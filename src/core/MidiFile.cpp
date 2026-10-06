#include "MidiFile.h"

#include <juce_audio_basics/juce_audio_basics.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <deque>
#include <limits>
#include <map>
#include <optional>
#include <utility>

namespace sv
{
namespace
{
// High-resolution PPQ (Pulses Per Quarter note) used for MIDI export.
// 9600 ticks/quarter note gives high timing precision while remaining standard.
constexpr int exportTicksPerQuarter = 9600;

// Maximum valid MIDI tick timestamp (28-bit unsigned integer in standard MIDI specs).
constexpr int maxMidiTick = 0x0fffffff;

/**
 * @brief Converts MIDI ticks (at a specified PPQ) to OpenSV Blicks.
 *
 * Uses extended precision (long double) to avoid overflow before rounding.
 * Formula: position = ticks * (blicksPerQuarter / ticksPerQuarter).
 */
std::optional<Blick> ticksToBlick(double ticks, int ticksPerQuarter)
{
    const auto position = static_cast<long double>(ticks) * blicksPerQuarter / ticksPerQuarter;
    if (!std::isfinite(ticks) || position < 0 || position >= static_cast<long double>(std::numeric_limits<Blick>::max()))
    {
        return std::nullopt;
    }

    return static_cast<Blick>(std::round(position));
}

/**
 * @brief Converts OpenSV Blicks to MIDI export ticks at 9600 PPQ.
 */
std::optional<double> blickToTicks(Blick position)
{
    const auto ticks = std::round(static_cast<long double>(position) * exportTicksPerQuarter / blicksPerQuarter);
    if (position < 0 || ticks > maxMidiTick)
    {
        return std::nullopt;
    }

    return static_cast<double>(ticks);
}

/**
 * @brief Validates time signature numerator and denominator.
 * Denominator must be a positive power of 2 (1, 2, 4, 8, 16, 32, 64, 128).
 */
bool isValidTimeSignature(int numerator, int denominator)
{
    return numerator > 0 && numerator <= 255 && denominator > 0 && denominator <= 128 && (denominator & (denominator - 1)) == 0;
}

/**
 * @brief Computes length of one musical measure/bar in Blicks for a given time signature.
 *
 * Example: 4/4 meter -> 705600000 * 4 * 4 / 4 = 4 * 705600000 blicks.
 * Example: 3/4 meter -> 705600000 * 4 * 3 / 4 = 3 * 705600000 blicks.
 * Example: 6/8 meter -> 705600000 * 4 * 6 / 8 = 3 * 705600000 blicks.
 */
Blick barLength(const TimeSignature& signature)
{
    return blicksPerQuarter * 4 * signature.numerator / signature.denominator;
}

/**
 * @brief Adds a MIDI message to a sequence at the specified tick timestamp.
 */
void addEvent(juce::MidiMessageSequence& sequence, juce::MidiMessage message, double ticks)
{
    message.setTimeStamp(ticks);
    sequence.addEvent(message);
}

/**
 * @brief Extracts tempo and time signature events from a loaded MIDI file into a TempoMap.
 */
juce::Result importTempoMap(const juce::MidiFile& midi, TempoMap& tempoMap)
{
    // Step 1: Extract all tempo events
    juce::MidiMessageSequence tempoEvents;
    midi.findAllTempoEvents(tempoEvents);
    std::map<Blick, double> tempos{{0, 120.0}}; // Default fallback at position 0
    for (const auto* event : tempoEvents)
    {
        const auto& message = event->message;
        const auto position = ticksToBlick(message.getTimeStamp(), midi.getTimeFormat());
        if (!position || message.getMetaEventLength() != 3 || message.getTempoSecondsPerQuarterNote() <= 0.0)
        {
            return juce::Result::fail("MIDI import: invalid tempo event.");
        }

        tempos[*position] = 60.0 / message.getTempoSecondsPerQuarterNote();
    }

    tempoMap.tempos.clear();
    for (const auto& [position, bpm] : tempos)
    {
        tempoMap.tempos.push_back({position, bpm, {}});
    }

    // Step 2: Extract all time signature events
    juce::MidiMessageSequence signatureEvents;
    midi.findAllTimeSigEvents(signatureEvents);
    std::map<Blick, TimeSignature> signatures{{0, {0, 4, 4, {}}}}; // Default 4/4 meter
    for (const auto* event : signatureEvents)
    {
        const auto& message = event->message;
        const auto position = ticksToBlick(message.getTimeStamp(), midi.getTimeFormat());
        if (!position || message.getMetaEventLength() != 4 || message.getMetaEventData()[1] > 7)
        {
            return juce::Result::fail("MIDI import: invalid time signature event.");
        }

        TimeSignature signature;
        message.getTimeSignatureInfo(signature.numerator, signature.denominator);
        if (!isValidTimeSignature(signature.numerator, signature.denominator))
        {
            return juce::Result::fail("MIDI import: invalid time signature.");
        }

        signatures[*position] = signature;
    }

    // Step 3: Align time signature changes to musical bars
    tempoMap.timeSignatures.clear();
    Blick previousPosition = 0;
    TimeSignature previousSignature;
    for (auto& [position, signature] : signatures)
    {
        const auto length = barLength(previousSignature);
        const auto distance = position - previousPosition;
        const auto bars = distance / length;
        // In OpenSV / SV, meter changes must happen on bar boundaries
        if (distance % length != 0)
        {
            return juce::Result::fail("MIDI import: time signature changes inside a bar are not supported.");
        }
        if (bars > std::numeric_limits<int>::max() - previousSignature.bar)
        {
            return juce::Result::fail("MIDI import: bar number exceeds the supported range.");
        }

        signature.bar = previousSignature.bar + static_cast<int>(bars);
        tempoMap.timeSignatures.push_back(signature);
        previousPosition = position;
        previousSignature = signature;
    }

    return juce::Result::ok();
}

/**
 * @brief Imports notes and lyrics from a single MIDI sequence into Track objects.
 *
 * Handles MIDI channels: notes on different channels within the same MIDI track
 * are split into distinct tracks.
 */
juce::Result importTrack(const juce::MidiMessageSequence& sequence, int sourceIndex, int ticksPerQuarter, std::vector<Track>& tracks)
{
    std::string name = "Track " + std::to_string(sourceIndex + 1);
    std::map<Blick, std::string> lyrics;

    // Scan for track name meta-events and lyric meta-events (meta-type 5)
    for (const auto* event : sequence)
    {
        const auto& message = event->message;
        const auto position = ticksToBlick(message.getTimeStamp(), ticksPerQuarter);
        if (!position)
        {
            return juce::Result::fail("MIDI import: event position exceeds the supported range.");
        }
        if (message.isTrackNameEvent())
        {
            name = message.getTextFromTextMetaEvent().toStdString();
        }
        else if (message.isMetaEvent() && message.getMetaEventType() == 5)
        {
            lyrics[*position] += message.getTextFromTextMetaEvent().toStdString();
        }
    }

    // Pair Note-On with Note-Off events per (channel, pitch) key
    std::array<std::optional<Track>, 16> channelTracks;
    std::map<int, std::deque<std::size_t>> activeNotes;

    for (const auto* event : sequence)
    {
        const auto& message = event->message;
        if (!message.isNoteOnOrOff())
        {
            continue;
        }

        const int channel = message.getChannel() - 1;
        const int pitch = message.getNoteNumber();
        const auto position = *ticksToBlick(message.getTimeStamp(), ticksPerQuarter);
        auto& active = activeNotes[channel * 128 + pitch];
        auto& track = channelTracks[static_cast<std::size_t>(channel)];

        if (message.isNoteOn())
        {
            if (!track)
            {
                track.emplace();
                track->name = name;
                track->mainGroup = createNoteGroup(name);
            }

            Note note;
            note.id = createNoteId();
            note.onset = position;
            note.duration = 0; // Completed when matching note-off is encountered
            note.pitch = pitch;
            if (const auto lyric = lyrics.find(position); lyric != lyrics.end())
            {
                note.lyrics = lyric->second;
            }
            active.push_back(track->mainGroup.notes.size());
            track->mainGroup.notes.push_back(std::move(note));
        }
        else if (!active.empty())
        {
            // Note-Off matches the earliest pending Note-On of the same pitch
            auto& note = track->mainGroup.notes[active.front()];
            active.pop_front();
            if (position <= note.onset)
            {
                return juce::Result::fail("MIDI import: a note has zero or negative duration.");
            }

            note.duration = position - note.onset;
        }
    }

    // Verify all notes were properly closed
    for (const auto& entry : activeNotes)
    {
        if (!entry.second.empty())
        {
            return juce::Result::fail("MIDI import: a note is missing its note-off event.");
        }
    }

    // Append populated tracks to output vector
    const auto channelCount = std::count_if(channelTracks.begin(), channelTracks.end(), [](const auto& track)
                                            { return track.has_value(); });
    for (std::size_t channel = 0; channel < channelTracks.size(); ++channel)
    {
        auto& track = channelTracks[channel];
        if (track)
        {
            if (channelCount > 1)
            {
                track->name += " (channel " + std::to_string(channel + 1) + ")";
            }
            tracks.push_back(std::move(*track));
        }
    }

    return juce::Result::ok();
}

/**
 * @brief Exports tempo and meter timelines to MIDI Track 0 (conductor track).
 */
juce::Result exportTempoMap(const TempoMap& tempoMap, juce::MidiMessageSequence& sequence)
{
    if (tempoMap.tempos.empty() || tempoMap.tempos.front().position != 0 || tempoMap.timeSignatures.empty() || tempoMap.timeSignatures.front().bar != 0)
    {
        return juce::Result::fail("MIDI export: tempo and time signature maps must begin at the project start.");
    }

    // Export tempo changes (microseconds per quarter note)
    Blick previousTempoPosition = -1;
    for (const auto& tempo : tempoMap.tempos)
    {
        const auto ticks = blickToTicks(tempo.position);
        const auto microseconds = 60000000.0 / tempo.bpm;
        if (!ticks || !std::isfinite(microseconds) || microseconds < 1.0 || microseconds > 16777215.0 || tempo.position <= previousTempoPosition)
        {
            return juce::Result::fail("MIDI export: invalid tempo or tempo position.");
        }

        addEvent(sequence, juce::MidiMessage::tempoMetaEvent(static_cast<int>(std::round(microseconds))), *ticks);
        previousTempoPosition = tempo.position;
    }

    // Export time signatures
    Blick position = 0;
    TimeSignature previousSignature;
    for (std::size_t index = 0; index < tempoMap.timeSignatures.size(); ++index)
    {
        const auto& signature = tempoMap.timeSignatures[index];
        if (!isValidTimeSignature(signature.numerator, signature.denominator) || signature.bar < 0 || (index > 0 && signature.bar <= previousSignature.bar))
        {
            return juce::Result::fail("MIDI export: invalid time signature or bar number.");
        }

        const auto bars = static_cast<Blick>(signature.bar) - previousSignature.bar;
        const auto length = barLength(previousSignature);
        if (bars > (std::numeric_limits<Blick>::max() - position) / length)
        {
            return juce::Result::fail("MIDI export: time signature position exceeds the supported range.");
        }

        position += bars * length;
        const auto ticks = blickToTicks(position);
        if (!ticks)
        {
            return juce::Result::fail("MIDI export: time signature position exceeds the supported range.");
        }
        addEvent(sequence, juce::MidiMessage::timeSignatureMetaEvent(signature.numerator, signature.denominator), *ticks);
        previousSignature = signature;
    }

    return juce::Result::ok();
}

/**
 * @brief Exports notes from a NoteGroup referenced on a track to MIDI events.
 */
juce::Result exportGroup(const NoteGroup& group, const GroupReference& reference, int channel, juce::MidiMessageSequence& sequence)
{
    if (reference.isInstrumental)
    {
        return juce::Result::ok();
    }
    if (reference.absoluteBegin < 0 || reference.absoluteEnd < -1 || (reference.absoluteEnd >= 0 && reference.absoluteEnd < reference.absoluteBegin))
    {
        return juce::Result::fail("MIDI export: invalid note group boundaries.");
    }

    for (const auto& note : group.notes)
    {
        const auto pitch = static_cast<std::int64_t>(note.pitch) + reference.pitchOffset;
        const auto timeOffset = reference.timeOffset;
        if (note.duration <= 0 || pitch < 0 || pitch > 127 || (timeOffset > 0 && note.onset > std::numeric_limits<Blick>::max() - timeOffset) || (timeOffset < 0 && note.onset < std::numeric_limits<Blick>::min() - timeOffset))
        {
            return juce::Result::fail("MIDI export: invalid note duration, position, or pitch.");
        }

        const auto onset = note.onset + timeOffset;
        if (onset > std::numeric_limits<Blick>::max() - note.duration)
        {
            return juce::Result::fail("MIDI export: note position exceeds the supported range.");
        }

        // Apply crop boundaries (clipping window)
        const auto clippedStart = std::max(onset, reference.absoluteBegin);
        const auto clippedEnd = reference.absoluteEnd < 0 ? onset + note.duration : std::min(onset + note.duration, reference.absoluteEnd);
        if (clippedEnd <= clippedStart)
        {
            continue;
        }

        const auto startTicks = blickToTicks(clippedStart);
        const auto endTicks = blickToTicks(clippedEnd);
        if (!startTicks || !endTicks || *endTicks <= *startTicks)
        {
            return juce::Result::fail("MIDI export: note position or duration cannot be represented at 9600 ticks per quarter note.");
        }

        // Export lyrics as MIDI meta-event type 5 on note onset
        if (!note.lyrics.empty())
        {
            addEvent(sequence, juce::MidiMessage::textMetaEvent(5, juce::String::fromUTF8(note.lyrics.c_str())), *startTicks);
        }

        // Standard Note-On with default velocity 100, and Note-Off
        addEvent(sequence, juce::MidiMessage::noteOn(channel, static_cast<int>(pitch), static_cast<juce::uint8>(100)), *startTicks);
        addEvent(sequence, juce::MidiMessage::noteOff(channel, static_cast<int>(pitch)), *endTicks);
    }

    return juce::Result::ok();
}
} // namespace

juce::Result importMidiFile(const juce::File& file, Project& project)
{
    auto stream = file.createInputStream();
    if (!stream || stream->getStatus().failed())
    {
        return juce::Result::fail("MIDI import: could not open " + file.getFullPathName());
    }

    juce::MidiFile midi;
    int fileType = -1;
    if (!midi.readFrom(*stream, false, &fileType))
    {
        return juce::Result::fail("MIDI import: invalid or truncated MIDI file.");
    }
    if (fileType != 0 && fileType != 1)
    {
        return juce::Result::fail("MIDI import: only synchronous MIDI formats 0 and 1 are supported.");
    }
    if (midi.getTimeFormat() <= 0)
    {
        return juce::Result::fail("MIDI import: SMPTE timing is not supported; use a file with ticks per quarter note.");
    }

    Project imported;
    imported.name = file.getFileNameWithoutExtension().toStdString();
    if (const auto result = importTempoMap(midi, imported.tempoMap); result.failed())
    {
        return result;
    }
    for (int index = 0; index < midi.getNumTracks(); ++index)
    {
        if (const auto result = importTrack(*midi.getTrack(index), index, midi.getTimeFormat(), imported.tracks); result.failed())
        {
            return result;
        }
    }
    if (imported.tracks.empty())
    {
        return juce::Result::fail("MIDI import: the file contains no notes.");
    }

    normaliseProject(imported);
    project = std::move(imported);
    return juce::Result::ok();
}

juce::Result exportMidiFile(const juce::File& file, const Project& project)
{
    if (project.tracks.size() > 65534)
    {
        return juce::Result::fail("MIDI export: too many tracks.");
    }

    juce::MidiFile midi;
    midi.setTicksPerQuarterNote(exportTicksPerQuarter);

    // Track 0: Conductor track with song title, tempo, and time signatures
    juce::MidiMessageSequence conductor;
    addEvent(conductor, juce::MidiMessage::textMetaEvent(3, juce::String::fromUTF8(project.name.c_str())), 0.0);
    if (const auto result = exportTempoMap(project.tempoMap, conductor); result.failed())
    {
        return result;
    }
    midi.addTrack(conductor);

    // Track 1..N: Musical vocal tracks
    for (std::size_t index = 0; index < project.tracks.size(); ++index)
    {
        const auto& track = project.tracks[index];
        // Channel assignment: modulo 15, skipping GM Channel 10 (reserved for drums/percussion)
        const auto channelIndex = static_cast<int>(index % 15);
        const int channel = channelIndex < 9 ? channelIndex + 1 : channelIndex + 2;

        juce::MidiMessageSequence sequence;
        addEvent(sequence, juce::MidiMessage::textMetaEvent(3, juce::String::fromUTF8(track.name.c_str())), 0.0);

        if (const auto result = exportGroup(track.mainGroup, track.mainRef, channel, sequence); result.failed())
        {
            return result;
        }
        for (const auto& reference : track.groups)
        {
            if (reference.isInstrumental)
            {
                continue;
            }
            const auto* group = findNoteGroup(project, reference.groupId);
            if (group == nullptr)
            {
                return juce::Result::fail("MIDI export: a referenced note group is missing.");
            }
            if (const auto result = exportGroup(*group, reference, channel, sequence); result.failed())
            {
                return result;
            }
        }
        midi.addTrack(sequence);
    }

    // Atomic write to avoid partial/corrupt files on interruption
    juce::TemporaryFile temporary(file);
    auto stream = temporary.getFile().createOutputStream();
    if (!stream || stream->getStatus().failed())
    {
        return juce::Result::fail("MIDI export: could not create " + file.getFullPathName());
    }
    const bool written = midi.writeTo(*stream, 1); // Format 1 (multi-track)
    stream->flush();
    if (!written || stream->getStatus().failed())
    {
        return juce::Result::fail("MIDI export: could not write " + file.getFullPathName());
    }
    stream.reset();
    if (!temporary.overwriteTargetFileWithTemporary())
    {
        return juce::Result::fail("MIDI export: could not replace " + file.getFullPathName());
    }

    return juce::Result::ok();
}

} // namespace sv
