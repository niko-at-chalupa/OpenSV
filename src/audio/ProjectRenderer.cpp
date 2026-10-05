#include "ProjectRenderer.h"

#include "core/ParameterCurve.h"
#include "synthesis/PitchCurve.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <memory>
#include <new>
#include <numbers>
#include <span>
#include <string>
#include <string_view>
#include <utility>

namespace sv::audio
{
namespace
{
constexpr double restContextSeconds = 0.1;
constexpr double phraseBreakSeconds = restContextSeconds * 2.0;
constexpr double maximumPhraseSeconds = 30.0;
constexpr std::size_t maximumPhrasePhonemes = 4096;
constexpr std::size_t maximumAudioBytes = 128 * 1024 * 1024;
constexpr std::size_t maximumPhraseCacheBytes = 128 * 1024 * 1024;
constexpr std::size_t maximumCachedPhrases = 1024;
constexpr std::size_t maximumInferenceCacheBytes = 256 * 1024 * 1024;
constexpr std::size_t maximumCachedInferenceStates = 64;

struct RenderNote
{
    const Note* note = nullptr;
    const NoteGroup* group = nullptr;
    const GroupReference* reference = nullptr;
    Blick start = 0;
    Blick end = 0;
    double startSeconds = 0.0;
    double endSeconds = 0.0;
    int pitch = 60;
};

struct RenderTrack
{
    const Track* track = nullptr;
    std::size_t projectIndex = 0;
    bool audible = false;
    std::vector<RenderNote> notes;
};

bool cancelled(const std::function<bool()>& shouldCancel)
{
    return shouldCancel && shouldCancel();
}

juce::Result cancellation()
{
    return juce::Result::fail("Singing rendering was cancelled.");
}

juce::Result trackError(const Track& track, const juce::String& message)
{
    return juce::Result::fail("Track '" + juce::String::fromUTF8(track.name.c_str()) + "': " + message);
}

juce::String noteContext(const RenderNote& note)
{
    return "note '" + juce::String::fromUTF8(note.note->lyrics.c_str()) + "' at " + juce::String(note.startSeconds, 3) + " s: ";
}

bool samePitchNotes(std::span<const synthesis::PitchNote> first, std::span<const synthesis::PitchNote> second)
{
    return std::equal(first.begin(), first.end(), second.begin(), second.end(), [](const auto& left, const auto& right)
                      { return left.syllable.language == right.syllable.language && left.syllable.phonemes == right.syllable.phonemes && left.syllable.durationSeconds == right.syllable.durationSeconds && left.syllable.midiPitch == right.syllable.midiPitch && left.isBreath == right.isBreath && left.isSilence == right.isSilence && left.isContinuation == right.isContinuation && left.isRap == right.isRap && left.tone == right.tone && left.vibratoModulation == right.vibratoModulation; });
}

bool isEnglishVowelPhoneme(std::string_view symbol)
{
    static constexpr std::array<std::string_view, 16> vowels{
        "aa", "ae", "ah", "ao", "aw", "ax", "ay", "eh", "er", "ey", "ih", "iy", "ow", "oy", "uh", "uw"};
    return std::find(vowels.begin(), vowels.end(), symbol) != vowels.end();
}

void mergeContinuationPhones(std::span<const synthesis::PhonemeDuration> durations, std::span<const synthesis::TimingSyllable> syllables, std::vector<synthesis::TimedPhoneme>& phonemes)
{
    if (durations.size() != phonemes.size())
    {
        return;
    }
    std::vector<synthesis::TimedPhoneme> merged;
    merged.reserve(phonemes.size());
    for (std::size_t index = 0; index < phonemes.size(); ++index)
    {
        const auto& duration = durations[index];
        const auto& phoneme = phonemes[index];
        const bool continuation = duration.syllableIndex < syllables.size() && syllables[duration.syllableIndex].isContinuation;
        if (continuation && !merged.empty() && merged.back().language == phoneme.language && merged.back().symbol == phoneme.symbol && phoneme.frameCount <= std::numeric_limits<std::size_t>::max() - merged.back().frameCount)
        {
            merged.back().frameCount += phoneme.frameCount;
        }
        else
        {
            merged.push_back(phoneme);
        }
    }
    phonemes = std::move(merged);
}

bool addBlicks(Blick first, Blick second, Blick& result)
{
    if ((second > 0 && first > std::numeric_limits<Blick>::max() - second) || (second < 0 && first < std::numeric_limits<Blick>::min() - second))
    {
        return false;
    }
    result = first + second;
    return true;
}

Blick localCurvePosition(Blick absolutePosition, Blick timeOffset)
{
    // Parameter curves hold their endpoints outside their stored range.
    if (timeOffset > 0 && absolutePosition < std::numeric_limits<Blick>::min() + timeOffset)
    {
        return std::numeric_limits<Blick>::min();
    }
    if (timeOffset < 0 && absolutePosition > std::numeric_limits<Blick>::max() + timeOffset)
    {
        return std::numeric_limits<Blick>::max();
    }
    return absolutePosition - timeOffset;
}

juce::Result collectNotes(const Project& project, const NoteGroup& group, const GroupReference& reference, std::vector<RenderNote>& notes)
{
    if (reference.isInstrumental)
    {
        return juce::Result::ok();
    }
    if (reference.absoluteBegin < 0 || reference.absoluteEnd < -1 || (reference.absoluteEnd >= 0 && reference.absoluteEnd <= reference.absoluteBegin))
    {
        return juce::Result::fail("The group has an invalid crop range.");
    }
    for (const auto& note : group.notes)
    {
        Blick start = 0;
        Blick end = 0;
        if (note.duration <= 0 || !addBlicks(note.onset, reference.timeOffset, start) || !addBlicks(start, note.duration, end))
        {
            return juce::Result::fail("A note has an invalid time range.");
        }
        start = std::max({Blick{0}, reference.absoluteBegin, start});
        end = reference.absoluteEnd >= 0 ? std::min(end, reference.absoluteEnd) : end;
        if (end <= start)
        {
            continue;
        }
        const auto pitch = static_cast<std::int64_t>(note.pitch) + static_cast<std::int64_t>(reference.pitchOffset);
        if (pitch < 0 || pitch > 127)
        {
            return juce::Result::fail("A transposed note is outside the MIDI pitch range.");
        }
        const double startSeconds = project.tempoMap.blickToSeconds(start);
        const double endSeconds = project.tempoMap.blickToSeconds(end);
        if (!std::isfinite(startSeconds) || !std::isfinite(endSeconds) || startSeconds < 0.0 || endSeconds <= startSeconds)
        {
            return juce::Result::fail("A note has an invalid duration after applying the tempo map.");
        }
        notes.push_back({&note, &group, &reference, start, end, startSeconds, endSeconds, static_cast<int>(pitch)});
    }
    return juce::Result::ok();
}

juce::Result prepareTrack(const Project& project, RenderTrack& prepared)
{
    const auto& track = *prepared.track;
    if (!std::isfinite(track.gain) || track.gain < 0.0 || track.gain > static_cast<double>(std::numeric_limits<float>::max()) || !std::isfinite(track.pan))
    {
        return juce::Result::fail("The mixer gain or pan is invalid.");
    }
    if (const auto result = collectNotes(project, track.mainGroup, track.mainRef, prepared.notes); result.failed())
    {
        return result;
    }
    for (const auto& reference : track.groups)
    {
        const auto* group = findNoteGroup(project, reference.groupId);
        if (group == nullptr)
        {
            return juce::Result::fail("A referenced note group is missing.");
        }
        if (const auto result = collectNotes(project, *group, reference, prepared.notes); result.failed())
        {
            return result;
        }
    }
    std::stable_sort(prepared.notes.begin(), prepared.notes.end(), [](const RenderNote& left, const RenderNote& right)
                     { return left.start < right.start; });
    for (std::size_t index = 1; index < prepared.notes.size(); ++index)
    {
        if (prepared.notes[index].start < prepared.notes[index - 1].end)
        {
            return juce::Result::fail(noteContext(prepared.notes[index]) + "overlapping singing notes require separate tracks.");
        }
    }
    return juce::Result::ok();
}

std::shared_ptr<const PhraseVisualization> makeVisualization(const synthesis::NeuralVocoderOutput& rendered, float frameInterval, std::span<const float> logF0)
{
    auto data = std::make_shared<PhraseVisualization>();
    data->sampleRate = rendered.sampleRate;
    data->frameIntervalSeconds = static_cast<double>(frameInterval);
    data->sampleCount = rendered.samples.size();
    data->midiPitch.reserve(logF0.size());
    for (const float logarithmicPitch : logF0)
    {
        // Voicing controls excitation, not the continuous pitch control curve.
        // Preserve the predicted transitions through unvoiced consonants.
        data->midiPitch.push_back(static_cast<float>(69.0 + 12.0 * (static_cast<double>(logarithmicPitch) - std::log(440.0)) / std::numbers::ln2));
    }
    WaveformLevel base;
    base.peaks.reserve((rendered.samples.size() + base.samplesPerPeak - 1) / base.samplesPerPeak);
    for (std::size_t first = 0; first < rendered.samples.size(); first += base.samplesPerPeak)
    {
        const auto end = std::min(first + base.samplesPerPeak, rendered.samples.size());
        const auto range = juce::FloatVectorOperations::findMinAndMax(rendered.samples.data() + first, end - first);
        const WaveformPeak peak{range.getStart(), range.getEnd()};
        data->peakMagnitude = std::max({data->peakMagnitude, std::abs(peak.minimum), std::abs(peak.maximum)});
        base.peaks.push_back(peak);
    }
    data->waveform.push_back(std::move(base));
    while (data->waveform.back().peaks.size() > 1)
    {
        const auto& previous = data->waveform.back();
        WaveformLevel level;
        level.samplesPerPeak = previous.samplesPerPeak * 2;
        level.peaks.reserve((previous.peaks.size() + 1) / 2);
        for (std::size_t first = 0; first < previous.peaks.size(); first += 2)
        {
            auto peak = previous.peaks[first];
            if (first + 1 < previous.peaks.size())
            {
                peak.minimum = std::min(peak.minimum, previous.peaks[first + 1].minimum);
                peak.maximum = std::max(peak.maximum, previous.peaks[first + 1].maximum);
            }
            level.peaks.push_back(peak);
        }
        data->waveform.push_back(std::move(level));
    }
    return data;
}

void appendVisualization(TrackVisualization& destination, const Track& track, std::span<const RenderNote> notes, double startSeconds, double stopSeconds, std::shared_ptr<const PhraseVisualization> data)
{
    VisualPhrase phrase;
    phrase.startSeconds = startSeconds;
    phrase.endSeconds = std::min(stopSeconds, startSeconds + static_cast<double>(data->sampleCount) / data->sampleRate);
    phrase.notes.reserve(notes.size());
    for (const auto& note : notes)
    {
        phrase.notes.push_back({note.note->id, note.startSeconds, note.endSeconds, note.note->pitch, note.reference->pitchOffset, note.reference == &track.mainRef});
    }
    phrase.data = std::move(data);
    destination.phrases.push_back(std::move(phrase));
}

double resolvePitchAttribute(const RenderNote& note, std::optional<double> PitchAttributes::* field, double manualDefault, std::optional<double> generatedValue = {})
{
    if ((note.note->attributes.*field).has_value())
    {
        return *(note.note->attributes.*field);
    }
    if (note.note->instantMode && generatedValue.has_value())
    {
        return *generatedValue;
    }
    return (note.reference->voicePitch.*field).value_or(manualDefault);
}

double resolveRapAttribute(const RenderNote& note, std::optional<double> PitchAttributes::* field)
{
    if ((note.note->attributes.*field).has_value())
    {
        return *(note.note->attributes.*field);
    }
    if (note.note->instantMode && (note.note->systemAttributes.*field).has_value())
    {
        return *(note.note->systemAttributes.*field);
    }
    return 0.0;
}

std::vector<float> buildVibratoEnvelope(const Project& project, std::span<const RenderNote> notes, double startSeconds, double frameIntervalSeconds, std::size_t frameCount)
{
    std::vector<float> envelope(frameCount, 1.0f);
    std::size_t noteIndex = 0;
    for (std::size_t frame = 0; frame < frameCount; ++frame)
    {
        const double seconds = startSeconds + static_cast<double>(frame) * frameIntervalSeconds;
        while (noteIndex + 1 < notes.size() && seconds >= notes[noteIndex + 1].startSeconds)
        {
            ++noteIndex;
        }
        const auto& note = notes[noteIndex];
        const auto position = localCurvePosition(project.tempoMap.secondsToBlick(seconds), note.reference->timeOffset);
        envelope[frame] = static_cast<float>(sampleParameterCurve(note.group->vibratoEnv, position, 1.0));
    }
    return envelope;
}

juce::Result buildPitch(const Project& project, std::span<const RenderNote> notes, double startSeconds, float frameIntervalSeconds, std::size_t frameCount, std::span<const float> automaticPitch, double pitchStartSeconds, float pitchFrameIntervalSeconds, std::vector<float>& logF0)
{
    std::vector<synthesis::PitchCurveNote> editorNotes;
    std::vector<synthesis::PitchCurveNote> predictionNotes;
    editorNotes.reserve(notes.size());
    predictionNotes.reserve(notes.size());
    for (const auto& note : notes)
    {
        synthesis::PitchCurveNote curve;
        curve.startSeconds = note.startSeconds;
        curve.endSeconds = note.endSeconds;
        curve.secondsPerQuarter = 60.0 / project.tempoMap.getTempoAt(note.start);
        curve.instantMode = note.note->instantMode;
        curve.midiPitch = note.pitch;
        curve.tF0Left = note.note->attributes.tF0Left.value_or(curve.secondsPerQuarter * 0.2);
        curve.tF0Right = note.note->attributes.tF0Right.value_or(curve.secondsPerQuarter * 0.2);
        predictionNotes.push_back(curve);
        // Regenerated gen5 attributes supersede an imported system curve. The
        // latter may belong to a different score, tempo, voice or model version.
        curve.midiPitch += note.note->detune / 100.0;
        if (note.note->musicalType == "rap")
        {
            curve.midiPitch += resolveRapAttribute(note, &PitchAttributes::rTone);
            curve.rapIntonation = resolveRapAttribute(note, &PitchAttributes::rIntonation);
        }
        curve.tF0Offset = resolvePitchAttribute(note, &PitchAttributes::tF0Offset, 0.0, 0.0);
        curve.tF0Left = resolvePitchAttribute(note, &PitchAttributes::tF0Left, 0.1, curve.secondsPerQuarter * 0.2);
        curve.tF0Right = resolvePitchAttribute(note, &PitchAttributes::tF0Right, 0.07, curve.secondsPerQuarter * 0.2);
        curve.dF0Left = resolvePitchAttribute(note, &PitchAttributes::dF0Left, 0.15, 0.0);
        curve.dF0Right = resolvePitchAttribute(note, &PitchAttributes::dF0Right, 0.15, 0.0);
        curve.tF0VbrStart = resolvePitchAttribute(note, &PitchAttributes::tF0VbrStart, 0.25);
        curve.tF0VbrLeft = resolvePitchAttribute(note, &PitchAttributes::tF0VbrLeft, 0.2);
        curve.tF0VbrRight = resolvePitchAttribute(note, &PitchAttributes::tF0VbrRight, 0.2);
        curve.dF0Vbr = resolvePitchAttribute(note, &PitchAttributes::dF0Vbr, 1.0, 0.0);
        curve.pF0Vbr = resolvePitchAttribute(note, &PitchAttributes::pF0Vbr, 0.0);
        curve.fF0Vbr = resolvePitchAttribute(note, &PitchAttributes::fF0Vbr, 5.5);
        editorNotes.push_back(curve);
    }
    const double firstFrameSeconds = startSeconds + 0.5 * static_cast<double>(frameIntervalSeconds);
    const auto transitions = synthesis::renderPitchTransitions(editorNotes, firstFrameSeconds, frameIntervalSeconds, frameCount);
    const auto vibrato = synthesis::renderPitchVibrato(editorNotes, firstFrameSeconds, frameIntervalSeconds, frameCount);
    const auto mask = synthesis::renderAutomaticPitchMask(editorNotes, firstFrameSeconds, frameIntervalSeconds, frameCount);
    const auto envelope = buildVibratoEnvelope(project, notes, firstFrameSeconds, frameIntervalSeconds, frameCount);
    std::vector<float> residual;
    if (!automaticPitch.empty())
    {
        residual = synthesis::renderPitchPredictionBase(predictionNotes, pitchStartSeconds, pitchFrameIntervalSeconds, automaticPitch.size());
        for (std::size_t frame = 0; frame < residual.size(); ++frame)
        {
            residual[frame] = automaticPitch[frame] - residual[frame];
        }
    }
    logF0.resize(frameCount);
    std::size_t noteIndex = 0;
    for (std::size_t frame = 0; frame < frameCount; ++frame)
    {
        // Sample at the frame centre so float frame periods cannot push a
        // transition on an exact score boundary into the following frame.
        const double seconds = startSeconds + (static_cast<double>(frame) + 0.5) * static_cast<double>(frameIntervalSeconds);
        while (noteIndex + 1 < notes.size() && seconds >= notes[noteIndex + 1].startSeconds)
        {
            ++noteIndex;
        }
        const auto& note = notes[noteIndex];
        const auto position = localCurvePosition(project.tempoMap.secondsToBlick(seconds), note.reference->timeOffset);
        double automaticDelta = 0.0;
        if (!residual.empty())
        {
            const double modelFrame = std::clamp((seconds - pitchStartSeconds) / static_cast<double>(pitchFrameIntervalSeconds), 0.0, static_cast<double>(residual.size() - 1));
            const auto left = static_cast<std::size_t>(modelFrame);
            const auto right = std::min(left + 1, residual.size() - 1);
            automaticDelta = std::lerp(residual[left], residual[right], modelFrame - static_cast<double>(left));
        }
        const double cents = sampleParameterCurve(note.group->pitchDelta, position);
        const double noteProgress = std::clamp((seconds - note.startSeconds) / (note.endSeconds - note.startSeconds), 0.0, 1.0);
        const double rapIntonation = note.note->musicalType == "rap" ? editorNotes[noteIndex].rapIntonation * (noteProgress - 0.5) : 0.0;
        const double pitch = transitions[frame] + mask[frame] * automaticDelta + envelope[frame] * vibrato[frame] + cents / 100.0 + rapIntonation;
        const double logarithmicPitch = std::log(440.0) + (pitch - 69.0) * std::numbers::ln2 / 12.0;
        const float frequency = std::exp(static_cast<float>(logarithmicPitch));
        if (!std::isfinite(pitch) || !std::isfinite(frequency) || frequency <= 0.0f)
        {
            return juce::Result::fail(noteContext(note) + "pitch automation produces an invalid fundamental frequency.");
        }
        logF0[frame] = static_cast<float>(logarithmicPitch);
    }
    return juce::Result::ok();
}

juce::Result mixPhrase(std::span<const float> samples, double sourceSampleRate, double startSeconds, double cropStartSeconds, double stopSeconds, double sampleRate, const Track& track, juce::AudioBuffer<float>& output, const std::function<bool()>& shouldCancel)
{
    if (!std::isfinite(sourceSampleRate) || sourceSampleRate <= 0.0 || samples.empty() || samples.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    {
        return juce::Result::fail("The voice renderer returned invalid audio.");
    }
    std::vector<float> resampled;
    if (sourceSampleRate != sampleRate)
    {
        const double ratio = sourceSampleRate / sampleRate;
        const auto outputCount = static_cast<std::size_t>(std::ceil(static_cast<double>(samples.size()) / ratio));
        if (outputCount > maximumAudioBytes / sizeof(float))
        {
            return juce::Result::fail("The resampled phrase exceeds the rendering memory limit.");
        }
        resampled.resize(outputCount);
        juce::WindowedSincInterpolator interpolator;
        constexpr auto latency = static_cast<int>(juce::WindowedSincInterpolator::getBaseLatency());
        std::array<float, latency> priming{};
        // Prime in input-sample time to remove the interpolator's 100-sample delay.
        const int consumed = interpolator.process(1.0, samples.data(), priming.data(), latency, static_cast<int>(samples.size()), 0);
        const float zero = 0.0f;
        const int remaining = static_cast<int>(samples.size()) - consumed;
        const auto* source = remaining > 0 ? samples.data() + consumed : &zero;
        interpolator.process(ratio, source, resampled.data(), static_cast<int>(outputCount), std::max(1, remaining), 0);
        samples = resampled;
    }
    if (cancelled(shouldCancel))
    {
        return cancellation();
    }
    const auto phraseOffset = static_cast<int>(std::llround(startSeconds * sampleRate));
    const auto offset = std::max({0, phraseOffset, static_cast<int>(std::ceil(cropStartSeconds * sampleRate))});
    const auto stop = std::min(output.getNumSamples(), static_cast<int>(std::ceil(stopSeconds * sampleRate)));
    const auto skippedSamples = static_cast<std::size_t>(offset - phraseOffset);
    if (offset >= stop || skippedSamples >= samples.size())
    {
        return juce::Result::ok();
    }
    // Keep complete model context in the cache; crop only when placing audio
    // on the project timeline, after resampling into its sample coordinates.
    samples = samples.subspan(skippedSamples);
    const auto count = std::min(samples.size(), static_cast<std::size_t>(stop - offset));
    const double panAngle = (std::clamp(track.pan, -1.0, 1.0) + 1.0) * std::numbers::pi / 4.0;
    const float leftGain = static_cast<float>(track.gain * std::cos(panAngle));
    const float rightGain = static_cast<float>(track.gain * std::sin(panAngle));
    auto* left = output.getWritePointer(0, offset);
    auto* right = output.getWritePointer(1, offset);
    for (std::size_t sample = 0; sample < count; ++sample)
    {
        if ((sample & 4095) == 0 && cancelled(shouldCancel))
        {
            return cancellation();
        }
        if (!std::isfinite(samples[sample]))
        {
            return juce::Result::fail("The voice renderer returned a non-finite sample.");
        }
        left[sample] += samples[sample] * leftGain;
        right[sample] += samples[sample] * rightGain;
    }
    return juce::Result::ok();
}

juce::String dictionaryStem(const std::string& language)
{
    if (language == "japanese")
    {
        return "japanese-romaji";
    }
    if (language == "mandarin")
    {
        return "mandarin-xsampa";
    }
    if (language == "english")
    {
        return "english-arpabet";
    }
    if (language == "cantonese")
    {
        return "cantonese-xsampa";
    }
    if (language == "spanish")
    {
        return "spanish-xsampa";
    }
    return {};
}
} // namespace

juce::Result ProjectRenderer::readFileStamp(const juce::File& file, FileStamp& stamp)
{
    if (!file.existsAsFile())
    {
        return juce::Result::fail("Cannot read synthesis resource: " + file.getFullPathName());
    }
    stamp = {file, file.getSize(), file.getLastModificationTime().toMilliseconds()};
    return juce::Result::ok();
}

juce::Result ProjectRenderer::findVoice(const FileStamp& source, synthesis::VoiceSynthesizer*& synthesizer)
{
    const auto found = std::find_if(voices.begin(), voices.end(), [&source](const CachedVoice& entry)
                                    { return entry.source == source; });
    if (found != voices.end())
    {
        std::rotate(found, std::next(found), voices.end());
        synthesizer = voices.back().synthesizer.get();
        return juce::Result::ok();
    }
    auto loaded = std::make_unique<synthesis::VoiceSynthesizer>();
    const double started = juce::Time::getMillisecondCounterHiRes();
    const auto result = loaded->load(source.file);
    statistics.modelLoadMilliseconds += juce::Time::getMillisecondCounterHiRes() - started;
    if (result.failed())
    {
        return result;
    }
    if (voices.size() == 2)
    {
        voices.erase(voices.begin());
    }
    voices.push_back({source, std::move(loaded)});
    synthesizer = voices.back().synthesizer.get();
    return juce::Result::ok();
}

juce::Result ProjectRenderer::resolvePhonemes(const VoiceSettings& settings, const Note& note, std::vector<std::string>& phonemes, std::string& continuationPhoneme, std::string& phonemeLanguage, bool preferEnglishContext)
{
    phonemeLanguage = settings.language;
    const auto selectContinuationPhoneme = [&phonemes, &continuationPhoneme](const synthesis::PhonemeDictionary& dictionary)
    {
        continuationPhoneme = phonemes.empty() ? std::string{} : phonemes.back();
        const auto& definitions = dictionary.getPhonemes();
        for (auto phone = phonemes.rbegin(); phone != phonemes.rend(); ++phone)
        {
            const auto definition = std::find_if(definitions.begin(), definitions.end(), [&phone](const synthesis::PhonemeDefinition& candidate)
                                                 { return candidate.symbol == *phone && (candidate.category == "vowel" || candidate.category == "diphthong"); });
            if (definition != definitions.end())
            {
                continuationPhoneme = *phone;
                return;
            }
        }
    };
    std::string_view explicitPhonemeText = note.phonemes;
    bool hasExplicitPhonemes = !explicitPhonemeText.empty();
    if (!hasExplicitPhonemes && !note.lyrics.empty() && note.lyrics.front() == '.')
    {
        explicitPhonemeText = std::string_view(note.lyrics).substr(1);
        hasExplicitPhonemes = true;
    }
    if (hasExplicitPhonemes)
    {
        if (explicitPhonemeText.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()) || explicitPhonemeText.find('\0') != std::string_view::npos || !juce::CharPointer_UTF8::isValidString(explicitPhonemeText.data(), static_cast<int>(explicitPhonemeText.size())))
        {
            return juce::Result::fail("Explicit phonemes must be valid UTF-8 text.");
        }
        auto fields = juce::StringArray::fromTokens(juce::String::fromUTF8(explicitPhonemeText.data(), static_cast<int>(explicitPhonemeText.size())), " \t\r\n", "");
        fields.removeEmptyStrings();
        if (fields.isEmpty())
        {
            return juce::Result::fail("The explicit phoneme sequence is empty.");
        }
        phonemes.clear();
        for (const auto& field : fields)
        {
            phonemes.push_back(field.toStdString());
        }
        continuationPhoneme = phonemes.back();
        if (settings.language == "english")
        {
            const auto vowel = std::find_if(phonemes.rbegin(), phonemes.rend(), [](const std::string& symbol)
                                            { return isEnglishVowelPhoneme(symbol); });
            if (vowel != phonemes.rend())
            {
                continuationPhoneme = *vowel;
            }
        }
        return juce::Result::ok();
    }
    const bool latinWord = !note.lyrics.empty() && std::all_of(note.lyrics.begin(), note.lyrics.end(), [](unsigned char character)
                                                               { return std::isalpha(character) != 0 || character == '\'' || character == '-'; });
    const auto loadEnglishDictionary = [&]() -> juce::Result
    {
        const juce::File directory(juce::String::fromUTF8(settings.dictionaryDirectory.c_str()));
        const auto phonesFile = directory.getChildFile("english-arpabet-phones.txt");
        auto dictionaryFile = directory.getChildFile("cmudict-07b.txt");
        if (!dictionaryFile.existsAsFile())
        {
            dictionaryFile = directory.getChildFile("english-arpabet-dict.txt");
        }
        if (!phonesFile.existsAsFile() || !dictionaryFile.existsAsFile())
        {
            return juce::Result::fail("English pronunciation dictionary files are unavailable.");
        }
        FileStamp phonesSource;
        FileStamp dictionarySource;
        if (const auto result = readFileStamp(phonesFile, phonesSource); result.failed())
        {
            return result;
        }
        if (const auto result = readFileStamp(dictionaryFile, dictionarySource); result.failed())
        {
            return result;
        }
        const auto found = std::find_if(dictionaries.begin(), dictionaries.end(), [&phonesSource, &dictionarySource](const CachedDictionary& entry)
                                        { return entry.phonesSource == phonesSource && entry.dictionarySource == dictionarySource; });
        if (found != dictionaries.end())
        {
            std::rotate(found, std::next(found), dictionaries.end());
            return juce::Result::ok();
        }
        synthesis::PhonemeDictionary dictionary;
        if (const auto result = dictionary.load(phonesSource.file, dictionarySource.file); result.failed())
        {
            return result;
        }
        if (dictionaries.size() == 4)
        {
            dictionaries.erase(dictionaries.begin());
        }
        dictionaries.push_back({phonesSource, dictionarySource, {}, {}, {}, {}, std::move(dictionary)});
        return juce::Result::ok();
    };
    const auto resolveAsEnglish = [&](const synthesis::PhonemeDictionary& localDictionary)
    {
        if (settings.language != "japanese" || !latinWord || settings.dictionaryDirectory.empty())
        {
            return false;
        }
        const bool localHasEntry = localDictionary.hasEntry(note.lyrics);
        if (loadEnglishDictionary().failed())
        {
            return false;
        }
        auto& englishDictionary = dictionaries.back().dictionary;
        const bool startsUppercase = !note.lyrics.empty() && std::isupper(static_cast<unsigned char>(note.lyrics.front())) != 0;
        if (!englishDictionary.hasEntry(note.lyrics) || (!preferEnglishContext && !startsUppercase && localHasEntry))
        {
            return false;
        }
        if (englishDictionary.lookup(note.lyrics, phonemes).failed())
        {
            return false;
        }
        phonemeLanguage = "english";
        selectContinuationPhoneme(englishDictionary);
        return true;
    };
    const auto stem = dictionaryStem(settings.language);
    if (stem.isEmpty())
    {
        return juce::Result::fail("This language requires explicit phonemes; automatic grapheme-to-phoneme conversion is not implemented.");
    }
    if (settings.dictionaryDirectory.empty())
    {
        return juce::Result::fail("Select the clf-data pronunciation dictionary directory or enter explicit phonemes.");
    }
    const juce::File directory(juce::String::fromUTF8(settings.dictionaryDirectory.c_str()));
    auto file = directory.getChildFile(stem + "-dict.txt");
    if (settings.language == "english")
    {
        const auto cmuDictionary = directory.getChildFile("cmudict-07b.txt");
        if (cmuDictionary.existsAsFile())
        {
            file = cmuDictionary;
        }
    }
    FileStamp phonesSource;
    FileStamp dictionarySource;
    if (const auto result = readFileStamp(directory.getChildFile(stem + "-phones.txt"), phonesSource); result.failed())
    {
        return result;
    }
    if (const auto result = readFileStamp(file, dictionarySource); result.failed())
    {
        return result;
    }
    std::optional<FileStamp> readingsSource;
    std::optional<FileStamp> hiraganaSource;
    std::optional<FileStamp> katakanaSource;
    std::optional<FileStamp> smallKanaSource;
    if (settings.language == "mandarin")
    {
        readingsSource.emplace();
        if (const auto result = readFileStamp(directory.getChildFile("cedict.txt"), *readingsSource); result.failed())
        {
            return result;
        }
    }
    else if (settings.language == "japanese")
    {
        hiraganaSource.emplace();
        katakanaSource.emplace();
        smallKanaSource.emplace();
        if (const auto result = readFileStamp(directory.getChildFile("japanese-hira2romaji-dict.txt"), *hiraganaSource); result.failed())
        {
            return result;
        }
        if (const auto result = readFileStamp(directory.getChildFile("japanese-kata2romaji-dict.txt"), *katakanaSource); result.failed())
        {
            return result;
        }
        if (const auto result = readFileStamp(directory.getChildFile("japanese-sute2romaji-dict.txt"), *smallKanaSource); result.failed())
        {
            return result;
        }
    }
    const auto found = std::find_if(dictionaries.begin(), dictionaries.end(), [&phonesSource, &dictionarySource, &readingsSource, &hiraganaSource, &katakanaSource, &smallKanaSource](const CachedDictionary& entry)
                                    { return entry.phonesSource == phonesSource && entry.dictionarySource == dictionarySource && entry.readingsSource == readingsSource && entry.hiraganaSource == hiraganaSource && entry.katakanaSource == katakanaSource && entry.smallKanaSource == smallKanaSource; });
    if (found != dictionaries.end())
    {
        std::rotate(found, std::next(found), dictionaries.end());
        if (resolveAsEnglish(dictionaries.back().dictionary))
        {
            return juce::Result::ok();
        }
        const auto local = std::find_if(dictionaries.begin(), dictionaries.end(), [&phonesSource, &dictionarySource, &readingsSource, &hiraganaSource, &katakanaSource, &smallKanaSource](const CachedDictionary& entry)
                                        { return entry.phonesSource == phonesSource && entry.dictionarySource == dictionarySource && entry.readingsSource == readingsSource && entry.hiraganaSource == hiraganaSource && entry.katakanaSource == katakanaSource && entry.smallKanaSource == smallKanaSource; });
        if (local != dictionaries.end())
        {
            std::rotate(local, std::next(local), dictionaries.end());
        }
        const auto result = dictionaries.back().dictionary.lookup(note.lyrics, phonemes);
        if (result.wasOk())
        {
            selectContinuationPhoneme(dictionaries.back().dictionary);
        }
        return result;
    }
    synthesis::PhonemeDictionary dictionary;
    const auto loadResult = readingsSource.has_value() ? dictionary.loadMandarin(phonesSource.file, dictionarySource.file, readingsSource->file)
                              : hiraganaSource.has_value() && katakanaSource.has_value() && smallKanaSource.has_value() ? dictionary.loadJapanese(phonesSource.file, dictionarySource.file, hiraganaSource->file, katakanaSource->file, smallKanaSource->file)
                                                                                          : dictionary.load(phonesSource.file, dictionarySource.file);
    if (loadResult.failed())
    {
        return loadResult;
    }
    if (dictionaries.size() == 4)
    {
        dictionaries.erase(dictionaries.begin());
    }
    dictionaries.push_back({phonesSource, dictionarySource, readingsSource, hiraganaSource, katakanaSource, smallKanaSource, std::move(dictionary)});
    if (resolveAsEnglish(dictionaries.back().dictionary))
    {
        return juce::Result::ok();
    }
    const auto local = std::find_if(dictionaries.begin(), dictionaries.end(), [&phonesSource, &dictionarySource, &readingsSource, &hiraganaSource, &katakanaSource, &smallKanaSource](const CachedDictionary& entry)
                                    { return entry.phonesSource == phonesSource && entry.dictionarySource == dictionarySource && entry.readingsSource == readingsSource && entry.hiraganaSource == hiraganaSource && entry.katakanaSource == katakanaSource && entry.smallKanaSource == smallKanaSource; });
    if (local != dictionaries.end())
    {
        std::rotate(local, std::next(local), dictionaries.end());
    }
    const auto result = dictionaries.back().dictionary.lookup(note.lyrics, phonemes);
    if (result.wasOk())
    {
        selectContinuationPhoneme(dictionaries.back().dictionary);
        return result;
    }
    if (settings.language == "japanese" && latinWord && loadEnglishDictionary().wasOk())
    {
        const auto english = dictionaries.back().dictionary.lookup(note.lyrics, phonemes);
        if (english.wasOk())
        {
            phonemeLanguage = "english";
            selectContinuationPhoneme(dictionaries.back().dictionary);
            return english;
        }
    }
    return result;
}

bool ProjectRenderer::matchesTiming(const CachedPhrase& cached, const FileStamp& voiceSource, const std::vector<synthesis::TimingSyllable>& syllables)
{
    return cached.voiceSource == voiceSource && cached.syllables.size() == syllables.size() && std::equal(cached.syllables.begin(), cached.syllables.end(), syllables.begin(), [](const auto& first, const auto& second)
                                                                                                          { return first.language == second.language && first.phonemes == second.phonemes && first.durationSeconds == second.durationSeconds && first.midiPitch == second.midiPitch && first.isContinuation == second.isContinuation; });
}

std::size_t ProjectRenderer::phraseBytes(const CachedPhrase& phrase)
{
    std::size_t bytes = sizeof(CachedPhrase) + (phrase.samples.capacity() + phrase.logF0.capacity() + phrase.pitchEnvelope.capacity() + phrase.automaticPitch.capacity()) * sizeof(float) + phrase.syllables.capacity() * sizeof(synthesis::TimingSyllable) + phrase.phonemes.capacity() * sizeof(synthesis::TimedPhoneme) + phrase.pitchNotes.capacity() * sizeof(synthesis::PitchNote);
    bytes += phrase.voiceSource.file.getFullPathName().getNumBytesAsUTF8();
    if (phrase.visualization != nullptr)
    {
        bytes += phrase.visualization->getBytes();
    }
    for (const auto& syllable : phrase.syllables)
    {
        bytes += syllable.language.capacity() + syllable.phonemes.capacity() * sizeof(std::string);
        for (const auto& phoneme : syllable.phonemes)
        {
            bytes += phoneme.capacity();
        }
    }
    for (const auto& note : phrase.pitchNotes)
    {
        bytes += note.syllable.language.capacity() + note.syllable.phonemes.capacity() * sizeof(std::string);
        for (const auto& phoneme : note.syllable.phonemes)
        {
            bytes += phoneme.capacity();
        }
    }
    for (const auto& phoneme : phrase.phonemes)
    {
        bytes += phoneme.language.capacity() + phoneme.symbol.capacity();
    }
    return bytes;
}

void ProjectRenderer::retainPhrase(CachedPhrase phrase)
{
    phrase.bytes = phraseBytes(phrase);
    if (phrase.bytes > maximumPhraseCacheBytes)
    {
        return;
    }
    while (!phrases.empty() && (phrases.size() >= maximumCachedPhrases || phraseCacheBytes > maximumPhraseCacheBytes - phrase.bytes))
    {
        phraseCacheBytes -= phrases.front().bytes;
        phrases.pop_front();
    }
    phrases.push_back(std::move(phrase));
    phraseCacheBytes += phrases.back().bytes;
}

const RenderStatistics& ProjectRenderer::getStatistics() const noexcept
{
    return statistics;
}

ProjectRenderer::CachedInference ProjectRenderer::takeInference(const FileStamp& voiceSource, const std::string& trackGroupId, Blick phraseStart)
{
    const auto found = std::find_if(inferenceStates.begin(), inferenceStates.end(), [&](const CachedInference& entry)
                                    { return entry.voiceSource == voiceSource && entry.trackGroupId == trackGroupId && entry.phraseStart == phraseStart; });
    if (found == inferenceStates.end())
    {
        return {voiceSource, trackGroupId, phraseStart, {}, 0};
    }
    auto state = std::move(*found);
    inferenceCacheBytes -= state.bytes;
    inferenceStates.erase(found);
    return state;
}

void ProjectRenderer::retainInference(CachedInference inference)
{
    const auto stateBytes = inference.state.getBytes();
    if (stateBytes == 0)
    {
        return;
    }
    inference.bytes = stateBytes + sizeof(CachedInference) + inference.trackGroupId.capacity() + inference.voiceSource.file.getFullPathName().getNumBytesAsUTF8();
    if (inference.bytes > maximumInferenceCacheBytes)
    {
        return;
    }
    while (!inferenceStates.empty() && (inferenceStates.size() >= maximumCachedInferenceStates || inferenceCacheBytes > maximumInferenceCacheBytes - inference.bytes))
    {
        inferenceCacheBytes -= inferenceStates.front().bytes;
        inferenceStates.pop_front();
    }
    inferenceStates.push_back(std::move(inference));
    inferenceCacheBytes += inferenceStates.back().bytes;
}

juce::Result ProjectRenderer::render(const Project& project, double sampleRate, juce::AudioBuffer<float>& output, const std::function<bool()>& shouldCancel, RenderVisualization* visualization)
{
    statistics = {};
    if (visualization != nullptr)
    {
        *visualization = {};
        visualization->tracks.resize(project.tracks.size());
        for (std::size_t index = 0; index < project.tracks.size(); ++index)
        {
            visualization->tracks[index].mainGroupId = project.tracks[index].mainGroup.id;
        }
    }
    const double started = juce::Time::getMillisecondCounterHiRes();
    const juce::ScopeGuard finishStatistics([this, started]
                                            { statistics.elapsedMilliseconds = juce::Time::getMillisecondCounterHiRes() - started;
                                             statistics.cachedAudioBytes = phraseCacheBytes;
                                             statistics.cachedInferenceBytes = inferenceCacheBytes; });
    if (!std::isfinite(sampleRate) || sampleRate < 8000.0 || sampleRate > 384000.0)
    {
        return juce::Result::fail("The output sample rate must be between 8000 and 384000 Hz.");
    }
    for (const auto& tempo : project.tempoMap.tempos)
    {
        if (!std::isfinite(tempo.bpm) || tempo.bpm <= 0.0)
        {
            return juce::Result::fail("The tempo map contains an invalid tempo.");
        }
    }
    const bool hasSolo = std::any_of(project.tracks.begin(), project.tracks.end(), [](const Track& track)
                                     { return track.solo; });
    std::vector<RenderTrack> tracks;
    double endSeconds = 0.0;
    const Blick projectEnd = getProjectEnd(project);
    if (projectEnd > 0)
    {
        const double scoreEndSeconds = project.tempoMap.blickToSeconds(projectEnd);
        if (!std::isfinite(scoreEndSeconds) || scoreEndSeconds <= 0.0)
        {
            return juce::Result::fail("The project has an invalid duration after applying the tempo map.");
        }
        // Mixer state changes audibility, never the score's playback timeline.
        endSeconds = scoreEndSeconds + restContextSeconds;
    }
    for (std::size_t index = 0; index < project.tracks.size(); ++index)
    {
        if (cancelled(shouldCancel))
        {
            return cancellation();
        }
        const auto& track = project.tracks[index];
        const bool audible = !track.mute && (!hasSolo || track.solo) && track.gain != 0.0;
        if (!audible && visualization == nullptr)
        {
            continue;
        }
        RenderTrack prepared{&track, index, audible, {}};
        if (const auto result = prepareTrack(project, prepared); result.failed())
        {
            if (audible)
            {
                return trackError(track, result.getErrorMessage());
            }
            continue;
        }
        if (!prepared.notes.empty())
        {
            if (audible)
            {
                const auto& last = prepared.notes.back();
                const double releaseEnd = last.reference->absoluteEnd >= 0 ? std::min(last.endSeconds + restContextSeconds, project.tempoMap.blickToSeconds(last.reference->absoluteEnd)) : last.endSeconds + restContextSeconds;
                endSeconds = std::max(endSeconds, releaseEnd);
            }
            tracks.push_back(std::move(prepared));
        }
    }
    const double sampleCount = std::ceil(endSeconds * sampleRate);
    constexpr auto maximumSamples = maximumAudioBytes / (2 * sizeof(float));
    if (!std::isfinite(sampleCount) || sampleCount > static_cast<double>(maximumSamples))
    {
        return juce::Result::fail("The project exceeds the 128 MiB stereo rendering buffer limit at this sample rate.");
    }
    output.setSize(2, static_cast<int>(sampleCount));
    output.clear();
    for (const auto& prepared : tracks)
    {
        const auto& track = *prepared.track;
        const auto renderTrack = [&]() -> juce::Result
        {
            synthesis::VoiceSynthesizer* voice = nullptr;
            if (cancelled(shouldCancel))
            {
                return cancellation();
            }
            if (track.voice.databasePath.empty())
            {
                return trackError(track, "Select a voice.nofs database before playing singing notes.");
            }
            FileStamp voiceSource;
            if (const auto result = readFileStamp(juce::File(juce::String::fromUTF8(track.voice.databasePath.c_str())), voiceSource); result.failed())
            {
                return trackError(track, result.getErrorMessage());
            }
            for (std::size_t begin = 0; begin < prepared.notes.size();)
            {
                if (cancelled(shouldCancel))
                {
                    return cancellation();
                }
                std::size_t end = begin + 1;
                while (end < prepared.notes.size() && prepared.notes[end].startSeconds - prepared.notes[end - 1].endSeconds < phraseBreakSeconds)
                {
                    ++end;
                }
                const auto notes = std::span(prepared.notes).subspan(begin, end - begin);
                // The duration model places onset consonants in the preceding rest.
                // Keep that context even before zero or a group crop boundary; only
                // the mixed output is cropped, never the model's input intervals.
                const double cropStartSeconds = project.tempoMap.blickToSeconds(notes.front().reference->absoluteBegin);
                const double startSeconds = notes.front().startSeconds - restContextSeconds;
                const double releaseEnd = notes.back().endSeconds + restContextSeconds;
                const double stopSeconds = notes.back().reference->absoluteEnd >= 0 ? std::min(releaseEnd, project.tempoMap.blickToSeconds(notes.back().reference->absoluteEnd)) : releaseEnd;
                if (releaseEnd - startSeconds > maximumPhraseSeconds)
                {
                    return trackError(track, noteContext(notes.front()) + "the connected phrase exceeds 30 seconds; add a rest of at least 0.2 seconds to separate phrases.");
                }
                std::vector<synthesis::TimingSyllable> syllables;
                std::vector<synthesis::PitchNote> pitchNotes;
                std::string continuationPhoneme;
                std::string continuationLanguage = track.voice.language;
                double cursorSeconds = startSeconds;
                std::size_t phonemeCount = 0;
                for (std::size_t noteIndex = 0; noteIndex < notes.size(); ++noteIndex)
                {
                    const auto& note = notes[noteIndex];
                    if (cancelled(shouldCancel))
                    {
                        return cancellation();
                    }
                    const double gapSeconds = noteIndex == 0 ? restContextSeconds : note.startSeconds - cursorSeconds;
                    if (gapSeconds > 0.0)
                    {
                        continuationPhoneme.clear();
                        syllables.push_back({track.voice.language, {"sil"}, gapSeconds, note.pitch, false});
                        synthesis::PitchNote silence;
                        silence.syllable = syllables.back();
                        silence.isSilence = true;
                        if (noteIndex != 0)
                        {
                            pitchNotes.push_back(std::move(silence));
                        }
                        ++phonemeCount;
                    }
                    std::vector<std::string> phonemes;
                    std::string phonemeLanguage = track.voice.language;
                    const bool isContinuation = note.note->lyrics == "+" || note.note->lyrics == "-";
                    if (isContinuation)
                    {
                        if (continuationPhoneme.empty())
                        {
                            return trackError(track, noteContext(note) + "a continuation lyric (+/-) must immediately follow a note with a resolved syllable.");
                        }
                        phonemes.push_back(continuationPhoneme);
                        phonemeLanguage = continuationLanguage;
                    }
                    else if (const auto result = resolvePhonemes(track.voice, *note.note, phonemes, continuationPhoneme, phonemeLanguage, continuationLanguage == "english"); result.failed())
                    {
                        return trackError(track, noteContext(note) + result.getErrorMessage());
                    }
                    continuationLanguage = phonemeLanguage;
                    if (phonemes.size() == 1 && (phonemes.front() == "sil" || phonemes.front() == "br"))
                    {
                        continuationPhoneme.clear();
                        continuationLanguage = track.voice.language;
                    }
                    phonemeCount += phonemes.size();
                    if (phonemeCount >= maximumPhrasePhonemes)
                    {
                        return trackError(track, "The phrase exceeds the 4096-phoneme inference limit.");
                    }
                    syllables.push_back({phonemeLanguage, std::move(phonemes), note.endSeconds - note.startSeconds, note.pitch, isContinuation});
                    synthesis::PitchNote pitchNote;
                    pitchNote.syllable = syllables.back();
                    pitchNote.isSilence = pitchNote.syllable.phonemes.size() == 1 && pitchNote.syllable.phonemes.front() == "sil";
                    pitchNote.isBreath = pitchNote.syllable.phonemes.size() == 1 && pitchNote.syllable.phonemes.front() == "br";
                    pitchNote.isContinuation = isContinuation;
                    pitchNote.isRap = note.note->musicalType == "rap";
                    if (note.note->accent.size() == 1 && note.note->accent.front() >= '1' && note.note->accent.front() <= '5')
                    {
                        pitchNote.tone = note.note->accent.front() - '0';
                    }
                    pitchNote.vibratoModulation = static_cast<float>(note.note->attributes.dF0VbrMod.value_or(note.reference->voicePitch.dF0VbrMod.value_or(1.0)));
                    pitchNotes.push_back(std::move(pitchNote));
                    cursorSeconds = note.endSeconds;
                }
                syllables.push_back({track.voice.language, {"sil"}, restContextSeconds, notes.back().pitch, false});
                ++statistics.totalPhrases;
                // Timing can be reused before loading a model. The pitch comparison
                // below uses exactly the samples sent to the acoustic model, including
                // curve points and tempo effects outside the edited note itself.
                const auto timingEntry = std::find_if(phrases.begin(), phrases.end(), [&voiceSource, &syllables](const CachedPhrase& entry)
                                                      { return matchesTiming(entry, voiceSource, syllables); });
                std::vector<synthesis::TimedPhoneme> phonemes;
                float frameInterval = 0.0f;
                if (timingEntry != phrases.end())
                {
                    frameInterval = timingEntry->frameIntervalSeconds;
                    phonemes = timingEntry->phonemes;
                }
                else
                {
                    if (voice == nullptr)
                    {
                        if (const auto result = findVoice(voiceSource, voice); result.failed())
                        {
                            return trackError(track, result.getErrorMessage());
                        }
                    }
                    frameInterval = voice->getFrameIntervalSeconds();
                    std::vector<synthesis::PhonemeDuration> durations;
                    const double durationStarted = juce::Time::getMillisecondCounterHiRes();
                    const auto durationResult = voice->predict(syllables, durations);
                    statistics.durationMilliseconds += juce::Time::getMillisecondCounterHiRes() - durationStarted;
                    if (durationResult.failed())
                    {
                        return trackError(track, noteContext(notes.front()) + "duration prediction: " + durationResult.getErrorMessage());
                    }
                    if (const auto result = synthesis::quantizePhonemeDurations(durations, frameInterval, phonemes); result.failed())
                    {
                        return trackError(track, noteContext(notes.front()) + "duration quantization: " + result.getErrorMessage());
                    }
                    mergeContinuationPhones(durations, syllables, phonemes);
                }
                if (cancelled(shouldCancel))
                {
                    return cancellation();
                }
                std::size_t frameCount = 0;
                for (const auto& phoneme : phonemes)
                {
                    frameCount += phoneme.frameCount;
                }
                if (static_cast<double>(frameCount) * static_cast<double>(frameInterval) > maximumPhraseSeconds + static_cast<double>(frameInterval))
                {
                    return trackError(track, "The predicted phrase exceeds the 30-second inference limit.");
                }
                auto inference = takeInference(voiceSource, track.mainGroup.id, notes.front().start);
                const juce::ScopeGuard retainState([this, &inference]() noexcept
                                                   {
                    try
                    {
                        retainInference(std::move(inference));
                    }
                    catch (const std::bad_alloc&)
                    {
                        // A cache allocation is optional; never terminate while
                        // unwinding a cancelled or failed synthesis operation.
                    } });
                std::vector<float> automaticPitch;
                std::vector<float> pitchEnvelope;
                const double pitchStartSeconds = notes.front().startSeconds - synthesis::VoiceSynthesizer::pitchContextSeconds;
                float pitchFrameInterval = 0.0f;
                if (std::any_of(notes.begin(), notes.end(), [](const RenderNote& note)
                                { return note.note->instantMode; }))
                {
                    const auto pitchConfiguration = std::find_if(phrases.begin(), phrases.end(), [&voiceSource](const CachedPhrase& entry)
                                                                 { return entry.voiceSource == voiceSource && entry.pitchFrameIntervalSeconds > 0.0f; });
                    if (pitchConfiguration != phrases.end())
                    {
                        pitchFrameInterval = pitchConfiguration->pitchFrameIntervalSeconds;
                    }
                    else
                    {
                        if (voice == nullptr)
                        {
                            if (const auto result = findVoice(voiceSource, voice); result.failed())
                            {
                                return trackError(track, result.getErrorMessage());
                            }
                        }
                        pitchFrameInterval = voice->getPitchFrameIntervalSeconds();
                    }
                    const double pitchDuration = notes.back().endSeconds - notes.front().startSeconds + 2.0 * synthesis::VoiceSynthesizer::pitchContextSeconds;
                    const auto pitchEnvelopeCount = static_cast<std::size_t>(std::ceil(pitchDuration / static_cast<double>(pitchFrameInterval))) + 2;
                    pitchEnvelope = buildVibratoEnvelope(project, notes, pitchStartSeconds, pitchFrameInterval, pitchEnvelopeCount);
                    const auto pitchEntry = std::find_if(phrases.begin(), phrases.end(), [&voiceSource, &syllables, &pitchNotes, &pitchEnvelope, pitchFrameInterval](const CachedPhrase& entry)
                                                         { return matchesTiming(entry, voiceSource, syllables) && entry.pitchFrameIntervalSeconds == pitchFrameInterval && !entry.automaticPitch.empty() && samePitchNotes(entry.pitchNotes, pitchNotes) && entry.pitchEnvelope == pitchEnvelope; });
                    if (pitchEntry != phrases.end())
                    {
                        automaticPitch = pitchEntry->automaticPitch;
                    }
                    else
                    {
                        if (voice == nullptr)
                        {
                            if (const auto result = findVoice(voiceSource, voice); result.failed())
                            {
                                return trackError(track, result.getErrorMessage());
                            }
                        }
                        if (const auto result = voice->predictPitch(pitchNotes, automaticPitch, shouldCancel, &statistics.synthesis, &inference.state, pitchEnvelope); result.failed())
                        {
                            return trackError(track, noteContext(notes.front()) + "pitch prediction: " + result.getErrorMessage());
                        }
                    }
                }
                std::vector<float> logF0;
                if (const auto result = buildPitch(project, notes, startSeconds, frameInterval, frameCount, automaticPitch, pitchStartSeconds, pitchFrameInterval, logF0); result.failed())
                {
                    return trackError(track, result.getErrorMessage());
                }
                const auto cached = std::find_if(phrases.begin(), phrases.end(), [&voiceSource, &syllables, &logF0](const CachedPhrase& entry)
                                                 { return matchesTiming(entry, voiceSource, syllables) && entry.logF0 == logF0; });
                if (cached != phrases.end())
                {
                    ++statistics.reusedPhrases;
                    phrases.splice(phrases.end(), phrases, cached);
                    if (prepared.audible)
                    {
                        if (const auto result = mixPhrase(cached->samples, cached->sampleRate, startSeconds, cropStartSeconds, stopSeconds, sampleRate, track, output, shouldCancel); result.failed())
                        {
                            return trackError(track, result.getErrorMessage());
                        }
                    }
                    if (visualization != nullptr)
                    {
                        appendVisualization(visualization->tracks[prepared.projectIndex], track, notes, startSeconds, stopSeconds, cached->visualization);
                    }
                    if (!automaticPitch.empty() && (cached->pitchFrameIntervalSeconds != pitchFrameInterval || !samePitchNotes(cached->pitchNotes, pitchNotes) || cached->pitchEnvelope != pitchEnvelope))
                    {
                        // Distinct prediction inputs may still produce identical
                        // final F0. Preserve their complete prediction on this hit.
                        auto updated = std::move(*cached);
                        phraseCacheBytes -= updated.bytes;
                        phrases.erase(cached);
                        updated.pitchNotes = std::move(pitchNotes);
                        updated.pitchEnvelope = std::move(pitchEnvelope);
                        updated.automaticPitch = std::move(automaticPitch);
                        updated.pitchFrameIntervalSeconds = pitchFrameInterval;
                        retainPhrase(std::move(updated));
                    }
                    begin = end;
                    continue;
                }
                if (voice == nullptr)
                {
                    if (const auto result = findVoice(voiceSource, voice); result.failed())
                    {
                        return trackError(track, result.getErrorMessage());
                    }
                }
                synthesis::NeuralVocoderOutput rendered;
                {
                    // Each network commits only complete results. Even if a later
                    // stage is cancelled, those independent cache entries remain valid.
                    const auto result = voice->render(phonemes, logF0, rendered, 5489, shouldCancel, &statistics.synthesis, &inference.state);
                    if (result.failed())
                    {
                        return trackError(track, noteContext(notes.front()) + result.getErrorMessage());
                    }
                }
                if (prepared.audible)
                {
                    if (const auto result = mixPhrase(rendered.samples, rendered.sampleRate, startSeconds, cropStartSeconds, stopSeconds, sampleRate, track, output, shouldCancel); result.failed())
                    {
                        return trackError(track, result.getErrorMessage());
                    }
                }
                auto display = makeVisualization(rendered, frameInterval, logF0);
                if (visualization != nullptr)
                {
                    appendVisualization(visualization->tracks[prepared.projectIndex], track, notes, startSeconds, stopSeconds, display);
                }
                ++statistics.renderedPhrases;
                // Cache only complete native mono output. Mixer settings, track/note
                // identities and placement do not participate in neural inference.
                retainPhrase({voiceSource, std::move(syllables), std::move(phonemes), std::move(pitchNotes), std::move(pitchEnvelope), std::move(automaticPitch), pitchFrameInterval, std::move(logF0), frameInterval, rendered.sampleRate, std::move(rendered.samples), std::move(display), 0});
                begin = end;
            }
            return juce::Result::ok();
        };
        const auto result = renderTrack();
        if (cancelled(shouldCancel))
        {
            return cancellation();
        }
        if (result.failed())
        {
            if (prepared.audible)
            {
                return result;
            }
            // Inaudible tracks may have incomplete voice settings. Their display
            // is optional and must not prevent the audible project from playing.
            visualization->tracks[prepared.projectIndex].phrases.clear();
        }
    }
    for (int channel = 0; channel < output.getNumChannels(); ++channel)
    {
        auto* samples = output.getWritePointer(channel);
        for (int sample = 0; sample < output.getNumSamples(); ++sample)
        {
            if ((sample & 4095) == 0 && cancelled(shouldCancel))
            {
                return cancellation();
            }
            if (!std::isfinite(samples[sample]))
            {
                return juce::Result::fail("Track mixing produced a non-finite audio sample.");
            }
            samples[sample] = std::clamp(samples[sample], -1.0f, 1.0f);
        }
    }
    return juce::Result::ok();
}
} // namespace sv::audio
