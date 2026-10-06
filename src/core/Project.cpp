#include "Project.h"

#include <juce_core/juce_core.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <limits>
#include <unordered_set>

namespace sv
{
namespace
{
/**
 * @brief Computes the duration in seconds of a single Blick at a given BPM.
 *
 * Formula:
 * 1 quarter note = 60 / BPM seconds.
 * 1 quarter note = blicksPerQuarter (705,600,000) blicks.
 * Therefore, seconds_per_blick = 60.0 / (BPM * blicksPerQuarter).
 */
double secondsPerBlick(double bpm)
{
    return 60.0 / (bpm * static_cast<double>(blicksPerQuarter));
}

/**
 * @brief Safely rounds a floating-point value to a 64-bit integer Blick with saturation.
 *
 * Prevents undefined behavior when floating-point values exceed the range of std::int64_t.
 */
Blick roundedBlick(double value)
{
    constexpr auto lower = std::numeric_limits<Blick>::lowest();
    constexpr auto upper = std::numeric_limits<Blick>::max();
    if (value <= static_cast<double>(lower))
    {
        return lower;
    }
    if (value >= static_cast<double>(upper))
    {
        return upper;
    }
    return static_cast<Blick>(std::llround(value));
}

/**
 * @brief Saturating addition of two Blicks to prevent signed integer overflow.
 */
Blick addBlick(Blick left, Blick right)
{
    if (right > 0 && left > std::numeric_limits<Blick>::max() - right)
    {
        return std::numeric_limits<Blick>::max();
    }
    if (right < 0 && left < std::numeric_limits<Blick>::min() - right)
    {
        return std::numeric_limits<Blick>::min();
    }
    return left + right;
}

/**
 * @brief Computes the effective timeline end position (in blicks) of a referenced group.
 *
 * Takes into account:
 * - Each note's onset and duration.
 * - The reference's timeOffset placement on the track timeline.
 * - Clipping window boundaries: absoluteBegin and absoluteEnd.
 */
Blick referencedGroupEnd(const NoteGroup& group, const GroupReference& reference)
{
    Blick end = 0;
    for (const auto& note : group.notes)
    {
        auto noteEnd = addBlick(addBlick(note.onset, note.duration), reference.timeOffset);
        // Clip against right boundary if specified (non-negative)
        if (reference.absoluteEnd >= 0)
        {
            noteEnd = std::min(noteEnd, reference.absoluteEnd);
        }
        // Only consider notes that extend past the left clipping boundary
        if (noteEnd > reference.absoluteBegin)
        {
            end = std::max(end, noteEnd);
        }
    }
    return end;
}

/**
 * @brief Normalises an automation curve in-place.
 *
 * Enforces curve invariants:
 * 1. Filters out non-finite point values (NaN, +/- Infinity).
 * 2. Sorts points strictly ascending by position (time in blicks).
 * 3. Deduplicates identical positions: if multiple points share the same position,
 *    the last one is kept (matching SV's UI behavior where later edits overwrite).
 */
void normaliseCurve(ParameterCurve& curve)
{
    auto& points = curve.points;
    // Step 1: Remove invalid non-finite numbers
    std::erase_if(points, [](const AutomationPoint& point)
                  { return !std::isfinite(point.value); });

    // Step 2: Stable sort chronologically by position
    std::stable_sort(points.begin(), points.end(), [](const AutomationPoint& left, const AutomationPoint& right)
                     { return left.position < right.position; });

    // Step 3: Remove duplicate positions, preserving the latest point
    points.erase(points.begin(), std::unique(points.rbegin(), points.rend(), [](const AutomationPoint& left, const AutomationPoint& right)
                                             { return left.position == right.position; })
                                     .base());
}
} // namespace

double TempoMap::blickToSeconds(Blick position) const
{
    if (tempos.empty())
    {
        return static_cast<double>(position) * secondsPerBlick(120.0);
    }
    double seconds = 0.0;
    Blick cursor = 0;
    double bpm = tempos.front().bpm;

    // Accumulate duration across each constant-tempo segment
    for (const auto& tempo : tempos)
    {
        if (tempo.position > position)
        {
            break;
        }
        seconds += static_cast<double>(tempo.position - cursor) * secondsPerBlick(bpm);
        cursor = tempo.position;
        bpm = tempo.bpm;
    }

    // Add remaining time from last tempo change to target position
    return seconds + static_cast<double>(position - cursor) * secondsPerBlick(bpm);
}

Blick TempoMap::secondsToBlick(double seconds) const
{
    if (!std::isfinite(seconds))
    {
        return 0;
    }
    if (tempos.empty())
    {
        return roundedBlick(seconds / secondsPerBlick(120.0));
    }
    double elapsed = 0.0;
    Blick cursor = 0;
    double bpm = tempos.front().bpm;

    // Traverse tempo segments until target elapsed time falls within the segment
    for (const auto& tempo : tempos)
    {
        const double segment = static_cast<double>(tempo.position - cursor) * secondsPerBlick(bpm);
        if (elapsed + segment > seconds)
        {
            break;
        }
        elapsed += segment;
        cursor = tempo.position;
        bpm = tempo.bpm;
    }

    // Linearly interpolate remaining seconds at current BPM
    return roundedBlick(static_cast<double>(cursor) + (seconds - elapsed) / secondsPerBlick(bpm));
}

double TempoMap::getTempoAt(Blick position) const
{
    double bpm = tempos.empty() ? 120.0 : tempos.front().bpm;
    for (const auto& tempo : tempos)
    {
        if (tempo.position > position)
        {
            break;
        }
        bpm = tempo.bpm;
    }
    return bpm;
}

NoteId createNoteId()
{
    // Monotonically increasing atomic counter ensures thread-safe unique IDs.
    static std::atomic<NoteId> nextId{1};
    return nextId.fetch_add(1, std::memory_order_relaxed);
}

NoteGroup createNoteGroup(const std::string& name)
{
    NoteGroup group;
    // Assign a standard UUID string for group identity
    group.id = juce::Uuid().toDashedString().toStdString();
    group.name = name;
    return group;
}

Project createEmptyProject()
{
    Project project;
    Track track;
    track.mainGroup = createNoteGroup();
    track.mainRef.groupId = track.mainGroup.id;
    project.tracks.push_back(std::move(track));
    return project;
}

const NoteGroup* findNoteGroup(const Project& project, const std::string& groupId)
{
    // First search the project library
    for (const auto& group : project.library)
    {
        if (group.id == groupId)
        {
            return &group;
        }
    }
    // Next search each track's inline main group
    for (const auto& track : project.tracks)
    {
        if (track.mainGroup.id == groupId)
        {
            return &track.mainGroup;
        }
    }
    return nullptr;
}

Blick getProjectEnd(const Project& project)
{
    Blick end = 0;
    for (const auto& track : project.tracks)
    {
        // Check track's main inline group
        end = std::max(end, referencedGroupEnd(track.mainGroup, track.mainRef));
        // Check all referenced library groups on this track
        for (const auto& reference : track.groups)
        {
            if (const auto* group = findNoteGroup(project, reference.groupId))
            {
                end = std::max(end, referencedGroupEnd(*group, reference));
            }
        }
    }
    return end;
}

void normaliseProject(Project& project)
{
    // Invariant: At least one track must always exist
    if (project.tracks.empty())
    {
        project.tracks = createEmptyProject().tracks;
    }

    std::unordered_set<NoteId> noteIds;

    // Helper lambda to validate and sort notes and curves in a note group
    const auto normaliseGroup = [&noteIds](NoteGroup& group)
    {
        if (group.id.empty())
        {
            group.id = juce::Uuid().toDashedString().toStdString();
        }
        for (auto& note : group.notes)
        {
            // Ensure unique, non-zero NoteId
            while (note.id == 0 || noteIds.contains(note.id))
            {
                note.id = createNoteId();
            }
            noteIds.insert(note.id);

            // Invariant: Duration must be strictly positive
            note.duration = std::max<Blick>(1, note.duration);
            // Invariant: Pitch must be in standard MIDI range [0, 127]
            note.pitch = std::clamp(note.pitch, 0, 127);
            // Invariant: Detune must be a finite floating-point number
            if (!std::isfinite(note.detune))
            {
                note.detune = 0.0;
            }
        }
        // Invariant: Notes in a group are sorted chronologically by onset
        std::stable_sort(group.notes.begin(), group.notes.end(), [](const Note& left, const Note& right)
                         { return left.onset < right.onset; });

        // Normalise automation curves
        normaliseCurve(group.pitchDelta);
        normaliseCurve(group.vibratoEnv);
    };

    // Normalise all groups in library
    for (auto& group : project.library)
    {
        normaliseGroup(group);
    }

    // Normalise all tracks and their groups
    for (auto& track : project.tracks)
    {
        normaliseGroup(track.mainGroup);
        track.mainRef.groupId = track.mainGroup.id;
        normaliseCurve(track.mainRef.systemPitchDelta);
        for (auto& reference : track.groups)
        {
            normaliseCurve(reference.systemPitchDelta);
        }
    }

    // Invariant: Tempo map must have at least one tempo marker at position 0
    auto& tempos = project.tempoMap.tempos;
    if (tempos.empty())
    {
        tempos.push_back({0, 120.0, {}});
    }
    std::stable_sort(tempos.begin(), tempos.end(), [](const Tempo& left, const Tempo& right)
                     { return left.position < right.position; });
    if (tempos.front().position > 0)
    {
        tempos.insert(tempos.begin(), {0, 120.0, {}});
    }

    // Invariant: Time signature map must have at least one entry at measure 0
    auto& signatures = project.tempoMap.timeSignatures;
    if (signatures.empty())
    {
        signatures.push_back({0, 4, 4, {}});
    }
    std::stable_sort(signatures.begin(), signatures.end(), [](const TimeSignature& left, const TimeSignature& right)
                     { return left.bar < right.bar; });
    if (signatures.front().bar > 0)
    {
        signatures.insert(signatures.begin(), {0, 4, 4, {}});
    }
}

} // namespace sv
