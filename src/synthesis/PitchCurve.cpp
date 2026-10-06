#include "PitchCurve.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <numbers>
#include <span>
#include <vector>

namespace sv::synthesis
{
namespace
{
// Mathematical constants for analytical pitch curve generation.
// curvePi is kept at float-precision equivalent (3.14159) to match legacy Synthesizer V analytical curve evaluations.
constexpr double curvePi = 3.14159;

/// Transition width scaling constant. Transition duration is scaled by 0.3 relative to tF0Left / tF0Right.
constexpr double transitionWidthScale = 0.3;

/// Offset multiplier (in units of transition width) for the Gaussian overshoot/undershoot lobes.
constexpr double transitionSwingOffset = 1.5;

/**
 * @struct PitchSegment
 * @brief Discrete temporal slice between notes or rests used for analytical piecewise interpolation.
 */
struct PitchSegment
{
    double startSeconds = 0.0;
    double endSeconds = 0.0;
    double midiPitch = 0.0;
    const PitchCurveNote* note = nullptr; ///< nullptr indicates a rest/silence segment.
};

/**
 * @brief Numerically clamped standard logistic sigmoid function: sigma(x) = 1 / (1 + exp(-x)).
 * Clamped to [-10.0, 10.0] to prevent floating point underflow/overflow.
 */
double sigmoid(double value)
{
    if (value < -10.0)
    {
        return 0.0;
    }
    if (value > 10.0)
    {
        return 1.0;
    }
    return 1.0 / (1.0 + std::exp(-value));
}

/**
 * @brief Evaluates an unnormalized symmetric Gaussian bell curve: exp(-0.5 * (x / width)^2).
 * Used to shape the pitch swing lobes (overshoot/undershoot) at transition boundaries.
 */
double gaussian(double position, double width)
{
    const auto normalized = position / width;
    return std::exp(-0.5 * normalized * normalized);
}

/**
 * @brief Computes effective transition width in seconds.
 * Clamps to a minimum floor of 1/256th of a quarter note to avoid infinite slopes.
 */
double transitionWidth(const PitchCurveNote& note, double seconds)
{
    return std::max(seconds * transitionWidthScale, note.secondsPerQuarter / 256.0);
}

/**
 * @brief Partitions the note sequence into alternating note and rest segments.
 *
 * Adds 1 quarter-note silent padding segments before the first note and after the last note
 * so that attack and release curves have well-defined boundary conditions even when rendering
 * partial phrase sub-windows.
 */
std::vector<PitchSegment> makeSegments(std::span<const PitchCurveNote> notes)
{
    std::vector<PitchSegment> segments;
    segments.reserve(notes.size() * 2 + 1);

    // Explicit silent neighbors retain the attack/release shape even when only
    // a small window inside the phrase is being sampled. Their pitches merely
    // continue the nearest note; voicing is supplied by the acoustic model.
    const auto& first = notes.front();
    segments.push_back({first.startSeconds - first.secondsPerQuarter, first.startSeconds, first.midiPitch, nullptr});
    for (const auto& note : notes)
    {
        if (segments.back().endSeconds < note.startSeconds)
        {
            segments.push_back({segments.back().endSeconds, note.startSeconds, note.midiPitch, nullptr});
        }
        segments.push_back({note.startSeconds, note.endSeconds, note.midiPitch, &note});
    }
    const auto& last = notes.back();
    segments.push_back({last.endSeconds, last.endSeconds + last.secondsPerQuarter, last.midiPitch, nullptr});
    return segments;
}

/**
 * @brief Evaluates the continuous pitch for a note/rest transition pair at a given time.
 *
 * Models three transition topologies:
 * 1. Rest -> Note (Attack):
 *    - Sigmoid transition over rest progress: sigmoid(20.0 * (progress - 0.5)).
 *    - Incoming overshoot swing lobe centered at +1.5 * incomingWidth.
 * 2. Note -> Rest (Release):
 *    - Pitch held then transitioning towards following pitch.
 *    - Outgoing undershoot swing lobe centered at -1.5 * outgoingWidth.
 * 3. Note -> Note (Portamento Legato):
 *    - Sigmoid portamento: current + (next - current) * sigmoid(3.0 * relSec / (w_out + w_in)).
 *    - Two opposing Gaussian swing lobes: outgoing swing (-direction * dF0Right) and incoming swing (+direction * dF0Left).
 */
double samplePair(std::span<const PitchSegment> segments, std::size_t index, double seconds)
{
    const auto& current = segments[index];
    if (index + 1 == segments.size())
    {
        return current.midiPitch;
    }

    const auto& next = segments[index + 1];
    const auto boundarySeconds = current.endSeconds;
    double pitch = current.midiPitch;
    double outgoingDepth = 0.0;
    double incomingDepth = 0.0;
    double outgoingWidth = 1.0;
    double incomingWidth = 1.0;
    double offset = 0.0;

    if (current.note == nullptr)
    {
        // A rest bridges the preceding and following score pitches, with the
        // incoming note's attack dip centered on the end of the rest.
        const auto previousPitch = index == 0 ? next.midiPitch : segments[index - 1].midiPitch;
        const auto progress = std::clamp((seconds - current.startSeconds) / (current.endSeconds - current.startSeconds), 0.0, 1.0);
        pitch = seconds < boundarySeconds ? previousPitch + (next.midiPitch - previousPitch) * sigmoid(20.0 * (progress - 0.5)) : next.midiPitch;
        if (next.note != nullptr)
        {
            incomingWidth = transitionWidth(*next.note, next.note->tF0Left);
            incomingDepth = -next.note->dF0Left;
            offset = transitionSwingOffset * incomingWidth;
        }
    }
    else if (next.note == nullptr)
    {
        const auto followingPitch = index + 2 < segments.size() ? segments[index + 2].midiPitch : current.midiPitch;
        const auto progress = std::clamp((seconds - next.startSeconds) / (next.endSeconds - next.startSeconds), 0.0, 1.0);
        pitch = seconds < boundarySeconds ? current.midiPitch : current.midiPitch + (followingPitch - current.midiPitch) * sigmoid(20.0 * (progress - 0.5));
        outgoingWidth = transitionWidth(*current.note, current.note->tF0Right);
        outgoingDepth = -current.note->dF0Right;
        offset = -transitionSwingOffset * outgoingWidth;
    }
    else
    {
        outgoingWidth = transitionWidth(*current.note, current.note->tF0Right);
        incomingWidth = transitionWidth(*next.note, next.note->tF0Left);
        offset = -next.note->tF0Offset;
        const auto relativeSeconds = seconds - boundarySeconds + offset;
        pitch = current.midiPitch + (next.midiPitch - current.midiPitch) * sigmoid(3.0 * relativeSeconds / (outgoingWidth + incomingWidth));
        const auto direction = next.midiPitch > current.midiPitch ? 1.0 : -1.0;
        outgoingDepth = -direction * current.note->dF0Right;
        incomingDepth = direction * next.note->dF0Left;
    }

    const auto relativeSeconds = seconds - boundarySeconds + offset;
    return pitch + outgoingDepth * gaussian(relativeSeconds + transitionSwingOffset * outgoingWidth, outgoingWidth) + incomingDepth * gaussian(relativeSeconds - transitionSwingOffset * incomingWidth, incomingWidth);
}

/**
 * @brief Computes the earliest frame index whose timestamp >= seconds.
 */
std::size_t frameAtOrAfter(double seconds, double firstFrameSeconds, double frameIntervalSeconds, std::size_t frameCount)
{
    const auto frame = std::ceil((seconds - firstFrameSeconds) / frameIntervalSeconds);
    return static_cast<std::size_t>(std::clamp(frame, 0.0, static_cast<double>(frameCount)));
}

/**
 * @brief Evaluates the simplified pitch reference pair for neural pitch prediction.
 * Excludes Gaussian swing lobes and offset nudges, producing a clean, monotonic transition.
 */
double samplePredictionPair(std::span<const PitchSegment> segments, std::size_t index, double seconds)
{
    const auto& current = segments[index];
    if (index + 1 == segments.size())
    {
        return current.midiPitch;
    }
    const auto& next = segments[index + 1];
    if (current.note == nullptr)
    {
        const auto previousPitch = index == 0 ? next.midiPitch : segments[index - 1].midiPitch;
        const auto progress = std::clamp((seconds - current.startSeconds) / (current.endSeconds - current.startSeconds), 0.0, 1.0);
        return previousPitch + (next.midiPitch - previousPitch) * progress;
    }
    if (next.note == nullptr)
    {
        const auto followingPitch = index + 2 < segments.size() ? segments[index + 2].midiPitch : current.midiPitch;
        const auto progress = std::clamp((seconds - next.startSeconds) / (next.endSeconds - next.startSeconds), 0.0, 1.0);
        return current.midiPitch + (followingPitch - current.midiPitch) * progress;
    }

    // Unlike the editor base, the prediction reference does not clamp widths in
    // quarter-note units. Both zero widths have a well-defined step limit.
    const auto width = transitionWidthScale * (current.note->tF0Right + next.note->tF0Left);
    const auto relativeSeconds = seconds - current.endSeconds;
    const auto weight = width > 0.0 ? sigmoid(3.0 * relativeSeconds / width) : (relativeSeconds < 0.0 ? 0.0 : (relativeSeconds > 0.0 ? 1.0 : 0.5));
    return current.midiPitch + (next.midiPitch - current.midiPitch) * weight;
}

/**
 * @brief Core driver for sampling pitch curves over discrete time frames.
 *
 * For each frame, finds the active segment and performs a raised-cosine crossfade:
 * weight = 0.5 - 0.5 * cos(pi * progress)
 * between the previous transition pair and the current transition pair.
 */
std::vector<float> renderPitchBase(std::span<const PitchCurveNote> notes, double firstFrameSeconds, double frameIntervalSeconds, std::size_t frameCount, bool predictionReference)
{
    assert(std::isfinite(firstFrameSeconds) && std::isfinite(frameIntervalSeconds) && frameIntervalSeconds > 0.0);
    std::vector<float> result(frameCount, 0.0f);
    if (notes.empty() || frameCount == 0)
    {
        return result;
    }

    const auto segments = makeSegments(notes);
    const auto sample = predictionReference ? samplePredictionPair : samplePair;
    std::size_t segmentIndex = 0;
    for (std::size_t frame = 0; frame < frameCount; ++frame)
    {
        const auto seconds = firstFrameSeconds + static_cast<double>(frame) * frameIntervalSeconds;
        while (segmentIndex + 1 < segments.size() && segments[segmentIndex].endSeconds <= seconds)
        {
            ++segmentIndex;
        }
        const auto& segment = segments[segmentIndex];
        if (segmentIndex == 0)
        {
            result[frame] = static_cast<float>(sample(segments, segmentIndex, seconds));
            continue;
        }
        if (segmentIndex + 1 == segments.size())
        {
            // The terminal release is not crossfaded against another note, and
            // must not depend on the length of the requested render window.
            result[frame] = static_cast<float>(sample(segments, segmentIndex - 1, seconds));
            continue;
        }

        const auto progress = std::clamp((seconds - segment.startSeconds) / (segment.endSeconds - segment.startSeconds), 0.0, 1.0);
        const auto weight = 0.5 - 0.5 * std::cos(curvePi * progress);
        const auto incoming = sample(segments, segmentIndex - 1, seconds);
        const auto outgoing = sample(segments, segmentIndex, seconds);
        result[frame] = static_cast<float>((1.0 - weight) * incoming + weight * outgoing);
    }
    return result;
}
} // namespace

std::vector<float> renderPitchTransitions(std::span<const PitchCurveNote> notes, double firstFrameSeconds, double frameIntervalSeconds, std::size_t frameCount)
{
    // Evaluates base portamento transitions with full Gaussian overshoot/undershoot dynamics.
    return renderPitchBase(notes, firstFrameSeconds, frameIntervalSeconds, frameCount, false);
}

std::vector<float> renderPitchPredictionBase(std::span<const PitchCurveNote> notes, double firstFrameSeconds, double frameIntervalSeconds, std::size_t frameCount)
{
    // Evaluates neutral reference transitions for neural residual prediction.
    return renderPitchBase(notes, firstFrameSeconds, frameIntervalSeconds, frameCount, true);
}

std::vector<float> renderPitchVibrato(std::span<const PitchCurveNote> notes, double firstFrameSeconds, double frameIntervalSeconds, std::size_t frameCount)
{
    assert(std::isfinite(firstFrameSeconds) && std::isfinite(frameIntervalSeconds) && frameIntervalSeconds > 0.0);
    std::vector<float> result(frameCount, 0.0f);

    // Iterate through each note and synthesize its active vibrato segment
    for (std::size_t index = 0; index < notes.size(); ++index)
    {
        const auto& note = notes[index];
        const auto delay = std::max(0.0, note.tF0VbrStart);
        // If note duration is shorter than vibrato delay, vibrato never begins
        if (delay >= note.endSeconds - note.startSeconds)
        {
            continue;
        }

        const auto startSeconds = note.startSeconds + delay;
        auto endSeconds = note.endSeconds;
        // If connected seamlessly to the next note, extend vibrato up to the next note's transition offset
        if (index + 1 < notes.size() && notes[index + 1].startSeconds == note.endSeconds)
        {
            endSeconds += notes[index + 1].tF0Offset;
        }

        // Clamp fade-in and fade-out times to a minimum floor of 1/128th of a quarter note
        const auto fadeInSeconds = std::max(note.tF0VbrLeft, note.secondsPerQuarter / 128.0);
        const auto fadeOutSeconds = std::max(note.tF0VbrRight, note.secondsPerQuarter / 128.0);
        const auto firstFrame = frameAtOrAfter(startSeconds, firstFrameSeconds, frameIntervalSeconds, frameCount);
        const auto endFrame = frameAtOrAfter(endSeconds, firstFrameSeconds, frameIntervalSeconds, frameCount);

        for (auto frame = firstFrame; frame < endFrame; ++frame)
        {
            const auto seconds = firstFrameSeconds + static_cast<double>(frame) * frameIntervalSeconds;
            const auto elapsed = seconds - startSeconds;
            // Linear attack / release trapezoidal envelope clamped to [0.0, 1.0]
            const auto envelope = std::clamp(std::min(elapsed / fadeInSeconds, (endSeconds - seconds) / fadeOutSeconds), 0.0, 1.0);
            // Instantaneous phase in radians: theta(t) = 2*pi*f*t + phase_0
            const auto phase = 2.0 * curvePi * note.fF0Vbr * elapsed + note.pF0Vbr;
            // Vibrato amplitude is 0.5 * peak-to-peak depth (dF0Vbr)
            result[frame] = static_cast<float>(0.5 * note.dF0Vbr * envelope * std::sin(phase));
        }
    }
    return result;
}

std::vector<float> renderAutomaticPitchMask(std::span<const PitchCurveNote> notes, double firstFrameSeconds, double frameIntervalSeconds, std::size_t frameCount)
{
    assert(std::isfinite(firstFrameSeconds) && std::isfinite(frameIntervalSeconds) && frameIntervalSeconds > 0.0);
    std::vector<float> result(frameCount, 0.0f);
    if (notes.empty())
    {
        return result;
    }

    // Raised cosine interpolation curve: 0.5 - 0.5 * cos(pi * clamp(p, 0, 1))
    const auto raisedCosine = [](double position)
    {
        return 0.5 - 0.5 * std::cos(std::numbers::pi * std::clamp(position, 0.0, 1.0));
    };

    std::size_t noteIndex = 0;
    for (std::size_t frame = 0; frame < frameCount; ++frame)
    {
        const auto seconds = firstFrameSeconds + static_cast<double>(frame) * frameIntervalSeconds;
        while (noteIndex + 1 < notes.size() && notes[noteIndex + 1].startSeconds <= seconds)
        {
            ++noteIndex;
        }
        const auto& note = notes[noteIndex];
        const auto current = note.instantMode ? 1.0 : 0.0;
        auto mask = current;

        // Transition through a rest gap between notes
        if (noteIndex + 1 < notes.size() && note.endSeconds < seconds && note.endSeconds < notes[noteIndex + 1].startSeconds)
        {
            const auto& next = notes[noteIndex + 1];
            const auto nextMode = next.instantMode ? 1.0 : 0.0;
            const auto progress = (seconds - note.endSeconds) / (next.startSeconds - note.endSeconds);
            mask += (nextMode - current) * raisedCosine(progress);
        }
        else
        {
            // Transition near note boundaries (attack and release edges)
            const auto width = std::min((note.endSeconds - note.startSeconds) * 0.5, note.secondsPerQuarter * 0.125);
            if (seconds < note.startSeconds + width)
            {
                auto previous = current;
                if (noteIndex > 0 && notes[noteIndex - 1].endSeconds == note.startSeconds)
                {
                    previous = notes[noteIndex - 1].instantMode ? 1.0 : 0.0;
                }
                mask = previous + (current - previous) * raisedCosine(0.5 + 0.5 * (seconds - note.startSeconds) / width);
            }
            else if (seconds > note.endSeconds - width)
            {
                auto next = current;
                if (noteIndex + 1 < notes.size() && notes[noteIndex + 1].startSeconds == note.endSeconds)
                {
                    next = notes[noteIndex + 1].instantMode ? 1.0 : 0.0;
                }
                mask += (next - current) * raisedCosine(0.5 * (seconds - (note.endSeconds - width)) / width);
            }
        }
        result[frame] = static_cast<float>(mask);
    }
    return result;
}
} // namespace sv::synthesis
