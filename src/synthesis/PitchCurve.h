#pragma once

#include <cstddef>
#include <span>
#include <vector>

namespace sv::synthesis
{
struct PitchCurveNote
{
    double startSeconds = 0.0;
    double endSeconds = 0.0;
    double midiPitch = 60.0;
    double secondsPerQuarter = 0.5;
    bool instantMode = true;

    // Resolved note/voice attributes: times in seconds, depths in semitones,
    // frequency in Hz, and phase in radians. Vibrato depth is peak to peak.
    double tF0Offset = 0.0;
    double tF0Left = 0.1;
    double tF0Right = 0.07;
    double dF0Left = 0.15;
    double dF0Right = 0.15;
    double tF0VbrStart = 0.25;
    double tF0VbrLeft = 0.2;
    double tF0VbrRight = 0.2;
    double dF0Vbr = 1.0;
    double pF0Vbr = 0.0;
    double fF0Vbr = 5.5;
    double rapIntonation = 0.0;
};

// All note fields must be finite. Notes must be ordered and nonoverlapping,
// with positive durations and secondsPerQuarter > 0.
// The first sample is at firstFrameSeconds; callers choose their own frame origin.
// Gaps connect neighboring pitches but do not define the acoustic voiced mask.
[[nodiscard]] std::vector<float> renderPitchTransitions(std::span<const PitchCurveNote> notes, double firstFrameSeconds, double frameIntervalSeconds, std::size_t frameCount);
// Kept separate so the parameter curve can scale vibrato without scaling portamento.
[[nodiscard]] std::vector<float> renderPitchVibrato(std::span<const PitchCurveNote> notes, double firstFrameSeconds, double frameIntervalSeconds, std::size_t frameCount);
// Gen5's residual reference after preprocessing: no offset, overshoot or vibrato.
// Supply the prediction records' score pitches and tF0Left/tF0Right, rather than
// the final editor's resolved user/system/voice attributes.
[[nodiscard]] std::vector<float> renderPitchPredictionBase(std::span<const PitchCurveNote> notes, double firstFrameSeconds, double frameIntervalSeconds, std::size_t frameCount);
// Smoothly gates the predicted residual across adjacent manual/automatic notes.
[[nodiscard]] std::vector<float> renderAutomaticPitchMask(std::span<const PitchCurveNote> notes, double firstFrameSeconds, double frameIntervalSeconds, std::size_t frameCount);
} // namespace sv::synthesis
