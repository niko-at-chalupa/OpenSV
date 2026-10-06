/**
 * @file PitchCurve.h
 * @brief Analytical pitch contour generation (portamento transitions, vibrato, and prediction reference).
 *
 * In Synthesizer V and OpenSV, musical pitch consists of two layers:
 * 1. An analytical base pitch curve generated from score notes and transition/vibrato parameters.
 * 2. A neural residual pitch curve predicted by the DNNI pitch model (in automatic / instant mode).
 *
 * This header defines:
 * - `PitchCurveNote`: Per-note musical and phonetic attributes resolved from project data,
 *   including transition times/depths and vibrato parameters.
 * - `renderPitchTransitions`: Generates the continuous portamento and transition overshoot/undershoot
 *   contour based on sigmoid interpolations and Gaussian swing lobes.
 * - `renderPitchVibrato`: Generates the time-domain sinusoidal vibrato modulation with trapezoidal
 *   fade-in and fade-out envelopes.
 * - `renderPitchPredictionBase`: Generates a simplified, neutral pitch reference contour that serves
 *   as the baseline input for Gen5 neural pitch residual prediction.
 * - `renderAutomaticPitchMask`: Generates a smooth gating mask ([0.0, 1.0]) for blending between
 *   manual user pitch curves and AI-generated automatic pitch curves.
 */

#pragma once

#include <cstddef>
#include <span>
#include <vector>

namespace sv::synthesis
{
/**
 * @struct PitchCurveNote
 * @brief Resolved note attributes required for analytical pitch curve evaluation.
 *
 * All time parameters are expressed in seconds, frequencies in Hertz, depths in semitones,
 * and phases in radians.
 */
struct PitchCurveNote
{
    /// Absolute start time of the note in seconds.
    double startSeconds = 0.0;

    /// Absolute end time of the note in seconds. Must satisfy `endSeconds > startSeconds`.
    double endSeconds = 0.0;

    /// Base musical pitch in semitones (e.g., 60.0 = Middle C, C4).
    double midiPitch = 60.0;

    /// Tempo context: duration of one quarter note in seconds (60.0 / BPM).
    double secondsPerQuarter = 0.5;

    /// Whether "Instant Mode" (AI automatic pitch prediction) is enabled for this note.
    bool instantMode = true;

    // --- Pitch Transition (Portamento & Transition Dynamics) Parameters ---

    /// Transition center time offset in seconds relative to the note boundary.
    double tF0Offset = 0.0;

    /// Duration/width scale of the incoming pitch transition (attack) in seconds.
    double tF0Left = 0.1;

    /// Duration/width scale of the outgoing pitch transition (release) in seconds.
    double tF0Right = 0.07;

    /// Peak depth of the incoming transition overshoot/undershoot in semitones.
    double dF0Left = 0.15;

    /// Peak depth of the outgoing transition overshoot/undershoot in semitones.
    double dF0Right = 0.15;

    // --- Vibrato Parameters ---

    /// Vibrato start delay relative to note onset in seconds.
    double tF0VbrStart = 0.25;

    /// Vibrato attack fade-in duration in seconds.
    double tF0VbrLeft = 0.2;

    /// Vibrato release fade-out duration before note offset in seconds.
    double tF0VbrRight = 0.2;

    /// Peak-to-peak vibrato depth in semitones (e.g. 1.0 = +/- 0.5 semitones amplitude).
    double dF0Vbr = 1.0;

    /// Vibrato starting phase in radians.
    double pF0Vbr = 0.0;

    /// Vibrato oscillation frequency in Hertz (typically ~5.5 Hz).
    double fF0Vbr = 5.5;

    /// Optional rap intonation / speech inflection coefficient.
    double rapIntonation = 0.0;
};

/**
 * @brief Renders the analytical portamento pitch transitions and attack/release swings.
 *
 * Evaluates smooth sigmoid transitions between adjacent notes/rests and adds Gaussian
 * overshoot/undershoot lobes scaled by `dF0Left`/`dF0Right`.
 *
 * @param notes Span of ordered, non-overlapping notes with finite values and positive durations.
 * @param firstFrameSeconds Timestamp of the first frame (frame index 0) in seconds.
 * @param frameIntervalSeconds Temporal hop size between successive frames in seconds (e.g., 0.005 for 5ms).
 * @param frameCount Total number of frames to generate.
 * @return Vector of continuous pitch values in MIDI semitones for each frame.
 */
[[nodiscard]] std::vector<float> renderPitchTransitions(
    std::span<const PitchCurveNote> notes,
    double firstFrameSeconds,
    double frameIntervalSeconds,
    std::size_t frameCount);

/**
 * @brief Renders the analytical vibrato modulation contour in semitones.
 *
 * Evaluates a sinusoidal oscillation `0.5 * dF0Vbr * envelope * sin(2*pi*f*t + phase)`,
 * where `envelope` is a trapezoidal window fading in over `tF0VbrLeft` and fading out over `tF0VbrRight`.
 *
 * @note Rendered separately from portamento so user parameter curves can scale vibrato depth
 *       independently without distorting portamento transitions.
 *
 * @param notes Span of ordered notes.
 * @param firstFrameSeconds Timestamp of frame 0 in seconds.
 * @param frameIntervalSeconds Time step per frame in seconds.
 * @param frameCount Number of frames to evaluate.
 * @return Vector of vibrato offsets in semitones (centered around 0.0).
 */
[[nodiscard]] std::vector<float> renderPitchVibrato(
    std::span<const PitchCurveNote> notes,
    double firstFrameSeconds,
    double frameIntervalSeconds,
    std::size_t frameCount);

/**
 * @brief Renders the neutral pitch reference contour for neural pitch prediction.
 *
 * Unlike `renderPitchTransitions`, this curve excludes vibrato, pitch offsets, and transition
 * overshoots, providing the neutral baseline from which the neural pitch model predicts
 * the expressive pitch residual.
 *
 * @param notes Span of ordered notes with prediction score pitches.
 * @param firstFrameSeconds Timestamp of frame 0 in seconds.
 * @param frameIntervalSeconds Time step per frame in seconds.
 * @param frameCount Number of frames to evaluate.
 * @return Vector of baseline pitch values in MIDI semitones.
 */
[[nodiscard]] std::vector<float> renderPitchPredictionBase(
    std::span<const PitchCurveNote> notes,
    double firstFrameSeconds,
    double frameIntervalSeconds,
    std::size_t frameCount);

/**
 * @brief Renders a smooth gating mask [0.0, 1.0] across automatic and manual notes.
 *
 * Uses raised-cosine crossfades across note boundaries and rests to seamlessly blend
 * between user-edited manual pitch curves (mask = 0.0) and neural automatic pitch predictions (mask = 1.0).
 *
 * @param notes Span of ordered notes.
 * @param firstFrameSeconds Timestamp of frame 0 in seconds.
 * @param frameIntervalSeconds Time step per frame in seconds.
 * @param frameCount Number of frames to evaluate.
 * @return Vector of blend coefficients [0.0, 1.0] for each frame.
 */
[[nodiscard]] std::vector<float> renderAutomaticPitchMask(
    std::span<const PitchCurveNote> notes,
    double firstFrameSeconds,
    double frameIntervalSeconds,
    std::size_t frameCount);

} // namespace sv::synthesis

