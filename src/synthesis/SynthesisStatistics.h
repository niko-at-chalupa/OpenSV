#pragma once

#include <cstddef>

namespace sv::synthesis
{
/**
 * @brief Performance and execution profiling statistics accumulated during synthesis.
 *
 * Tracks the workload and timings across all stages of singing synthesis:
 * - Phoneme timing prediction
 * - Neural pitch prediction (PitchModel)
 * - Neural acoustic feature prediction (AcousticModel)
 * - Neural vocoder synthesis (NeuralVocoder: periodic excitation, residual noise, filter banks)
 *
 * Invariants:
 * - Owned and reset by the caller on the synthesis worker thread.
 * - Thread-confined (not thread-safe by itself; do not share across threads without locks).
 */
struct SynthesisStatistics
{
    // --- Frame counts across synthesis domains ---
    std::size_t pitchFrames = 0;       ///< Number of frames evaluated by the pitch prediction model.
    std::size_t acousticFrames = 0;    ///< Number of acoustic frames evaluated (e.g. 5ms or 10ms frame rate).
    std::size_t vocoderFrames = 0;     ///< Number of frames rendered by the vocoder spectral processing.

    // --- DNNI neural network cache & inference metrics ---
    // Note: Neural networks can operate on different time resolutions (phoneme-rate,
    // acoustic frame-rate, subband rate), so these numbers reflect network execution windows.
    std::size_t dnniComputedFrames = 0; ///< Total neural network output frames computed from scratch.
    std::size_t dnniReusedFrames = 0;   ///< Frames retrieved from the phrase cache without recomputation.
    std::size_t dnniContextFrames = 0;  ///< Overlapping receptive field / context frames required.

    // --- Elapsed execution time (in milliseconds) ---
    double pitchMilliseconds = 0.0;           ///< Wall-clock time spent in pitch inference & decoding.
    double acousticMilliseconds = 0.0;        ///< Wall-clock time spent in acoustic model inference.
    double vocoderMilliseconds = 0.0;         ///< Total wall-clock time spent in neural vocoder synthesis.
    double vocoderNetworkMilliseconds = 0.0;  ///< Time spent running vocoder neural network layers.
    double vocoderSourceMilliseconds = 0.0;   ///< Time spent generating periodic (pulse) glottal excitation.
    double vocoderResidualMilliseconds = 0.0; ///< Time spent generating aperiodic / noise residual.
    double vocoderFilterMilliseconds = 0.0;   ///< Time spent in filter bank convolution & synthesis.
};

} // namespace sv::synthesis
