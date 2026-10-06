/**
 * @file PitchDecoder.h
 * @brief Gen5 two-pass neural pitch decoder.
 *
 * In Synthesizer V Gen5, neural pitch synthesis uses a sophisticated two-pass architecture
 * comprising 9 sub-networks:
 * 1. Network 0: Frame-level scalar projection.
 * 2. Network 5: Coarse residual network (WaveNet/dilated residual blocks) conditioned on pooled context,
 *               sinusoidal control embeddings, and random noise.
 * 3. Network 7: Preliminary pitch head generating preliminary normalized pitch.
 * 4. Analytical extraction: Computes per-note pitch tilt (median slope) and pitch shift (80th percentile)
 *    on vowel frames. If the user provided requested tilt/shift overrides, target weights are computed
 *    via tanh activation.
 * 5. Network 1: Vibrato control embedding.
 * 6. Network 2: Pitch tilt embedding.
 * 7. Network 3: Pitch shift embedding.
 * 8. Network 4: Control correction network fusing pooled context and weighted control embeddings.
 * 9. Network 6: Refined residual network conditioning coarse features with the corrected context.
 * 10. Network 8: Final pitch projection head generating the final normalized pitch contour.
 */

#pragma once

#include "DnniInference.h"
#include "DnniReader.h"
#include "PitchFeatures.h"

#include <juce_core/juce_core.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <span>
#include <vector>

namespace sv::synthesis
{
/**
 * @struct PitchDecoderNotes
 * @brief Per-note metadata and pitch statistics exchanged between caller and PitchDecoder.
 */
struct PitchDecoderNotes
{
    /// Number of 5ms frames per note.
    std::span<const std::size_t> frameCounts;

    /// Score MIDI pitch for each note.
    std::span<const float> midiPitch;

    /// Binary mask (1 for vowel, 0 for consonant/silence) indicating frames where pitch statistics are evaluated.
    std::span<const std::uint8_t> vowelFrames;

    /// User-requested pitch tilt override per note (semitones/sec). Set to NaN to use model prediction.
    std::span<const float> requestedTilt;

    /// User-requested pitch shift override per note (semitones). Set to NaN to use model prediction.
    std::span<const float> requestedShift;

    /// Output: Model-predicted pitch tilt (semitones/sec) extracted from the first preliminary pass.
    std::vector<float> predictedTilt;

    /// Output: Model-predicted pitch shift (semitones) extracted from the first preliminary pass.
    std::vector<float> predictedShift;
};

/**
 * @class PitchDecoder
 * @brief Manages the 9 neural inference sub-networks of the Synthesizer V Gen5 pitch decoder.
 *
 * Parameters are immutable after loading from the DNNI voice database node. Working tensors
 * and temporary buffers are allocated dynamically on the calling synthesis thread.
 */
class PitchDecoder
{
public:
    /**
     * @brief Loads the 9 neural sub-networks and decoder configuration from the DNNI container.
     *
     * Validates that the node type matches one of the 12 recognized Gen5 pitch decoder type hashes,
     * parses the 12-byte payload (channels, scale, frame stride), and recursively loads all 9 child networks.
     *
     * @param reader DNNI archive reader.
     * @param nodeIndex Index of the pitch decoder container node.
     * @return `juce::Result::ok()` on success, error message on failure.
     */
    [[nodiscard]] juce::Result load(const DnniReader& reader, std::size_t nodeIndex);

    /**
     * @brief Executes two-pass neural pitch decoding.
     *
     * @param context High-dimensional phoneme/note context tensor [frames, channels].
     * @param scalarContext Scalar projection feature tensor [frames, scalarChannels].
     * @param noise Gaussian noise vector [frames].
     * @param controls 3-channel user control tensor [frames, 3] (enhancement, expression, ornamentation).
     * @param vibratoControl Vibrato modulation control stream [frames].
     * @param frontend Pitch features frontend for pitch denormalization during the first pass.
     * @param notes Note metadata and statistics struct (inputs and populated outputs).
     * @param output Output tensor [frames, 1] populated with final normalized pitch.
     * @param shouldCancel Optional cancellation callback queried between groups/layers.
     * @param statistics Optional profiling statistics collector.
     * @return `juce::Result::ok()` on success, failure error description otherwise.
     */
    [[nodiscard]] juce::Result run(
        const DnniTensor& context,
        const DnniTensor& scalarContext,
        std::span<const float> noise,
        const DnniTensor& controls,
        std::span<const float> vibratoControl,
        const PitchFeatures& frontend,
        PitchDecoderNotes& notes,
        DnniTensor& output,
        const std::function<bool()>& shouldCancel = {},
        DnniRunStatistics* statistics = nullptr) const;

private:
    std::array<DnniInference, 9> networks; ///< 9 inference sub-networks.
    std::size_t embeddingChannels = 0;     ///< Number of channels in sinusoidal control embeddings.
    std::size_t frameStride = 0;           ///< Temporal decimation factor (e.g. 4 frames per group).
    float embeddingScale = 0.0f;           ///< Scaling factor for sinusoidal embedding position.
    bool loaded = false;                   ///< True once all 9 networks are successfully loaded.
};
} // namespace sv::synthesis

