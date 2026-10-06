#pragma once

#include "DnniInference.h"
#include "DnniReader.h"

#include <juce_core/juce_core.h>

#include <cstddef>
#include <vector>

namespace sv::synthesis
{
/**
 * @brief Per-channel feature normalization and denormalization using min-max scaling (`cmpu0`).
 *
 * In Synthesizer V DNNI models, input features (phoneme representations, pitch, energy)
 * and output predictions (acoustic features, vocoder parameters) are scaled to/from
 * normalized ranges (e.g. [0, 1] or [-1, 1]).
 *
 * The `cmpu0` node contains three vector children:
 * 1. lower: per-channel minimum source values [C]
 * 2. upper: per-channel maximum source values [C]
 * 3. range: global target range [targetLower, targetUpper] (typically 2 elements)
 *
 * Normalization formula:
 * y = targetLower + (targetUpper - targetLower) * (x - lower[c]) / max(upper[c] - lower[c], 1e-5)
 *
 * Denormalization formula:
 * x = lower[c] + (upper[c] - lower[c]) * (y - targetLower) / (targetUpper - targetLower)
 */
class FeatureNormalizer
{
public:
    /**
     * @brief Loads normalization bounds from a `cmpu0` node in a DNNI model.
     * @param reader Parsed DNNI reader.
     * @param nodeIndex Index of the `cmpu0` node.
     * @return juce::Result::ok() or failure description.
     */
    [[nodiscard]] juce::Result load(const DnniReader& reader, std::size_t nodeIndex);

    /**
     * @brief Returns the number of channels supported by this normalizer.
     */
    [[nodiscard]] std::size_t getChannelCount() const noexcept;

    /**
     * @brief Maps raw features into the normalized target range (forward transform).
     * @param input Raw feature tensor [frames, channels].
     * @param output Output normalized tensor [frames, channels].
     * @param clamp If true, clamps output to [targetLower, targetUpper].
     */
    [[nodiscard]] juce::Result normalize(const DnniTensor& input, DnniTensor& output, bool clamp = true) const;

    /**
     * @brief Maps normalized features back to physical units (inverse transform).
     * @param input Normalized feature tensor [frames, channels].
     * @param output Output denormalized tensor [frames, channels].
     */
    [[nodiscard]] juce::Result denormalize(const DnniTensor& input, DnniTensor& output) const;

private:
    [[nodiscard]] juce::Result transform(const DnniTensor& input, DnniTensor& output, bool inverse, bool clamp) const;

    std::vector<float> lower;
    std::vector<float> upper;
    float targetLower = 0.0f;
    float targetUpper = 1.0f;
};

} // namespace sv::synthesis
