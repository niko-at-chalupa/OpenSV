#include "FeatureNormalizer.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

namespace sv::synthesis
{
juce::Result FeatureNormalizer::load(const DnniReader& reader, std::size_t nodeIndex)
{
    const auto& nodes = reader.getNodes();
    // Invariant: node must be of type "cmpu0", have 0-byte payload, and exactly 3 vector children
    if (nodeIndex >= nodes.size() || nodes[nodeIndex].type != "cmpu0" || nodes[nodeIndex].payloadSize != 0 || nodes[nodeIndex].children.size() != 3)
    {
        return juce::Result::fail("Feature normalization requires an empty cmpu0 node with three vector children.");
    }

    FeatureNormalizer candidate;
    std::vector<float> range;
    const auto& children = nodes[nodeIndex].children;

    // Child 0: lower bound vector
    if (const auto result = reader.readFloatVector(children[0], candidate.lower); result.failed())
    {
        return result;
    }
    // Child 1: upper bound vector
    if (const auto result = reader.readFloatVector(children[1], candidate.upper); result.failed())
    {
        return result;
    }
    // Child 2: target range vector (2 floats: [targetLower, targetUpper])
    if (const auto result = reader.readFloatVector(children[2], range); result.failed())
    {
        return result;
    }

    // Validate dimensions and ranges
    if (candidate.lower.empty() || candidate.lower.size() != candidate.upper.size() || range.size() != 2 || !(range[0] < range[1]) || !std::isfinite(range[1] - range[0]))
    {
        return juce::Result::fail("Invalid cmpu0 channel count or target range.");
    }
    for (std::size_t channel = 0; channel < candidate.lower.size(); ++channel)
    {
        if (candidate.lower[channel] > candidate.upper[channel] || !std::isfinite(candidate.upper[channel] - candidate.lower[channel]))
        {
            return juce::Result::fail("Invalid cmpu0 source range at channel " + juce::String(static_cast<juce::int64>(channel)) + ".");
        }
    }

    candidate.targetLower = range[0];
    candidate.targetUpper = range[1];
    *this = std::move(candidate);
    return juce::Result::ok();
}

std::size_t FeatureNormalizer::getChannelCount() const noexcept
{
    return lower.size();
}

juce::Result FeatureNormalizer::normalize(const DnniTensor& input, DnniTensor& output, bool clamp) const
{
    return transform(input, output, false, clamp);
}

juce::Result FeatureNormalizer::denormalize(const DnniTensor& input, DnniTensor& output) const
{
    return transform(input, output, true, false);
}

juce::Result FeatureNormalizer::transform(const DnniTensor& input, DnniTensor& output, bool inverse, bool clamp) const
{
    if (lower.empty())
    {
        return juce::Result::fail("Feature normalization has not been loaded.");
    }
    if (input.channels != lower.size() || input.frames > std::numeric_limits<std::size_t>::max() / input.channels || input.frames * input.channels != input.values.size())
    {
        return juce::Result::fail("Feature normalization input tensor has an invalid shape.");
    }

    DnniTensor result{input.frames, input.channels, {}};
    result.values.resize(input.values.size());
    const float targetRange = targetUpper - targetLower;

    for (std::size_t index = 0; index < input.values.size(); ++index)
    {
        const float value = input.values[index];
        if (!std::isfinite(value))
        {
            return juce::Result::fail("Feature normalization input contains a non-finite value.");
        }
        const std::size_t channel = index % input.channels;
        const float sourceRange = upper[channel] - lower[channel];
        float transformed = 0.0f;

        // Inverse transform (normalized -> raw physical scale)
        if (inverse)
        {
            transformed = lower[channel] + sourceRange * (value - targetLower) / targetRange;
        }
        // Forward transform (raw physical scale -> normalized)
        else
        {
            // Floor of 1e-5 in denominator prevents division by zero for constant features
            transformed = targetLower + targetRange * (value - lower[channel]) / std::max(sourceRange, 1.0e-5f);
            if (clamp)
            {
                transformed = std::clamp(transformed, targetLower, targetUpper);
            }
        }

        if (!std::isfinite(transformed))
        {
            return juce::Result::fail("Feature normalization produced a non-finite value.");
        }
        result.values[index] = transformed;
    }

    output = std::move(result);
    return juce::Result::ok();
}

} // namespace sv::synthesis
