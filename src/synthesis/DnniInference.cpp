#include "DnniInference.h"

#include <juce_audio_basics/juce_audio_basics.h>

#include <algorithm>
#include <atomic>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <new>
#include <span>
#include <utility>

namespace sv::synthesis
{
namespace
{
constexpr std::size_t maximumElements = 64 * 1024 * 1024;
constexpr std::size_t maximumCacheBytes = 64 * 1024 * 1024;
std::atomic<std::uint64_t> nextModelIdentity{1};

juce::Result nodeError(std::size_t offset, const juce::String& reason)
{
    return juce::Result::fail("DNNI inference at 0x" + juce::String::toHexString(static_cast<juce::int64>(offset)) + ": " + reason);
}

std::uint32_t readUint32(std::span<const std::uint8_t> payload, std::size_t offset)
{
    return static_cast<std::uint32_t>(payload[offset]) | (static_cast<std::uint32_t>(payload[offset + 1]) << 8) | (static_cast<std::uint32_t>(payload[offset + 2]) << 16) | (static_cast<std::uint32_t>(payload[offset + 3]) << 24);
}

bool validShape(std::size_t frames, std::size_t channels)
{
    return channels > 0 && channels <= maximumElements && frames <= maximumElements / channels;
}

bool finiteValues(const std::vector<float>& values)
{
    return std::all_of(values.begin(), values.end(), [](float value)
                       { return std::isfinite(value); });
}
/**
 * @brief Numerically stable logistic sigmoid activation: 1 / (1 + exp(-x)).
 * Avoids floating-point overflow for large negative or positive inputs.
 */
float sigmoid(float value)
{
    const auto exponential = std::exp(-std::abs(value));
    return value >= 0.0f ? 1.0f / (1.0f + exponential) : exponential / (1.0f + exponential);
}

bool sameShape(const DnniTensor& first, const DnniTensor& second)
{
    return first.frames == second.frames && first.channels == second.channels && first.values.size() == second.values.size();
}

/**
 * @brief Represents a continuous range of frames [first, end) in time.
 */
struct FrameRange
{
    std::size_t first = 0;
    std::size_t end = 0;
};

/**
 * @brief Expands a frame range by a given context radius (receptive field padding),
 * clamped to total frame count [0, frames].
 */
FrameRange expandRange(FrameRange range, std::size_t radius, std::size_t frames)
{
    return {range.first > radius ? range.first - radius : 0, range.end + std::min(radius, frames - range.end)};
}

/**
 * @brief Identifies segments of output frames that must be recomputed due to changed inputs.
 *
 * Compares current input tensor (and optional condition tensor) against cached inputs.
 * When differences are detected, the affected frame ranges are expanded by the receptive
 * field radius of the convolutional network, merging adjacent or overlapping windows.
 */
std::vector<FrameRange> findChangedOutputRanges(const DnniTensor& previous, const DnniTensor& current, const DnniTensor* previousCondition, const DnniTensor* currentCondition, std::size_t radius)
{
    std::vector<FrameRange> ranges;
    const auto inputBytes = current.channels * sizeof(float);
    const auto conditionBytes = currentCondition == nullptr ? 0 : currentCondition->channels * sizeof(float);
    for (std::size_t frame = 0; frame < current.frames; ++frame)
    {
        const auto inputOffset = frame * current.channels;
        const bool inputChanged = std::memcmp(previous.values.data() + inputOffset, current.values.data() + inputOffset, inputBytes) != 0;
        const auto conditionOffset = currentCondition == nullptr ? 0 : frame * currentCondition->channels;
        const bool conditionChanged = !inputChanged && currentCondition != nullptr && std::memcmp(previousCondition->values.data() + conditionOffset, currentCondition->values.data() + conditionOffset, conditionBytes) != 0;
        if (!inputChanged && !conditionChanged)
        {
            continue;
        }
        const auto range = expandRange({frame, frame + 1}, radius, current.frames);
        if (ranges.empty() || ranges.back().end < range.first)
        {
            ranges.push_back(range);
        }
        else
        {
            ranges.back().end = std::max(ranges.back().end, range.end);
        }
    }
    return ranges;
}

DnniTensor sliceFrames(const DnniTensor& input, std::size_t first, std::size_t end)
{
    const auto begin = input.values.begin() + static_cast<std::ptrdiff_t>(first * input.channels);
    const auto finish = input.values.begin() + static_cast<std::ptrdiff_t>(end * input.channels);
    return {end - first, input.channels, std::vector<float>(begin, finish)};
}
} // namespace

void DnniInference::Cache::clear()
{
    *this = Cache{};
}

std::size_t DnniInference::Cache::getBytes() const noexcept
{
    return (input.values.capacity() + condition.values.capacity() + output.values.capacity()) * sizeof(float);
}

juce::Result DnniInference::load(const DnniReader& reader, std::size_t rootNode)
{
    try
    {
        Layer replacement;
        std::size_t parameterCount = 0;
        if (const auto result = loadLayer(reader, rootNode, replacement, parameterCount); result.failed())
        {
            return result;
        }
        contextRadius = findContextRadius(replacement);
        root = std::move(replacement);
        modelIdentity = nextModelIdentity.fetch_add(1, std::memory_order_relaxed);
        loaded = true;
        return juce::Result::ok();
    }
    catch (const std::bad_alloc&)
    {
        return juce::Result::fail("Insufficient memory to load DNNI inference parameters.");
    }
}

juce::Result DnniInference::loadMatrix(const DnniReader& reader, std::size_t nodeIndex, Matrix& packed, std::size_t& parameterCount)
{
    DnniMatrix matrix;
    if (const auto result = reader.readFloatMatrix(nodeIndex, matrix); result.failed())
    {
        return result;
    }
    if (matrix.rows == 0 || !validShape(matrix.rows, matrix.columns) || matrix.values.size() != static_cast<std::size_t>(matrix.rows) * matrix.columns)
    {
        return nodeError(reader.getNodes()[nodeIndex].offset, "invalid matrix shape.");
    }
    const auto blockCount = (static_cast<std::size_t>(matrix.rows) + channelsPerBlock - 1) / channelsPerBlock;
    const auto paddedRows = blockCount * channelsPerBlock;
    if (matrix.columns > (maximumElements - parameterCount) / paddedRows)
    {
        return nodeError(reader.getNodes()[nodeIndex].offset, "packed model exceeds the parameter memory limit.");
    }
    parameterCount += paddedRows * matrix.columns;
    packed.rows = matrix.rows;
    packed.columns = matrix.columns;
    packed.values.resize(blockCount * packed.columns);
    for (std::size_t block = 0; block < blockCount; ++block)
    {
        const auto firstRow = block * channelsPerBlock;
        const auto rowCount = std::min(channelsPerBlock, packed.rows - firstRow);
        for (std::size_t column = 0; column < packed.columns; ++column)
        {
            alignas(Vector) std::array<float, channelsPerBlock> values{};
            for (std::size_t row = 0; row < rowCount; ++row)
            {
                values[row] = matrix.values[(firstRow + row) * packed.columns + column];
            }
            auto& weights = packed.values[block * packed.columns + column];
            for (std::size_t vector = 0; vector < vectorsPerBlock; ++vector)
            {
                weights[vector] = Vector::fromRawArray(values.data() + vector * Vector::size());
            }
        }
    }
    return juce::Result::ok();
}

juce::Result DnniInference::loadLayer(const DnniReader& reader, std::size_t nodeIndex, Layer& layer, std::size_t& parameterCount)
{
    const auto& nodes = reader.getNodes();
    if (nodeIndex >= nodes.size())
    {
        return juce::Result::fail("DNNI inference root or child node is out of range.");
    }
    const auto& node = nodes[nodeIndex];
    const auto payload = reader.getPayload(nodeIndex);
    layer.sourceOffset = node.offset;

    if (node.type == "modm0")
    {
        if (!payload.empty())
        {
            return nodeError(node.offset, "sequence payload must be empty.");
        }
        layer.operation = Operation::sequence;
        for (const auto childIndex : node.children)
        {
            Layer child;
            if (const auto result = loadLayer(reader, childIndex, child, parameterCount); result.failed())
            {
                return result;
            }
            // Nested sequences have no state; flattening prevents tensor copies accumulating with depth.
            if (child.operation == Operation::sequence)
            {
                for (auto& nested : child.children)
                {
                    layer.children.push_back(std::move(nested));
                }
            }
            else
            {
                layer.children.push_back(std::move(child));
            }
        }
        return juce::Result::ok();
    }

    if (node.type == "_ncwnv0")
    {
        if (payload.size() != 12)
        {
            return nodeError(node.offset, "residual convolution v0 requires three int32 channel dimensions.");
        }
        const auto inputChannels = std::bit_cast<std::int32_t>(readUint32(payload, 0));
        const auto hiddenChannels = std::bit_cast<std::int32_t>(readUint32(payload, 4));
        const auto conditionChannels = std::bit_cast<std::int32_t>(readUint32(payload, 8));
        if (inputChannels <= 0 || hiddenChannels <= 0 || conditionChannels < 0)
        {
            return nodeError(node.offset, "residual convolution channel dimensions are invalid.");
        }
        layer.operation = Operation::residualConvolution;
        layer.inputChannels = static_cast<std::size_t>(inputChannels);
        layer.gateChannels = static_cast<std::size_t>(hiddenChannels);
        layer.conditionChannels = static_cast<std::size_t>(conditionChannels);
        const std::size_t childCount = conditionChannels > 0 ? 4 : 3;
        if (node.children.size() != childCount)
        {
            return nodeError(node.offset, "residual convolution requires three parameter groups and an optional condition projection.");
        }
        for (std::size_t groupIndex = 0; groupIndex < 3; ++groupIndex)
        {
            const auto childIndex = node.children[groupIndex];
            if (childIndex >= nodes.size() || nodes[childIndex].type != "cmpg1" || !reader.getPayload(childIndex).empty())
            {
                return nodeError(node.offset, "residual convolution requires empty-payload cmpg1 parameter groups.");
            }
            const auto count = nodes[childIndex].children.size();
            if (groupIndex == 0)
            {
                layer.stageCount = count;
            }
            if (count == 0 || count != layer.stageCount)
            {
                return nodeError(nodes[childIndex].offset, "residual convolution parameter groups must have matching non-zero lengths.");
            }
        }
        const auto preservesFrames = [](const Layer& convolution)
        {
            return convolution.operation == Operation::convolution && convolution.stride == 1 && 2 * static_cast<std::uint64_t>(convolution.padding) == static_cast<std::uint64_t>(convolution.matrices.size() - 1) * convolution.dilation;
        };
        // Each stage stores the gate, its skip projection, and the input's residual projection.
        layer.children.resize(layer.stageCount * 3 + (conditionChannels > 0 ? 1 : 0));
        for (std::size_t stage = 0; stage < layer.stageCount; ++stage)
        {
            const auto stageInput = stage == 0 ? layer.inputChannels : layer.gateChannels;
            for (std::size_t groupIndex = 0; groupIndex < 3; ++groupIndex)
            {
                const auto childIndex = nodes[node.children[groupIndex]].children[stage];
                const auto expectedType = groupIndex == 0 ? "_gnc1v0" : "modl1";
                if (childIndex >= nodes.size() || nodes[childIndex].type != expectedType)
                {
                    return nodeError(node.offset, "residual convolution contains an unsupported stage operator.");
                }
                auto& child = layer.children[stage * 3 + groupIndex];
                if (const auto result = loadLayer(reader, childIndex, child, parameterCount); result.failed())
                {
                    return result;
                }
                if (groupIndex == 0)
                {
                    if (child.inputChannels != stageInput || child.gateChannels != layer.gateChannels || child.conditionChannels != layer.conditionChannels)
                    {
                        return nodeError(nodes[childIndex].offset, "residual gate dimensions do not match its parent network.");
                    }
                    for (const auto& convolution : child.children)
                    {
                        if (!preservesFrames(convolution))
                        {
                            return nodeError(convolution.sourceOffset, "whole-sequence residual gates require stride-one convolutions that preserve frame count.");
                        }
                    }
                }
                else
                {
                    const auto& matrix = child.matrices.front();
                    const auto expectedInput = groupIndex == 1 ? layer.gateChannels : stageInput;
                    if (matrix.columns != expectedInput || matrix.rows != layer.gateChannels || !preservesFrames(child))
                    {
                        return nodeError(nodes[childIndex].offset, "residual or skip projection dimensions and timing do not match the network.");
                    }
                }
            }
        }
        if (conditionChannels > 0)
        {
            const auto childIndex = node.children[3];
            if (childIndex >= nodes.size() || nodes[childIndex].type != "modl1")
            {
                return nodeError(node.offset, "residual network condition projection must be Conv1D.");
            }
            auto& projection = layer.children.back();
            if (const auto result = loadLayer(reader, childIndex, projection, parameterCount); result.failed())
            {
                return result;
            }
            const auto& matrix = projection.matrices.front();
            if (matrix.columns != layer.conditionChannels || matrix.rows != layer.gateChannels || projection.matrices.size() != 1 || !preservesFrames(projection))
            {
                return nodeError(nodes[childIndex].offset, "residual network condition projection must preserve frames and map condition channels to hidden channels.");
            }
        }
        return juce::Result::ok();
    }

    if (node.type == "_gnc1v0")
    {
        if (payload.size() != 12)
        {
            return nodeError(node.offset, "gated Conv1D v0 requires three int32 channel dimensions.");
        }
        const auto inputChannels = std::bit_cast<std::int32_t>(readUint32(payload, 0));
        const auto gateChannels = std::bit_cast<std::int32_t>(readUint32(payload, 4));
        const auto conditionChannels = std::bit_cast<std::int32_t>(readUint32(payload, 8));
        if (inputChannels <= 0 || gateChannels <= 0 || conditionChannels < 0 || static_cast<std::size_t>(gateChannels) > maximumElements / 2)
        {
            return nodeError(node.offset, "gated Conv1D channel dimensions are invalid.");
        }
        layer.operation = Operation::gatedConvolution;
        layer.inputChannels = static_cast<std::size_t>(inputChannels);
        layer.gateChannels = static_cast<std::size_t>(gateChannels);
        layer.conditionChannels = static_cast<std::size_t>(conditionChannels);
        const std::size_t childCount = conditionChannels > 0 ? 2 : 1;
        if (node.children.size() != childCount)
        {
            return nodeError(node.offset, "gated Conv1D v0 requires an input convolution and an optional condition convolution.");
        }
        layer.children.resize(childCount);
        for (std::size_t index = 0; index < childCount; ++index)
        {
            const auto childIndex = node.children[index];
            if (childIndex >= nodes.size() || nodes[childIndex].type != "modl1")
            {
                return nodeError(node.offset, "gated Conv1D v0 children must be Conv1D operators.");
            }
            auto& convolution = layer.children[index];
            if (const auto result = loadLayer(reader, childIndex, convolution, parameterCount); result.failed())
            {
                return result;
            }
            const auto& matrix = convolution.matrices.front();
            const auto expectedInput = index == 0 ? layer.inputChannels : layer.conditionChannels;
            if (matrix.columns != expectedInput || matrix.rows != 2 * layer.gateChannels)
            {
                return nodeError(nodes[childIndex].offset, "gated convolution matrices do not match the declared channel dimensions.");
            }
            if (index == 1 && convolution.matrices.size() != 1)
            {
                return nodeError(nodes[childIndex].offset, "gated Conv1D v0 condition convolution must have a one-frame kernel.");
            }
        }
        return juce::Result::ok();
    }

    if (node.type == "modl6")
    {
        if (!payload.empty() || node.children.size() != 2)
        {
            return nodeError(node.offset, "bidirectional GRU requires two GRU children and an empty payload.");
        }
        layer.operation = Operation::bidirectionalGru;
        layer.children.resize(2);
        for (std::size_t direction = 0; direction < 2; ++direction)
        {
            const auto childIndex = node.children[direction];
            if (childIndex >= nodes.size() || nodes[childIndex].type != "modl3")
            {
                return nodeError(node.offset, "bidirectional GRU children must be modl3 operators.");
            }
            if (const auto result = loadLayer(reader, childIndex, layer.children[direction], parameterCount); result.failed())
            {
                return result;
            }
        }
        layer.inputChannels = layer.children[0].inputChannels;
        if (layer.inputChannels != layer.children[1].inputChannels || layer.children[0].gateChannels > maximumElements - layer.children[1].gateChannels)
        {
            return nodeError(node.offset, "bidirectional GRU directions have incompatible input channels or excessive output channels.");
        }
        layer.gateChannels = layer.children[0].gateChannels + layer.children[1].gateChannels;
        return juce::Result::ok();
    }

    if (node.type == "modl3")
    {
        if (!payload.empty() || node.children.size() != 12)
        {
            return nodeError(node.offset, "GRU requires six matrix/bias pairs and an empty payload.");
        }
        layer.operation = Operation::gru;
        layer.matrices.resize(6);
        for (std::size_t projection = 0; projection < layer.matrices.size(); ++projection)
        {
            const auto matrixIndex = node.children[2 * projection];
            const auto biasIndex = node.children[2 * projection + 1];
            if (matrixIndex >= nodes.size() || biasIndex >= nodes.size() || !nodes[matrixIndex].children.empty() || !nodes[biasIndex].children.empty() || nodes[biasIndex].type != "prim1")
            {
                return nodeError(node.offset, "GRU parameters must alternate leaf matrices and float bias vectors.");
            }
            auto& matrix = layer.matrices[projection];
            if (const auto result = loadMatrix(reader, matrixIndex, matrix, parameterCount); result.failed())
            {
                return result;
            }
            if (projection == 0)
            {
                layer.inputChannels = matrix.columns;
                layer.gateChannels = matrix.rows;
            }
            const auto expectedColumns = projection < 3 ? layer.inputChannels : layer.gateChannels;
            if (matrix.rows != layer.gateChannels || matrix.columns != expectedColumns)
            {
                return nodeError(nodes[matrixIndex].offset, "GRU input and recurrent matrices have incompatible dimensions.");
            }
            std::vector<float> bias;
            if (const auto result = reader.readFloatVector(biasIndex, bias); result.failed())
            {
                return result;
            }
            if (bias.size() != layer.gateChannels || bias.size() > maximumElements - parameterCount)
            {
                return nodeError(nodes[biasIndex].offset, "GRU bias dimensions are invalid or exceed the parameter memory limit.");
            }
            parameterCount += bias.size();
            layer.bias.insert(layer.bias.end(), bias.begin(), bias.end());
        }
        return juce::Result::ok();
    }

    if (node.type == "modl0" || node.type == "modl1")
    {
        std::size_t kernelCount = 1;
        if (node.type == "modl0")
        {
            layer.operation = Operation::dense;
            if (!payload.empty() || node.children.empty() || node.children.size() > 2)
            {
                return nodeError(node.offset, "dense requires an empty payload, one matrix and an optional bias.");
            }
        }
        else
        {
            layer.operation = Operation::convolution;
            if (payload.size() != 20)
            {
                return nodeError(node.offset, "Conv1D requires five int32 parameters.");
            }
            const auto kernel = std::bit_cast<std::int32_t>(readUint32(payload, 0));
            const auto stride = std::bit_cast<std::int32_t>(readUint32(payload, 4));
            const auto padding = std::bit_cast<std::int32_t>(readUint32(payload, 8));
            const auto dilation = std::bit_cast<std::int32_t>(readUint32(payload, 12));
            const auto groups = std::bit_cast<std::int32_t>(readUint32(payload, 16));
            if (kernel <= 0 || stride <= 0 || padding < 0 || dilation <= 0 || groups != 1)
            {
                return nodeError(node.offset, "Conv1D requires positive kernel, stride and dilation, non-negative padding, and groups=1.");
            }
            kernelCount = static_cast<std::size_t>(kernel);
            layer.stride = static_cast<std::size_t>(stride);
            layer.padding = static_cast<std::size_t>(padding);
            layer.dilation = static_cast<std::size_t>(dilation);
            if (node.children.size() < kernelCount || node.children.size() > kernelCount + 1)
            {
                return nodeError(node.offset, "Conv1D child count does not match its kernel count and optional bias.");
            }
        }

        bool hasBias = false;
        for (std::size_t index = 0; index < node.children.size(); ++index)
        {
            const auto childIndex = node.children[index];
            if (childIndex >= nodes.size())
            {
                return nodeError(node.offset, "parameter node is out of range.");
            }
            const auto& child = nodes[childIndex];
            if (!child.children.empty())
            {
                return nodeError(child.offset, "tensor parameters cannot have children.");
            }
            if (child.type == "prim1")
            {
                if (hasBias || (layer.operation == Operation::dense && index != 1))
                {
                    return nodeError(child.offset, "unexpected or duplicate bias vector.");
                }
                if (const auto result = reader.readFloatVector(childIndex, layer.bias); result.failed())
                {
                    return result;
                }
                hasBias = true;
                if (layer.bias.size() > maximumElements - parameterCount)
                {
                    return nodeError(child.offset, "decoded model exceeds the parameter memory limit.");
                }
                parameterCount += layer.bias.size();
            }
            else
            {
                Matrix packed;
                if (const auto result = loadMatrix(reader, childIndex, packed, parameterCount); result.failed())
                {
                    return result;
                }
                layer.matrices.push_back(std::move(packed));
            }
        }
        if (layer.matrices.size() != kernelCount)
        {
            return nodeError(node.offset, "matrix count does not match the operator.");
        }
        const auto& firstMatrix = layer.matrices.front();
        for (const auto& matrix : layer.matrices)
        {
            if (matrix.rows != firstMatrix.rows || matrix.columns != firstMatrix.columns)
            {
                return nodeError(node.offset, "Conv1D kernel matrices must have equal shapes.");
            }
        }
        if (hasBias && layer.bias.size() != firstMatrix.rows)
        {
            return nodeError(node.offset, "bias length does not match output channels.");
        }
        return juce::Result::ok();
    }

    if (node.type == "moda0")
    {
        layer.operation = Operation::relu;
    }
    else if (node.type == "moda1")
    {
        layer.operation = Operation::tanh;
    }
    else if (node.type == "moda2")
    {
        layer.operation = Operation::sigmoid;
    }
    else if (node.type == "moda3")
    {
        layer.operation = Operation::leakyRelu;
    }
    else if (node.type == "moda4")
    {
        layer.operation = Operation::elu;
    }
    else if (node.type == "moda5")
    {
        layer.operation = Operation::identity;
    }
    else if (node.type == "moda7")
    {
        layer.operation = Operation::silu;
    }
    else
    {
        const auto type = node.type.empty() ? "0x" + juce::String::toHexString(static_cast<juce::int64>(node.typeId)) : juce::String::fromUTF8(node.type.c_str());
        return nodeError(node.offset, "unsupported operator " + type + ".");
    }
    if (!node.children.empty())
    {
        return nodeError(node.offset, "activation cannot have child parameters.");
    }
    if (layer.operation == Operation::leakyRelu || layer.operation == Operation::elu)
    {
        if (payload.size() != sizeof(float))
        {
            return nodeError(node.offset, "activation requires one float32 alpha parameter.");
        }
        layer.alpha = std::bit_cast<float>(readUint32(payload, 0));
        if (!std::isfinite(layer.alpha))
        {
            return nodeError(node.offset, "activation alpha must be finite.");
        }
    }
    else if (!payload.empty())
    {
        return nodeError(node.offset, "activation payload must be empty.");
    }
    return juce::Result::ok();
}

std::optional<std::size_t> DnniInference::findContextRadius(const Layer& layer)
{
    if (layer.operation == Operation::gru || layer.operation == Operation::bidirectionalGru)
    {
        // Recurrent state depends on the whole preceding sequence (both
        // directions for modl6), so fixed-radius cache windows are not exact.
        return std::nullopt;
    }
    if (layer.operation == Operation::convolution)
    {
        const auto extent = static_cast<std::uint64_t>(layer.matrices.size() - 1) * layer.dilation;
        if (layer.stride != 1 || extent != 2 * static_cast<std::uint64_t>(layer.padding) || layer.padding > maximumElements)
        {
            return std::nullopt;
        }
        return layer.padding;
    }
    if (layer.operation == Operation::sequence)
    {
        std::size_t radius = 0;
        for (const auto& child : layer.children)
        {
            const auto childRadius = findContextRadius(child);
            if (!childRadius.has_value() || *childRadius > maximumElements - radius)
            {
                return std::nullopt;
            }
            radius += *childRadius;
        }
        return radius;
    }
    if (layer.operation == Operation::gatedConvolution)
    {
        std::size_t radius = 0;
        for (const auto& child : layer.children)
        {
            const auto childRadius = findContextRadius(child);
            if (!childRadius.has_value())
            {
                return std::nullopt;
            }
            radius = std::max(radius, *childRadius);
        }
        return radius;
    }
    if (layer.operation == Operation::residualConvolution)
    {
        std::size_t radius = 0;
        for (std::size_t stage = 0; stage < layer.stageCount; ++stage)
        {
            const auto gate = findContextRadius(layer.children[stage * 3]);
            const auto skip = findContextRadius(layer.children[stage * 3 + 1]);
            const auto residual = findContextRadius(layer.children[stage * 3 + 2]);
            if (!gate.has_value() || !skip.has_value() || !residual.has_value() || *skip > maximumElements - *gate)
            {
                return std::nullopt;
            }
            // The next stage receives gate + residual. Every skip also contributes
            // to the final output, so include the longest branch at each stage.
            const auto stageRadius = std::max(*gate + *skip, *residual);
            if (stageRadius > maximumElements - radius)
            {
                return std::nullopt;
            }
            radius += stageRadius;
        }
        if (layer.conditionChannels > 0)
        {
            const auto conditionRadius = findContextRadius(layer.children.back());
            if (!conditionRadius.has_value())
            {
                return std::nullopt;
            }
            radius = std::max(radius, *conditionRadius);
        }
        return radius;
    }
    return 0;
}

juce::Result DnniInference::run(const DnniTensor& input, DnniTensor& output, const DnniTensor* condition, Cache* cache, const std::function<bool()>& shouldCancel, DnniRunStatistics* statistics) const
{
    if (!loaded)
    {
        return juce::Result::fail("No DNNI inference model has been loaded.");
    }
    if (!validShape(input.frames, input.channels) || input.values.size() != input.frames * input.channels || !finiteValues(input.values))
    {
        return juce::Result::fail("DNNI input must contain finite frame-major values with valid dimensions within 64 Mi elements.");
    }
    if (condition != nullptr && (!validShape(condition->frames, condition->channels) || condition->values.size() != condition->frames * condition->channels || !finiteValues(condition->values)))
    {
        return juce::Result::fail("DNNI condition must contain finite frame-major values with valid dimensions within 64 Mi elements.");
    }
    try
    {
        if (shouldCancel && shouldCancel())
        {
            return juce::Result::fail("DNNI inference cancelled.");
        }
        const bool canCache = cache != nullptr && contextRadius.has_value() && (condition == nullptr || condition->frames == input.frames);
        const bool hasCachedOutput = canCache && cache->modelIdentity == modelIdentity && sameShape(cache->input, input) && cache->hasCondition == (condition != nullptr) && (condition == nullptr || sameShape(cache->condition, *condition));
        std::vector<FrameRange> outputRanges;
        if (hasCachedOutput)
        {
            outputRanges = findChangedOutputRanges(cache->input, input, condition == nullptr ? nullptr : &cache->condition, condition, *contextRadius);
            if (outputRanges.empty())
            {
                output = cache->output;
                if (statistics != nullptr)
                {
                    statistics->reusedFrames += output.frames;
                }
                return juce::Result::ok();
            }
        }

        DnniTensor replacement;
        std::size_t computedFrames = 0;
        std::size_t reusedFrames = 0;
        std::size_t contextFrames = 0;
        if (hasCachedOutput)
        {
            const auto radius = *contextRadius;
            std::size_t updatedFrames = 0;
            replacement = cache->output;
            for (std::size_t firstRange = 0; firstRange < outputRanges.size();)
            {
                if (shouldCancel && shouldCancel())
                {
                    return juce::Result::fail("DNNI inference cancelled.");
                }
                auto window = expandRange(outputRanges[firstRange], radius, input.frames);
                auto endRange = firstRange + 1;
                while (endRange < outputRanges.size())
                {
                    const auto nextWindow = expandRange(outputRanges[endRange], radius, input.frames);
                    if (nextWindow.first > window.end)
                    {
                        break;
                    }
                    window.end = std::max(window.end, nextWindow.end);
                    ++endRange;
                }
                DnniTensor localOutput;
                if (window.first == 0 && window.end == input.frames)
                {
                    if (const auto result = runLayer(root, input, localOutput, condition, shouldCancel); result.failed())
                    {
                        return result;
                    }
                }
                else
                {
                    const auto localInput = sliceFrames(input, window.first, window.end);
                    const auto localCondition = condition == nullptr ? DnniTensor{} : sliceFrames(*condition, window.first, window.end);
                    if (const auto result = runLayer(root, localInput, localOutput, condition == nullptr ? nullptr : &localCondition, shouldCancel); result.failed())
                    {
                        return result;
                    }
                }
                if (localOutput.frames != window.end - window.first || localOutput.channels != cache->output.channels)
                {
                    return juce::Result::fail("DNNI cached inference produced an unexpected output shape.");
                }
                // Merge overlapping halos into one computation, but replace only
                // affected centres. Unchanged gaps retain their cached values.
                for (auto index = firstRange; index < endRange; ++index)
                {
                    const auto range = outputRanges[index];
                    const auto source = localOutput.values.begin() + static_cast<std::ptrdiff_t>((range.first - window.first) * localOutput.channels);
                    const auto destination = replacement.values.begin() + static_cast<std::ptrdiff_t>(range.first * replacement.channels);
                    std::copy_n(source, (range.end - range.first) * replacement.channels, destination);
                    updatedFrames += range.end - range.first;
                }
                computedFrames += localOutput.frames;
                firstRange = endRange;
            }
            reusedFrames = input.frames - updatedFrames;
            contextFrames = computedFrames - updatedFrames;
        }
        else
        {
            if (const auto result = runLayer(root, input, replacement, condition, shouldCancel); result.failed())
            {
                return result;
            }
            computedFrames = replacement.frames;
        }
        if (shouldCancel && shouldCancel())
        {
            return juce::Result::fail("DNNI inference cancelled.");
        }
        if (cache != nullptr)
        {
            const auto conditionElements = condition == nullptr ? 0 : condition->values.size();
            const auto snapshotElements = input.values.size() + conditionElements + replacement.values.size();
            if (canCache && replacement.frames == input.frames && snapshotElements <= maximumCacheBytes / sizeof(float))
            {
                Cache updated;
                updated.input = input;
                if (condition != nullptr)
                {
                    updated.condition = *condition;
                }
                updated.output = replacement;
                updated.modelIdentity = modelIdentity;
                updated.hasCondition = condition != nullptr;
                if (updated.getBytes() <= maximumCacheBytes)
                {
                    *cache = std::move(updated);
                }
                else
                {
                    cache->clear();
                }
            }
            else
            {
                cache->clear();
            }
        }
        output = std::move(replacement);
        if (statistics != nullptr)
        {
            statistics->computedFrames += computedFrames;
            statistics->reusedFrames += reusedFrames;
            statistics->contextFrames += contextFrames;
        }
        return juce::Result::ok();
    }
    catch (const std::bad_alloc&)
    {
        return juce::Result::fail("Insufficient memory for DNNI inference.");
    }
}

/**
 * @brief Performs SIMD-vectorized matrix-vector multiplication: output = matrix * input.
 *
 * Vectorization strategy:
 * - Operates in blocks of channelsPerBlock (4 SIMD registers = 16 or 32 floats).
 * - For each block of rows, broadcasts input[column] across the register and accumulates dot products.
 */
void DnniInference::multiplyMatrix(const Matrix& matrix, const float* input, float* output)
{
    for (std::size_t firstChannel = 0; firstChannel < matrix.rows; firstChannel += channelsPerBlock)
    {
        Vector sum0(0.0f);
        Vector sum1(0.0f);
        Vector sum2(0.0f);
        Vector sum3(0.0f);
        const auto* weights = matrix.values.data() + firstChannel / channelsPerBlock * matrix.columns;
        for (std::size_t column = 0; column < matrix.columns; ++column)
        {
            const auto value = Vector::expand(input[column]);
            sum0 += weights[column][0] * value;
            sum1 += weights[column][1] * value;
            sum2 += weights[column][2] * value;
            sum3 += weights[column][3] * value;
        }
        alignas(Vector) std::array<float, channelsPerBlock> values;
        sum0.copyToRawArray(values.data());
        sum1.copyToRawArray(values.data() + Vector::size());
        sum2.copyToRawArray(values.data() + 2 * Vector::size());
        sum3.copyToRawArray(values.data() + 3 * Vector::size());
        std::copy_n(values.data(), std::min(channelsPerBlock, matrix.rows - firstChannel), output + firstChannel);
    }
}

/**
 * @brief Executes Gated Recurrent Unit (GRU) forward pass over the sequence.
 *
 * Model weights structure (`modl3`):
 * - 6 projection matrices:
 *   - Projections 0..2: Input-to-hidden projections for reset (r), update (z), and candidate (n) gates
 *   - Projections 3..5: Hidden-to-hidden recurrent projections for r, z, and n
 * - Bias vector has 6 * hiddenChannels entries: [b_ir, b_iz, b_in, b_hr, b_hz, b_hn]
 *
 * Recurrence equations (PyTorch reset_after=True convention):
 * 1. Reset gate:  r_t = sigmoid(W_ir * x_t + b_ir + W_hr * h_{t-1} + b_hr)
 * 2. Update gate: z_t = sigmoid(W_iz * x_t + b_iz + W_hz * h_{t-1} + b_hz)
 * 3. Candidate:   n_t = tanh(W_in * x_t + b_in + r_t * (W_hn * h_{t-1} + b_hn))
 * 4. Hidden state update: h_t = z_t * h_{t-1} + (1 - z_t) * n_t
 *
 * @param reverse If true, runs backward from last frame to first (for bidirectional GRU).
 */
juce::Result DnniInference::runGru(const Layer& layer, const DnniTensor& input, DnniTensor& output, bool reverse, const std::function<bool()>& shouldCancel)
{
    if (input.channels != layer.inputChannels || !validShape(input.frames, layer.gateChannels))
    {
        return nodeError(layer.sourceOffset, "GRU input channels do not match the model or output exceeds the memory limit.");
    }
    const auto hiddenChannels = layer.gateChannels;
    output.frames = input.frames;
    output.channels = hiddenChannels;
    output.values.resize(output.frames * output.channels);
    std::vector<float> hidden(hiddenChannels, 0.0f);
    std::vector<float> projections(6 * hiddenChannels);
    for (std::size_t step = 0; step < input.frames; ++step)
    {
        if (step % 32 == 0 && shouldCancel && shouldCancel())
        {
            return juce::Result::fail("DNNI inference cancelled.");
        }
        const auto frame = reverse ? input.frames - 1 - step : step;
        const auto* inputFrame = input.values.data() + frame * input.channels;
        for (std::size_t projection = 0; projection < 6; ++projection)
        {
            const auto* source = projection < 3 ? inputFrame : hidden.data();
            multiplyMatrix(layer.matrices[projection], source, projections.data() + projection * hiddenChannels);
        }
        if (!finiteValues(projections))
        {
            return nodeError(layer.sourceOffset, "GRU projection produced a non-finite value.");
        }
        // Evaluate gates per channel
        for (std::size_t channel = 0; channel < hiddenChannels; ++channel)
        {
            const auto resetInput = layer.bias[channel] + layer.bias[3 * hiddenChannels + channel] + projections[channel] + projections[3 * hiddenChannels + channel];
            const auto updateInput = layer.bias[hiddenChannels + channel] + layer.bias[4 * hiddenChannels + channel] + projections[hiddenChannels + channel] + projections[4 * hiddenChannels + channel];
            const auto candidateRecurrent = layer.bias[5 * hiddenChannels + channel] + projections[5 * hiddenChannels + channel];
            const auto candidateInput = sigmoid(resetInput) * candidateRecurrent + projections[2 * hiddenChannels + channel] + layer.bias[2 * hiddenChannels + channel];
            if (!std::isfinite(resetInput) || !std::isfinite(updateInput) || !std::isfinite(candidateInput))
            {
                return nodeError(layer.sourceOffset, "GRU gate produced a non-finite value.");
            }
            const auto update = sigmoid(updateInput);
            hidden[channel] = update * hidden[channel] + (1.0f - update) * std::tanh(candidateInput);
        }
        std::copy(hidden.begin(), hidden.end(), output.values.begin() + static_cast<std::ptrdiff_t>(frame * hiddenChannels));
    }
    return juce::Result::ok();
}

juce::Result DnniInference::runLayer(const Layer& layer, const DnniTensor& input, DnniTensor& output, const DnniTensor* condition, const std::function<bool()>& shouldCancel)
{
    if (shouldCancel && shouldCancel())
    {
        return juce::Result::fail("DNNI inference cancelled.");
    }
    if (layer.operation == Operation::sequence)
    {
        auto current = input;
        for (const auto& child : layer.children)
        {
            DnniTensor next;
            if (const auto result = runLayer(child, current, next, condition, shouldCancel); result.failed())
            {
                return result;
            }
            current = std::move(next);
        }
        output = std::move(current);
        return juce::Result::ok();
    }

    if (layer.operation == Operation::gru)
    {
        return runGru(layer, input, output, false, shouldCancel);
    }

    if (layer.operation == Operation::bidirectionalGru)
    {
        if (!validShape(input.frames, layer.gateChannels))
        {
            return nodeError(layer.sourceOffset, "bidirectional GRU output exceeds the memory limit.");
        }
        DnniTensor forward;
        DnniTensor backward;
        if (const auto result = runGru(layer.children[0], input, forward, false, shouldCancel); result.failed())
        {
            return result;
        }
        if (const auto result = runGru(layer.children[1], input, backward, true, shouldCancel); result.failed())
        {
            return result;
        }
        output.frames = input.frames;
        output.channels = layer.gateChannels;
        output.values.resize(output.frames * output.channels);
        // modl6 buffers through end-of-sequence and concatenates forward and
        // backward channels at their original frame positions, in that order.
        for (std::size_t frame = 0; frame < output.frames; ++frame)
        {
            auto* destination = output.values.data() + frame * output.channels;
            std::copy_n(forward.values.data() + frame * forward.channels, forward.channels, destination);
            std::copy_n(backward.values.data() + frame * backward.channels, backward.channels, destination + forward.channels);
        }
        return juce::Result::ok();
    }

    // --- Residual Dilated Convolution Stack (_ncwnv0 / WaveNet architecture) ---
    // Contains stageCount stages of:
    // 1. Dilated gated convolution unit (children[stage * 3])
    // 2. Skip projection (children[stage * 3 + 1]) -> accumulated directly into output
    // 3. Residual projection (children[stage * 3 + 2]) -> added back to input for the next stage
    if (layer.operation == Operation::residualConvolution)
    {
        if (input.channels != layer.inputChannels)
        {
            return nodeError(layer.sourceOffset, "residual convolution input channel count does not match the network.");
        }
        if (layer.conditionChannels > 0)
        {
            if (condition == nullptr || condition->channels != layer.conditionChannels || condition->frames != input.frames)
            {
                return nodeError(layer.sourceOffset, "residual convolution requires a condition tensor with matching frames and declared channels.");
            }
            if (const auto result = runLayer(layer.children.back(), *condition, output, nullptr, shouldCancel); result.failed())
            {
                return result;
            }
        }
        else
        {
            output.frames = input.frames;
            output.channels = layer.gateChannels;
            if (!validShape(output.frames, output.channels))
            {
                return nodeError(layer.sourceOffset, "residual convolution output exceeds the memory limit.");
            }
            output.values.assign(output.frames * output.channels, 0.0f);
        }
        auto current = input;
        for (std::size_t stage = 0; stage < layer.stageCount; ++stage)
        {
            DnniTensor residual;
            if (const auto result = runLayer(layer.children[stage * 3 + 2], current, residual, nullptr, shouldCancel); result.failed())
            {
                return result;
            }
            DnniTensor gated;
            if (const auto result = runLayer(layer.children[stage * 3], current, gated, condition, shouldCancel); result.failed())
            {
                return result;
            }
            DnniTensor skip;
            if (const auto result = runLayer(layer.children[stage * 3 + 1], gated, skip, nullptr, shouldCancel); result.failed())
            {
                return result;
            }
            if (residual.frames != output.frames || gated.frames != output.frames || skip.frames != output.frames || residual.channels != output.channels || gated.channels != output.channels || skip.channels != output.channels)
            {
                return nodeError(layer.sourceOffset, "residual convolution branches produced incompatible shapes.");
            }
            // Skip connection contributes to overall network output
            juce::FloatVectorOperations::add(output.values.data(), skip.values.data(), output.values.size());
            // Residual connection feeds into next dilated stage
            juce::FloatVectorOperations::add(gated.values.data(), residual.values.data(), gated.values.size());
            if (!finiteValues(output.values) || !finiteValues(gated.values))
            {
                return nodeError(layer.sourceOffset, "residual convolution accumulation produced a non-finite value.");
            }
            current = std::move(gated);
        }
        return juce::Result::ok();
    }

    // --- Gated Convolutional Unit (_gnc1v0) ---
    // Formula: output = tanh(filter) * sigmoid(gate)
    // where input is projected to 2 * gateChannels (first half = filter, second half = gate)
    if (layer.operation == Operation::gatedConvolution)
    {
        if (input.channels != layer.inputChannels)
        {
            return nodeError(layer.sourceOffset, "gated Conv1D input channels do not match the declared input.");
        }
        if (layer.conditionChannels > 0 && (condition == nullptr || condition->channels != layer.conditionChannels))
        {
            return nodeError(layer.sourceOffset, "gated Conv1D requires a condition tensor with the declared channel count.");
        }
        DnniTensor gateInput;
        if (const auto result = runLayer(layer.children[0], input, gateInput, nullptr, shouldCancel); result.failed())
        {
            return result;
        }
        if (gateInput.channels != 2 * layer.gateChannels)
        {
            return nodeError(layer.sourceOffset, "gated Conv1D input projection must produce two gate-channel groups.");
        }
        if (layer.conditionChannels > 0)
        {
            DnniTensor projectedCondition;
            if (const auto result = runLayer(layer.children[1], *condition, projectedCondition, nullptr, shouldCancel); result.failed())
            {
                return result;
            }
            if (projectedCondition.frames != gateInput.frames || projectedCondition.channels != gateInput.channels)
            {
                return nodeError(layer.sourceOffset, "input and condition convolutions must produce matching frames and channels.");
            }
            juce::FloatVectorOperations::add(gateInput.values.data(), projectedCondition.values.data(), gateInput.values.size());
            if (!finiteValues(gateInput.values))
            {
                return nodeError(layer.sourceOffset, "gated Conv1D conditioning produced a non-finite value.");
            }
        }
        output.frames = gateInput.frames;
        output.channels = layer.gateChannels;
        output.values.resize(output.frames * output.channels);
        for (std::size_t frame = 0; frame < output.frames; ++frame)
        {
            const auto gateOffset = frame * gateInput.channels;
            for (std::size_t channel = 0; channel < output.channels; ++channel)
            {
                output.values[frame * output.channels + channel] = std::tanh(gateInput.values[gateOffset + channel]) * sigmoid(gateInput.values[gateOffset + layer.gateChannels + channel]);
            }
        }
        return juce::Result::ok();
    }

    if (layer.operation == Operation::dense || layer.operation == Operation::convolution)
    {
        const auto& firstMatrix = layer.matrices.front();
        if (input.channels != firstMatrix.columns)
        {
            return nodeError(layer.sourceOffset, "input channels do not match the matrix columns.");
        }
        output.channels = firstMatrix.rows;
        output.frames = input.frames;
        if (layer.operation == Operation::convolution)
        {
            // Parameters are int32 and tensor dimensions are bounded, so these fit uint64.
            const auto receptiveField = static_cast<std::uint64_t>(layer.matrices.size() - 1) * layer.dilation + 1;
            const auto paddedFrames = static_cast<std::uint64_t>(input.frames) + 2 * static_cast<std::uint64_t>(layer.padding);
            output.frames = paddedFrames < receptiveField ? 0 : static_cast<std::size_t>((paddedFrames - receptiveField) / layer.stride + 1);
        }
        if (!validShape(output.frames, output.channels))
        {
            return nodeError(layer.sourceOffset, "output tensor exceeds the memory limit.");
        }
        output.values.assign(output.frames * output.channels, 0.0f);
        for (std::size_t frame = 0; frame < output.frames; frame += 2)
        {
            if (frame % 32 == 0 && shouldCancel && shouldCancel())
            {
                return juce::Result::fail("DNNI inference cancelled.");
            }
            auto* outputFrame = output.values.data() + frame * output.channels;
            const auto hasSecondFrame = frame + 1 < output.frames;
            for (std::size_t firstChannel = 0; firstChannel < output.channels; firstChannel += channelsPerBlock)
            {
                // Share each weight load between two frames while preserving each output's
                // kernel/column summation order and keeping both sets of sums in registers.
                Vector sum0(0.0f);
                Vector sum1(0.0f);
                Vector sum2(0.0f);
                Vector sum3(0.0f);
                Vector secondSum0(0.0f);
                Vector secondSum1(0.0f);
                Vector secondSum2(0.0f);
                Vector secondSum3(0.0f);
                const auto block = firstChannel / channelsPerBlock;
                for (std::size_t kernel = 0; kernel < layer.matrices.size(); ++kernel)
                {
                    const auto inputFrame = static_cast<std::int64_t>(frame * layer.stride + kernel * layer.dilation) - static_cast<std::int64_t>(layer.padding);
                    const auto secondInputFrame = inputFrame + static_cast<std::int64_t>(layer.stride);
                    const auto hasFirstInput = inputFrame >= 0 && static_cast<std::uint64_t>(inputFrame) < input.frames;
                    const auto hasSecondInput = hasSecondFrame && secondInputFrame >= 0 && static_cast<std::uint64_t>(secondInputFrame) < input.frames;
                    if (!hasFirstInput && !hasSecondInput)
                    {
                        continue;
                    }
                    const auto& matrix = layer.matrices[kernel];
                    const auto* weights = matrix.values.data() + block * input.channels;
                    if (hasFirstInput && hasSecondInput)
                    {
                        const auto* inputValues = input.values.data() + static_cast<std::size_t>(inputFrame) * input.channels;
                        const auto* secondInputValues = input.values.data() + static_cast<std::size_t>(secondInputFrame) * input.channels;
                        for (std::size_t column = 0; column < input.channels; ++column)
                        {
                            const auto value = Vector::expand(inputValues[column]);
                            const auto secondValue = Vector::expand(secondInputValues[column]);
                            const auto weight0 = weights[column][0];
                            sum0 += weight0 * value;
                            secondSum0 += weight0 * secondValue;
                            const auto weight1 = weights[column][1];
                            sum1 += weight1 * value;
                            secondSum1 += weight1 * secondValue;
                            const auto weight2 = weights[column][2];
                            sum2 += weight2 * value;
                            secondSum2 += weight2 * secondValue;
                            const auto weight3 = weights[column][3];
                            sum3 += weight3 * value;
                            secondSum3 += weight3 * secondValue;
                        }
                    }
                    else if (hasFirstInput)
                    {
                        // Preserve kernel skips at padding boundaries and for an unpaired tail frame.
                        const auto* inputValues = input.values.data() + static_cast<std::size_t>(inputFrame) * input.channels;
                        for (std::size_t column = 0; column < input.channels; ++column)
                        {
                            const auto value = Vector::expand(inputValues[column]);
                            sum0 += weights[column][0] * value;
                            sum1 += weights[column][1] * value;
                            sum2 += weights[column][2] * value;
                            sum3 += weights[column][3] * value;
                        }
                    }
                    else
                    {
                        const auto* inputValues = input.values.data() + static_cast<std::size_t>(secondInputFrame) * input.channels;
                        for (std::size_t column = 0; column < input.channels; ++column)
                        {
                            const auto value = Vector::expand(inputValues[column]);
                            secondSum0 += weights[column][0] * value;
                            secondSum1 += weights[column][1] * value;
                            secondSum2 += weights[column][2] * value;
                            secondSum3 += weights[column][3] * value;
                        }
                    }
                }
                alignas(Vector) std::array<float, channelsPerBlock> values;
                sum0.copyToRawArray(values.data());
                sum1.copyToRawArray(values.data() + Vector::size());
                sum2.copyToRawArray(values.data() + 2 * Vector::size());
                sum3.copyToRawArray(values.data() + 3 * Vector::size());
                std::copy_n(values.data(), std::min(channelsPerBlock, output.channels - firstChannel), outputFrame + firstChannel);
                if (hasSecondFrame)
                {
                    secondSum0.copyToRawArray(values.data());
                    secondSum1.copyToRawArray(values.data() + Vector::size());
                    secondSum2.copyToRawArray(values.data() + 2 * Vector::size());
                    secondSum3.copyToRawArray(values.data() + 3 * Vector::size());
                    std::copy_n(values.data(), std::min(channelsPerBlock, output.channels - firstChannel), outputFrame + output.channels + firstChannel);
                }
            }
            if (!layer.bias.empty())
            {
                juce::FloatVectorOperations::add(outputFrame, layer.bias.data(), output.channels);
                if (hasSecondFrame)
                {
                    juce::FloatVectorOperations::add(outputFrame + output.channels, layer.bias.data(), output.channels);
                }
            }
        }
    }
    else
    {
        output = input;
        for (auto& value : output.values)
        {
            switch (layer.operation)
            {
                case Operation::relu:
                    value = std::max(value, 0.0f);
                    break;
                case Operation::tanh:
                    value = std::tanh(value);
                    break;
                case Operation::sigmoid:
                case Operation::silu:
                {
                    // Equivalent to 1/(1+exp(-x)), avoiding overflow for negative inputs.
                    const auto activation = sigmoid(value);
                    value = layer.operation == Operation::silu ? value * activation : activation;
                    break;
                }
                case Operation::leakyRelu:
                    value *= value > 0.0f ? 1.0f : layer.alpha;
                    break;
                case Operation::elu:
                    // Preserve the reference formula for every finite alpha, including negative values.
                    // A non-negative alpha leaves positive inputs unchanged without evaluating exp(x).
                    if (layer.alpha < 0.0f || value < 0.0f)
                    {
                        value = std::max(value, 0.0f) + std::min(layer.alpha * std::expm1(value), 0.0f);
                    }
                    break;
                case Operation::sequence:
                case Operation::dense:
                case Operation::convolution:
                case Operation::gatedConvolution:
                case Operation::residualConvolution:
                case Operation::gru:
                case Operation::bidirectionalGru:
                case Operation::identity:
                    break;
            }
        }
    }
    if (!finiteValues(output.values))
    {
        return nodeError(layer.sourceOffset, "operator produced a non-finite output.");
    }
    return juce::Result::ok();
}
} // namespace sv::synthesis
