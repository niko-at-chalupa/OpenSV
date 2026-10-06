#pragma once

#include "DnniReader.h"

#include <juce_core/juce_core.h>
#include <juce_dsp/juce_dsp.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <vector>

namespace sv::synthesis
{
/**
 * @brief 2D tensor representation for neural network inputs, activations, and outputs.
 *
 * Data layout:
 * - frames: Time steps / sequence length (rows).
 * - channels: Feature dimension / channels per time step (columns).
 * - values: Flattened row-major array: values[frame * channels + channel].
 */
struct DnniTensor
{
    std::size_t frames = 0;    ///< Sequence length (time dimension).
    std::size_t channels = 0;  ///< Number of channels/features per frame.
    std::vector<float> values; ///< Frame-major flattened array of floats.
};

/**
 * @brief Operational statistics for a single neural inference pass.
 */
struct DnniRunStatistics
{
    std::size_t computedFrames = 0; ///< Output frames evaluated from scratch.
    std::size_t reusedFrames = 0;   ///< Frames retrieved from cache without evaluating layers.
    std::size_t contextFrames = 0;  ///< Additional receptive field context frames processed.
};

/**
 * @brief High-performance neural network inference engine with SIMD vectorization and caching.
 *
 * Supported neural architectures and layer types:
 * - Linear / Dense layers (`modl0`)
 * - 1D Dilated / Causal Convolutions (`modl1`)
 * - WaveNet-style Residual Dilated Convolution blocks (`_ncwnv0`) with Gated Convolutions (`_gnc1v0`)
 * - Gated Recurrent Units (GRU) (`modl3`) and Bidirectional GRUs (`modl4`)
 * - Non-linear activation functions: ReLU (`moda0`), Tanh (`moda1`), Sigmoid (`moda2`),
 *   LeakyReLU (`moda3`), ELU (`moda4`), Identity (`moda5`), SiLU/Swish (`moda7`)
 *
 * SIMD Optimization:
 * - Uses `juce::dsp::SIMDRegister<float>` (AVX/SSE/NEON) with blocked weight layouts
 *   (4 vectors per block = 16 or 32 channels per block).
 *
 * Incremental Inference (Cache):
 * - For convolutional networks with finite receptive fields, `DnniInference::Cache` detects
 *   which parts of the input sequence changed, expands only the affected frames by the
 *   receptive field context radius, runs inference on that slice, and pastes the result back.
 */
class DnniInference
{
public:
    /**
     * @brief Per-phrase inference cache supporting incremental dirty-range re-synthesis.
     *
     * Invariants:
     * - One Cache instance belongs to one phrase on the synthesis worker thread.
     * - Must not be accessed concurrently across multiple threads.
     */
    class Cache
    {
    public:
        void clear();
        [[nodiscard]] std::size_t getBytes() const noexcept;

    private:
        friend class DnniInference;
        DnniTensor input;
        DnniTensor condition;
        DnniTensor output;
        std::uint64_t modelIdentity = 0;
        bool hasCondition = false;
    };

    /**
     * @brief Loads and initializes neural network weights and structure from a parsed DNNI model.
     *
     * @param reader Parsed DNNI reader.
     * @param rootNode Root node index in reader's node list (defaults to 0).
     * @return juce::Result::ok() or failure description.
     */
    [[nodiscard]] juce::Result load(const DnniReader& reader, std::size_t rootNode = 0);

    /**
     * @brief Executes neural network forward inference.
     *
     * Invariants:
     * - Runs outside the real-time audio thread.
     * - Cancellation: `shouldCancel` is queried periodically between layer evaluations.
     * - Output tensor is populated on success; left untouched on failure or cancellation.
     *
     * @param input Input feature tensor [frames, inputChannels].
     * @param output Output tensor [frames, outputChannels].
     * @param condition Optional conditioning tensor (e.g. speaker embeddings, pitch contour).
     * @param cache Optional cache for incremental re-rendering of dirty phrase segments.
     * @param shouldCancel Callback returning true if rendering has been aborted by user.
     * @param statistics Optional pointer to record frame counters.
     * @return juce::Result::ok() or error.
     */
    [[nodiscard]] juce::Result run(const DnniTensor& input, DnniTensor& output, const DnniTensor* condition = nullptr, Cache* cache = nullptr, const std::function<bool()>& shouldCancel = {}, DnniRunStatistics* statistics = nullptr) const;

private:
    using Vector = juce::dsp::SIMDRegister<float>;
    static constexpr std::size_t vectorsPerBlock = 4;
    static constexpr std::size_t channelsPerBlock = vectorsPerBlock * Vector::size();
    using WeightBlock = std::array<Vector, vectorsPerBlock>;

    enum class Operation
    {
        sequence,               ///< Sequential container of layers
        dense,                  ///< Fully-connected affine transformation (x * W^T + b)
        convolution,            ///< 1D convolution with kernel size, stride, padding, dilation
        gatedConvolution,       ///< Gated activation unit: tanh(W_f * x) * sigmoid(W_g * x)
        residualConvolution,    ///< WaveNet residual stack with skip connections
        gru,                    ///< Unidirectional Gated Recurrent Unit
        bidirectionalGru,       ///< Bidirectional GRU (forward + backward concatenated)
        relu,                   ///< Rectified Linear Unit: max(0, x)
        tanh,                   ///< Hyperbolic tangent
        sigmoid,                ///< Logistic sigmoid: 1 / (1 + exp(-x))
        leakyRelu,              ///< Leaky ReLU: x >= 0 ? x : alpha * x
        elu,                    ///< Exponential Linear Unit
        identity,               ///< Pass-through identity
        silu                    ///< Sigmoid Linear Unit (Swish): x * sigmoid(x)
    };

    /**
     * @brief Blocked weight matrix layout tailored for SIMD dot products.
     */
    struct Matrix
    {
        std::size_t rows = 0;
        std::size_t columns = 0;
        std::vector<WeightBlock> values;
    };

    struct Layer
    {
        Operation operation = Operation::sequence;
        std::size_t sourceOffset = 0;
        std::vector<Layer> children;
        std::vector<Matrix> matrices;
        std::vector<float> bias;
        std::size_t stride = 1;
        std::size_t padding = 0;
        std::size_t dilation = 1;
        std::size_t inputChannels = 0;
        std::size_t gateChannels = 0;
        std::size_t conditionChannels = 0;
        std::size_t stageCount = 0;
        float alpha = 0.0f;
    };

    [[nodiscard]] static juce::Result loadLayer(const DnniReader& reader, std::size_t nodeIndex, Layer& layer, std::size_t& parameterCount);
    [[nodiscard]] static juce::Result loadMatrix(const DnniReader& reader, std::size_t nodeIndex, Matrix& packed, std::size_t& parameterCount);
    [[nodiscard]] static std::optional<std::size_t> findContextRadius(const Layer& layer);
    [[nodiscard]] static juce::Result runLayer(const Layer& layer, const DnniTensor& input, DnniTensor& output, const DnniTensor* condition, const std::function<bool()>& shouldCancel);
    [[nodiscard]] static juce::Result runGru(const Layer& layer, const DnniTensor& input, DnniTensor& output, bool reverse, const std::function<bool()>& shouldCancel);
    static void multiplyMatrix(const Matrix& matrix, const float* input, float* output);

    Layer root;
    std::optional<std::size_t> contextRadius;
    std::uint64_t modelIdentity = 0;
    bool loaded = false;
};

} // namespace sv::synthesis
