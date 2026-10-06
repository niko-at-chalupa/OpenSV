#include "PitchDecoder.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <new>
#include <string>
#include <utility>

namespace sv::synthesis
{
namespace
{
/// Memory limit safety ceiling (64 million single-precision elements ~ 256 MB) to prevent out-of-memory crashes.
constexpr std::size_t maximumElements = 64 * 1024 * 1024;

/**
 * @brief Reads a 32-bit little-endian integer from a byte buffer.
 */
std::uint32_t readUint32(std::span<const std::uint8_t> bytes, std::size_t offset)
{
    return static_cast<std::uint32_t>(bytes[offset]) | (static_cast<std::uint32_t>(bytes[offset + 1]) << 8) | (static_cast<std::uint32_t>(bytes[offset + 2]) << 16) | (static_cast<std::uint32_t>(bytes[offset + 3]) << 24);
}

/**
 * @brief Validates tensor dimensions and checks that all values are finite.
 */
bool validTensor(const DnniTensor& tensor)
{
    return tensor.channels > 0 && tensor.channels <= maximumElements && tensor.frames <= maximumElements / tensor.channels && tensor.values.size() == tensor.frames * tensor.channels && std::all_of(tensor.values.begin(), tensor.values.end(), [](float value)
                                                                                                                                                                                                      { return std::isfinite(value); });
}

/**
 * @brief Checks if all elements in a float slice are finite numbers (no NaN or Inf).
 */
bool finiteValues(std::span<const float> values)
{
    return std::all_of(values.begin(), values.end(), [](float value)
                       { return std::isfinite(value); });
}

/**
 * @brief Helper for generating standardized failure results from the pitch decoder.
 */
juce::Result decoderError(const juce::String& reason)
{
    return juce::Result::fail("Pitch decoder: " + reason);
}

/**
 * @brief Upsamples a downsampled/grouped tensor by repeating each group's channel vector `stride` times.
 *
 * For example, if input has `groups` frames, output will have `frames` frames where frame `t` copies
 * from group `t / stride`.
 */
DnniTensor repeatFrames(const DnniTensor& input, std::size_t frames, std::size_t stride)
{
    DnniTensor output{frames, input.channels, std::vector<float>(frames * input.channels)};
    for (std::size_t frame = 0; frame < frames; ++frame)
    {
        std::copy_n(input.values.begin() + static_cast<std::ptrdiff_t>((frame / stride) * input.channels), input.channels, output.values.begin() + static_cast<std::ptrdiff_t>(frame * input.channels));
    }
    return output;
}

/**
 * @brief Computes a statistical quantile (in-place partial sort using std::nth_element).
 *
 * @param values Mutable vector of values to partition.
 * @param position Normalized quantile in [0.0, 1.0] (e.g. 0.5 for median, 0.8 for 80th percentile).
 * @return The value at the quantile index.
 */
float quantile(std::vector<float>& values, float position)
{
    if (values.empty())
    {
        return 0.0f;
    }
    const auto index = static_cast<std::size_t>(static_cast<float>(values.size() - 1) * position);
    auto selected = values.begin() + static_cast<std::ptrdiff_t>(index);
    std::nth_element(values.begin(), selected, values.end());
    return *selected;
}

/**
 * @brief Expands per-note control scalars (e.g. tilt, shift) into a continuous per-frame signal.
 *
 * Synthesizer V Gen5 smooths transitions across note boundaries by extending each note's influence
 * by 5 frames (25 ms) before onset and after offset, weighting the value by a two-sided sigmoid:
 * weight = 1 / [ (1 + exp(-(duration - localFrame))) * (1 + exp(-localFrame)) ].
 *
 * @param counts Frame duration count of each note.
 * @param values Control value per note.
 * @param frames Total number of frames in the phrase.
 * @return Continuous per-frame expanded signal.
 */
std::vector<float> expandNoteValues(std::span<const std::size_t> counts, std::span<const float> values, std::size_t frames)
{
    std::vector<float> output(frames, 0.0f);
    std::int64_t onset = 0;
    for (std::size_t note = 0; note < counts.size(); ++note)
    {
        const auto duration = static_cast<std::int64_t>(counts[note]);
        if (values[note] != 0.0f)
        {
            const auto first = std::max<std::int64_t>(-5, -onset);
            const auto end = std::min(duration + 5, static_cast<std::int64_t>(frames) - onset);
            for (auto localFrame = first; localFrame < end; ++localFrame)
            {
                // Gen5 extends a note's control by five 5 ms frames on each side with smooth sigmoid tapering.
                const auto weight = static_cast<float>(1.0 / ((1.0 + std::exp(-static_cast<double>(duration - localFrame))) * (1.0 + std::exp(-static_cast<double>(localFrame)))));
                output[static_cast<std::size_t>(onset + localFrame)] += values[note] * weight;
            }
        }
        onset += duration;
    }
    return output;
}
} // namespace

juce::Result PitchDecoder::load(const DnniReader& reader, std::size_t nodeIndex)
{
    loaded = false;
    const auto& nodes = reader.getNodes();

    // Recognized 64-bit type identifiers for Gen5 pitch decoder containers across various voice versions.
    constexpr std::array<std::uint64_t, 12> decoderTypes{
        0x67133569b1e0c9a7, 0x0833163a9cfbe985, 0x9d1ceeb019e4ee82, 0xc2d683aaffb9c539,
        0x7878dd761d208ab5, 0x833c88b73fdaad72, 0x83853dd7b2ffc31b, 0xcd0750935e49d6e3,
        0x37277fa85c86b715, 0x0dcc23701223d7ca, 0xe886cc8e9fe491d4, 0xca7b4a8a02707df0};

    if (nodeIndex >= nodes.size() || std::find(decoderTypes.begin(), decoderTypes.end(), nodes[nodeIndex].typeId) == decoderTypes.end() || nodes[nodeIndex].children.size() != networks.size())
    {
        return decoderError("expected the gen5 nine-network decoder.");
    }

    // 12-byte payload:
    // [0..3]: int32_t embeddingChannels (e.g. 64)
    // [4..7]: float   embeddingScale
    // [8..11]: int32_t frameStride (e.g. 4)
    const auto payload = reader.getPayload(nodeIndex);
    if (payload.size() != 12)
    {
        return decoderError("embedding and frame-group parameters are missing.");
    }
    const auto channels = std::bit_cast<std::int32_t>(readUint32(payload, 0));
    const auto scale = std::bit_cast<float>(readUint32(payload, 4));
    const auto stride = std::bit_cast<std::int32_t>(readUint32(payload, 8));
    if (channels < 2 || channels % 2 != 0 || channels > 4096 || stride <= 0 || stride > 4096 || !std::isfinite(scale) || scale <= 0.0f)
    {
        return decoderError("invalid sinusoidal embedding or frame-group dimensions.");
    }

    // Load each of the 9 child inference sub-networks
    for (std::size_t index = 0; index < networks.size(); ++index)
    {
        if (const auto result = networks[index].load(reader, nodes[nodeIndex].children[index]); result.failed())
        {
            return decoderError("network " + juce::String(static_cast<int>(index)) + ": " + result.getErrorMessage());
        }
    }
    embeddingChannels = static_cast<std::size_t>(channels);
    embeddingScale = scale;
    frameStride = static_cast<std::size_t>(stride);
    loaded = true;
    return juce::Result::ok();
}


juce::Result PitchDecoder::run(const DnniTensor& context, const DnniTensor& scalarContext, std::span<const float> noise, const DnniTensor& controls, std::span<const float> vibratoControl, const PitchFeatures& frontend, PitchDecoderNotes& notes, DnniTensor& output, const std::function<bool()>& shouldCancel, DnniRunStatistics* statistics) const
{
    if (!loaded)
    {
        return decoderError("model has not been loaded.");
    }
    if (!validTensor(context) || context.frames == 0 || !validTensor(scalarContext) || scalarContext.frames != context.frames || !validTensor(controls) || controls.frames != context.frames || controls.channels != 3 || noise.size() != context.frames || vibratoControl.size() != context.frames || !finiteValues(noise) || !finiteValues(vibratoControl))
    {
        return decoderError("context, noise, and control frames do not match.");
    }
    const auto noteCount = notes.frameCounts.size();
    if (notes.midiPitch.size() != noteCount || notes.vowelFrames.size() != context.frames || (!notes.requestedTilt.empty() && notes.requestedTilt.size() != noteCount) || (!notes.requestedShift.empty() && notes.requestedShift.size() != noteCount) || !finiteValues(notes.midiPitch))
    {
        return decoderError("note durations, pitches, and vowel mask do not match.");
    }
    std::size_t countedFrames = 0;
    for (const auto count : notes.frameCounts)
    {
        if (count > context.frames - countedFrames)
        {
            return decoderError("note durations extend beyond the input frames.");
        }
        countedFrames += count;
    }
    const auto frames = context.frames;
    const auto groups = (frames + frameStride - 1) / frameStride;
    const auto conditionChannels = context.channels + embeddingChannels * 3 + frameStride;
    if (conditionChannels > maximumElements || groups > maximumElements / conditionChannels || context.channels > maximumElements / (frames * 2))
    {
        return decoderError("working tensors exceed the memory limit.");
    }
    try
    {
        const auto runNetwork = [this, &shouldCancel, statistics](std::size_t index, const DnniTensor& input, DnniTensor& result, const DnniTensor* condition = nullptr)
        {
            const auto status = networks[index].run(input, result, condition, nullptr, shouldCancel, statistics);
            if (status.failed())
            {
                return decoderError("network " + juce::String(static_cast<int>(index)) + ": " + status.getErrorMessage());
            }
            return status;
        };
        DnniTensor scalar;
        if (const auto result = runNetwork(0, scalarContext, scalar); result.failed())
        {
            return result;
        }
        if (scalar.frames != frames || scalar.channels != 1)
        {
            return decoderError("initial projection must produce one scalar per frame.");
        }
        DnniTensor groupedScalar{groups, frameStride, std::vector<float>(groups * frameStride)};
        DnniTensor pooledContext{groups, context.channels, std::vector<float>(groups * context.channels, 0.0f)};
        DnniTensor condition{groups, conditionChannels, std::vector<float>(groups * conditionChannels, 0.0f)};
        const auto divisor = static_cast<float>(frameStride);
        const auto embeddingHalf = embeddingChannels / 2;
        for (std::size_t group = 0; group < groups; ++group)
        {
            if (shouldCancel && shouldCancel())
            {
                return decoderError("cancelled.");
            }
            std::array<float, 3> controlSum{};
            auto* grouped = groupedScalar.values.data() + group * frameStride;
            auto* pooled = pooledContext.values.data() + group * context.channels;
            auto* conditional = condition.values.data() + group * conditionChannels;
            for (std::size_t localFrame = 0; localFrame < frameStride; ++localFrame)
            {
                // The last incomplete group repeats its final frame before pooling.
                const auto frame = std::min(group * frameStride + localFrame, frames - 1);
                grouped[localFrame] = scalar.values[frame];
                conditional[context.channels + embeddingChannels * 3 + localFrame] = noise[frame];
                for (std::size_t channel = 0; channel < context.channels; ++channel)
                {
                    pooled[channel] += context.values[frame * context.channels + channel];
                }
                for (std::size_t control = 0; control < controlSum.size(); ++control)
                {
                    controlSum[control] += controls.values[frame * 3 + control];
                }
            }
            for (std::size_t channel = 0; channel < context.channels; ++channel)
            {
                pooled[channel] /= divisor;
                conditional[channel] = pooled[channel];
            }
            for (std::size_t control = 0; control < controlSum.size(); ++control)
            {
                const auto position = controlSum[control] * embeddingScale / divisor;
                auto* embedding = conditional + context.channels + control * embeddingChannels;
                for (std::size_t channel = 0; channel < embeddingHalf; ++channel)
                {
                    // -2 * log2(10000), as stored by the original sinusoidal encoder.
                    const auto phase = std::exp2(static_cast<float>(channel) * -26.575424194335938f / static_cast<float>(embeddingChannels)) * position;
                    embedding[channel] = std::sin(phase);
                    embedding[channel + embeddingHalf] = std::cos(phase);
                }
            }
        }
        DnniTensor coarse;
        if (const auto result = runNetwork(5, groupedScalar, coarse, &condition); result.failed())
        {
            return result;
        }
        if (coarse.frames != groups || coarse.channels != context.channels)
        {
            return decoderError("first residual pass changed the expected frame groups or context width.");
        }
        DnniTensor preliminary;
        if (const auto result = runNetwork(7, repeatFrames(coarse, frames, frameStride), preliminary); result.failed())
        {
            return result;
        }
        if (preliminary.frames != frames || preliminary.channels != 1)
        {
            return decoderError("first pitch head must produce one scalar per frame.");
        }
        std::vector<float> preliminaryMidi;
        if (const auto result = frontend.denormalizePitch(preliminary.values, preliminaryMidi); result.failed())
        {
            return result;
        }
        std::vector<float> predictedTilt(noteCount, 0.0f);
        std::vector<float> predictedShift(noteCount, 0.0f);
        std::vector<float> desiredTilt(noteCount, 0.0f);
        std::vector<float> desiredShift(noteCount, 0.0f);
        std::vector<float> tiltWeight(noteCount, 0.0f);
        std::vector<float> shiftWeight(noteCount, 0.0f);
        std::vector<float> differences;
        std::vector<float> pitches;
        std::size_t onset = 0;
        for (std::size_t note = 0; note < noteCount; ++note)
        {
            differences.clear();
            pitches.clear();
            for (std::size_t frame = onset; frame < onset + notes.frameCounts[note]; ++frame)
            {
                if (notes.vowelFrames[frame] != 0)
                {
                    differences.push_back(frame == 0 ? 0.0f : preliminaryMidi[frame] - preliminaryMidi[frame - 1]);
                    pitches.push_back(preliminaryMidi[frame]);
                }
            }
            predictedTilt[note] = static_cast<float>(static_cast<double>(differences.size()) * 0.05 * static_cast<double>(quantile(differences, 0.5f)));
            predictedShift[note] = pitches.empty() ? 0.0f : quantile(pitches, 0.8f) - notes.midiPitch[note];
            const auto applyTarget = [note](std::span<const float> requested, const std::vector<float>& predicted, float scale, std::vector<float>& desired, std::vector<float>& weights)
            {
                if (!requested.empty() && !std::isnan(requested[note]))
                {
                    desired[note] = requested[note];
                    weights[note] = 1.5f * std::tanh(std::abs(requested[note] - predicted[note]) * scale);
                }
            };
            applyTarget(notes.requestedTilt, predictedTilt, 10.0f, desiredTilt, tiltWeight);
            applyTarget(notes.requestedShift, predictedShift, 5.0f, desiredShift, shiftWeight);
            onset += notes.frameCounts[note];
        }
        const auto expandedTilt = expandNoteValues(notes.frameCounts, desiredTilt, frames);
        const auto expandedShift = expandNoteValues(notes.frameCounts, desiredShift, frames);
        const auto expandedTiltWeight = expandNoteValues(notes.frameCounts, tiltWeight, frames);
        const auto expandedShiftWeight = expandNoteValues(notes.frameCounts, shiftWeight, frames);
        DnniTensor pooledVibrato{groups, 1, std::vector<float>(groups)};
        DnniTensor pooledTilt{groups, 1, std::vector<float>(groups)};
        DnniTensor pooledShift{groups, 1, std::vector<float>(groups)};
        std::vector<float> pooledTiltWeight(groups, 0.0f);
        std::vector<float> pooledShiftWeight(groups, 0.0f);
        for (std::size_t group = 0; group < groups; ++group)
        {
            for (std::size_t localFrame = 0; localFrame < frameStride; ++localFrame)
            {
                const auto frame = std::min(group * frameStride + localFrame, frames - 1);
                pooledVibrato.values[group] += vibratoControl[frame];
                pooledTilt.values[group] += expandedTilt[frame];
                pooledShift.values[group] += expandedShift[frame];
                pooledTiltWeight[group] += expandedTiltWeight[frame];
                pooledShiftWeight[group] += expandedShiftWeight[frame];
            }
            pooledVibrato.values[group] = std::clamp(pooledVibrato.values[group] / divisor, -0.999f, 0.999f);
            pooledTilt.values[group] /= divisor;
            pooledShift.values[group] /= divisor;
            pooledTiltWeight[group] /= divisor;
            pooledShiftWeight[group] /= divisor;
        }
        DnniTensor vibratoEmbedding;
        DnniTensor tiltEmbedding;
        DnniTensor shiftEmbedding;
        if (const auto result = runNetwork(1, pooledVibrato, vibratoEmbedding); result.failed())
        {
            return result;
        }
        if (const auto result = runNetwork(2, pooledTilt, tiltEmbedding); result.failed())
        {
            return result;
        }
        if (const auto result = runNetwork(3, pooledShift, shiftEmbedding); result.failed())
        {
            return result;
        }
        if (vibratoEmbedding.frames != groups || vibratoEmbedding.channels != context.channels || tiltEmbedding.frames != groups || tiltEmbedding.channels != context.channels || shiftEmbedding.frames != groups || shiftEmbedding.channels != context.channels)
        {
            return decoderError("control embeddings must match the context width.");
        }
        DnniTensor correctionInput{groups, context.channels * 2, std::vector<float>(groups * context.channels * 2)};
        for (std::size_t group = 0; group < groups; ++group)
        {
            for (std::size_t channel = 0; channel < context.channels; ++channel)
            {
                const auto index = group * context.channels + channel;
                correctionInput.values[group * correctionInput.channels + channel] = pooledContext.values[index];
                correctionInput.values[group * correctionInput.channels + context.channels + channel] = vibratoEmbedding.values[index] + pooledTiltWeight[group] * tiltEmbedding.values[index] + pooledShiftWeight[group] * shiftEmbedding.values[index];
            }
        }
        DnniTensor correction;
        if (const auto result = runNetwork(4, correctionInput, correction); result.failed())
        {
            return result;
        }
        if (correction.frames != groups || correction.channels != context.channels)
        {
            return decoderError("control correction must match the pooled context.");
        }
        for (std::size_t group = 0; group < groups; ++group)
        {
            for (std::size_t channel = 0; channel < context.channels; ++channel)
            {
                const auto index = group * context.channels + channel;
                condition.values[group * condition.channels + channel] = pooledContext.values[index] + correction.values[index];
            }
        }
        DnniTensor refined;
        if (const auto result = runNetwork(6, coarse, refined, &condition); result.failed())
        {
            return result;
        }
        if (refined.frames != groups || refined.channels != context.channels)
        {
            return decoderError("second residual pass changed the expected frame groups or context width.");
        }
        DnniTensor result;
        if (const auto status = runNetwork(8, repeatFrames(refined, frames, frameStride), result); status.failed())
        {
            return status;
        }
        if (result.frames != frames || result.channels != 1)
        {
            return decoderError("final pitch head must produce one scalar per frame.");
        }
        output = std::move(result);
        notes.predictedTilt = std::move(predictedTilt);
        notes.predictedShift = std::move(predictedShift);
        return juce::Result::ok();
    }
    catch (const std::bad_alloc&)
    {
        return decoderError("insufficient memory for inference.");
    }
}
} // namespace sv::synthesis
