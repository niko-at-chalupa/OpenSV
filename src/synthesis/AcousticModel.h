#pragma once

#include "AcousticFeatures.h"
#include "DnniInference.h"
#include "DnniReader.h"

#include <juce_core/juce_core.h>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <span>
#include <vector>

namespace sv::synthesis
{
class AcousticModel
{
public:
    struct State
    {
        DnniInference::Cache phonemeEncoder;
        DnniInference::Cache contextProjection;
        DnniInference::Cache contextNetwork;
        DnniInference::Cache latentProjection;
        DnniInference::Cache latentNetwork;
        DnniInference::Cache latentHead;
        DnniInference::Cache spectralNetwork;
        DnniInference::Cache spectralHead;
        DnniInference::Cache auxiliaryNetwork;
        DnniInference::Cache auxiliaryHead;

        [[nodiscard]] std::size_t getBytes() const noexcept;
    };

    [[nodiscard]] juce::Result load(const DnniReader& reader, std::size_t rootNode = 0);
    [[nodiscard]] juce::Result load(const DnniReader& reader, std::span<const std::string> vocalModeNames, std::size_t rootNode = 0);
    // Whole-utterance inference with the database's default vocal mode and supplied timing/F0.
    // The seed selects our deterministic noise stream, not the original editor's retake schedule.
    [[nodiscard]] juce::Result run(std::span<const TimedPhoneme> phonemes, std::span<const float> logF0, DnniTensor& output, std::uint32_t noiseSeed = 5489, State* state = nullptr, const std::function<bool()>& shouldCancel = {}, DnniRunStatistics* statistics = nullptr, std::span<const float> vocalModeWeights = {}) const;

private:
    [[nodiscard]] juce::Result makeContext(std::span<const TimedPhoneme> phonemes, const DnniTensor& pitch, DnniTensor& output, State* state, const std::function<bool()>& shouldCancel, DnniRunStatistics* statistics, std::span<const float> vocalModeWeights) const;
    [[nodiscard]] juce::Result sampleLatent(const DnniTensor& context, const DnniTensor& projected, std::uint32_t noiseSeed, DnniTensor& output, State* state, const std::function<bool()>& shouldCancel, DnniRunStatistics* statistics) const;

    AcousticFeatures features;
    DnniInference phonemeEncoder;
    DnniInference contextProjection;
    DnniInference contextNetwork;
    DnniInference latentProjection;
    DnniInference latentNetwork;
    DnniInference latentHead;
    DnniInference spectralNetwork;
    DnniInference spectralHead;
    DnniInference auxiliaryNetwork;
    DnniInference auxiliaryHead;
    DnniMatrix languageEmbedding;
    std::vector<float> speaker;
    std::vector<float> defaultStyle;
    std::vector<std::vector<float>> vocalModeStyles;
    bool loaded = false;
};
} // namespace sv::synthesis
