#pragma once

#include "DnniInference.h"
#include "DnniReader.h"
#include "PitchFeatures.h"

#include <juce_core/juce_core.h>

#include <cstddef>
#include <functional>
#include <span>

namespace sv::synthesis
{
/**
 * @brief Neural context encoder module for the pitch prediction model.
 *
 * Responsibilities:
 * 1. Categorical embeddings: Maps phoneme categories and languages into 32-channel dense embeddings.
 * 2. Recurrent encoders: Evaluates bidirectional note encoder and phoneme encoder.
 * 3. Frame expansion: Expands note-rate and phoneme-rate features onto the 5ms acoustic frame grid.
 * 4. Dual-branch projection: Evaluates shared projection, feeding into separate feed-forward
 *    and residual output conditioning representations for the pitch decoder.
 */
class PitchContext
{
public:
    /**
     * @brief Retained inference snapshots for phrase-level incremental caching.
     */
    struct State
    {
        DnniInference::Cache projection;
        DnniInference::Cache feedForward;
        DnniInference::Cache residual;

        [[nodiscard]] std::size_t getBytes() const noexcept;
    };

    /**
     * @brief Loads neural layers and embedding matrices from the DNNI context node.
     */
    [[nodiscard]] juce::Result load(const DnniReader& reader, std::size_t nodeIndex, std::size_t phonemeCategoryCount, std::size_t languageCount);

    /**
     * @brief Executes the context encoding pipeline.
     *
     * @param features Encoded note and phoneme features from PitchFeatures.
     * @param speaker Speaker / voice embedding vector.
     * @param feedForwardOutput Output tensor for the feed-forward conditioning branch.
     * @param residualOutput Output tensor for the residual conditioning branch.
     * @param state Optional incremental cache state.
     * @param shouldCancel Callback to check for user cancellation.
     * @param statistics Optional profiling stats recorder.
     */
    [[nodiscard]] juce::Result run(const PitchFeatureOutput& features, std::span<const float> speaker, DnniTensor& feedForwardOutput, DnniTensor& residualOutput, State* state = nullptr, const std::function<bool()>& shouldCancel = {}, DnniRunStatistics* statistics = nullptr) const;

private:
    DnniMatrix phonemeEmbedding;
    DnniMatrix languageEmbedding;
    DnniInference noteEncoder;
    DnniInference phonemeEncoder;
    DnniInference projection;
    DnniInference feedForward;
    DnniInference residual;
    bool loaded = false;
};

} // namespace sv::synthesis
