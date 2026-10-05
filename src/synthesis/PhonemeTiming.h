#pragma once

#include "AcousticFeatures.h"
#include "DnniInference.h"
#include "FeatureNormalizer.h"
#include "PhoneSet.h"

#include <juce_core/juce_core.h>

#include <cstddef>
#include <span>
#include <string>
#include <vector>

namespace sv::synthesis
{
struct TimingSyllable
{
    std::string language;
    std::vector<std::string> phonemes;
    double durationSeconds = 0.0;
    int midiPitch = 60;
    bool isContinuation = false;
};

struct PhonemeDuration
{
    std::string language;
    std::string symbol;
    std::size_t syllableIndex = 0;
    std::size_t timingIntervalIndex = 0;
    double durationSeconds = 0.0;
};

// Quantizes cumulative boundaries from zero; every phoneme receives at least one frame.
// Supply the frame interval read from the acoustic model, preserving its float precision.
[[nodiscard]] juce::Result quantizePhonemeDurations(std::span<const PhonemeDuration> durations, float frameIntervalSeconds, std::vector<TimedPhoneme>& output);

class PhonemeTiming
{
public:
    // Copies the verified gen2a duration model; a failed load preserves the current model.
    [[nodiscard]] juce::Result load(const DnniReader& reader, std::size_t rootNode = 0);
    // Moves leading consonants into the preceding interval, then splits multiple nuclei.
    // Include an explicit leading rest if the first syllable's consonants should start early.
    // Processes the complete phrase together and preserves source syllable/language ownership.
    // Returns seconds before user timing adjustments or acoustic-frame quantization.
    // This allocates and runs neural inference; do not call from the audio callback.
    [[nodiscard]] juce::Result predict(std::span<const TimingSyllable> syllables, std::vector<PhonemeDuration>& output) const;

private:
    std::vector<PhoneSet> phoneSets;
    PhoneSet unifiedPhoneSet;
    FeatureNormalizer inputNormalization;
    FeatureNormalizer outputNormalization;
    DnniMatrix languageEmbedding;
    DnniMatrix phonemeEmbedding;
    DnniMatrix positionEmbedding;
    std::vector<float> voiceEmbedding;
    DnniInference network;
    int minimumPitch = 0;
    int maximumPitch = 0;
};
} // namespace sv::synthesis
