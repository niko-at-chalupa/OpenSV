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
/**
 * @brief Input musical syllable provided to the timing prediction model.
 */
struct TimingSyllable
{
    std::string language;               ///< Language identifier (e.g. "japanese", "english", "mandarin").
    std::vector<std::string> phonemes;  ///< Phoneme sequence belonging to this syllable.
    double durationSeconds = 0.0;       ///< Musical duration of the note in seconds (from tempo map).
    int midiPitch = 60;                 ///< Musical pitch (MIDI note number).
    bool isContinuation = false;        ///< True if this note is a slurred continuation of a previous lyric.
};

/**
 * @brief Output duration assigned to an individual phoneme by the neural timing model.
 */
struct PhonemeDuration
{
    std::string language;               ///< Language of this phoneme.
    std::string symbol;                 ///< Phoneme token (e.g. "k", "aa", "sil").
    std::size_t syllableIndex = 0;       ///< 0-based index of the parent syllable.
    std::size_t timingIntervalIndex = 0; ///< Index of the timing alignment interval.
    double durationSeconds = 0.0;       ///< Predicted duration in seconds (before frame quantization).
};

/**
 * @brief Quantizes continuous phoneme durations (seconds) into discrete acoustic frames.
 *
 * Invariants:
 * - Every phoneme receives at least one frame.
 * - Cumulative boundaries are accumulated from zero to prevent drift.
 *
 * @param durations Input predicted durations in seconds.
 * @param frameIntervalSeconds Acoustic model frame interval in seconds (e.g. 0.005f for 5ms).
 * @param output Output list of timed phonemes with start/end frames.
 */
[[nodiscard]] juce::Result quantizePhonemeDurations(std::span<const PhonemeDuration> durations, float frameIntervalSeconds, std::vector<TimedPhoneme>& output);

/**
 * @brief Neural phoneme duration prediction model (`gen2a` duration model).
 *
 * Predicts the continuous time duration (in seconds) of each phoneme within a singing phrase.
 *
 * Key algorithmic steps:
 * 1. Interval alignment (`alignIntervals`):
 *    - In vocal singing, vowels (syllable nuclei) must land on the beat / note onset.
 *    - Leading consonants of a syllable are shifted earlier into the preceding note or rest.
 *    - Syllables with multiple vowels are split into sub-intervals.
 * 2. Feature extraction (272 channels):
 *    - Language embedding + Phoneme embedding + Position embedding.
 *    - Musical pitch, normalized note duration, voice timbre embedding.
 * 3. Neural inference:
 *    - Input normalization via `FeatureNormalizer` (`cmpu0`).
 *    - Forward pass through `DnniInference` network.
 *    - Denormalization to produce physical duration seconds.
 */
class PhonemeTiming
{
public:
    /**
     * @brief Loads the duration model weights and embeddings from a DNNI file.
     */
    [[nodiscard]] juce::Result load(const DnniReader& reader, std::size_t rootNode = 0);

    /**
     * @brief Predicts duration in seconds for each phoneme in a sequence of syllables.
     *
     * @param syllables Sequence of notes/syllables in the phrase.
     * @param output Output vector populated with predicted phoneme durations.
     */
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
