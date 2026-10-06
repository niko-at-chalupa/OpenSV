#pragma once

#include "AcousticModel.h"
#include "NeuralVocoder.h"
#include "PhonemeTiming.h"
#include "PitchModel.h"
#include "SynthesisStatistics.h"

#include <juce_core/juce_core.h>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <span>
#include <string>
#include <vector>

namespace sv::synthesis
{
// Owns decoded independent models; copying duplicates their parameters.
// Use on a worker thread, never the audio callback. The caller must serialize
// loading, inference and destruction, and keep input spans alive until return.
// No input views, file handles or original-engine objects are retained.
class VoiceSynthesizer
{
public:
    static constexpr double pitchContextSeconds = 0.5;

    // One state belongs to one phrase on the synthesis worker. Model parameters
    // remain shared; the state owns all retained inference snapshots.
    struct State
    {
        PitchModel::State pitch;
        AcousticModel::State acoustic;
        NeuralVocoder::State vocoder;

        [[nodiscard]] std::size_t getBytes() const noexcept;
    };

    // Opens the NOFS once. A failed load preserves the previously loaded voice.
    [[nodiscard]] juce::Result load(const juce::File& voice);
    // Returns zero until a voice has been loaded successfully.
    [[nodiscard]] float getFrameIntervalSeconds() const noexcept;
    [[nodiscard]] float getPitchFrameIntervalSeconds() const noexcept;
    [[nodiscard]] const std::vector<std::string>& getVocalModeNames() const noexcept;
    [[nodiscard]] juce::Result predict(std::span<const TimingSyllable> syllables, std::vector<PhonemeDuration>& output) const;
    // Takes score notes and internal rests, without acoustic preroll/release.
    // Predicts phoneme timing independently in the pitch model's context.
    // Output and optional envelope start pitchContextSeconds before the first
    // score note, and include the same amount of context after the last note.
    [[nodiscard]] juce::Result predictPitch(std::span<const PitchNote> notes, std::vector<float>& midiPitch, const std::function<bool()>& shouldCancel = {}, SynthesisStatistics* statistics = nullptr, State* state = nullptr, std::span<const float> vibratoEnvelope = {}) const;
    // The cancellation callback runs synchronously between inference stages and blocks.
    // Outputs are owned by the caller and replaced only on success.
    [[nodiscard]] juce::Result render(std::span<const TimedPhoneme> phonemes, std::span<const float> logF0, NeuralVocoderOutput& output, std::uint32_t seed = 5489, const std::function<bool()>& shouldCancel = {}, SynthesisStatistics* statistics = nullptr, State* state = nullptr, std::span<const float> vocalModeWeights = {}) const;

private:
    PhonemeTiming timing;
    PitchModel pitch;
    AcousticModel acoustic;
    NeuralVocoder vocoder;
    std::vector<std::string> rapLanguages;
    std::vector<std::string> vocalModeNames;
    float frameIntervalSeconds = 0.0f;
};
} // namespace sv::synthesis
