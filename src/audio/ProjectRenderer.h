#pragma once

#include "RenderStatistics.h"
#include "RenderVisualization.h"
#include "core/Project.h"
#include "synthesis/PhonemeDictionary.h"
#include "synthesis/VoiceSynthesizer.h"

#include <juce_audio_basics/juce_audio_basics.h>
#include <juce_core/juce_core.h>

#include <functional>
#include <list>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace sv::audio
{
// Own on a single rendering worker. All loading, inference, mixing, and cache
// eviction allocate; neither render nor destruction belongs on the audio thread.
class ProjectRenderer
{
public:
    // The caller owns the project snapshot and output throughout this call.
    // Only successful output may be published; cancellation can leave partial audio.
    // Optional visualization is built on this worker and follows the same success rule.
    [[nodiscard]] juce::Result render(const Project& project, double sampleRate, juce::AudioBuffer<float>& output, const std::function<bool()>& shouldCancel, RenderVisualization* visualization = nullptr);
    [[nodiscard]] const RenderStatistics& getStatistics() const noexcept;

private:
    struct FileStamp
    {
        juce::File file;
        juce::int64 size = 0;
        juce::int64 modifiedMilliseconds = 0;

        bool operator==(const FileStamp&) const = default;
    };

    struct CachedVoice
    {
        FileStamp source;
        std::unique_ptr<synthesis::VoiceSynthesizer> synthesizer;
    };

    struct CachedDictionary
    {
        FileStamp phonesSource;
        FileStamp dictionarySource;
        std::optional<FileStamp> readingsSource;
        std::optional<FileStamp> hiraganaSource;
        std::optional<FileStamp> katakanaSource;
        std::optional<FileStamp> smallKanaSource;
        synthesis::PhonemeDictionary dictionary;
    };

    struct CachedPhrase
    {
        FileStamp voiceSource;
        std::vector<synthesis::TimingSyllable> syllables;
        // Acoustic timing is independent of the pitch prediction context.
        std::vector<synthesis::TimedPhoneme> phonemes;
        // Keep complete F0 predictions with the audio, independently of the
        // larger network snapshots and loaded-model LRU lifetimes.
        std::vector<synthesis::PitchNote> pitchNotes;
        std::vector<float> pitchEnvelope;
        // Frame-major voice timbre-mode weights used by acoustic inference.
        std::vector<float> vocalModeWeights;
        std::vector<float> automaticPitch;
        float pitchFrameIntervalSeconds = 0.0f;
        std::vector<float> logF0;
        float frameIntervalSeconds = 0.0f;
        double sampleRate = 0.0;
        std::vector<float> samples;
        std::shared_ptr<const PhraseVisualization> visualization;
        std::size_t bytes = 0;
    };

    struct CachedInference
    {
        FileStamp voiceSource;
        std::string trackGroupId;
        Blick phraseStart = 0;
        synthesis::VoiceSynthesizer::State state;
        std::size_t bytes = 0;
    };

    [[nodiscard]] static juce::Result readFileStamp(const juce::File& file, FileStamp& stamp);
    [[nodiscard]] juce::Result findVoice(const FileStamp& source, synthesis::VoiceSynthesizer*& synthesizer);
    [[nodiscard]] juce::Result resolvePhonemes(const VoiceSettings& settings, const Note& note, std::vector<std::string>& phonemes, std::string& continuationPhoneme, std::string& phonemeLanguage, bool preferEnglishContext);
    [[nodiscard]] static bool matchesTiming(const CachedPhrase& cached, const FileStamp& voiceSource, const std::vector<synthesis::TimingSyllable>& syllables);
    [[nodiscard]] static std::size_t phraseBytes(const CachedPhrase& phrase);
    void retainPhrase(CachedPhrase phrase);
    [[nodiscard]] CachedInference takeInference(const FileStamp& voiceSource, const std::string& trackGroupId, Blick phraseStart);
    void retainInference(CachedInference inference);

    std::vector<CachedVoice> voices;
    std::vector<CachedDictionary> dictionaries;
    // Native-rate mono audio is independent of placement, output rate and mixer.
    // All entries and LRU reclamation belong exclusively to the render worker.
    std::list<CachedPhrase> phrases;
    std::size_t phraseCacheBytes = 0;
    // Lineage only selects a candidate. Each network checks its exact inputs
    // and model identity before reusing any cached output.
    std::list<CachedInference> inferenceStates;
    std::size_t inferenceCacheBytes = 0;
    RenderStatistics statistics;
};
} // namespace sv::audio
