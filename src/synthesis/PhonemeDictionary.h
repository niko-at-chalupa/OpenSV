#pragma once

#include <juce_core/juce_core.h>

#include <cstddef>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace sv::synthesis
{
struct PhonemeDefinition
{
    std::string symbol;
    std::string category;
};

// File order is retained for inspection; it is not a model's phoneme-index mapping.
// Loading and queries allocate memory and must run outside the audio callback.
// Complete load before concurrent read-only queries; reloading requires exclusive access.
class PhonemeDictionary final
{
public:
    // Failure leaves the previously loaded dictionary intact.
    [[nodiscard]] juce::Result load(const juce::File& phonesFile, const juce::File& dictionaryFile);
    [[nodiscard]] juce::Result loadMandarin(const juce::File& phonesFile, const juce::File& dictionaryFile, const juce::File& cedictFile);

    // Generic dictionaries try exact and ASCII case-insensitive keys. Uppercase English single
    // letters are spelled out by name. Mandarin also accepts normalized pinyin and the first
    // supported single-syllable CEDICT reading; it does not segment lyrics.
    // Both query methods leave output intact on failure and reject empty input.
    [[nodiscard]] juce::Result lookup(std::string_view lyrics, std::vector<std::string>& output) const;
    [[nodiscard]] juce::Result parseExplicitPhonemes(std::string_view text, std::vector<std::string>& output) const;

    [[nodiscard]] bool isLoaded() const;
    [[nodiscard]] const std::vector<PhonemeDefinition>& getPhonemes() const;
    [[nodiscard]] std::size_t getEntryCount() const;

private:
    std::vector<PhonemeDefinition> phonemes;
    std::unordered_set<std::string> symbols;
    std::unordered_map<std::string, std::vector<std::string>> entries;
    bool isMandarin = false;
    bool isEnglishArpabet = false;
};
} // namespace sv::synthesis
