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
/**
 * @brief Definition of an individual phoneme in the active language dictionary.
 */
struct PhonemeDefinition
{
    std::string symbol;   ///< Phoneme token identifier (e.g. "k", "aa", "sil").
    std::string category; ///< Phonetic class category (e.g. "vowel", "consonant", "silence").
};

/**
 * @brief Multi-language Grapheme-to-Phoneme (G2P) pronouncing dictionary and parser.
 *
 * Supports three primary language pipelines used in Synthesizer V:
 * 1. Japanese Romaji / Kana:
 *    - Full Hiragana and Katakana transliteration to Romaji.
 *    - Small kana digraph compound contraction (e.g. き + ゃ -> kya).
 * 2. English ARPABET:
 *    - CMUDict pronunciations with stress digit stripping (AA0/AA1/AA2 -> AA).
 *    - Single-letter alphabet spelling out ("A" -> "EY", "B" -> "B IY").
 *    - CMU dialect alias normalization (e.g. "ax" -> "ah", "ix" -> "ih").
 * 3. Mandarin Pinyin / Hanzi:
 *    - Pinyin tone normalization (e.g. "ni3" -> "ni", "lü" / "lu:" -> "lv").
 *    - CEDICT Chinese character to Pinyin dictionary lookup.
 *
 * Thread safety & performance:
 * - Dictionaries are loaded once on a background worker thread.
 * - Read-only queries (`lookup`, `parseExplicitPhonemes`, `hasEntry`) are thread-safe concurrently.
 * - Never execute dictionary loading or lookups in the real-time audio thread.
 */
class PhonemeDictionary final
{
public:
    /**
     * @brief Loads a generic two-column pronouncing dictionary (word -> phonemes).
     */
    [[nodiscard]] juce::Result load(const juce::File& phonesFile, const juce::File& dictionaryFile);

    /**
     * @brief Loads full Japanese G2P tables (phones, romaji dict, hiragana, katakana, small kana).
     */
    [[nodiscard]] juce::Result loadJapanese(const juce::File& phonesFile, const juce::File& dictionaryFile, const juce::File& hiraganaFile, const juce::File& katakanaFile, const juce::File& smallKanaFile);

    /**
     * @brief Loads Mandarin G2P tables (phones, pinyin dict, CEDICT hanzi dictionary).
     */
    [[nodiscard]] juce::Result loadMandarin(const juce::File& phonesFile, const juce::File& dictionaryFile, const juce::File& cedictFile);

    /**
     * @brief Translates a lyric syllable into a list of phoneme strings.
     */
    [[nodiscard]] juce::Result lookup(std::string_view lyrics, std::vector<std::string>& output) const;

    /**
     * @brief Parses and validates an explicit space-separated phoneme override string (e.g. "k aa").
     */
    [[nodiscard]] juce::Result parseExplicitPhonemes(std::string_view text, std::vector<std::string>& output) const;

    /**
     * @brief Returns true if the dictionary has an entry for the given word/syllable.
     */
    [[nodiscard]] bool hasEntry(std::string_view key) const;

    /**
     * @brief Returns true if dictionary files are successfully loaded.
     */
    [[nodiscard]] bool isLoaded() const;

    /**
     * @brief Returns the list of all defined phonemes and their categories.
     */
    [[nodiscard]] const std::vector<PhonemeDefinition>& getPhonemes() const;

    /**
     * @brief Returns total count of vocabulary words in the pronouncing dictionary.
     */
    [[nodiscard]] std::size_t getEntryCount() const;

private:
    std::vector<PhonemeDefinition> phonemes;
    std::unordered_set<std::string> symbols;
    std::unordered_map<std::string, std::vector<std::string>> entries;
    std::unordered_map<std::string, std::string> kanaToRomaji;
    std::unordered_map<std::string, std::string> smallKanaToRomaji;
    bool isMandarin = false;
    bool isJapanese = false;
    bool isEnglishArpabet = false;
};

} // namespace sv::synthesis
