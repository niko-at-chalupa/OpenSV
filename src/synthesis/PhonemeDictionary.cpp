#include "PhonemeDictionary.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <limits>
#include <optional>
#include <utility>

namespace sv::synthesis
{
namespace
{
/**
 * @brief Decodes a UTF-8 string view into a juce::String, checking for invalid sequences,
 * embedded NULs, and non-printable control characters.
 */
juce::Result decodeUtf8(std::string_view bytes, juce::String& text)
{
    if (bytes.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    {
        return juce::Result::fail("Text exceeds the supported UTF-8 input size.");
    }
    if (bytes.find('\0') != std::string_view::npos)
    {
        return juce::Result::fail("Text contains an embedded NUL.");
    }
    if (bytes.empty())
    {
        text.clear();
        return juce::Result::ok();
    }
    if (!juce::CharPointer_UTF8::isValidString(bytes.data(), static_cast<int>(bytes.size())))
    {
        return juce::Result::fail("Text is not valid UTF-8.");
    }
    for (const auto byte : bytes)
    {
        const auto code = static_cast<unsigned char>(byte);
        if ((code < 0x20 && byte != '\t' && byte != '\r' && byte != '\n') || code == 0x7f)
        {
            return juce::Result::fail("Text contains an unsupported control character.");
        }
    }
    text = juce::String::fromUTF8(bytes.data(), static_cast<int>(bytes.size()));
    return juce::Result::ok();
}

/**
 * @brief Reads a text file, stripping optional UTF-8 Byte Order Mark (BOM).
 */
juce::Result readText(const juce::File& file, juce::String& text)
{
    if (file.getSize() > std::numeric_limits<int>::max())
    {
        return juce::Result::fail("Dictionary file is too large: " + file.getFullPathName());
    }
    juce::MemoryBlock contents;
    if (!file.loadFileAsData(contents))
    {
        return juce::Result::fail("Could not read dictionary file: " + file.getFullPathName());
    }
    if (contents.getSize() == 0)
    {
        return juce::Result::fail("Dictionary file is empty: " + file.getFullPathName());
    }
    std::string_view bytes(static_cast<const char*>(contents.getData()), contents.getSize());
    // Strip UTF-8 BOM if present (\xEF\xBB\xBF)
    if (bytes.starts_with("\xef\xbb\xbf"))
    {
        bytes.remove_prefix(3);
    }
    const auto result = decodeUtf8(bytes, text);
    if (result.failed())
    {
        return juce::Result::fail(file.getFullPathName() + ": " + result.getErrorMessage());
    }
    return juce::Result::ok();
}

/**
 * @brief Splits a whitespace-separated line into tokens.
 */
juce::StringArray splitFields(const juce::String& line)
{
    auto fields = juce::StringArray::fromTokens(line, " \t", "");
    fields.removeEmptyStrings();
    return fields;
}

/**
 * @brief Normalizes English dictionary keys (lowercasing, and stripping CMU variant suffixes like "word(2)").
 */
std::string normalizeEnglishKey(std::string key)
{
    std::transform(key.begin(), key.end(), key.begin(), [](unsigned char character)
                   { return static_cast<char>(std::tolower(character)); });
    // Remove CMUDict pronunciation variation markers like "(1)", "(2)"
    if (const auto variant = key.rfind('('); variant != std::string::npos && key.back() == ')'
        && variant + 2 < key.size()
        && std::all_of(key.begin() + static_cast<std::ptrdiff_t>(variant + 1), key.end() - 1, [](unsigned char character)
                       { return std::isdigit(character) != 0; }))
    {
        key.erase(variant);
    }
    return key;
}

/**
 * @brief Normalizes CMU ARPABET phoneme symbols:
 * 1. Strips lexical stress digit suffixes (0 = unstressed, 1 = primary, 2 = secondary).
 * 2. Maps historical or dialectal CMU phone variants to standard SV ARPABET symbols.
 */
void normalizeEnglishPhone(std::string& symbol)
{
    // Strip stress numbers: AA0/AA1/AA2 -> AA
    if (!symbol.empty() && symbol.back() >= '0' && symbol.back() <= '2')
    {
        symbol.pop_back();
    }
    static const std::unordered_map<std::string, std::string> cmuAliases{
        {"ax", "ah"}, {"ax-h", "ah"}, {"axr", "er"}, {"el", "l"}, {"em", "m"}, {"en", "n"},
        {"eng", "ng"}, {"hv", "hh"}, {"ix", "ih"}, {"nx", "n"}, {"ux", "uw"}, {"wh", "w"}};
    if (const auto alias = cmuAliases.find(symbol); alias != cmuAliases.end())
    {
        symbol = alias->second;
    }
}

juce::Result lineError(const juce::File& file, int line, const juce::String& message)
{
    return juce::Result::fail(file.getFullPathName() + ":" + juce::String(line) + ": " + message);
}

/**
 * @brief Normalizes Mandarin Pinyin:
 * - Lowercases.
 * - Replaces 'u:' and 'ü' with 'v'.
 * - Strips optional trailing tone numbers (1 to 5).
 */
juce::String normalizePinyin(const juce::String& text)
{
    auto normalized = text.toLowerCase().replace("u:", "v").replace(juce::String::charToString(0x00fc), "v");
    if (normalized.isNotEmpty() && normalized.getLastCharacter() >= '1' && normalized.getLastCharacter() <= '5')
    {
        normalized = normalized.dropLastCharacters(1);
    }
    for (const auto character : normalized)
    {
        if (character < 'a' || character > 'z')
        {
            return {};
        }
    }
    return normalized;
}

/**
 * @brief Parses a line from CC-CEDICT Chinese-English dictionary:
 * Format: Traditional Simplified [pin1 yin1] /definition 1/definition 2/
 */
juce::Result parseCedictEntry(const juce::String& line, juce::StringArray& headwords, juce::StringArray& reading)
{
    const auto openingBracket = line.indexOfChar('[');
    const auto closingBracket = line.indexOfChar(']');
    if (openingBracket <= 0 || closingBracket <= openingBracket || !juce::CharacterFunctions::isWhitespace(line[openingBracket - 1]))
    {
        return juce::Result::fail("Expected a CEDICT entry: traditional simplified [pinyin].");
    }
    headwords = splitFields(line.substring(0, openingBracket));
    if (headwords.size() != 2 || headwords[0].containsAnyOf("[]") || headwords[1].containsAnyOf("[]"))
    {
        return juce::Result::fail("Expected exactly two CEDICT headwords before the pinyin reading.");
    }
    const auto pinyin = line.substring(openingBracket + 1, closingBracket);
    if (pinyin.containsAnyOf("[]"))
    {
        return juce::Result::fail("CEDICT pinyin reading contains an unexpected bracket.");
    }
    reading = splitFields(pinyin);
    if (reading.isEmpty())
    {
        return juce::Result::fail("CEDICT pinyin reading is empty.");
    }
    const auto suffix = line.substring(closingBracket + 1);
    const auto definitions = suffix.trim();
    if (definitions.isNotEmpty() && (!juce::CharacterFunctions::isWhitespace(suffix[0]) || !definitions.startsWithChar('/') || !definitions.endsWithChar('/')))
    {
        return juce::Result::fail("Unexpected text after the CEDICT pinyin reading; definitions must be slash-delimited.");
    }
    return juce::Result::ok();
}
} // namespace

juce::Result PhonemeDictionary::load(const juce::File& phonesFile, const juce::File& dictionaryFile)
{
    juce::String phonesText;
    if (const auto result = readText(phonesFile, phonesText); result.failed())
    {
        return result;
    }
    juce::String dictionaryText;
    if (const auto result = readText(dictionaryFile, dictionaryText); result.failed())
    {
        return result;
    }

    PhonemeDictionary loaded;
    loaded.isEnglishArpabet = phonesFile.getFileName() == "english-arpabet-phones.txt";
    const bool isCmuDictionary = loaded.isEnglishArpabet && dictionaryFile.getFileName() == "cmudict-07b.txt";

    // 1. Parse phoneme list (phoneme, category)
    const auto phoneLines = juce::StringArray::fromLines(phonesText);
    for (int line = 0; line < phoneLines.size(); ++line)
    {
        const auto fields = splitFields(phoneLines[line]);
        if (fields.isEmpty())
        {
            continue;
        }
        if (fields.size() != 2)
        {
            return lineError(phonesFile, line + 1, "Expected exactly two fields: phoneme category.");
        }
        const auto symbol = fields[0].toStdString();
        if (!loaded.symbols.insert(symbol).second)
        {
            return lineError(phonesFile, line + 1, "Duplicate phoneme '" + fields[0] + "'.");
        }
        loaded.phonemes.push_back({symbol, fields[1].toStdString()});
    }
    if (loaded.phonemes.empty())
    {
        return juce::Result::fail("Phoneme inventory contains no definitions: " + phonesFile.getFullPathName());
    }

    // 2. Parse dictionary (key, phoneme1, phoneme2, ...)
    const auto dictionaryLines = juce::StringArray::fromLines(dictionaryText);
    for (int line = 0; line < dictionaryLines.size(); ++line)
    {
        // Skip CMU comments (lines starting with ';;;')
        if (isCmuDictionary && dictionaryLines[line].trimStart().startsWith(";;;"))
        {
            continue;
        }
        const auto fields = splitFields(dictionaryLines[line]);
        if (fields.isEmpty())
        {
            continue;
        }
        if (fields.size() < 2)
        {
            return lineError(dictionaryFile, line + 1, "Expected a dictionary key followed by one or more phonemes.");
        }
        const auto sourceKey = fields[0].toStdString();
        const auto key = loaded.isEnglishArpabet ? normalizeEnglishKey(sourceKey) : sourceKey;
        if (loaded.entries.contains(key))
        {
            // For CMU dict, retain first pronunciation variant
            if (isCmuDictionary || sourceKey != key)
            {
                continue;
            }
            return lineError(dictionaryFile, line + 1, "Duplicate dictionary key '" + fields[0] + "'.");
        }
        std::vector<std::string> pronunciation;
        pronunciation.reserve(static_cast<std::size_t>(fields.size() - 1));
        bool supportedPronunciation = true;
        for (int field = 1; field < fields.size(); ++field)
        {
            auto symbol = fields[field].toStdString();
            if (isCmuDictionary)
            {
                normalizeEnglishPhone(symbol);
            }
            if (!loaded.symbols.contains(symbol))
            {
                if (isCmuDictionary)
                {
                    supportedPronunciation = false;
                    break;
                }
                return lineError(dictionaryFile, line + 1, "Undefined phoneme '" + fields[field] + "' in entry '" + fields[0] + "'.");
            }
            pronunciation.push_back(symbol);
        }
        if (!supportedPronunciation)
        {
            continue;
        }
        loaded.entries.emplace(key, std::move(pronunciation));
    }
    if (loaded.entries.empty())
    {
        return juce::Result::fail("Pronunciation dictionary contains no entries: " + dictionaryFile.getFullPathName());
    }
    *this = std::move(loaded);
    return juce::Result::ok();
}

juce::Result PhonemeDictionary::loadJapanese(const juce::File& phonesFile, const juce::File& dictionaryFile, const juce::File& hiraganaFile, const juce::File& katakanaFile, const juce::File& smallKanaFile)
{
    PhonemeDictionary loaded;
    if (const auto result = loaded.load(phonesFile, dictionaryFile); result.failed())
    {
        return result;
    }

    const auto loadKanaMappings = [](const juce::File& mappingFile, std::unordered_map<std::string, std::string>& mappings) -> juce::Result
    {
        juce::String mappingText;
        if (const auto result = readText(mappingFile, mappingText); result.failed())
        {
            return result;
        }
        const auto lines = juce::StringArray::fromLines(mappingText);
        for (int line = 0; line < lines.size(); ++line)
        {
            const auto fields = splitFields(lines[line]);
            if (fields.isEmpty())
            {
                continue;
            }
            if (fields.size() != 2)
            {
                return lineError(mappingFile, line + 1, "Expected exactly two fields: kana and romaji.");
            }
            const auto kana = fields[0].toStdString();
            const auto romaji = fields[1].toStdString();
            const auto [found, inserted] = mappings.emplace(kana, romaji);
            if (!inserted && found->second != romaji)
            {
                return lineError(mappingFile, line + 1, "Conflicting romaji for kana '" + fields[0] + "'.");
            }
        }
        return juce::Result::ok();
    };

    // Load standard Hiragana and Katakana tables
    for (const auto& mappingFile : {hiraganaFile, katakanaFile})
    {
        if (const auto result = loadKanaMappings(mappingFile, loaded.kanaToRomaji); result.failed())
        {
            return result;
        }
    }
    // Load small kana tables (ゃ, ゅ, ょ, etc.)
    if (const auto result = loadKanaMappings(smallKanaFile, loaded.smallKanaToRomaji); result.failed())
    {
        return result;
    }
    if (loaded.kanaToRomaji.empty() || loaded.smallKanaToRomaji.empty())
    {
        return juce::Result::fail("Japanese kana conversion dictionaries contain no entries.");
    }
    loaded.isJapanese = true;
    *this = std::move(loaded);
    return juce::Result::ok();
}

juce::Result PhonemeDictionary::loadMandarin(const juce::File& phonesFile, const juce::File& dictionaryFile, const juce::File& cedictFile)
{
    PhonemeDictionary loaded;
    if (const auto result = loaded.load(phonesFile, dictionaryFile); result.failed())
    {
        return result;
    }

    // Normalize pinyin keys in dictionary (stripping tones)
    std::unordered_map<std::string, std::vector<std::string>> normalizedEntries;
    normalizedEntries.reserve(loaded.entries.size());
    for (auto& [key, pronunciation] : loaded.entries)
    {
        const auto normalized = normalizePinyin(juce::String::fromUTF8(key.c_str()));
        const auto normalizedKey = normalized.isEmpty() ? key : normalized.toStdString();
        const auto found = normalizedEntries.find(normalizedKey);
        if (found != normalizedEntries.end())
        {
            if (found->second != pronunciation)
            {
                return juce::Result::fail(dictionaryFile.getFullPathName() + ": Conflicting pronunciations for normalized pinyin '" + juce::String::fromUTF8(normalizedKey.c_str()) + "'.");
            }
            continue;
        }
        normalizedEntries.emplace(normalizedKey, std::move(pronunciation));
    }
    loaded.entries = normalizedEntries;

    // Parse CC-CEDICT for Hanzi -> Pinyin mappings
    juce::String cedictText;
    if (const auto result = readText(cedictFile, cedictText); result.failed())
    {
        return result;
    }
    const auto lines = juce::StringArray::fromLines(cedictText);
    std::size_t supportedReadings = 0;
    for (int line = 0; line < lines.size(); ++line)
    {
        const auto content = lines[line].trim();
        if (content.isEmpty() || content.startsWithChar('#'))
        {
            continue;
        }
        juce::StringArray headwords;
        juce::StringArray reading;
        if (const auto result = parseCedictEntry(content, headwords, reading); result.failed())
        {
            return lineError(cedictFile, line + 1, result.getErrorMessage());
        }
        // Only accept single-syllable Chinese character mappings
        if (reading.size() != 1)
        {
            continue;
        }
        const auto pinyin = normalizePinyin(reading[0]);
        if (pinyin.isEmpty())
        {
            continue;
        }
        const auto found = normalizedEntries.find(pinyin.toStdString());
        if (found == normalizedEntries.end())
        {
            continue;
        }
        // Map both Traditional and Simplified characters to this pronunciation
        for (const auto& headword : headwords)
        {
            loaded.entries.try_emplace(headword.toStdString(), found->second);
        }
        ++supportedReadings;
    }
    if (supportedReadings == 0)
    {
        return juce::Result::fail("CEDICT contains no supported single-syllable readings: " + cedictFile.getFullPathName());
    }
    loaded.isMandarin = true;
    *this = std::move(loaded);
    return juce::Result::ok();
}

juce::Result PhonemeDictionary::lookup(std::string_view lyrics, std::vector<std::string>& output) const
{
    if (!isLoaded())
    {
        return juce::Result::fail("Phoneme dictionary is not loaded.");
    }
    juce::String text;
    if (const auto result = decodeUtf8(lyrics, text); result.failed())
    {
        return result;
    }
    if (text.isEmpty())
    {
        return juce::Result::fail("Dictionary lookup key is empty.");
    }
    if (text.containsAnyOf(" \t\r\n"))
    {
        return juce::Result::fail("Dictionary lookup requires exactly one key without surrounding whitespace.");
    }

    // Special case: Single English capital letter spelled out (A -> "EY", B -> "B IY", etc.)
    static constexpr std::array<std::string_view, 26> englishLetterNames{
        "ey", "b iy", "s iy", "d iy", "iy", "eh f", "jh iy", "ey ch", "ay", "jh ey", "k ey", "eh l", "eh m",
        "eh n", "ow", "p iy", "k y uw", "aa r", "eh s", "t iy", "y uw", "v iy", "d ah b ah l y uw", "eh k s", "w ay", "z iy"};
    if (isEnglishArpabet && lyrics.size() == 1 && lyrics.front() >= 'A' && lyrics.front() <= 'Z')
    {
        const auto spelling = englishLetterNames[static_cast<std::size_t>(lyrics.front() - 'A')];
        const auto fields = juce::StringArray::fromTokens(juce::String::fromUTF8(spelling.data(), static_cast<int>(spelling.size())), " ", "");
        std::vector<std::string> pronunciation;
        pronunciation.reserve(static_cast<std::size_t>(fields.size()));
        for (const auto& field : fields)
        {
            const auto symbol = field.toStdString();
            if (!symbols.contains(symbol))
            {
                return juce::Result::fail("The phoneme inventory cannot spell the English letter '" + text + "'.");
            }
            pronunciation.push_back(symbol);
        }
        output = std::move(pronunciation);
        return juce::Result::ok();
    }

    std::string key(lyrics);

    // Japanese Kana to Romaji conversion (with small kana contraction)
    if (isJapanese)
    {
        std::string romanized;
        bool allKana = true;
        for (int index = 0; index < text.length(); ++index)
        {
            const auto kana = juce::String::charToString(text[index]).toStdString();
            const auto foundKana = kanaToRomaji.find(kana);
            if (foundKana != kanaToRomaji.end())
            {
                romanized += foundKana->second;
                continue;
            }
            const auto foundSmallKana = smallKanaToRomaji.find(kana);
            if (foundSmallKana == smallKanaToRomaji.end())
            {
                allKana = false;
                break;
            }
            // Small kana digraph contraction:
            // e.g. "ki" + "ya" -> "kya"; "shi" + "ya" -> "sha"; "chi" + "ya" -> "cha"
            const auto& smallRomaji = foundSmallKana->second;
            if (smallRomaji.size() == 2 && smallRomaji.front() == 'y' && !romanized.empty() && romanized.back() == 'i')
            {
                romanized.pop_back();
                const bool digraphOnset = romanized.ends_with("sh") || romanized.ends_with("ch") || romanized.ends_with('j');
                if (!digraphOnset)
                {
                    romanized.push_back('y');
                }
                romanized.push_back(smallRomaji.back());
            }
            else
            {
                romanized += smallRomaji;
            }
        }
        if (allKana)
        {
            key = std::move(romanized);
        }
    }

    // Mandarin Pinyin tone normalization
    const auto normalized = isMandarin ? normalizePinyin(text) : juce::String{};
    if (!normalized.isEmpty())
    {
        key = normalized.toStdString();
    }

    // Direct match in dictionary table
    const auto found = entries.find(key);
    if (found != entries.end())
    {
        output = found->second;
        return juce::Result::ok();
    }
    if (isMandarin)
    {
        return juce::Result::fail("No Mandarin pronunciation for '" + text + "'. Use one dictionary-listed syllable per note: a Chinese character or pinyin (optional tone 1-5). Split multi-character lyrics across notes, or enter phonemes explicitly.");
    }

    // Case-insensitive lookup
    std::string candidate = key;
    std::transform(candidate.begin(), candidate.end(), candidate.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (const auto caseInsensitive = entries.find(candidate); caseInsensitive != entries.end())
    {
        output = caseInsensitive->second;
        return juce::Result::ok();
    }

    // Strip punctuation
    candidate.erase(std::remove_if(candidate.begin(), candidate.end(), [](unsigned char c) { return !std::isalnum(c); }), candidate.end());
    if (candidate.empty())
    {
        return juce::Result::fail("No pronunciation for dictionary key '" + text + "'.");
    }

    // English hardcoded common word fallbacks
    static const std::unordered_map<std::string, std::vector<std::string>> englishFallbacks{
        {"a", {"ae"}},
        {"i", {"ay"}},
        {"hello", {"hh", "ah", "l", "ow"}},
        {"chat", {"ch", "ae", "t"}},
        {"hate", {"hh", "ey", "t"}},
        {"now", {"n", "aw"}},
        {"hua", {"hh", "w", "ah"}},
        {"kanru", {"k", "ae", "n", "r", "uw"}},
        {"hue", {"hh", "y", "uw"}},
        {"you", {"y", "uw"}},
        {"the", {"dh", "ah"}},
        {"she", {"sh", "iy"}},
        {"we", {"w", "iy"}},
        {"me", {"m", "iy"}}
    };

    if (const auto fallback = englishFallbacks.find(candidate); fallback != englishFallbacks.end())
    {
        output = fallback->second;
        return juce::Result::ok();
    }

    // Rule-based heuristic G2P grapheme-to-phoneme synthesizer for unknown English words
    std::vector<std::string> guessed;
    for (std::size_t index = 0; index < candidate.size();)
    {
        const auto next = candidate.substr(index);
        const std::array<std::pair<std::string_view, std::string>, 11> patterns{
            std::pair{"sh", "sh"},
            std::pair{"ch", "ch"},
            std::pair{"th", "th"},
            std::pair{"ph", "f"},
            std::pair{"ng", "ng"},
            std::pair{"qu", "kw"},
            std::pair{"ee", "iy"},
            std::pair{"ea", "iy"},
            std::pair{"oo", "uw"},
            std::pair{"ow", "aw"},
            std::pair{"ai", "ey"}
        };
        std::optional<std::pair<std::string, std::string>> matched;
        for (const auto& [start, phoneme] : patterns)
        {
            if (next.rfind(start, 0) == 0)
            {
                matched.emplace(std::string(start), std::string(phoneme));
                break;
            }
        }
        if (matched.has_value())
        {
            guessed.push_back(matched->second);
            index += matched->first.size();
            continue;
        }

        const auto letter = candidate[index];
        if (letter == 'a') guessed.push_back("ae");
        else if (letter == 'e') guessed.push_back("eh");
        else if (letter == 'i') guessed.push_back("ih");
        else if (letter == 'o') guessed.push_back("aa");
        else if (letter == 'u') guessed.push_back("uh");
        else if (letter == 'y') guessed.push_back("y");
        else if (letter == 'h') guessed.push_back("hh");
        else if (letter == 't') guessed.push_back("t");
        else if (letter == 'd') guessed.push_back("d");
        else if (letter == 'k') guessed.push_back("k");
        else if (letter == 'p') guessed.push_back("p");
        else if (letter == 'b') guessed.push_back("b");
        else if (letter == 'm') guessed.push_back("m");
        else if (letter == 'n') guessed.push_back("n");
        else if (letter == 'l') guessed.push_back("l");
        else if (letter == 'r') guessed.push_back("r");
        else if (letter == 's') guessed.push_back("s");
        else if (letter == 'f') guessed.push_back("f");
        else if (letter == 'v') guessed.push_back("v");
        else if (letter == 'z') guessed.push_back("z");
        else if (letter == 'j') guessed.push_back("jh");
        else if (letter == 'g') guessed.push_back("g");
        else if (letter == 'w') guessed.push_back("w");
        else guessed.push_back("ah");
        ++index;
    }

    if (guessed.empty())
    {
        return juce::Result::fail("No pronunciation for dictionary key '" + text + "'.");
    }

    bool valid = true;
    for (const auto& symbol : guessed)
    {
        if (!symbols.contains(symbol))
        {
            valid = false;
            break;
        }
    }
    if (!valid)
    {
        return juce::Result::fail("No pronunciation for dictionary key '" + text + "'.");
    }

    output = std::move(guessed);
    return juce::Result::ok();
}

juce::Result PhonemeDictionary::parseExplicitPhonemes(std::string_view text, std::vector<std::string>& output) const
{
    if (!isLoaded())
    {
        return juce::Result::fail("Phoneme dictionary is not loaded.");
    }
    juce::String decoded;
    if (const auto result = decodeUtf8(text, decoded); result.failed())
    {
        return result;
    }
    auto fields = juce::StringArray::fromTokens(decoded, " \t\r\n", "");
    fields.removeEmptyStrings();
    if (fields.isEmpty())
    {
        return juce::Result::fail("Explicit phoneme sequence is empty.");
    }
    std::vector<std::string> parsed;
    parsed.reserve(static_cast<std::size_t>(fields.size()));
    for (const auto& field : fields)
    {
        const auto symbol = field.toStdString();
        if (!symbols.contains(symbol))
        {
            return juce::Result::fail("Unknown explicit phoneme '" + field + "'.");
        }
        parsed.push_back(symbol);
    }
    output = std::move(parsed);
    return juce::Result::ok();
}

bool PhonemeDictionary::hasEntry(std::string_view key) const
{
    if (key.empty())
    {
        return false;
    }
    std::string candidate(key);
    if (isEnglishArpabet)
    {
        candidate = normalizeEnglishKey(std::move(candidate));
    }
    else
    {
        std::transform(candidate.begin(), candidate.end(), candidate.begin(), [](unsigned char character)
                       { return static_cast<char>(std::tolower(character)); });
    }
    if (entries.contains(candidate))
    {
        return true;
    }
    candidate.erase(std::remove_if(candidate.begin(), candidate.end(), [](unsigned char character)
                                   { return !std::isalnum(character); }), candidate.end());
    return !candidate.empty() && entries.contains(candidate);
}

bool PhonemeDictionary::isLoaded() const
{
    return !phonemes.empty() && !entries.empty();
}

const std::vector<PhonemeDefinition>& PhonemeDictionary::getPhonemes() const
{
    return phonemes;
}

std::size_t PhonemeDictionary::getEntryCount() const
{
    return entries.size();
}

} // namespace sv::synthesis
