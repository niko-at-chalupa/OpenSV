#pragma once

#include "DnniReader.h"

#include <juce_core/juce_core.h>

#include <cstddef>
#include <string>
#include <vector>

namespace sv::synthesis
{
/**
 * @brief Phonetic table and inventory loaded from a DNNI `_psv2` node.
 *
 * Defines the phoneme vocabulary of a voice model or language:
 * - `symbols`: Discrete phoneme tokens (e.g. "a", "k", "s", "pau", "sil").
 * - `categories`: Phonetic category for each phoneme (e.g. vowel, stop, fricative, nasal).
 * - `unifiedSymbols`: Cross-lingual standardized phonetic representation (e.g. X-SAMPA / IPA mapped).
 * - `classes`: List of distinct phonetic classes referenced by the categories.
 */
struct PhoneSet
{
    std::string name;                           ///< Identifier of this phoneme set (e.g. "japanese-romaji").
    std::vector<std::string> symbols;           ///< List of unique phoneme symbols in index order.
    std::vector<std::string> categories;        ///< Category string assigned to each symbol (parallel to symbols).
    std::vector<std::string> unifiedSymbols;    ///< Cross-lingual unified phone mapped to each symbol.
    std::vector<std::string> classes;           ///< Set of unique valid category names.
};

/**
 * @brief Reads and validates a `_psv2` (Phone Set Version 2) node from a DNNI model.
 *
 * Binary payload layout:
 * - 32-bit uint: name length, followed by UTF-8 bytes of name.
 * - 32-bit uint: symbols count, followed by count length-prefixed strings.
 * - 32-bit uint: categories count, followed by count length-prefixed strings.
 * - 32-bit uint: unifiedSymbols count, followed by count length-prefixed strings.
 * - 32-bit uint: classes count, followed by count length-prefixed strings.
 *
 * Invariant:
 * - The destination `output` is only modified on complete validation success.
 *
 * @param reader Loaded DNNI reader holding the parsed node table and payload.
 * @param nodeIndex Index of the `_psv2` node.
 * @param output Output PhoneSet populated on success.
 * @return juce::Result::ok() or failure error message.
 */
[[nodiscard]] juce::Result readPhoneSet(const DnniReader& reader, std::size_t nodeIndex, PhoneSet& output);

} // namespace sv::synthesis
