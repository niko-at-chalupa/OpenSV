#include "PhoneSet.h"

#include <algorithm>
#include <cstdint>
#include <set>
#include <utility>

namespace sv::synthesis
{
namespace
{
/**
 * @brief Reads a 32-bit unsigned little-endian integer from a byte buffer.
 */
std::uint32_t readUint32(std::span<const std::uint8_t> bytes, std::size_t offset)
{
    return static_cast<std::uint32_t>(bytes[offset])
         | (static_cast<std::uint32_t>(bytes[offset + 1]) << 8)
         | (static_cast<std::uint32_t>(bytes[offset + 2]) << 16)
         | (static_cast<std::uint32_t>(bytes[offset + 3]) << 24);
}

/**
 * @brief Reads a length-prefixed UTF-8 string from binary payload.
 * Format: 4-byte little-endian length, followed by raw UTF-8 bytes (without NUL terminator).
 */
bool readString(std::span<const std::uint8_t> bytes, std::size_t& position, std::string& value)
{
    if (bytes.size() - position < 4)
    {
        return false;
    }
    const std::size_t length = readUint32(bytes, position);
    position += 4;
    // Bound check against available bytes and sanity threshold of 4096 bytes
    if (length > bytes.size() - position || length > 4096)
    {
        return false;
    }
    const auto* text = reinterpret_cast<const char*>(bytes.data() + position);
    // Must be valid UTF-8 and contain no embedded NUL characters
    if (!juce::CharPointer_UTF8::isValidString(text, static_cast<int>(length)) || std::find(text, text + length, '\0') != text + length)
    {
        return false;
    }
    value.assign(text, length);
    position += length;
    return true;
}

/**
 * @brief Reads an array of length-prefixed strings.
 * Format: 4-byte count, followed by `count` string records.
 */
bool readStringList(std::span<const std::uint8_t> bytes, std::size_t& position, std::vector<std::string>& strings)
{
    if (bytes.size() - position < 4)
    {
        return false;
    }
    const std::size_t count = readUint32(bytes, position);
    position += 4;
    if (count > 4096 || count > (bytes.size() - position) / 4)
    {
        return false;
    }
    strings.resize(count);
    for (auto& string : strings)
    {
        if (!readString(bytes, position, string))
        {
            return false;
        }
    }
    return true;
}

} // namespace

juce::Result readPhoneSet(const DnniReader& reader, std::size_t nodeIndex, PhoneSet& output)
{
    const auto& nodes = reader.getNodes();
    // Invariant: The node must be of type '_psv2' and have no children (leaf node)
    if (nodeIndex >= nodes.size() || nodes[nodeIndex].type != "_psv2" || !nodes[nodeIndex].children.empty())
    {
        return juce::Result::fail("Phoneme tables require a leaf _psv2 node.");
    }

    PhoneSet phoneSet;
    const auto bytes = reader.getPayload(nodeIndex);
    std::size_t position = 0;

    // Decode sequentially: name, symbols, categories, unifiedSymbols, classes
    if (!readString(bytes, position, phoneSet.name) ||
        !readStringList(bytes, position, phoneSet.symbols) ||
        !readStringList(bytes, position, phoneSet.categories) ||
        !readStringList(bytes, position, phoneSet.unifiedSymbols) ||
        !readStringList(bytes, position, phoneSet.classes) ||
        position != bytes.size())
    {
        return juce::Result::fail("Invalid _psv2 string table.");
    }

    // Invariant check: symbols and categories must match in length
    // The stored multiphone mapping has one extra empty entry. Only symbol-indexed entries are used.
    if (phoneSet.name.empty() || phoneSet.symbols.empty() ||
        phoneSet.categories.size() != phoneSet.symbols.size() ||
        phoneSet.unifiedSymbols.size() < phoneSet.symbols.size())
    {
        return juce::Result::fail("Phoneme _psv2 table has inconsistent symbol metadata.");
    }

    // Validate uniqueness of phonetic class categories
    std::set<std::string> symbols;
    std::set<std::string> classes;
    for (const auto& category : phoneSet.classes)
    {
        if (category.empty() || !classes.insert(category).second)
        {
            return juce::Result::fail("Phoneme _psv2 class names must be nonempty and unique.");
        }
    }

    // Validate that every symbol has a recognized category and non-empty mapping
    for (std::size_t index = 0; index < phoneSet.symbols.size(); ++index)
    {
        if (phoneSet.symbols[index].empty() ||
            !symbols.insert(phoneSet.symbols[index]).second ||
            !classes.contains(phoneSet.categories[index]) ||
            phoneSet.unifiedSymbols[index].empty())
        {
            return juce::Result::fail("Phoneme _psv2 contains an invalid symbol, category, or mapping.");
        }
    }

    output = std::move(phoneSet);
    return juce::Result::ok();
}

} // namespace sv::synthesis
