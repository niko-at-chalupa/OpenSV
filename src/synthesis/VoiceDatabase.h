#pragma once

#include <juce_core/juce_core.h>

#include <cstdint>
#include <memory>
#include <vector>

namespace sv::synthesis
{
/**
 * @brief Represents an individual file or resource entry stored within a NOFS database.
 */
struct VoiceEntry
{
    juce::MemoryBlock key;          ///< Raw key identifier (can be printable text or arbitrary binary).
    juce::String name;              ///< De-obfuscated or ASCII representation if valid UTF-8, else empty.
    std::uint64_t valueOffset = 0;  ///< Absolute byte offset in the NOFS archive file.
    std::uint32_t valueSize = 0;    ///< Size in bytes of the resource payload.
};

/**
 * @brief High-level voice database metadata extracted from NOFS archive headers.
 */
struct VoiceMetadata
{
    juce::String name;                  ///< Singer / voice name (e.g. "Kasane Teto", "Eleanor Forte").
    juce::String vendor;                ///< Voice developer / vendor (e.g. "Dreamtonics", "Volor").
    int version = -1;                   ///< Database format version.
    juce::String language;              ///< Primary native language code (e.g. "japanese", "english").
    juce::String phoneset;              ///< Phoneme dictionary / phoneset tag (e.g. "romaji", "arpabet").
    juce::String type;                  ///< Voice type (e.g. "ai", "standard").
    juce::StringArray languages;        ///< List of supported cross-lingual languages.
    juce::StringArray timbreStyles;     ///< List of supported vocal mode timbre styles (e.g. "Power", "Soft").
    juce::StringPairArray properties{false}; ///< Additional key-value properties from the metadata table.
};

/**
 * @brief Reader for Synthesizer V's Native Object File System (NOFS) archive format.
 *
 * NOFS (`.nofs` or `.svp` packages) is a specialized binary archive container format:
 * - Magic bytes: 0xf580 (little endian 32-bit integer)
 * - Supported format version: 10
 * - Encapsulates neural network weights (DNNI), dictionaries, phoneme tables,
 *   samples, audio impulse responses, and vocal mode parameters.
 *
 * Thread safety & performance:
 * - Instances must be accessed only from a single synthesis worker thread or message thread.
 * - Never call open() or readEntry() from the real-time audio callback.
 */
class VoiceDatabase
{
public:
    /**
     * @brief Opens a NOFS archive from disk, validating headers and parsing entry index tables.
     */
    [[nodiscard]] juce::Result open(const juce::File& file);

    /**
     * @brief Returns the table of all entries present in the database.
     */
    [[nodiscard]] const std::vector<VoiceEntry>& getEntries() const noexcept;

    /**
     * @brief Returns high-level singer and voice metadata.
     */
    [[nodiscard]] const VoiceMetadata& getMetadata() const noexcept;

    /**
     * @brief Returns the underlying file path.
     */
    [[nodiscard]] const juce::File& getFile() const noexcept;

    /**
     * @brief Finds an entry by name. Returns nullptr if not found.
     */
    [[nodiscard]] const VoiceEntry* findEntry(juce::StringRef name) const;

    /**
     * @brief Reads the binary payload of an entry into destination memory.
     */
    [[nodiscard]] juce::Result readEntry(const VoiceEntry& entry, juce::MemoryBlock& destination);

private:
    [[nodiscard]] juce::Result readMetadata();

    juce::File file;
    std::unique_ptr<juce::FileInputStream> stream;
    std::vector<VoiceEntry> entries;
    VoiceMetadata metadata;
};

} // namespace sv::synthesis
