#pragma once

#include "VoiceDatabase.h"

#include <juce_core/juce_core.h>

#include <cstdint>
#include <string>
#include <vector>

namespace sv::synthesis
{
/**
 * @brief Data types supported in the binary voice configuration block.
 */
enum class VoiceConfigurationType : std::uint32_t
{
    strings = 0,   ///< Array of byte/string blocks
    numbers = 1,   ///< Array of 64-bit IEEE-754 double floats
    integers = 2   ///< Array of 32-bit unsigned integers
};

/**
 * @brief An individual configuration property entry within a voice configuration block.
 */
struct VoiceConfigurationEntry
{
    VoiceConfigurationType type = VoiceConfigurationType::strings;
    std::uint32_t ordinal = 0;              ///< Unique sequence number / property index.
    juce::MemoryBlock key;                  ///< Raw obfuscated key bytes from file.
    std::string name;                       ///< De-obfuscated UTF-8 property name (decoded via LCG XOR cipher).
    std::vector<juce::MemoryBlock> strings; ///< Value payloads if type == strings.
    std::vector<double> numbers;            ///< Value payloads if type == numbers.
    std::vector<std::uint32_t> integers;    ///< Value payloads if type == integers.
};

/**
 * @brief Descriptor locating a neural network model within the NOFS voice package.
 */
struct ModelReference
{
    juce::MemoryBlock key;      ///< NOFS entry key for the DNNI neural network file.
    std::string architecture;   ///< Model architecture identifier (e.g. "ds-dur-v1", "sv-aco-v2").
    std::string name;           ///< Descriptive model name.
};

/**
 * @brief Parser and container for the voice database configuration table.
 *
 * In Synthesizer V voice databases, the configuration table defines:
 * - Pointers to neural network models (duration/timing, acoustic, neural vocoder).
 * - Supported language mappings and pitch ranges.
 * - Vocal mode timbre definitions.
 *
 * The configuration block is stored in an obfuscated binary format (magic 0xFEFF)
 * inside the NOFS database container.
 */
class VoiceConfiguration
{
public:
    /**
     * @brief Parses and validates voice configuration from the provided VoiceDatabase.
     *
     * Decodes all entries and resolves model references for duration, acoustic, and vocoder.
     * The resulting VoiceConfiguration owns all its parsed data.
     *
     * @param database The open VoiceDatabase archive.
     * @return juce::Result::ok() or failure description.
     */
    [[nodiscard]] juce::Result load(VoiceDatabase& database);

    /**
     * @brief Returns all parsed configuration entries.
     */
    [[nodiscard]] const std::vector<VoiceConfigurationEntry>& getEntries() const noexcept;

    /**
     * @brief Returns the reference to the phoneme timing / duration prediction neural model.
     */
    [[nodiscard]] const ModelReference& getDurationModel() const noexcept;

    /**
     * @brief Returns the reference to the acoustic feature prediction neural model.
     */
    [[nodiscard]] const ModelReference& getAcousticModel() const noexcept;

    /**
     * @brief Returns the reference to the neural vocoder model.
     */
    [[nodiscard]] const ModelReference& getVocoderModel() const noexcept;

private:
    std::vector<VoiceConfigurationEntry> entries;
    ModelReference duration;
    ModelReference acoustic;
    ModelReference vocoder;
};

} // namespace sv::synthesis
