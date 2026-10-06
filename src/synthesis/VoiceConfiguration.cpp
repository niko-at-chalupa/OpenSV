#include "VoiceConfiguration.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>
#include <new>
#include <span>
#include <string_view>
#include <unordered_set>
#include <utility>

namespace sv::synthesis
{
namespace
{
// Maximum size limit for configuration blob to guard against malformed files
constexpr std::size_t maximumConfigurationBytes = 1024 * 1024; // 1 MiB
constexpr std::uint32_t maximumEntries = 16384;

/**
 * @brief Helper stream reader for little-endian primitive types in configuration buffers.
 */
class ConfigurationReader
{
public:
    explicit ConfigurationReader(std::span<const std::uint8_t> input) : bytes(input) {}

    bool readUint16(std::uint16_t& value)
    {
        if (remaining() < sizeof(value))
        {
            return false;
        }
        value = juce::ByteOrder::littleEndianShort(bytes.data() + position);
        position += sizeof(value);
        return true;
    }

    bool readUint32(std::uint32_t& value)
    {
        if (remaining() < sizeof(value))
        {
            return false;
        }
        value = juce::ByteOrder::littleEndianInt(bytes.data() + position);
        position += sizeof(value);
        return true;
    }

    bool readDouble(double& value)
    {
        if (remaining() < sizeof(value))
        {
            return false;
        }
        value = std::bit_cast<double>(static_cast<std::uint64_t>(juce::ByteOrder::littleEndianInt64(bytes.data() + position)));
        position += sizeof(value);
        return std::isfinite(value);
    }

    bool readString(juce::MemoryBlock& value)
    {
        std::uint16_t size = 0;
        if (!readUint16(size) || size > remaining())
        {
            return false;
        }
        value.replaceAll(bytes.data() + position, size);
        position += size;
        return true;
    }

    [[nodiscard]] std::size_t remaining() const noexcept
    {
        return bytes.size() - position;
    }

    [[nodiscard]] juce::Result error(const juce::String& detail) const
    {
        return juce::Result::fail("Voice configuration at 0x" + juce::String::toHexString(static_cast<juce::int64>(position)) + ": " + detail);
    }

private:
    std::span<const std::uint8_t> bytes;
    std::size_t position = 0;
};

/**
 * @brief Checks if a string view is valid printable UTF-8 text.
 */
bool isText(std::string_view value)
{
    return value.size() <= static_cast<std::size_t>(std::numeric_limits<int>::max()) &&
           std::none_of(value.begin(), value.end(), [](char character)
                        { return static_cast<unsigned char>(character) < 0x20 || character == 0x7f; }) &&
           juce::CharPointer_UTF8::isValidString(value.data(), static_cast<int>(value.size()));
}

/**
 * @brief De-obfuscates an entry name key using a Linear Congruential Generator (LCG) XOR cipher.
 *
 * Reverse-engineering details:
 * In Synthesizer V binaries (e.g. Mach-O symbol 0x100023E90), entry keys in the
 * voice configuration block are lightly obfuscated using a pseudo-random sequence.
 * The loop at 0x100023272 XORs each raw byte with the lowest 8 bits of the current LCG state.
 *
 * LCG parameters:
 * - Initial state: 0xbefcb5e0
 * - Multiplier: 0x41c64e6d (1103515245, identical to standard POSIX / MSVC rand())
 * - Additive constant: 0x3039 (12345)
 *
 * Formula per byte:
 * decoded[i] = bytes[i] ^ (state & 0xFF)
 * state = state * 0x41c64e6d + 0x3039
 */
std::string decodeName(const juce::MemoryBlock& value)
{
    const auto* bytes = static_cast<const std::uint8_t*>(value.getData());
    std::string decoded(value.getSize(), '\0');
    std::uint32_t state = 0xbefcb5e0;
    for (std::size_t index = 0; index < value.getSize(); ++index)
    {
        decoded[index] = static_cast<char>(bytes[index] ^ static_cast<std::uint8_t>(state));
        state = state * 0x41c64e6d + 0x3039;
    }
    return decoded;
}

/**
 * @brief Parses the raw binary configuration payload into VoiceConfigurationEntry records.
 *
 * Binary layout:
 * - 4 bytes: Magic signature (0x0000FEFF)
 * - 4 bytes: Entry count N
 * - Followed by N entry records:
 *   - 4 bytes: Type (0 = strings, 1 = numbers, 2 = integers)
 *   - 4 bytes: Ordinal (entry sequence ID)
 *   - 2 bytes length + string bytes: Obfuscated key name
 *   - 4 bytes: Value count M
 *   - Followed by M values corresponding to Type
 */
juce::Result parseConfiguration(const juce::MemoryBlock& bytes, std::vector<VoiceConfigurationEntry>& entries)
{
    if (bytes.getSize() < 8 || bytes.getSize() > maximumConfigurationBytes)
    {
        return juce::Result::fail("Voice configuration is truncated or exceeds the 1 MiB limit.");
    }
    ConfigurationReader reader({static_cast<const std::uint8_t*>(bytes.getData()), bytes.getSize()});
    std::uint32_t signature = 0;
    std::uint32_t count = 0;
    if (!reader.readUint32(signature) || signature != 0xfeff || !reader.readUint32(count))
    {
        return reader.error("invalid configuration header.");
    }
    if (count > maximumEntries || count > reader.remaining() / 14)
    {
        return reader.error("entry count exceeds the remaining data or resource limit.");
    }

    std::unordered_set<std::uint32_t> ordinals;
    entries.reserve(count);

    for (std::uint32_t index = 0; index < count; ++index)
    {
        VoiceConfigurationEntry entry;
        std::uint32_t type = 0;
        std::uint32_t valueCount = 0;
        if (!reader.readUint32(type) || !reader.readUint32(entry.ordinal) || !reader.readString(entry.key) || !reader.readUint32(valueCount))
        {
            return reader.error("truncated entry header.");
        }
        if (!ordinals.insert(entry.ordinal).second)
        {
            return reader.error("duplicate entry ordinal.");
        }

        entry.name = decodeName(entry.key);
        if (entry.name.empty() || !isText(entry.name))
        {
            return reader.error("entry name is not valid printable UTF-8.");
        }

        if (type == static_cast<std::uint32_t>(VoiceConfigurationType::strings))
        {
            if (valueCount > reader.remaining() / 2)
            {
                return reader.error("string count exceeds the remaining data.");
            }
            entry.type = VoiceConfigurationType::strings;
            entry.strings.resize(valueCount);
            for (auto& value : entry.strings)
            {
                if (!reader.readString(value))
                {
                    return reader.error("truncated string value.");
                }
            }
        }
        else if (type == static_cast<std::uint32_t>(VoiceConfigurationType::numbers))
        {
            if (valueCount > reader.remaining() / sizeof(double))
            {
                return reader.error("number count exceeds the remaining data.");
            }
            entry.type = VoiceConfigurationType::numbers;
            entry.numbers.resize(valueCount);
            for (auto& value : entry.numbers)
            {
                if (!reader.readDouble(value))
                {
                    return reader.error("truncated or non-finite number value.");
                }
            }
        }
        else if (type == static_cast<std::uint32_t>(VoiceConfigurationType::integers))
        {
            if (valueCount > reader.remaining() / sizeof(std::uint32_t))
            {
                return reader.error("integer count exceeds the remaining data.");
            }
            entry.type = VoiceConfigurationType::integers;
            entry.integers.resize(valueCount);
            for (auto& value : entry.integers)
            {
                if (!reader.readUint32(value))
                {
                    return reader.error("truncated integer value.");
                }
            }
        }
        else
        {
            return reader.error("unsupported value type " + juce::String(type) + ".");
        }
        entries.push_back(std::move(entry));
    }
    if (reader.remaining() != 0)
    {
        return reader.error("unexpected bytes after the final entry.");
    }
    return juce::Result::ok();
}

/**
 * @brief Finds a single string-valued configuration entry by its de-obfuscated name.
 */
juce::Result findSingleString(const std::vector<VoiceConfigurationEntry>& entries, std::string_view name, juce::MemoryBlock& value)
{
    const VoiceConfigurationEntry* found = nullptr;
    for (const auto& entry : entries)
    {
        if (entry.name != name)
        {
            continue;
        }
        if (found != nullptr)
        {
            return juce::Result::fail("Voice configuration has duplicate model field: " + juce::String::fromUTF8(name.data(), static_cast<int>(name.size())));
        }
        found = &entry;
    }
    if (found == nullptr || found->type != VoiceConfigurationType::strings || found->strings.size() != 1 || found->strings.front().isEmpty())
    {
        return juce::Result::fail("Voice configuration requires one non-empty string for: " + juce::String::fromUTF8(name.data(), static_cast<int>(name.size())));
    }
    value = found->strings.front();
    return juce::Result::ok();
}

/**
 * @brief Resolves a neural network model reference from configuration fields into a ModelReference struct.
 */
juce::Result readModelReference(const std::vector<VoiceConfigurationEntry>& entries, const VoiceDatabase& database, std::string_view keyField, std::string_view architectureField, ModelReference& model)
{
    if (auto result = findSingleString(entries, keyField, model.key); result.failed())
    {
        return result;
    }
    juce::MemoryBlock architecture;
    if (auto result = findSingleString(entries, architectureField, architecture); result.failed())
    {
        return result;
    }
    model.architecture.assign(static_cast<const char*>(architecture.getData()), architecture.getSize());
    if (!isText(model.architecture))
    {
        return juce::Result::fail("Voice configuration model architecture is not valid UTF-8 text.");
    }
    const auto& databaseEntries = database.getEntries();
    const auto found = std::find_if(databaseEntries.begin(), databaseEntries.end(), [&model](const VoiceEntry& entry)
                                    { return entry.key == model.key; });
    if (found == databaseEntries.end())
    {
        return juce::Result::fail("Voice configuration references a missing model: " + juce::String::fromUTF8(keyField.data(), static_cast<int>(keyField.size())));
    }
    auto decoded = decodeName(model.key);
    if (isText(decoded))
    {
        model.name = std::move(decoded);
    }
    return juce::Result::ok();
}
} // namespace

juce::Result VoiceConfiguration::load(VoiceDatabase& database)
{
    try
    {
        const auto& databaseEntries = database.getEntries();
        // In NOFS voice archives, the main configuration table is located by a 1-byte key: 0x91.
        const auto configuration = std::find_if(databaseEntries.begin(), databaseEntries.end(), [](const VoiceEntry& entry)
                                                { return entry.key.getSize() == 1 && *static_cast<const std::uint8_t*>(entry.key.getData()) == 0x91; });
        if (configuration == databaseEntries.end())
        {
            return juce::Result::fail("The NOFS database has no supported voice configuration entry.");
        }
        if (configuration->valueSize > maximumConfigurationBytes)
        {
            return juce::Result::fail("Voice configuration exceeds the 1 MiB limit.");
        }

        // Read and parse configuration entry payload
        juce::MemoryBlock bytes;
        if (auto result = database.readEntry(*configuration, bytes); result.failed())
        {
            return result;
        }
        VoiceConfiguration replacement;
        if (auto result = parseConfiguration(bytes, replacement.entries); result.failed())
        {
            return result;
        }

        // Resolve model references for duration, acoustic, and vocoder models:
        // 1. Duration / phoneme timing model: "model_duration", architecture: "model_duration_arch"
        if (auto result = readModelReference(replacement.entries, database, "model_duration", "model_duration_arch", replacement.duration); result.failed())
        {
            return result;
        }
        // 2. Acoustic / timbre model: "model_timbre_pred", architecture: "model_timbre_arch"
        if (auto result = readModelReference(replacement.entries, database, "model_timbre_pred", "model_timbre_arch", replacement.acoustic); result.failed())
        {
            return result;
        }
        // 3. Vocoder model: "model_vocoder", architecture: "model_vocoder_arch"
        if (auto result = readModelReference(replacement.entries, database, "model_vocoder", "model_vocoder_arch", replacement.vocoder); result.failed())
        {
            return result;
        }

        *this = std::move(replacement);
        return juce::Result::ok();
    }
    catch (const std::bad_alloc&)
    {
        return juce::Result::fail("Insufficient memory to load voice configuration.");
    }
}

const std::vector<VoiceConfigurationEntry>& VoiceConfiguration::getEntries() const noexcept
{
    return entries;
}

const ModelReference& VoiceConfiguration::getDurationModel() const noexcept
{
    return duration;
}

const ModelReference& VoiceConfiguration::getAcousticModel() const noexcept
{
    return acoustic;
}

const ModelReference& VoiceConfiguration::getVocoderModel() const noexcept
{
    return vocoder;
}

} // namespace sv::synthesis
