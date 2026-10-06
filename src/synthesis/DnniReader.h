#pragma once

#include <juce_core/juce_core.h>

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace sv::synthesis
{
/**
 * @brief Represents a single node in a DNNI neural network tree.
 *
 * DNNI files store neural networks as an hierarchical tree of modules and primitives:
 * - Module nodes represent compound layers (e.g. GRU, Dilated Conv, ResNet blocks).
 * - Leaf primitive nodes store weights, biases, or metadata (e.g. `prim0` matrix, `prim1` vector).
 */
struct DnniNode
{
    std::uint32_t marker = 0;              ///< Node header marker word (0x7fca40ff or 0x7fca41ff).
    std::uint64_t typeId = 0;              ///< 64-bit type hash / identifier.
    std::string type;                      ///< Human-readable layer or primitive name (e.g. "prim0", "modl0").
    std::size_t offset = 0;                ///< Byte offset of the 20-byte node header in the file.
    std::size_t payloadOffset = 0;         ///< Byte offset of the node's raw payload data.
    std::size_t payloadSize = 0;           ///< Length in bytes of the node's payload.
    std::vector<std::size_t> children;     ///< Indices of child nodes in the flat node array.
};

/**
 * @brief Decoded 2D weight matrix stored in row-major layout.
 *
 * In DNNI neural networks:
 * - rows: number of output channels / units.
 * - columns: number of input channels / units.
 * - values: row-major array of size rows * columns (values[row * columns + column]).
 */
struct DnniMatrix
{
    std::uint32_t rows = 0;                ///< Output dimension (rows).
    std::uint32_t columns = 0;             ///< Input dimension (columns).
    std::vector<float> values;             ///< Dequantized, unpacked 32-bit floating-point weights.
};

/**
 * @brief Binary reader and parser for Synthesizer V DNNI neural network files.
 *
 * Format specification:
 * - Magic header (4 bytes): 0x7fca00ff
 * - Format version (4 bytes): 1 or 2
 * - Node tree layout:
 *   - 20-byte node header per node:
 *     - [0..3]: Marker (0x7fca40ff or 0x7fca41ff)
 *     - [4..11]: 8-byte type identifier / FNV-1a hash
 *     - [12..15]: Child node count (in v2, obfuscated via XOR mask)
 *     - [16..19]: Payload byte length
 *     - Followed by raw payload bytes
 *     - Followed by child node tree records
 * - Supports weight dequantization (8-bit and 16-bit integer quantization with row-wise scaling).
 * - Supports Block Compressed Sparse Row (BCSR) sparse matrix decompression.
 */
class DnniReader
{
public:
    /**
     * @brief Loads and parses a DNNI neural network file from disk.
     */
    [[nodiscard]] juce::Result load(const juce::File& file);

    /**
     * @brief Loads and parses a DNNI model from an in-memory byte block.
     */
    [[nodiscard]] juce::Result load(juce::MemoryBlock bytes);

    /**
     * @brief Returns format version (1 or 2).
     */
    [[nodiscard]] std::uint32_t getVersion() const noexcept;

    /**
     * @brief Returns the flat list of parsed nodes.
     */
    [[nodiscard]] const std::vector<DnniNode>& getNodes() const noexcept;

    /**
     * @brief Returns a zero-copy byte view of the payload for a given node.
     */
    [[nodiscard]] std::span<const std::uint8_t> getPayload(std::size_t nodeIndex) const;

    /**
     * @brief Decodes a `prim1` 1D float vector node (e.g. bias weights).
     */
    [[nodiscard]] juce::Result readFloatVector(std::size_t nodeIndex, std::vector<float>& values) const;

    /**
     * @brief Decodes a matrix node (`prim0`, `prim2`, `prim3`, `prim4`, `prim5`) into floating-point weights.
     */
    [[nodiscard]] juce::Result readFloatMatrix(std::size_t nodeIndex, DnniMatrix& matrix) const;

private:
    [[nodiscard]] juce::Result parse();
    [[nodiscard]] juce::Result parseNode(std::size_t& position, unsigned depth, std::size_t& nodeIndex);

    juce::MemoryBlock data;
    std::uint32_t version = 0;
    std::vector<DnniNode> nodes;
};

} // namespace sv::synthesis
