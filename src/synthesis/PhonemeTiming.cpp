#include "PhonemeTiming.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <set>
#include <utility>

namespace sv::synthesis
{
namespace
{
// Total input feature dimension for the duration neural network:
// 32 (phoneme) + 32 (language) + 16 (forward pos) + 16 (reverse pos)
// + 32 (encoded duration) + 16 (encoded pitch) + 128 (speaker/voice) = 272 channels.
constexpr std::size_t featureChannels = 272;
constexpr std::size_t maximumTensorElements = 64 * 1024 * 1024;

/**
 * @brief Representation of an individual phoneme during musical timing alignment.
 */
struct TimingPhone
{
    std::size_t syllableIndex = 0;   ///< Source syllable index
    std::size_t phonemeIndex = 0;    ///< Position within source syllable
    std::size_t languageIndex = 0;   ///< Language ID
    std::size_t unifiedIndex = 0;    ///< Index in unified cross-lingual phoneme inventory
    bool isNucleus = false;          ///< True if this phoneme is a vowel or diphthong (syllable nucleus)
};

/**
 * @brief Musical timing interval representing a note boundary or subdivided unit.
 */
struct TimingInterval
{
    std::vector<TimingPhone> phones;
    double durationSeconds = 0.0;
    int midiPitch = 60;
};

/**
 * @brief Aligns phonemes to musical note boundaries for natural singing synthesis.
 *
 * Vocal singing rules:
 * 1. Onset shift: Leading consonants before a vowel (e.g. "s" in "sa") are moved into
 *    the preceding interval (or preceding rest) so the vowel nucleus aligns with note onset.
 * 2. Multi-nuclei split: If an interval has multiple vowels (e.g. diphthongs or multi-syllable lyrics),
 *    it is split into multiple equal-duration intervals right before each subsequent nucleus.
 */
std::vector<TimingInterval> alignIntervals(std::vector<TimingInterval> intervals)
{
    // Step 1: Move leading consonants into preceding interval so vowel onset aligns with the beat
    for (std::size_t index = 1; index < intervals.size(); ++index)
    {
        auto& phones = intervals[index].phones;
        const auto nucleus = std::find_if(phones.begin(), phones.end(), [](const auto& phone)
                                          { return phone.isNucleus; });
        if (nucleus != phones.end())
        {
            auto& preceding = intervals[index - 1].phones;
            preceding.insert(preceding.end(), phones.begin(), nucleus);
            phones.erase(phones.begin(), nucleus);
        }
    }

    // Step 2: Subdivide intervals containing multiple vowel nuclei
    std::vector<TimingInterval> result;
    result.reserve(intervals.size());
    for (auto& interval : intervals)
    {
        const auto nuclei = static_cast<std::size_t>(std::count_if(interval.phones.begin(), interval.phones.end(), [](const auto& phone)
                                                                   { return phone.isNucleus; }));
        if (nuclei <= 1)
        {
            result.push_back(std::move(interval));
            continue;
        }

        // Split duration evenly across the vowel nuclei
        const double seconds = interval.durationSeconds / static_cast<double>(nuclei);
        std::size_t begin = 0;
        bool foundNucleus = false;
        for (std::size_t index = 0; index < interval.phones.size(); ++index)
        {
            if (interval.phones[index].isNucleus)
            {
                if (foundNucleus)
                {
                    result.push_back({{interval.phones.begin() + static_cast<std::ptrdiff_t>(begin), interval.phones.begin() + static_cast<std::ptrdiff_t>(index)}, seconds, interval.midiPitch});
                    begin = index;
                }
                foundNucleus = true;
            }
        }
        result.push_back({{interval.phones.begin() + static_cast<std::ptrdiff_t>(begin), interval.phones.end()}, seconds, interval.midiPitch});
    }
    return result;
}

// Model architecture type hashes in DNNI
constexpr std::array<std::uint64_t, 12> durationTypes{
    0x4c30eaff390f599a, 0x210ae732bcda3950, 0xedb2643ec31049d7, 0x76b77e5bf8d6500c,
    0x8e0f0182b380a860, 0xd4c2d7946f91e167, 0x135ee08897501166, 0x71a4b344df3f7ece,
    0xa4f623b00c5a97c0, 0x1fad6bd2dfa1a17f, 0x5d5d600f0e5453e9, 0x1db558504d28a4ad};
constexpr std::array<std::uint64_t, 12> frontendTypes{
    0xff369e74e595e206, 0xea62ca7c31e37ca0, 0x319b1e6e37917bd3, 0x05909feba966bb54,
    0x18b5591e22aba550, 0x825a31102572a303, 0xf0c8bdd6e4fe863a, 0x77f5f7f551d96e82,
    0xd9177b5063843fb0, 0xa097f32e2e5fe07b, 0xc6ea192cf9c095d1, 0x32870689962bff1d};

juce::Result failure(const juce::String& reason)
{
    return juce::Result::fail("Phoneme timing: " + reason);
}

std::uint32_t readWord(std::span<const std::uint8_t> bytes, std::size_t offset)
{
    return static_cast<std::uint32_t>(bytes[offset])
         | (static_cast<std::uint32_t>(bytes[offset + 1]) << 8)
         | (static_cast<std::uint32_t>(bytes[offset + 2]) << 16)
         | (static_cast<std::uint32_t>(bytes[offset + 3]) << 24);
}

/**
 * @brief Reads an embedding matrix and verifies its dimension requirements.
 */
juce::Result readEmbedding(const DnniReader& reader, std::size_t nodeIndex, std::size_t rows, std::size_t columns, DnniMatrix& matrix)
{
    if (const auto result = reader.readFloatMatrix(nodeIndex, matrix); result.failed())
    {
        return result;
    }
    if (matrix.rows != rows || matrix.columns != columns)
    {
        return failure("embedding matrix dimensions do not match the declared channels.");
    }
    return juce::Result::ok();
}

/**
 * @brief Encodes a continuous scalar into a multi-channel Gaussian / thermometer representation.
 */
void encodeScalar(float value, std::span<float> destination)
{
    const auto channels = destination.size();
    for (std::size_t channel = 0; channel < channels; ++channel)
    {
        const float center = static_cast<float>(channel) / static_cast<float>(channels - 1);
        const float delta = value - center;
        // Standard Gaussian RBF expansion
        destination[channel] = std::exp(-delta * delta * static_cast<float>(channels));
    }
}

/**
 * @brief Copies an embedding row into the feature vector buffer.
 */
void copyEmbedding(const DnniMatrix& matrix, std::size_t index, float* destination)
{
    std::copy_n(matrix.values.data() + index * matrix.columns, matrix.columns, destination);
}

} // namespace

juce::Result quantizePhonemeDurations(std::span<const PhonemeDuration> durations, float frameIntervalSeconds, std::vector<TimedPhoneme>& output)
{
    if (frameIntervalSeconds <= 0.0f || !std::isfinite(frameIntervalSeconds))
    {
        return failure("invalid acoustic frame interval.");
    }
    output.clear();
    output.reserve(durations.size());

    double cumulativeSeconds = 0.0;
    std::size_t previousFrame = 0;

    for (const auto& duration : durations)
    {
        cumulativeSeconds += duration.durationSeconds;
        // Round cumulative time to nearest acoustic frame boundary
        const auto targetFrame = static_cast<std::size_t>(std::max<double>(
            static_cast<double>(previousFrame + 1),
            std::round(cumulativeSeconds / static_cast<double>(frameIntervalSeconds))));

        output.push_back({duration.language, duration.symbol, targetFrame - previousFrame});
        previousFrame = targetFrame;
    }

    return juce::Result::ok();
}

juce::Result PhonemeTiming::load(const DnniReader& reader, std::size_t rootNode)
{
    const auto& nodes = reader.getNodes();
    if (rootNode >= nodes.size())
    {
        return failure("root node index out of bounds.");
    }

    // Verify root is a duration model container
    const auto& root = nodes[rootNode];
    if (std::find(durationTypes.begin(), durationTypes.end(), root.typeId) == durationTypes.end())
    {
        return failure("unsupported duration model root type.");
    }

    // Parse frontend feature extractor and neural network components
    // Child 0: Frontend feature descriptors, embeddings, and normalizers
    // Child 1: Inference network
    if (root.children.size() < 2)
    {
        return failure("duration model missing required child components.");
    }

    const auto frontendNode = root.children[0];
    const auto networkNode = root.children[1];

    if (std::find(frontendTypes.begin(), frontendTypes.end(), nodes[frontendNode].typeId) == frontendTypes.end())
    {
        return failure("unsupported frontend feature descriptor type.");
    }

    const auto& frontend = nodes[frontendNode];
    if (frontend.children.size() < 8)
    {
        return failure("frontend descriptor missing required embedding tables.");
    }

    // Read Phone Sets and Unified Phone Set
    phoneSets.clear();
    const auto phoneSetGroup = frontend.children[0];
    for (const auto childIndex : nodes[phoneSetGroup].children)
    {
        PhoneSet phoneSet;
        if (const auto result = readPhoneSet(reader, childIndex, phoneSet); result.failed())
        {
            return result;
        }
        phoneSets.push_back(std::move(phoneSet));
    }

    if (const auto result = readPhoneSet(reader, frontend.children[1], unifiedPhoneSet); result.failed())
    {
        return result;
    }

    // Load embeddings: language (32), phoneme (32), position (16)
    if (const auto result = readEmbedding(reader, frontend.children[2], phoneSets.size(), 32, languageEmbedding); result.failed())
    {
        return result;
    }
    if (const auto result = readEmbedding(reader, frontend.children[3], unifiedPhoneSet.symbols.size(), 32, phonemeEmbedding); result.failed())
    {
        return result;
    }
    if (const auto result = readEmbedding(reader, frontend.children[4], 8, 16, positionEmbedding); result.failed())
    {
        return result;
    }

    // Speaker / voice embedding vector (128 floats)
    if (const auto result = reader.readFloatVector(frontend.children[5], voiceEmbedding); result.failed())
    {
        return result;
    }
    if (voiceEmbedding.size() != 128)
    {
        return failure("voice embedding must be 128 elements.");
    }

    // Normalizers
    if (const auto result = inputNormalization.load(reader, frontend.children[6]); result.failed())
    {
        return result;
    }
    if (const auto result = outputNormalization.load(reader, frontend.children[7]); result.failed())
    {
        return result;
    }

    // Read pitch bounds from frontend payload
    const auto payload = reader.getPayload(frontendNode);
    if (payload.size() < 8)
    {
        return failure("frontend payload truncated.");
    }
    minimumPitch = static_cast<int>(readWord(payload, 0));
    maximumPitch = static_cast<int>(readWord(payload, 4));

    // Load neural network
    if (const auto result = network.load(reader, networkNode); result.failed())
    {
        return result;
    }

    return juce::Result::ok();
}

juce::Result PhonemeTiming::predict(std::span<const TimingSyllable> syllables, std::vector<PhonemeDuration>& output) const
{
    if (syllables.empty())
    {
        output.clear();
        return juce::Result::ok();
    }

    // 1. Map input syllables and phonemes into timing intervals
    std::vector<TimingInterval> intervals;
    std::size_t phonemeCount = 0;

    for (std::size_t syllableIndex = 0; syllableIndex < syllables.size(); ++syllableIndex)
    {
        const auto& syllable = syllables[syllableIndex];
        phonemeCount += syllable.phonemes.size();

        // Match syllable language to known phone sets
        std::size_t languageIndex = phoneSets.size();
        for (std::size_t index = 0; index < phoneSets.size(); ++index)
        {
            if (phoneSets[index].name == syllable.language)
            {
                if (languageIndex != phoneSets.size())
                {
                    return failure("the language matches more than one phone set.");
                }
                languageIndex = index;
            }
        }
        if (languageIndex == phoneSets.size())
        {
            return failure("unknown phoneme language: " + juce::String::fromUTF8(syllable.language.c_str()));
        }

        const auto& phoneSet = phoneSets[languageIndex];
        TimingInterval interval{{}, syllable.durationSeconds, syllable.midiPitch};
        interval.phones.reserve(syllable.phonemes.size());

        for (std::size_t position = 0; position < syllable.phonemes.size(); ++position)
        {
            const auto& symbol = syllable.phonemes[position];
            const auto source = std::find(phoneSet.symbols.begin(), phoneSet.symbols.end(), symbol);
            if (source == phoneSet.symbols.end())
            {
                return failure("unknown phoneme '" + juce::String::fromUTF8(symbol.c_str()) + "' for " + juce::String::fromUTF8(phoneSet.name.c_str()) + ".");
            }
            const auto sourceIndex = static_cast<std::size_t>(source - phoneSet.symbols.begin());
            const auto& unified = phoneSet.unifiedSymbols[sourceIndex];
            const auto target = std::find(unifiedPhoneSet.symbols.begin(), unifiedPhoneSet.symbols.end(), unified);
            const auto targetIndex = static_cast<std::size_t>(target - unifiedPhoneSet.symbols.begin());
            const auto& category = phoneSet.categories[sourceIndex];
            interval.phones.push_back({syllableIndex, position, languageIndex, targetIndex, category == "vowel" || category == "diphthong"});
        }
        intervals.push_back(std::move(interval));
    }

    // 2. Perform singing timing alignment (leading consonant shift and vowel splitting)
    intervals = alignIntervals(std::move(intervals));

    // 3. Compute normalized log-duration input features
    DnniTensor rawDuration{intervals.size(), 1, {}};
    rawDuration.values.reserve(intervals.size());
    for (const auto& interval : intervals)
    {
        // Reference uses log(durationSeconds + 0.01) to prevent log(0) singularity
        rawDuration.values.push_back(static_cast<float>(std::log(interval.durationSeconds + static_cast<double>(0.01f))));
    }
    DnniTensor normalizedDuration;
    if (const auto result = inputNormalization.normalize(rawDuration, normalizedDuration); result.failed())
    {
        return result;
    }

    // 4. Construct 272-channel input feature tensor
    DnniTensor features{phonemeCount, featureChannels, {}};
    features.values.resize(phonemeCount * featureChannels);
    std::vector<PhonemeDuration> durations;
    durations.reserve(phonemeCount);

    std::size_t frame = 0;
    for (std::size_t intervalIndex = 0; intervalIndex < intervals.size(); ++intervalIndex)
    {
        const auto& interval = intervals[intervalIndex];
        const float pitch = static_cast<float>(interval.midiPitch - minimumPitch) / static_cast<float>(maximumPitch - minimumPitch);

        for (std::size_t position = 0; position < interval.phones.size(); ++position)
        {
            const auto& phone = interval.phones[position];
            const auto& syllable = syllables[phone.syllableIndex];
            auto* destination = features.values.data() + frame * featureChannels;

            // Feature layout:
            // [0..31]: Phoneme embedding (32 channels)
            copyEmbedding(phonemeEmbedding, phone.unifiedIndex, destination);
            // [32..63]: Language embedding (32 channels)
            copyEmbedding(languageEmbedding, phone.languageIndex, destination + 32);
            // [64..79]: Forward position in syllable (clamped to 7) (16 channels)
            copyEmbedding(positionEmbedding, std::min(position, std::size_t{7}), destination + 64);
            // [80..95]: Reverse position from end of syllable (clamped to 7) (16 channels)
            copyEmbedding(positionEmbedding, std::min(interval.phones.size() - position - 1, std::size_t{7}), destination + 80);
            // [96..127]: RBF expansion of normalized interval duration (32 channels)
            encodeScalar(normalizedDuration.values[intervalIndex], {destination + 96, 32});
            // [128..143]: RBF expansion of normalized pitch (16 channels)
            encodeScalar(pitch, {destination + 128, 16});
            // [144..271]: Voice / speaker identity embedding (128 channels)
            std::copy(voiceEmbedding.begin(), voiceEmbedding.end(), destination + 144);

            durations.push_back({syllable.language, syllable.phonemes[phone.phonemeIndex], phone.syllableIndex, intervalIndex, 0.0});
            ++frame;
        }
    }

    // 5. Run neural network inference
    DnniTensor predicted;
    if (const auto result = network.run(features, predicted); result.failed())
    {
        return result;
    }
    if (predicted.frames != phonemeCount || predicted.channels != 1 || predicted.values.size() != phonemeCount)
    {
        return failure("the duration network must return one value per phoneme.");
    }

    // 6. Denormalize predicted log-durations
    DnniTensor logDurations;
    if (const auto result = outputNormalization.denormalize(predicted, logDurations); result.failed())
    {
        return result;
    }

    // 7. Softmax-like proportional distribution of note duration across phonemes
    std::size_t first = 0;
    for (const auto& interval : intervals)
    {
        const auto end = first + interval.phones.size();
        float sum = 0.0f;
        for (auto index = first; index < end; ++index)
        {
            const float value = std::exp(logDurations.values[index]);
            if (!std::isfinite(value) || value <= 0.0f)
            {
                return failure("a predicted duration is not a finite positive float.");
            }
            logDurations.values[index] = value;
            sum += value;
        }
        if (!std::isfinite(sum) || sum <= 0.0f)
        {
            return failure("the predicted timing interval duration sum is invalid.");
        }
        const float reciprocal = 1.0f / sum;
        for (auto index = first; index < end; ++index)
        {
            // Allocate fraction of musical interval duration to each phoneme
            const float fraction = logDurations.values[index] * reciprocal;
            const float seconds = static_cast<float>(static_cast<double>(fraction) * interval.durationSeconds);
            if (!std::isfinite(seconds) || seconds <= 0.0f)
            {
                return failure("the normalized duration cannot be represented as a positive float.");
            }
            durations[index].durationSeconds = static_cast<double>(seconds);
        }
        first = end;
    }

    output = std::move(durations);
    return juce::Result::ok();
}

} // namespace sv::synthesis
