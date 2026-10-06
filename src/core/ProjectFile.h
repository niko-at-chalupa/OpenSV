#pragma once

#include "Project.h"

#include <juce_core/juce_core.h>

namespace sv
{
/**
 * @brief Deserializes a Synthesizer V Project (.svp) JSON file into a Project struct.
 *
 * Implements full format version 153 specification:
 * - Parses tracks, main groups, library groups, group references.
 * - Parses notes, pitch, lyrics, phonemes, detune, pitch attributes.
 * - Parses parameter curves (pitchDelta, vibratoEnv, vocalModes) with point arrays.
 * - Parses tempo map and time signature markers.
 * - Preserves unrecognized / vendor-specific JSON fields in `preservedFieldsJson`
 *   to ensure lossless round-tripping with commercial Synthesizer V Studio.
 *
 * @param file The .svp file to load from disk.
 * @param project Output project struct populated with deserialized data.
 * @return juce::Result::ok() on success, or failure with error description.
 */
[[nodiscard]] juce::Result loadProjectFile(const juce::File& file, Project& project);

/**
 * @brief Serializes a Project struct into a Synthesizer V Project (.svp) JSON file.
 *
 * Emits schema version 153 compliant JSON:
 * - Writes time signatures, tempos, tracks, groups, notes, and curves.
 * - Merges back any preserved unknown fields previously read from `preservedFieldsJson`.
 *
 * @param file The destination file path to write.
 * @param project The project struct to serialize.
 * @return juce::Result::ok() on success, or failure with error description.
 */
[[nodiscard]] juce::Result saveProjectFile(const juce::File& file, const Project& project);

} // namespace sv
