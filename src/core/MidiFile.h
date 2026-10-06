#pragma once

#include "Project.h"

#include <juce_core/juce_core.h>

namespace sv
{
/**
 * @brief Imports a Standard MIDI File (.mid) into a Project structure.
 *
 * Capabilities and semantics:
 * - Supports Standard MIDI Files format 0 (single multi-channel track) and format 1 (multi-track).
 * - Reconstructs notes from Note-On and Note-Off event pairs.
 * - Associates lyrics meta-events (type 5) with corresponding note onsets.
 * - Extracts tempo map (tempo meta-events) and time signatures (time signature meta-events).
 * - Converts MIDI PPQ ticks into OpenSV's exact high-precision Blicks (705,600,000 per quarter note).
 * - On success, replaces the destination project and normalises all note IDs and structures.
 *
 * @param file Source MIDI file on disk.
 * @param project Destination Project struct populated on success.
 * @return juce::Result::ok() or failure description.
 */
[[nodiscard]] juce::Result importMidiFile(const juce::File& file, Project& project);

/**
 * @brief Exports a Project structure to a Standard MIDI File format 1 (.mid).
 *
 * Output characteristics:
 * - Writes a Conductor Track (Track 0) containing project name, tempo changes, and meter changes.
 * - Writes separate tracks for each vocal track in the project.
 * - Exports lyrics as MIDI meta-event type 5 on note onsets.
 * - Uses high resolution 9600 ticks per quarter note to preserve fine microtiming.
 * - Assigns General MIDI channels (skipping channel 10 which is reserved for GM percussion).
 * - Performs atomic writing via temporary file replacement.
 *
 * @param file Destination file on disk.
 * @param project The project to export.
 * @return juce::Result::ok() or failure description.
 */
[[nodiscard]] juce::Result exportMidiFile(const juce::File& file, const Project& project);

} // namespace sv
