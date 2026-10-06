#pragma once

#include "Project.h"

#include <juce_data_structures/juce_data_structures.h>
#include <juce_events/juce_events.h>

#include <cstdint>
#include <functional>
#include <vector>

namespace sv
{
/**
 * @brief Thread-safe document model and undo/redo manager for the active project.
 *
 * Architecture and Concurrency Invariants:
 * - All access to ProjectDocument is strictly confined to the main GUI (JUCE message) thread.
 * - The background audio renderer and preview engines NEVER access ProjectDocument directly;
 *   they receive immutable, deep-copied `Project` snapshots via message-passing.
 * - ProjectDocument inherits from juce::ChangeBroadcaster. Whenever any property, track, note,
 *   selection, or document state changes, `sendChangeMessage()` is emitted to notify listening UI views.
 *
 * Versioning & Invalidation:
 * - `revision`: Incremented on EVERY mutation (user edit, undo, redo, track addition, document load).
 *   UI views compare their local revision against getRevision() to detect dirty state.
 * - `generation`: Incremented ONLY when a project is created from scratch or loaded from disk.
 *   Background synthesis pipelines use the generation counter to invalidate all phrase caches
 *   when switching between different project files.
 */
class ProjectDocument final : public juce::ChangeBroadcaster
{
public:
    ProjectDocument();

    /**
     * @brief Returns a reference to the active project state.
     */
    [[nodiscard]] const Project& getProject() const;

    /**
     * @brief Gets the 0-based index of the currently selected track in the UI.
     */
    [[nodiscard]] int getActiveTrackIndex() const;

    /**
     * @brief Sets the active track index, clamping to valid range and clearing note selection.
     */
    void setActiveTrackIndex(int index);

    /**
     * @brief Gets the active NoteGroup belonging to the current track.
     */
    [[nodiscard]] const NoteGroup& getActiveGroup() const;

    /**
     * @brief Gets the timeline time offset (in blicks) of the active track's main group reference.
     */
    [[nodiscard]] Blick getActiveGroupOffset() const;

    /**
     * @brief Gets the list of currently selected note IDs in the active track.
     */
    [[nodiscard]] const std::vector<NoteId>& getSelectedNoteIds() const;

    /**
     * @brief Sets the note selection, automatically pruning IDs that do not exist in the active track.
     */
    void setSelectedNoteIds(std::vector<NoteId> ids);

    /**
     * @brief Applies a mutating edit transaction with undo/redo support.
     *
     * Creates an undo transaction, clones the project state, executes the mutator callback,
     * normalises the result via normaliseProject(), and registers the change with the UndoManager.
     *
     * @param name User-visible name of the action (e.g. "Move Notes", "Change Lyric").
     * @param edit Callback taking `Project&` to apply mutations.
     */
    void performEdit(const juce::String& name, const std::function<void(Project&)>& edit);

    /**
     * @brief Undoes the most recent edit transaction.
     */
    void undo();

    /**
     * @brief Redoes the most recently undone transaction.
     */
    void redo();

    [[nodiscard]] bool canUndo() const;
    [[nodiscard]] bool canRedo() const;

    /**
     * @brief Resets the document to a clean, empty project with default track and tempo.
     */
    void newProject();

    /**
     * @brief Loads a project from an SVP file on disk.
     */
    [[nodiscard]] juce::Result load(const juce::File& file);

    /**
     * @brief Saves the current project state to an SVP file on disk.
     */
    [[nodiscard]] juce::Result save(const juce::File& file);

    /**
     * @brief Imports notes, tracks, and tempos from a standard MIDI file into the project.
     */
    [[nodiscard]] juce::Result importMidi(const juce::File& file);

    /**
     * @brief Exports the project's notes and tempo map to a standard MIDI file.
     */
    [[nodiscard]] juce::Result exportMidi(const juce::File& file) const;

    /**
     * @brief Appends a new vocal track to the project.
     */
    void addTrack();

    /**
     * @brief Removes the track at the specified index.
     */
    void removeTrack(int index);

    /**
     * @brief Manually overrides the dirty/modified flag.
     */
    void setModified(bool modified = true);

    /**
     * @brief Checks if the project has unsaved changes.
     */
    [[nodiscard]] bool isModified() const;

    /**
     * @brief Gets the disk file currently associated with this document (or empty File if unsaved).
     */
    [[nodiscard]] const juce::File& getFile() const;

    /**
     * @brief Monotonically increasing revision counter (changes on every edit, undo, redo, load).
     */
    [[nodiscard]] std::uint64_t getRevision() const;

    /**
     * @brief Generation counter that changes ONLY when a new document is loaded or created.
     */
    [[nodiscard]] std::uint64_t getGeneration() const;

private:
    class EditAction;

    /**
     * @brief Restores project state during undo/redo and broadcasts change notification.
     */
    void applyState(const Project& state, std::uint64_t stateId);

    /**
     * @brief Replaces project completely, clears undo history, and bumps generation counter.
     */
    void reset(Project replacement, const juce::File& file);

    /**
     * @brief Removes note IDs from selection that are no longer present in the active group.
     */
    void pruneSelection();

    Project project;
    juce::UndoManager undoManager;
    juce::File file;
    std::vector<NoteId> selectedNoteIds;
    int activeTrackIndex = 0;
    std::uint64_t revision = 0;
    std::uint64_t generation = 0;
    std::uint64_t currentStateId = 0;
    std::uint64_t savedStateId = 0;
    std::uint64_t nextStateId = 1;
    bool explicitlyModified = false;
};
} // namespace sv
