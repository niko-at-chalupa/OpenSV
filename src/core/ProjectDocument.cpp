#include "ProjectDocument.h"

#include "MidiFile.h"
#include "ProjectFile.h"

#include <algorithm>
#include <cstddef>
#include <utility>

namespace sv
{
namespace
{
// Total memory allocated for undo/redo snapshots before old actions are evicted.
constexpr int undoBudgetBytes = 64 * 1024 * 1024; // 64 MB budget

/**
 * @brief Approximates memory usage in bytes of a ParameterCurve.
 */
std::size_t curveStorageBytes(const ParameterCurve& curve)
{
    return curve.mode.size() + curve.preservedFieldsJson.size() + curve.points.size() * sizeof(AutomationPoint);
}

/**
 * @brief Approximates memory usage in bytes of a NoteGroup (including all notes and curves).
 */
std::size_t groupBytes(const NoteGroup& group)
{
    std::size_t bytes = sizeof(group) + group.id.size() + group.name.size() + group.preservedFieldsJson.size()
                      + curveStorageBytes(group.pitchDelta) + curveStorageBytes(group.vibratoEnv);
    for (const auto& [name, curve] : group.vocalModes)
    {
        bytes += name.size() + curveStorageBytes(curve.curve) + curve.preservedFieldsJson.size();
    }
    for (const auto& note : group.notes)
    {
        bytes += sizeof(note) + note.lyrics.size() + note.phonemes.size() + note.preservedFieldsJson.size();
        bytes += note.musicalType.size() + note.accent.size() + note.attributes.preservedFieldsJson.size()
               + note.systemAttributes.preservedFieldsJson.size();
    }
    return bytes;
}

/**
 * @brief Approximates memory usage in bytes of a GroupReference.
 */
std::size_t referenceBytes(const GroupReference& reference)
{
    std::size_t bytes = sizeof(reference) + reference.groupId.size() + reference.audioFile.size()
                      + reference.preservedFieldsJson.size() + reference.voicePitch.preservedFieldsJson.size()
                      + reference.vocalModePreset.size() + curveStorageBytes(reference.systemPitchDelta);
    for (const auto& [name, amount] : reference.vocalModeParams)
    {
        static_cast<void>(amount);
        bytes += name.size() + sizeof(double);
    }
    return bytes;
}

/**
 * @brief Approximates memory usage in bytes of an entire Project snapshot.
 *
 * Used by UndoableAction::getSizeInUnits to ensure undoManager does not exceed undoBudgetBytes.
 */
std::size_t projectBytes(const Project& project)
{
    std::size_t bytes = sizeof(project) + project.name.size() + project.preservedFieldsJson.size();
    for (const auto& track : project.tracks)
    {
        bytes += sizeof(track) + track.name.size() + track.preservedFieldsJson.size()
               + groupBytes(track.mainGroup) + referenceBytes(track.mainRef);
        bytes += track.voice.databasePath.size() + track.voice.language.size() + track.voice.dictionaryDirectory.size();
        for (const auto& reference : track.groups)
        {
            bytes += referenceBytes(reference);
        }
    }
    for (const auto& group : project.library)
    {
        bytes += groupBytes(group);
    }
    for (const auto& tempo : project.tempoMap.tempos)
    {
        bytes += sizeof(tempo) + tempo.preservedFieldsJson.size();
    }
    for (const auto& signature : project.tempoMap.timeSignatures)
    {
        bytes += sizeof(signature) + signature.preservedFieldsJson.size();
    }
    return bytes;
}
} // namespace

/**
 * @brief JUCE UndoableAction implementation storing before/after mementos of the Project.
 *
 * When an edit occurs, deep copies of both previous and resulting states are retained.
 * Performing restores the 'after' state; undoing restores the 'before' state.
 */
class ProjectDocument::EditAction final : public juce::UndoableAction
{
public:
    EditAction(ProjectDocument& owner, Project before, Project after, std::uint64_t beforeId, std::uint64_t afterId)
        : owner(owner), before(std::move(before)), after(std::move(after)), beforeId(beforeId), afterId(afterId)
    {
    }

    bool perform() override
    {
        owner.applyState(after, afterId);
        return true;
    }

    bool undo() override
    {
        owner.applyState(before, beforeId);
        return true;
    }

    int getSizeInUnits() override
    {
        const auto units = projectBytes(before) + projectBytes(after);
        return static_cast<int>(std::min(units, static_cast<std::size_t>(undoBudgetBytes)));
    }

private:
    ProjectDocument& owner;
    Project before;
    Project after;
    std::uint64_t beforeId;
    std::uint64_t afterId;
};

ProjectDocument::ProjectDocument()
    : project(createEmptyProject()), undoManager(undoBudgetBytes, 1)
{
}

const Project& ProjectDocument::getProject() const
{
    return project;
}

int ProjectDocument::getActiveTrackIndex() const
{
    return activeTrackIndex;
}

void ProjectDocument::setActiveTrackIndex(int index)
{
    const int bounded = std::clamp(index, 0, static_cast<int>(project.tracks.size()) - 1);
    if (activeTrackIndex != bounded)
    {
        activeTrackIndex = bounded;
        selectedNoteIds.clear(); // Switching tracks clears note selection
        sendChangeMessage();
    }
}

const NoteGroup& ProjectDocument::getActiveGroup() const
{
    return project.tracks[static_cast<std::size_t>(activeTrackIndex)].mainGroup;
}

Blick ProjectDocument::getActiveGroupOffset() const
{
    return project.tracks[static_cast<std::size_t>(activeTrackIndex)].mainRef.timeOffset;
}

const std::vector<NoteId>& ProjectDocument::getSelectedNoteIds() const
{
    return selectedNoteIds;
}

void ProjectDocument::setSelectedNoteIds(std::vector<NoteId> ids)
{
    selectedNoteIds = std::move(ids);
    pruneSelection();
    sendChangeMessage();
}

void ProjectDocument::performEdit(const juce::String& name, const std::function<void(Project&)>& edit)
{
    // Clone project to prepare new state
    Project next = project;
    edit(next);
    // Enforce model invariants (sorted notes, valid IDs, clamp pitch)
    normaliseProject(next);

    // Record transaction in UndoManager
    undoManager.beginNewTransaction(name);
    undoManager.perform(new EditAction(*this, project, std::move(next), currentStateId, nextStateId++));
}

void ProjectDocument::undo()
{
    undoManager.undo();
}

void ProjectDocument::redo()
{
    undoManager.redo();
}

bool ProjectDocument::canUndo() const
{
    return undoManager.canUndo();
}

bool ProjectDocument::canRedo() const
{
    return undoManager.canRedo();
}

void ProjectDocument::newProject()
{
    reset(createEmptyProject(), {});
}

juce::Result ProjectDocument::load(const juce::File& source)
{
    Project replacement;
    const auto result = loadProjectFile(source, replacement);
    if (result.wasOk())
    {
        reset(std::move(replacement), source);
    }
    return result;
}

juce::Result ProjectDocument::save(const juce::File& destination)
{
    const auto result = saveProjectFile(destination, project);
    if (result.wasOk())
    {
        file = destination;
        savedStateId = currentStateId;
        explicitlyModified = false;
        sendChangeMessage();
    }
    return result;
}

juce::Result ProjectDocument::importMidi(const juce::File& source)
{
    Project imported;
    const auto result = importMidiFile(source, imported);
    if (result.wasOk())
    {
        performEdit("Import MIDI", [&imported](Project& target)
                    { target = std::move(imported); });
        setActiveTrackIndex(0);
    }
    return result;
}

juce::Result ProjectDocument::exportMidi(const juce::File& destination) const
{
    return exportMidiFile(destination, project);
}

void ProjectDocument::addTrack()
{
    performEdit("Add Track", [](Project& target)
                {
        Track track;
        track.name = "Track " + std::to_string(target.tracks.size() + 1);
        track.mainGroup = createNoteGroup();
        track.mainRef.groupId = track.mainGroup.id;
        target.tracks.push_back(std::move(track)); });
    setActiveTrackIndex(static_cast<int>(project.tracks.size()) - 1);
}

void ProjectDocument::removeTrack(int index)
{
    if (index < 0 || index >= static_cast<int>(project.tracks.size()))
    {
        return;
    }
    performEdit("Remove Track", [index](Project& target)
                { target.tracks.erase(target.tracks.begin() + index); });
}

void ProjectDocument::setModified(bool modified)
{
    explicitlyModified = modified;
    if (!modified)
    {
        savedStateId = currentStateId;
    }
    sendChangeMessage();
}

bool ProjectDocument::isModified() const
{
    return explicitlyModified || currentStateId != savedStateId;
}

const juce::File& ProjectDocument::getFile() const
{
    return file;
}

std::uint64_t ProjectDocument::getRevision() const
{
    return revision;
}

std::uint64_t ProjectDocument::getGeneration() const
{
    return generation;
}

void ProjectDocument::applyState(const Project& state, std::uint64_t stateId)
{
    project = state;
    currentStateId = stateId;
    ++revision; // Signal UI that model was modified
    activeTrackIndex = std::clamp(activeTrackIndex, 0, static_cast<int>(project.tracks.size()) - 1);
    pruneSelection();
    sendChangeMessage();
}

void ProjectDocument::reset(Project replacement, const juce::File& source)
{
    normaliseProject(replacement);
    undoManager.clearUndoHistory();
    project = std::move(replacement);
    file = source;
    selectedNoteIds.clear();
    activeTrackIndex = 0;
    currentStateId = nextStateId++;
    savedStateId = currentStateId;
    explicitlyModified = false;
    ++revision;
    ++generation; // Bumping generation causes background renderer to purge voice caches
    sendChangeMessage();
}

void ProjectDocument::pruneSelection()
{
    const auto& notes = getActiveGroup().notes;
    // Remove any note ID that does not exist in the active note group
    std::erase_if(selectedNoteIds, [&notes](NoteId id)
                  { return std::none_of(notes.begin(), notes.end(), [id](const Note& note)
                                        { return note.id == id; }); });
    // Sort and deduplicate selected note IDs
    std::sort(selectedNoteIds.begin(), selectedNoteIds.end());
    selectedNoteIds.erase(std::unique(selectedNoteIds.begin(), selectedNoteIds.end()), selectedNoteIds.end());
}

} // namespace sv
