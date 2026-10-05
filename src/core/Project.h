#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace sv
{
using Blick = std::int64_t;
using NoteId = std::uint64_t;
inline constexpr Blick blicksPerQuarter = 705600000;

struct PitchAttributes
{
    // Missing values inherit from the group reference and then engine defaults.
    std::optional<double> tF0Offset;
    std::optional<double> tF0Left;
    std::optional<double> tF0Right;
    std::optional<double> dF0Left;
    std::optional<double> dF0Right;
    std::optional<double> tF0VbrStart;
    std::optional<double> tF0VbrLeft;
    std::optional<double> tF0VbrRight;
    std::optional<double> dF0Vbr;
    std::optional<double> pF0Vbr;
    std::optional<double> fF0Vbr;
    std::optional<double> dF0VbrMod;
    std::optional<double> rTone;
    std::optional<double> rIntonation;
    std::string preservedFieldsJson;
};

struct Note
{
    NoteId id = 0;
    Blick onset = 0;
    Blick duration = blicksPerQuarter;
    int pitch = 60;
    std::string lyrics = "la";
    std::string phonemes;
    std::string preservedFieldsJson;
    double detune = 0.0; // Cents.
    bool instantMode = true;
    std::string musicalType = "singing";
    PitchAttributes attributes;
    PitchAttributes systemAttributes;
    std::string accent;
};

struct AutomationPoint
{
    Blick position = 0;
    double value = 0.0;
};

struct ParameterCurve
{
    std::string mode = "cubic";
    std::vector<AutomationPoint> points;
    std::string preservedFieldsJson;
};

struct NoteGroup
{
    std::string id;
    std::string name;
    std::vector<Note> notes;
    ParameterCurve pitchDelta;
    std::string preservedFieldsJson;
    ParameterCurve vibratoEnv;
};

struct GroupReference
{
    std::string groupId;
    Blick timeOffset = 0;
    int pitchOffset = 0;
    Blick absoluteBegin = 0;
    Blick absoluteEnd = -1;
    bool isInstrumental = false;
    std::string audioFile;
    double audioDurationSeconds = 0.0;
    std::string preservedFieldsJson;
    PitchAttributes voicePitch;
    ParameterCurve systemPitchDelta;
};

struct VoiceSettings
{
    std::string databasePath;
    std::string language = "japanese";
    std::string dictionaryDirectory;
};

struct Track
{
    std::string name = "Track 1";
    NoteGroup mainGroup;
    GroupReference mainRef;
    std::vector<GroupReference> groups;
    double gain = 1.0;
    double pan = 0.0;
    bool mute = false;
    bool solo = false;
    VoiceSettings voice;
    std::string preservedFieldsJson;
};

struct Tempo
{
    Blick position = 0;
    double bpm = 120.0;
    std::string preservedFieldsJson;
};

struct TimeSignature
{
    int bar = 0;
    int numerator = 4;
    int denominator = 4;
    std::string preservedFieldsJson;
};

struct TempoMap
{
    std::vector<Tempo> tempos{{0, 120.0, {}}};
    std::vector<TimeSignature> timeSignatures{{0, 4, 4, {}}};

    [[nodiscard]] double blickToSeconds(Blick position) const;
    [[nodiscard]] Blick secondsToBlick(double seconds) const;
    [[nodiscard]] double getTempoAt(Blick position) const;
};

struct Project
{
    std::string name = "Untitled";
    std::vector<Track> tracks;
    std::vector<NoteGroup> library;
    TempoMap tempoMap;
    std::string preservedFieldsJson;
};

[[nodiscard]] NoteId createNoteId();
[[nodiscard]] NoteGroup createNoteGroup(const std::string& name = "Main");
[[nodiscard]] Project createEmptyProject();
[[nodiscard]] const NoteGroup* findNoteGroup(const Project& project, const std::string& groupId);
[[nodiscard]] Blick getProjectEnd(const Project& project);

// Sorts timelines and establishes runtime note identities after an edit or import.
void normaliseProject(Project& project);
} // namespace sv
