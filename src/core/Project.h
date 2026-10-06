#pragma once

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace sv
{
/**
 * @brief Musical time unit in Synthesizer V (SV).
 *
 * 1 quarter note = 705,600,000 blicks.
 * Why this specific large number?
 * 705,600,000 = 480 (standard MIDI ticks/quarter note) * 1,470,000.
 * Furthermore, 1,470,000 is divisible by common audio sample rates:
 * - 44,100 Hz (1,470,000 / 44,100 = 100/3)
 * - 48,000 Hz (1,470,000 / 48,000 = 1225/40)
 * It also factors nicely for arbitrary musical tuplets (triplets: /3, quintuplets: /5,
 * septuplets: /7, etc.) without roundoff errors across bars.
 */
using Blick = std::int64_t;

/**
 * @brief Runtime unique identifier assigned to each note.
 *
 * Used for GUI selection tracking, undo/redo diffing, and matching notes
 * across synthesis phrase partitions. Note IDs are unique across a session.
 */
using NoteId = std::uint64_t;

/**
 * @brief Constant defining the number of blicks per quarter note (one beat in 4/4).
 */
inline constexpr Blick blicksPerQuarter = 705600000;

/**
 * @brief Manual pitch transition and vibrato parameters.
 *
 * In Synthesizer V, pitch curves can be driven by high-level macro parameters
 * (pitch transitions between notes, and sinusoidal/modulated vibrato envelopes).
 * All properties are optional: when std::nullopt, the value is inherited from
 * the enclosing group or defaults to the engine's default value.
 */
struct PitchAttributes
{
    // --- Pitch Transition Parameters (between adjacent notes) ---
    // tF0Offset: Shift of the transition midpoint relative to note onset (in seconds).
    std::optional<double> tF0Offset;
    // tF0Left: Transition duration to the left of the transition point (in seconds).
    std::optional<double> tF0Left;
    // tF0Right: Transition duration to the right of the transition point (in seconds).
    std::optional<double> tF0Right;
    // dF0Left: Pitch overshoot / depth on the left side of transition (in semitones).
    std::optional<double> dF0Left;
    // dF0Right: Pitch overshoot / depth on the right side of transition (in semitones).
    std::optional<double> dF0Right;

    // --- Vibrato Envelope & Modulation Parameters ---
    // tF0VbrStart: Delay before vibrato onset from note start (in seconds).
    std::optional<double> tF0VbrStart;
    // tF0VbrLeft: Attack time / fade-in duration for vibrato depth (in seconds).
    std::optional<double> tF0VbrLeft;
    // tF0VbrRight: Release time / fade-out duration before note end (in seconds).
    std::optional<double> tF0VbrRight;
    // dF0Vbr: Peak vibrato depth (in semitones, e.g. 1.0 = +/- 1 semitone modulation).
    std::optional<double> dF0Vbr;
    // pF0Vbr: Initial vibrato phase (in radians / normalized fraction).
    std::optional<double> pF0Vbr;
    // fF0Vbr: Vibrato oscillation frequency (in Hz, typically ~5.0 - 6.0 Hz).
    std::optional<double> fF0Vbr;
    // dF0VbrMod: Frequency modulation / jitter applied to vibrato speed.
    std::optional<double> dF0VbrMod;

    // --- Timbre & Intonation Modifiers ---
    // rTone: Tone color / formant shift parameter.
    std::optional<double> rTone;
    // rIntonation: Intonation strength multiplier for AI pitch model.
    std::optional<double> rIntonation;

    /**
     * Preserved raw JSON of unrecognized vendor fields in SVP files.
     * Ensures lossless round-trip serialization when saving.
     */
    std::string preservedFieldsJson;
};

/**
 * @brief Representation of a single musical note in a track or group.
 */
struct Note
{
    NoteId id = 0;              ///< Unique runtime note ID (nonzero).
    Blick onset = 0;           ///< Start position relative to group origin (in blicks).
    Blick duration = blicksPerQuarter; ///< Length of note (in blicks, strictly > 0).
    int pitch = 60;            ///< MIDI note number (0 to 127, e.g., 60 = Middle C / C4).
    std::string lyrics = "la"; ///< Syllable text to be sung (e.g. "la", "hello", "き").
    std::string phonemes;      ///< Optional space-separated phoneme overrides (e.g. "l aa").
    std::string preservedFieldsJson; ///< Extra vendor JSON fields preserved across edits.
    double detune = 0.0;       ///< Microtonal pitch offset in cents (100 cents = 1 semitone).
    bool instantMode = true;   ///< Whether instant pitch mode (smooth neural transitions) is enabled.
    std::string musicalType = "singing"; ///< Musical style: "singing" or "rap".
    PitchAttributes attributes;        ///< User-specified manual pitch/vibrato overrides.
    PitchAttributes systemAttributes;  ///< Automatically calculated pitch/vibrato defaults.
    std::string accent;        ///< Stress / emphasis markings (e.g. for English pronunciation).
};

/**
 * @brief Single control point on a continuous parameter automation curve.
 */
struct AutomationPoint
{
    Blick position = 0;        ///< Time location of the point in blicks.
    double value = 0.0;        ///< Parameter value at this position.
};

/**
 * @brief Piecewise parameter automation curve (pitch delta, vibrato env, tension, etc.).
 *
 * Supported interpolation modes:
 * - "cubic": Hermite cubic spline interpolation with custom Synthesizer V weight blending.
 * - "cosine": Half-cosine smoothstep interpolation.
 * - "linear": Standard piecewise linear interpolation.
 */
struct ParameterCurve
{
    std::string mode = "cubic";         ///< Interpolation mode ("cubic", "cosine", "linear").
    std::vector<AutomationPoint> points;///< Ordered control points (sorted strictly by position).
    std::string preservedFieldsJson;    ///< Unrecognized vendor fields retained for serialization.
};

/**
 * @brief Configuration for a specific vocal mode / singing style (e.g., "Power", "Soft").
 */
struct VocalModeSetting
{
    bool enabled = true;       ///< Whether this vocal mode is currently active.
    double amount = 0.0;       ///< Static weight/intensity of this mode (typically 0.0 to 1.0).
    bool hasCurve = false;     ///< Whether dynamic automation curve is present.
    ParameterCurve curve;      ///< Continuous automation curve for mode intensity over time.
    std::string preservedFieldsJson; ///< Retained unknown vendor properties.
};

/**
 * @brief Container of notes and automation curves.
 *
 * Corresponds to a note group in Synthesizer V. Tracks have a default "main" group,
 * and additional reusable note groups can be stored in the project library and
 * referenced across tracks.
 */
struct NoteGroup
{
    std::string id;            ///< UUID string identifying this note group.
    std::string name;          ///< Display name of the group.
    std::vector<Note> notes;   ///< List of notes sorted by onset.
    ParameterCurve pitchDelta; ///< Continuous pitch bend curve in semitones (offset from note pitch).
    std::string preservedFieldsJson; ///< Retained unknown vendor properties.
    ParameterCurve vibratoEnv; ///< Dynamic vibrato depth envelope curve.
    std::map<std::string, VocalModeSetting> vocalModes; ///< Vocal style parameters keyed by mode name.
};

/**
 * @brief Reference (instance) of a NoteGroup placed on a track timeline.
 *
 * Allows a single NoteGroup in the library to be re-used multiple times at different
 * positions or pitch transpositions.
 */
struct GroupReference
{
    std::string groupId;       ///< UUID of the referenced NoteGroup.
    Blick timeOffset = 0;      ///< Timeline start offset on the track (in blicks).
    int pitchOffset = 0;       ///< Semitone transposition applied to all notes in the group.
    Blick absoluteBegin = 0;   ///< Left clip boundary (in blicks relative to track).
    Blick absoluteEnd = -1;    ///< Right clip boundary (-1 means unclipped / infinite).
    bool isInstrumental = false; ///< True if this reference plays an audio backing track instead of singing.
    std::string audioFile;     ///< File path if isInstrumental is true.
    double audioDurationSeconds = 0.0; ///< Audio duration if isInstrumental is true.
    std::string preservedFieldsJson;   ///< Retained vendor fields.
    PitchAttributes voicePitch;        ///< Track/reference-level default pitch attributes.
    ParameterCurve systemPitchDelta;   ///< Engine-generated pitch deviation curve.
    bool vocalModeInherited = true;    ///< If true, inherit vocal mode settings from parent track.
    std::string vocalModePreset;       ///< Name of applied vocal mode preset.
    std::map<std::string, double> vocalModeParams; ///< Per-mode weight overrides.
};

/**
 * @brief Voice database and language settings for a track.
 */
struct VoiceSettings
{
    std::string databasePath;       ///< Path to the voice database (.nofs file or directory).
    std::string language = "japanese"; ///< Default language for G2P ("japanese", "english", "mandarin", etc.).
    std::string dictionaryDirectory;///< Optional custom directory containing CLF dictionary files.
};

/**
 * @brief A single musical track in the project.
 *
 * Contains its own inline notes (`mainGroup` and `mainRef`), optional references
 * to library groups (`groups`), audio mixing parameters (gain, pan, mute, solo),
 * and voice database configuration.
 */
struct Track
{
    std::string name = "Track 1";   ///< Track display name.
    NoteGroup mainGroup;            ///< Inline note group directly edited on this track.
    GroupReference mainRef;         ///< Reference positioning mainGroup at offset 0.
    std::vector<GroupReference> groups; ///< Additional reusable note groups placed on this track.
    double gain = 1.0;              ///< Linear volume multiplier (1.0 = 0 dB).
    double pan = 0.0;               ///< Stereo panning (-1.0 = full left, +1.0 = full right, 0.0 = center).
    bool mute = false;              ///< Whether the track is muted.
    bool solo = false;              ///< Whether the track is soloed.
    VoiceSettings voice;            ///< Synthesis voice settings and language for this track.
    std::string preservedFieldsJson;///< Retained vendor fields.
};

/**
 * @brief Tempo marker specifying BPM at a given timeline position.
 */
struct Tempo
{
    Blick position = 0;        ///< Position in blicks where this tempo takes effect.
    double bpm = 120.0;        ///< Beats per minute (must be positive).
    std::string preservedFieldsJson; ///< Retained vendor fields.
};

/**
 * @brief Time signature marker specifying meter at a given measure/bar.
 */
struct TimeSignature
{
    int bar = 0;               ///< 0-indexed measure/bar number.
    int numerator = 4;         ///< Beats per measure (e.g. 4 in 4/4, 3 in 3/4).
    int denominator = 4;       ///< Note value of each beat (e.g. 4 for quarter note, 8 for eighth note).
    std::string preservedFieldsJson; ///< Retained vendor fields.
};

/**
 * @brief Converts between musical time (Blicks) and continuous real time (Seconds).
 *
 * Manages piecewise-constant tempo curves. Each tempo marker defines the BPM
 * from its position until the next tempo marker.
 */
struct TempoMap
{
    std::vector<Tempo> tempos{{0, 120.0, {}}}; ///< Chronologically sorted tempo changes.
    std::vector<TimeSignature> timeSignatures{{0, 4, 4, {}}}; ///< Chronologically sorted time signatures.

    /**
     * @brief Converts a position in musical blicks to elapsed seconds.
     * Integrates across tempo segments from position 0 to the target position.
     */
    [[nodiscard]] double blickToSeconds(Blick position) const;

    /**
     * @brief Converts elapsed wall-clock seconds to musical blicks.
     * Inverts the piecewise tempo integration.
     */
    [[nodiscard]] Blick secondsToBlick(double seconds) const;

    /**
     * @brief Retrieves the active tempo (BPM) at a given musical position.
     */
    [[nodiscard]] double getTempoAt(Blick position) const;
};

/**
 * @brief Root document structure representing an entire Synthesizer V project (.svp).
 */
struct Project
{
    std::string name = "Untitled";     ///< Project name / song title.
    std::vector<Track> tracks;          ///< List of vocal and instrumental tracks.
    std::vector<NoteGroup> library;     ///< Reusable note groups shared across tracks.
    TempoMap tempoMap;                  ///< Project-wide tempo and meter timeline.
    std::string preservedFieldsJson;    ///< Unrecognized vendor fields retained for serialization.
};

/**
 * @brief Thread-safe generator of unique Note IDs for new notes.
 */
[[nodiscard]] NoteId createNoteId();

/**
 * @brief Creates a new NoteGroup with a generated UUID and display name.
 */
[[nodiscard]] NoteGroup createNoteGroup(const std::string& name = "Main");

/**
 * @brief Factory for an initial empty project with one default track.
 */
[[nodiscard]] Project createEmptyProject();

/**
 * @brief Finds a NoteGroup by UUID in either the project library or track main groups.
 * @return Pointer to the note group if found, or nullptr.
 */
[[nodiscard]] const NoteGroup* findNoteGroup(const Project& project, const std::string& groupId);

/**
 * @brief Calculates the latest timeline end point (in blicks) among all notes in the project.
 */
[[nodiscard]] Blick getProjectEnd(const Project& project);

/**
 * @brief Normalises project state to maintain core invariants.
 *
 * Specifically:
 * - Ensures at least one track exists.
 * - Assigns unique nonzero Note IDs to all notes.
 * - Enforces minimum note duration >= 1 blick and MIDI pitch clamped to [0, 127].
 * - Sanitizes note detune to finite floats.
 * - Sorts notes chronologically by onset.
 * - Cleans up and sorts automation curve points (removing duplicates and non-finite values).
 * - Sorts tempo markers and time signatures, ensuring a default at position 0.
 */
void normaliseProject(Project& project);

} // namespace sv
