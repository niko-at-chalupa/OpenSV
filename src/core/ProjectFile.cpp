#include "ProjectFile.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <initializer_list>
#include <limits>
#include <optional>
#include <unordered_set>

namespace sv
{
namespace
{
constexpr int projectFormatVersion = 153;
constexpr Blick maximumFileBlick = std::numeric_limits<Blick>::max() / 4;

struct PitchAttributeField
{
    const char* name;
    std::optional<double> PitchAttributes::* member;
};

constexpr std::array pitchAttributeFields{
    PitchAttributeField{"tF0Offset", &PitchAttributes::tF0Offset},
    PitchAttributeField{"tF0Left", &PitchAttributes::tF0Left},
    PitchAttributeField{"tF0Right", &PitchAttributes::tF0Right},
    PitchAttributeField{"dF0Left", &PitchAttributes::dF0Left},
    PitchAttributeField{"dF0Right", &PitchAttributes::dF0Right},
    PitchAttributeField{"tF0VbrStart", &PitchAttributes::tF0VbrStart},
    PitchAttributeField{"tF0VbrLeft", &PitchAttributes::tF0VbrLeft},
    PitchAttributeField{"tF0VbrRight", &PitchAttributes::tF0VbrRight},
    PitchAttributeField{"dF0Vbr", &PitchAttributes::dF0Vbr},
    PitchAttributeField{"pF0Vbr", &PitchAttributes::pF0Vbr},
    PitchAttributeField{"fF0Vbr", &PitchAttributes::fF0Vbr},
    PitchAttributeField{"dF0VbrMod", &PitchAttributes::dF0VbrMod},
    PitchAttributeField{"rTone", &PitchAttributes::rTone},
    PitchAttributeField{"rIntonation", &PitchAttributes::rIntonation}};

juce::var makeObject()
{
    return juce::var(new juce::DynamicObject());
}

juce::String utf8(const std::string& text)
{
    return juce::String::fromUTF8(text.data(), static_cast<int>(text.size()));
}

juce::var retainedObject(const std::string& json)
{
    if (!json.empty())
    {
        auto value = juce::JSON::parse(utf8(json));
        if (value.isObject())
        {
            return value;
        }
    }
    return makeObject();
}

juce::var objectProperty(const juce::var& parent, const juce::Identifier& name)
{
    const auto& value = parent[name];
    return value.isObject() ? value.clone() : makeObject();
}

void set(juce::var& object, const juce::Identifier& name, const juce::var& value)
{
    object.getDynamicObject()->setProperty(name, value);
}

void setDefault(juce::var& object, const juce::Identifier& name, const juce::var& value)
{
    if (!object.hasProperty(name))
    {
        set(object, name, value);
    }
}

std::string preserve(const juce::var& original, std::initializer_list<const char*> fields)
{
    auto extra = original.clone();
    for (const auto* field : fields)
    {
        extra.getDynamicObject()->removeProperty(juce::Identifier(field));
    }
    return juce::JSON::toString(extra, true, std::numeric_limits<double>::max_digits10).toStdString();
}

bool isNumber(const juce::var& value)
{
    return value.isInt() || value.isInt64() || value.isDouble();
}

bool readInteger(const juce::var& object, const juce::Identifier& key, Blick& result)
{
    const auto& value = object[key];
    if (!isNumber(value))
    {
        return false;
    }
    const double number = static_cast<double>(value);
    if (!std::isfinite(number) || std::abs(number) > static_cast<double>(maximumFileBlick) || std::trunc(number) != number)
    {
        return false;
    }
    result = static_cast<juce::int64>(value);
    return true;
}

bool readNumber(const juce::var& object, const juce::Identifier& key, double& result)
{
    const auto& value = object[key];
    if (!isNumber(value))
    {
        return false;
    }
    result = static_cast<double>(value);
    return std::isfinite(result);
}

juce::Result invalid(const juce::String& context)
{
    return juce::Result::fail("Invalid Synthesizer V project: " + context);
}

juce::Result readPitchAttributes(const juce::var& value, PitchAttributes& attributes, const juce::String& context)
{
    if (value.isVoid())
    {
        return juce::Result::ok();
    }
    if (!value.isObject())
    {
        return invalid(context + " must be an object.");
    }
    auto extra = value.clone();
    for (const auto& field : pitchAttributeFields)
    {
        if (value.hasProperty(field.name))
        {
            double amount = 0.0;
            if (!readNumber(value, field.name, amount))
            {
                return invalid(context + "." + field.name + " must be a finite number.");
            }
            attributes.*field.member = amount;
            extra.getDynamicObject()->removeProperty(field.name);
        }
    }
    extra.getDynamicObject()->removeProperty("vocalModeInherited");
    extra.getDynamicObject()->removeProperty("vocalModePreset");
    extra.getDynamicObject()->removeProperty("vocalModeParams");
    attributes.preservedFieldsJson = juce::JSON::toString(extra, true, std::numeric_limits<double>::max_digits10).toStdString();
    return juce::Result::ok();
}

juce::Result readCurve(const juce::var& value, ParameterCurve& curve, const juce::String& name)
{
    if (value.isVoid())
    {
        return juce::Result::ok();
    }
    if (!value.isObject() || !value["mode"].isString() || !value["points"].isArray())
    {
        return invalid(name + " must contain a mode and points array.");
    }
    curve.mode = value["mode"].toString().toStdString();
    if (curve.mode != "linear" && curve.mode != "cubic" && curve.mode != "cosine")
    {
        return invalid(name + " has an unsupported interpolation mode.");
    }
    const auto& points = *value["points"].getArray();
    if (points.size() % 2 != 0)
    {
        return invalid(name + " points must contain time/value pairs.");
    }
    for (int i = 0; i < points.size(); i += 2)
    {
        const auto& position = points[i];
        const auto& amount = points[i + 1];
        if (!isNumber(position) || !isNumber(amount))
        {
            return invalid(name + " contains non-numeric points.");
        }
        const double positionNumber = static_cast<double>(position);
        const double valueNumber = static_cast<double>(amount);
        if (!std::isfinite(positionNumber) || std::trunc(positionNumber) != positionNumber || std::abs(positionNumber) > static_cast<double>(maximumFileBlick) || !std::isfinite(valueNumber))
        {
            return invalid(name + " contains an invalid point.");
        }
        curve.points.push_back({static_cast<juce::int64>(position), valueNumber});
    }
    curve.preservedFieldsJson = preserve(value, {"mode", "points"});
    return juce::Result::ok();
}

juce::Result readGroup(const juce::var& value, NoteGroup& group)
{
    if (!value.isObject() || !value["uuid"].isString() || value["uuid"].toString().isEmpty() || !value["notes"].isArray())
    {
        return invalid("a note group needs a UUID and notes array.");
    }
    group.id = value["uuid"].toString().toStdString();
    group.name = value["name"].toString().toStdString();
    auto extra = value.clone();
    auto parameters = objectProperty(extra, "parameters");
    parameters.getDynamicObject()->removeProperty("pitchDelta");
    parameters.getDynamicObject()->removeProperty("vibratoEnv");
    set(extra, "parameters", parameters);
    group.preservedFieldsJson = preserve(extra, {"uuid", "name", "notes", "vocalModes"});
    for (const auto& item : *value["notes"].getArray())
    {
        Note note;
        Blick pitch = 0;
        if (!item.isObject() || !readInteger(item, "onset", note.onset) || !readInteger(item, "duration", note.duration) || note.duration <= 0 || !readInteger(item, "pitch", pitch) || pitch < 0 || pitch > 127 || !item["lyrics"].isString())
        {
            return invalid("a note has invalid timing, pitch, or lyrics.");
        }
        note.id = createNoteId();
        note.pitch = static_cast<int>(pitch);
        note.lyrics = item["lyrics"].toString().toStdString();
        note.phonemes = item["phonemes"].toString().toStdString();
        if (item.hasProperty("accent") && !item["accent"].isString())
        {
            return invalid("a note's accent must be a string.");
        }
        note.accent = item["accent"].toString().toStdString();
        if (item.hasProperty("detune") && !readNumber(item, "detune", note.detune))
        {
            return invalid("a note has invalid detune.");
        }
        if (item.hasProperty("instantMode"))
        {
            if (!item["instantMode"].isBool())
            {
                return invalid("a note's instantMode must be a boolean.");
            }
            note.instantMode = static_cast<bool>(item["instantMode"]);
        }
        if (item.hasProperty("musicalType"))
        {
            if (!item["musicalType"].isString())
            {
                return invalid("a note's musicalType must be a string.");
            }
            note.musicalType = item["musicalType"].toString().toStdString();
        }
        if (const auto result = readPitchAttributes(item["attributes"], note.attributes, "note attributes"); result.failed())
        {
            return result;
        }
        if (const auto result = readPitchAttributes(item["systemAttributes"], note.systemAttributes, "note systemAttributes"); result.failed())
        {
            return result;
        }
        note.preservedFieldsJson = preserve(item, {"onset", "duration", "pitch", "lyrics", "phonemes", "accent", "detune", "instantMode", "musicalType", "attributes", "systemAttributes"});
        group.notes.push_back(std::move(note));
    }
    if (const auto result = readCurve(value["parameters"]["pitchDelta"], group.pitchDelta, "pitchDelta"); result.failed())
    {
        return result;
    }
    if (const auto result = readCurve(value["parameters"]["vibratoEnv"], group.vibratoEnv, "vibratoEnv"); result.failed())
    {
        return result;
    }
    const auto& modes = value["vocalModes"];
    if (!modes.isVoid() && !modes.isObject())
    {
        return invalid("vocalModes must be an object of mode curves.");
    }
    if (const auto* object = modes.getDynamicObject())
    {
        const auto& properties = object->getProperties();
        for (int index = 0; index < properties.size(); ++index)
        {
            const auto name = properties.getName(index).toString().toStdString();
            const auto& modeValue = properties.getValueAt(index);
            VocalModeSetting setting;
            if (isNumber(modeValue))
            {
                if (!readNumber(modes, properties.getName(index), setting.amount))
                {
                    return invalid("vocalModes." + utf8(name) + " must have a finite amount.");
                }
            }
            else if (modeValue.isObject())
            {
                if (modeValue.hasProperty("enabled"))
                {
                    if (!modeValue["enabled"].isBool())
                    {
                        return invalid("vocalModes." + utf8(name) + ".enabled must be a boolean.");
                    }
                    setting.enabled = static_cast<bool>(modeValue["enabled"]);
                }
                if (modeValue.hasProperty("amount") && !readNumber(modeValue, "amount", setting.amount))
                {
                    return invalid("vocalModes." + utf8(name) + ".amount must be a finite number.");
                }
                if (modeValue.hasProperty("mode") || modeValue.hasProperty("points"))
                {
                    if (const auto result = readCurve(modeValue, setting.curve, "vocalModes." + utf8(name)); result.failed())
                    {
                        return result;
                    }
                    setting.hasCurve = true;
                }
                setting.preservedFieldsJson = preserve(modeValue, {"enabled", "amount", "mode", "points"});
            }
            else
            {
                return invalid("vocalModes." + utf8(name) + " must be an amount or settings object.");
            }
            group.vocalModes.emplace(name, std::move(setting));
        }
    }
    return juce::Result::ok();
}

juce::Result readReference(const juce::var& value, GroupReference& reference, const juce::File& directory)
{
    Blick pitchOffset = 0;
    if (!value.isObject() || !value["groupID"].isString() || !readInteger(value, "blickOffset", reference.timeOffset) || !readInteger(value, "pitchOffset", pitchOffset) || pitchOffset < -127 || pitchOffset > 127)
    {
        return invalid("a group reference has invalid identity or offsets.");
    }
    reference.groupId = value["groupID"].toString().toStdString();
    reference.pitchOffset = static_cast<int>(pitchOffset);
    if ((value.hasProperty("blickAbsoluteBegin") && !readInteger(value, "blickAbsoluteBegin", reference.absoluteBegin)) || (value.hasProperty("blickAbsoluteEnd") && !readInteger(value, "blickAbsoluteEnd", reference.absoluteEnd)))
    {
        return invalid("a group reference has invalid crop boundaries.");
    }
    if (reference.absoluteEnd != -1 && reference.absoluteEnd <= reference.absoluteBegin)
    {
        return invalid("a group reference ends before its beginning.");
    }
    reference.isInstrumental = static_cast<bool>(value["isInstrumental"]);
    const auto& audio = value["audio"];
    if (!audio.isVoid())
    {
        if (!audio.isObject() || !audio["filename"].isString() || !readNumber(audio, "duration", reference.audioDurationSeconds) || reference.audioDurationSeconds < 0.0)
        {
            return invalid("an instrumental reference has invalid audio metadata.");
        }
        const auto filename = audio["filename"].toString();
        if (filename.isNotEmpty())
        {
            reference.audioFile = directory.getChildFile(filename).getFullPathName().toStdString();
        }
    }
    const auto& voice = value["voice"];
    if (!voice.isVoid() && !voice.isObject())
    {
        return invalid("a group reference's voice settings must be an object.");
    }
    if (voice.hasProperty("vocalModeInherited"))
    {
        if (!voice["vocalModeInherited"].isBool())
        {
            return invalid("vocalModeInherited must be a boolean.");
        }
        reference.vocalModeInherited = static_cast<bool>(voice["vocalModeInherited"]);
    }
    if (voice.hasProperty("vocalModePreset"))
    {
        if (!voice["vocalModePreset"].isString())
        {
            return invalid("vocalModePreset must be a string.");
        }
        reference.vocalModePreset = voice["vocalModePreset"].toString().toStdString();
    }
    const auto& modeParams = voice["vocalModeParams"];
    if (!modeParams.isVoid() && !modeParams.isObject())
    {
        return invalid("vocalModeParams must be an object of mode amounts.");
    }
    if (const auto* object = modeParams.getDynamicObject())
    {
        const auto& properties = object->getProperties();
        for (int index = 0; index < properties.size(); ++index)
        {
            double amount = 0.0;
            const auto name = properties.getName(index).toString().toStdString();
            if (!readNumber(modeParams, properties.getName(index), amount))
            {
                return invalid("vocalModeParams." + utf8(name) + " must be a finite number.");
            }
            reference.vocalModeParams.emplace(name, amount);
        }
    }
    if (const auto result = readPitchAttributes(voice, reference.voicePitch, "reference voice"); result.failed())
    {
        return result;
    }
    if (const auto result = readCurve(value["systemPitchDelta"], reference.systemPitchDelta, "systemPitchDelta"); result.failed())
    {
        return result;
    }
    reference.preservedFieldsJson = preserve(value, {"groupID", "blickOffset", "pitchOffset", "blickAbsoluteBegin", "blickAbsoluteEnd", "isInstrumental", "voice", "systemPitchDelta"});
    return juce::Result::ok();
}

juce::Result readTrack(const juce::var& value, Track& track, const juce::File& directory)
{
    if (!value.isObject() || !value["groups"].isArray())
    {
        return invalid("a track needs a groups array.");
    }
    track.name = value["name"].toString().toStdString();
    track.preservedFieldsJson = preserve(value, {"name", "mainGroup", "mainRef", "groups", "svVoice"});
    const auto& voice = value["svVoice"];
    if (!voice.isVoid())
    {
        if (!voice.isObject() || !voice["databasePath"].isString() || !voice["language"].isString() || !voice["dictionaryDirectory"].isString())
        {
            return invalid("svVoice requires databasePath, language and dictionaryDirectory strings.");
        }
        const auto databasePath = voice["databasePath"].toString();
        const auto dictionaryDirectory = voice["dictionaryDirectory"].toString();
        track.voice.databasePath = databasePath.isEmpty() ? std::string{} : directory.getChildFile(databasePath).getFullPathName().toStdString();
        track.voice.dictionaryDirectory = dictionaryDirectory.isEmpty() ? std::string{} : directory.getChildFile(dictionaryDirectory).getFullPathName().toStdString();
        track.voice.language = voice["language"].toString().toStdString();
        if (track.voice.language != "japanese" && track.voice.language != "mandarin" && track.voice.language != "english" && track.voice.language != "cantonese" && track.voice.language != "spanish")
        {
            return invalid("svVoice has an unsupported language.");
        }
    }
    if (const auto result = readGroup(value["mainGroup"], track.mainGroup); result.failed())
    {
        return result;
    }
    if (const auto result = readReference(value["mainRef"], track.mainRef, directory); result.failed())
    {
        return result;
    }
    if (track.mainRef.groupId != track.mainGroup.id)
    {
        return invalid("a track's mainRef does not refer to its mainGroup.");
    }
    const auto& mixer = value["mixer"];
    double gainDecibel = 0.0;
    if (!mixer.isObject() || !readNumber(mixer, "gainDecibel", gainDecibel) || !readNumber(mixer, "pan", track.pan) || track.pan < -1.0 || track.pan > 1.0)
    {
        return invalid("a track has invalid mixer settings.");
    }
    track.gain = std::pow(10.0, gainDecibel / 20.0);
    if (!std::isfinite(track.gain))
    {
        return invalid("a track's gain is out of range.");
    }
    track.mute = static_cast<bool>(mixer["mute"]);
    track.solo = static_cast<bool>(mixer["solo"]);
    for (const auto& item : *value["groups"].getArray())
    {
        GroupReference reference;
        if (const auto result = readReference(item, reference, directory); result.failed())
        {
            return result;
        }
        track.groups.push_back(std::move(reference));
    }
    return juce::Result::ok();
}

juce::var defaultTakes()
{
    auto take = makeObject();
    set(take, "id", 0);
    set(take, "expr", 0.0);
    set(take, "liked", false);
    auto value = makeObject();
    set(value, "activeTakeId", 0);
    set(value, "takes", juce::Array<juce::var>{take});
    return value;
}

juce::var writePitchAttributes(const PitchAttributes& attributes)
{
    auto value = retainedObject(attributes.preservedFieldsJson);
    for (const auto& field : pitchAttributeFields)
    {
        if (const auto& amount = attributes.*field.member)
        {
            set(value, field.name, *amount);
        }
        else
        {
            value.getDynamicObject()->removeProperty(field.name);
        }
    }
    return value;
}

juce::var writeCurve(const ParameterCurve& curve)
{
    auto value = retainedObject(curve.preservedFieldsJson);
    set(value, "mode", utf8(curve.mode));
    juce::Array<juce::var> points;
    for (const auto& point : curve.points)
    {
        points.add(static_cast<juce::int64>(point.position));
        points.add(point.value);
    }
    set(value, "points", points);
    return value;
}

juce::var writeGroup(const NoteGroup& group)
{
    auto value = retainedObject(group.preservedFieldsJson);
    set(value, "uuid", utf8(group.id));
    set(value, "name", utf8(group.name));
    juce::Array<juce::var> notes;
    for (const auto& note : group.notes)
    {
        auto item = retainedObject(note.preservedFieldsJson);
        set(item, "onset", static_cast<juce::int64>(note.onset));
        set(item, "duration", static_cast<juce::int64>(note.duration));
        set(item, "pitch", note.pitch);
        set(item, "lyrics", utf8(note.lyrics));
        set(item, "phonemes", utf8(note.phonemes));
        set(item, "musicalType", utf8(note.musicalType));
        set(item, "accent", utf8(note.accent));
        set(item, "detune", note.detune);
        set(item, "instantMode", note.instantMode);
        auto attributes = writePitchAttributes(note.attributes);
        setDefault(attributes, "evenSyllableDuration", true);
        set(item, "attributes", attributes);
        auto systemAttributes = writePitchAttributes(note.systemAttributes);
        if (systemAttributes.getDynamicObject()->getProperties().size() > 0)
        {
            set(item, "systemAttributes", systemAttributes);
        }
        else
        {
            item.getDynamicObject()->removeProperty("systemAttributes");
        }
        setDefault(item, "pitchTakes", defaultTakes());
        setDefault(item, "timbreTakes", defaultTakes());
        notes.add(std::move(item));
    }
    set(value, "notes", notes);
    auto parameters = objectProperty(value, "parameters");
    for (const auto* name : {"loudness", "tension", "breathiness", "voicing", "gender", "toneShift"})
    {
        auto curve = objectProperty(parameters, name);
        setDefault(curve, "mode", "cubic");
        setDefault(curve, "points", juce::Array<juce::var>());
        set(parameters, name, curve);
    }
    set(parameters, "pitchDelta", writeCurve(group.pitchDelta));
    set(parameters, "vibratoEnv", writeCurve(group.vibratoEnv));
    set(value, "parameters", parameters);
    auto vocalModes = makeObject();
    for (const auto& [name, setting] : group.vocalModes)
    {
        auto mode = retainedObject(setting.preservedFieldsJson);
        set(mode, "enabled", setting.enabled);
        set(mode, "amount", setting.amount);
        if (setting.hasCurve)
        {
            auto curve = writeCurve(setting.curve);
            set(mode, "mode", curve["mode"]);
            set(mode, "points", curve["points"]);
        }
        set(vocalModes, juce::Identifier(utf8(name)), mode);
    }
    set(value, "vocalModes", vocalModes);
    return value;
}

juce::var writeReference(const GroupReference& reference, const juce::File& directory)
{
    auto value = retainedObject(reference.preservedFieldsJson);
    set(value, "groupID", utf8(reference.groupId));
    set(value, "blickOffset", static_cast<juce::int64>(reference.timeOffset));
    set(value, "pitchOffset", reference.pitchOffset);
    set(value, "blickAbsoluteBegin", static_cast<juce::int64>(reference.absoluteBegin));
    set(value, "blickAbsoluteEnd", static_cast<juce::int64>(reference.absoluteEnd));
    set(value, "isInstrumental", reference.isInstrumental);
    auto database = objectProperty(value, "database");
    for (const auto* name : {"name", "language", "phoneset", "languageOverride", "phonesetOverride", "backendType"})
    {
        setDefault(database, name, "");
    }
    setDefault(database, "version", "-2");
    set(value, "database", database);
    setDefault(value, "dictionary", "");
    auto voice = writePitchAttributes(reference.voicePitch);
    set(voice, "vocalModeInherited", reference.vocalModeInherited);
    set(voice, "vocalModePreset", utf8(reference.vocalModePreset));
    auto modeParams = makeObject();
    for (const auto& [name, amount] : reference.vocalModeParams)
    {
        set(modeParams, juce::Identifier(utf8(name)), amount);
    }
    set(voice, "vocalModeParams", modeParams);
    set(value, "voice", voice);
    setDefault(value, "pitchTakes", defaultTakes());
    setDefault(value, "timbreTakes", defaultTakes());
    set(value, "systemPitchDelta", writeCurve(reference.systemPitchDelta));
    if (!reference.audioFile.empty())
    {
        auto audio = objectProperty(value, "audio");
        set(audio, "filename", juce::File(utf8(reference.audioFile)).getRelativePathFrom(directory));
        set(audio, "duration", reference.audioDurationSeconds);
        set(value, "audio", audio);
    }
    return value;
}

juce::var writeTrack(const Track& track, int index, const juce::File& directory)
{
    auto value = retainedObject(track.preservedFieldsJson);
    set(value, "name", utf8(track.name));
    setDefault(value, "dispColor", "ff7db235");
    set(value, "dispOrder", index);
    setDefault(value, "renderEnabled", false);
    set(value, "mainGroup", writeGroup(track.mainGroup));
    set(value, "mainRef", writeReference(track.mainRef, directory));
    if (!track.voice.databasePath.empty() || !track.voice.dictionaryDirectory.empty() || track.voice.language != "japanese")
    {
        auto voice = makeObject();
        set(voice, "databasePath", track.voice.databasePath.empty() ? juce::String() : juce::File(utf8(track.voice.databasePath)).getRelativePathFrom(directory));
        set(voice, "language", utf8(track.voice.language));
        set(voice, "dictionaryDirectory", track.voice.dictionaryDirectory.empty() ? juce::String() : juce::File(utf8(track.voice.dictionaryDirectory)).getRelativePathFrom(directory));
        set(value, "svVoice", voice);
    }
    auto mixer = objectProperty(value, "mixer");
    const double gainDecibel = track.gain > 0.0 ? 20.0 * std::log10(track.gain) : -100.0;
    // Preserve the source dB value when no linear gain edit has occurred.
    const double originalDb = static_cast<double>(mixer["gainDecibel"]);
    if (!mixer.hasProperty("gainDecibel") || std::pow(10.0, originalDb / 20.0) != track.gain)
    {
        set(mixer, "gainDecibel", gainDecibel);
    }
    set(mixer, "pan", track.pan);
    set(mixer, "mute", track.mute);
    set(mixer, "solo", track.solo);
    setDefault(mixer, "display", true);
    set(value, "mixer", mixer);
    juce::Array<juce::var> groups;
    for (const auto& reference : track.groups)
    {
        groups.add(writeReference(reference, directory));
    }
    set(value, "groups", groups);
    return value;
}
} // namespace

juce::Result loadProjectFile(const juce::File& file, Project& project)
{
    if (file.getSize() > std::numeric_limits<int>::max())
    {
        return invalid("the file exceeds the supported size.");
    }
    juce::MemoryBlock bytes;
    if (!file.loadFileAsData(bytes))
    {
        return juce::Result::fail("Could not read project: " + file.getFullPathName());
    }
    auto size = bytes.getSize();
    if (size == 0)
    {
        return invalid("the file is empty.");
    }
    const auto* data = static_cast<const char*>(bytes.getData());
    while (size > 0 && data[size - 1] == '\0')
    {
        --size;
    }
    if (size > static_cast<std::size_t>(std::numeric_limits<int>::max()) || std::find(data, data + size, '\0') != data + size)
    {
        return invalid("the JSON contains an embedded NUL or exceeds the supported size.");
    }
    juce::var root;
    const auto parsed = juce::JSON::parse(juce::String::fromUTF8(data, static_cast<int>(size)), root);
    if (parsed.failed())
    {
        return juce::Result::fail("Could not parse project JSON: " + parsed.getErrorMessage());
    }
    if (!root.isObject() || static_cast<int>(root["version"]) != projectFormatVersion)
    {
        return juce::Result::fail("Only Synthesizer V Studio 1.11.2 projects (format version 153) are supported.");
    }
    if (!root["tracks"].isArray() || !root["library"].isArray() || !root["time"].isObject() || !root["time"]["tempo"].isArray() || !root["time"]["meter"].isArray())
    {
        return invalid("tracks, library, tempo, or meter is missing.");
    }
    Project loaded;
    loaded.name = file.getFileNameWithoutExtension().toStdString();
    loaded.preservedFieldsJson = preserve(root, {"tracks", "library"});
    for (const auto& value : *root["library"].getArray())
    {
        NoteGroup group;
        if (const auto result = readGroup(value, group); result.failed())
        {
            return result;
        }
        loaded.library.push_back(std::move(group));
    }
    for (const auto& value : *root["tracks"].getArray())
    {
        Track track;
        if (const auto result = readTrack(value, track, file.getParentDirectory()); result.failed())
        {
            return result;
        }
        loaded.tracks.push_back(std::move(track));
    }
    loaded.tempoMap.tempos.clear();
    for (const auto& value : *root["time"]["tempo"].getArray())
    {
        Tempo tempo;
        if (!value.isObject() || !readInteger(value, "position", tempo.position) || tempo.position < 0 || !readNumber(value, "bpm", tempo.bpm) || tempo.bpm <= 0.0)
        {
            return invalid("a tempo marker is invalid.");
        }
        tempo.preservedFieldsJson = preserve(value, {"position", "bpm"});
        loaded.tempoMap.tempos.push_back(std::move(tempo));
    }
    loaded.tempoMap.timeSignatures.clear();
    for (const auto& value : *root["time"]["meter"].getArray())
    {
        Blick bar = 0;
        Blick numerator = 0;
        Blick denominator = 0;
        if (!value.isObject() || !readInteger(value, "index", bar) || bar < 0 || bar > std::numeric_limits<int>::max() || !readInteger(value, "numerator", numerator) || numerator < 1 || numerator > 256 || !readInteger(value, "denominator", denominator) || denominator < 1 || denominator > 256 || (denominator & (denominator - 1)) != 0)
        {
            return invalid("a meter marker is invalid.");
        }
        loaded.tempoMap.timeSignatures.push_back({static_cast<int>(bar), static_cast<int>(numerator), static_cast<int>(denominator), preserve(value, {"index", "numerator", "denominator"})});
    }
    std::unordered_set<std::string> groupIds;
    for (const auto& group : loaded.library)
    {
        if (!groupIds.insert(group.id).second)
        {
            return invalid("duplicate group UUIDs.");
        }
    }
    for (const auto& track : loaded.tracks)
    {
        if (!groupIds.insert(track.mainGroup.id).second)
        {
            return invalid("duplicate main group UUIDs.");
        }
    }
    for (const auto& track : loaded.tracks)
    {
        for (const auto& reference : track.groups)
        {
            if (findNoteGroup(loaded, reference.groupId) == nullptr && !reference.isInstrumental)
            {
                return invalid("a group reference points to a missing group.");
            }
        }
    }
    normaliseProject(loaded);
    project = std::move(loaded);
    return juce::Result::ok();
}

juce::Result saveProjectFile(const juce::File& file, const Project& project)
{
    auto root = retainedObject(project.preservedFieldsJson);
    set(root, "version", projectFormatVersion);
    auto time = objectProperty(root, "time");
    juce::Array<juce::var> tempos;
    for (const auto& tempo : project.tempoMap.tempos)
    {
        auto value = retainedObject(tempo.preservedFieldsJson);
        set(value, "position", static_cast<juce::int64>(tempo.position));
        set(value, "bpm", tempo.bpm);
        tempos.add(std::move(value));
    }
    set(time, "tempo", tempos);
    juce::Array<juce::var> meter;
    for (const auto& signature : project.tempoMap.timeSignatures)
    {
        auto value = retainedObject(signature.preservedFieldsJson);
        set(value, "index", signature.bar);
        set(value, "numerator", signature.numerator);
        set(value, "denominator", signature.denominator);
        meter.add(std::move(value));
    }
    set(time, "meter", meter);
    set(root, "time", time);
    juce::Array<juce::var> library;
    for (const auto& group : project.library)
    {
        library.add(writeGroup(group));
    }
    set(root, "library", library);
    juce::Array<juce::var> tracks;
    for (std::size_t i = 0; i < project.tracks.size(); ++i)
    {
        tracks.add(writeTrack(project.tracks[i], static_cast<int>(i), file.getParentDirectory()));
    }
    set(root, "tracks", tracks);
    auto render = objectProperty(root, "renderConfig");
    setDefault(render, "destination", "");
    setDefault(render, "filename", utf8(project.name));
    setDefault(render, "numChannels", 1);
    setDefault(render, "aspirationFormat", "noAspiration");
    setDefault(render, "bitDepth", 16);
    setDefault(render, "sampleRate", 44100);
    setDefault(render, "exportMixDown", true);
    setDefault(render, "exportPitch", false);
    set(root, "renderConfig", render);
    const auto json = juce::JSON::toString(root, false, std::numeric_limits<double>::max_digits10);
    juce::TemporaryFile temporary(file);
    {
        auto output = temporary.getFile().createOutputStream();
        if (output == nullptr || !output->write(json.toRawUTF8(), json.getNumBytesAsUTF8()) || !output->writeByte(0))
        {
            return juce::Result::fail("Could not write project: " + file.getFullPathName());
        }
        output->flush();
        if (output->getStatus().failed())
        {
            return output->getStatus();
        }
    }
    if (!temporary.overwriteTargetFileWithTemporary())
    {
        return juce::Result::fail("Could not replace project: " + file.getFullPathName());
    }
    return juce::Result::ok();
}
} // namespace sv
