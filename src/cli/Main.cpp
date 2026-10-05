// opensv-cli: headless SVP -> WAV renderer built on the OpenSVEngine library.
// No GUI, no audio device. Reads a project, renders offline, writes a 24-bit WAV.

#include "audio/ProjectRenderer.h"
#include "audio/WaveFile.h"
#include "core/Project.h"
#include "core/ProjectFile.h"

#include <juce_core/juce_core.h>

#include <atomic>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <string>
#include <vector>

namespace
{
std::atomic<bool> cancelRequested{false};

void onSignal(int)
{
    cancelRequested.store(true);
}

struct Options
{
    std::string input;
    std::string output;
    std::string voiceOverride;
    std::string dictionaryOverride;
    std::string languageOverride;
    double sampleRate = 48000.0;
    bool infoOnly = false;
    bool quiet = false;
};

void printUsage(std::FILE* out)
{
    std::fputs(
        "Usage: opensv-cli <project.svp> -o <out.wav> [options]\n"
        "       opensv-cli <project.svp> --info\n"
        "\n"
        "Options:\n"
        "  -o, --output <file>     Output WAV path (24-bit stereo)\n"
        "      --voice <voice.nofs>  Use this voice database on every track\n"
        "      --dict <clf-data>   Use this pronunciation dictionary directory on every track\n"
        "      --language <name>   Override the singing language on every track (e.g. japanese)\n"
        "      --rate <hz>         Output sample rate (default 48000)\n"
        "      --info              List tracks, voices and whether their files exist, then exit\n"
        "  -q, --quiet             Suppress progress messages\n"
        "  -h, --help              Show this help\n"
        "\n"
        "Voice databases and dictionaries are not bundled; point to files you own.\n"
        "Exit codes: 0 ok, 1 usage error, 2 load error, 3 render/write error, 130 cancelled.\n",
        out);
}

bool parseArgs(int argc, char** argv, Options& options, std::string& error)
{
    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        auto next = [&](std::string& target) -> bool
        {
            if (i + 1 >= argc)
            {
                error = "missing value for " + arg;
                return false;
            }
            target = argv[++i];
            return true;
        };

        if (arg == "-h" || arg == "--help")
        {
            printUsage(stdout);
            std::exit(0);
        }
        else if (arg == "-o" || arg == "--output")
        {
            if (!next(options.output)) return false;
        }
        else if (arg == "--voice")
        {
            if (!next(options.voiceOverride)) return false;
        }
        else if (arg == "--dict")
        {
            if (!next(options.dictionaryOverride)) return false;
        }
        else if (arg == "--language")
        {
            if (!next(options.languageOverride)) return false;
        }
        else if (arg == "--rate")
        {
            std::string value;
            if (!next(value)) return false;
            options.sampleRate = std::atof(value.c_str());
            if (options.sampleRate < 8000.0 || options.sampleRate > 192000.0)
            {
                error = "--rate must be between 8000 and 192000";
                return false;
            }
        }
        else if (arg == "--info")
        {
            options.infoOnly = true;
        }
        else if (arg == "-q" || arg == "--quiet")
        {
            options.quiet = true;
        }
        else if (!arg.empty() && arg[0] == '-')
        {
            error = "unknown option " + arg;
            return false;
        }
        else if (options.input.empty())
        {
            options.input = arg;
        }
        else
        {
            error = "unexpected extra argument " + arg;
            return false;
        }
    }

    if (options.input.empty())
    {
        error = "no input project given";
        return false;
    }
    if (!options.infoOnly && options.output.empty())
    {
        error = "no output file given (use -o)";
        return false;
    }
    return true;
}

juce::File resolve(const std::string& path)
{
    return juce::File::getCurrentWorkingDirectory().getChildFile(juce::String::fromUTF8(path.c_str()));
}

void printInfo(const sv::Project& project)
{
    std::printf("Project: %s\n", project.name.c_str());
    std::printf("Tracks: %d\n", static_cast<int>(project.tracks.size()));
    int index = 0;
    for (const auto& track : project.tracks)
    {
        const bool hasVoice = !track.voice.databasePath.empty();
        const bool voiceExists = hasVoice && juce::File(juce::String::fromUTF8(track.voice.databasePath.c_str())).existsAsFile();
        const bool hasDict = !track.voice.dictionaryDirectory.empty();
        const bool dictExists = hasDict && juce::File(juce::String::fromUTF8(track.voice.dictionaryDirectory.c_str())).isDirectory();
        std::printf("  [%d] %s  notes=%d  language=%s\n", index, track.name.c_str(), static_cast<int>(track.mainGroup.notes.size()), track.voice.language.c_str());
        std::printf("      voice: %s%s\n", hasVoice ? track.voice.databasePath.c_str() : "(none)", hasVoice ? (voiceExists ? "" : "  [MISSING]") : "");
        std::printf("      dict:  %s%s\n", hasDict ? track.voice.dictionaryDirectory.c_str() : "(none)", hasDict ? (dictExists ? "" : "  [MISSING]") : "");
        ++index;
    }
}
} // namespace

int main(int argc, char** argv)
{
    Options options;
    std::string error;
    if (!parseArgs(argc, argv, options, error))
    {
        std::fprintf(stderr, "opensv-cli: %s\n\n", error.c_str());
        printUsage(stderr);
        return 1;
    }

    std::signal(SIGINT, onSignal);
    std::signal(SIGTERM, onSignal);

    try
    {
        sv::Project project;
        const auto inputFile = resolve(options.input);
        if (const auto loaded = sv::loadProjectFile(inputFile, project); loaded.failed())
        {
            std::fprintf(stderr, "opensv-cli: %s\n", loaded.getErrorMessage().toRawUTF8());
            return 2;
        }

        for (auto& track : project.tracks)
        {
            if (!options.voiceOverride.empty())
            {
                track.voice.databasePath = resolve(options.voiceOverride).getFullPathName().toStdString();
            }
            if (!options.dictionaryOverride.empty())
            {
                track.voice.dictionaryDirectory = resolve(options.dictionaryOverride).getFullPathName().toStdString();
            }
            if (!options.languageOverride.empty())
            {
                track.voice.language = options.languageOverride;
            }
        }

        if (options.infoOnly)
        {
            printInfo(project);
            return 0;
        }

        if (!options.quiet)
        {
            std::fprintf(stderr, "Rendering %s (%d tracks) at %.0f Hz...\n", options.input.c_str(), static_cast<int>(project.tracks.size()), options.sampleRate);
        }

        sv::audio::ProjectRenderer renderer;
        juce::AudioBuffer<float> samples;
        const auto shouldCancel = []
        { return cancelRequested.load(); };

        const auto rendered = renderer.render(project, options.sampleRate, samples, shouldCancel);
        if (cancelRequested.load())
        {
            std::fprintf(stderr, "opensv-cli: cancelled\n");
            return 130;
        }
        if (rendered.failed())
        {
            std::fprintf(stderr, "opensv-cli: render failed: %s\n", rendered.getErrorMessage().toRawUTF8());
            return 3;
        }

        const auto outputFile = resolve(options.output);
        const auto written = sv::audio::writeWaveFile(outputFile, samples, options.sampleRate, shouldCancel);
        if (cancelRequested.load())
        {
            std::fprintf(stderr, "opensv-cli: cancelled\n");
            return 130;
        }
        if (written.failed())
        {
            std::fprintf(stderr, "opensv-cli: could not write %s: %s\n", options.output.c_str(), written.getErrorMessage().toRawUTF8());
            return 3;
        }

        if (!options.quiet)
        {
            std::fprintf(stderr, "Wrote %s (%.2f s)\n", options.output.c_str(), static_cast<double>(samples.getNumSamples()) / options.sampleRate);
        }
        return 0;
    }
    catch (const std::exception& exception)
    {
        std::fprintf(stderr, "opensv-cli: unexpected error: %s\n", exception.what());
        return 3;
    }
    catch (...)
    {
        std::fprintf(stderr, "opensv-cli: unexpected error\n");
        return 3;
    }
}
