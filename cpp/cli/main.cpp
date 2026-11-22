// VideoMetricsLab from a terminal, in C++: the Python app's command line
// (vmaf_app/cli.py), as far as it has been ported (compare.h says how far).
//
//     vml-cli compare REFERENCE TEST [TEST ...] [options]
//     vml-cli bitrate VIDEO [VIDEO ...] [options]
//     vml-cli bdrate --anchor FILE --test FILE [options]
//     vml-cli devices
//     vml-cli --version
//
// Progress goes to stderr, results to stdout. The exit codes (arguments.h):
// 0 when every video got every metric asked for that applies to it (or was
// read, for bitrate; the files gave BD-rates, for bdrate), 1 when one failed
// or a file could not be written, 2 for a wrong command (or one this version
// cannot do yet), 3 when a quality gate is not met, 4 when an input
// could not be read, 130 when cancelled with Ctrl+C.
#include <csignal>
#include <exception>
#include <string>
#include <vector>

#include "arguments.h"
#include "bdrate.h"
#include "bitrate.h"
#include "codes.h"
#include "common.h"
#include "compare.h"
#include "devices.h"
#include "loop.h"
#include "saved.h"
#include "vml/gpu_choice.h"
#include "vml/hardware_decoding.h"
#include "vml/settings.h"
#include "vml/app_log.h"
#include "vml/isolated.h"
#include "vml/platform/console.h"
#include "vml/platform/system.h"

int main(int argc, char** argv) {
    using namespace vml;
#ifndef _WIN32
    // Output to a pipe whose reader has gone (`vml-cli compare ... | head`)
    // fails, as on Windows, rather than ending the program with SIGPIPE
    // after the run, with exit code 141.
    std::signal(SIGPIPE, SIG_IGN);
#endif
    const std::vector<std::string> words = platform::utf8_arguments(argc, argv);
    // FFmpeg's D3D11VA decoding on the AMD GPU, not D3D11's default adapter
    // (another maker's on a PC with two), here and in the isolated tasks.
    use_d3d11va_adapter(d3d11va_adapter());
    // This program again, as the process of its own that native GPU code
    // runs in (isolated.h): a task and its request, not a command.
    if (words.size() == 3 && words[0] == isolated_argument) {
        return run_isolated_task(words[1], platform::path_from_text(words[2]));
    }
    set_isolation_program(platform::executable_path());
    try {
        const cli::Arguments arguments = cli::parse_arguments(words);
        if (arguments.print_and_exit) {
            platform::write_text(platform::Stream::output, *arguments.print_and_exit);
            return cli::exit_ok;
        }
        // The settings the command runs with (--settings, --no-settings, else
        // the profile's), read once; bdrate reads none.
        const cli::ProfileArguments* profile = arguments.command == cli::Command::compare  ? &arguments.compare.profile
                                               : arguments.command == cli::Command::bitrate ? &arguments.bitrate.profile
                                               : arguments.command == cli::Command::devices ? &arguments.devices.profile
                                                                                            : nullptr;
        cli::ChosenSettings chosen;
        if (profile != nullptr) {
            try {
                chosen = cli::choose_settings(*profile);
            } catch (const cli::SettingsUnreadable& error) {
                platform::write_text(platform::Stream::error, std::string(error.what()) + "\n");
                if (arguments.command == cli::Command::compare) {
                    cli::write_error_json(arguments.compare, error.what(), cli::codes::settings_unreadable);
                }
                return cli::exit_input;
            }
        }
        // The GPU to score on: --gpu, else one named in the environment,
        // else the settings' (Settings::gpu); the processes this one starts
        // inherit it.
        bool gpu_from_settings = false;
        if (arguments.command == cli::Command::compare && arguments.compare.gpu) {
            const std::string& gpu = *arguments.compare.gpu;
            choose_gpu(gpu == "auto" ? std::nullopt : std::optional(gpu));
        } else if (arguments.command == cli::Command::compare || arguments.command == cli::Command::devices) {
            if (!chosen_gpu()) {
                const std::string& gpu = chosen.settings.gpu;
                choose_gpu(gpu.empty() ? std::nullopt : std::optional(gpu));
                gpu_from_settings = true;
            }
        }
        resolve_chosen_gpu();
        use_d3d11va_adapter(d3d11va_adapter());
        // The app's log on standard error: its failures always, everything
        // with --verbose.
        const bool verbose = arguments.command == cli::Command::compare && arguments.compare.verbose;
        app_log::log_to_console(verbose ? app_log::Level::info : app_log::Level::error);
        switch (arguments.command) {
            case cli::Command::compare:
                return cli::compare(arguments.compare, chosen, gpu_from_settings);
            case cli::Command::bitrate:
                return cli::bitrate(arguments.bitrate, chosen);
            case cli::Command::bdrate:
                return cli::bdrate(arguments.bdrate);
            case cli::Command::saved:
                return cli::saved(arguments.saved);
            case cli::Command::report:
                return cli::report(arguments.saved);
            case cli::Command::target:
                return cli::target(arguments.target);
            case cli::Command::diff:
                return cli::diff(arguments.saved);
            case cli::Command::devices:
                return cli::devices(chosen.settings, arguments.devices.json_path);
            case cli::Command::none:
                platform::write_text(platform::Stream::output, cli::top_help());
                return cli::exit_usage;
        }
    } catch (const cli::UsageError& error) {
        platform::write_text(platform::Stream::error, std::string(error.what()) + "\n");
        return cli::exit_usage;
    } catch (const std::exception& error) {
        platform::write_text(platform::Stream::error, std::string("Error: ") + error.what() + "\n");
        return cli::exit_failed;
    }
    return cli::exit_usage;
}
