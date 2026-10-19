// VideoMetricsLab from a terminal, in C++: the Python app's command line
// (vmaf_app/cli.py), as far as it has been ported (compare.h says how far).
//
//     vml-cli compare REFERENCE TEST [TEST ...] [options]
//     vml-cli devices
//     vml-cli --version
//
// Progress goes to stderr, results to stdout. Exit code 0 when every video
// got every metric asked for, 1 when one failed, 2 for a wrong command (or
// one this version cannot do yet), 130 when cancelled with Ctrl+C.
#include <csignal>
#include <exception>
#include <string>
#include <vector>

#include "arguments.h"
#include "compare.h"
#include "devices.h"
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
        // The app's log on standard error: its failures always, everything
        // with --verbose.
        const bool verbose = arguments.command == cli::Command::compare && arguments.compare.verbose;
        app_log::log_to_console(verbose ? app_log::Level::info : app_log::Level::error);
        switch (arguments.command) {
            case cli::Command::compare:
                return cli::compare(arguments.compare);
            case cli::Command::devices:
                return cli::devices();
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
