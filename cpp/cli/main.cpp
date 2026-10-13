// VideoMetricsLab from a terminal, in C++: the Python app's command line
// (vmaf_app/cli.py), as far as it has been ported (compare.h says how far).
//
//     vml-cli compare REFERENCE TEST [TEST ...] [options]
//     vml-cli --version
//
// Progress goes to stderr, results to stdout. Exit code 0 when every video
// got every metric asked for, 1 when one failed, 2 for a wrong command (or
// one this version cannot do yet), 130 when cancelled with Ctrl+C.
#include <exception>

#include "arguments.h"
#include "compare.h"
#include "vml/platform/console.h"

int main(int argc, char** argv) {
    using namespace vml;
    try {
        const cli::Arguments arguments = cli::parse_arguments(platform::utf8_arguments(argc, argv));
        if (arguments.print_and_exit) {
            platform::write_text(platform::Stream::output, *arguments.print_and_exit);
            return cli::exit_ok;
        }
        switch (arguments.command) {
            case cli::Command::compare:
                return cli::compare(arguments.compare);
            case cli::Command::devices:
                platform::write_text(platform::Stream::error,
                                     "The C++ command line cannot do this yet: the devices command.\n");
                return cli::exit_usage;
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
