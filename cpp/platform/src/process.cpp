// What every system's ChildProcess shares.
#include "vml/platform/process.h"

#include <thread>

namespace vml::platform {

std::string error_tail(const std::string& kept) {
    if (kept.size() <= error_tail_bytes) {
        return kept;
    }
    const std::size_t cut = kept.size() - error_tail_bytes;
    if (kept[cut - 1] == '\n') {
        return kept.substr(cut);  // a line starts there
    }
    const std::size_t line_end = kept.find('\n', cut);
    if (line_end == std::string::npos || line_end + 1 == kept.size()) {
        return kept.substr(cut);  // the last line alone is longer
    }
    return kept.substr(line_end + 1);
}

void keep_error_tail(std::string& kept, std::string_view written) {
    kept.append(written);
    if (kept.size() >= 2 * error_tail_bytes) {
        kept = error_tail(kept);
    }
}

ProcessOutput run_program(const std::vector<std::string>& arguments, std::chrono::milliseconds timeout,
                          const ProcessOptions& options) {
    ChildProcess child = ChildProcess::start(arguments, options);
    // Read on a thread of its own, so that a program writing a lot is never
    // held up by a full pipe while this one waits for it to end.
    std::string output;
    std::thread reader([&child, &output] {
        while (const std::optional<std::string> line = child.read_line()) {
            output += *line;
            output += '\n';
        }
    });
    ProcessOutput result;
    if (!child.wait_for(timeout)) {
        child.terminate();
        result.timed_out = true;
    }
    result.exit_code = child.wait();
    reader.join();
    result.standard_output = std::move(output);
    result.standard_error = child.standard_error();
    return result;
}

}  // namespace vml::platform
