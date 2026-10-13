// A program's standard error handed on a line at a time
// (ProcessOptions::on_error_line), from the pieces the pipe gives it in.
#pragma once

#include <functional>
#include <string>
#include <string_view>
#include <utility>

namespace vml::platform {

class ErrorLines {
public:
    explicit ErrorLines(std::function<void(std::string_view)> on_line) : on_line_(std::move(on_line)) {}

    // Hands on each line `written` completes, without its line ending.
    void add(std::string_view written) {
        if (!on_line_) {
            return;
        }
        partial_.append(written);
        std::size_t start = 0;
        for (std::size_t end = partial_.find('\n'); end != std::string::npos; end = partial_.find('\n', start)) {
            hand_on(std::string_view(partial_).substr(start, end - start));
            start = end + 1;
        }
        partial_.erase(0, start);
    }

    // Hands on the last line, when the program ended it without a line ending.
    void finish() {
        if (on_line_ && !partial_.empty()) {
            hand_on(partial_);
            partial_.clear();
        }
    }

private:
    void hand_on(std::string_view line) {
        if (!line.empty() && line.back() == '\r') {
            line.remove_suffix(1);
        }
        on_line_(line);
    }

    std::function<void(std::string_view)> on_line_;
    std::string partial_;
};

}  // namespace vml::platform
