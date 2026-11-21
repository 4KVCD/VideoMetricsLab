// http_post on Linux and macOS: the system's curl, its request (with the URL
// and the body, which can hold a key) read from a pipe it inherits, never
// from its command line, which any user of the system can see.
#include <fcntl.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <string>

#include "vml/platform/http.h"
#include "vml/platform/process.h"

namespace vml::platform {

HttpResponse http_post(std::string_view url, std::string_view content_type, std::string_view body,
                       std::chrono::seconds timeout) {
    HttpResponse response;
    if (url.substr(0, 8) != "https://") {
        response.error = "not an https URL";
        return response;
    }
    const std::string config = curl_config(url, content_type, body, timeout);
    int ends[2];
    if (pipe(ends) != 0) {
        response.error = std::string("a pipe could not be made: ") + std::strerror(errno);
        return response;
    }
    // The request fits the pipe (a notification is a few KB): written whole
    // before curl starts, then the end closed, so that curl reads it to its end.
    std::size_t written = 0;
    while (written < config.size()) {
        const ssize_t now = write(ends[1], config.data() + written, config.size() - written);
        if (now <= 0) {
            break;
        }
        written += static_cast<std::size_t>(now);
    }
    close(ends[1]);
    if (written < config.size()) {
        close(ends[0]);
        response.error = "the request could not be handed to curl";
        return response;
    }
    ProcessOptions options;
    options.inherited_descriptors = {ends[0]};  // curl's now; closed here once it has started
    try {
        const ProcessOutput output = run_program(curl_command(ends[0]), timeout + std::chrono::seconds(5), options);
        // The status is the output's last line (curl_config's write-out).
        const std::string& text = output.standard_output;
        const std::size_t line = text.rfind('\n');
        const std::string code = line == std::string::npos ? text : text.substr(line + 1);
        response.status = code.empty() ? 0 : std::atoi(code.c_str());
        response.body = line == std::string::npos ? std::string() : text.substr(0, line);
        if (response.status == 0) {
            response.error = output.timed_out ? std::string("no answer in time")
                                              : universal_newlines(output.standard_error);
            while (!response.error.empty() && response.error.back() == '\n') {
                response.error.pop_back();
            }
            if (response.error.empty()) {
                response.error = "curl ended with " + exit_code_text(output.exit_code);
            }
        }
    } catch (const ProgramNotFound&) {
        response.error = "curl was not found: it sends the notifications from the command line";
    } catch (const ProcessError& error) {
        response.error = error.what();
    }
    return response;
}

}  // namespace vml::platform
