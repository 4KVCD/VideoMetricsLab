// http_post on Linux and macOS: the system's curl, its request (with the URL
// and the body, which can hold a key) read from a pipe it inherits, never
// from its command line, which any user of the system can see.
#include <fcntl.h>
#include <unistd.h>

#include <cerrno>
#include <csignal>
#include <cstring>
#include <string>
#include <thread>

#include "vml/platform/http.h"
#include "vml/platform/process.h"

namespace vml::platform {

HttpResponse http_post(const HttpRequest& request, std::chrono::seconds timeout) {
    HttpResponse response;
    const std::string_view url = request.url;
    if (url.substr(0, 8) != "https://" && !(request.plain_http && url.substr(0, 7) == "http://")) {
        response.error = request.plain_http ? "not an http or https URL" : "not an https URL";
        return response;
    }
    const std::string config = curl_config(request, timeout);
    int ends[2];
    if (pipe(ends) != 0) {
        response.error = std::string("a pipe could not be made: ") + std::strerror(errno);
        return response;
    }
    // The request is written while curl reads it, from a thread: a webhook's
    // statistics of many videos, or a long {videos} message, fill a pipe
    // (16 to 64 KB) that nobody reads before curl starts, and a write
    // before then would wait for ever, past every timeout. The writing end
    // is this process's only (curl gets none of it: close on exec), so that
    // curl finds the request's end once it is closed; once curl has ended
    // (or never started), a write fails (EPIPE) rather than waiting, as this
    // process's copy of the reading end is closed once curl has started.
    fcntl(ends[1], F_SETFD, FD_CLOEXEC);
#ifdef F_SETNOSIGPIPE
    fcntl(ends[1], F_SETNOSIGPIPE, 1);
#else
    std::signal(SIGPIPE, SIG_IGN);
#endif
    std::thread writer([&config, end = ends[1]] {
        std::size_t written = 0;
        while (written < config.size()) {
            const ssize_t now = write(end, config.data() + written, config.size() - written);
            if (now < 0 && errno == EINTR) {
                continue;
            }
            if (now <= 0) {
                break;
            }
            written += static_cast<std::size_t>(now);
        }
        close(end);
    });
    // Joined however curl's run ends: curl's end closed, it has written all
    // or found curl gone.
    const struct Joined {
        std::thread& thread;
        ~Joined() { thread.join(); }
    } joined{writer};
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
