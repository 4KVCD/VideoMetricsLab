// An HTTPS POST without a GUI toolkit, for the command line's notifications
// (--notify): WinHTTP on Windows; elsewhere the system's curl, given its
// request on a descriptor of its own, so that a URL or a body holding a key
// is never on a command line.
#pragma once

#include <chrono>
#include <string>
#include <string_view>
#include <vector>

namespace vml::platform {

// What the server answered: its HTTP status (0 when none came) and body, or
// why there was no answer.
struct HttpResponse {
    int status = 0;
    std::string body;
    std::string error;
};

// POSTs `body` as `content_type` to `url` (https only), following no
// redirect, giving up after `timeout`.
HttpResponse http_post(std::string_view url, std::string_view content_type, std::string_view body,
                       std::chrono::seconds timeout = std::chrono::seconds(20));

// curl's configuration for http_post's request (curl -K), each value quoted
// as curl reads a quoted value: backslashes, quotes and line breaks escaped.
// It asks for the status on the last line of the output.
std::string curl_config(std::string_view url, std::string_view content_type, std::string_view body,
                        std::chrono::seconds timeout);

// The command that runs curl with that configuration, read from the
// descriptor `config` (curl -K /dev/fd/N). -q, first (curl reads it only
// there), keeps the user's ~/.curlrc out: a line there such as `insecure`
// (no TLS checks for the keys), `location` (the POST followed elsewhere) or
// `verbose` (the webhook's token on standard error) would apply to this
// request too.
std::vector<std::string> curl_command(int config);

}  // namespace vml::platform
