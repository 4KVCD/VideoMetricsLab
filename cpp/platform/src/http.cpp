#include "vml/platform/http.h"

namespace vml::platform {

namespace {

// `text` as curl reads a quoted value in its configuration.
std::string curl_quoted(std::string_view text) {
    std::string quoted = "\"";
    for (const char c : text) {
        switch (c) {
        case '\\':
            quoted += "\\\\";
            break;
        case '"':
            quoted += "\\\"";
            break;
        case '\n':
            quoted += "\\n";
            break;
        case '\r':
            quoted += "\\r";
            break;
        case '\t':
            quoted += "\\t";
            break;
        default:
            quoted += c;
        }
    }
    return quoted + "\"";
}

}  // namespace

std::string curl_config(const HttpRequest& request, std::chrono::seconds timeout) {
    std::string config;
    config += "url = " + curl_quoted(request.url) + "\n";
    config += request.plain_http ? "proto = \"=https,http\"\n" : "proto = \"=https\"\n";
    config += "header = " + curl_quoted("Content-Type: " + request.content_type) + "\n";
    for (const auto& [name, value] : request.headers) {
        config += "header = " + curl_quoted(name + ": " + value) + "\n";
    }
    config += "data-binary = " + curl_quoted(request.body) + "\n";
    config += "max-time = " + std::to_string(timeout.count()) + "\n";
    config += "silent\n";
    config += "show-error\n";
    config += "write-out = \"\\n%{http_code}\"\n";
    return config;
}

std::vector<std::string> curl_command(int config) { return {"curl", "-q", "-K", "/dev/fd/" + std::to_string(config)}; }

}  // namespace vml::platform
