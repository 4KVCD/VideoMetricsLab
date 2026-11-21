// http_post on Windows: WinHTTP.
#include "text_windows.h"  // before <winhttp.h>: it sets <windows.h> up

#include <winhttp.h>

#include <condition_variable>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

#include "vml/platform/http.h"

namespace vml::platform {

namespace {

struct InternetHandle {
    HINTERNET handle = nullptr;
    explicit InternetHandle(HINTERNET opened) : handle(opened) {}
    InternetHandle(const InternetHandle&) = delete;
    InternetHandle& operator=(const InternetHandle&) = delete;
    ~InternetHandle() {
        if (handle != nullptr) {
            WinHttpCloseHandle(handle);
        }
    }
};

// Why the last WinHTTP call failed, in the system's words.
std::string last_error(const char* what) {
    const DWORD code = GetLastError();
    wchar_t* text = nullptr;
    const HMODULE module = GetModuleHandleW(L"winhttp.dll");
    FormatMessageW(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_FROM_HMODULE |
                       FORMAT_MESSAGE_IGNORE_INSERTS,
                   module, code, 0, reinterpret_cast<wchar_t*>(&text), 0, nullptr);
    std::string message = std::string(what) + " failed (" + std::to_string(code) + ")";
    if (text != nullptr) {
        std::string said = narrow(text);
        while (!said.empty() && (said.back() == '\n' || said.back() == '\r' || said.back() == ' ')) {
            said.pop_back();
        }
        message += ": " + said;
        LocalFree(text);
    }
    return message;
}

// One deadline for the whole request. WinHttpSetTimeouts gives each step
// (finding the host, connecting, sending, receiving) the whole timeout, so a
// host slow at each step could hold the run's end for four times as long,
// for each service. At the deadline the request's handle is closed, which
// ends the WinHTTP call waiting on it, as curl's --max-time ends curl.
class Deadline {
public:
    Deadline(HINTERNET request, std::chrono::milliseconds timeout)
        : request_(request), watch_([this, timeout] {
              std::unique_lock lock(mutex_);
              if (!woken_.wait_for(lock, timeout, [this] { return stopped_; })) {
                  WinHttpCloseHandle(request_);
                  closed_ = true;
              }
          }) {}
    Deadline(const Deadline&) = delete;
    Deadline& operator=(const Deadline&) = delete;
    ~Deadline() { stop(); }

    // Stops the watch; true when the deadline passed first, and the request's
    // handle is closed.
    bool stop() {
        {
            const std::lock_guard lock(mutex_);
            stopped_ = true;
        }
        woken_.notify_all();
        if (watch_.joinable()) {
            watch_.join();
        }
        return closed_;
    }

private:
    HINTERNET request_;
    std::mutex mutex_;
    std::condition_variable woken_;
    bool stopped_ = false;
    bool closed_ = false;
    std::thread watch_;  // last: started once the rest is ready
};

// Sends the request on `request` and reads the answer into `response`.
void exchange(HINTERNET request, const HttpRequest& asked, HttpResponse& response) {
    std::wstring headers = L"Content-Type: " + widen(asked.content_type);
    for (const auto& [name, value] : asked.headers) {
        headers += L"\r\n" + widen(name) + L": " + widen(value);
    }
    const std::string_view body = asked.body;
    if (!WinHttpSendRequest(request, headers.c_str(), static_cast<DWORD>(headers.size()),
                            const_cast<char*>(body.data()), static_cast<DWORD>(body.size()),
                            static_cast<DWORD>(body.size()), 0) ||
        !WinHttpReceiveResponse(request, nullptr)) {
        response.error = last_error("The request");
        return;
    }
    DWORD status = 0;
    DWORD size = sizeof status;
    WinHttpQueryHeaders(request, WINHTTP_QUERY_STATUS_CODE | WINHTTP_QUERY_FLAG_NUMBER, WINHTTP_HEADER_NAME_BY_INDEX,
                        &status, &size, WINHTTP_NO_HEADER_INDEX);
    response.status = static_cast<int>(status);
    for (;;) {
        DWORD available = 0;
        if (!WinHttpQueryDataAvailable(request, &available) || available == 0) {
            break;
        }
        std::string chunk(available, '\0');
        DWORD read = 0;
        if (!WinHttpReadData(request, chunk.data(), available, &read) || read == 0) {
            break;
        }
        response.body.append(chunk.data(), read);
        if (response.body.size() > 64 * 1024) {
            break;  // a service's answer to a notification is short; enough to say why
        }
    }
}

}  // namespace

HttpResponse http_post(const HttpRequest& asked, std::chrono::seconds timeout) {
    HttpResponse response;
    const std::wstring wide_url = widen(asked.url);
    URL_COMPONENTS parts{};
    parts.dwStructSize = sizeof parts;
    parts.dwHostNameLength = static_cast<DWORD>(-1);
    parts.dwUrlPathLength = static_cast<DWORD>(-1);
    parts.dwExtraInfoLength = static_cast<DWORD>(-1);
    const bool cracked = WinHttpCrackUrl(wide_url.c_str(), 0, 0, &parts);
    const bool secure = cracked && parts.nScheme == INTERNET_SCHEME_HTTPS;
    if (!secure && !(cracked && asked.plain_http && parts.nScheme == INTERNET_SCHEME_HTTP)) {
        response.error = asked.plain_http ? "not an http or https URL" : "not an https URL";
        return response;
    }
    const std::wstring host(parts.lpszHostName, parts.dwHostNameLength);
    const std::wstring path = std::wstring(parts.lpszUrlPath, parts.dwUrlPathLength) +
                              std::wstring(parts.lpszExtraInfo, parts.dwExtraInfoLength);
    const InternetHandle session(WinHttpOpen(L"VideoMetricsLab", WINHTTP_ACCESS_TYPE_AUTOMATIC_PROXY,
                                             WINHTTP_NO_PROXY_NAME, WINHTTP_NO_PROXY_BYPASS, 0));
    if (session.handle == nullptr) {
        response.error = last_error("WinHttpOpen");
        return response;
    }
    const int milliseconds = static_cast<int>(std::chrono::milliseconds(timeout).count());
    WinHttpSetTimeouts(session.handle, milliseconds, milliseconds, milliseconds, milliseconds);
    const InternetHandle connection(WinHttpConnect(session.handle, host.c_str(), parts.nPort, 0));
    if (connection.handle == nullptr) {
        response.error = last_error("WinHttpConnect");
        return response;
    }
    InternetHandle request(WinHttpOpenRequest(connection.handle, L"POST", path.c_str(), nullptr, WINHTTP_NO_REFERER,
                                              WINHTTP_DEFAULT_ACCEPT_TYPES, secure ? WINHTTP_FLAG_SECURE : 0));
    if (request.handle == nullptr) {
        response.error = last_error("WinHttpOpenRequest");
        return response;
    }
    // A webhook's URL is its key: never followed elsewhere.
    DWORD no_redirects = WINHTTP_DISABLE_REDIRECTS;
    WinHttpSetOption(request.handle, WINHTTP_OPTION_DISABLE_FEATURE, &no_redirects, sizeof no_redirects);
    Deadline deadline(request.handle, timeout);
    exchange(request.handle, asked, response);
    if (deadline.stop()) {
        request.handle = nullptr;  // closed at the deadline
        response = HttpResponse();
        response.error = "no answer in time";
    }
    return response;
}

}  // namespace vml::platform
