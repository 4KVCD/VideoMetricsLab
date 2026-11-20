// http_post on Windows: WinHTTP.
#include "text_windows.h"  // before <winhttp.h>: it sets <windows.h> up

#include <winhttp.h>

#include <memory>
#include <string>

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

}  // namespace

HttpResponse http_post(std::string_view url, std::string_view content_type, std::string_view body,
                       std::chrono::seconds timeout) {
    HttpResponse response;
    const std::wstring wide_url = widen(url);
    URL_COMPONENTS parts{};
    parts.dwStructSize = sizeof parts;
    parts.dwHostNameLength = static_cast<DWORD>(-1);
    parts.dwUrlPathLength = static_cast<DWORD>(-1);
    parts.dwExtraInfoLength = static_cast<DWORD>(-1);
    if (!WinHttpCrackUrl(wide_url.c_str(), 0, 0, &parts) || parts.nScheme != INTERNET_SCHEME_HTTPS) {
        response.error = "not an https URL";
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
    const InternetHandle request(WinHttpOpenRequest(connection.handle, L"POST", path.c_str(), nullptr,
                                                    WINHTTP_NO_REFERER, WINHTTP_DEFAULT_ACCEPT_TYPES,
                                                    WINHTTP_FLAG_SECURE));
    if (request.handle == nullptr) {
        response.error = last_error("WinHttpOpenRequest");
        return response;
    }
    // A webhook's URL is its key: never followed elsewhere.
    DWORD no_redirects = WINHTTP_DISABLE_REDIRECTS;
    WinHttpSetOption(request.handle, WINHTTP_OPTION_DISABLE_FEATURE, &no_redirects, sizeof no_redirects);
    const std::wstring header = L"Content-Type: " + widen(content_type);
    if (!WinHttpSendRequest(request.handle, header.c_str(), static_cast<DWORD>(header.size()),
                            const_cast<char*>(body.data()), static_cast<DWORD>(body.size()),
                            static_cast<DWORD>(body.size()), 0) ||
        !WinHttpReceiveResponse(request.handle, nullptr)) {
        response.error = last_error("The request");
        return response;
    }
    DWORD status = 0;
    DWORD size = sizeof status;
    WinHttpQueryHeaders(request.handle, WINHTTP_QUERY_STATUS_CODE | WINHTTP_QUERY_FLAG_NUMBER,
                        WINHTTP_HEADER_NAME_BY_INDEX, &status, &size, WINHTTP_NO_HEADER_INDEX);
    response.status = static_cast<int>(status);
    for (;;) {
        DWORD available = 0;
        if (!WinHttpQueryDataAvailable(request.handle, &available) || available == 0) {
            break;
        }
        std::string chunk(available, '\0');
        DWORD read = 0;
        if (!WinHttpReadData(request.handle, chunk.data(), available, &read) || read == 0) {
            break;
        }
        response.body.append(chunk.data(), read);
        if (response.body.size() > 64 * 1024) {
            break;  // a service's answer to a notification is short; enough to say why
        }
    }
    return response;
}

}  // namespace vml::platform
