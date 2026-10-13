// ChildProcess on Windows: CreateProcessW with pipes, in a job object of its
// own.
//
// Each program is put in a job object that ends every process in it when
// the job's last handle closes (JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE). The
// handle is the app's alone, so the program and whatever it starts end with
// the app, however the app ends, and terminate() ends them together.
//
// Only the program's own pipe ends are handed down to it
// (PROC_THREAD_ATTRIBUTE_HANDLE_LIST), as Python's subprocess does. Handed
// down by plain inheritance, two programs started at the same moment could
// each take a copy of the other's pipe, and a reader would then wait for an
// end of output that never comes.
#include "vml/platform/process.h"

#include <condition_variable>
#include <mutex>
#include <thread>
#include <utility>

#include "text_windows.h"

namespace vml::platform {

namespace {

// A Windows handle, closed when it goes.
class Handle {
public:
    Handle() = default;
    explicit Handle(HANDLE handle) : handle_(handle == INVALID_HANDLE_VALUE ? nullptr : handle) {}
    Handle(Handle&& other) noexcept : handle_(other.release()) {}
    Handle& operator=(Handle&& other) noexcept {
        if (this != &other) {
            reset(other.release());
        }
        return *this;
    }
    ~Handle() { reset(); }

    HANDLE get() const { return handle_; }
    explicit operator bool() const { return handle_ != nullptr; }
    HANDLE release() { return std::exchange(handle_, nullptr); }
    void reset(HANDLE handle = nullptr) {
        if (handle_ != nullptr) {
            CloseHandle(handle_);
        }
        handle_ = handle;
    }

private:
    HANDLE handle_ = nullptr;
};

[[noreturn]] void fail(const std::string& what) {
    throw ProcessError(what + " (Windows error " + std::to_string(GetLastError()) + ")");
}

// One argument quoted as the Microsoft C run-time reads a command line back
// into arguments (CommandLineToArgvW): in double quotes when it has a space,
// a tab or a quote or is empty, with its quotes and the backslashes before
// them escaped.
std::wstring quote_argument(const std::wstring& argument) {
    if (!argument.empty() && argument.find_first_of(L" \t\n\v\"") == std::wstring::npos) {
        return argument;
    }
    std::wstring quoted = L"\"";
    std::size_t backslashes = 0;
    for (const wchar_t c : argument) {
        if (c == L'\\') {
            ++backslashes;
            continue;
        }
        // Backslashes count double before a quote, which itself is escaped.
        quoted.append(c == L'"' ? backslashes * 2 + 1 : backslashes, L'\\');
        quoted.push_back(c);
        backslashes = 0;
    }
    quoted.append(backslashes * 2, L'\\');  // before the closing quote
    quoted.push_back(L'"');
    return quoted;
}

std::wstring command_line(const std::vector<std::string>& arguments) {
    std::wstring line;
    for (const std::string& argument : arguments) {
        if (!line.empty()) {
            line.push_back(L' ');
        }
        line += quote_argument(widen(argument));
    }
    return line;
}

// A pipe whose write end is handed to the program.
struct Pipe {
    Handle read;
    Handle write;
};

Pipe make_pipe() {
    SECURITY_ATTRIBUTES inheritable{sizeof(SECURITY_ATTRIBUTES), nullptr, TRUE};
    HANDLE read = nullptr;
    HANDLE write = nullptr;
    if (!CreatePipe(&read, &write, &inheritable, 0)) {
        fail("a pipe could not be made");
    }
    Pipe pipe{Handle(read), Handle(write)};
    // Only the write end goes to the program.
    SetHandleInformation(pipe.read.get(), HANDLE_FLAG_INHERIT, 0);
    return pipe;
}

// A job object that ends its processes when its last handle closes.
Handle make_job() {
    Handle job(CreateJobObjectW(nullptr, nullptr));
    if (!job) {
        fail("a job object could not be made");
    }
    JOBOBJECT_EXTENDED_LIMIT_INFORMATION limits{};
    limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
    if (!SetInformationJobObject(job.get(), JobObjectExtendedLimitInformation, &limits, sizeof limits)) {
        fail("a job object could not be set up");
    }
    return job;
}

}  // namespace

struct ChildProcess::State {
    Handle process;
    Handle job;
    Handle output;  // the program's standard output, read by read_line
    DWORD id = 0;
    std::string unread;  // read from the pipe, not yet given as lines
    bool output_ended = false;

    // Standard error, read on a thread of its own so the program never
    // waits on a full pipe.
    std::thread error_reader;
    std::mutex error_mutex;
    std::condition_variable error_ended_signal;
    std::string error_text;
    bool error_ended = false;

    ~State() {
        if (process) {
            TerminateJobObject(job.get(), 1);
        }
        if (error_reader.joinable()) {
            error_reader.join();
        }
    }
};

ChildProcess ChildProcess::start(const std::vector<std::string>& arguments, const ProcessOptions& options) {
    if (arguments.empty()) {
        throw ProcessError("no program to start");
    }
    Pipe output = make_pipe();
    Pipe error = make_pipe();
    SECURITY_ATTRIBUTES inheritable{sizeof(SECURITY_ATTRIBUTES), nullptr, TRUE};
    Handle input(CreateFileW(L"NUL", GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE, &inheritable, OPEN_EXISTING,
                             0, nullptr));
    if (!input) {
        fail("NUL could not be opened");
    }

    // The three handles the program is given, and no others.
    HANDLE handed_down[3] = {input.get(), output.write.get(), error.write.get()};
    SIZE_T list_size = 0;
    InitializeProcThreadAttributeList(nullptr, 1, 0, &list_size);
    std::vector<char> list_memory(list_size);
    auto* attributes = reinterpret_cast<LPPROC_THREAD_ATTRIBUTE_LIST>(list_memory.data());
    if (!InitializeProcThreadAttributeList(attributes, 1, 0, &list_size) ||
        !UpdateProcThreadAttribute(attributes, 0, PROC_THREAD_ATTRIBUTE_HANDLE_LIST, handed_down,
                                   sizeof handed_down, nullptr, nullptr)) {
        fail("the program's handles could not be listed");
    }

    STARTUPINFOEXW startup{};
    startup.StartupInfo.cb = sizeof startup;
    startup.StartupInfo.dwFlags = STARTF_USESTDHANDLES;
    startup.StartupInfo.hStdInput = input.get();
    startup.StartupInfo.hStdOutput = output.write.get();
    startup.StartupInfo.hStdError = error.write.get();

    std::wstring line = command_line(arguments);
    const std::wstring folder = options.working_directory.empty() ? std::wstring() : options.working_directory.wstring();
    PROCESS_INFORMATION started{};
    // Started suspended, so that it is in its job before it can start
    // anything of its own.
    const BOOL made = CreateProcessW(nullptr, line.data(), nullptr, nullptr, TRUE,
                                     CREATE_NO_WINDOW | CREATE_SUSPENDED | EXTENDED_STARTUPINFO_PRESENT |
                                         CREATE_UNICODE_ENVIRONMENT,
                                     nullptr, folder.empty() ? nullptr : folder.c_str(), &startup.StartupInfo,
                                     &started);
    DeleteProcThreadAttributeList(attributes);
    if (!made) {
        fail("could not start " + arguments.front());
    }
    auto state = std::make_unique<State>();
    state->process = Handle(started.hProcess);
    Handle thread(started.hThread);
    state->id = started.dwProcessId;
    state->job = make_job();
    if (!AssignProcessToJobObject(state->job.get(), state->process.get())) {
        TerminateProcess(state->process.get(), 1);
        fail("could not put " + arguments.front() + " in a job object");
    }
    ResumeThread(thread.get());

    // The program has its copies; ours must close, or the pipes never end.
    output.write.reset();
    error.write.reset();
    state->output = std::move(output.read);
    state->error_reader = std::thread([raw = state.get(), pipe = std::move(error.read)]() {
        char buffer[4096];
        DWORD count = 0;
        while (ReadFile(pipe.get(), buffer, sizeof buffer, &count, nullptr) && count > 0) {
            const std::lock_guard lock(raw->error_mutex);
            raw->error_text.append(buffer, count);
        }
        const std::lock_guard lock(raw->error_mutex);
        raw->error_ended = true;
        raw->error_ended_signal.notify_all();
    });
    return ChildProcess(std::move(state));
}

ChildProcess::ChildProcess(std::unique_ptr<State> state) : state_(std::move(state)) {}
ChildProcess::ChildProcess(ChildProcess&&) noexcept = default;
ChildProcess& ChildProcess::operator=(ChildProcess&&) noexcept = default;
ChildProcess::~ChildProcess() = default;

std::optional<std::string> ChildProcess::read_line() {
    while (true) {
        const std::size_t end = state_->unread.find('\n');
        if (end != std::string::npos) {
            std::string line = state_->unread.substr(0, end);
            state_->unread.erase(0, end + 1);
            if (!line.empty() && line.back() == '\r') {
                line.pop_back();
            }
            return line;
        }
        if (state_->output_ended) {
            if (state_->unread.empty()) {
                return std::nullopt;
            }
            return std::exchange(state_->unread, std::string());
        }
        char buffer[4096];
        DWORD count = 0;
        if (ReadFile(state_->output.get(), buffer, sizeof buffer, &count, nullptr) && count > 0) {
            state_->unread.append(buffer, count);
        } else {
            state_->output_ended = true;
        }
    }
}

int ChildProcess::wait() {
    WaitForSingleObject(state_->process.get(), INFINITE);
    DWORD code = 0;
    GetExitCodeProcess(state_->process.get(), &code);
    return static_cast<int>(code);
}

bool ChildProcess::wait_for(std::chrono::milliseconds timeout) {
    return WaitForSingleObject(state_->process.get(), static_cast<DWORD>(timeout.count())) == WAIT_OBJECT_0;
}

void ChildProcess::terminate() { TerminateJobObject(state_->job.get(), 1); }

std::string ChildProcess::standard_error() {
    std::unique_lock lock(state_->error_mutex);
    // Once the program has ended its error output ends with it, unless a
    // program it started still holds the pipe: then what came is enough.
    state_->error_ended_signal.wait_for(lock, std::chrono::seconds(5), [this] { return state_->error_ended; });
    return state_->error_text;
}

int ChildProcess::process_id() const { return static_cast<int>(state_->id); }

}  // namespace vml::platform
