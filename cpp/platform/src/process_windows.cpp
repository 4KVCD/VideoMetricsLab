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

#include <algorithm>
#include <condition_variable>
#include <cstddef>
#include <cstring>
#include <mutex>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include "error_lines.h"
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

// `bytes` is how much it holds; 0 is the system's size.
Pipe make_pipe(std::size_t bytes = 0) {
    SECURITY_ATTRIBUTES inheritable{sizeof(SECURITY_ATTRIBUTES), nullptr, TRUE};
    HANDLE read = nullptr;
    HANDLE write = nullptr;
    if (!CreatePipe(&read, &write, &inheritable, static_cast<DWORD>(std::min<std::size_t>(bytes, 1u << 30)))) {
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

// The ids of the processes in `job`, in the order they joined it: the
// program, then what it started.
std::vector<DWORD> job_processes(HANDLE job) {
    // The two counts, then room for far more ids than a program starts.
    std::vector<ULONG_PTR> buffer(256);
    auto* list = reinterpret_cast<JOBOBJECT_BASIC_PROCESS_ID_LIST*>(buffer.data());
    if (!QueryInformationJobObject(job, JobObjectBasicProcessIdList, list,
                                   static_cast<DWORD>(buffer.size() * sizeof(ULONG_PTR)), nullptr)) {
        return {};
    }
    const std::size_t first = offsetof(JOBOBJECT_BASIC_PROCESS_ID_LIST, ProcessIdList) / sizeof(ULONG_PTR);
    std::vector<DWORD> ids;
    for (DWORD i = 0; i < list->NumberOfProcessIdsInList; ++i) {
        ids.push_back(static_cast<DWORD>(buffer[first + i]));
    }
    return ids;
}

// Whether the process is a console host (conhost.exe), which Windows starts
// for each console program: it does no work, so it is not suspended.
bool is_console_host(HANDLE process) {
    wchar_t path[MAX_PATH];
    DWORD length = MAX_PATH;
    if (!QueryFullProcessImageNameW(process, 0, path, &length)) {
        return false;
    }
    const std::wstring_view name = std::wstring_view(path, length).substr(
        std::wstring_view(path, length).find_last_of(L'\\') + 1);
    return CompareStringOrdinal(name.data(), static_cast<int>(name.size()), L"conhost.exe", -1, TRUE) == CSTR_EQUAL;
}

// ntdll's NtSuspendProcess or NtResumeProcess, which suspend or resume every
// thread of a process at once, as psutil does it for the Python app (Windows
// documents no call that does); nullptr if ntdll lacks it.
using ProcessCall = LONG(WINAPI*)(HANDLE);
ProcessCall ntdll_call(const char* name) {
    const HMODULE ntdll = GetModuleHandleW(L"ntdll.dll");
    if (ntdll == nullptr) {
        return nullptr;
    }
    // Through void (*)(), which any function type may be cast from and to.
    return reinterpret_cast<ProcessCall>(reinterpret_cast<void (*)()>(GetProcAddress(ntdll, name)));
}

}  // namespace

struct ChildProcess::State {
    Handle process;
    Handle job;
    Handle output;  // the program's standard output, read by read_line
    DWORD id = 0;
    std::string unread;  // read from the pipe, not yet given as lines
    bool output_ended = false;
    // For gather's waits of a millisecond, as Python's time.sleep(0.001)
    // makes them: a plain Sleep(1) can last a whole 15.6 ms tick.
    Handle timer;
    // The processes suspend() has suspended, each once, in that order. The
    // handles keep their ids from going to other processes meanwhile.
    // `holds` counts the suspend() calls not yet let go; `hold_mutex` guards
    // both.
    std::mutex hold_mutex;
    int holds = 0;
    std::vector<std::pair<DWORD, Handle>> suspended;

    // Standard error, read on a thread of its own so the program never
    // waits on a full pipe; its end is kept (keep_error_tail).
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
    Pipe output = make_pipe(options.output_pipe_bytes);
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
    startup.lpAttributeList = attributes;

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
    state->error_reader = std::thread([raw = state.get(), pipe = std::move(error.read),
                                       lines = ErrorLines(options.on_error_line)]() mutable {
        char buffer[4096];
        DWORD count = 0;
        while (ReadFile(pipe.get(), buffer, sizeof buffer, &count, nullptr) && count > 0) {
            lines.add(std::string_view(buffer, count));
            const std::lock_guard lock(raw->error_mutex);
            keep_error_tail(raw->error_text, std::string_view(buffer, count));
        }
        lines.finish();
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

std::size_t ChildProcess::read(void* into, std::size_t bytes) {
    if (bytes == 0) {
        return 0;
    }
    if (!state_->unread.empty()) {  // what read_line read ahead
        const std::size_t count = std::min(bytes, state_->unread.size());
        std::memcpy(into, state_->unread.data(), count);
        state_->unread.erase(0, count);
        return count;
    }
    if (state_->output_ended) {
        return 0;
    }
    DWORD count = 0;
    const DWORD wanted = static_cast<DWORD>(std::min<std::size_t>(bytes, 1u << 30));
    if (ReadFile(state_->output.get(), into, wanted, &count, nullptr) && count > 0) {
        return count;
    }
    state_->output_ended = true;
    return 0;
}

void ChildProcess::gather(std::size_t bytes, const std::atomic<bool>& stop) {
    if (!state_->unread.empty() || state_->output_ended) {
        return;
    }
    DWORD held = 0;
    if (!GetNamedPipeInfo(state_->output.get(), nullptr, nullptr, &held, nullptr)) {
        return;
    }
    const std::size_t wanted = std::min<std::size_t>(bytes, held / 2);
    if (!state_->timer) {
        state_->timer.reset(
            CreateWaitableTimerExW(nullptr, nullptr, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, TIMER_ALL_ACCESS));
        if (!state_->timer) {  // before Windows 10 1803
            state_->timer.reset(CreateWaitableTimerExW(nullptr, nullptr, 0, TIMER_ALL_ACCESS));
        }
    }
    while (!stop) {
        DWORD available = 0;
        if (!PeekNamedPipe(state_->output.get(), nullptr, 0, nullptr, &available, nullptr)) {
            return;  // closed and empty: the read says so
        }
        if (available >= wanted || WaitForSingleObject(state_->process.get(), 0) == WAIT_OBJECT_0) {
            return;
        }
        LARGE_INTEGER due;
        due.QuadPart = -10'000;  // a millisecond from now, in 100 ns units
        if (state_->timer && SetWaitableTimer(state_->timer.get(), &due, 0, nullptr, nullptr, FALSE)) {
            WaitForSingleObject(state_->timer.get(), INFINITE);
        } else {
            Sleep(1);
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

void ChildProcess::suspend() {
    static const ProcessCall suspend_process = ntdll_call("NtSuspendProcess");
    const std::lock_guard lock(state_->hold_mutex);
    ++state_->holds;
    if (suspend_process == nullptr) {
        return;
    }
    for (const DWORD id : job_processes(state_->job.get())) {
        if (std::any_of(state_->suspended.begin(), state_->suspended.end(),
                        [id](const auto& done) { return done.first == id; })) {
            continue;
        }
        Handle process(OpenProcess(PROCESS_SUSPEND_RESUME | PROCESS_QUERY_LIMITED_INFORMATION, FALSE, id));
        // The id may have gone to another process since the job listed it.
        BOOL in_job = FALSE;
        if (!process || !IsProcessInJob(process.get(), state_->job.get(), &in_job) || !in_job ||
            is_console_host(process.get())) {
            continue;
        }
        if (suspend_process(process.get()) >= 0) {
            state_->suspended.emplace_back(id, std::move(process));
        }
    }
}

void ChildProcess::resume() {
    static const ProcessCall resume_process = ntdll_call("NtResumeProcess");
    const std::lock_guard lock(state_->hold_mutex);
    if (state_->holds == 0 || --state_->holds > 0) {
        return;  // not held, or still held by another
    }
    if (resume_process != nullptr) {
        for (auto done = state_->suspended.rbegin(); done != state_->suspended.rend(); ++done) {
            resume_process(done->second.get());
        }
    }
    state_->suspended.clear();
}

std::string ChildProcess::standard_error() {
    std::unique_lock lock(state_->error_mutex);
    // Once the program has ended its error output ends with it, unless a
    // program it started still holds the pipe: then what came is enough.
    state_->error_ended_signal.wait_for(lock, std::chrono::seconds(5), [this] { return state_->error_ended; });
    return universal_newlines(error_tail(state_->error_text));
}

int ChildProcess::process_id() const { return static_cast<int>(state_->id); }

}  // namespace vml::platform
