"""Launching ffmpeg/ffprobe without a console window flashing up.

The app runs under pythonw.exe, which has no console of its own, so every
child process Windows starts gets a brand new console window -- ffprobe on
each added video, the version checks at startup, crop detection during a
run. They appear and vanish, and with several videos it looks like the app
is misbehaving.
"""
from __future__ import annotations

import contextlib
import functools
import logging
import os
import subprocess
import threading

_log = logging.getLogger(__name__)

# CREATE_NO_WINDOW. Defined here rather than imported from subprocess so the
# module still imports cleanly off Windows, where the flag does not exist.
_CREATE_NO_WINDOW = 0x0800_0000


def hidden_kwargs() -> dict:
    """Extra Popen/run keyword arguments that suppress the console window."""
    if os.name != "nt":
        return {}
    return {"creationflags": _CREATE_NO_WINDOW}


def run(cmd, **kwargs):
    """subprocess.run with the console window suppressed on Windows."""
    return subprocess.run(cmd, **{**hidden_kwargs(), **kwargs})


def popen(cmd, **kwargs):
    """subprocess.Popen with the console window suppressed on Windows, and
    the process ended with the app's (end_with_app)."""
    process = subprocess.Popen(cmd, **{**hidden_kwargs(), **kwargs})
    end_with_app(process.pid)
    return process


# A Windows job object that ends every process in it when the app's process
# ends, however it ends: the handle is the app's alone, and Windows closes
# it then (JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE). The app's FFmpeg and its
# isolated scorers used to run on after a crash or End task -- a whole
# film's CPU fallback, hours of it. Only what the app starts for its work
# is put in it (popen, isolated.run_isolated), and what those start goes
# with them; what it opens for the user (a link, a folder) is not, so
# closing the app never closes those.
_JOB_LOCK = threading.Lock()
_job: int | None = None
_job_failed = False


def _kill_on_close_job() -> int | None:
    global _job, _job_failed
    with _JOB_LOCK:
        if _job is not None or _job_failed:
            return _job
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateJobObjectW.restype = wintypes.HANDLE
        kernel32.CreateJobObjectW.argtypes = [wintypes.LPVOID, wintypes.LPCWSTR]
        kernel32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD]

        class _Basic(ctypes.Structure):
            _fields_ = [("PerProcessUserTimeLimit", ctypes.c_longlong), ("PerJobUserTimeLimit", ctypes.c_longlong),
                        ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                        ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                        ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD),
                        ("SchedulingClass", wintypes.DWORD)]

        class _Extended(ctypes.Structure):
            _fields_ = [("BasicLimitInformation", _Basic), ("IoInfo", ctypes.c_ulonglong * 6),
                        ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                        ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]

        job = kernel32.CreateJobObjectW(None, None)
        limits = _Extended()
        limits.BasicLimitInformation.LimitFlags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not job or not kernel32.SetInformationJobObject(job, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
            _log.info("Child processes are not ended with the app here (job object: error %d)",
                      ctypes.get_last_error())
            _job_failed = True
            return None
        _job = job
        return _job


def end_with_app(pid: int) -> None:
    """Has process `pid` -- just started by the app, before it starts any of
    its own -- ended when the app's process ends (_kill_on_close_job)."""
    if os.name != "nt":
        return
    job = _kill_on_close_job()
    if job is None:
        return
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel32.OpenProcess(0x0100 | 0x0001, False, pid)  # PROCESS_SET_QUOTA | PROCESS_TERMINATE
    if not handle:
        return  # already gone
    try:
        # Fails where the process is in a job that cannot nest (Windows 7);
        # it then outlives a crashed app, as before.
        if not kernel32.AssignProcessToJobObject(job, handle):
            _log.debug("Process %d is not ended with the app (error %d)", pid, ctypes.get_last_error())
    finally:
        kernel32.CloseHandle(handle)


#: The pipe a program's raw video frames come through. subprocess.PIPE is
#: Windows' default of 4 KB: a writer hands a 4K frame over in thousands of
#: pieces, each a switch between the two processes. Measured on one 4K
#: 10-bit HEVC stream piped as YUV: 55 fps with it, 107 with 64 MB; 4K RGBA
#: for playback: 30 fps against 72.
FRAME_PIPE_BYTES = 64 * 1024 * 1024


def popen_piped(cmd, pipe_bytes: int = FRAME_PIPE_BYTES):
    """Starts `cmd` writing its standard output to a pipe of `pipe_bytes`
    (see FRAME_PIPE_BYTES); returns (process, reader), the reader
    unbuffered. Standard error is a pipe, standard input nothing."""
    if os.name == "nt":
        import _winapi
        import msvcrt

        read_handle, write_handle = _winapi.CreatePipe(None, pipe_bytes)
        write_fd = msvcrt.open_osfhandle(write_handle, 0)
        try:
            process = popen(cmd, stdin=subprocess.DEVNULL, stdout=write_fd, stderr=subprocess.PIPE)
        except BaseException:
            os.close(write_fd)
            _winapi.CloseHandle(read_handle)
            raise
        os.close(write_fd)  # the child holds its own copy; EOF arrives when it exits
        return process, open(msvcrt.open_osfhandle(read_handle, os.O_RDONLY), "rb", buffering=0)
    process = popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
    return process, process.stdout


def process_tree(pid: int) -> list:
    """The process and every process it started, oldest first.

    FFmpeg is not always the process the app starts. Chocolatey installs
    ffmpeg.exe as a "shim", a launcher that starts the real ffmpeg.exe as a
    child and waits for it. Suspending or ending the launcher alone left
    FFmpeg running: Pause did not pause, Cancel did not stop it, and the
    CPU perceptual-metric extraction ran ahead of scoring unchecked (364
    images waiting on disk on the GitHub runner, where the limit is 52).

    On Windows the PC's processes are listed by Windows' own snapshot
    (_parent_pids), which lets go of Python's lock while it is taken:
    psutil's children() held the lock all along, 14-22 ms a call here, and
    the window's thread waited behind it -- for each decoder a seek in
    Video Compare stopped.
    """
    import psutil

    try:
        root = psutil.Process(pid)
        parents = _parent_pids() if os.name == "nt" else None
        children = root.children(recursive=True) if parents is None else _descendants(root, parents)
    except psutil.Error:
        return []
    # Not the console host Windows gives each console program: it does no
    # work, and suspending it serves nothing.
    tree = [root]
    for child in children:
        with contextlib.suppress(psutil.Error):
            if child.name().lower() != "conhost.exe":
                tree.append(child)
    return tree


def _descendants(root, parents: dict[int, int]) -> list:
    """root.children(recursive=True) from `parents` ({pid: parent pid}):
    what root started, and what those started, none older than root (a
    process whose pid Windows has since given to another)."""
    import psutil

    started_by: dict[int, list[int]] = {}
    for pid, parent in parents.items():
        if pid != parent:  # the System Idle Process is its own parent
            started_by.setdefault(parent, []).append(pid)
    found, seen, waiting = [], set(), [root.pid]
    while waiting:
        pid = waiting.pop()
        if pid in seen:
            continue
        seen.add(pid)
        for child_pid in started_by.get(pid, ()):
            with contextlib.suppress(psutil.Error):
                child = psutil.Process(child_pid)
                if root.create_time() <= child.create_time():
                    found.append(child)
                    waiting.append(child_pid)
    return found


@functools.cache
def _toolhelp():
    """(kernel32 letting go of Python's lock, kernel32 keeping it, and
    PROCESSENTRY32W) for _parent_pids."""
    import ctypes
    from ctypes import wintypes

    class Entry(ctypes.Structure):
        _fields_ = [("dwSize", wintypes.DWORD), ("cntUsage", wintypes.DWORD), ("th32ProcessID", wintypes.DWORD),
                    ("th32DefaultHeapID", ctypes.c_size_t), ("th32ModuleID", wintypes.DWORD),
                    ("cntThreads", wintypes.DWORD), ("th32ParentProcessID", wintypes.DWORD),
                    ("pcPriClassBase", ctypes.c_long), ("dwFlags", wintypes.DWORD),
                    ("szExeFile", wintypes.WCHAR * 260)]

    releasing = ctypes.WinDLL("kernel32", use_last_error=True)
    releasing.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    releasing.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
    holding = ctypes.PyDLL("kernel32")
    holding.CloseHandle.argtypes = [wintypes.HANDLE]
    for walk in (holding.Process32FirstW, holding.Process32NextW):
        walk.restype = wintypes.BOOL
        walk.argtypes = [wintypes.HANDLE, ctypes.POINTER(Entry)]
    return releasing, holding, Entry


def _parent_pids() -> dict[int, int] | None:
    """{pid: parent pid} for every process on the PC (Windows), or None
    when Windows would not list them.

    Windows takes the snapshot (8-10 ms) with Python's lock let go; the
    walk through it (2 ms) and closing it keep the lock. Letting it go for each of the
    PC's ~460 processes as well, the walk waited up to Python's switch
    interval (5 ms) to have it back, each time, whenever another thread
    was busy: 1.5-2.5 s, the window frozen."""
    import ctypes

    releasing, holding, entry_type = _toolhelp()
    snapshot = releasing.CreateToolhelp32Snapshot(0x2, 0)  # TH32CS_SNAPPROCESS
    if not snapshot or snapshot == ctypes.c_void_p(-1).value:
        return None
    entry = entry_type()
    entry.dwSize = ctypes.sizeof(entry)
    parents = {}
    try:
        found = holding.Process32FirstW(snapshot, ctypes.byref(entry))
        while found:
            parents[entry.th32ProcessID] = entry.th32ParentProcessID
            found = holding.Process32NextW(snapshot, ctypes.byref(entry))
    finally:
        holding.CloseHandle(snapshot)
    return parents


def process_root(pid: int) -> list:
    """Just the process, as a one-item tree: quick, for when its children
    have not been listed yet."""
    import psutil

    try:
        return [psutil.Process(pid)]
    except psutil.Error:
        return []


def signal_tree(pid: int, action: str) -> None:
    """Applies "suspend", "resume", "terminate" or "kill" to a process and
    its children (see process_tree). Suspending starts at the top, so the
    launcher cannot start anything meanwhile; resuming and ending start at
    the bottom, so FFmpeg is never left running under a stopped launcher.
    A process that has already exited is skipped."""
    signal_processes(process_tree(pid), action)


def signal_processes(tree: list, action: str) -> None:
    """signal_tree for a tree already listed by process_tree, for callers
    that switch it often: listing it takes ~13 ms on Windows."""
    import psutil

    for process in (tree if action == "suspend" else reversed(tree)):
        with contextlib.suppress(psutil.NoSuchProcess, psutil.AccessDenied):
            getattr(process, action)()


def terminate(process) -> None:
    """Popen.terminate(), after ending anything the process started (the
    real FFmpeg under a launcher). The process itself is ended through its
    own Popen, which knows whether it has already exited."""
    signal_processes(process_tree(process.pid)[1:], "terminate")
    process.terminate()


def kill(process) -> None:
    """Popen.kill(), after killing anything the process started."""
    signal_processes(process_tree(process.pid)[1:], "kill")
    process.kill()


def raise_current_thread_priority() -> None:
    """Makes the calling thread preempt ordinary work (Windows
    THREAD_PRIORITY_HIGHEST). For light, timing-critical threads only.
    Does nothing elsewhere or if Windows refuses."""
    if os.name != "nt":
        return
    with contextlib.suppress(Exception):
        import ctypes

        kernel32 = ctypes.windll.kernel32
        kernel32.GetCurrentThread.restype = ctypes.c_void_p
        kernel32.SetThreadPriority.argtypes = [ctypes.c_void_p, ctypes.c_int]
        kernel32.SetThreadPriority(kernel32.GetCurrentThread(), 2)  # THREAD_PRIORITY_HIGHEST

