"""The app's temporary files and folders, and the removal of those that runs
left behind.

A run removes its temporary files as it ends, however it ends: done, failed
or cancelled. An app that is itself ended outright (End task, a crash, a
power cut, Windows restarting) cannot, and the pictures of SSIMULACRA2 and
Butteraugli on the CPU alone left 1 to 1.4 GB in the temporary folder per
run at 4K. So each name is one of the app's prefixes, then the id of the
process that made it (prefix), and the next start removes those whose
process no longer runs (remove_leftovers). The C++ app names them the same
way, so either app removes what the other left. Names without a process id
were made by earlier versions: they are removed once nothing has changed
them for a week, as Windows' Disk Cleanup removes temporary files.
"""
from __future__ import annotations

import logging
import os
import re
import shutil
import tempfile
import threading
import time
from pathlib import Path

import psutil

_log = logging.getLogger(__name__)

#: The start of every temporary file or folder name the app uses.
PREFIXES = (
    "videometricslab-perceptual-",  # the pictures of SSIMULACRA2 and Butteraugli on the CPU (perceptual_cpu)
    "vml-isolated-",  # a process of its own's folder (isolated)
    "vmaf_run_", "vmaf_gpu_run_", "vmaf_cpu_run_", "vmaf_xpsnr_run_", "vmaf_resample_",  # FFmpeg's runs (vmaf_runner)
    "vml-vship-pts-",  # FFmpeg's frame timestamps for Vship (perceptual_vship)
    "vml-vmaf-v1-",  # libvmaf's list of features (vmaf_v1_gpu)
)

#: How long a name without a process id is kept untouched before it goes.
_KEPT_WITHOUT_OWNER_SECONDS = 7 * 24 * 3600

#: The process id after a prefix, as prefix() writes it.
_OWNER = re.compile(r"(\d+)-")


def prefix(name: str) -> str:
    """`name`, one of PREFIXES, followed by this process's id: the prefix of
    a temporary file or folder that remove_leftovers can find the owner of."""
    return f"{name}{os.getpid()}-"


def remove_leftovers(folder: Path | None = None) -> int:
    """Removes the app's temporary files and folders in `folder` (the
    temporary folder) whose process no longer runs, and those without a
    process id that nothing has changed for a week. Leaves everything else,
    and anything it cannot remove (a file still open). Returns how many it
    removed."""
    folder = Path(tempfile.gettempdir()) if folder is None else folder
    try:
        entries = list(os.scandir(folder))
    except OSError:
        return 0
    now = time.time()
    removed = 0
    for entry in entries:
        owner = _owner(entry.name)
        if owner is None:
            continue  # not the app's
        if owner >= 0:
            if _running(owner):
                continue
        else:
            try:
                if now - entry.stat(follow_symlinks=False).st_mtime < _KEPT_WITHOUT_OWNER_SECONDS:
                    continue
            except OSError:
                continue
        try:
            if entry.is_dir(follow_symlinks=False):
                shutil.rmtree(entry.path)
            else:
                os.unlink(entry.path)
        except OSError:
            continue
        removed += 1
    return removed


def start_removing_leftovers() -> None:
    """Runs remove_leftovers on a thread of its own, so that a large folder
    does not hold up the start."""
    def remove() -> None:
        """Removes the leftovers, and says how many there were."""
        if removed := remove_leftovers():
            _log.info("Temporary files and folders left by runs that never ended: %d removed", removed)

    threading.Thread(target=remove, name="remove-leftovers", daemon=True).start()


def _running(pid: int) -> bool:
    """Whether the process that made a temporary file or folder still runs."""
    if pid == os.getpid():
        return True
    try:
        return psutil.pid_exists(pid)
    except (OverflowError, ValueError):
        return False  # an id too large for any process


def _owner(name: str) -> int | None:
    """The id of the process that made the temporary file or folder `name`;
    -1 for one of the app's names without one; None for a name that is not
    the app's."""
    for known in PREFIXES:
        if name.startswith(known):
            match = _OWNER.match(name, len(known))
            return int(match.group(1)) if match else -1
    return None
