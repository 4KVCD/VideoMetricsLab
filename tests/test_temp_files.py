"""vmaf_app.core.temp_files: what runs left in the temporary folder."""
import os
import time

import psutil

from vmaf_app.core import temp_files


def _a_process_id_not_running() -> int:
    pid = 4_000_000
    while psutil.pid_exists(pid):
        pid += 4
    return pid


def test_what_runs_ended_outright_left_is_removed_and_nothing_else(tmp_path):
    """An app ended outright (End task, a crash) cannot remove its temporary
    files, and one CPU SSIMULACRA2 run's pictures alone left 1 to 1.4 GB in
    the temporary folder. The next start removes what processes that no
    longer run left, and what earlier versions left without a process id
    once nothing has changed it for a week: nothing of a running process's,
    and nothing that is not the app's."""
    gone, running = _a_process_id_not_running(), os.getpid()
    temp = tmp_path / "temp"
    temp.mkdir()
    pictures = temp / f"videometricslab-perceptual-{gone}-k2x9"
    pictures.mkdir()
    (pictures / "reference-00000001.png").write_bytes(b"png")
    (temp / f"vml-isolated-{gone}-a1b2").mkdir()
    (temp / f"vmaf_gpu_run_{gone}-c3d4").mkdir()
    (temp / f"vml-vship-pts-{gone}-e5f6.txt").write_text("1 1/1000\n")
    (temp / "vml-vmaf-v1-99999999999-i9j0.json").write_text("{}")  # an id too large for any process
    (temp / f"vml-isolated-{running}-g7h8").mkdir()  # this process's, running
    old = temp / "videometricslab-perceptual-oldname1"  # an earlier version's, untouched for 8 days
    old.mkdir()
    eight_days_ago = time.time() - 8 * 24 * 3600
    os.utime(old, (eight_days_ago, eight_days_ago))
    (temp / "vmaf_run_recent99").mkdir()  # an earlier version's, changed today
    (temp / f"other-program-{gone}-x").mkdir()  # not the app's

    assert temp_files.remove_leftovers(temp) == 6
    assert sorted(path.name for path in temp.iterdir()) == [
        f"other-program-{gone}-x", "vmaf_run_recent99", f"vml-isolated-{running}-g7h8"]
