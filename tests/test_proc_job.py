"""What the app starts for its work ends with it, however the app ends: its
FFmpeg and isolated scorers ran on after a crash or End task -- a whole
film's CPU fallback, hours of it (proc.end_with_app)."""
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import psutil
import pytest

pytestmark = pytest.mark.skipif(os.name != "nt", reason="a Windows job object")

ROOT = Path(__file__).resolve().parents[1]

# The "app": starts a long-running program and an isolated child, says their
# process ids, and waits to be killed.
_APP = textwrap.dedent("""
    import sys, threading, time
    sys.path.insert(0, sys.argv[1])
    from vmaf_app.core import isolated, proc

    def main():
        program = proc.popen([sys.executable, "-c", "import time; time.sleep(600)"])
        threading.Thread(target=isolated.run_isolated, args=(time.sleep, 600), kwargs={"what": "test"},
                         daemon=True).start()
        while True:
            children = [p for p in __import__("psutil").Process().children() if p.pid != program.pid]
            if children:
                print(program.pid, children[0].pid, flush=True)
                break
            time.sleep(0.05)
        time.sleep(600)

    if __name__ == "__main__":
        main()
""")


def test_what_the_app_started_ends_when_the_app_is_killed(tmp_path):
    script = tmp_path / "app.py"
    script.write_text(_APP, encoding="utf-8")
    app = subprocess.Popen([sys.executable, str(script), str(ROOT)], stdout=subprocess.PIPE, text=True)
    started = []
    try:
        started = [psutil.Process(int(pid)) for pid in app.stdout.readline().split()]
        assert len(started) == 2 and all(p.is_running() for p in started)
        psutil.Process(app.pid).kill()  # TerminateProcess: no cleanup runs, as in a crash or End task
        _gone, alive = psutil.wait_procs(started, timeout=30)
        assert not alive, [p.pid for p in alive]
    finally:
        for process in started:  # what a failed run would leave behind
            try:
                for child in [process, *process.children(recursive=True)]:
                    child.kill()
            except psutil.Error:
                pass
        app.kill()
