from pathlib import Path

from PySide6.QtWidgets import QApplication

from vmaf_app.core.models import VmafOptions
from vmaf_app.ui import probe_worker as probe_worker_module
from vmaf_app.ui.probe_worker import ProbeWorker


def test_unexpected_probe_failure_is_reported_and_worker_always_finishes(monkeypatch):
    app = QApplication.instance() or QApplication([])
    path = Path("broken.mp4")
    monkeypatch.setattr(
        probe_worker_module, "probe_video",
        lambda _path, **kwargs: (_ for _ in ()).throw(TimeoutError("unexpected timeout")),
    )
    worker = ProbeWorker(
        [path], None, False, {path: VmafOptions()}
    )
    errors = []
    finished = []
    worker.probed.connect(lambda _path, info, error: errors.append((info, error)))
    worker.finished_all.connect(lambda: finished.append(True))

    worker.run()

    assert errors == [(None, "unexpected timeout")]
    assert finished == [True]
    assert app is not None
