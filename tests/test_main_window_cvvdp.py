"""CVVDP in the Videos tab: its column, availability, display presets, and requests."""
from __future__ import annotations

from pathlib import Path

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QMessageBox

from tests.factories import fake_run_result, fake_video_info
from vmaf_app.core import result_cache
from vmaf_app.core.cvvdp import BUILTIN_PRESETS, DEFAULT_PRESET, CvvdpSettings
from vmaf_app.core.metric_results import MetricProvenance, MetricResultSet, SequenceMetricResult
from vmaf_app.ui import main_window as main_window_module
from vmaf_app.ui.main_window import COL_CVVDP, COL_VMAF, CompletedRun, CvvdpDisplayDialog, MainWindow


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture(autouse=True)
def gpu_present(monkeypatch):
    """A supported GPU unless a test says otherwise, so the suite does not
    depend on the machine running it."""
    monkeypatch.setattr(MainWindow, "_vship_available", staticmethod(lambda: True))


def _cvvdp(settings: CvvdpSettings, score=9.8099) -> SequenceMetricResult:
    parameters = {**dict(settings.spec_parameters()), "gpu_backend": "cuda"}
    return SequenceMetricResult(
        "cvvdp", score, MetricProvenance("Vship/cvvdp", "Vship 5.1.1", "gpu", "cvvdp-vship-gpu-v1", parameters),
        frame=[0, 30], time=[0.0, 1.0], values=[9.9, 9.7],
    )


def _window_with_row(name="test.mp4", *, cvvdp_score: float | None = None):
    win = MainWindow()
    win._source_info = fake_video_info("source.mp4")
    row = win._add_table_row(Path(name))
    row_data = win._rows[row]
    row_data.video_info = fake_video_info(name)
    row_data.extra_metric_keys.add("cvvdp")
    if cvvdp_score is not None:
        result = fake_run_result(name)
        result.merge_metric_results(MetricResultSet([_cvvdp(row_data.cvvdp, cvvdp_score)]))
        row_data.completed_run = CompletedRun(result, name)
    win._set_row_metrics(row)
    return win, row, row_data


def _select(win, row):
    win.distorted_table.selectRow(row)
    win._on_table_selection_changed()


def test_a_cvvdp_score_is_shown_with_what_it_means_and_its_display(qapp):
    win, row, row_data = _window_with_row(cvvdp_score=9.80987)
    cell = win.distorted_table.item(row, COL_CVVDP)
    assert cell.text() == "9.810"
    assert "JOD" in cell.toolTip() and DEFAULT_PRESET.name in cell.toolTip()
    assert "Metric Graphs" in cell.toolTip()
    # The score counts as calculated: the next run does not redo it.
    assert win._reusable_results(row_data).has("cvvdp")
    win.close()


def test_rows_start_with_the_default_display_and_request_cvvdp_with_it(qapp):
    win, _row, row_data = _window_with_row()
    assert row_data.cvvdp == DEFAULT_PRESET.settings
    request = win._analysis_request(row_data)
    spec = next(spec for spec in request.metrics if spec.key == "cvvdp")
    assert dict(spec.parameters)["display"]["peak_luminance"] == 200
    win.close()


def test_choosing_another_display_drops_only_the_cvvdp_score(qapp):
    win, row, row_data = _window_with_row(cvvdp_score=9.5)
    _select(win, row)
    hdr = BUILTIN_PRESETS[2]
    win.cvvdp_preset_combo.setCurrentIndex(win.cvvdp_preset_combo.findData(hdr.name))
    assert row_data.cvvdp == hdr.settings
    assert row_data.completed_run is not None
    assert row_data.completed_run.result.has_metric("vmaf")
    assert not row_data.completed_run.result.has_metric("cvvdp")
    assert win.distorted_table.item(row, COL_VMAF).text() != ""
    assert win.distorted_table.item(row, COL_CVVDP).checkState() == Qt.Checked
    assert not win._reusable_results(row_data).has("cvvdp")
    win.close()


def _dialog_answers(monkeypatch, action, *, name=None, ambient_lux=None, peak=None):
    """Makes the display dialog behave as if the user edited it and clicked
    `action`'s button ("save", "save_new" or "apply"), through its own
    validation. Returns the list of warnings it raised."""
    warnings = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *args, **kwargs: warnings.append(args[2]))

    def exec_(dialog):
        if name is not None:
            dialog.name_edit.setText(name)
        if ambient_lux is not None:
            dialog.ambient_spin.setValue(ambient_lux)
        if peak is not None:
            dialog.peak_spin.setValue(peak)
        dialog._finish(action)
        return dialog.result()

    monkeypatch.setattr(CvvdpDisplayDialog, "exec", exec_)
    return warnings


def test_a_run_carries_each_rows_display_and_skips_a_row_already_scored(qapp, monkeypatch):
    win, row, row_data = _window_with_row()
    row_data.cvvdp = BUILTIN_PRESETS[1].settings
    scored = win._add_table_row(Path("scored.mp4"))
    win._rows[scored].video_info = fake_video_info("scored.mp4")
    win._rows[scored].extra_metric_keys.add("cvvdp")
    result = fake_run_result("scored.mp4")
    result.merge_metric_results(MetricResultSet([_cvvdp(win._rows[scored].cvvdp)]))
    win._rows[scored].completed_run = CompletedRun(result, "scored")
    for r in (row, scored):
        for key in ("psnr", "ssim", "xpsnr", "vmaf_neg"):
            win._rows[r].options.set_metric_enabled(key, False)
    monkeypatch.setattr(main_window_module.VmafWorker, "start", lambda self: None)
    monkeypatch.setattr(main_window_module, "validate_video_pair", lambda *a: None)
    win._on_run_clicked()
    (job,) = win._worker.scheduler.jobs
    assert "cvvdp" in job.metric_keys and job.cvvdp == BUILTIN_PRESETS[1].settings
    win._worker = None
    win.close()


def test_a_cached_cvvdp_score_comes_back_only_for_its_own_display(qapp, tmp_path, monkeypatch):
    monkeypatch.setattr(result_cache, "cache_dir", lambda: tmp_path)
    source, distorted = tmp_path / "source.mp4", tmp_path / "test.mp4"
    source.write_bytes(b"s" * 100)
    distorted.write_bytes(b"d" * 50)
    win = MainWindow()
    win._source_info = fake_video_info(str(source))
    win._source_info.path = source
    row = win._add_table_row(distorted)
    row_data = win._rows[row]
    row_data.video_info = fake_video_info(str(distorted))
    row_data.extra_metric_keys.add("cvvdp")
    result = fake_run_result(distorted, source=source)
    result.merge_metric_results(MetricResultSet([_cvvdp(row_data.cvvdp, 9.25)]))
    result_cache.store(source, distorted, result, "test", win._analysis_request(row_data), tmp_path)

    assert win._try_load_cached_result(row)
    assert win.distorted_table.item(row, COL_CVVDP).text() == "9.250"
    loaded = row_data.completed_run.result.sequence_metric("cvvdp")
    assert loaded.has_timeline and list(loaded.values) == pytest.approx([9.9, 9.7])

    row_data.completed_run = None
    row_data.cvvdp = BUILTIN_PRESETS[3].settings
    assert win._try_load_cached_result(row)  # VMAF is still cached for this recipe
    assert not row_data.completed_run.result.has_metric("cvvdp")
    win.close()


def test_a_loaded_result_file_brings_the_display_its_cvvdp_was_scored_for(qapp, tmp_path, monkeypatch):
    from vmaf_app.core.run_io import save_run

    tv = BUILTIN_PRESETS[5].settings
    result = fake_run_result("test.mp4")
    result.merge_metric_results(MetricResultSet([_cvvdp(tv, 8.5)]))
    path = tmp_path / "run.metrics.json"
    save_run(result, path, label="run")
    monkeypatch.setattr(main_window_module.QFileDialog, "getOpenFileName", lambda *a, **k: (str(path), ""))
    win = MainWindow()
    win._on_load_saved_run()
    row_data = win._rows[-1]
    assert row_data.cvvdp == tv
    assert "cvvdp" in row_data.extra_metric_keys
    assert win.distorted_table.item(len(win._rows) - 1, COL_CVVDP).text() == "8.500"
    win.close()


def _row_with_cached_scores(win, tmp_path, *keys):
    """A row whose recipe has VMAF plus `keys` saved, none of `keys` ticked."""
    from vmaf_app.core.metric_results import FrameMetricResult

    source, distorted = tmp_path / "source.mp4", tmp_path / "test.mp4"
    source.write_bytes(b"s" * 100)
    distorted.write_bytes(b"d" * 50)
    win._source_info = fake_video_info(str(source))
    win._source_info.path = source
    row = win._add_table_row(distorted)
    row_data = win._rows[row]
    row_data.video_info = fake_video_info(str(distorted))
    stored = fake_run_result(distorted, source=source)
    extra = []
    for key, value in (("ssimulacra2", 80.0), ("butteraugli", 1.5)):
        if key in keys:
            extra.append(FrameMetricResult(
                key, list(range(10)), [i / 30 for i in range(10)], [value] * 10,
                MetricProvenance(f"Vship/{key}", "5.1.1", "gpu", f"{key}-vship-gpu-v1")))
    if "cvvdp" in keys:
        extra.append(_cvvdp(row_data.cvvdp, 9.42))
    stored.merge_metric_results(MetricResultSet(extra))
    ticked = set(row_data.extra_metric_keys)
    row_data.extra_metric_keys |= set(keys)  # store as a run that calculated them would
    result_cache.store(source, distorted, stored, "test", win._analysis_request(row_data), tmp_path)
    row_data.extra_metric_keys = ticked
    return row, row_data, source, distorted


def test_saved_scores_show_even_when_their_column_is_not_ticked(qapp, tmp_path, monkeypatch):
    """Brian: saved scores should show whether or not their metric is ticked.
    The lookup used to ask only for ticked metrics, so a saved SSIMULACRA2,
    Butteraugli or CVVDP stayed hidden behind an empty tick box."""
    monkeypatch.setattr(result_cache, "cache_dir", lambda: tmp_path)
    win = MainWindow()
    row, row_data, *_ = _row_with_cached_scores(win, tmp_path, "ssimulacra2", "butteraugli", "cvvdp")
    assert not {"ssimulacra2", "butteraugli", "cvvdp"} & set(win._requested_metrics(row_data))
    assert win._try_load_cached_result(row)
    assert win.distorted_table.item(row, main_window_module.COL_SSIMULACRA2).text() == "80.00"
    assert win.distorted_table.item(row, main_window_module.COL_BUTTERAUGLI).text() == "1.5000"
    assert win.distorted_table.item(row, COL_CVVDP).text() == "9.420"
    # Shown, but not asked for: the next run does not calculate them.
    assert not {"ssimulacra2", "butteraugli", "cvvdp"} & set(win._requested_metrics(row_data))
    win.close()


def test_recalculating_does_not_delete_saved_scores_of_unticked_metrics(qapp, tmp_path, monkeypatch):
    """Showing unticked saved scores must not widen "Recalculate selected
    metrics": it still clears only what the row asks for."""
    monkeypatch.setattr(result_cache, "cache_dir", lambda: tmp_path)
    win = MainWindow()
    row, _row_data, *_ = _row_with_cached_scores(win, tmp_path, "ssimulacra2")
    win._recompute_rows([row])
    win._file_writes.wait_until_idle(10)
    assert list(tmp_path.rglob("ssimulacra2_*.npz")), "an unticked metric's saved score was deleted"
    assert not list(tmp_path.rglob("vmaf_*.npz"))
    win.close()


def _two_rows_on_desk(monkeypatch):
    win, first, first_data = _window_with_row("a.mp4")
    second = win._add_table_row(Path("b.mp4"))
    second_data = win._rows[second]
    second_data.video_info = fake_video_info("b.mp4")
    _select(win, first)
    _dialog_answers(monkeypatch, "save_new", name="Desk", peak=300)
    win._on_cvvdp_add_preset()
    win.distorted_table.selectAll()
    win._on_table_selection_changed()
    win.cvvdp_preset_combo.setCurrentIndex(win.cvvdp_preset_combo.findData("Desk"))
    _select(win, first)
    return win, first_data, second_data


def test_renaming_a_preset_keeps_other_videos_on_it_without_asking(qapp, monkeypatch):
    win, _first, second_data = _two_rows_on_desk(monkeypatch)
    monkeypatch.setattr(QMessageBox, "question", lambda *args, **kwargs: pytest.fail("nothing to ask"))
    _dialog_answers(monkeypatch, "save", name="Desk monitor")
    win._on_cvvdp_edit_display()
    assert second_data.cvvdp_preset == "Desk monitor"
    assert second_data.cvvdp.display.peak_luminance == 300
    win.close()


def test_an_empty_cvvdp_cell_names_the_scores_saved_for_other_displays(qapp, tmp_path, monkeypatch):
    """A video whose display differed from its earlier run showed no CVVDP
    score and no hint that one had been saved, or for which display."""
    monkeypatch.setattr(result_cache, "cache_dir", lambda: tmp_path)
    monkeypatch.setattr(main_window_module.ProbeWorker, "start", lambda worker: worker.run())
    source, distorted = tmp_path / "source.mp4", tmp_path / "test.mp4"
    source.write_bytes(b"s" * 100)
    distorted.write_bytes(b"d" * 50)
    win = MainWindow()
    win._settings.use_cache = True
    win._source_info = fake_video_info(str(source))
    win._source_info.path = source
    row = win._add_table_row(distorted)
    row_data = win._rows[row]
    row_data.video_info = fake_video_info(str(distorted))
    row_data.extra_metric_keys.add("cvvdp")
    scored_on = BUILTIN_PRESETS[3]
    row_data.cvvdp = scored_on.settings
    result = fake_run_result(distorted, source=source)
    result.merge_metric_results(MetricResultSet([_cvvdp(row_data.cvvdp, 9.25)]))
    result_cache.store(source, distorted, result, "test", win._analysis_request(row_data), tmp_path)

    row_data.cvvdp = DEFAULT_PRESET.settings
    win._start_cache_lookup([distorted])
    item = win.distorted_table.item(row, COL_CVVDP)
    assert item.text() == ""  # never shown as this display's score
    assert f"{scored_on.name}: 9.250 JOD" in item.toolTip()
    assert win.distorted_table.item(row, COL_VMAF).text() != ""  # the rest of the saved run came back

    row_data.completed_run = None
    row_data.cvvdp = scored_on.settings
    win._start_cache_lookup([distorted])
    assert win.distorted_table.item(row, COL_CVVDP).text() == "9.250"
    row_data.completed_run = None
    win._set_row_metrics(row)
    assert "Saved for other displays" not in win.distorted_table.item(row, COL_CVVDP).toolTip()
    win.close()
