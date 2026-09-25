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
from vmaf_app.core.models import ResampleTarget
from vmaf_app.core.settings import Settings
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
    parameters = {**dict(settings.spec_parameters()), "gpu_vendor": "nvidia"}
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


@pytest.mark.parametrize("case", ["subsample", "no gpu", "round trip"])
def test_cvvdp_is_na_where_it_cannot_be_calculated(qapp, monkeypatch, case):
    win, row, row_data = _window_with_row()
    if case == "subsample":
        row_data.options.n_subsample = 3
        expected = "subsample"
    elif case == "no gpu":
        monkeypatch.setattr(MainWindow, "_vship_available", staticmethod(lambda: False))
        expected = "GPU"
    else:
        row_data.options.resample_test = ResampleTarget(width=1280, label="720p")
        expected = "round-trip"
    win._set_row_metrics(row)
    cell = win.distorted_table.item(row, COL_CVVDP)
    assert cell.text() == "n/a" and not cell.flags() & Qt.ItemIsUserCheckable
    assert expected in cell.toolTip()
    assert "cvvdp" not in win._requested_metrics(row_data)
    assert "vmaf" in win._requested_metrics(row_data)
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


def test_the_resize_box_and_apply_without_saving_change_the_selected_rows(qapp, monkeypatch):
    win, row, row_data = _window_with_row()
    _select(win, row)
    win.cvvdp_resize_check.setChecked(True)
    assert row_data.cvvdp.resize_to_display
    # Scaling is the video's own setting, not part of the preset.
    assert win.cvvdp_preset_combo.currentData() == DEFAULT_PRESET.name

    _dialog_answers(monkeypatch, "apply", ambient_lux=15.0)
    win._on_cvvdp_edit_display()
    assert row_data.cvvdp.display.ambient_lux == 15.0 and row_data.cvvdp.resize_to_display
    assert "15 lux" in win.cvvdp_display_label.text()
    assert win._settings.cvvdp_presets == []  # nothing saved
    win.close()


def test_the_display_dialog_reads_back_what_it_was_given(qapp):
    display = BUILTIN_PRESETS[4].settings.display  # the 65-inch TV
    dialog = CvvdpDisplayDialog(display)
    assert dialog.display() == display
    # 1.98 m from a 65-inch 16:9 screen (0.81 m tall) is 2.45 heights.
    assert dialog.distance_note.text() == "= 2.45 x screen height"
    dialog.distance_spin.setValue(1.0)
    assert dialog.distance_note.text() == "= 1.24 x screen height"
    # A built-in display offers saving as new and applying, not renaming.
    assert dialog.save_button is None and dialog.apply_button is not None
    assert dialog.name_edit.text() == ""


def test_the_panel_has_add_preset_and_no_save_as_preset_button(qapp):
    win, _row, _row_data = _window_with_row()
    assert win.cvvdp_add_btn.text() == "Add preset..."
    assert not hasattr(win, "cvvdp_save_btn")
    new_form = CvvdpDisplayDialog(DEFAULT_PRESET.settings.display, new_preset=True)
    assert new_form.apply_button is None and new_form.save_button is None
    win.close()


def test_adding_a_preset_makes_it_the_default_for_new_videos(qapp, monkeypatch):
    win, row, row_data = _window_with_row()
    _select(win, row)
    _dialog_answers(monkeypatch, "save_new", name="My monitor", peak=350)
    win._on_cvvdp_add_preset()

    # Making a preset is not choosing it: the selected video keeps its display.
    assert row_data.cvvdp == DEFAULT_PRESET.settings
    assert win.cvvdp_preset_combo.currentData() == DEFAULT_PRESET.name
    assert win.cvvdp_preset_combo.findData("My monitor") >= 0
    assert win._settings.cvvdp_default_preset == "My monitor"
    assert win.settings_cvvdp_default.currentData() == "My monitor"
    win.cvvdp_preset_combo.setCurrentIndex(win.cvvdp_preset_combo.findData("My monitor"))
    assert row_data.cvvdp.display.peak_luminance == 350
    new_row = win._add_table_row(Path("next.mp4"))
    assert win._rows[new_row].cvvdp.display.peak_luminance == 350
    # It survives a restart.
    assert Settings.load().cvvdp_default_preset == "My monitor"
    restarted = MainWindow()
    assert restarted._default_cvvdp.display.peak_luminance == 350
    restarted.close()

    # Deleting it puts the built-in default back; the rows keep their settings.
    monkeypatch.setattr(QMessageBox, "question", lambda *args, **kwargs: QMessageBox.Yes)
    win._on_cvvdp_delete_preset()
    assert win._settings.cvvdp_presets == [] and win._settings.cvvdp_default_preset == ""
    assert win._default_cvvdp == DEFAULT_PRESET.settings
    assert row_data.cvvdp.display.peak_luminance == 350
    assert win.cvvdp_preset_combo.currentData() is None  # now "Custom"
    win.close()


def test_the_display_editor_renames_and_updates_your_preset(qapp, monkeypatch):
    win, row, row_data = _window_with_row()
    _select(win, row)
    _dialog_answers(monkeypatch, "save_new", name="Desk", peak=300)
    win._on_cvvdp_add_preset()
    win.cvvdp_preset_combo.setCurrentIndex(win.cvvdp_preset_combo.findData("Desk"))

    _dialog_answers(monkeypatch, "save", name="Desk monitor", peak=320)
    win._on_cvvdp_edit_display()
    assert [p["name"] for p in win._settings.cvvdp_presets] == ["Desk monitor"]
    assert win._settings.cvvdp_default_preset == "Desk monitor"  # the default follows the rename
    assert row_data.cvvdp.display.peak_luminance == 320
    assert win.cvvdp_preset_combo.currentData() == "Desk monitor"
    assert "Renamed" in win.status_label.text()
    win.close()


def test_the_display_editor_saves_a_copy_as_a_new_preset(qapp, monkeypatch):
    win, row, row_data = _window_with_row()
    _select(win, row)
    _dialog_answers(monkeypatch, "save_new", name="Desk", peak=300)
    win._on_cvvdp_add_preset()
    win.cvvdp_preset_combo.setCurrentIndex(win.cvvdp_preset_combo.findData("Desk"))
    _dialog_answers(monkeypatch, "save_new", name="Desk, dark room", ambient_lux=0)
    win._on_cvvdp_edit_display()
    assert [p["name"] for p in win._settings.cvvdp_presets] == ["Desk", "Desk, dark room"]
    assert row_data.cvvdp.display.ambient_lux == 0 and row_data.cvvdp.display.peak_luminance == 300
    assert win.cvvdp_preset_combo.currentData() == "Desk, dark room"
    win.close()


@pytest.mark.parametrize("name", ["", "  ", BUILTIN_PRESETS[0].name, "Desk"])
def test_a_preset_needs_a_new_name(qapp, monkeypatch, name):
    win, row, row_data = _window_with_row()
    _select(win, row)
    _dialog_answers(monkeypatch, "save_new", name="Desk")
    win._on_cvvdp_add_preset()
    warnings = _dialog_answers(monkeypatch, "save_new", name=name, peak=999)
    win._on_cvvdp_add_preset()
    assert warnings, "the dialog should refuse the name and stay open"
    assert [p["name"] for p in win._settings.cvvdp_presets] == ["Desk"]
    assert row_data.cvvdp.display.peak_luminance != 999
    win.close()


def test_the_settings_tab_chooses_the_default_display_and_default_tick(qapp):
    win = MainWindow()
    phone = BUILTIN_PRESETS[-1]
    win.settings_cvvdp_default.setCurrentIndex(win.settings_cvvdp_default.findData(phone.name))
    win.settings_default_cvvdp.setChecked(True)
    assert win._settings.cvvdp_default_preset == phone.name
    row = win._add_table_row(Path("new.mp4"))
    assert win._rows[row].cvvdp == phone.settings
    assert "cvvdp" in win._rows[row].extra_metric_keys
    win.close()


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
    (job,) = win._worker._jobs
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


def test_the_display_editor_stops_at_8k(qapp):
    """Scaled to fill, Vship works at the display's resolution: an 8K display
    took CVVDP to +7.3 GB of VRAM on a 4K video, and 16384x16384 to +15.9 GB."""
    dialog = CvvdpDisplayDialog(DEFAULT_PRESET.settings.display)
    dialog.width_spin.setValue(16384)
    dialog.height_spin.setValue(16384)
    assert (dialog.display().width, dialog.display().height) == (8192, 8192)
    dialog.width_spin.setValue(7680)
    assert dialog.display().width == 7680


def test_cached_ssimulacra2_appears_while_cvvdp_is_ticked_but_never_calculated(qapp, tmp_path, monkeypatch):
    """Brian's Beekeeper rows: VMAF showed, SSIMULACRA2/Butteraugli ticked but
    empty. A cached answer had to hold every ticked metric, and CVVDP had
    never been calculated, so the answer carrying SSIMULACRA2 was dropped."""
    from vmaf_app.core.metric_results import FrameMetricResult

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
    row_data.extra_metric_keys |= {"ssimulacra2"}
    stored = fake_run_result(distorted, source=source)
    stored.merge_metric_results(MetricResultSet([FrameMetricResult(
        "ssimulacra2", list(range(10)), [i / 30 for i in range(10)], [80.0] * 10,
        MetricProvenance("Vship/ssimulacra2", "5.1.1", "gpu", "ssimulacra2-vship-gpu-v1"))]))
    result_cache.store(source, distorted, stored, "test", win._analysis_request(row_data), tmp_path)
    # The row shows VMAF only, as after adding it with SSIMULACRA2 unticked.
    row_data.completed_run = CompletedRun(fake_run_result(distorted, source=source), "test")
    row_data.extra_metric_keys |= {"cvvdp"}
    cached, _label = result_cache.load_cached(source, distorted, win._analysis_request(row_data), tmp_path)
    win._on_cached_found(distorted, cached, "test")
    assert win.distorted_table.item(row, main_window_module.COL_SSIMULACRA2).text() == "80.00"
    assert win.distorted_table.item(row, COL_CVVDP).text() == ""  # still to calculate
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


def test_a_saved_unticked_score_is_added_to_a_row_already_showing_others(qapp, tmp_path, monkeypatch):
    monkeypatch.setattr(result_cache, "cache_dir", lambda: tmp_path)
    win = MainWindow()
    row, row_data, source, distorted = _row_with_cached_scores(win, tmp_path, "ssimulacra2")
    row_data.completed_run = CompletedRun(fake_run_result(distorted, source=source), "test")
    cached, _label = result_cache.load_cached(
        source, distorted, win._analysis_request(row_data), tmp_path,
        main_window_module.displayable_metric_specs(row_data.options, row_data.cvvdp))
    win._on_cached_found(distorted, cached, "test")
    assert win.distorted_table.item(row, main_window_module.COL_SSIMULACRA2).text() == "80.00"
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



class _PendingLookup:
    """Stands in for ProbeWorker: records what each cache lookup was asked
    about, and stays "running" so the next lookup has to replace it."""

    started: list = []

    def __init__(self, paths, *args, **kwargs):
        self.paths = list(paths)
        self.cancelled = False
        _PendingLookup.started.append(self)
        self.cached_found = self.other_cvvdp_found = self.finished_all = self
        self.probed = self

    def connect(self, *_args):
        pass

    def start(self):
        pass

    def isRunning(self):
        return not self.cancelled

    def cancel(self):
        self.cancelled = True

    def deleteLater(self):
        pass


def test_editing_one_row_while_saved_results_load_keeps_asking_for_the_others(qapp, monkeypatch):
    """Editing one row while saved results were still loading cancelled the
    lookup for every row but re-asked only for the edited one: the others
    stayed empty and the next run recalculated them in full."""
    _PendingLookup.started = []
    monkeypatch.setattr(main_window_module, "ProbeWorker", _PendingLookup)
    win = MainWindow()
    win._source_info = fake_video_info("source.mp4")
    paths = [Path(f"{name}.mp4") for name in "abc"]
    for path in paths:
        win._add_table_row(path)
    win._start_cache_lookup(paths)
    win._start_cache_lookup([paths[2]])  # c's display changed while loading
    assert _PendingLookup.started[0].cancelled
    assert _PendingLookup.started[-1].paths == paths
    # Recalculate drops its own rows' pending answers but re-asks for the rest.
    win._recompute_rows([0])
    assert _PendingLookup.started[-1].paths == paths[1:]
    win._probe_workers.clear()
    win.close()


@pytest.mark.parametrize("preset", BUILTIN_PRESETS, ids=lambda preset: preset.name)
def test_the_display_editor_changes_nothing_that_was_not_edited(qapp, preset):
    """0.7472 m (the default display's distance) came back as 0.747: "Apply
    without saving" with nothing changed dropped the CVVDP score and made
    the dropdown say "Custom"."""
    dialog = CvvdpDisplayDialog(preset.settings.display)
    assert dialog.display() == preset.settings.display
    dialog.distance_spin.setValue(1.2345)  # an edit is still read as typed
    assert dialog.display().viewing_distance_m == 1.2345


def test_apply_without_saving_and_no_edit_keeps_the_cvvdp_score(qapp, monkeypatch):
    win, row, row_data = _window_with_row(cvvdp_score=9.81)
    _select(win, row)
    _dialog_answers(monkeypatch, "apply")
    win._on_cvvdp_edit_display()
    assert row_data.completed_run.result.has_metric("cvvdp")
    assert win.cvvdp_preset_combo.currentData() == DEFAULT_PRESET.name
    win.close()


def test_recalculating_keeps_the_saved_scores_of_unticked_ffmpeg_metrics(qapp, tmp_path, monkeypatch):
    """Recalculate cleared every FFmpeg metric whenever a libvmaf one was
    ticked: a saved PSNR on show but unticked was deleted with VMAF."""
    from vmaf_app.core.metric_results import FrameMetricResult

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
    stored = fake_run_result(distorted, source=source)
    stored.merge_metric_results(MetricResultSet([FrameMetricResult(
        "psnr", list(range(10)), [i / 30 for i in range(10)], [40.0] * 10,
        MetricProvenance("ffmpeg/libvmaf", "ffmpeg 9.0.1", "cpu", "ffmpeg-libvmaf-v1"))]))
    result_cache.store(source, distorted, stored, "test", win._analysis_request(row_data), tmp_path)
    assert list(tmp_path.rglob("psnr_*.npz")), f"setup: PSNR not saved; metrics {stored.metric_results.keys()}"
    row_data.options.set_metric_enabled("psnr", False)
    win._recompute_rows([row])
    win._file_writes.wait_until_idle(10)
    assert list(tmp_path.rglob("psnr_*.npz")), "the unticked PSNR's saved score was deleted"
    assert not list(tmp_path.rglob("vmaf_*.npz"))
    win.close()


def test_a_run_keeps_showing_saved_scores_of_unticked_metrics(qapp):
    """A row showing VMAF and an unticked saved CVVDP: after running PSNR,
    the result held only what the run calculated and CVVDP went back to a
    tick box, although its score was still saved."""
    from vmaf_app.core.models import clone_options

    win, row, row_data = _window_with_row(cvvdp_score=9.42)
    win._source_info.path = row_data.completed_run.result.source
    row_data.extra_metric_keys.discard("cvvdp")
    fresh = fake_run_result("test.mp4", vmaf=91.0)
    win._job_rows = [row_data]
    win._job_cache_options = [clone_options(row_data.options)]
    win._job_cvvdp = [row_data.cvvdp]
    win._on_job_finished(0, fresh)
    assert win.distorted_table.item(row, COL_CVVDP).text() == "9.420"
    assert win.distorted_table.item(row, COL_VMAF).text() == "91.00"
    assert not fresh.has_metric("cvvdp"), "the result handed to the cache write must not change"
    win.close()


def test_recalculating_keeps_unticked_saved_scores_on_show(qapp):
    win, row, row_data = _window_with_row(cvvdp_score=9.42)
    row_data.extra_metric_keys.discard("cvvdp")
    win._recompute_rows([row])
    assert win.distorted_table.item(row, COL_CVVDP).text() == "9.420"
    assert not row_data.completed_run.result.has_metric("vmaf")
    assert not row_data.completed_run.result.frames.has("vmaf")
    win._file_writes.wait_until_idle(10)
    win.close()


def test_choosing_a_preset_leaves_scale_to_fill_alone_and_saving_one_does_not_store_it(qapp, monkeypatch):
    """Presets used to carry "Scale the video to fill the display": choosing
    a built-in switched it off without a word, and "Add preset..." saved the
    selected video's setting invisibly -- then turned it on for every new
    video once that preset became the default."""
    win, row, row_data = _window_with_row()
    _select(win, row)
    win.cvvdp_resize_check.setChecked(True)
    tv = BUILTIN_PRESETS[4]
    win.cvvdp_preset_combo.setCurrentIndex(win.cvvdp_preset_combo.findData(tv.name))
    assert row_data.cvvdp.display == tv.settings.display and row_data.cvvdp.resize_to_display
    assert win.cvvdp_resize_check.isChecked()

    _dialog_answers(monkeypatch, "save_new", name="Scaled TV", peak=900)
    win._on_cvvdp_add_preset()
    assert win._settings.cvvdp_presets[0]["settings"]["resize_to_display"] is False
    new_row = win._add_table_row(Path("next.mp4"))
    assert win._rows[new_row].cvvdp.display.peak_luminance == 900
    assert not win._rows[new_row].cvvdp.resize_to_display
    win.close()



def test_a_mixed_selection_shows_as_mixed_and_one_choice_sets_every_row(qapp):
    """With A on the default display and B on HDR, selecting both showed A's
    preset: re-choosing it fired nothing and B kept HDR. The resize box
    likewise showed A's state, so a click set both rows to the opposite of
    A's value."""
    from PySide6.QtWidgets import QTableWidgetSelectionRange

    win, _row_a, a = _window_with_row("a.mp4")
    row_b = win._add_table_row(Path("b.mp4"))
    b = win._rows[row_b]
    b.cvvdp = BUILTIN_PRESETS[2].settings
    b.cvvdp = type(b.cvvdp)(b.cvvdp.display, True)
    win.distorted_table.setRangeSelected(
        QTableWidgetSelectionRange(0, 0, 1, win.distorted_table.columnCount() - 1), True)
    win._on_table_selection_changed()
    assert win.cvvdp_preset_combo.currentData() is None
    assert "Mixed" in win.cvvdp_preset_combo.currentText()
    assert win.cvvdp_resize_check.checkState() == Qt.PartiallyChecked

    win.cvvdp_preset_combo.setCurrentIndex(win.cvvdp_preset_combo.findData(DEFAULT_PRESET.name))
    assert a.cvvdp.display == b.cvvdp.display == DEFAULT_PRESET.settings.display
    win.cvvdp_resize_check.click()  # from partly ticked to ticked: every row
    assert a.cvvdp.resize_to_display and b.cvvdp.resize_to_display
    assert win.cvvdp_resize_check.checkState() == Qt.Checked and not win.cvvdp_resize_check.isTristate()
    win.close()


def test_a_new_resolution_test_row_shows_n_a_for_the_perceptual_metrics(qapp, monkeypatch):
    from PySide6.QtWidgets import QInputDialog

    win = MainWindow()
    win._source_info = fake_video_info("source.mp4")
    win._default_extra_metric_keys |= {"ssimulacra2", "cvvdp"}
    monkeypatch.setattr(QInputDialog, "getItem", lambda *a, **k: ("720p", True))
    win._on_add_resample_test()
    row = len(win._rows) - 1
    for column in (main_window_module.COL_SSIMULACRA2, main_window_module.COL_BUTTERAUGLI, COL_CVVDP):
        assert win.distorted_table.item(row, column).text() == "n/a"
    win.close()


def test_adding_a_preset_keeps_the_selected_videos_cvvdp_score(qapp, monkeypatch):
    """"Add preset..." gave the new display to the selected video too,
    silently dropping its CVVDP score."""
    win, row, row_data = _window_with_row(cvvdp_score=9.81)
    _select(win, row)
    _dialog_answers(monkeypatch, "save_new", name="Dark room", ambient_lux=0)
    win._on_cvvdp_add_preset()
    assert row_data.completed_run.result.has_metric("cvvdp")
    assert win.distorted_table.item(row, COL_CVVDP).text() == "9.810"
    assert "choose it" in win.status_label.text()
    win.close()


def test_a_finished_lookup_does_not_overwrite_a_message_with_ready(qapp):
    """The confirmation after "Add preset..." was replaced by "Ready." as
    soon as the cache lookup it started finished."""
    win = MainWindow()
    win.status_label.setText('Saved the CVVDP preset "Mine".')
    win._on_probe_finished()
    assert win.status_label.text() == 'Saved the CVVDP preset "Mine".'
    win.status_label.setText("Reading 2 video(s)...")
    win._on_probe_finished()
    assert win.status_label.text() == "Ready."
    win.close()


def test_the_header_ticks_show_the_settings_defaults_for_cvvdp(qapp):
    settings = Settings.load()
    settings.default_compute_cvvdp = True
    settings.default_compute_ssimulacra2 = True
    settings.save()
    win = MainWindow()
    assert win.metric_header.is_checked(COL_CVVDP)
    assert win.metric_header.is_checked(main_window_module.COL_SSIMULACRA2)
    assert not win.metric_header.is_checked(main_window_module.COL_BUTTERAUGLI)
    win.close()


def test_a_test_both_companion_row_copies_the_metric_choices(qapp):
    from vmaf_app.core.models import VideoInfo

    win = MainWindow()
    win._default_extra_metric_keys = {"ssimulacra2", "butteraugli", "cvvdp"}  # the new defaults
    win._source_info = fake_video_info("source.mp4")
    row = win._add_table_row(Path("test.mp4"))
    original = win._rows[row]
    original.video_info = VideoInfo(Path("test.mp4"), 1280, 720, 30.0, 5.0, 150, "h264")
    original.extra_metric_keys = {"ssimulacra2", "cvvdp"}
    original.metric_backends["ssimulacra2"] = "cpu"
    original.cvvdp = BUILTIN_PRESETS[1].settings
    win._add_opposite_scale_direction_rows([row])
    companion = win._rows[-1]
    assert companion.scale_direction_pinned
    assert companion.extra_metric_keys == {"ssimulacra2", "cvvdp"}
    assert companion.metric_backends["ssimulacra2"] == "cpu"
    assert companion.cvvdp == BUILTIN_PRESETS[1].settings
    companion_row = len(win._rows) - 1
    # The cells show the copied choices, not the defaults it was drawn with.
    assert win.distorted_table.item(companion_row, main_window_module.COL_BUTTERAUGLI).checkState() == Qt.Unchecked
    assert win.distorted_table.item(companion_row, COL_CVVDP).checkState() == Qt.Checked
    companion.extra_metric_keys.add("butteraugli")
    assert "butteraugli" not in original.extra_metric_keys  # copies, not shared
    win.close()


def test_changing_the_display_keeps_the_graph_series_colour_and_visibility(qapp):
    win, row, row_data = _window_with_row(cvvdp_score=9.5)
    graph = win.graph_panel
    graph.add_run(row_data.completed_run.result, "test", identity=row_data.completed_run.graph_identity)
    (series_id, entry), = graph._entries.items()
    colour = entry.color
    graph.set_series_visible(series_id, False)
    _select(win, row)
    win.cvvdp_preset_combo.setCurrentIndex(win.cvvdp_preset_combo.findData(BUILTIN_PRESETS[2].name))
    (series_id_after, entry_after), = graph._entries.items()
    assert series_id_after == series_id and entry_after.color == colour and not entry_after.visible
    assert not entry_after.result.has_metric("cvvdp")

    graph.remove_run(series_id)  # removed with the x: stays removed
    win.cvvdp_preset_combo.setCurrentIndex(win.cvvdp_preset_combo.findData(DEFAULT_PRESET.name))
    assert graph._entries == {}
    win.close()


def test_a_fresh_install_ticks_every_metric(qapp):
    """Every metric starts ticked, for new videos and in the column headers
    (the suite's settings file pins the older four-metric default)."""
    from vmaf_app.core.metrics import METRICS

    Settings.path().unlink()
    fresh = Settings()
    assert all(getattr(fresh, f"default_compute_{metric.key}") for metric in METRICS)
    win = MainWindow()
    win._source_info = fake_video_info("source.mp4")
    row = win._add_table_row(Path("test.mp4"))
    assert set(win._selected_metrics(win._rows[row])) == {metric.key for metric in METRICS}
    for item in main_window_module._METRIC_COLUMNS:
        assert win.metric_header.is_checked(item.column), item.key
    win.close()


def test_a_lookup_whose_thread_ended_but_whose_answers_are_queued_is_still_asked_again(qapp, monkeypatch):
    """The thread can exit while its answers still wait in the UI thread's
    queue; an edit then discarded them without asking again."""
    class _Exited(_PendingLookup):
        def isRunning(self):
            return False

    _PendingLookup.started = []
    monkeypatch.setattr(main_window_module, "ProbeWorker", _Exited)
    win = MainWindow()
    win._source_info = fake_video_info("source.mp4")
    paths = [Path(f"{name}.mp4") for name in "abc"]
    for path in paths:
        win._add_table_row(path)
    win._start_cache_lookup(paths)
    win._start_cache_lookup([paths[2]])
    assert _PendingLookup.started[-1].paths == paths
    # Once its finished signal has been handled, nothing is pending.
    win._on_cache_lookup_finished(win._cache_generation, win._cache_worker)
    win._start_cache_lookup([paths[0]])
    assert _PendingLookup.started[-1].paths == [paths[0]]
    win._probe_workers.clear()
    win.close()


def test_switching_back_to_a_display_with_a_saved_score_keeps_the_graph_series(qapp, tmp_path, monkeypatch):
    """Switching A -> B -> A: the saved CVVDP score for A comes back from
    the cache, and that replaced the graph series under a new identity --
    new colour, visible again, back even after being removed."""
    monkeypatch.setattr(result_cache, "cache_dir", lambda: tmp_path)
    win = MainWindow()
    row, row_data, source, distorted = _row_with_cached_scores(win, tmp_path, "cvvdp")
    row_data.extra_metric_keys.add("cvvdp")
    assert win._try_load_cached_result(row)
    graph = win.graph_panel
    graph.add_run(row_data.completed_run.result, "test", identity=row_data.completed_run.graph_identity)
    (series_id, entry), = graph._entries.items()
    colour = entry.color
    graph.set_series_visible(series_id, False)

    _select(win, row)
    win.cvvdp_preset_combo.setCurrentIndex(win.cvvdp_preset_combo.findData(BUILTIN_PRESETS[2].name))
    win.cvvdp_preset_combo.setCurrentIndex(win.cvvdp_preset_combo.findData(DEFAULT_PRESET.name))
    cached, _label = result_cache.load_cached(
        source, distorted, win._analysis_request(row_data), tmp_path,
        main_window_module.displayable_metric_specs(row_data.options, row_data.cvvdp))
    win._on_cached_found(distorted, cached, "test")  # the lookup's answer for the display switched back to
    assert win.distorted_table.item(row, COL_CVVDP).text() == "9.420"
    (series_after, entry_after), = graph._entries.items()
    assert series_after == series_id and entry_after.color == colour and not entry_after.visible

    graph.remove_run(series_id)
    win.cvvdp_preset_combo.setCurrentIndex(win.cvvdp_preset_combo.findData(BUILTIN_PRESETS[2].name))
    win._on_cached_found(distorted, cached, "test")
    assert graph._entries == {}
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


@pytest.mark.parametrize("answer", [QMessageBox.Yes, QMessageBox.No])
def test_saving_a_preset_asks_whether_other_videos_using_it_follow(qapp, monkeypatch, answer):
    """Only the selected videos were updated; the others kept the old
    values and turned into "Custom" without a word."""
    win, first_data, second_data = _two_rows_on_desk(monkeypatch)
    asked = []
    monkeypatch.setattr(QMessageBox, "question", lambda *args, **kwargs: asked.append(args[2]) or answer)
    _dialog_answers(monkeypatch, "save", name="Desk", peak=320)
    win._on_cvvdp_edit_display()
    assert len(asked) == 1 and '1 other video uses your preset "Desk"' in asked[0]
    assert first_data.cvvdp.display.peak_luminance == 320
    assert second_data.cvvdp.display.peak_luminance == (320 if answer == QMessageBox.Yes else 300)
    win.close()


def test_renaming_a_preset_keeps_other_videos_on_it_without_asking(qapp, monkeypatch):
    win, _first, second_data = _two_rows_on_desk(monkeypatch)
    monkeypatch.setattr(QMessageBox, "question", lambda *args, **kwargs: pytest.fail("nothing to ask"))
    _dialog_answers(monkeypatch, "save", name="Desk monitor")
    win._on_cvvdp_edit_display()
    assert second_data.cvvdp_preset == "Desk monitor"
    assert second_data.cvvdp.display.peak_luminance == 300
    win.close()


def test_of_two_presets_with_the_same_values_the_chosen_one_is_shown(qapp, monkeypatch):
    """The dropdown jumped to the later of the two names."""
    win, row, _row_data = _window_with_row()
    _select(win, row)
    for name in ("Desk", "Desk copy"):
        _dialog_answers(monkeypatch, "save_new", name=name, peak=300)
        win._on_cvvdp_add_preset()
    win.cvvdp_preset_combo.setCurrentIndex(win.cvvdp_preset_combo.findData("Desk"))
    assert win.cvvdp_preset_combo.currentData() == "Desk"
    _select(win, row)  # redrawn from the row
    assert win.cvvdp_preset_combo.currentData() == "Desk"
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

