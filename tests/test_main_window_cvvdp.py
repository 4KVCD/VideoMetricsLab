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
    assert win.cvvdp_preset_combo.currentData() is None  # no preset has resize on: "Custom"

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

    assert row_data.cvvdp.display.peak_luminance == 350
    assert win._settings.cvvdp_default_preset == "My monitor"
    assert win.cvvdp_preset_combo.currentData() == "My monitor"
    assert win.settings_cvvdp_default.currentData() == "My monitor"
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
