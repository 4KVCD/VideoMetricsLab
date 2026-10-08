"""vmaf_app.ui.row_state: a row's analysis state."""
from vmaf_app.i18n import tr
from vmaf_app.ui.row_state import RowState


def test_a_state_is_its_english_text_and_is_shown_translated():
    assert RowState.FAILED == "Failed" and str(RowState.QUEUED) == "Queued"
    assert tr(RowState.CALCULATING) == tr("Calculating")
