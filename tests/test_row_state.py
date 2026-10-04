"""vmaf_app.ui.row_state: a row's analysis state."""
from vmaf_app.i18n import tr
from vmaf_app.ui.row_state import RowState


def test_a_state_is_its_english_text_and_is_shown_translated():
    assert RowState.FAILED == "Failed" and str(RowState.QUEUED) == "Queued"
    assert tr(RowState.CALCULATING) == tr("Calculating")


def test_only_the_two_finished_but_not_shown_states_say_so():
    assert {state for state in RowState if state.finished_not_shown} == {
        RowState.FOR_PREVIOUS_SOURCE, RowState.FOR_PREVIOUS_SETTINGS}
