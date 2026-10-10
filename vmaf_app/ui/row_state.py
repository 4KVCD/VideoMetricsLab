"""How a test-video row's analysis stands: the phrase shown on its file
name.

Each state is a member here, rather than an English string set in one place
and compared in others, where rewording it would quietly stop it matching.
A member's value is the English text, marked for translation (N_) and shown
with tr().
"""
from __future__ import annotations

from enum import StrEnum

from vmaf_app.i18n import N_


class RowState(StrEnum):
    """The states a test-video row can be in."""

    # Set while something happens to the row (RowData.analysis_status).
    READING = N_("Reading...")
    QUEUED = N_("Queued")
    CALCULATING = N_("Calculating")
    FAILED = N_("Failed")
    #: Some metrics finished and others failed (MainWindow._on_job_partially_failed).
    PARTLY_FAILED = N_("Partly failed")
    CANCELLED = N_("Cancelled")
    CACHED = N_("Complete (cached)")
    #: Finished, and not shown: the run was for another source or other
    #: settings than the row's now (MainWindow._on_job_finished).
    FOR_PREVIOUS_SOURCE = N_("Finished for the previous source; select it again to load the result.")
    FOR_PREVIOUS_SETTINGS = N_("Finished with the previous settings; change them back to see the result.")
    # Otherwise, what the row's metric cells add up to (MainWindow._row_state).
    NO_METRICS = N_("No metrics selected")
    COMPLETE = N_("Complete")
    PARTIAL = N_("Partially calculated")
    NOT_CALCULATED = N_("Not calculated")

    @property
    def finished_not_shown(self) -> bool:
        """Whether the run finished but its result is not shown, because the
        row's source or settings have changed since."""
        return self in (RowState.FOR_PREVIOUS_SOURCE, RowState.FOR_PREVIOUS_SETTINGS)
