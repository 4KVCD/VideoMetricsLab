// How a test-video row's analysis stands: the phrase shown on its file name
// (the Python app's vmaf_app/ui/row_state.py). Each state is a member here,
// rather than an English string set in one place and compared in others; its
// text is the English, marked for translation and shown with tr().
#pragma once

#include <QString>

namespace vml::app {

enum class RowState {
    // Set while something happens to the row.
    reading,
    queued,
    calculating,
    failed,
    // Some metrics finished and others failed.
    partly_failed,
    cancelled,
    cached,
    // Finished, and not shown: the run was for another source or other
    // settings than the row's now.
    for_previous_source,
    for_previous_settings,
    // Otherwise, what the row's metric cells add up to.
    no_metrics,
    complete,
    partial,
    not_calculated,
};

// The state's English text, as the Python app's RowState value.
const char* row_state_text(RowState state);

// The state's text in the window's language.
QString row_state_shown(RowState state);

// Whether the run finished but its result is not shown, because the row's
// source or settings have changed since.
bool finished_not_shown(RowState state);

}  // namespace vml::app
