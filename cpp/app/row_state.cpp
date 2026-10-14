#include "row_state.h"

#include "i18n.h"

namespace vml::app {

const char* row_state_text(RowState state) {
    switch (state) {
        case RowState::reading:
            return N_("Reading...");
        case RowState::queued:
            return N_("Queued");
        case RowState::calculating:
            return N_("Calculating");
        case RowState::failed:
            return N_("Failed");
        case RowState::partly_failed:
            return N_("Partly failed");
        case RowState::cancelled:
            return N_("Cancelled");
        case RowState::cached:
            return N_("Complete (cached)");
        case RowState::for_previous_source:
            return N_("Finished for the previous source; select it again to load the result.");
        case RowState::for_previous_settings:
            return N_("Finished with the previous settings; change them back to see the result.");
        case RowState::no_metrics:
            return N_("No metrics selected");
        case RowState::complete:
            return N_("Complete");
        case RowState::partial:
            return N_("Partially calculated");
        case RowState::not_calculated:
            return N_("Not calculated");
    }
    return "";
}

QString row_state_shown(RowState state) { return tr(row_state_text(state)); }

bool finished_not_shown(RowState state) {
    return state == RowState::for_previous_source || state == RowState::for_previous_settings;
}

}  // namespace vml::app
