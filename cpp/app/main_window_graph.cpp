// The window's side of Metric Graphs (the Python app's _sync_graph,
// _open_or_update_graph and _on_show_graph_clicked): the rows' results on
// the graph.
#include "main_window.h"

#include <QTabWidget>

#include <atomic>
#include <string>

#include "frame_compare_panel.h"
#include "graph_panel.h"
#include "i18n.h"
#include "settings_panel.h"
#include "widgets.h"

namespace vml::app {

std::string new_graph_identity() {
    static std::atomic<long long> next{0};
    return "run:" + std::to_string(++next);
}

void MainWindow::on_tab_changed(int index) {
    // The tab just opened, brought up to date.
    if (index == tab_graph) {
        sync_graph();
    } else if (index == tab_frame_compare) {
        sync_frame_compare();
    } else if (index == tab_settings) {
        settings_panel_->refresh_status();
    }
    // Ctrl+Tab and Ctrl+Shift+Tab reach Video Compare from the tabs beside
    // it without the pointer crossing its tab, so those tabs prepare its
    // playback as the pointer over its tab does (released again if it is not
    // opened within FrameComparePanel's release_after_ms): its first pair
    // then comes about 125 ms after it opens rather than about 285 ms.
    if (index == tab_graph || index == tab_bitrate) {
        frame_compare_panel_->prepare_playback();
    }
}

void MainWindow::sync_graph() {
    // Every row's result on the graph. add_runs updates a line of the same
    // identity rather than adding another, leaves one already showing its
    // result alone, and brings the table up to date once, so this is safe at
    // every visit to the tab. A line the user removed stays removed.
    std::vector<GraphPanel::Run> runs;
    for (const std::shared_ptr<RowData>& row_data : rows_) {
        if (const std::shared_ptr<CompletedRun>& run = row_data->completed_run) {
            runs.push_back(
                {std::shared_ptr<const ComparisonResult>(run, &run->result), run->label, run->graph_identity, false});
        }
    }
    graph_panel_->add_runs(runs);
}

void MainWindow::open_or_update_graph(const std::vector<std::shared_ptr<CompletedRun>>& runs, bool show) {
    // The runs on the graph, removed lines too; with `show`, the tab brought
    // to the front.
    std::vector<GraphPanel::Run> lines;
    for (const std::shared_ptr<CompletedRun>& run : runs) {
        lines.push_back(
            {std::shared_ptr<const ComparisonResult>(run, &run->result), run->label, run->graph_identity, true});
    }
    graph_panel_->add_runs(lines);
    if (show) {
        tabs_->setCurrentIndex(tab_graph);
    }
}

void MainWindow::on_show_graph_clicked() {
    // Every row's result, not only the first ones finished: looked at again
    // during a run (4 of 8 done), it shows all 4.
    std::vector<std::shared_ptr<CompletedRun>> runs;
    for (const std::shared_ptr<RowData>& row_data : rows_) {
        if (row_data->completed_run) {
            runs.push_back(row_data->completed_run);
        }
    }
    if (runs.empty() && graph_panel_->empty()) {
        dialogs_.inform(app::tr("No results yet"),
                        app::tr("Calculate metrics or load analysis results to view metric graphs."));
        return;
    }
    open_or_update_graph(runs);
}

}  // namespace vml::app
