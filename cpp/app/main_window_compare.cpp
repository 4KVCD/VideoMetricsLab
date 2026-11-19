// The window's side of the Video Compare tab: which pairs it lists, and where
// black bars found are kept.
#include <QtGlobal>

#include <algorithm>
#include <atomic>

#include "frame_compare_panel.h"
#include "graph_panel.h"
#include "main_window.h"
#include "window_helpers.h"
#include "vml/crop_detect.h"
#include "vml/ffmpeg_run.h"
#include "vml/metric_cache.h"

namespace vml::app {

namespace {

// The frames the tab offers: the shorter video's (a resolution test's
// source's), as a run would bound them, but whatever range the metrics
// have. That range is the metrics' alone: playback stopped at it, three
// seconds into a four-minute video, and the frames outside it show that
// they were not scored.
long long whole_video_frames(const FrameComparison& comparison, VmafOptions options) {
    options.range_start = 0;
    options.range_end = 0;
    options.range_in_frames = false;
    options.range_in_milliseconds = false;
    const bool resample = comparison.resample_target.has_value();
    return estimate_total_frames(resample ? comparison.source_info : comparison.distorted_info, options,
                                 resample ? nullptr : &comparison.source_info);
}

}  // namespace

std::string new_frame_identity() {
    static std::atomic<long long> next{0};
    return "row:" + std::to_string(++next);
}

void MainWindow::sync_frame_compare() {
    // Every row of the table the tab can show, in the table's order: not
    // only scored rows, on purpose. Seeing a source and an encode side by
    // side is useful in itself: to check framing, to find a scene worth
    // measuring, or to see whether an encode is the same content at all,
    // before a feature-length calculation. Metric Graphs lists its lines in
    // the table's order too: called after every change to the rows, this
    // keeps both in step with it.
    std::vector<FrameComparisonEntry> entries;
    std::vector<std::string> graph_order;
    for (const std::shared_ptr<RowData>& row_data : rows_) {
        if (std::optional<FrameComparisonEntry> entry = frame_comparison_entry(*row_data)) {
            entries.push_back(std::move(*entry));
        }
        if (row_data->completed_run) {
            graph_order.push_back(row_data->completed_run->graph_identity);
        }
    }
    frame_compare_panel_->set_runs(std::move(entries));
    graph_panel_->set_order(std::move(graph_order));
    open_videos_changed();  // the rows removed, reordered or given a new reference
}

std::optional<FrameComparisonEntry> MainWindow::frame_comparison_entry(const RowData& row_data) const {
    // One entry for a row, scored or not.
    if (const std::shared_ptr<CompletedRun>& run = row_data.completed_run) {
        // A finished run knows the geometry it really used, with the black
        // bars it found.
        FrameComparison comparison = FrameComparison::from_result(run->result);
        comparison.source_info = source_info_ ? *source_info_ : run->result.source_info;
        comparison.distorted_info = row_data.video_info ? *row_data.video_info : run->result.distorted_info;
        comparison.frame_count = std::max(comparison.frame_count, whole_video_frames(comparison, row_data.options));
        return FrameComparisonEntry{run->graph_identity, run->label, std::move(comparison),
                                    std::shared_ptr<const ComparisonResult>(run, &run->result),
                                    row_data.frame_identity};
    }
    if (!source_info_) {
        return std::nullopt;  // nothing to compare against yet
    }
    const VmafOptions& options = row_data.options;
    const std::optional<ResampleTarget>& resample = options.resample_test;
    if (!resample && !row_data.video_info) {
        return std::nullopt;  // the test file has not been read yet
    }
    // The crops a run applies are known only once it has run: auto
    // detection measures the video. None and Manual are known now, so they
    // are exact; auto is previewed uncropped, and says so.
    FrameComparison comparison;
    comparison.source_info = *source_info_;
    comparison.distorted_info = resample ? *source_info_ : *row_data.video_info;
    comparison.scale_direction = options.scale_direction;
    comparison.scale_algorithm = options.scale_algorithm;
    comparison.resample_target = resample;
    comparison.fps = comparison.distorted_info.fps;
    comparison.frame_count = whole_video_frames(comparison, options);
    comparison.auto_crop_pending = detects_black_bars(options.crop_mode);
    comparison.test_full_frame = options.crop_mode == CropMode::source_only;
    if (options.crop_mode == CropMode::manual) {
        try {
            comparison.source_crop = box_from_edges(options.manual_source_crop, comparison.source_info, "the source");
            // A resolution test's test video is the source's own picture.
            comparison.distorted_crop =
                resample ? comparison.source_crop
                         : box_from_edges(options.manual_test_crop, comparison.distorted_info, "the test video");
        } catch (const CropEntryError&) {
            // Shown uncropped: the row's run says why the crop cannot be cut.
            comparison.source_crop.reset();
            comparison.distorted_crop.reset();
        }
    }
    return FrameComparisonEntry{row_data.frame_identity, text_of(row_data.path.stem()), std::move(comparison),
                                nullptr, row_data.frame_identity};
}

void MainWindow::use_crop_store() {
    // Black bars found are kept with the saved scores, in the folder
    // Settings names now. Detection asks for the store on threads of its
    // own, so the store holds the folder rather than reading the settings.
    const std::filesystem::path folder = cache_folder();
    set_crop_store([folder] {
        std::filesystem::create_directories(folder);
        return folder / black_bars_file;
    });
}

}  // namespace vml::app
