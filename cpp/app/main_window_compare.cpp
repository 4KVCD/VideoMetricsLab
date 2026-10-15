// The window's side of the Video Compare tab: which pairs it lists, and where
// black bars found are kept.
#include <QtGlobal>

#include <atomic>

#include "frame_compare_panel.h"
#include "main_window.h"
#include "window_helpers.h"
#include "vml/crop_detect.h"
#include "vml/ffmpeg_run.h"
#include "vml/metric_cache.h"

namespace vml::app {

std::string new_frame_identity() {
    static std::atomic<long long> next{0};
    return "row:" + std::to_string(++next);
}

void MainWindow::sync_frame_compare() {
    // Every row of the table the tab can show, in the table's order: not
    // only scored rows, on purpose. Seeing a source and an encode side by
    // side is useful in itself: to check framing, to find a scene worth
    // measuring, or to see whether an encode is the same content at all,
    // before a feature-length calculation.
    std::vector<FrameComparisonEntry> entries;
    for (const std::shared_ptr<RowData>& row_data : rows_) {
        if (std::optional<FrameComparisonEntry> entry = frame_comparison_entry(*row_data)) {
            entries.push_back(std::move(*entry));
        }
    }
    frame_compare_panel_->set_runs(std::move(entries));
}

std::optional<FrameComparisonEntry> MainWindow::frame_comparison_entry(const RowData& row_data) const {
    // One entry for a row, scored or not.
    if (const std::shared_ptr<CompletedRun>& run = row_data.completed_run) {
        // A finished run knows the geometry it really used, with the black
        // bars it found.
        FrameComparison comparison = FrameComparison::from_result(run->result);
        comparison.source_info = source_info_ ? *source_info_ : run->result.source_info;
        comparison.distorted_info = row_data.video_info ? *row_data.video_info : run->result.distorted_info;
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
    // detection measures the video. None is known now, so it is exact; auto
    // is previewed uncropped, and says so.
    FrameComparison comparison;
    comparison.source_info = *source_info_;
    comparison.distorted_info = resample ? *source_info_ : *row_data.video_info;
    comparison.scale_direction = options.scale_direction;
    comparison.scale_algorithm = options.scale_algorithm;
    comparison.resample_target = resample;
    const VideoInfo& reference = comparison.distorted_info;
    comparison.fps = reference.fps;
    // The same bound a run uses, so that the timeline cannot offer frames
    // past the end of the shorter video.
    comparison.frame_count = estimate_total_frames(reference, options, resample ? nullptr : &*source_info_);
    comparison.auto_crop_pending = options.crop_mode == CropMode::automatic;
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
