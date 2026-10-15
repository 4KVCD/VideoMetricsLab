// The app's preferences, saved in settings.json between sessions. A port of
// vmaf_app/core/settings.py: the same file, read the same way, so both apps
// share it.
#pragma once

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace vml {

// Above this many logical cores, a new install scores two videos at once.
constexpr unsigned parallel_by_default_above_cores = 12;

// The parallel-jobs setting a PC with `cores` logical cores starts with.
int default_parallel_jobs(unsigned cores);

struct Settings {
    // The folder holding ffmpeg and ffprobe; empty means "search the PATH".
    std::string ffmpeg_dir;
    // Where finished runs are saved; empty means the per-user default.
    std::string cache_dir;
    // The folder CSV and PNG exports go to; empty means "ask each time".
    std::string export_dir;

    // What a newly added video starts with (each row's own settings are
    // edited in the window's Options panel).
    bool default_gpu_decode = true;
    bool default_compute_psnr = true;
    bool default_compute_ssim = true;
    bool default_compute_xpsnr = true;
    bool default_compute_vmaf = true;
    bool default_compute_vmaf_neg = true;
    bool default_compute_vmaf_v1 = true;
    bool default_compute_ssimulacra2 = true;
    bool default_compute_butteraugli = true;
    bool default_compute_cvvdp = true;
    // "gpu" (Vship) or "cpu" (libjxl's tools).
    std::string default_ssimulacra2_backend = "gpu";
    std::string default_butteraugli_backend = "gpu";
    // Whether VMAF v0.6.1, NEG and v1 run on the GPU.
    bool default_vmaf_on_gpu = true;
    std::string graph_metric = "vmaf";

    // How many videos to score at once (1 or 2).
    int parallel_jobs = 1;
    // Calculate a video's GPU metrics in one Vship pass always; otherwise in
    // one when the GPU has the memory for it, and one at a time when not.
    bool gpu_metrics_together = false;
    // Which Vship build calculates the GPU metrics: "auto", "vulkan", "cuda"
    // or "hip".
    std::string gpu_backend = "auto";
    // Use a saved result instead of calculating again.
    bool use_cache = true;

    bool check_for_updates = true;
    std::string skipped_update_version;
    std::string language;
    std::string theme;
    bool remember_window_size = true;
    int window_width = 1280;
    int window_height = 800;
    std::string last_reference_dir;
    std::string last_test_dir;
    bool remember_videos = false;
    std::string remembered_reference;
    std::vector<std::string> remembered_tests;
    bool highlight_best_worst = false;
    std::string frame_preview_color_mode = "display_aware";
    int compare_decoded_videos = 3;
    // The metrics left out of the test-video table (and not calculated).
    std::vector<std::string> hidden_metrics;
    // The CVVDP presets the user saved, as JSON text of their list
    // ([{"name": ..., "settings": ...}]), and the preset new videos start
    // with ("" is the built-in default).
    std::string cvvdp_presets_json = "[]";
    std::string cvvdp_default_preset;

    // The saved settings: defaults for a missing or unreadable file, and
    // for each value of the wrong type (that setting alone); unknown keys
    // are ignored. Reading makes the settings folder if it is missing, as
    // the Python app does.
    static Settings load();
    // load() from a given file, for tests.
    static Settings load_from(const std::filesystem::path& file);
    // Where settings.json is: the user's home/.videometricslab.
    static std::filesystem::path file();

    // Saves the settings as the Python app does (json.dumps(asdict(self),
    // indent=2)): every setting, in its order. The file is written whole
    // beside the old one, then put in its place, so a crash or a full disk
    // part-way through never leaves a cut-off file. Returns nothing on
    // success, or a message to show the user.
    std::optional<std::string> save() const;
    // save() to a given file, for tests.
    std::optional<std::string> save_to(const std::filesystem::path& file) const;
    // The text save() writes.
    std::string text() const;

    // The folders, or nothing when the setting is empty.
    std::optional<std::filesystem::path> ffmpeg_dir_path() const;
    std::optional<std::filesystem::path> cache_dir_path() const;
    // The libvmaf features these defaults ask for: PSNR and SSIM.
    std::vector<std::string> default_extra_features() const;
};

}  // namespace vml
