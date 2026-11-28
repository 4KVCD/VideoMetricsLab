#include "vml/run_record.h"

#include <algorithm>
#include <stdexcept>

#include "vml/analysis_request.h"
#include "vml/compared_range.h"
#include "vml/metrics.h"
#include "vml/model_select.h"
#include "vml/number_format.h"
#include "vml/platform/system.h"
#include "vml/result_json.h"
#include "vml/results_summary.h"

namespace vml {

namespace {

using Json = nlohmann::ordered_json;

// A size as the record writes it: "1920x804".
std::string size_text(int width, int height) { return std::to_string(width) + "x" + std::to_string(height); }

std::string size_text(const std::pair<int, int>& size) { return size_text(size.first, size.second); }

Json gpu_json(const VulkanGpu& gpu) {
    return Json{{"name", gpu.name},
                {"vendor", gpu.vendor},
                {"driver_version", gpu.driver_version},
                {"driver_version_number", gpu.driver_version_number},
                {"driver_name", gpu.driver_name},
                {"driver_info", gpu.driver_info}};
}

// How each metric's score and statistics come from its frames (or, for
// CVVDP, its seconds): what results_summary's statistics mean.
Json summary_rules() {
    Json per_metric = Json::object();
    for (const MetricDefinition& metric : metrics()) {
        Json bands = Json::array();
        for (const Threshold& band : metric.thresholds) {
            bands.push_back(std::string(1, band.comparison) + " " + format_general(band.value));
        }
        const char* const score =
            metric.kind == MetricKind::sequence ? "the whole video's score, as the metric gives it"
            : metric.aggregation == MetricAggregation::square_mean_root_db
                ? "-20 log10(mean(10^(-dB/20))) of the frames (FFmpeg's XPSNR average)"
            : metric.aggregation == MetricAggregation::mean_error_db
                ? "-10 log10(mean(10^(-dB/10))) of the frames (the mean of their error, as Daala's tool averages it)"
                : "the mean of the frames";
        per_metric[std::string(metric.key)] =
            Json{{"score", score},
                 {"worst_is", metric.direction == MetricDirection::lower_is_better ? "high" : "low"},
                 {"bands", std::move(bands)}};
    }
    return Json{{"median", "the middle frame's score, or the mean of the middle two"},
                {"stdev", "the population standard deviation of the finite scores"},
                {"worst_percent",
                 "worst_10_percent ... worst_0.1_percent: the score below which that share of the frames falls "
                 "(above which, where worst_is is high), numpy's linear percentile"},
                {"percent_of_frames", "the share of frames above (>) or below (<) each band's value, in %"},
                {"worst_frames",
                 "the worst frame of each of the 10 worst dips, worst first; a frame within 1 s (or the scores' "
                 "step, if longer) of a worse one is part of its dip"},
                {"frames_with_no_score", "left out (NaN); an identical frame scores inf and is counted"},
                {"metrics", std::move(per_metric)}};
}

std::string decoder_text(const std::optional<std::string>& decoder) { return decoder.value_or("software"); }

// Each video's decoder, and for one decoded in software why ("source_why",
// "test_why"; absent for one on the GPU).
Json decode_json(const std::optional<HardwareDecoding>& decode, const SoftwareDecodingWhy& why) {
    if (!decode) {
        return nullptr;
    }
    Json json{{"source", decoder_text(decode->source)}, {"test", decoder_text(decode->distorted)}};
    if (!why.source.empty()) {
        json["source_why"] = why.source;
    }
    if (!why.distorted.empty()) {
        json["test_why"] = why.distorted;
    }
    return json;
}

Json box_json(const std::optional<CropBox>& box) { return summary_crop_json(box); }

Json edges_json(const BarEdges& edges) {
    return Json{{"top", edges.top}, {"bottom", edges.bottom}, {"left", edges.left}, {"right", edges.right}};
}

Json crop_edges_json(const CropEdges& edges) {
    return Json{{"top", edges.top}, {"bottom", edges.bottom}, {"left", edges.left}, {"right", edges.right}};
}

// One video's black bars: the box compared, and where detection ran, what
// it found and what from; each detail null where it is not known (no box
// kept: a video whose size changes keeps none, detection failed, or saved
// scores answered it and the store holds no box of this version's; a box
// kept by a build before these records has no findings or time).
Json side_json(const std::optional<CropAnswer>& kept, const std::optional<CropBox>& compared,
               const std::string& started_utc, bool detects) {
    Json side{{"compared", box_json(compared)}};
    if (!detects) {
        return side;
    }
    const bool timed = kept && !kept->found_utc.empty();
    const BarFindings* found = kept && kept->findings ? &*kept->findings : nullptr;
    const bool measured = found != nullptr && !found->too_few;
    side["detected"] = kept ? box_json(kept->box) : Json(nullptr);
    side["adjusted"] = kept ? Json(compared != std::optional<CropBox>(kept->box)) : Json(nullptr);
    side["found_utc"] = timed ? Json(kept->found_utc) : Json(nullptr);
    // ISO 8601 texts of one form compare as the moments they name
    side["found_before_run"] =
        timed && !started_utc.empty() ? Json(kept->found_utc < started_utc) : Json(nullptr);
    side["detection_seconds"] = timed ? Json(kept->seconds) : Json(nullptr);
    side["previews_decoded"] = found ? Json(found->decoded) : Json(nullptr);
    side["previews_judged"] = found ? Json(found->judged) : Json(nullptr);
    side["too_few_judged"] = found ? Json(found->too_few) : Json(nullptr);
    side["median_bars"] = measured ? edges_json(found->median) : Json(nullptr);
    side["shape_changes"] = measured ? Json(found->shape_changes) : Json(nullptr);
    side["bars_cut_before_rounding"] = measured ? edges_json(found->taken) : Json(nullptr);
    return side;
}

Json black_bars_json(const VideoSettings& settings, const ComparisonResult* result, const std::string& started_utc) {
    const VmafOptions& options = settings.options;
    Json bars{{"mode", crop_mode_text(options.crop_mode)}};
    if (options.crop_mode == CropMode::manual) {
        bars["manual_source"] = crop_edges_json(options.manual_source_crop);
        bars["manual_test"] = crop_edges_json(options.manual_test_crop);
    }
    const std::optional<CropBox> source_compared = result ? result->source_crop : std::nullopt;
    const std::optional<CropBox> test_compared = result ? result->distorted_crop : std::nullopt;
    // Source only detects the source's alone
    const bool detects_source = detects_black_bars(options.crop_mode);
    const bool detects_test = options.crop_mode == CropMode::automatic;
    bars["source"] = side_json(detects_source ? kept_crop_answer(settings.source) : std::nullopt, source_compared,
                               started_utc, detects_source);
    bars["test"] = side_json(detects_test ? kept_crop_answer(settings.test) : std::nullopt, test_compared,
                             started_utc, detects_test);
    return bars;
}

// What each side of the comparison was scaled from and to: null for a side
// left at its size.
Json scaling_json(const VideoSettings& settings, const ComparisonResult* result) {
    const VmafOptions& options = settings.options;
    Json scaling{{"direction", scale_direction_value(options.scale_direction)},
                 {"algorithm", options.scale_algorithm}};
    if (options.resample_test) {
        // Both videos come from the source, which is scaled to the test's
        // width and back
        scaling["resolution_test"] = Json{{"label", options.resample_test->label},
                                          {"width", options.resample_test->width},
                                          {"scaler", options.resample_test->scaler}};
        return scaling;
    }
    const auto picture = [](const std::optional<CropBox>& box, const VideoInfo& info) {
        return box ? std::pair(box->w, box->h) : info.frame_size();
    };
    const std::pair<int, int> source = picture(result ? result->source_crop : std::nullopt, settings.source);
    const std::pair<int, int> test = picture(result ? result->distorted_crop : std::nullopt, settings.test);
    const bool to_test = options.scale_direction == ScaleDirection::source_to_distorted;
    const std::pair<int, int> compared_at = to_test ? test : source;
    const auto side = [&compared_at](const std::pair<int, int>& size) {
        return size == compared_at ? Json(nullptr) : Json{{"from", size_text(size)}, {"to", size_text(compared_at)}};
    };
    scaling["compared_at"] = size_text(compared_at);
    scaling["source"] = side(source);
    scaling["test"] = side(test);
    return scaling;
}

// A model: as chosen, as run (a model= value: a file's install path, or a
// version libvmaf has built in), and its "id", the same on every install: a
// bundled model's or libvmaf's own version ("vmaf_v0.6.1",
// "hdr_vmaf_v1.1.8_3d0h"), or null for a model file of the user's.
Json model_json(const std::string& choice, const std::string& resolved) {
    const std::string shown = model_shown(resolved);
    const std::string_view version = "version=";
    return Json{{"choice", choice},
                {"model", resolved},
                {"id", shown.starts_with(version) ? Json(shown.substr(version.size())) : Json(nullptr)}};
}

Json settings_json(const VideoSettings& settings, const ComparisonResult* result, const std::string& cvvdp_display) {
    const VmafOptions& options = settings.options;
    Json json{{"metrics", settings.metrics}};
    // Auto's model is resolved once the size compared at is known: the
    // result's, where there is one
    const auto asks = [&settings](const char* key) {
        return std::find(settings.metrics.begin(), settings.metrics.end(), key) != settings.metrics.end();
    };
    const bool vmaf_v0 = asks("vmaf") || asks("vmaf_neg");  // NEG runs VMAF v0.6.1's model too
    json["vmaf_model"] =
        vmaf_v0 ? model_json(options.model_choice, result && !result->model.empty() ? result->model : options.model)
                : Json(nullptr);
    json["vmaf_v1_model"] =
        asks("vmaf_v1") ? model_json(options.model_choice_v1,
                                     result && !result->model_v1.empty() ? result->model_v1 : options.model_v1)
                        : Json(nullptr);
    if (asks("vmaf_hdr")) {  // only with HDR VMAF, so other runs' records stay as they were
        json["vmaf_hdr_model"] = model_json(options.model_choice_hdr, result && !result->model_hdr.empty()
                                                                          ? result->model_hdr
                                                                          : options.model_hdr);
    }
    json["custom_model_path"] = options.custom_model_path.empty() ? Json(nullptr) : Json(options.custom_model_path);
    const bool asks_cvvdp = asks("cvvdp");
    json["cvvdp"] = asks_cvvdp ? Json{{"display", cvvdp_display}, {"parameters", settings.cvvdp.parameters()}}
                               : Json(nullptr);
    // Each metric's frames: every frame, or every n-th (libvmaf's metrics
    // share its subsampling; XPSNR and CVVDP score every frame)
    Json every = Json::object();
    try {
        const AnalysisRequest request = analysis_request(options, settings.metrics,
                                                         {settings.perceptual_backends.begin(),
                                                          settings.perceptual_backends.end()},
                                                         settings.cvvdp, settings.source, settings.test);
        for (const MetricRequestSpec& spec : request.metrics) {
            every[spec.key] = spec.coverage ? spec.coverage->step : 1;
        }
    } catch (const std::invalid_argument&) {
        every = nullptr;  // a metric with no way to calculate it: the run said why
    }
    json["frame_subsample"] = Json{{"asked", options.n_subsample}, {"every_nth_frame", std::move(every)}};
    const ComparedRange range = options.compared_range();
    json["range"] = Json{{"whole", range.whole()},
                         {"start", range.start},
                         {"end", range.end},
                         {"unit", range.in_frames ? "frames" : range.in_milliseconds ? "milliseconds" : "seconds"}};
    if (asks_cvvdp && !range.in_seconds() && !range.whole()) {
        // CVVDP scores whole seconds: those nearest the frames, as the run
        // scored them (an end of 0: to the videos' end)
        const ComparedRange seconds =
            cvvdp_seconds(range, settings.source.fps,
                          std::min(settings.source.estimated_frame_count(), settings.test.estimated_frame_count()));
        json["range"]["cvvdp_seconds"] = Json{{"start", seconds.start}, {"end", seconds.end}};
    }
    if (result && result->compared_frame_count > 0) {
        const long long first = compared_frames(range, settings.source.fps).first;
        json["compared_frames"] = Json{
            {"first", first}, {"last", first + result->compared_frame_count - 1}, {"count", result->compared_frame_count}};
    } else {
        json["compared_frames"] = nullptr;
    }
    // Frame N of one video with frame N of the other, each counted from its
    // own first frame: nothing dropped or repeated to line them up
    json["frame_pairing"] = "frame number";
    json["scaling"] = scaling_json(settings, result);
    json["decoding"] = Json{{"gpu_decode", options.gpu_decode},
                            {"gpu_maker", options.gpu_vendor == GpuVendor::automatic
                                              ? std::string("automatic")
                                              : std::string(vendor_label(options.gpu_vendor))},
                            {"vvc", options.vvc_on_gpu ? "gpu" : "cpu"}};
    Json perceptual = Json::object();
    for (const auto& [key, backend] : settings.perceptual_backends) {
        perceptual[key] = backend;
    }
    json["calculated_on"] = Json{{"vmaf", options.vmaf_on_gpu ? "gpu" : "cpu"},
                                 {"cambi", options.cambi_on_gpu ? "gpu" : "cpu"},
                                 {"ciede2000", options.ciede2000_on_gpu ? "gpu" : "cpu"},
                                 {"ms_ssim", options.ms_ssim_on_gpu ? "gpu" : "cpu"},
                                 {"perceptual", std::move(perceptual)},
                                 {"vship", vship_build_value(options.vship_build)},
                                 {"vship_in_use", vship_build_value(vship_build_in_use(options.vship_build))},
                                 {"libvmaf_threads", options.n_threads}};
    return json;
}

Json pass_json(const PassTiming& pass) {
    return Json{{"task", pass.task},
                {"metrics", pass.metric_keys},
                {"together", pass.metric_keys.size() > 1},
                {"on", pass.on},
                {"seconds", pass.seconds},
                {"paused_seconds", pass.paused_seconds},
                {"setup_seconds", pass.setup_seconds ? Json(*pass.setup_seconds) : Json(nullptr)},
                {"frames", pass.frames},
                {"fps", pass.fps > 0 ? Json(pass.fps) : Json(nullptr)},
                {"end", pass.end}};
}

}  // namespace

std::string outcome_code(const MetricOutcome& outcome) {
    if (outcome.why.empty()) {
        return "";
    }
    if (!outcome.code.empty()) {
        return outcome.code;
    }
    return outcome.status == metric_not_run          ? not_planned_code
           : outcome.status == metric_not_applicable ? not_applicable_code
                                                     : calculation_failed_code;
}

MachineFacts this_machine(const std::string& ffmpeg_path, const std::string& ffmpeg_version) {
    MachineFacts machine;
    machine.os = platform::system_description();
    machine.cpu = platform::cpu_model_name();
    machine.ffmpeg_path = ffmpeg_path;
    machine.ffmpeg_version = ffmpeg_version;
    for (const VshipBuild build : {VshipBuild::fast, VshipBuild::stock}) {
        const auto known = known_vship_device(build);
        if (known && known->first) {
            machine.gpus = known->first->gpus;
            machine.vship_gpu = known->first->name;
            break;
        }
    }
    return machine;
}

void add_run_record(Json& document, const RunRecord& record) {
    const auto seconds = [](const std::optional<double>& value) { return value ? Json(*value) : Json(nullptr); };
    document["run"] = Json{{"interface", record.user_interface},
                           {"command_line", record.command_line ? Json(*record.command_line) : Json(nullptr)},
                           {"started_utc", record.started_utc.empty() ? Json(nullptr) : Json(record.started_utc)},
                           {"finished_utc", record.finished_utc.empty() ? Json(nullptr) : Json(record.finished_utc)},
                           {"wall_seconds", seconds(record.wall_seconds)},
                           {"paused_seconds", seconds(record.paused_seconds)},
                           {"runs", record.runs}};
    const MachineFacts& machine = record.machine;
    Json gpus = Json::array();
    for (const VulkanGpu& gpu : machine.gpus) {
        gpus.push_back(gpu_json(gpu));
    }
    document["machine"] = Json{{"os", machine.os},
                               {"cpu", machine.cpu},
                               {"gpus", std::move(gpus)},
                               {"vship_gpu", machine.vship_gpu.empty() ? Json(nullptr) : Json(machine.vship_gpu)},
                               {"ffmpeg", Json{{"version", machine.ffmpeg_version}, {"path", machine.ffmpeg_path}}}};
    document["summary_rules"] = summary_rules();
}

void add_video_record(Json& entry, const RunFacts* facts, const ComparisonResult* result,
                      const std::string& cvvdp_display) {
    if (result != nullptr) {
        // From the scores themselves, so known without the run's settings
        // too: whether they are of the whole videos (rate_of_scores pairs
        // only those with the file's bitrate), and a resolution test, whose
        // test video is the reference's own.
        entry["whole_file"] = !scores_of_a_range(*result);
        entry["resolution_test"] = result->resample_target ? Json{{"label", result->resample_target->label},
                                                                   {"width", result->resample_target->width},
                                                                   {"scaler", result->resample_target->scaler}}
                                                           : Json(nullptr);
    }
    if (facts == nullptr) {
        for (const char* key : {"settings", "black_bars", "metric_status", "decode_plan", "timings"}) {
            entry[key] = nullptr;
        }
        return;
    }
    entry["settings"] = settings_json(facts->settings, result, cvvdp_display);
    entry["black_bars"] = black_bars_json(facts->settings, result, facts->started_utc);
    Json status = Json::object();
    for (const auto& [key, outcome] : facts->metrics) {
        status[key] = outcome.why.empty()
                          ? Json{{"status", outcome.status}}
                          : Json{{"status", outcome.status}, {"why", outcome.why}, {"code", outcome_code(outcome)}};
    }
    entry["metric_status"] = std::move(status);
    Json plans = Json::array();
    Json passes = Json::array();
    for (const PassTiming& pass : facts->passes) {
        plans.push_back(
            Json{{"metrics", pass.metric_keys}, {"decoders", decode_json(pass.decode, pass.decode_why)}});
        passes.push_back(pass_json(pass));
    }
    // A video that never started (cancelled first, or answered by its saved
    // scores) decoded nothing and took no time
    const bool started = !facts->started_utc.empty();
    entry["decode_plan"] = started ? std::move(plans) : Json(nullptr);
    entry["timings"] = started ? Json{{"started_utc", facts->started_utc},
                                      {"seconds", facts->seconds},
                                      {"paused_seconds", facts->paused_seconds},
                                      {"passes", std::move(passes)}}
                               : Json(nullptr);
}

}  // namespace vml
