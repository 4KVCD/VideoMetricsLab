// SSIMULACRA2 and Butteraugli on the CPU (perceptual_cpu.h), held to the
// Python app's (perceptual_cpu.json, from perceptual_cpu.py and colour.py):
// the filter graphs and FFmpeg commands that make the pictures, the scores
// read from the tools' output, how many pairs are scored at once,
// Butteraugli's 3-norm, and the pictures' colour chunks. Where FFmpeg runs:
// the pictures are each video's frame N, a task scores every pair with
// stand-ins for the tools, and Cancel ends the tools at once. And each
// metric goes where the Python app sends it (run_perceptual_task).
#include "vml/perceptual_cpu.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <fstream>
#include <functional>
#include <iterator>
#include <thread>

#include "reference.h"
#include "test_videos.h"
#include "vml/colour.h"
#include "vml/ffmpeg_tools.h"
#include "vml/ffprobe.h"
#include "vml/film_grain_scoring.h"
#include "vml/frame_sync.h"
#include "vml/platform/process.h"
#include "vml/platform/system.h"
#include "vml/size_changes.h"
#include "vml/vship_run.h"

namespace vml {
namespace {

using Json = nlohmann::json;

const Json& reference() {
    static const Json loaded = test::load_reference("perceptual_cpu.json");
    return loaded;
}

VideoInfo video_from(const Json& info) {
    VideoInfo video;
    video.path = "video.mkv";
    video.width = info["width"].get<int>();
    video.height = info["height"].get<int>();
    video.fps = *test::reference_number(info["fps"]);
    video.duration = 10.0;
    video.nb_frames = 240;
    video.codec_name = "hevc";
    video.pix_fmt = info["pix_fmt"].get<std::string>();
    video.color_range = info["color_range"].get<std::string>();
    video.color_space = info["color_space"].get<std::string>();
    video.color_transfer = info["color_transfer"].get<std::string>();
    video.color_primaries = info["color_primaries"].get<std::string>();
    video.chroma_location = info["chroma_location"].get<std::string>();
    return video;
}

std::optional<CropBox> crop_from(const Json& crop) {
    if (crop.is_null()) {
        return std::nullopt;
    }
    return CropBox{crop[0].get<int>(), crop[1].get<int>(), crop[2].get<int>(), crop[3].get<int>()};
}

std::string bytes_from_hex(const std::string& hex) {
    std::string bytes;
    for (std::size_t at = 0; at + 1 < hex.size(); at += 2) {
        bytes.push_back(static_cast<char>(std::stoi(hex.substr(at, 2), nullptr, 16)));
    }
    return bytes;
}

std::string hex_of(const std::string& bytes) {
    static const char digits[] = "0123456789abcdef";
    std::string hex;
    for (const char byte : bytes) {
        hex.push_back(digits[static_cast<unsigned char>(byte) >> 4]);
        hex.push_back(digits[static_cast<unsigned char>(byte) & 0xF]);
    }
    return hex;
}

void write_file(const std::filesystem::path& path, const std::string& bytes) {
    std::ofstream(path, std::ios::binary) << bytes;
}

std::string read_file(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
}

// A video as the Python app's reference cases make one by default.
VideoInfo plain_video(const std::filesystem::path& path) {
    VideoInfo video;
    video.path = path;
    video.width = 1920;
    video.height = 1080;
    video.fps = 24.0;
    video.duration = 10.0;
    video.nb_frames = 240;
    video.codec_name = "hevc";
    video.pix_fmt = "yuv420p";
    return video;
}

// The filter graph and the command FFmpeg is given, by frame number and by
// timestamp.
TEST(PerceptualCpu, MakesThePicturesAsThePythonApp) {
    for (const Json& expected : reference()["graphs"]) {
        SCOPED_TRACE(expected["name"].get<std::string>());
        const test::ScopedVariable pairing(std::string(pairing_variable),
                                           expected["by_time"].get<bool>() ? "timestamp" : "");
        VmafOptions options;
        options.scale_direction = expected["to_source"].get<bool>() ? ScaleDirection::distorted_to_source
                                                                     : ScaleDirection::source_to_distorted;
        options.scale_algorithm = expected["algorithm"].get<std::string>();
        EXPECT_EQ(image_filtergraph(video_from(expected["source"]), video_from(expected["test"]), options,
                                    crop_from(expected["source_crop"]), crop_from(expected["test_crop"]),
                                    expected["step"].get<int>()),
                  expected["graph"].get<std::string>());
    }
    const std::filesystem::path folder = std::filesystem::temp_directory_path() / "pictures";
    const std::string separator(1, static_cast<char>(std::filesystem::path::preferred_separator));
    for (const Json& expected : reference()["commands"]) {
        SCOPED_TRACE(expected.dump());
        VmafOptions options;
        options.duration_limit = *test::reference_number(expected["limit"]);
        std::vector<std::string> command;
        for (std::string argument : expected["command"].get<std::vector<std::string>>()) {
            if (argument == "<source>" || argument == "<test>") {
                argument = input_path(argument == "<source>" ? "source.mkv" : "test.mkv");
            } else if (argument.starts_with("<folder>/")) {
                argument = platform::path_text(folder) + separator + argument.substr(9);
            }
            command.push_back(argument);
        }
        EXPECT_EQ(png_pairs_command("ffmpeg", plain_video("source.mkv"), plain_video("test.mkv"), options,
                                    std::nullopt, std::nullopt, expected["step"].get<int>(), folder),
                  command);
    }
}

// The score a tool printed, or the same refusal; and how many pairs are
// scored at once for a size, the metrics, the CPU's tasks, cores and memory.
TEST(PerceptualCpu, ReadsScoresAndSharesTheCpuAsThePythonApp) {
    for (const Json& expected : reference()["scores"]) {
        const std::string text = expected["text"].get<std::string>();
        SCOPED_TRACE(text);
        if (expected.contains("error")) {
            try {
                parse_score("ssimulacra2", text);
                ADD_FAILURE() << "no error";
            } catch (const PerceptualRunError& error) {
                EXPECT_EQ(std::string(error.what()), expected["error"].get<std::string>());
            }
            continue;
        }
        const double wanted = *test::reference_number(expected["score"]);
        const double score = parse_score("ssimulacra2", text);
        EXPECT_EQ(score, wanted);
        EXPECT_EQ(std::signbit(score), std::signbit(wanted));
    }
    for (const Json& expected : reference()["workers"]) {
        SCOPED_TRACE(expected.dump());
        EXPECT_EQ(scoring_workers(expected["pixels"].get<long long>(),
                                  expected["metrics"].get<std::vector<std::string>>(),
                                  expected["concurrent"].get<int>(), expected["cores"].get<unsigned>(),
                                  expected["free_memory"].get<unsigned long long>()),
                  expected["workers"].get<int>());
    }
}

// Butteraugli's 3-norm of a distortion map is numpy's, to the last digit on
// Windows; a colour map and a file that is not a map are refused as the
// Python app refuses them.
TEST(PerceptualCpu, TakesButteraugli3NormAsNumpy) {
    const test::Folder folder;
    for (const Json& expected : reference()["norms"]) {
        SCOPED_TRACE(expected["name"].get<std::string>());
        const std::filesystem::path map = folder / "map.pfm";
        write_file(map, bytes_from_hex(expected["bytes"].get<std::string>()));
        if (expected.contains("norm")) {
            EXPECT_TRUE(test::same_number(*test::reference_number(expected["norm"]), butteraugli_norm3(map)));
            continue;
        }
        const std::string error = expected["error"].get<std::string>();
        try {
            butteraugli_norm3(map);
            ADD_FAILURE() << "no error";
        } catch (const PerceptualRunError& refused) {
            EXPECT_EQ(std::string(refused.what()), error);
        } catch (const std::runtime_error&) {
            EXPECT_EQ(error, "unreadable");
        }
    }
}

// A PNG's chunks, whole, in order.
std::vector<std::string> png_chunks(const std::string& png) {
    std::vector<std::string> chunks;
    std::size_t position = 8;
    while (position + 12 <= png.size()) {
        std::size_t length = 0;
        for (std::size_t i = 0; i < 4; ++i) {
            length = (length << 8) | static_cast<unsigned char>(png[position + i]);
        }
        chunks.push_back(png.substr(position, 12 + length));
        position += 12 + length;
    }
    return chunks;
}

// The chunks that tell libjxl's tools how to read a picture's values, and
// FFmpeg's pictures with them in place of their own, as the Python app
// writes them. A picture as FFmpeg writes one, its colour chunks before its
// data, is described in place: the same picture but for a padding chunk
// ("vmPd") where FFmpeg's chunks took more room, and as long as before.
TEST(PerceptualCpu, DescribesThePicturesColoursAsThePythonApp) {
    for (const Json& expected : reference()["chunks"]) {
        const VideoColour colour{1, expected["transfer"].get<int>(), expected["primaries"].get<int>(), false};
        EXPECT_EQ(hex_of(png_colour_chunks(colour)), expected["chunks"].get<std::string>()) << expected.dump();
    }
    const test::Folder folder;
    const std::filesystem::path picture = folder / "picture.png";
    int in_place = 0;
    for (const Json& expected : reference()["pictures"]) {
        SCOPED_TRACE(expected["name"].get<std::string>());
        const VideoColour colour{1, expected["transfer"].get<int>(), expected["primaries"].get<int>(), false};
        const std::string written = bytes_from_hex(expected["bytes"].get<std::string>());
        write_file(picture, written);
        describe_png(picture, colour);
        EXPECT_EQ(hex_of(read_file(picture)), expected["described"].get<std::string>());

        // The same picture less any colour chunk after its data, which
        // FFmpeg does not write: the Python app's picture, padded.
        std::string as_ffmpeg = written.substr(0, std::min<std::size_t>(written.size(), 8));
        bool data_seen = false;
        for (const std::string& chunk : png_chunks(written)) {
            const std::string kind = chunk.substr(4, 4);
            data_seen = data_seen || kind == "IDAT";
            if (!(data_seen && (kind == "cICP" || kind == "gAMA" || kind == "cHRM"))) {
                as_ffmpeg += chunk;
            }
        }
        write_file(picture, as_ffmpeg);
        describe_png(picture, colour);
        const std::string described = read_file(picture);
        std::string unpadded = described.substr(0, std::min<std::size_t>(described.size(), 8));
        for (const std::string& chunk : png_chunks(described)) {
            if (chunk.substr(4, 4) == "vmPd") {
                ++in_place;
            } else {
                unpadded += chunk;
            }
        }
        EXPECT_EQ(hex_of(unpadded), expected["described"].get<std::string>());
    }
    EXPECT_EQ(in_place, 1);  // FFmpeg's chunks replaced; the others have none, or no picture
}

// ---------------------------------------------------------------- with FFmpeg

// ------------------------------------------------------------ where they run

// A task's output with `key`'s score, as one place or the other gives it.
PerceptualTaskOutput scored(const std::string& key, float value, const std::string& backend) {
    PerceptualTaskOutput output;
    const MetricProvenance provenance{key, "1", backend, key, {}};
    if (key == "cvvdp") {
        output.metrics.add(SequenceMetricResult{key, value, provenance, {}, {}, {}});
    } else {
        output.metrics.add(FrameMetricResult{key, {0}, {0.0}, {value}, provenance});
    }
    output.compared_frame_count = 1;
    return output;
}

// What the stand-in routes were asked to do, in order.
struct Routed {
    bool probed = false;
    std::optional<VshipBuild> build;  // the build whose GPU was asked for
    std::optional<CancelToken> probe_cancel;  // what the GPU was asked for under
    std::vector<std::string> gpu;
    std::vector<std::string> vship51;  // what stock Vship 5.1.2 was asked for
    std::vector<std::string> cpu;
    std::optional<CropPair> cpu_crops;
    std::vector<std::string> events;  // progress reports and the CPU's announcements
    std::vector<std::string> statuses;
    std::vector<StatusKind> kinds;  // the statuses' kinds
};

// Routes whose GPU is `device` (none: "no supported GPU"), and whose GPU and
// CPU work is `on_gpu` and `on_cpu`, and stock Vship 5.1.2's `on_vship51`
// (none: not bundled).
PerceptualRoutes stand_in_routes(
    Routed& routed, std::optional<VshipDevice> device,
    std::function<PerceptualTaskOutput(const std::vector<std::string>&, const PerceptualHooks&)> on_gpu,
    std::function<PerceptualTaskOutput(const std::vector<std::string>&, const ProgressCallback&)> on_cpu,
    std::function<PerceptualTaskOutput(const std::vector<std::string>&)> on_vship51 = {}) {
    PerceptualRoutes routes;
    routes.find_gpu = [&routed, device](VshipBuild build, const CancelToken& cancel) {
        routed.probed = true;
        routed.build = build;
        routed.probe_cancel = cancel;
        return std::pair(device, std::string(device ? "" : "no supported GPU"));
    };
    routes.find_crops = [](const PerceptualRequest&, const StatusCallback&, const CancelToken&) { return CropPair{}; };
    routes.run_gpu = [&routed, on_gpu](const PerceptualRequest&, const std::vector<std::string>& keys,
                                       const VshipDevice&, const CropPair&, bool, const PerceptualHooks& hooks,
                                       const CancelToken&) {
        routed.gpu = keys;
        return on_gpu(keys, hooks);
    };
    if (on_vship51) {
        routes.run_vship51 = [&routed, on_vship51](const PerceptualRequest&, const std::vector<std::string>& keys,
                                                   const VshipDevice&, const CropPair&, bool, const PerceptualHooks&,
                                                   const CancelToken&) {
            routed.vship51 = keys;
            return on_vship51(keys);
        };
    }
    routes.run_cpu = [&routed, on_cpu](const PerceptualRequest&, const std::vector<std::string>& keys,
                                       const std::optional<CropPair>& crops, const ProgressCallback& on_progress,
                                       const StatusCallback&, const CancelToken&) {
        routed.cpu = keys;
        routed.cpu_crops = crops;
        return on_cpu(keys, on_progress);
    };
    return routes;
}

PerceptualHooks recording_hooks(Routed& routed) {
    PerceptualHooks hooks;
    hooks.on_progress = [&routed](long long current, long long total, double fps) {
        routed.events.push_back(std::to_string(current) + "/" + std::to_string(total) + " at " +
                                std::to_string(static_cast<int>(fps)));
    };
    hooks.on_status = [&routed](const RunStatus& status) {
        routed.statuses.push_back(status.text);
        routed.kinds.push_back(status.kind);
    };
    hooks.on_cpu = [&routed](const std::vector<std::string>& keys) {
        std::string joined;
        for (const std::string& key : keys) {
            joined += (joined.empty() ? "" : ", ") + key;
        }
        routed.events.push_back("cpu: " + joined);
    };
    return hooks;
}

PerceptualRequest routed_request(double seconds = 60.0) {
    PerceptualRequest request;
    request.source = plain_video("source.mkv");
    request.distorted = plain_video("test.mkv");
    for (VideoInfo* video : {&request.source, &request.distorted}) {
        video->duration = seconds;
        video->nb_frames = static_cast<long long>(seconds * 24);
    }
    request.options.crop_mode = CropMode::none;
    return request;
}

const VshipDevice test_gpu{"cuda", "test GPU", 0, "5.1.2", GpuVendor::nvidia};

// Without a GPU Vship can use, SSIMULACRA2 goes to the CPU, and CVVDP,
// which runs on the GPU only, fails with the reason; alone, CVVDP fails the
// task. Said as a GPU fallback only where the GPU's check failed.
TEST(PerceptualRouting, WithoutAGpuCvvdpFailsAndTheOthersRunOnTheCpu) {
    Routed routed;
    PerceptualRoutes routes = stand_in_routes(
        routed, std::nullopt, [](const auto&, const auto&) -> PerceptualTaskOutput { throw std::logic_error("GPU"); },
        [](const auto&, const auto&) { return scored("ssimulacra2", 80.0f, "cpu"); });
    const PerceptualRequest request = routed_request();
    const PerceptualTaskOutput output =
        run_perceptual_task(request, {"ssimulacra2", "cvvdp"}, false, recording_hooks(routed), CancelToken(), routes);
    EXPECT_EQ(routed.cpu, (std::vector<std::string>{"ssimulacra2"}));
    EXPECT_TRUE(output.metrics.has("ssimulacra2"));
    ASSERT_NE(failure_of(output.failures, "cvvdp"), nullptr);
    EXPECT_EQ(*failure_of(output.failures, "cvvdp"),
              "CVVDP needs a GPU that Vship can use, and could not use one: no supported GPU");
    EXPECT_EQ(routed.statuses,
              (std::vector<std::string>{"Vship GPU unavailable (no supported GPU); using CPU reference metrics\xE2\x80\xA6"}));
    EXPECT_EQ(routed.kinds, (std::vector<StatusKind>{StatusKind::step}));
    EXPECT_EQ(routed.events, (std::vector<std::string>{"cpu: ssimulacra2"}));

    routes.gpu_check_failed = [](VshipBuild) { return true; };
    Routed failed_check;
    run_perceptual_task(request, {"ssimulacra2"}, false, recording_hooks(failed_check), CancelToken(), routes);
    EXPECT_EQ(failed_check.kinds, (std::vector<StatusKind>{StatusKind::gpu_perceptual_failed}));

    try {
        run_perceptual_task(request, {"cvvdp"}, false, recording_hooks(routed), CancelToken(), routes);
        ADD_FAILURE() << "CVVDP alone did not fail";
    } catch (const PerceptualRunError& error) {
        EXPECT_EQ(std::string(error.what()),
                  "CVVDP needs a GPU that Vship can use, and could not use one: no supported GPU");
    }
}

// The GPU is asked for under the run's own token, so that the run's Cancel
// ends a check of the GPU asked for again inside the run (a copy of a token
// shares its Cancel).
TEST(PerceptualRouting, TheGpuIsAskedForUnderTheRunsToken) {
    Routed routed;
    const PerceptualRoutes routes = stand_in_routes(
        routed, test_gpu, [](const auto&, const auto&) { return scored("ssimulacra2", 91.0f, "gpu"); },
        [](const auto&, const auto&) -> PerceptualTaskOutput { throw std::logic_error("CPU"); });
    const CancelToken run;
    run_perceptual_task(routed_request(), {"ssimulacra2"}, false, recording_hooks(routed), run, routes);
    ASSERT_TRUE(routed.probe_cancel.has_value());
    routed.probe_cancel->cancel();
    EXPECT_TRUE(run.cancelled());
}

// Each metric runs where it is set to, the CPU's with the black bars the GPU
// task found; each stage's progress is its own, the CPU's from 0 once it is
// told which metrics it takes; and a choice of the CPU alone does not probe
// the GPU.
TEST(PerceptualRouting, EachMetricRunsWhereItIsSet) {
    Routed routed;
    const PerceptualRoutes routes = stand_in_routes(
        routed, test_gpu,
        [](const auto&, const PerceptualHooks& hooks) {
            hooks.on_progress(1, 1, 10.0);
            return scored("ssimulacra2", 91.0f, "gpu");
        },
        [](const auto&, const ProgressCallback& on_progress) {
            on_progress(1, 1, 8.0);
            return scored("butteraugli", 0.2f, "cpu");
        });
    PerceptualRequest request = routed_request();
    request.backends = {{"ssimulacra2", "gpu"}, {"butteraugli", "cpu"}};
    const PerceptualTaskOutput output = run_perceptual_task(request, {"ssimulacra2", "butteraugli"}, false,
                                                            recording_hooks(routed), CancelToken(), routes);
    EXPECT_EQ(routed.gpu, (std::vector<std::string>{"ssimulacra2"}));
    EXPECT_EQ(routed.cpu, (std::vector<std::string>{"butteraugli"}));
    EXPECT_TRUE(routed.cpu_crops.has_value());
    EXPECT_EQ(output.metrics.keys(), (std::vector<std::string>{"ssimulacra2", "butteraugli"}));
    EXPECT_EQ(output.metrics.frame("ssimulacra2")->provenance.compute_backend, "gpu");
    EXPECT_EQ(output.metrics.frame("butteraugli")->provenance.compute_backend, "cpu");
    EXPECT_EQ(routed.events, (std::vector<std::string>{"1/1 at 10", "cpu: butteraugli", "1/1 at 8"}));

    Routed cpu_only;
    const PerceptualRoutes cpu_routes = stand_in_routes(
        cpu_only, test_gpu, [](const auto&, const auto&) -> PerceptualTaskOutput { throw std::logic_error("GPU"); },
        [](const auto&, const auto&) { return scored("butteraugli", 0.2f, "cpu"); });
    run_perceptual_task(request, {"butteraugli"}, false, recording_hooks(cpu_only), CancelToken(), cpu_routes);
    EXPECT_FALSE(cpu_only.probed);
    EXPECT_FALSE(cpu_only.cpu_crops.has_value());  // the CPU task finds the black bars itself
}

// A metric that failed on Vship-fast while CVVDP finished is calculated
// again with stock Vship 5.1.2 where it is bundled, and on the CPU where it
// is not or fails too, however long the video (two hours here).
TEST(PerceptualRouting, AMetricThatFailedOnTheGpuMovesToVship51ThenToTheCpuAtAnyLength) {
    const auto failed_ssimulacra2 = [](const auto&, const auto&) {
        PerceptualTaskOutput output = scored("cvvdp", 9.4f, "gpu");
        output.failures = {{"ssimulacra2", "Vship SSIMULACRA2 failed: out of memory"}};
        return output;
    };
    const auto on_cpu = [](const auto&, const auto&) { return scored("ssimulacra2", 80.0f, "cpu"); };
    const std::string on_vship51 = "SSIMULACRA2 failed on the GPU; calculating it with Vship 5.1.2\xE2\x80\xA6";
    const std::string to_cpu = "SSIMULACRA2 failed on the GPU; calculating it on the CPU\xE2\x80\xA6";
    for (const std::string vship51 : {"not bundled", "scores", "fails"}) {
        SCOPED_TRACE(vship51);
        Routed routed;
        std::function<PerceptualTaskOutput(const std::vector<std::string>&)> stock;
        if (vship51 == "scores") {
            stock = [](const auto&) { return scored("ssimulacra2", 81.0f, "gpu"); };
        } else if (vship51 == "fails") {
            stock = [](const auto&) -> PerceptualTaskOutput { throw VshipUnavailableError("device lost"); };
        }
        const PerceptualRoutes routes = stand_in_routes(routed, test_gpu, failed_ssimulacra2, on_cpu, stock);
        const PerceptualTaskOutput output = run_perceptual_task(
            routed_request(7200.0), {"ssimulacra2", "cvvdp"}, false, recording_hooks(routed), CancelToken(), routes);
        EXPECT_TRUE(output.metrics.has("cvvdp"));
        EXPECT_TRUE(output.failures.empty());
        const std::string backend = output.metrics.frame("ssimulacra2")->provenance.compute_backend;
        if (vship51 == "scores") {
            EXPECT_EQ(routed.vship51, (std::vector<std::string>{"ssimulacra2"}));
            EXPECT_TRUE(routed.cpu.empty());
            EXPECT_EQ(backend, "gpu");
            EXPECT_EQ(routed.statuses, (std::vector<std::string>{on_vship51}));
        } else {
            EXPECT_EQ(routed.cpu, (std::vector<std::string>{"ssimulacra2"}));
            EXPECT_EQ(backend, "cpu");
            EXPECT_EQ(routed.statuses.back(), to_cpu);
        }
    }
}

// When Vship-fast's task fails, stock Vship 5.1.2 calculates its metrics
// again where it is bundled; where it is not, fails too, or is what failed
// (on a GPU only 5.1.2 found, which is said, or with 5.1.2 chosen for the
// video, which is not), the CPU calculates what it can, however long the
// video, and CVVDP fails with the GPU's reason, which does not blame the GPU.
TEST(PerceptualRouting, AFailedGpuTaskMovesToVship51ThenToTheCpuAtAnyLength) {
    const auto frame_rates = [](const auto&, const auto&) -> PerceptualTaskOutput {
        throw PerceptualRunError("the frame rates differ");
    };
    const auto on_cpu = [](const auto&, const auto&) { return scored("ssimulacra2", 80.0f, "cpu"); };
    for (const std::string vship51 : {"not bundled", "scores", "fails", "found the GPU", "chosen"}) {
        SCOPED_TRACE(vship51);
        Routed routed;
        std::function<PerceptualTaskOutput(const std::vector<std::string>&)> stock;
        if (vship51 == "scores" || vship51 == "found the GPU" || vship51 == "chosen") {
            stock = [](const auto&) {
                PerceptualTaskOutput output = scored("ssimulacra2", 81.0f, "gpu");
                output.metrics.add_all(scored("cvvdp", 9.4f, "gpu").metrics);
                return output;
            };
        } else if (vship51 == "fails") {
            stock = [](const auto&) -> PerceptualTaskOutput { throw PerceptualRunError("the frame rates differ"); };
        }
        PerceptualRequest request = routed_request(7200.0);
        VshipDevice gpu = test_gpu;
        gpu.vship51 = vship51 == "found the GPU" || vship51 == "chosen";
        if (vship51 == "found the GPU") {
            gpu.fast_failure = "a shader failed";
        } else if (vship51 == "chosen") {
            request.options.vship_build = VshipBuild::stock;  // 5.1.2's own probe found the GPU
        }
        const PerceptualRoutes routes = stand_in_routes(routed, gpu, frame_rates, on_cpu, stock);
        const PerceptualTaskOutput output = run_perceptual_task(request, {"ssimulacra2", "cvvdp"}, false,
                                                                recording_hooks(routed), CancelToken(), routes);
        EXPECT_EQ(routed.build, request.options.vship_build);
        EXPECT_EQ(routed.vship51.empty(), vship51 == "not bundled" || gpu.vship51);
        if (vship51 == "found the GPU") {
            // Vship-fast's failed check is said first, as a GPU fallback.
            EXPECT_EQ(routed.statuses.front(), "Vship-fast could not use the GPU (a shader failed); calculating "
                                               "SSIMULACRA2 and CVVDP with Vship 5.1.2\xE2\x80\xA6");
            EXPECT_EQ(routed.kinds.front(), StatusKind::gpu_perceptual_failed);
        }
        if (!routed.vship51.empty()) {
            EXPECT_EQ(routed.vship51, (std::vector<std::string>{"ssimulacra2", "cvvdp"}));
            EXPECT_EQ(routed.statuses.front(), "Vship-fast failed (the frame rates differ); calculating SSIMULACRA2 "
                                               "and CVVDP with Vship 5.1.2\xE2\x80\xA6");
        }
        if (vship51 == "scores") {
            EXPECT_TRUE(routed.cpu.empty());
            EXPECT_TRUE(output.failures.empty());
            EXPECT_TRUE(output.metrics.has("cvvdp"));
            continue;
        }
        EXPECT_EQ(routed.cpu, (std::vector<std::string>{"ssimulacra2"}));
        EXPECT_TRUE(output.metrics.has("ssimulacra2"));
        ASSERT_NE(failure_of(output.failures, "cvvdp"), nullptr);
        EXPECT_EQ(*failure_of(output.failures, "cvvdp"), "CVVDP could not be calculated: the frame rates differ");
        EXPECT_EQ(routed.statuses.back(),
                  "Vship GPU compute failed (the frame rates differ); using CPU reference metrics\xE2\x80\xA6");
    }
}

// A film grain refusal is final: a GPU task refused is not made again with
// stock Vship 5.1.2 or on the CPU, a metric refused by 5.1.2 is not taken to
// the CPU, and what the GPU scored stays when the CPU refuses its metrics.
TEST(PerceptualRouting, AFilmGrainRefusalIsFinal) {
    const auto refused = [](const auto&...) -> PerceptualTaskOutput { throw FilmGrainRefusal("refused"); };
    const auto on_cpu = [](const auto&, const auto&) { return scored("ssimulacra2", 80.0f, "cpu"); };
    const auto stock = [](const auto&) { return scored("ssimulacra2", 81.0f, "gpu"); };
    Routed routed;
    EXPECT_THROW(run_perceptual_task(routed_request(), {"ssimulacra2", "cvvdp"}, false, recording_hooks(routed),
                                     CancelToken(), stand_in_routes(routed, test_gpu, refused, on_cpu, stock)),
                 FilmGrainRefusal);
    EXPECT_TRUE(routed.vship51.empty());
    EXPECT_TRUE(routed.cpu.empty());

    const auto failed_ssimulacra2 = [](const auto&, const auto&) {
        PerceptualTaskOutput output = scored("cvvdp", 9.4f, "gpu");
        output.failures = {{"ssimulacra2", "Vship SSIMULACRA2 failed: out of memory"}};
        return output;
    };
    Routed by_vship51;
    PerceptualTaskOutput output =
        run_perceptual_task(routed_request(), {"ssimulacra2", "cvvdp"}, false, recording_hooks(by_vship51),
                            CancelToken(), stand_in_routes(by_vship51, test_gpu, failed_ssimulacra2, on_cpu, refused));
    EXPECT_EQ(by_vship51.vship51, (std::vector<std::string>{"ssimulacra2"}));
    EXPECT_TRUE(by_vship51.cpu.empty());
    EXPECT_TRUE(output.metrics.has("cvvdp"));
    ASSERT_NE(failure_of(output.failures, "ssimulacra2"), nullptr);
    EXPECT_EQ(*failure_of(output.failures, "ssimulacra2"), "refused");

    Routed by_cpu;
    PerceptualRequest request = routed_request();
    request.backends = {{"ssimulacra2", "cpu"}};
    output = run_perceptual_task(
        request, {"ssimulacra2", "cvvdp"}, false, recording_hooks(by_cpu), CancelToken(),
        stand_in_routes(by_cpu, test_gpu, [](const auto&, const auto&) { return scored("cvvdp", 9.4f, "gpu"); },
                        refused));
    EXPECT_TRUE(output.metrics.has("cvvdp"));
    ASSERT_NE(failure_of(output.failures, "ssimulacra2"), nullptr);
    EXPECT_EQ(*failure_of(output.failures, "ssimulacra2"), "refused");
}

// A cancelled GPU task does not start the CPU's.
TEST(PerceptualRouting, CancellingDoesNotStartTheCpu) {
    Routed routed;
    const PerceptualRoutes routes = stand_in_routes(
        routed, test_gpu, [](const auto&, const auto&) -> PerceptualTaskOutput { throw Cancelled(); },
        [](const auto&, const auto&) { return scored("ssimulacra2", 80.0f, "cpu"); });
    EXPECT_THROW(run_perceptual_task(routed_request(), {"ssimulacra2"}, false, recording_hooks(routed), CancelToken(),
                                     routes),
                 Cancelled);
    EXPECT_TRUE(routed.cpu.empty());
}

// A Vship pass whose FFmpeg failed to give it the frames is made once more
// the same way, saying so; a second such failure, or another kind of
// failure, is the pass's own, for the next Vship or the CPU.
TEST(PerceptualRouting, APassWhoseInputFailedIsMadeOnceMore) {
    std::vector<std::string> statuses;
    const StatusCallback on_status = [&statuses](const RunStatus& status) { statuses.push_back(status.text); };
    int attempts = 0;
    // A pass that fails `failures` times with `error`, then gives its score.
    const auto pass = [&attempts](int failures, const std::exception_ptr& error) {
        return [&attempts, failures, error](bool, bool) {
            if (++attempts <= failures) {
                std::rethrow_exception(error);
            }
            return scored("cvvdp", 9.5f, "gpu");
        };
    };
    const auto input_error = std::make_exception_ptr(VshipInputError("FFmpeg ended partway through a frame."));
    const PerceptualTaskOutput output = remade_vship_pass(pass(1, input_error), on_status, CancelToken());
    EXPECT_EQ(attempts, 2);
    EXPECT_TRUE(output.metrics.has("cvvdp"));
    EXPECT_EQ(statuses, (std::vector<std::string>{
                            "Reading the videos failed (FFmpeg ended partway through a frame.); trying again\xE2\x80\xA6"}));
    attempts = 0;
    EXPECT_THROW(remade_vship_pass(pass(2, input_error), on_status, CancelToken()), VshipInputError);
    EXPECT_EQ(attempts, 2);
    attempts = 0;
    const auto gpu_error = std::make_exception_ptr(VshipUnavailableError("Vship CVVDP failed"));
    EXPECT_THROW(remade_vship_pass(pass(1, gpu_error), on_status, CancelToken()), VshipUnavailableError);
    EXPECT_EQ(attempts, 1);
    attempts = 0;
    const auto refusal = std::make_exception_ptr(FilmGrainRefusal("the app could not decode its pictures"));
    EXPECT_THROW(remade_vship_pass(pass(1, refusal), on_status, CancelToken()), FilmGrainRefusal);
    EXPECT_EQ(attempts, 1);
}

// A task that FFmpeg's rebuilt filters stop (a video's picture size changes
// partway through) is made again with the video the stop names scaled, the
// test video to the source's width at its own shape, the source to its own
// size. Where none is named, the test video is tried, then the source alone,
// then both, and a fourth stop fails the task, without another route scoring
// it. A video found to change in a task that finished is scaled from the
// start of its later tasks.
TEST(PerceptualRouting, AVideoWhoseSizeChangesIsScoredAtOneSize) {
    using Scaled = std::optional<std::pair<int, int>>;
    std::vector<std::pair<Scaled, Scaled>> asked;  // each GPU run's source and test video, as marked
    int stops = 0;
    std::optional<std::filesystem::path> named;  // the video each stop names
    Routed routed;
    PerceptualRoutes routes = stand_in_routes(
        routed, test_gpu, nullptr, [](const auto&, const auto&) { return scored("ssimulacra2", 80.0f, "cpu"); });
    routes.run_gpu = [&asked, &stops, &named](const PerceptualRequest& request, const std::vector<std::string>&,
                                              const VshipDevice&, const CropPair&, bool, const PerceptualHooks&,
                                              const CancelToken&) {
        asked.emplace_back(request.source.scaled_to, request.distorted.scaled_to);
        if (static_cast<int>(asked.size()) <= stops) {
            throw VideoSizeChanged(named);
        }
        return scored("ssimulacra2", 70.0f, "gpu");
    };
    // Each case with videos of its own: the videos found to change are this
    // process's.
    const auto task = [&](const std::string& name, int stopped, bool source_named = false) {
        PerceptualRequest request = routed_request();
        request.source.path = name + "-source.mkv";
        request.distorted.path = name + "-test.mkv";
        request.distorted.width = 1280;
        request.distorted.height = 534;
        asked.clear();
        stops = stopped;
        named = source_named ? std::optional(request.source.path) : std::nullopt;
        return run_perceptual_task(request, {"ssimulacra2"}, false, recording_hooks(routed), CancelToken(), routes);
    };
    const Scaled test_size = std::pair(1920, 802);  // 534 * 1920 / 1280 = 801, made even
    const Scaled source_size = std::pair(1920, 1080);
    EXPECT_TRUE(task("once", 1).metrics.has("ssimulacra2"));
    EXPECT_EQ(asked, (std::vector<std::pair<Scaled, Scaled>>{{std::nullopt, std::nullopt}, {std::nullopt, test_size}}));
    task("once", 0);
    EXPECT_EQ(asked, (std::vector<std::pair<Scaled, Scaled>>{{std::nullopt, test_size}}));
    task("twice", 2);
    EXPECT_EQ(asked, (std::vector<std::pair<Scaled, Scaled>>{
                         {std::nullopt, std::nullopt}, {std::nullopt, test_size}, {source_size, std::nullopt}}));
    task("twice", 0);  // the source, which changes, alone
    EXPECT_EQ(asked, (std::vector<std::pair<Scaled, Scaled>>{{source_size, std::nullopt}}));
    task("named", 1, true);
    EXPECT_EQ(asked,
              (std::vector<std::pair<Scaled, Scaled>>{{std::nullopt, std::nullopt}, {source_size, std::nullopt}}));
    try {
        task("always", 4);
        ADD_FAILURE() << "a task stopped with both videos scaled did not fail";
    } catch (const PerceptualRunError& error) {
        EXPECT_EQ(std::string(error.what()), size_change_unfixed);
    }
    EXPECT_EQ(asked, (std::vector<std::pair<Scaled, Scaled>>{{std::nullopt, std::nullopt},
                                                            {std::nullopt, test_size},
                                                            {source_size, std::nullopt},
                                                            {source_size, test_size}}));
    task("always", 0);  // nothing noted by a task that failed
    EXPECT_EQ(asked, (std::vector<std::pair<Scaled, Scaled>>{{std::nullopt, std::nullopt}}));
    EXPECT_TRUE(routed.cpu.empty());
}

// Two pictures of different shapes are refused, as FFmpeg's metrics refuse
// them (they were scored stretched), black bars kept or not; one rounded to
// an even size still compares.
TEST(PerceptualRouting, PicturesOfDifferentShapesAreRefused) {
    const VideoInfo source = plain_video("source.mkv");
    VideoInfo squeezed = plain_video("test.mkv");
    squeezed.width = 1440;
    EXPECT_THROW(resolve_crops("ffmpeg", source, squeezed, CropMode::none, "bicubic", CancelToken(), nullptr),
                 PerceptualRunError);
    VideoInfo rounded = plain_video("test.mkv");
    rounded.width = 1918;
    EXPECT_NO_THROW(resolve_crops("ffmpeg", source, rounded, CropMode::none, "bicubic", CancelToken(), nullptr));
}

// The bundled tool is the one built for this system: the repository's
// resources have Windows' ssimulacra2.exe, which macOS found first and could
// not run ("Permission denied").
TEST(PerceptualCpu, TheBundledToolIsTheOneForThisSystem) {
    const test::Folder resources;
    std::filesystem::create_directories(resources / "tools/libjxl");
    const test::ScopedVariable named("VML_RESOURCES", platform::path_text(resources / ""));
    const test::ScopedVariable chosen("SSIMULACRA2_PATH", "");
    const test::ScopedVariable path("PATH", platform::path_text(resources / "tools"));
    std::ofstream(resources / "tools/libjxl/ssimulacra2.exe") << "";
#ifdef _WIN32
    EXPECT_EQ(find_metric_executable("ssimulacra2"), resources / "tools/libjxl/ssimulacra2.exe");
#else
    EXPECT_EQ(find_metric_executable("ssimulacra2"), std::nullopt);
    std::ofstream(resources / "tools/libjxl/ssimulacra2") << "";
    EXPECT_EQ(find_metric_executable("ssimulacra2"), resources / "tools/libjxl/ssimulacra2");
#endif
}

}  // namespace
}  // namespace vml
