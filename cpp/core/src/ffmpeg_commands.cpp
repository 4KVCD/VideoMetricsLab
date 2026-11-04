// FFmpeg's commands and filter graphs for a run (ffmpeg_run.h): libvmaf's
// and xpsnr's options, each video's filters before the two are paired, and
// the commands of a run, of a resample test and of a run whose videos are
// decoded by FFmpegs of their own.
#include "ffmpeg_run_parts.h"

#include <sstream>
#include <tuple>

#include "vml/append.h"
#include "vml/ffmpeg_run.h"
#include "vml/ffmpeg_tools.h"
#include "vml/frame_sync.h"
#include "vml/geometry.h"
#include "vml/gpu_decoders.h"
#include "vml/model_select.h"
#include "vml/platform/system.h"
#include "vml/python_text.h"

namespace vml {

namespace {

using namespace run_parts;

// Separates the filter graphs of a run that has more than one; each becomes
// a -filter_complex of its own, ending in [graph0], [graph1].
constexpr char graph_separator = '\n';

// libvmaf's options: its log, the models, threads, subsampling, extra
// features (PSNR, SSIM) and frame sync. FFmpeg's filter-graph parser cannot
// take a Windows path with its drive letter's colon, so FFmpeg runs in the
// log's folder and every file is named by its bare name.
std::vector<std::string> libvmaf_options(const VmafOptions& options, const std::string& log_name,
                                         const std::string& model, unsigned cores) {
    std::string model_value = model;
    const std::optional<std::filesystem::path> v1_file = v1_model_file(options);
    if (options.compute_vmaf_neg || v1_file) {
        // Each model gets a name, so each has its own scores in the log
        // ("\\:" is a colon escaped for the filter graph).
        std::vector<std::string> models;
        if (options.compute_vmaf) {
            models.push_back(model_value + "\\\\:name=vmaf");
        }
        if (v1_file) {
            models.push_back(model_option_for_file(platform::path_text(v1_file->filename())) + "\\\\:name=vmaf_v1");
        }
        if (options.compute_vmaf_neg) {
            models.emplace_back("version=vmaf_v0.6.1neg\\\\:name=vmaf_neg");
        }
        model_value = join(models, "|");
    }
    std::vector<std::string> made{"log_path=" + log_name, "log_fmt=json",
                                  uses_vmaf_model(options) ? "model=" + model_value : "model=''"};
    // libvmaf runs on one thread unless told otherwise, so "Auto" is every
    // core (or this job's share of them, set before the run).
    made.push_back("n_threads=" + std::to_string(options.n_threads > 0 ? options.n_threads : static_cast<int>(cores)));
    if (options.n_subsample > 1) {
        made.push_back("n_subsample=" + std::to_string(options.n_subsample));
    }
    if (!options.extra_features.empty()) {
        made.push_back("feature=" + join(options.extra_features, "|"));
    }
    for (const std::string& option : framesync_opts()) {
        made.push_back(option);
    }
    return made;
}

// FFmpeg's xpsnr filter on the two videos, the source first: it weights
// each block's error by its first input's activity, which must be the
// original's.
std::string xpsnr_filter(const std::string& source_label, const std::string& test_label,
                         const std::string& xpsnr_log_name) {
    return "[" + source_label + "][" + test_label + "]xpsnr=stats_file=" + xpsnr_log_name + ":" +
           join(framesync_opts(), ":");
}

// The end of the filter graph: XPSNR and libvmaf.
std::string libvmaf_stage(const VmafOptions& options, const std::string& log_name, const std::string& model,
                          const std::optional<std::string>& xpsnr_log_name, unsigned cores) {
    if (options.requested_metrics().empty()) {
        throw VmafRunError("Select at least one metric to calculate.");
    }
    std::string main_label = "main";
    std::string ref_label = "ref";
    if (!uses_vmaf_model(options) && options.extra_features.empty()) {
        return xpsnr_filter(ref_label, main_label, *xpsnr_log_name);  // XPSNR alone needs no libvmaf
    }
    std::vector<std::string> chains;
    if (options.compute_xpsnr && xpsnr_log_name) {
        // xpsnr takes [main] and libvmaf needs it too, but a label can be
        // taken once: without this split, FFmpeg silently gives libvmaf the
        // wrong stream (one video against itself: VMAF 100 for every frame).
        chains.emplace_back("[" + main_label + "]split=2[main_xpsnr][main_vmaf]");
        chains.push_back(xpsnr_filter(ref_label, "main_xpsnr", *xpsnr_log_name) + "[xref]");
        main_label = "main_vmaf";
        ref_label = "xref";
    }
    chains.push_back("[" + main_label + "][" + ref_label + "]libvmaf=" +
                     join(libvmaf_options(options, log_name, model, cores), ":"));
    return join(chains, ";");
}

// The hardware decoders that hand frames over in system memory themselves
// when no output format is given (QSV's still hands over its surfaces).
bool decoder_downloads(const std::optional<std::string>& hwaccel) {
    const GpuDecoder* decoder = hwaccel ? decoder_of_hwaccel(*hwaccel) : nullptr;
    return decoder && decoder->ffmpeg_downloads;
}

// The -hwaccel options for one input of a run with several graphs: where the
// decoder can, it downloads each picture once and every graph gets that copy.
std::vector<std::string> split_hwaccel_args(const std::optional<std::string>& hwaccel) {
    if (decoder_downloads(hwaccel)) {
        return {"-hwaccel", *hwaccel};
    }
    return hwaccel_args(hwaccel);
}

// The options after the inputs: the graphs and the outputs (a range's frames
// are kept by each chain's head, video_chains).
std::vector<std::string> output_args(const std::string& filtergraph) {
    std::vector<std::string> graphs;
    std::stringstream lines(filtergraph);
    for (std::string graph; std::getline(lines, graph, graph_separator);) {
        graphs.push_back(graph);
    }
    std::vector<std::string> args;
    if (graphs.size() > 1) {
        for (const std::string& graph : graphs) {
            args.insert(args.end(), {"-filter_complex", graph});
        }
        args.insert(args.end(), {"-progress", "pipe:1", "-nostats"});
        for (std::size_t index = 0; index < graphs.size(); ++index) {
            args.push_back("-map");
            args.push_back("[graph" + std::to_string(index) + "]");
            args.insert(args.end(), {"-f", "null", "-"});
        }
        return args;
    }
    return {"-lavfi", filtergraph, "-progress", "pipe:1", "-nostats", "-f", "null", "-"};
}

// Each video's filters before the two are paired: a range's frames kept
// (first, so that the frames are counted as decoded, as a run over the whole
// videos numbers them), downloaded from its GPU
// decoder if need be (in a run of several graphs, a decoder that downloads
// its pictures itself does it once for all of them), cropped, converted to
// the analysis format (the deeper video's) and, for a test video of the
// other range, to the source's range (VideoInfo::full_range), scaled to the
// compared size, for a resolution test's test video scaled down and back up
// (VideoInfo::resample_via, as build_resample_filtergraph does), and its frames
// numbered.
struct VideoChains {
    std::vector<std::string> test;    // the test video's (input 0)
    std::vector<std::string> source;  // the source's
};

VideoChains video_chains(const VideoInfo& source, const VideoInfo& distorted, const VmafOptions& options,
                         const std::optional<CropBox>& source_crop, const std::optional<CropBox>& distorted_crop,
                         const HardwareDecoding& decoding, bool split) {
    const auto [distorted_w, distorted_h] = content_size(distorted, distorted_crop);
    const auto [source_w, source_h] = content_size(source, source_crop);
    const bool sizes_differ = source_w != distorted_w || source_h != distorted_h;
    const bool upscale_distorted = sizes_differ && options.scale_direction == ScaleDirection::distorted_to_source;
    const std::string analysis_format = analysis_pix_fmt({source.pix_fmt, distorted.pix_fmt});
    const std::vector<std::string> numbering = numbering_filters(frame_rate(distorted));
    const auto scaled_to = [&options](int width, int height) {
        return "scale=" + std::to_string(width) + ":" + std::to_string(height) + ":flags=" + options.scale_algorithm;
    };
    // A GPU decoder's 8-bit picture (NV12) widened to a deeper analysis
    // format is made planar first: FFmpeg widens NV12 with a plain shift, but
    // a planar picture, as its software decoding gives it, with full-range
    // luma's top bits repeated, as the app's own decoders widen (widen_to).
    const auto planar_first = [&analysis_format](const std::optional<std::string>& hwaccel, const VideoInfo& info) {
        return hwaccel.has_value() && bit_depth(info.pix_fmt) <= 8 && bit_depth(analysis_format) > 8;
    };
    VideoChains chains;
    const FrameRange range = run_range(options, source);
    if (const std::optional<std::string> trim = range_trim_filter(range, options.range_seeks.distorted)) {
        chains.test.push_back(*trim);
    }
    if (const std::optional<std::string> trim = range_trim_filter(range, options.range_seeks.source)) {
        chains.source.push_back(*trim);
    }

    if (decoding.distorted && !(split && decoder_downloads(decoding.distorted))) {
        // Frames come as GPU surfaces and must come back to system memory
        // before any other filter can use them.
        chains.test.emplace_back("hwdownload");
        chains.test.push_back("format=" + hw_native_format(distorted.pix_fmt));
    }
    if (const std::optional<std::string> scale = size_change_scale(distorted, options.scale_algorithm)) {
        chains.test.push_back(*scale);
    }
    if (distorted_crop && !distorted_crop->is_noop(distorted.frame_size().first, distorted.frame_size().second)) {
        chains.test.push_back(distorted_crop->as_filter());
    }
    if (planar_first(decoding.distorted, distorted)) {
        chains.test.emplace_back("format=yuv420p");
    }
    chains.test.push_back("format=" + analysis_format);
    if (distorted.full_range() != source.full_range()) {
        // As they are, a full-range video's samples would all be off from a
        // limited one's: black is 0 in one and 16 in the other.
        chains.test.push_back(std::string("scale=out_range=") + (source.full_range() ? "pc" : "tv"));
    }
    if (upscale_distorted) {
        chains.test.push_back(scaled_to(source_w, source_h));
    }
    if (distorted.resample_via) {
        chains.test.push_back(scaled_to(distorted.resample_via->first, distorted.resample_via->second));
        chains.test.push_back(scaled_to(distorted_w, distorted_h));
    }
    append(chains.test, numbering);

    // The source is cropped before the format conversion, as the test video
    // is, so chroma at a crop's edge is not filtered with the bars' samples.
    if (decoding.source && !(split && decoder_downloads(decoding.source))) {
        chains.source.emplace_back("hwdownload");
        chains.source.push_back("format=" + hw_native_format(source.pix_fmt));
    }
    if (const std::optional<std::string> scale = size_change_scale(source, options.scale_algorithm)) {
        chains.source.push_back(*scale);
    }
    if (source_crop && !source_crop->is_noop(source.frame_size().first, source.frame_size().second)) {
        chains.source.push_back(source_crop->as_filter());
    }
    if (planar_first(decoding.source, source)) {
        chains.source.emplace_back("format=yuv420p");
    }
    chains.source.push_back("format=" + analysis_format);
    if (sizes_differ && !upscale_distorted) {
        chains.source.push_back(scaled_to(distorted_w, distorted_h));
    }
    append(chains.source, numbering);
    return chains;
}

}  // namespace

namespace run_parts {

bool uses_vmaf_model(const VmafOptions& options) {
    return options.compute_vmaf || options.compute_vmaf_neg || options.compute_vmaf_v1;
}

std::optional<std::filesystem::path> v1_model_file(const VmafOptions& options) {
    if (const std::optional<std::string_view> file = model_file_of(options.model_v1); options.compute_vmaf_v1 && file) {
        return platform::path_from_text(*file);
    }
    return std::nullopt;
}

}  // namespace run_parts

std::string build_filtergraph(const VideoInfo& source, const VideoInfo& distorted, const VmafOptions& options,
                              const std::optional<CropBox>& source_crop, const std::optional<CropBox>& distorted_crop,
                              const HardwareDecoding& decoding, const std::string& log_name, const std::string& model,
                              const std::optional<std::string>& xpsnr_log_name, unsigned cores) {
    // XPSNR beside libvmaf's metrics: each in a graph of its own, which
    // FFmpeg runs on a thread of its own (44 fps at 4K, against 31 chained).
    const bool split = options.compute_xpsnr && xpsnr_log_name &&
                       (uses_vmaf_model(options) || !options.extra_features.empty());
    VideoChains chains = video_chains(source, distorted, options, source_crop, distorted_crop, decoding, split);
    // Both videos' frames with the same colour tags: where they differ,
    // FFmpeg 9 converts one video's colours to the other's before a filter
    // that takes both (for an untagged test video and a BT.2020 source,
    // PSNR 43.7 rather than 52.1), and libvmaf and xpsnr compare samples
    // as they are. The ranges are the same by now (video_chains).
    const std::string same_tags =
        std::string("setparams=colorspace=unknown:range=") + (source.full_range() ? "pc" : "tv");
    chains.test.push_back(same_tags);
    chains.source.push_back(same_tags);
    const std::string stream = std::string(video_stream);
    if (split) {
        // The graphs decode nothing themselves: FFmpeg gives every graph
        // that reads an input the same decoded frames.
        const auto both = [&](const std::string& suffix) {
            return "[0:" + stream + "]" + join(chains.test, ",") + "[main_" + suffix + "];[1:" + stream + "]" +
                   join(chains.source, ",") + "[ref_" + suffix + "]";
        };
        return both("v") + ";[main_v][ref_v]libvmaf=" + join(libvmaf_options(options, log_name, model, cores), ":") +
               "[graph0]" + graph_separator + both("x") + ";" + xpsnr_filter("ref_x", "main_x", *xpsnr_log_name) +
               "[graph1]";
    }
    return "[0:" + stream + "]" + join(chains.test, ",") + "[main];[1:" + stream + "]" + join(chains.source, ",") +
           "[ref];" + libvmaf_stage(options, log_name, model, xpsnr_log_name, cores);
}

std::vector<std::string> build_ffmpeg_command(const std::string& ffmpeg, const std::filesystem::path& distorted,
                                              const std::filesystem::path& source, const std::string& filtergraph,
                                              const HardwareDecoding& decoding, const InputOptions& inputs) {
    const bool several_graphs = filtergraph.find(graph_separator) != std::string::npos;
    const auto decode = [several_graphs](const std::optional<std::string>& hwaccel) {
        return several_graphs ? split_hwaccel_args(hwaccel) : hwaccel_args(hwaccel);
    };
    std::vector<std::string> command = ffmpeg_command_start(ffmpeg, {"-y"});
    for (const auto& [hwaccel, path, own, fed] :
         {std::tuple{decoding.distorted, distorted, &inputs.distorted, &inputs.distorted_fed},
          std::tuple{decoding.source, source, &inputs.source, &inputs.source_fed}}) {
        if (*fed) {
            command.insert(command.end(), (*fed)->begin(), (*fed)->end());
            continue;
        }
        const std::vector<std::string> options = decode(hwaccel);
        append(command, options);
        command.insert(command.end(), own->begin(), own->end());
        command.insert(command.end(), {"-i", input_path(path)});
    }
    const std::vector<std::string> outputs = output_args(filtergraph);
    append(command, outputs);
    return command;
}

std::string build_resample_filtergraph(const VideoInfo& source, const VmafOptions& options,
                                       const std::optional<CropBox>& source_crop,
                                       const std::optional<std::string>& hwaccel, const std::string& log_name,
                                       const std::string& model, const std::optional<std::string>& xpsnr_log_name,
                                       unsigned cores) {
    if (!options.resample_test) {
        throw std::invalid_argument("not a resolution test");
    }
    const bool cropped = source_crop && !source_crop->is_noop(source.width, source.height);
    const int original_w = cropped ? source_crop->w : source.width;
    const int original_h = cropped ? source_crop->h : source.height;
    const int down_w = options.resample_test->width;
    const int down_h = resample_down_height(down_w, {original_w, original_h});
    // One input, so the analysis format comes from the source alone.
    const std::string analysis_format = analysis_pix_fmt({source.pix_fmt});

    // A range's frames kept first, as in video_chains: both videos are the source's, cut alike.
    std::vector<std::string> base;
    const FrameRange range = run_range(options, source);
    if (const std::optional<std::string> trim = range_trim_filter(range, options.range_seeks.source)) {
        base.push_back(*trim);
    }
    if (hwaccel) {
        // As in build_filtergraph: hwdownload can only give the hardware
        // surface's own format, not the analysis format.
        base.emplace_back("hwdownload");
        base.push_back("format=" + hw_native_format(source.pix_fmt));
    }
    if (cropped) {
        base.push_back(source_crop->as_filter());  // before the conversion, as build_filtergraph's
    }
    base.push_back("format=" + analysis_format);
    const std::string numbering = join(numbering_filters(frame_rate(source)), ",");
    const std::string scale_flags = ":flags=" + options.resample_test->scaler;
    return join({"[0:" + std::string(video_stream) + "]" + join(base, ",") + "[base]",
                   "[base]split=2[ref_src][dist_src]", "[ref_src]" + numbering + "[ref]",
                   "[dist_src]scale=" + std::to_string(down_w) + ":" + std::to_string(down_h) + scale_flags +
                       ",scale=" + std::to_string(original_w) + ":" + std::to_string(original_h) + scale_flags + "," +
                       numbering + "[main]",
                   libvmaf_stage(options, log_name, model, xpsnr_log_name, cores)},
                  ";");
}

std::vector<std::string> build_resample_command(const std::string& ffmpeg, const std::filesystem::path& source,
                                                const std::string& filtergraph,
                                                const std::optional<std::string>& hwaccel,
                                                const std::vector<std::string>& source_options,
                                                const std::optional<std::vector<std::string>>& source_fed) {
    std::vector<std::string> command = ffmpeg_command_start(ffmpeg, {"-y"});
    if (source_fed) {
        command.insert(command.end(), source_fed->begin(), source_fed->end());
    } else {
        const std::vector<std::string> decode = hwaccel_args(hwaccel);
        append(command, decode);
        append(command, source_options);
        command.insert(command.end(), {"-i", input_path(source)});
    }
    const std::vector<std::string> outputs = output_args(filtergraph);
    append(command, outputs);
    return command;
}

std::filesystem::path synthetic_resample_distorted_path(const std::filesystem::path& source,
                                                        const ResampleTarget& target) {
    std::filesystem::path name = source.stem();
    // The default scaler unnamed, as every test's path was before tests had one.
    const std::string scaler = target.scaler == default_resample_scaler ? "" : "-" + target.scaler;
    name += " [downscale-" + target.label + scaler + "-upscale]";
    name += source.extension();
    return source.parent_path() / name;
}

std::pair<std::string, std::string> build_stream_graphs(const VideoInfo& source, const VideoInfo& distorted,
                                                        const VmafOptions& options,
                                                        const std::optional<CropBox>& source_crop,
                                                        const std::optional<CropBox>& distorted_crop,
                                                        const HardwareDecoding& decoding) {
    const VideoChains chains = video_chains(source, distorted, options, source_crop, distorted_crop, decoding, false);
    const std::string input = "[0:" + std::string(video_stream) + "]";  // each FFmpeg has its video alone
    return {input + join(chains.test, ",") + "[vmaf_dist]", input + join(chains.source, ",") + "[vmaf_ref]"};
}

std::vector<std::vector<std::string>> build_stream_commands(
    const std::string& ffmpeg, const std::filesystem::path& distorted, const std::filesystem::path& source,
    const std::pair<std::string, std::string>& graphs, const HardwareDecoding& decoding,
    const std::pair<std::vector<std::string>, std::vector<std::string>>& outputs, const InputOptions& inputs) {
    const auto command = [&ffmpeg](const std::filesystem::path& path, const std::optional<std::string>& hwaccel,
                                   const std::vector<std::string>& input,
                                   const std::optional<std::vector<std::string>>& fed, const std::string& graph,
                                   const std::vector<std::string>& output) {
        std::vector<std::string> made = ffmpeg_command_start(ffmpeg, {"-y"});
        if (fed) {
            made.insert(made.end(), fed->begin(), fed->end());
        } else {
            const std::vector<std::string> decode = hwaccel_args(hwaccel);
            append(made, decode);
            append(made, input);
            made.insert(made.end(), {"-i", input_path(path)});
        }
        made.insert(made.end(), {"-lavfi", graph, "-progress", "pipe:1", "-nostats"});
        append(made, output);
        return made;
    };
    return {command(distorted, decoding.distorted, inputs.distorted, inputs.distorted_fed, graphs.first,
                    outputs.first),
            command(source, decoding.source, inputs.source, inputs.source_fed, graphs.second, outputs.second)};
}

}  // namespace vml
