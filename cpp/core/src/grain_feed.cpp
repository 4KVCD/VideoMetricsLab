#include "vml/grain_feed.h"

#include <array>
#include <condition_variable>
#include <cstdint>
#include <exception>
#include <mutex>
#include <set>
#include <thread>
#include <utility>

#include "vml/app_log.h"
#include "vml/film_grain_scoring.h"
#include "vml/frame_sync.h"
#include "vml/gpu_frames.h"
#include "vml/isolated.h"
#include "vml/number_format.h"
#include "vml/platform/pipe.h"
#include "vml/platform/system.h"
#include "vml/python_text.h"
#include "vml/run_json.h"

namespace vml {

namespace {

const app_log::Logger logger("vml.grain_feed");

// VML_FEED=all: every video the app's software decoder takes is fed.
bool feeding_all() { return platform::environment_variable("VML_FEED").value_or("") == "all"; }

// The pictures fed: the whole picture at its own size, every plane, 10-bit
// samples in the low bits of 16 (yuv420p10le's layout). Throws
// GpuDecodeUnavailableError where the software decoder takes no such plan.
DecodePlan feed_plan(const VideoInfo& info) { return plan_decode(info, std::nullopt, 6); }

// rawvideo's name for the plan's pictures: 4:2:0 alone, as the decoder
// gives them (any other format would be converted on the way, which FFmpeg's
// own decoding of the file does not do).
std::optional<std::string> raw_format(const VideoInfo& info) {
    const std::string format = ascii_lower(info.pix_fmt);
    if (format == "yuv420p" || format == "yuv420p10le") {
        return format;
    }
    return std::nullopt;
}

// A colour tag as a decoder option: ffprobe's name of it, renamed where the
// option's differs, and the names the option takes (libavcodec's
// options_table.h, FFmpeg 9).
struct DecoderTag {
    std::string option;
    std::string VideoInfo::* field;
    std::set<std::string, std::less<>> names;
};

const std::array<DecoderTag, 5>& decoder_tags() {
    static const std::array<DecoderTag, 5> tags{{
        {"-color_range", &VideoInfo::color_range, {"tv", "pc"}},
        {"-color_primaries", &VideoInfo::color_primaries,
         {"bt709", "bt470m", "bt470bg", "smpte170m", "smpte240m", "film", "bt2020", "smpte428", "smpte431", "smpte432",
          "jedec-p22", "ebu3213"}},
        {"-color_trc", &VideoInfo::color_transfer,
         {"bt709", "bt470m", "bt470bg", "smpte170m", "smpte240m", "linear", "log100", "log316", "iec61966-2-4",
          "bt1361e", "iec61966-2-1", "bt2020-10", "bt2020-12", "smpte2084", "smpte428", "arib-std-b67"}},
        {"-colorspace", &VideoInfo::color_space,
         {"rgb", "bt709", "fcc", "bt470bg", "smpte170m", "smpte240m", "ycgco", "bt2020nc", "bt2020c", "smpte2085",
          "chroma-derived-nc", "chroma-derived-c", "ictcp", "ipt-c2", "ycgco-re", "ycgco-ro"}},
        {"-chroma_sample_location", &VideoInfo::chroma_location,
         {"left", "center", "topleft", "top", "bottomleft", "bottom"}},
    }};
    return tags;
}

// A feeding's failure as the caller is to see it: Cancelled and refusals as
// they are, anything else final (FilmGrainRefusal), since the app's decoder
// is the only one for the video's pictures and no other route would do
// better.
[[noreturn]] void throw_final(const std::exception_ptr& failure) {
    try {
        std::rethrow_exception(failure);
    } catch (const Cancelled&) {
        throw;
    } catch (const FilmGrainRefusal&) {
        throw;
    } catch (const std::exception& error) {
        throw FilmGrainRefusal(std::string("the app could not decode its pictures: ") + error.what());
    }
}

// The frame rate FFmpeg gives the file's stream, which xpsnr's temporal mode
// goes by (libavformat's av_guess_frame_rate): its nominal rate
// (r_frame_rate), but the average where that is below 70 and the nominal
// above 210. (A codec coded in fields may get its own rate there too; VVC is
// not one.) A raw stream's is the one it is read at.
double fed_rate(const VideoInfo& info) {
    if (info.timed_raw || info.nominal_fps <= 0) {
        return info.fps;
    }
    if (info.average_fps > 0 && info.average_fps < 70 && info.nominal_fps > 210) {
        return info.average_fps;
    }
    return info.nominal_fps;
}

}  // namespace

bool fed_to_ffmpeg(const VideoInfo& info) {
    if (only_the_app_decodes(info)) {
        return true;  // refused where it cannot be fed: FFmpeg would score it wrongly, or not at all
    }
    return (decoded_by_the_app(info) || feeding_all()) && !feed_refusal(info);
}

std::optional<std::string> feed_refusal(const VideoInfo& info) {
#ifndef _WIN32
    static_cast<void>(info);
    return "the app decodes a video for FFmpeg on Windows only";
#else
    if (info.scaled_to) {
        return "its size changes during the video, which pictures fed to FFmpeg at one size cannot follow";
    }
    if (by_timestamp()) {
        return "frames are paired by their timestamps (VML_FRAME_PAIRING), which pictures fed to FFmpeg do not carry";
    }
    if (!software_decoder_bundled()) {
        return "the app's own decoder is not here (or VML_SOFTWARE_DECODE=ffmpeg turned it off)";
    }
    try {
        feed_plan(info);
    } catch (const GpuDecodeUnavailableError& error) {
        return std::string("the app's decoder does not decode it: ") + error.what();
    }
    if (!feed_input_options(info)) {
        return "its pixel format or a colour tag has no rawvideo equivalent for FFmpeg";
    }
    return std::nullopt;
#endif
}

std::optional<std::vector<std::string>> feed_input_options(const VideoInfo& info) {
    const std::optional<std::string> format = raw_format(info);
    if (!format) {
        return std::nullopt;
    }
    std::vector<std::string> options{"-f",           "rawvideo",
                                     "-video_size",  std::to_string(info.width) + "x" + std::to_string(info.height),
                                     "-pixel_format", *format,
                                     "-framerate",   python_repr(fed_rate(info))};
    for (const DecoderTag& tag : decoder_tags()) {
        std::string name = ascii_lower(info.*tag.field);
        if (name.empty() || name == "unknown" || name == "unspecified") {
            continue;  // untagged, as the file's frames are
        }
        if (name == "gbr") {
            name = "rgb";  // ffprobe's name of RGB's matrix, not the option's
        }
        if (!tag.names.contains(name)) {
            return std::nullopt;
        }
        options.insert(options.end(), {tag.option, name});
    }
    return options;
}

struct PictureFeed::State {
    std::string file_name;  // for the log
    std::string pipe_path;
    std::vector<std::string> arguments;
    std::thread feeding;
    std::mutex mutex;
    std::condition_variable changed;
    bool ready = false;  // the pipe is there to open
    bool ended = false;  // the feeding is over
    nlohmann::json result;
    std::exception_ptr failure;
};

PictureFeed::PictureFeed(const VideoInfo& info, const std::string& ffmpeg, const std::string& ffprobe,
                         const CancelToken& cancel)
    : state_(std::make_unique<State>()) {
    if (const std::optional<std::string> refusal = feed_refusal(info)) {
        throw FilmGrainRefusal(*refusal);
    }
    State& state = *state_;
    state.file_name = platform::path_text(info.path.filename());
    state.pipe_path = platform::OutboundPipe::unique_path("feed");
    state.arguments = *feed_input_options(info);
    state.arguments.insert(state.arguments.end(), {"-i", state.pipe_path});
    const nlohmann::json request = {
        {"video", video_json(info)}, {"pipe", state.pipe_path}, {"ffmpeg", ffmpeg}, {"ffprobe", ffprobe}};
    const auto say_ready = [&state] {
        const std::lock_guard lock(state.mutex);
        state.ready = true;
        state.changed.notify_all();
    };
    state.feeding = std::thread([&state, request, say_ready, &cancel] {
        nlohmann::json result;
        std::exception_ptr failure;
        try {
            if (isolation_program()) {
                result = run_isolated(
                    "grain-feed", "The app's decoder", request,
                    [&say_ready](const nlohmann::json& message) {
                        if (message.value("ready", false)) {
                            say_ready();
                        }
                    },
                    cancel);
            } else {
                result = picture_feed_task(request, say_ready);  // the tests: no program to run it in
            }
        } catch (...) {
            failure = std::current_exception();
        }
        const std::lock_guard lock(state.mutex);
        state.result = std::move(result);
        state.failure = failure;
        state.ended = true;
        state.changed.notify_all();
    });
    std::unique_lock lock(state.mutex);
    state.changed.wait(lock, [&state] { return state.ready || state.ended; });
    if (!state.ready) {
        lock.unlock();
        state.feeding.join();
        throw_final(state.failure ? state.failure
                                  : std::make_exception_ptr(std::runtime_error("it ended before its pipe was made")));
    }
}

PictureFeed::~PictureFeed() {
    if (state_->feeding.joinable()) {
        // A feeding still waiting for FFmpeg to open its pipe goes on, finds
        // no reader and ends.
        platform::OutboundPipe::touch(state_->pipe_path);
        state_->feeding.join();
    }
}

const std::vector<std::string>& PictureFeed::input_arguments() const { return state_->arguments; }

void PictureFeed::finish() {
    State& state = *state_;
    platform::OutboundPipe::touch(state.pipe_path);  // FFmpeg may have ended before opening it
    state.feeding.join();
    if (!state.failure) {
        // Fed to the video's end, or FFmpeg stopped reading first: where its
        // run was to end (a duration limit), or failing, which its caller
        // reports.
        logger.info(state.file_name + ": " + std::to_string(state.result.value("frames", 0LL)) +
                    " pictures decoded by the app for FFmpeg" +
                    (state.result.value("reader_gone", false) ? ", until FFmpeg stopped reading" : ""));
        return;
    }
    throw_final(state.failure);
}

PairFeeds::PairFeeds(const VideoInfo* source, const VideoInfo* distorted, const std::string& ffmpeg,
                     const std::string& ffprobe, const CancelToken& cancel) {
    // Both refusals before either feeding starts; a video the app decodes that
    // cannot be fed is decoded by FFmpeg where FFmpeg can (app_decoders.h),
    // and the log says why.
    for (const VideoInfo* info : {distorted, source}) {
        if (info == nullptr || !(decoded_by_the_app(*info) || fed_to_ffmpeg(*info))) {
            continue;
        }
        const std::optional<std::string> refusal = feed_refusal(*info);
        if (fed_to_ffmpeg(*info) && refusal) {
            throw FilmGrainRefusal(*refusal);
        }
        if (refusal && decoded_by_the_app(*info)) {
            logger.info(platform::path_text(info->path.filename()) + ": decoded by FFmpeg, not the app's decoder: " +
                        *refusal);
        }
    }
    if (distorted != nullptr && fed_to_ffmpeg(*distorted)) {
        distorted_.emplace(*distorted, ffmpeg, ffprobe, cancel);
    }
    if (source != nullptr && fed_to_ffmpeg(*source)) {
        source_.emplace(*source, ffmpeg, ffprobe, cancel);
    }
}

HardwareDecoding PairFeeds::decoding(HardwareDecoding decoding) const {
    if (source_) {
        decoding.source.reset();
    }
    if (distorted_) {
        decoding.distorted.reset();
    }
    return decoding;
}

void PairFeeds::finish() {
    if (distorted_) {
        distorted_->finish();
    }
    if (source_) {
        source_->finish();
    }
}

nlohmann::json picture_feed_task(const nlohmann::json& request, const std::function<void()>& on_ready) {
    const VideoInfo info = video_from(request.at("video"));
    const DecodePlan plan = feed_plan(info);
    if (const auto [supported, refusal] = decoder_supports(0, plan, software_decoder); !supported) {
        throw FilmGrainRefusal("the app's decoder does not decode it: " + refusal);
    }
    platform::OutboundPipe pipe(request.at("pipe").get<std::string>(), platform::frame_pipe_bytes);
    on_ready();
    // Decoding starts while FFmpeg starts and opens its inputs.
    GpuFrameStream stream(info, plan, 0, 4, software_decoder, request.at("ffmpeg").get<std::string>(),
                          request.at("ffprobe").get<std::string>());
    stream.start();
    pipe.connect();
    std::vector<std::uint8_t> picture(plan.frame_bytes());
    long long frames = 0;
    const CancelToken never;  // the parent ends this process to stop it
    while (const std::optional<GpuFrameStream::Picture> next = stream.next(never)) {
        stream.download(next->slot, picture.data());
        stream.release(next->slot);
        if (!pipe.write(picture.data(), picture.size())) {
            return {{"frames", frames}, {"reader_gone", true}};
        }
        ++frames;
    }
    const bool read = pipe.finish();
    return {{"frames", frames}, {"reader_gone", !read}};
}

}  // namespace vml
