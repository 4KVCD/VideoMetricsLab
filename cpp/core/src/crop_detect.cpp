#include "vml/crop_detect.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <semaphore>
#include <set>
#include <sstream>
#include <string_view>
#include <system_error>
#include <thread>
#include <tuple>
#include <vector>

#include "vml/app_decoders.h"
#include "vml/app_log.h"
#include "vml/append.h"
#include "vml/black_bars_rule.h"
#include "vml/ffmpeg_tools.h"
#include "vml/film_grain_scoring.h"
#include "vml/geometry.h"
#include "vml/gpu_frames.h"
#include "vml/grain_feed.h"
#include "vml/hardware_decoding.h"
#include "vml/hevc.h"
#include "vml/number_format.h"
#include "vml/platform/process.h"
#include "vml/platform/system.h"
#include "vml/python_text.h"
#include "vml/size_changes.h"

namespace vml {

namespace {
// A thread joined as it goes out of scope: std::jthread, which the macOS CI's
// standard library does not have.
class JoiningThread {
public:
    template <typename Function>
    explicit JoiningThread(Function function) : thread_(std::move(function)) {}
    JoiningThread(const JoiningThread&) = delete;
    JoiningThread& operator=(const JoiningThread&) = delete;
    ~JoiningThread() { thread_.join(); }

private:
    std::thread thread_;
};


using namespace std::chrono_literals;

const app_log::Logger logger("vml.crop_detect");

// A box for the log: "3840x1608 at 0,276".
std::string box_text(const CropBox& box) {
    return std::to_string(box.w) + "x" + std::to_string(box.h) + " at " + std::to_string(box.x) + "," +
           std::to_string(box.y);
}

// HandBrake's previews (black_bars_rule.h): how many, and how many first
// pictures stand for them in a video of unknown duration.
constexpr int preview_count = 10;
constexpr int unknown_duration_pictures = 20;
// How bright a dark line's mean may be (black_bars_rule.cpp's dark_mean),
// for the answers' key.
constexpr double dark_limit = 32.0 / 255.0;
// How many detection decoders may run at once in the whole app: one video's
// previews run in turn, so this limits two jobs detecting both their videos
// to two (2.5 GB of GPU memory for 4K on the GPU decoder).
std::counting_semaphore<2> decoder_slots(2);
// How many files' answers to remember (a bound against an extreme session).
constexpr std::size_t remembered_answers = 512;
// Raised when the detection itself changes what it finds, so that older
// answers in the store are ignored. 2: Intel's GPU decoded lossless H.264
// to other pixels before (pick_decode); 3: so did NVIDIA's before the RTX 50
// series; 4: HandBrake's rule in place of FFmpeg's cropdetect.
constexpr int store_version = 4;

// What an answer depends on: which bytes are in the file, and which stretch
// of it is sampled (not how it was decoded: the GPUs that decode a video
// give the pixels the CPU does, pick_decode). The Python app's key, so both
// apps read each other's store.
struct AnswerKey {
    std::string path;
    long long size = 0;
    long long modified_ns = 0;
    double sampled_duration = 0.0;
    double limit = 0.0;
    auto operator<=>(const AnswerKey&) const = default;
};

// The key for a file, or nothing when the file cannot be identified (then
// nothing is kept).
std::optional<AnswerKey> answer_key(const std::filesystem::path& path, double sampled_duration) {
    std::error_code error;
    const std::filesystem::path resolved = std::filesystem::weakly_canonical(std::filesystem::absolute(path, error), error);
    if (error) {
        return std::nullopt;
    }
    const auto size = std::filesystem::file_size(resolved, error);
    const std::optional<long long> written = platform::modified_ns(resolved);
    if (error || !written) {
        return std::nullopt;
    }
    return AnswerKey{platform::path_text(resolved), static_cast<long long>(size), *written,
                     python_round(sampled_duration, 3), python_round(dark_limit, 6)};
}

// The answers, newest last, and the files being detected now (a second
// caller for the same file waits for that answer instead of decoding it
// again).
class Answers {
public:
    void set_store(std::function<std::filesystem::path()> store) {
        const std::lock_guard lock(mutex_);
        store_ = std::move(store);
        store_read_.reset();
    }

    // Forgets every answer kept (not those being detected now), and reads
    // the store again when next asked.
    void forget() {
        const std::lock_guard lock(mutex_);
        boxes_.clear();
        store_read_.reset();
    }

    // Forgets one answer, in the store too (read first, so that its other
    // answers stay); nothing is written when it had none.
    void drop(const AnswerKey& key) {
        const std::lock_guard lock(mutex_);
        read_store();
        if (const auto found = find(key); found != boxes_.end()) {
            boxes_.erase(found);
            write_store();
        }
    }

    // The kept box, or nothing once this caller has been given the job of
    // detecting it (then it must call settle).
    std::optional<CropBox> claim(const AnswerKey& key, const CancelToken& cancel) {
        std::unique_lock lock(mutex_);
        while (true) {
            read_store();
            if (const auto found = find(key); found != boxes_.end()) {
                boxes_.splice(boxes_.end(), boxes_, found);
                return found->second;
            }
            if (!in_flight_.contains(key)) {
                in_flight_.insert(key);
                return std::nullopt;
            }
            // Another caller is on it: wait for it to finish, either way.
            settled_.wait_for(lock, 100ms);
            if (cancel.cancelled()) {
                throw Cancelled();
            }
        }
    }

    // Keeps a detected box (if any), and wakes the callers waiting for it.
    void settle(const AnswerKey& key, const std::optional<CropBox>& box) {
        {
            const std::lock_guard lock(mutex_);
            if (box) {
                if (const auto found = find(key); found != boxes_.end()) {
                    boxes_.erase(found);
                }
                boxes_.emplace_back(key, *box);
                while (boxes_.size() > remembered_answers) {
                    boxes_.pop_front();
                }
                write_store();
            }
            in_flight_.erase(key);
        }
        settled_.notify_all();
    }

private:
    std::list<std::pair<AnswerKey, CropBox>>::iterator find(const AnswerKey& key) {
        return std::find_if(boxes_.begin(), boxes_.end(), [&key](const auto& item) { return item.first == key; });
    }

    std::optional<std::filesystem::path> store_path() const {
        if (!store_) {
            return std::nullopt;
        }
        try {
            return store_();
        } catch (const std::exception&) {
            return std::nullopt;
        }
    }

    // Reads the store's answers, once per store file.
    void read_store() {
        const std::optional<std::filesystem::path> path = store_path();
        if (!path || path == store_read_) {
            return;
        }
        store_read_ = path;
        std::ifstream file(*path, std::ios::binary);
        if (!file) {
            return;
        }
        std::stringstream text;
        text << file.rdbuf();
        nlohmann::json data;
        try {
            data = nlohmann::json::parse(text.str());
        } catch (const nlohmann::json::exception&) {
            return;
        }
        if (!data.is_object() || data.value("version", 0) != store_version || !data.contains("boxes") ||
            !data["boxes"].is_array()) {
            return;
        }
        for (const nlohmann::json& item : data["boxes"]) {
            try {
                const nlohmann::json& key = item.at(0);
                const nlohmann::json& box = item.at(1);
                const AnswerKey answer{key.at(0).get<std::string>(), key.at(1).get<long long>(),
                                       key.at(2).get<long long>(), key.at(3).get<double>(), key.at(4).get<double>()};
                if (find(answer) == boxes_.end()) {
                    boxes_.emplace_back(answer, CropBox{box.at(0).get<int>(), box.at(1).get<int>(),
                                                        box.at(2).get<int>(), box.at(3).get<int>()});
                }
            } catch (const nlohmann::json::exception&) {
                continue;
            }
        }
        while (boxes_.size() > remembered_answers) {
            boxes_.pop_front();
        }
    }

    // Writes every answer to the store, replacing it whole. At worst, two
    // processes writing at once lose one's newest answer, detected again.
    void write_store() {
        const std::optional<std::filesystem::path> path = store_path();
        if (!path) {
            return;
        }
        nlohmann::ordered_json boxes = nlohmann::ordered_json::array();
        for (const auto& [key, box] : boxes_) {
            boxes.push_back({{key.path, key.size, key.modified_ns, key.sampled_duration, key.limit},
                             {box.w, box.h, box.x, box.y}});
        }
        // The keys in the Python app's order (crop_detect._write_store).
        const nlohmann::ordered_json data{{"version", store_version}, {"boxes", boxes}};
        std::filesystem::path partial = *path;
        partial += ".partial";
        {
            std::ofstream file(partial, std::ios::binary);
            file << data.dump();
            if (!file) {
                logger.warning("Could not keep black-bar answers in " + platform::path_text(*path) + ": " +
                               platform::path_text(partial) + " could not be written");
                return;
            }
        }
        std::error_code error;
        std::filesystem::rename(partial, *path, error);
        if (error) {
            logger.warning("Could not keep black-bar answers in " + platform::path_text(*path) + ": " +
                           error.message());
        }
    }

    std::mutex mutex_;
    std::condition_variable settled_;
    std::list<std::pair<AnswerKey, CropBox>> boxes_;
    std::set<AnswerKey> in_flight_;
    std::function<std::filesystem::path()> store_;
    std::optional<std::filesystem::path> store_read_;
};

Answers& answers() {
    static Answers kept;
    return kept;
}

// The previews' times: HandBrake's ten, at i / 11 of the video for i = 0 to
// 9, as its scan takes a stream's (the first at its start, which a dark
// opening leaves out as too dark to say anything). None for a video of
// unknown duration (a file still being written): its first pictures stand
// for them (unknown_duration_pictures).
std::vector<double> preview_times(double duration) {
    std::vector<double> times;
    if (duration > 0) {
        for (int i = 0; i < preview_count; ++i) {
            times.push_back(duration * i / (preview_count + 1));
        }
    }
    return times;
}

// What comes before the luma is taken: hwdownload where the frames are
// decoded on the GPU, which gives the surface's own format (`download_format`);
// then the scale of a video whose size changes (size_changes.h), as the run
// scales its frames before it crops them.
struct BeforeSampling {
    std::string download_format;
    std::optional<std::string> scale;

    // The filters for frames decoded with `hwaccel`, each followed by a
    // comma; nothing for a video decoded in software at one size.
    std::string filters(const std::optional<std::string>& hwaccel) const {
        return (hwaccel ? "hwdownload,format=" + download_format + "," : "") + (scale ? *scale + "," : "");
    }

    // Those filters and the luma alone at 16 bits (black_bars_rule.h's
    // Y4mLumaReader: the high byte is the luma at 8 bits, whatever its own
    // depth). extractplanes leaves the samples as they are; FFmpeg converts
    // a format it cannot take (RGB) to YUV before it.
    std::string luma(const std::optional<std::string>& hwaccel) const {
        return filters(hwaccel) + "extractplanes=y,format=gray16le";
    }
};

// FFmpeg's output of the pictures: Y4M on standard output (16-bit mono is
// "unofficial" Y4M, so -strict -1), each picture once: Y4M's muxer is
// constant frame rate, and would fill the gaps between pictures picked out
// (raw_previews_command) with copies of them.
const std::vector<std::string> luma_output{"-fps_mode", "passthrough", "-f", "yuv4mpegpipe", "-strict", "-1", "-"};

// The command that decodes `frames` pictures from `start` seconds (-ss, an
// accurate seek: the pictures from that time on, as the app's decoder feeds
// them), `input` the video's input options (a raw YUV file's size and
// format, VideoInfo::input_options).
std::vector<std::string> preview_command(const std::string& ffmpeg, const std::string& path,
                                         const std::vector<std::string>& input, double start, int frames,
                                         const std::optional<std::string>& hwaccel, const BeforeSampling& before) {
    std::vector<std::string> command = ffmpeg_command_start(ffmpeg, {"-nostats"});
    const std::vector<std::string> decoding = hwaccel_args(hwaccel);
    append(command, decoding);
    append(command, input);
    command.insert(command.end(), {"-ss", format_fixed(start, 3), "-i", path, "-map", "0:" + std::string(video_stream),
                                   "-frames:v", std::to_string(frames), "-vf", before.luma(hwaccel)});
    append(command, luma_output);
    return command;
}

// The frame of a raw stream (VideoInfo::timed_raw) HandBrake judges for each
// preview starting at `times`: the second from the first at or after its
// time (a frame's time its number over the frame rate, the time to the
// millisecond as -ss takes it); for a video of unknown duration, every
// second one of its first pictures.
std::vector<long long> preview_frames(const VideoInfo& info, const std::vector<double>& times) {
    std::vector<long long> frames;
    if (times.empty()) {
        for (int i = 1; i < unknown_duration_pictures; i += 2) {
            frames.push_back(i);
        }
        return frames;
    }
    for (const double time : times) {
        // The first frame at or after the time, against the rounding of the product
        const double millisecond = std::nearbyint(time * 1000) / 1000;
        frames.push_back(static_cast<long long>(std::ceil(millisecond * info.fps - 1e-6)) + 1);
    }
    return frames;
}

// Whether a raw stream can be read from a byte partway through
// (run_byte_previews): its format marks where each picture starts (start
// codes, JPEG markers) and its decoder waits for a picture it can decode on
// its own. Started at 30% and 60% of a file, these gave only right pictures;
// AV1's OBUs and EVC gave none, and H.263's and H.261's decoders broken ones.
bool reads_from_a_byte(const VideoInfo& info) {
    static const std::set<std::string, std::less<>> formats{"h264",  "hevc",      "vvc",  "mpegvideo",
                                                            "mjpeg", "jpeg_pipe", "dirac"};
    return formats.contains(info.format_name);
}

// A stretch of a file's bytes: [first, end), an end of 0 running to the
// file's end.
using ByteRange = std::pair<std::uintmax_t, std::uintmax_t>;

// Whether an HEVC unit that holds no picture ends the access unit before it
// rather than starting the next (H.265 7.4.2.4.4): end of sequence or of
// bitstream, filler data, suffix SEI, and the reserved and unspecified types
// allowed only after a picture's slices (Dolby Vision's 62 and 63 among them).
bool ends_access_unit(int type) {
    return (type >= 36 && type <= 38) || type == 40 || (type >= 45 && type <= 47) || type >= 56;
}

// One unit of an HEVC stream: where its start code begins in the file, its
// type and layer, and for a slice, whether it is its picture's first.
struct HevcUnit {
    std::uintmax_t position = 0;
    int type = 0;
    int layer = 0;
    bool first_slice = false;
};

// Hands each unit of an HEVC stream (H.265 Annex B: each after a 00 00 01
// start code, which the stream's bytes hold nowhere else) from byte `from`
// of its file to `visit`, in order, until `visit` returns false or the file
// ends. A file that cannot be read has no units.
void walk_hevc_units(const std::filesystem::path& path, std::uintmax_t from, const CancelToken& cancel,
                     const std::function<bool(const HevcUnit&)>& visit) {
    std::ifstream file(path, std::ios::binary);
    if (!file.seekg(static_cast<std::streamoff>(from))) {
        return;
    }
    constexpr std::string_view start_code("\0\0\1", 3);
    // A start code, the unit's two header bytes, and a slice header's first.
    constexpr std::size_t head = 6;
    std::string buffer;  // the file's bytes from buffer_at
    std::uintmax_t buffer_at = from;
    std::string chunk(std::size_t{1} << 20, '\0');
    while (file) {
        cancel.throw_if_cancelled();
        file.read(chunk.data(), static_cast<std::streamsize>(chunk.size()));
        buffer.append(chunk, 0, static_cast<std::size_t>(file.gcount()));
        for (std::size_t at = buffer.find(start_code); at != std::string::npos && at + head <= buffer.size();
             at = buffer.find(start_code, at + start_code.size())) {
            const auto byte = [&buffer, at](std::size_t index) {
                return static_cast<int>(static_cast<unsigned char>(buffer[at + index]));
            };
            const HevcUnit unit{buffer_at + at, hevc_unit::type_of(byte(3)), hevc_unit::layer_of(byte(3), byte(4)),
                                (byte(5) & 0x80) != 0};
            if (!visit(unit)) {
                return;
            }
        }
        // Every start code beginning before the last five bytes has been
        // handed on; one beginning in them is completed by the next read.
        const std::size_t kept = std::min(buffer.size(), head - 1);
        buffer_at += buffer.size() - kept;
        buffer.erase(0, buffer.size() - kept);
    }
}

// The bytes of an HEVC stream's parameter sets (VPS, SPS, PPS) from before
// its first picture, each from its start code to the next unit's, adjoining
// ones as one range; nothing for a stream with no picture.
std::vector<ByteRange> hevc_parameter_sets(const std::filesystem::path& path, const CancelToken& cancel) {
    std::vector<ByteRange> ranges;
    std::optional<std::uintmax_t> parameter_set;  // where the one before this unit began
    bool picture = false;
    walk_hevc_units(path, 0, cancel, [&](const HevcUnit& unit) {
        if (parameter_set) {
            if (!ranges.empty() && ranges.back().second == *parameter_set) {
                ranges.back().second = unit.position;
            } else {
                ranges.emplace_back(*parameter_set, unit.position);
            }
            parameter_set.reset();
        }
        if (unit.type <= hevc_unit::last_picture) {
            picture = true;
            return false;
        }
        if (unit.type >= hevc_unit::first_parameter_set && unit.type <= hevc_unit::last_parameter_set) {
            parameter_set = unit.position;
        }
        return true;
    });
    return picture ? ranges : std::vector<ByteRange>{};
}

// Where the access unit of the first random access picture (IDR, CRA or
// BLA) of an HEVC stream's base layer at or after byte `from` begins: at its
// first slice, or at the units before it that belong to it (a delimiter,
// parameter sets, SEI). Nothing when there is none.
std::optional<std::uintmax_t> hevc_random_access(const std::filesystem::path& path, std::uintmax_t from,
                                                 const CancelToken& cancel) {
    std::optional<std::uintmax_t> found;
    std::optional<std::uintmax_t> access_unit;  // the first unit since the last picture's
    walk_hevc_units(path, from, cancel, [&](const HevcUnit& unit) {
        if (unit.type <= hevc_unit::last_picture) {
            if (unit.layer == 0 && unit.first_slice && unit.type >= hevc_unit::first_random_access &&
                unit.type <= hevc_unit::last_random_access) {
                found = access_unit.value_or(unit.position);
                return false;
            }
            access_unit.reset();
        } else if (ends_access_unit(unit.type)) {
            access_unit.reset();
        } else if (!access_unit) {
            access_unit = unit.position;
        }
        return true;
    });
    return found;
}

// FFmpeg's input made of the byte ranges `parts` of the file `path`, one
// after the other (its concat and subfile protocols).
std::string pieced_input(const std::string& path, const std::vector<ByteRange>& parts) {
    std::string input;
    for (const auto& [first, end] : parts) {
        input += (input.empty() ? "concat:" : "|") + std::string("subfile,,start,") + std::to_string(first) + ",end," +
                 std::to_string(end) + ",,:" + path;
    }
    return input;
}

// The command that decodes a raw stream's preview from `input` (the input
// options and -i): its first `frames` pictures from there.
std::vector<std::string> byte_preview_command(const std::string& ffmpeg, const std::vector<std::string>& input,
                                              int frames, const std::optional<std::string>& hwaccel,
                                              const BeforeSampling& before) {
    std::vector<std::string> command = ffmpeg_command_start(ffmpeg, {"-nostats"});
    const std::vector<std::string> decoding = hwaccel_args(hwaccel);
    append(command, decoding);
    append(command, input);
    command.insert(command.end(), {"-map", "0:" + std::string(video_stream), "-frames:v", std::to_string(frames), "-vf",
                                   before.luma(hwaccel)});
    append(command, luma_output);
    return command;
}

// The command that decodes a raw stream's preview frames `frames` (in order,
// each once) in one pass from its start, for a stream that cannot be read
// from a byte partway (reads_from_a_byte), or whose previews found no picture
// from there (its only keyframe its first): select picks them out, and the
// pass stops after the last (trim).
std::vector<std::string> raw_previews_command(const std::string& ffmpeg, const VideoInfo& info,
                                              const std::vector<long long>& frames,
                                              const std::optional<std::string>& hwaccel,
                                              const BeforeSampling& before) {
    std::string picked;
    for (const long long frame : frames) {
        picked += (picked.empty() ? "" : "+") + std::string("eq(n\\,") + std::to_string(frame) + ")";
    }
    const std::string filters = "trim=end_frame=" + std::to_string(frames.back() + 1) + ",select=gt(" + picked +
                                "\\,0)," + before.luma(hwaccel);
    std::vector<std::string> command = ffmpeg_command_start(ffmpeg, {"-nostats"});
    const std::vector<std::string> decoding = hwaccel_args(hwaccel);
    append(command, decoding);
    const std::vector<std::string> input = info.input_options();
    append(command, input);
    command.insert(command.end(), {"-i", platform::path_text(info.path), "-map", "0:" + std::string(video_stream),
                                   "-vf", filters});
    append(command, luma_output);
    return command;
}

// The pictures a detection command decoded, judged as they came
// (picture_edges: nothing for one too dark to say), and their size.
struct Pictures {
    int width = 0;
    int height = 0;
    std::vector<std::optional<BarEdges>> edges;
};

// What a detection command gave: its exit code, the end of its error output
// (for a failure's message), the lines of it the detection reads (FFmpeg's
// word that it rebuilt its filters), its pictures, and why its output could
// not be read, if it could not.
struct DetectionOutput {
    int code = 0;
    std::string tail;
    std::string lines;
    Pictures pictures;
    std::string read_error;
};

// A detection decoder slot (decoder_slots), held while this lives. Waiting
// for one ends with a Cancel (Cancelled), and one that comes as the run is
// cancelled is not kept.
class DecoderSlot {
public:
    explicit DecoderSlot(const CancelToken& cancel) {
        while (!decoder_slots.try_acquire_for(100ms)) {
            cancel.throw_if_cancelled();
        }
        if (cancel.cancelled()) {
            decoder_slots.release();
            cancel.throw_if_cancelled();
        }
    }
    ~DecoderSlot() { decoder_slots.release(); }
    DecoderSlot(const DecoderSlot&) = delete;
    DecoderSlot& operator=(const DecoderSlot&) = delete;
};

// Reads `process`'s pictures (Y4M luma, luma_output) into `pictures` until
// its output ends; a stream it cannot read is drained, so that FFmpeg never
// waits for room in the pipe, and why goes to `error`.
void read_pictures(platform::ChildProcess& process, Pictures& pictures, std::string& error) {
    const auto read = [&process](void* into, std::size_t bytes) { return process.read(into, bytes); };
    try {
        Y4mLumaReader reader(read);
        while (const std::optional<LumaPicture> picture = reader.next()) {
            pictures.width = picture->width;
            pictures.height = picture->height;
            pictures.edges.push_back(picture_edges(*picture));
        }
    } catch (const std::exception& failure) {
        error = failure.what();
        char drained[65536];
        while (read(drained, sizeof drained) > 0) {
        }
    }
}

// Runs one detection command to its end, a decoder slot already held, for
// at most `limit` of running (paused time not counted), its pictures read
// on a thread of their own while it runs.
DetectionOutput run_ffmpeg(const std::vector<std::string>& command, const CancelToken& cancel,
                           std::chrono::seconds limit) {
    logger.info("FFmpeg: " + python_list2cmdline(command));
    struct Kept {
        std::mutex mutex;
        std::string lines;
    };
    const auto kept = std::make_shared<Kept>();
    platform::ProcessOptions options;
    options.on_error_line = [kept](std::string_view line) {
        if (filters_rebuilt(line)) {
            const std::lock_guard lock(kept->mutex);
            kept->lines.append(line).push_back('\n');
        }
    };
    try {
        DetectionOutput output;
        platform::ChildProcess ffmpeg_process = platform::ChildProcess::start(command, options);
        {
            // Joined before `ffmpeg_process` goes: the program has ended or been
            // ended by then, so its output has too
            const JoiningThread reader([&] { read_pictures(ffmpeg_process, output.pictures, output.read_error); });
            const CancelToken::Attached attached = cancel.attach(ffmpeg_process);
            auto running = std::chrono::steady_clock::duration::zero();  // paused time not counted
            while (!ffmpeg_process.wait_for(200ms)) {
                if (cancel.cancelled()) {
                    ffmpeg_process.terminate();
                }
                if (!cancel.paused()) {
                    running += 200ms;
                }
                if (running >= limit) {
                    ffmpeg_process.terminate();
                    ffmpeg_process.wait();
                    throw CropDetectError("Crop detection timed out");
                }
            }
            output.code = ffmpeg_process.wait();
        }
        cancel.throw_if_cancelled();
        output.tail = ffmpeg_process.standard_error();  // waits for the error output's end, and so its lines
        const std::lock_guard lock(kept->mutex);
        output.lines = std::move(kept->lines);
        return output;
    } catch (const platform::ProcessError& error) {
        throw CropDetectError(std::string("Could not run crop detection: ") + error.what());
    }
}

// The last line of FFmpeg's error output (the output as a whole stripped of
// spaces at both ends first, as Python strips it); empty if it wrote none.
std::string last_error_line(const std::string& errors) {
    const std::size_t first = errors.find_first_not_of(" \t\r\n\f\v");
    if (first == std::string::npos) {
        return {};
    }
    const std::string stripped = errors.substr(first, errors.find_last_not_of(" \t\r\n\f\v") - first + 1);
    const std::size_t line_break = stripped.find_last_of("\r\n");
    return line_break == std::string::npos ? stripped : stripped.substr(line_break + 1);
}

// Why a detection command failed, for its error: FFmpeg's last line, or its
// exit code, or why its pictures could not be read.
std::string failure_of(const DetectionOutput& output) {
    if (output.code == 0) {
        return "its pictures could not be read (" + output.read_error + ")";
    }
    const std::string line = last_error_line(output.tail);
    return line.empty() ? "ffmpeg exited with code " + platform::exit_code_text(output.code) : line;
}

// Runs the detection command `command` makes for a decoder, the GPU's
// (`hwaccel`) and else software, for at most `limit` each, and returns its
// pictures. A GPU decoder that fails is dropped from `hwaccel`, so that the
// video's next previews go to software at once. Throws CropDetectError when
// it fails, and VideoSizeChanged when FFmpeg rebuilt its filters partway.
Pictures run_detection(const std::function<std::vector<std::string>(const std::optional<std::string>&)>& command,
                       const std::string& path, const CancelToken& cancel, std::optional<std::string>& hwaccel,
                       std::chrono::seconds limit) {
    cancel.throw_if_cancelled();
    DetectionOutput output;
    while (true) {
        std::string failure;
        try {
            output = run_ffmpeg(command(hwaccel), cancel, limit);
            if ((output.code == 0 && output.read_error.empty()) || !hwaccel) {
                break;
            }
            failure = failure_of(output);
        } catch (const CropDetectError& error) {
            if (!hwaccel) {
                throw CropDetectError(std::string(error.what()) + " (" + path + ")");
            }
            failure = error.what();  // a GPU decoder that hung (timed out) or could not start
        }
        // A hardware attempt that fails is tried again in software; the
        // pixels, and so the bars, are the same. Info, not a warning: a GPU
        // that cannot decode the video is expected, and software answers.
        logger.info("Black bars in " + platform::path_text(platform::path_from_text(path).filename()) + ": " +
                    failure + " decoding with " + *hwaccel + "; sampling in software from here");
        hwaccel.reset();
    }
    if (filters_rebuilt(output.lines)) {
        throw VideoSizeChanged(platform::path_from_text(path));
    }
    if (output.code != 0 || !output.read_error.empty()) {
        throw CropDetectError("Crop detection failed for " + path + ": " + failure_of(output));
    }
    return std::move(output.pictures);
}

// One preview decoded by FFmpeg from the file: `frames` pictures from
// `start`.
Pictures run_preview(const std::string& ffmpeg, const VideoInfo& info, double start, int frames,
                     const CancelToken& cancel, std::optional<std::string>& hwaccel, const BeforeSampling& before) {
    const std::string path = platform::path_text(info.path);
    return run_detection(
        [&](const std::optional<std::string>& attempt) {
            return preview_command(ffmpeg, path, info.input_options(), start, frames, attempt, before);
        },
        path, cancel, hwaccel, 60s);
}

// One preview decoded by the app's decoder (samples_through_the_app): the
// pictures of `window` fed to FFmpeg, which takes their luma as
// preview_command does. Throws FilmGrainRefusal when the app could not feed
// the window, CropDetectError when FFmpeg failed, and Cancelled.
Pictures run_fed_preview(const std::string& ffmpeg, const VideoInfo& info, const FeedWindow& window,
                         const CancelToken& cancel) {
    // The coded pictures, without their grain, as FFmpeg's vvc gives a video
    // deeper than 8 bits (it adds H.274's grain at 8 bits only). VVC's
    // decoder asks ffprobe nothing (gpu_frames.cpp's configuration).
    VideoInfo coded = info;
    coded.with_film_grain = false;
    PictureFeed feed(coded, ffmpeg, "", cancel, window);
    std::vector<std::string> command = ffmpeg_command_start(ffmpeg, {"-nostats"});
    append(command, feed.input_arguments());
    command.insert(command.end(), {"-map", "0:" + std::string(video_stream), "-vf", BeforeSampling{}.luma({})});
    append(command, luma_output);
    DetectionOutput output = run_ffmpeg(command, cancel, 60s);
    feed.finish();  // a window fed in part gives no pictures
    if (output.code != 0 || !output.read_error.empty()) {
        throw CropDetectError("Crop detection failed for " + platform::path_text(info.path) + ": " +
                              failure_of(output));
    }
    return std::move(output.pictures);
}

// A preview from the app's decoder (run_fed_preview), or from FFmpeg as any
// other video's where the app cannot feed it and FFmpeg decodes the codec
// (app_decoders.h); else it fails.
Pictures run_preview_through_the_app(const std::string& ffmpeg, const VideoInfo& info, double start, int frames,
                                     const CancelToken& cancel, std::optional<std::string>& hwaccel,
                                     const BeforeSampling& before) {
    try {
        // From the millisecond preview_command's -ss gives FFmpeg (its text,
        // so that a start on a half millisecond is rounded as that text
        // rounds it); the window's length to the frame before the last
        // wanted, so that it ends with that one (WindowCut)
        const double start_ms = python_round(start, 3);
        const double frame = info.fps > 0 ? 1.0 / info.fps : 1.0 / 24;
        return run_fed_preview(ffmpeg, info, FeedWindow{start_ms, frame * (frames - 1)}, cancel);
    } catch (const FilmGrainRefusal& refusal) {
        if (const std::optional<std::string> failure = unfed_failure(info, refusal)) {
            throw CropDetectError("Crop detection failed for " + platform::path_text(info.path) + ": " + *failure);
        }
        logger.info("Black bars in " + platform::path_text(info.path.filename()) + ": a preview decoded by FFmpeg, " +
                    "as the app's decoder could not feed it (" + refusal.what() + ")");
    }
    return run_preview(ffmpeg, info, start, frames, cancel, hwaccel, before);
}

// The previews of a raw stream, each started from a byte partway through,
// and why any failed: FFmpeg cannot seek in a stream without timestamps (-ss
// lands on no picture), so each starts at the share of the file its time is
// of the video, and is its first pictures decoded from there. A preview that
// decoded no picture (none after its byte, past the stream's last keyframe)
// has none.
std::pair<std::vector<Pictures>, std::vector<std::string>> run_byte_previews(
    const std::string& ffmpeg, const VideoInfo& info, std::uintmax_t file_bytes, const std::vector<double>& times,
    const CancelToken& cancel, std::optional<std::string>& hwaccel, const BeforeSampling& before) {
    const std::string path = platform::path_text(info.path);
    // An HEVC stream's previews start at a random access picture, after the
    // parameter sets from before its first picture: x265 writes them only
    // there unless told to repeat them (--repeat-headers), and a preview
    // without them decodes nothing. ('|' in the path would split the input.)
    const bool hevc = info.format_name == "hevc";
    std::vector<ByteRange> parameter_sets;
    if (hevc) {
        parameter_sets = hevc_parameter_sets(info.path, cancel);
        if (parameter_sets.empty() || path.find('|') != std::string::npos) {
            return {};
        }
    }
    std::vector<Pictures> previews;
    std::vector<std::string> failures;
    for (const double time : times) {
        const auto offset = static_cast<std::uintmax_t>(static_cast<double>(file_bytes) * time / info.duration);
        std::vector<std::string> input = info.input_options();
        if (hevc) {
            const std::optional<std::uintmax_t> access = hevc_random_access(info.path, offset, cancel);
            if (!access) {
                break;  // none after this preview's byte, so none after a later one's
            }
            std::vector<ByteRange> parts = parameter_sets;
            parts.emplace_back(*access, 0);
            input.insert(input.begin(), {"-f", "hevc"});
            input.insert(input.end(), {"-i", pieced_input(path, parts)});
        } else {
            input.insert(input.end(), {"-skip_initial_bytes", std::to_string(offset), "-i", path});
        }
        try {
            previews.push_back(run_detection(
                [&](const std::optional<std::string>& attempt) {
                    return byte_preview_command(ffmpeg, input, 2, attempt, before);
                },
                path, cancel, hwaccel, 60s));
        } catch (const CropDetectError& error) {
            failures.emplace_back(error.what());
        }
    }
    return {previews, failures};
}

// Each preview of a raw stream from one pass from its start
// (raw_previews_command): the picture judged of each, by its frame number.
std::vector<Pictures> run_raw_previews(const std::string& ffmpeg, const VideoInfo& info,
                                       const std::vector<double>& times, const CancelToken& cancel,
                                       std::optional<std::string>& hwaccel, const BeforeSampling& before) {
    const std::vector<long long> wanted = preview_frames(info, times);
    std::vector<long long> frames = wanted;  // each once, in order (previews of a short video share frames)
    std::sort(frames.begin(), frames.end());
    frames.erase(std::unique(frames.begin(), frames.end()), frames.end());
    // A minute, and a tenth of a second for each frame the pass decodes: a
    // bound against a pass that hangs, which a slow decoder (4K VVC in
    // software) stays well inside.
    const std::chrono::seconds limit = 60s + std::chrono::seconds(frames.back() / 10);
    const std::string path = platform::path_text(info.path);
    const Pictures picked = run_detection(
        [&](const std::optional<std::string>& attempt) {
            return raw_previews_command(ffmpeg, info, frames, attempt, before);
        },
        path, cancel, hwaccel, limit);
    // The pictures come in the frames' order, those past the stream's end
    // missing: each preview's is its frame's
    std::vector<Pictures> previews;
    for (const long long frame : wanted) {
        Pictures preview{picked.width, picked.height, {}};
        const auto found = std::lower_bound(frames.begin(), frames.end(), frame);
        const auto at = static_cast<std::size_t>(found - frames.begin());
        if (at < picked.edges.size()) {
            preview.edges.push_back(picked.edges[at]);
        }
        previews.push_back(std::move(preview));
    }
    return previews;
}

// A video of unknown duration's first pictures (unknown_duration_pictures)
// as the previews they stand for: each two, in order.
std::vector<Pictures> as_previews(const Pictures& first) {
    std::vector<Pictures> previews;
    for (std::size_t at = 0; at < first.edges.size(); at += 2) {
        Pictures preview{first.width, first.height, {}};
        preview.edges.assign(first.edges.begin() + static_cast<std::ptrdiff_t>(at),
                             first.edges.begin() + static_cast<std::ptrdiff_t>(std::min(at + 2, first.edges.size())));
        previews.push_back(std::move(preview));
    }
    return previews;
}

// The picture HandBrake judges of a preview: its second (a stream's first
// may be a dummy picture), or its first where it gave one alone; nothing for
// a preview that decoded none.
std::optional<std::optional<BarEdges>> judged(const Pictures& preview) {
    if (preview.edges.empty()) {
        return std::nullopt;
    }
    return preview.edges.size() >= 2 ? preview.edges[1] : preview.edges[0];
}

// The previews of a raw stream: from bytes partway where the stream allows
// it (a pass from the start reaches the last preview only after decoding
// nine tenths of the video: for a 4K VVC film, still not there after 21
// minutes), else in one pass from the start. The first preview, from the
// first byte, always decodes: fewer than fewest_judged decoded from bytes
// says the stream cannot be read from partway (its only keyframe its first,
// or its parameter sets only at its start).
std::pair<std::vector<Pictures>, std::vector<std::string>> sample_raw_stream(
    const std::string& ffmpeg, const VideoInfo& info, const std::vector<double>& times, const CancelToken& cancel,
    std::optional<std::string>& hwaccel, const BeforeSampling& before) {
    std::vector<Pictures> previews;
    std::vector<std::string> failures;
    std::error_code unknown;
    const std::uintmax_t file_bytes = std::filesystem::file_size(info.path, unknown);
    if (reads_from_a_byte(info) && !unknown && !times.empty()) {
        std::tie(previews, failures) = run_byte_previews(ffmpeg, info, file_bytes, times, cancel, hwaccel, before);
    }
    const auto decoded = static_cast<std::size_t>(
        std::count_if(previews.begin(), previews.end(), [](const Pictures& p) { return !p.edges.empty(); }));
    if (decoded < fewest_judged) {
        logger.info("Black bars in " + platform::path_text(info.path.filename()) +
                    ": sampling in one pass from the start, as too few previews could start partway through");
        failures.clear();
        try {
            previews = run_raw_previews(ffmpeg, info, times, cancel, hwaccel, before);
        } catch (const CropDetectError& error) {
            failures.emplace_back(error.what());
        }
    }
    return {previews, failures};
}

// detect_crop's work, without the kept answers: HandBrake's previews
// (black_bars_rule.h), each a decoder's run of its own, one after another.
CropBox detect_uncached(const std::string& ffmpeg, const VideoInfo& info, double sampled_duration,
                        const CancelToken& cancel, const std::optional<std::string>& gpu_decoder,
                        const std::string& scale_algorithm) {
    const std::string path = platform::path_text(info.path);
    const BeforeSampling before{hw_native_format(info.pix_fmt), size_change_scale(info, scale_algorithm)};
    const std::vector<double> times = preview_times(sampled_duration);
    std::optional<std::string> hwaccel = gpu_decoder;  // dropped for the rest once it fails
    cancel.throw_if_cancelled();
    // One decoder slot for all of the video's previews: they run in turn
    const DecoderSlot slot(cancel);
    std::vector<Pictures> previews;
    std::vector<std::string> failures;
    // A raw YUV file's pictures are all one size, so FFmpeg's -ss finds a
    // preview's first frame in it exactly, as in a container. A raw stream
    // the app reads itself (AV2) has its previews from the app's decoder.
    const bool through_the_app = samples_through_the_app(info);
    if (info.timed_raw && info.format_name != "rawvideo" && !through_the_app) {
        std::tie(previews, failures) = sample_raw_stream(ffmpeg, info, times, cancel, hwaccel, before);
    } else {
        if (through_the_app) {
            logger.info("Black bars in " + platform::path_text(info.path.filename()) + ": sampled with " +
                        std::string(app_decoded_codec(ascii_lower(info.codec_name))->decoder) +
                        ", the app's decoder of its codec");
        }
        const auto preview = [&](double start, int frames) {
            return through_the_app
                       ? run_preview_through_the_app(ffmpeg, info, start, frames, cancel, hwaccel, before)
                       : run_preview(ffmpeg, info, start, frames, cancel, hwaccel, before);
        };
        try {
            if (times.empty()) {
                previews = as_previews(preview(0, unknown_duration_pictures));
            }
            for (const double time : times) {
                try {
                    previews.push_back(preview(time, 2));
                } catch (const CropDetectError& error) {
                    failures.emplace_back(error.what());
                }
            }
        } catch (const CropDetectError& error) {
            failures.emplace_back(error.what());
        }
    }
    cancel.throw_if_cancelled();  // a partial sampling is not an answer
    // Every preview's picture judged, at one size
    int width = 0;
    int height = 0;
    std::size_t decoded = 0;
    std::vector<BarEdges> counted;
    for (const Pictures& sampled : previews) {
        const std::optional<std::optional<BarEdges>> picture = judged(sampled);
        if (!picture) {
            continue;
        }
        if (decoded > 0 && (sampled.width != width || sampled.height != height)) {
            throw VideoSizeChanged(info.path);  // previews of two sizes: one box cannot cut both
        }
        width = sampled.width;
        height = sampled.height;
        ++decoded;
        if (*picture) {
            counted.push_back(**picture);
        }
    }
    if (decoded == 0 && failures.empty()) {
        throw CropDetectError("Could not auto-detect black bars in " + path +
                              ": FFmpeg found no pictures in the parts of it that were sampled. The file may be "
                              "damaged or cut short.");
    }
    if (decoded == 0) {
        throw CropDetectError("Could not auto-detect black bars in " + path + ": " + failures.back() +
                              ". Choose 'None (use full frame)' for this video to continue without cropping.");
    }
    logger.info("Black bars in " + platform::path_text(info.path.filename()) + ": " + std::to_string(counted.size()) +
                " of " + std::to_string(decoded) + " previews judged" +
                (counted.size() < std::min(fewest_judged, decoded) ? " (too few: the whole frame)" : ""));
    return handbrake_box(counted, decoded, width, height);
}

}  // namespace

bool samples_through_the_app(const VideoInfo& info) { return windows_from_the_app(info); }

void set_crop_store(std::function<std::filesystem::path()> store) { answers().set_store(std::move(store)); }

void forget_crop_answers() { answers().forget(); }

void forget_crop_answer(const VideoInfo& info) {
    // (a video whose size changes keeps no answer, but one found before it was seen to change may be kept)
    if (const std::optional<AnswerKey> key = answer_key(info.path, info.duration)) {
        answers().drop(*key);
    }
}

CropBox detect_crop(const std::string& ffmpeg, const VideoInfo& info, const CancelToken& cancel,
                    const std::optional<std::string>& hwaccel, const std::string& scale_algorithm) {
    const double sampled_duration = info.duration;  // the whole video: the runs give no limit
    // A video whose size changes is sampled scaled, and its box is not kept:
    // the key knows the file, not how its frames are scaled.
    const std::optional<AnswerKey> key =
        info.scaled_to ? std::nullopt : answer_key(info.path, sampled_duration);
    if (key) {
        if (const std::optional<CropBox> kept = answers().claim(*key, cancel)) {
            return *kept;
        }
    }
    std::optional<CropBox> box;
    try {
        box = detect_uncached(ffmpeg, info, sampled_duration, cancel, hwaccel, scale_algorithm);
        logger.info("Black bars in " + platform::path_text(info.path.filename()) + ": picture " + box_text(*box) +
                    " of " + std::to_string(info.frame_size().first) + "x" + std::to_string(info.frame_size().second));
    } catch (const CropDetectError& error) {
        logger.warning("Black-bar detection failed for " + platform::path_text(info.path.filename()) + ": " +
                       error.what());
        if (key) {
            answers().settle(*key, std::nullopt);
        }
        throw;
    } catch (...) {
        if (key) {
            answers().settle(*key, std::nullopt);
        }
        throw;
    }
    if (key) {
        // Not kept when it was found from the GPU's pictures while its decoding was being found wrong: the run is
        // made again on the CPU, and finds it again there
        const bool spoiled = hwaccel.has_value() && gpu_decoding_distrusted().has_value();
        answers().settle(*key, spoiled ? std::nullopt : box);
    }
    return *box;
}

std::pair<CropBox, CropBox> detect_pair(const std::string& ffmpeg, const VideoInfo& source,
                                        const VideoInfo& distorted, const CancelToken& cancel,
                                        const std::optional<std::string>& source_hwaccel,
                                        const std::optional<std::string>& distorted_hwaccel,
                                        const std::string& scale_algorithm) {
    // Each video's previews run in turn, so this is two decoders, in about
    // half the time of one video after the other. Both are waited for before
    // anything is thrown: a cancel first, then the source's error.
    std::optional<CropBox> boxes[2];
    std::exception_ptr errors[2];
    const auto detect = [&](int index, const VideoInfo& info, const std::optional<std::string>& hwaccel) {
        try {
            boxes[index] = detect_crop(ffmpeg, info, cancel, hwaccel, scale_algorithm);
        } catch (...) {
            errors[index] = std::current_exception();
        }
    };
    std::thread reference([&] {
        app_log::set_thread_name("crop-reference");  // in the log, as the Python app names it
        detect(0, source, source_hwaccel);
    });
    detect(1, distorted, distorted_hwaccel);
    reference.join();
    for (const std::exception_ptr& error : errors) {
        if (error) {
            try {
                std::rethrow_exception(error);
            } catch (const Cancelled&) {
                throw;
            } catch (...) {
            }
        }
    }
    for (const std::exception_ptr& error : errors) {
        if (error) {
            std::rethrow_exception(error);
        }
    }
    return {*boxes[0], *boxes[1]};
}

std::pair<std::optional<CropBox>, std::optional<CropBox>> common_picture(const VideoInfo& source,
                                                                         const VideoInfo& distorted,
                                                                         std::optional<CropBox> source_crop,
                                                                         std::optional<CropBox> distorted_crop) {
    const auto kept = std::make_pair(source_crop, distorted_crop);
    if (!source_crop || !distorted_crop) {
        return kept;
    }
    const auto [source_width, source_height] = source.frame_size();
    const auto [distorted_width, distorted_height] = distorted.frame_size();
    if (std::min({source_width, source_height, distorted_width, distorted_height}) <= 0) {
        return kept;
    }
    // The same picture at one size or two: the sizes one factor apart,
    // across and down, to within half a percent.
    const double across = static_cast<double>(source_width) / distorted_width;
    const double down = static_cast<double>(source_height) / distorted_height;
    if (std::abs(across - down) > 0.005 * std::max(across, down)) {
        return kept;
    }
    // Each box's edges as fractions of its frame.
    struct Edges {
        double left, top, right, bottom;
    };
    const auto edges = [](const CropBox& box, const VideoInfo& info) {
        const auto [width, height] = info.frame_size();
        return Edges{static_cast<double>(box.x) / width, static_cast<double>(box.y) / height,
                     static_cast<double>(box.x + box.w) / width, static_cast<double>(box.y + box.h) / height};
    };
    const Edges s = edges(*source_crop, source);
    const Edges d = edges(*distorted_crop, distorted);
    const double left = std::max(s.left, d.left);
    const double top = std::max(s.top, d.top);
    const double right = std::min(s.right, d.right);
    const double bottom = std::min(s.bottom, d.bottom);
    if (right <= left || bottom <= top) {
        return kept;
    }
    // The shared picture in one video's pixels, rounded inward (never a row
    // of the other's bars), its edges on even samples (4:2:0).
    const auto shared = [&](const VideoInfo& info) {
        const auto [width, height] = info.frame_size();
        int x = static_cast<int>(std::ceil(left * width - 1e-6));
        int y = static_cast<int>(std::ceil(top * height - 1e-6));
        x += x & 1;
        y += y & 1;
        const int w = static_cast<int>(std::floor(right * width + 1e-6)) - x;
        const int h = static_cast<int>(std::floor(bottom * height + 1e-6)) - y;
        return CropBox{w - (w & 1), h - (h & 1), x, y};
    };
    const CropBox shared_source = shared(source);
    const CropBox shared_distorted = shared(distorted);
    if (shared_source.w <= 0 || shared_source.h <= 0 || shared_distorted.w <= 0 || shared_distorted.h <= 0) {
        return kept;
    }
    if (shared_source != *source_crop || shared_distorted != *distorted_crop) {
        logger.info("Black bars: the picture both videos show is " + box_text(shared_source) + " of the source and " +
                    box_text(shared_distorted) + " of the test video (detected: " + box_text(*source_crop) +
                    " and " + box_text(*distorted_crop) + ")");
    }
    return {shared_source, shared_distorted};
}

std::pair<std::optional<CropBox>, std::optional<CropBox>> crops_for(CropMode mode, const std::string& ffmpeg,
                                                                    const VideoInfo& source,
                                                                    const VideoInfo& distorted,
                                                                    const CancelToken& cancel,
                                                                    const std::optional<std::string>& source_hwaccel,
                                                                    const std::optional<std::string>& distorted_hwaccel,
                                                                    const std::string& scale_algorithm) {
    switch (mode) {
    case CropMode::none:
        return {};
    case CropMode::source_only:  // (the run's settings line says so: "black bars source")
        return {detect_crop(ffmpeg, source, cancel, source_hwaccel, scale_algorithm), std::nullopt};
    case CropMode::automatic:
        break;
    }
    const auto [source_box, distorted_box] =
        detect_pair(ffmpeg, source, distorted, cancel, source_hwaccel, distorted_hwaccel, scale_algorithm);
    return common_picture(source, distorted, source_box, distorted_box);
}

}  // namespace vml
