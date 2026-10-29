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
#include <regex>
#include <semaphore>
#include <set>
#include <sstream>
#include <string_view>
#include <system_error>
#include <thread>
#include <tuple>
#include <vector>

#include "vml/app_log.h"
#include "vml/ffmpeg_tools.h"
#include "vml/hardware_decoding.h"
#include "vml/number_format.h"
#include "vml/platform/process.h"
#include "vml/platform/system.h"

namespace vml {

namespace {

using namespace std::chrono_literals;

const app_log::Logger logger("vml.crop_detect");

// A box for the log: "3840x1608 at 0,276".
std::string box_text(const CropBox& box) {
    return std::to_string(box.w) + "x" + std::to_string(box.h) + " at " + std::to_string(box.x) + "," +
           std::to_string(box.y);
}

constexpr double sample_window_seconds = 3.0;
constexpr int sample_count = 5;
constexpr double sample_span_from = 0.1;  // of the duration
constexpr double sample_span_to = 0.9;
// cropdetect's limit: how bright a pixel may be and still count as black.
constexpr double black_limit = 24.0 / 255.0;
// How many detection decoders may run at once in the whole app: one video's
// windows run in turn, so this limits two jobs detecting both their videos
// to two (2.5 GB of GPU memory for 4K on the GPU decoder).
std::counting_semaphore<2> decoder_slots(2);
// How many files' answers to remember (a bound against an extreme session).
constexpr std::size_t remembered_answers = 512;
// Raised when the detection itself changes what it finds, so that older
// answers in the store are ignored.
constexpr int store_version = 1;

// What an answer depends on: which bytes are in the file, and which stretch
// of it is sampled (not how it was decoded: the GPU and the CPU give the
// same pixels). The Python app's key, so both apps read each other's store.
struct AnswerKey {
    std::string path;
    long long size = 0;
    long long modified_ns = 0;
    double sampled_duration = 0.0;
    double limit = 0.0;
    auto operator<=>(const AnswerKey&) const = default;
};

// Python's round(value, digits): the decimal nearest, read back.
double python_round(double value, int digits) { return std::strtod(format_fixed(value, digits).c_str(), nullptr); }

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
                     python_round(sampled_duration, 3), python_round(black_limit, 6)};
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

// How many seconds each sample reads: at most the part being compared.
double sample_window(double duration) {
    return duration <= 0 ? sample_window_seconds : std::min(sample_window_seconds, duration);
}

// The windows' start times, spread over the middle of the video, each
// leaving enough video for the whole window.
std::vector<double> sample_offsets(double duration, double window) {
    if (duration <= 0) {
        return {0.0};
    }
    const double last_start = std::max(0.0, duration - window);
    const double from = std::min(duration * sample_span_from, last_start);
    const double to = std::min(duration * sample_span_to, last_start);
    if (to <= from) {
        return {from};
    }
    const double step = (to - from) / (sample_count - 1);
    std::vector<double> starts;
    for (int i = 0; i < sample_count; ++i) {
        starts.push_back(from + i * step);
    }
    return starts;
}

// cropdetect, named `name` if one is given, for the frames the timeline
// expression `enable` gives (every frame without one).
std::string cropdetect(const std::string& name = "", const std::string& enable = "") {
    return "cropdetect" + (name.empty() ? "" : "@" + name) + "=limit=" + python_repr(black_limit) +
           ":round=2:reset=0" + (enable.empty() ? "" : ":enable=" + enable);
}

// The filters that take the frames off the GPU where they are decoded there
// (`hwaccel`), each followed by a comma; nothing otherwise. hwdownload gives
// the surface's own format, which cropdetect cannot take, so the run's own
// conversion follows it.
std::string download_filters(const std::optional<std::string>& hwaccel, const std::string& download_format) {
    return hwaccel ? "hwdownload,format=" + download_format + "," : "";
}

// cropdetect, after hwdownload where the frames are on the GPU (`hwaccel`).
std::string box_filters(const std::optional<std::string>& hwaccel, const std::string& download_format) {
    return download_filters(hwaccel, download_format) + cropdetect();
}

// The command that samples the window from `start` to `start` + `window`
// seconds (-ss and -t), `input` the video's input options (a raw YUV file's
// size and format, VideoInfo::input_options).
std::vector<std::string> window_command(const std::string& ffmpeg, const std::string& path,
                                        const std::vector<std::string>& input, double start, double window,
                                        const std::optional<std::string>& hwaccel,
                                        const std::string& download_format) {
    std::vector<std::string> command{ffmpeg, "-nostdin", "-hide_banner"};
    const std::vector<std::string> decoding = hwaccel_args(hwaccel);
    command.insert(command.end(), decoding.begin(), decoding.end());
    command.insert(command.end(), input.begin(), input.end());
    command.insert(command.end(), {"-ss", format_fixed(start, 3), "-i", path, "-map", "0:" + std::string(video_stream),
                                   "-t", format_fixed(window, 3), "-vf", box_filters(hwaccel, download_format), "-f",
                                   "null", "-"});
    return command;
}

// The frames of a raw stream (VideoInfo::timed_raw) in each window, starting
// at `starts`, `window` long: [first, end) for each, by the times a
// container's windows are given (-ss and -t, to the millisecond), a frame's
// time being its number over the frame rate.
std::vector<std::pair<long long, long long>> raw_window_frames(const VideoInfo& info, const std::vector<double>& starts,
                                                               double window) {
    const auto millisecond = [](double seconds) { return std::nearbyint(seconds * 1000) / 1000; };
    // The first frame at or after a time, against the rounding of the product.
    const auto frame_at = [&info](double seconds) {
        return static_cast<long long>(std::ceil(seconds * info.fps - 1e-6));
    };
    std::vector<std::pair<long long, long long>> frames;
    for (const double start : starts) {
        frames.emplace_back(frame_at(millisecond(start)), frame_at(millisecond(start) + millisecond(window)));
    }
    return frames;
}

// Whether a raw stream can be read from a byte partway through
// (run_byte_windows): its format marks where each picture starts (start
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

// The HEVC NAL unit types (H.265 table 7-1) an HEVC stream's windows are
// started by.
namespace hevc_unit {
constexpr int last_picture = 31;         // 0-31 hold pictures' slices
constexpr int first_random_access = 16;  // 16-23: BLA, IDR and CRA pictures
constexpr int last_random_access = 23;
constexpr int first_parameter_set = 32;  // 32-34: VPS, SPS and PPS
constexpr int last_parameter_set = 34;
}  // namespace hevc_unit

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
            const HevcUnit unit{buffer_at + at, (byte(3) >> 1) & 0x3F, ((byte(3) & 1) << 5) | (byte(4) >> 3),
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

// The command that samples a raw stream's window from `input` (the input
// options and -i), the first `frames` pictures decoded from it.
std::vector<std::string> byte_window_command(const std::string& ffmpeg, const std::vector<std::string>& input,
                                             long long frames, const std::optional<std::string>& hwaccel,
                                             const std::string& download_format) {
    std::vector<std::string> command{ffmpeg, "-nostdin", "-hide_banner"};
    const std::vector<std::string> decoding = hwaccel_args(hwaccel);
    command.insert(command.end(), decoding.begin(), decoding.end());
    command.insert(command.end(), input.begin(), input.end());
    command.insert(command.end(), {"-map", "0:" + std::string(video_stream), "-frames:v", std::to_string(frames), "-vf",
                                   box_filters(hwaccel, download_format), "-f", "null", "-"});
    return command;
}

// The command that samples a raw stream's windows (raw_window_frames, in
// order) in one pass from its start, for a stream that cannot be read from a
// byte partway (reads_from_a_byte), or whose windows found no picture from
// there (its only keyframe its first): select picks out the windows' frames,
// a cropdetect for each window (@w0, @w1...) takes its own by their number
// among those picked (windows overlap in a short video), and the pass stops
// after the last window's frames (trim).
std::vector<std::string> raw_windows_command(const std::string& ffmpeg, const VideoInfo& info,
                                             const std::vector<std::pair<long long, long long>>& windows,
                                             const std::optional<std::string>& hwaccel,
                                             const std::string& download_format) {
    long long end = 0;
    std::string picked;
    for (const auto& [first, after] : windows) {
        picked += (picked.empty() ? "" : "+") + std::string("between(n\\,") + std::to_string(first) + "\\," +
                  std::to_string(after - 1) + ")";
        end = std::max(end, after);
    }
    // How many of the frames picked come before frame `frame`.
    const auto picked_before = [&windows](long long frame) {
        long long count = 0;
        long long counted_to = 0;
        for (const auto& [first, after] : windows) {
            count += std::max(0LL, std::min(after, frame) - std::max(first, counted_to));
            counted_to = std::max(counted_to, after);
        }
        return count;
    };
    std::string filters = "trim=end_frame=" + std::to_string(end) + ",select=gt(" + picked + "\\,0)," +
                          download_filters(hwaccel, download_format);
    for (std::size_t i = 0; i < windows.size(); ++i) {
        filters += (i == 0 ? "" : ",") + cropdetect("w" + std::to_string(i),
                                                    "between(n\\," + std::to_string(picked_before(windows[i].first)) +
                                                        "\\," + std::to_string(picked_before(windows[i].second) - 1) +
                                                        ")");
    }
    std::vector<std::string> command{ffmpeg, "-nostdin", "-hide_banner", "-nostats"};
    const std::vector<std::string> decoding = hwaccel_args(hwaccel);
    command.insert(command.end(), decoding.begin(), decoding.end());
    const std::vector<std::string> input = info.input_options();
    command.insert(command.end(), input.begin(), input.end());
    command.insert(command.end(), {"-i", platform::path_text(info.path), "-map", "0:" + std::string(video_stream),
                                   "-vf", filters, "-f", "null", "-"});
    return command;
}

// What a detection command wrote: its exit code, the end of its error output
// (for a failure's message), and every line the detection reads, whole
// (cropdetect's boxes, and FFmpeg's word that it rebuilt its filters): a
// pass of several windows writes more than the end that is kept
// (error_tail_bytes), and would lose an early window's last box.
struct DetectionOutput {
    int code = 0;
    std::string tail;
    std::string lines;
};

// Runs one detection command to its end, holding a decoder slot, for at
// most `limit` of running (paused time not counted).
DetectionOutput launch_window(const std::vector<std::string>& command, const CancelToken& cancel,
                              std::chrono::seconds limit) {
    while (!decoder_slots.try_acquire_for(100ms)) {
        cancel.throw_if_cancelled();
    }
    struct SlotRelease {
        ~SlotRelease() { decoder_slots.release(); }
    } release_slot;
    struct Kept {
        std::mutex mutex;
        std::string lines;
    };
    const auto kept = std::make_shared<Kept>();
    platform::ProcessOptions options;
    options.on_error_line = [kept](std::string_view line) {
        if (line.find("crop=") != std::string_view::npos ||
            line.find("Reconfiguring filter graph") != std::string_view::npos) {
            const std::lock_guard lock(kept->mutex);
            kept->lines.append(line).push_back('\n');
        }
    };
    try {
        platform::ChildProcess window = platform::ChildProcess::start(command, options);
        const CancelToken::Attached attached = cancel.attach(window);
        auto running = std::chrono::steady_clock::duration::zero();  // paused time not counted
        while (!window.wait_for(200ms)) {
            if (cancel.cancelled()) {
                window.terminate();
            }
            if (!cancel.paused()) {
                running += 200ms;
            }
            if (running >= limit) {
                window.terminate();
                window.wait();
                throw CropDetectError("Crop detection timed out");
            }
        }
        DetectionOutput output;
        output.code = window.wait();
        cancel.throw_if_cancelled();
        output.tail = window.standard_error();  // waits for the error output's end, and so its lines
        const std::lock_guard lock(kept->mutex);
        output.lines = std::move(kept->lines);
        return output;
    } catch (const platform::ProcessError& error) {
        throw CropDetectError(std::string("Could not run crop detection: ") + error.what());
    }
}

// Runs the detection command `command` makes for a decoder, the GPU's
// (`hwaccel`) and else software, for at most `limit` each, and returns the
// lines of FFmpeg's error output the detection reads (DetectionOutput),
// where cropdetect writes its boxes. Throws CropDetectError when it fails.
std::string run_detection(const std::function<std::vector<std::string>(const std::optional<std::string>&)>& command,
                          const std::string& path, const CancelToken& cancel,
                          const std::optional<std::string>& hwaccel, std::chrono::seconds limit) {
    cancel.throw_if_cancelled();
    std::vector<std::optional<std::string>> attempts{hwaccel};
    if (hwaccel) {
        attempts.emplace_back(std::nullopt);
    }
    DetectionOutput output;
    for (const std::optional<std::string>& attempt : attempts) {
        try {
            output = launch_window(command(attempt), cancel, limit);
        } catch (const CropDetectError& error) {
            throw CropDetectError(std::string(error.what()) + " (" + path + ")");
        }
        // A hardware attempt that fails is tried again in software; the
        // pixels, and so the box, are the same.
        if (output.code == 0) {
            break;
        }
    }
    if (output.code != 0) {
        // The last line of FFmpeg's error output (the output as a whole
        // stripped of spaces at both ends first, as Python strips it).
        const std::string& errors = output.tail;
        std::string tail = "ffmpeg exited with code " + platform::exit_code_text(output.code);
        const std::size_t first = errors.find_first_not_of(" \t\r\n\f\v");
        if (first != std::string::npos) {
            const std::string stripped = errors.substr(first, errors.find_last_not_of(" \t\r\n\f\v") - first + 1);
            const std::size_t line_break = stripped.find_last_of("\r\n");
            tail = line_break == std::string::npos ? stripped : stripped.substr(line_break + 1);
        }
        throw CropDetectError("Crop detection failed for " + path + ": " + tail);
    }
    return output.lines;
}

// A box cropdetect wrote, from its "crop=w:h:x:y".
CropBox box_from(const std::smatch& match, std::size_t first_group) {
    return CropBox{std::stoi(match[first_group].str()), std::stoi(match[first_group + 1].str()),
                   std::stoi(match[first_group + 2].str()), std::stoi(match[first_group + 3].str())};
}

// The last box a window's cropdetect wrote in FFmpeg's error output: the
// one from all its frames.
std::optional<CropBox> last_box(const std::string& errors) {
    static const std::regex crop(R"(crop=(\d+):(\d+):(\d+):(\d+))");
    std::optional<CropBox> box;
    for (auto match = std::sregex_iterator(errors.begin(), errors.end(), crop); match != std::sregex_iterator();
         ++match) {
        box = box_from(*match, 1);
    }
    return box;
}

// Each window's last box in a pass of several (cropdetect@w0, @w1...), as each
// window's own cropdetect wrote it; nothing for a window that wrote none.
std::vector<std::optional<CropBox>> boxes_by_window(const std::string& errors, std::size_t windows) {
    static const std::regex crop(R"(\[cropdetect@w(\d+) @[^\]]*\][^\n]*?crop=(\d+):(\d+):(\d+):(\d+))");
    std::vector<std::optional<CropBox>> boxes(windows);
    for (auto match = std::sregex_iterator(errors.begin(), errors.end(), crop); match != std::sregex_iterator();
         ++match) {
        const std::size_t index = std::stoul((*match)[1].str());
        if (index < boxes.size()) {
            boxes[index] = box_from(*match, 2);
        }
    }
    return boxes;
}

// Runs one window, on the GPU decoder and else in software, and returns the
// box it found, or nothing.
std::optional<CropBox> run_single_window(const std::string& ffmpeg, const VideoInfo& info, double start,
                                         double window, const CancelToken& cancel,
                                         const std::optional<std::string>& hwaccel,
                                         const std::string& download_format) {
    const std::string path = platform::path_text(info.path);
    return last_box(run_detection(
        [&](const std::optional<std::string>& attempt) {
            return window_command(ffmpeg, path, info.input_options(), start, window, attempt, download_format);
        },
        path, cancel, hwaccel, 60s));
}

// The boxes a raw stream's windows found, each started from a byte partway
// through, and why any window failed: FFmpeg cannot seek in a stream without
// timestamps (-ss lands on no picture), so each window starts at the share
// of the file its time is of the video, and is its first frames decoded from
// there. A window that decoded no picture (none after its byte, past the
// stream's last keyframe) finds no box.
std::pair<std::vector<CropBox>, std::vector<std::string>> run_byte_windows(
    const std::string& ffmpeg, const VideoInfo& info, std::uintmax_t file_bytes, const std::vector<double>& starts,
    double window, const CancelToken& cancel, const std::optional<std::string>& hwaccel,
    const std::string& download_format) {
    const std::string path = platform::path_text(info.path);
    const long long frames = std::max(1LL, static_cast<long long>(std::llround(window * info.fps)));
    // An HEVC stream's windows start at a random access picture, after the
    // parameter sets from before its first picture: x265 writes them only
    // there unless told to repeat them (--repeat-headers), and a window
    // without them decodes nothing. ('|' in the path would split the input.)
    const bool hevc = info.format_name == "hevc";
    std::vector<ByteRange> parameter_sets;
    if (hevc) {
        parameter_sets = hevc_parameter_sets(info.path, cancel);
        if (parameter_sets.empty() || path.find('|') != std::string::npos) {
            return {};
        }
    }
    std::vector<CropBox> boxes;
    std::vector<std::string> failures;
    for (const double start : starts) {
        const auto offset = static_cast<std::uintmax_t>(static_cast<double>(file_bytes) * start / info.duration);
        std::vector<std::string> input = info.input_options();
        if (hevc) {
            const std::optional<std::uintmax_t> access = hevc_random_access(info.path, offset, cancel);
            if (!access) {
                break;  // none after this window's byte, so none after a later one's
            }
            std::vector<ByteRange> parts = parameter_sets;
            parts.emplace_back(*access, 0);
            input.insert(input.begin(), {"-f", "hevc"});
            input.insert(input.end(), {"-i", pieced_input(path, parts)});
        } else {
            input.insert(input.end(), {"-skip_initial_bytes", std::to_string(offset), "-i", path});
        }
        try {
            if (const std::optional<CropBox> box = last_box(run_detection(
                    [&](const std::optional<std::string>& attempt) {
                        return byte_window_command(ffmpeg, input, frames, attempt, download_format);
                    },
                    path, cancel, hwaccel, 60s))) {
                boxes.push_back(*box);
            }
        } catch (const CropDetectError& error) {
            failures.emplace_back(error.what());
        }
    }
    return {boxes, failures};
}

// Each window's box of a raw stream (raw_windows_command), or nothing for a
// window that found none.
std::vector<std::optional<CropBox>> run_raw_windows(const std::string& ffmpeg, const VideoInfo& info,
                                                    const std::vector<double>& starts, double window,
                                                    const CancelToken& cancel,
                                                    const std::optional<std::string>& hwaccel,
                                                    const std::string& download_format) {
    const std::vector<std::pair<long long, long long>> windows = raw_window_frames(info, starts, window);
    long long frames = 0;
    for (const auto& [first, end] : windows) {
        frames = std::max(frames, end);
    }
    // A window's minute, and a tenth of a second for each frame the pass
    // decodes: a bound against a pass that hangs, which a slow decoder (4K
    // VVC in software) stays well inside.
    const std::chrono::seconds limit = 60s + std::chrono::seconds(frames / 10);
    const std::string path = platform::path_text(info.path);
    return boxes_by_window(run_detection(
                               [&](const std::optional<std::string>& attempt) {
                                   return raw_windows_command(ffmpeg, info, windows, attempt, download_format);
                               },
                               path, cancel, hwaccel, limit),
                           windows.size());
}

// --------------------------------------------- overlapping windows, one pass

// Runs of windows that overlap or touch, as indexes into `starts` (in order):
// each run's frames can be decoded once for all its windows.
std::vector<std::vector<std::size_t>> overlapping_runs(const std::vector<double>& starts, double window) {
    std::vector<std::vector<std::size_t>> runs;
    for (std::size_t i = 0; i < starts.size(); ++i) {
        if (!runs.empty() && starts[i] <= starts[runs.back().back()] + window) {
            runs.back().push_back(i);
        } else {
            runs.push_back({i});
        }
    }
    return runs;
}

// Whether one pass gives each window exactly the frames its own pass
// (window_command) gets: where FFmpeg's seek lands on a picture shown at or
// before the window's start. Matroska and WebM seek by presentation time,
// and an intra-only codec has a picture at every frame, in MP4/MOV or a raw
// file. MP4 and MOV seek by decode time (an open-GOP picture after the start
// can be where a window's own pass begins, missing frames one pass has), and
// other formats' timestamp corrections (MPEG-TS, AVI) differ in a pass that
// keeps the file's timestamps.
bool merges_windows(const VideoInfo& info) {
    static const std::set<std::string, std::less<>> intra_only{"ffv1", "prores", "dnxhd", "mjpeg", "rawvideo"};
    const bool mov = info.format_name.starts_with("mov,mp4");
    return info.format_name.find("matroska") != std::string::npos ||
           ((mov || info.format_name == "rawvideo") && intra_only.contains(info.codec_name));
}

// The file's start time as FFmpeg gives it on its "  Duration: ..., start:
// 0.000000" line (whole microseconds), from which its seek (-ss) counts: 0
// for a file that has none (FFmpeg's seek then counts from 0, and it writes
// no "start:"); nothing if FFmpeg could not read the file.
std::optional<long long> file_start_microseconds(const std::string& ffmpeg, const VideoInfo& info,
                                                 const CancelToken& cancel) {
    std::vector<std::string> command{ffmpeg, "-nostdin", "-hide_banner"};
    const std::vector<std::string> input = info.input_options();
    command.insert(command.end(), input.begin(), input.end());
    command.insert(command.end(), {"-i", platform::path_text(info.path)});
    // The input's own line, two spaces in, kept as it comes: a metadata
    // value naming a duration is further in, and a file with many streams or
    // chapters writes more than the end of its error output that is kept.
    struct Found {
        std::mutex mutex;
        std::optional<std::string> line;
    };
    const auto found = std::make_shared<Found>();
    platform::ProcessOptions options;
    options.on_error_line = [found](std::string_view line) {
        if (line.starts_with("  Duration: ")) {
            const std::lock_guard lock(found->mutex);
            if (!found->line) {
                found->line = std::string(line);
            }
        }
    };
    try {
        // No output named: it says what it read, and ends.
        platform::ChildProcess probe = platform::ChildProcess::start(command, options);
        const CancelToken::Attached attached = cancel.attach(probe);
        if (!probe.wait_for(30s)) {
            probe.terminate();
            probe.wait();
            return std::nullopt;
        }
        cancel.throw_if_cancelled();
        static_cast<void>(probe.standard_error());  // waits for the error output's end, and so its lines
    } catch (const platform::ProcessError&) {
        return std::nullopt;
    }
    const std::lock_guard lock(found->mutex);
    if (!found->line) {
        return std::nullopt;
    }
    static const std::regex start(R"(, start: (-?)(\d+)\.(\d{6}))");
    std::smatch match;
    if (!std::regex_search(*found->line, match, start)) {
        return 0;
    }
    const long long magnitude = std::stoll(match[2].str()) * 1'000'000 + std::stoll(match[3].str());
    return match[1].length() > 0 ? -magnitude : magnitude;
}

// Microseconds as seconds with six decimals, from whole numbers (no
// rounding of a double between).
std::string seconds_text(long long microseconds) {
    const long long whole = microseconds / 1'000'000;
    const long long part = microseconds % 1'000'000;
    std::string digits = std::to_string(part);
    return std::to_string(whole) + "." + std::string(6 - digits.size(), '0') + digits;
}

// One pass for the windows `run` (indexes into `starts`), each window seeing
// exactly the frames its own pass (window_command) sees. The pass keeps the
// file's timestamps (-copyts) from the first window's seek; each window's
// branch trims from its own start (the file's start time `file_start` added,
// as -ss counts from it), then its cropdetect, then a trim to the window's
// length, so that its cropdetect also sees the frame past the end that its
// own pass's -t drops after it. Each branch is an output of its own, so the
// pass ends when the last window does.
std::vector<std::string> merged_windows_command(const std::string& ffmpeg, const VideoInfo& info,
                                                const std::vector<double>& starts,
                                                const std::vector<std::size_t>& run, double window,
                                                long long file_start, const std::optional<std::string>& hwaccel,
                                                const std::string& download_format) {
    std::string graph = "[0:" + std::string(video_stream) + "]" + download_filters(hwaccel, download_format) +
                        "split=" + std::to_string(run.size());
    for (std::size_t i = 0; i < run.size(); ++i) {
        graph += "[s" + std::to_string(i) + "]";
    }
    for (std::size_t i = 0; i < run.size(); ++i) {
        // The window's start in whole milliseconds, as its own pass's -ss.
        std::string start = format_fixed(starts[run[i]], 3);
        start.erase(std::remove(start.begin(), start.end(), '.'), start.end());
        graph += ";[s" + std::to_string(i) + "]trim=start=" + seconds_text(std::stoll(start) * 1000 + file_start) +
                 "," + cropdetect("w" + std::to_string(i)) + ",trim=duration=" + format_fixed(window, 3) + "[o" +
                 std::to_string(i) + "]";
    }
    std::vector<std::string> command{ffmpeg, "-nostdin", "-hide_banner", "-nostats"};
    const std::vector<std::string> decoding = hwaccel_args(hwaccel);
    command.insert(command.end(), decoding.begin(), decoding.end());
    const std::vector<std::string> input = info.input_options();
    command.insert(command.end(), input.begin(), input.end());
    command.insert(command.end(), {"-copyts", "-ss", format_fixed(starts[run.front()], 3), "-i",
                                   platform::path_text(info.path), "-filter_complex", graph});
    for (std::size_t i = 0; i < run.size(); ++i) {
        command.insert(command.end(), {"-map", "[o" + std::to_string(i) + "]", "-f", "null", "-"});
    }
    return command;
}

// Each window's box of `run` from one pass (merged_windows_command), or
// nothing where the pass cannot stand for the windows' own passes: it failed
// or timed out, or FFmpeg rebuilt its filters partway (the video changed size
// or format), which starts the windows' cropdetects again.
std::optional<std::vector<std::optional<CropBox>>> run_merged_windows(
    const std::string& ffmpeg, const VideoInfo& info, const std::vector<double>& starts,
    const std::vector<std::size_t>& run, double window, long long file_start, const CancelToken& cancel,
    const std::optional<std::string>& hwaccel, const std::string& download_format) {
    const std::string path = platform::path_text(info.path);
    std::string errors;
    try {
        errors = run_detection(
            [&](const std::optional<std::string>& attempt) {
                return merged_windows_command(ffmpeg, info, starts, run, window, file_start, attempt,
                                              download_format);
            },
            path, cancel, hwaccel, 60s * static_cast<long long>(run.size()));
    } catch (const CropDetectError& error) {
        logger.info("Black bars in " + platform::path_text(info.path.filename()) +
                    ": windows sampled one by one, as one pass for them failed (" + error.what() + ")");
        return std::nullopt;
    }
    if (errors.find("Reconfiguring filter graph") != std::string::npos) {
        logger.info("Black bars in " + platform::path_text(info.path.filename()) +
                    ": windows sampled one by one, as the video changes size or format");
        return std::nullopt;
    }
    return boxes_by_window(errors, run.size());
}

// detect_crop's work, without the kept answers.
CropBox detect_uncached(const std::string& ffmpeg, const VideoInfo& info, double sampled_duration,
                        const CancelToken& cancel, const std::optional<std::string>& hwaccel) {
    const std::string path = platform::path_text(info.path);
    const double window = sample_window(sampled_duration);
    const std::string download_format = hw_native_format(info.pix_fmt);
    const std::vector<double> starts = sample_offsets(sampled_duration, window);
    cancel.throw_if_cancelled();
    std::vector<CropBox> boxes;
    std::vector<std::string> failures;
    // A raw YUV file's pictures are all one size, so FFmpeg's -ss finds a
    // window's first frame in it exactly, as in a container.
    if (info.timed_raw && info.format_name != "rawvideo") {
        // Each window from a byte partway through where the stream allows
        // it: a pass from the start reaches the last window only after
        // decoding nine tenths of the video (for a 4K VVC film, still not
        // there after 21 minutes).
        std::error_code unknown;
        const std::uintmax_t file_bytes = std::filesystem::file_size(info.path, unknown);
        if (reads_from_a_byte(info) && !unknown && info.duration > 0) {
            std::tie(boxes, failures) =
                run_byte_windows(ffmpeg, info, file_bytes, starts, window, cancel, hwaccel, download_format);
        }
        if (boxes.empty()) {
            // No picture from any window started partway, or a stream that
            // cannot be: every window again, in one pass from the start.
            logger.info("Black bars in " + platform::path_text(info.path.filename()) +
                        ": sampling in one pass from the start, as no window could start partway through");
            failures.clear();
            try {
                for (const std::optional<CropBox>& box :
                     run_raw_windows(ffmpeg, info, starts, window, cancel, hwaccel, download_format)) {
                    if (box) {
                        boxes.push_back(*box);
                    }
                }
            } catch (const CropDetectError& error) {
                failures.emplace_back(error.what());
            }
        }
    } else {
        // One decoder at a time: five at once held 6.2 GB of GPU memory for a
        // 4K source. Windows that overlap (a video under 17 s) are decoded in
        // one pass where it sees each window's frames exactly, each frame
        // once (a 6.4 s 4K video: 382 frames, not some 1240); the others,
        // and those of a pass that failed, one by one.
        std::vector<std::optional<CropBox>> found(starts.size());
        std::vector<bool> sampled(starts.size(), false);
        const std::vector<std::vector<std::size_t>> runs = overlapping_runs(starts, window);
        const bool shared = std::any_of(runs.begin(), runs.end(), [](const auto& run) { return run.size() > 1; });
        const std::optional<long long> file_start =
            shared && merges_windows(info) ? file_start_microseconds(ffmpeg, info, cancel) : std::nullopt;
        if (file_start && *file_start >= 0) {
            for (const std::vector<std::size_t>& run : runs) {
                if (run.size() < 2) {
                    continue;
                }
                if (const auto run_boxes = run_merged_windows(ffmpeg, info, starts, run, window, *file_start, cancel,
                                                              hwaccel, download_format)) {
                    for (std::size_t i = 0; i < run.size(); ++i) {
                        found[run[i]] = (*run_boxes)[i];
                        sampled[run[i]] = true;
                    }
                }
            }
        }
        for (std::size_t i = 0; i < starts.size(); ++i) {
            if (sampled[i]) {
                continue;
            }
            try {
                found[i] = run_single_window(ffmpeg, info, starts[i], window, cancel, hwaccel, download_format);
            } catch (const CropDetectError& error) {
                failures.emplace_back(error.what());
            }
        }
        // In the windows' order: between equally good boxes, the first wins.
        for (const std::optional<CropBox>& box : found) {
            if (box) {
                boxes.push_back(*box);
            }
        }
    }
    cancel.throw_if_cancelled();  // a partial vote is not an answer
    if (boxes.empty() && failures.empty()) {
        throw CropDetectError("Could not auto-detect black bars in " + path +
                              ": FFmpeg found no pictures in the parts of it that were sampled. The file may be "
                              "damaged or cut short.");
    }
    if (boxes.empty()) {
        throw CropDetectError("Could not auto-detect black bars in " + path + ": " + failures.back() +
                              ". Choose 'None (use full frame)' for this video to continue without cropping.");
    }
    // The most common box wins, and between equally common ones the largest:
    // a box from a dark stretch is too tight.
    std::vector<std::pair<CropBox, int>> counts;
    for (const CropBox& box : boxes) {
        const auto found = std::find_if(counts.begin(), counts.end(), [&box](const auto& item) { return item.first == box; });
        if (found == counts.end()) {
            counts.emplace_back(box, 1);
        } else {
            ++found->second;
        }
    }
    const auto better = [](const auto& a, const auto& b) {
        const long long area_a = static_cast<long long>(a.first.w) * a.first.h;
        const long long area_b = static_cast<long long>(b.first.w) * b.first.h;
        return std::tie(a.second, area_a) < std::tie(b.second, area_b);
    };
    // The first of equally good boxes, as Python's max() takes it.
    auto best = counts.begin();
    for (auto item = counts.begin(); item != counts.end(); ++item) {
        if (better(*best, *item)) {
            best = item;
        }
    }
    return best->first;
}

}  // namespace

void set_crop_store(std::function<std::filesystem::path()> store) { answers().set_store(std::move(store)); }

void forget_crop_answers() { answers().forget(); }

CropBox detect_crop(const std::string& ffmpeg, const VideoInfo& info, const CancelToken& cancel,
                    const std::optional<std::string>& hwaccel) {
    const double sampled_duration = info.duration;  // the whole video: the runs give no limit
    const std::optional<AnswerKey> key = answer_key(info.path, sampled_duration);
    if (key) {
        if (const std::optional<CropBox> kept = answers().claim(*key, cancel)) {
            return *kept;
        }
    }
    std::optional<CropBox> box;
    try {
        box = detect_uncached(ffmpeg, info, sampled_duration, cancel, hwaccel);
        logger.info("Black bars in " + platform::path_text(info.path.filename()) + ": picture " + box_text(*box) +
                    " of " + std::to_string(info.width) + "x" + std::to_string(info.height));
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
        answers().settle(*key, box);
    }
    return *box;
}

std::pair<CropBox, CropBox> detect_pair(const std::string& ffmpeg, const VideoInfo& source,
                                        const VideoInfo& distorted, const CancelToken& cancel,
                                        const std::optional<std::string>& source_hwaccel,
                                        const std::optional<std::string>& distorted_hwaccel) {
    // Each video's windows run in turn, so this is two decoders, in about
    // half the time of one video after the other. Both are waited for before
    // anything is thrown: a cancel first, then the source's error.
    std::optional<CropBox> boxes[2];
    std::exception_ptr errors[2];
    const auto detect = [&](int index, const VideoInfo& info, const std::optional<std::string>& hwaccel) {
        try {
            boxes[index] = detect_crop(ffmpeg, info, cancel, hwaccel);
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
    if (std::min({source.width, source.height, distorted.width, distorted.height}) <= 0) {
        return kept;
    }
    // The same picture at one size or two: the sizes one factor apart,
    // across and down, to within half a percent.
    const double across = static_cast<double>(source.width) / distorted.width;
    const double down = static_cast<double>(source.height) / distorted.height;
    if (std::abs(across - down) > 0.005 * std::max(across, down)) {
        return kept;
    }
    // Each box's edges as fractions of its frame.
    struct Edges {
        double left, top, right, bottom;
    };
    const auto edges = [](const CropBox& box, const VideoInfo& info) {
        return Edges{static_cast<double>(box.x) / info.width, static_cast<double>(box.y) / info.height,
                     static_cast<double>(box.x + box.w) / info.width, static_cast<double>(box.y + box.h) / info.height};
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
        int x = static_cast<int>(std::ceil(left * info.width - 1e-6));
        int y = static_cast<int>(std::ceil(top * info.height - 1e-6));
        x += x & 1;
        y += y & 1;
        const int w = static_cast<int>(std::floor(right * info.width + 1e-6)) - x;
        const int h = static_cast<int>(std::floor(bottom * info.height + 1e-6)) - y;
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

}  // namespace vml
