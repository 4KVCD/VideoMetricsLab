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
#include <numeric>
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
#include "vml/gpu_decoders.h"
#include "vml/gpu_frames.h"
#include "vml/grain_feed.h"
#include "vml/hardware_decoding.h"
#include "vml/hevc.h"
#include "vml/number_format.h"
#include "vml/platform/process.h"
#include "vml/platform/system.h"
#include "vml/python_json.h"
#include "vml/python_text.h"
#include "vml/size_changes.h"
#include "vml/time_format.h"

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
// series; 4: HandBrake's rule in place of FFmpeg's cropdetect; 5: an odd-sized
// video's picture is even-sized; 6: odd bars rounded up to even and the
// thinner bars counted edge by edge, as HandBrake does.
constexpr int store_version = 6;

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
std::optional<AnswerKey> answer_key(const VideoInfo& info, double sampled_duration) {
    const std::filesystem::path& path = info.path;
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
    std::string name = platform::path_text(resolved);
    if (info.format_name == "rawvideo") {  // the same file read at another --size has other pictures
        name += "|" + std::to_string(info.width) + "x" + std::to_string(info.height);
    }
    return AnswerKey{std::move(name), static_cast<long long>(size), *written,
                     python_round(sampled_duration, 3), python_round(dark_limit, 6)};
}

// Bars as the store writes them: [top, bottom, left, right].
nlohmann::ordered_json edges_json(const BarEdges& edges) {
    return nlohmann::ordered_json::array({edges.top, edges.bottom, edges.left, edges.right});
}

BarEdges edges_from(const nlohmann::json& text) {
    return {text.at(0).get<int>(), text.at(1).get<int>(), text.at(2).get<int>(), text.at(3).get<int>()};
}

// What an answer was decided from, as the store keeps it beside the box.
nlohmann::ordered_json findings_json(const CropAnswer& answer) {
    const BarFindings& found = *answer.findings;
    return {{"decoded", found.decoded},       {"judged", found.judged},
            {"median", edges_json(found.median)}, {"taken", edges_json(found.taken)},
            {"shape_changes", found.shape_changes}, {"too_few", found.too_few},
            {"found_utc", answer.found_utc}, {"seconds", answer.seconds}};
}

// The findings beside a stored box; left out where they are of another shape.
void read_findings(const nlohmann::json& text, CropAnswer& answer) {
    try {
        BarFindings found;
        found.decoded = text.at("decoded").get<std::size_t>();
        found.judged = text.at("judged").get<std::size_t>();
        found.median = edges_from(text.at("median"));
        found.taken = edges_from(text.at("taken"));
        found.shape_changes = text.at("shape_changes").get<bool>();
        found.too_few = text.at("too_few").get<bool>();
        answer.found_utc = text.at("found_utc").get<std::string>();
        answer.seconds = text.at("seconds").get<double>();
        answer.findings = found;
    } catch (const nlohmann::json::exception&) {
        answer.found_utc.clear();
    }
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
                return found->second.box;
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

    // The answer the store keeps for `key` now: read again, for answers a
    // process of its own wrote since.
    std::optional<CropAnswer> kept(const AnswerKey& key) {
        const std::lock_guard lock(mutex_);
        store_read_.reset();
        read_store();
        const auto found = find(key);
        return found != boxes_.end() ? std::optional(found->second) : std::nullopt;
    }

    // Keeps a detected answer (if any), and wakes the callers waiting for it.
    void settle(const AnswerKey& key, const std::optional<CropAnswer>& answer) {
        {
            const std::lock_guard lock(mutex_);
            if (answer) {
                if (const auto found = find(key); found != boxes_.end()) {
                    boxes_.erase(found);
                }
                boxes_.emplace_back(key, *answer);
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
    std::list<std::pair<AnswerKey, CropAnswer>>::iterator find(const AnswerKey& key) {
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
            data = nlohmann::json::parse(python_json_readable(text.str()));
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
                    CropAnswer kept;
                    kept.box = CropBox{box.at(0).get<int>(), box.at(1).get<int>(), box.at(2).get<int>(),
                                       box.at(3).get<int>()};
                    if (item.size() > 2) {  // what it was decided from: not kept by builds before these records
                        read_findings(item.at(2), kept);
                    }
                    boxes_.emplace_back(answer, std::move(kept));
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
        for (const auto& [key, answer] : boxes_) {
            const CropBox& box = answer.box;
            nlohmann::ordered_json item{{key.path, key.size, key.modified_ns, key.sampled_duration, key.limit},
                                        {box.w, box.h, box.x, box.y}};
            if (answer.findings) {
                item.push_back(findings_json(answer));
            }
            boxes.push_back(std::move(item));
        }
        // The keys in the Python app's order (crop_detect._write_store).
        const nlohmann::ordered_json data{{"version", store_version}, {"boxes", boxes}};
        std::filesystem::path partial = *path;
        partial += ".partial";
        {
            std::ofstream file(partial, std::ios::binary);
            file << data.dump(-1, ' ', false, nlohmann::ordered_json::error_handler_t::replace);  // Linux's names
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
    std::list<std::pair<AnswerKey, CropAnswer>> boxes_;
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
    command.insert(command.end(), {"-ss", format_fixed(start, 3), as_stored, "-i", path, "-map", "0:" + std::string(video_stream),
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
    command.insert(command.end(), {as_stored, "-i", platform::path_text(info.path), "-map", "0:" + std::string(video_stream),
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
        throw VideoSizeChanged(platform::path_from_text(path), rebuilt_size(output.lines));
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
            input.insert(input.end(), {as_stored, "-i", pieced_input(path, parts)});
        } else {
            input.insert(input.end(), {"-skip_initial_bytes", std::to_string(offset), as_stored, "-i", path});
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

// The decoder of a video's previews. After -ss, FFmpeg's QSV decoders give
// pictures from before it (285K, driver 32.0.101.8860, FFmpeg 9.0.1: usually
// 3 to 5 frames early, though a whole decode is exact), so another GPU would
// judge other pictures and could find another box. FFmpeg's D3D11VA gives the
// right ones, also where its default adapter is another GPU's. A codec it does
// not decode (VP8) goes to software.
std::optional<std::string> preview_decoder(const std::optional<std::string>& gpu_decoder, const VideoInfo& info) {
    if (gpu_decoder != "qsv") {
        return gpu_decoder;
    }
    const GpuDecoder* d3d11va = decoder_of_hwaccel("d3d11va");
    const std::string codec = ascii_lower(info.codec_name);
    const bool decodes =
        d3d11va != nullptr && std::ranges::find(d3d11va->hwaccel_codecs, codec) != d3d11va->hwaccel_codecs.end();
    return decodes ? std::optional<std::string>("d3d11va") : std::nullopt;
}

// detect_crop's work, without the kept answers: HandBrake's previews
// (black_bars_rule.h), each a decoder's run of its own, one after another.
CropBox detect_uncached(BarFindings& findings, const std::string& ffmpeg, const VideoInfo& info,
                        double sampled_duration,
                        const CancelToken& cancel, const std::optional<std::string>& gpu_decoder,
                        const std::string& scale_algorithm) {
    const std::string path = platform::path_text(info.path);
    const BeforeSampling before{hw_native_format(info.pix_fmt), size_change_scale(info, scale_algorithm)};
    const std::vector<double> times = preview_times(sampled_duration);
    std::optional<std::string> hwaccel = preview_decoder(gpu_decoder, info);  // dropped for the rest once it fails
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
    return handbrake_box(counted, decoded, width, height, &findings);
}

// The largest q of the sizes' ratio p/q for which common_picture lines the
// boxes up on one grid: the test video's edges then every 2 x 16 samples at
// most. A coarser grid could cut too much picture from each edge.
constexpr int largest_grid_denominator = 16;

// The shared picture (edges as fractions of the frame) as one box in each
// video, the source's exactly the test video's scaled by the sizes' ratio,
// when that ratio is one fraction p/q across and down (source/test, in lowest
// terms, q at most largest_grid_denominator): the edges rounded inward to the test
// video's multiples of 2q, so that both boxes' edges fall on even samples
// (4:2:0) and the test box scaled to the source's size covers its picture
// row for row. Each box rounded on its own may be a row apart once scaled
// (3840x2160 against 2560x1440: rows 68 and 46, which is 69). Nothing for other
// ratios.
std::optional<std::pair<CropBox, CropBox>> shared_on_one_grid(int source_width, int source_height, int test_width,
                                                              int test_height, double left, double top, double right,
                                                              double bottom) {
    if (static_cast<long long>(source_width) * test_height != static_cast<long long>(source_height) * test_width) {
        return std::nullopt;
    }
    const int common = std::gcd(source_width, test_width);
    const int p = source_width / common;
    const int q = test_width / common;
    if (q > largest_grid_denominator) {
        return std::nullopt;
    }
    const int step = 2 * q;
    const auto inward_up = [step](double fraction, int size) {
        return static_cast<int>(std::ceil(fraction * size / step - 1e-6)) * step;
    };
    const auto inward_down = [step](double fraction, int size) {
        return static_cast<int>(std::floor(fraction * size / step + 1e-6)) * step;
    };
    const int x = inward_up(left, test_width);
    const int y = inward_up(top, test_height);
    const CropBox test{inward_down(right, test_width) - x, inward_down(bottom, test_height) - y, x, y};
    const auto scaled = [p, q](int samples) { return samples / q * p; };  // exact: samples are multiples of 2q
    return std::pair(CropBox{scaled(test.w), scaled(test.h), scaled(test.x), scaled(test.y)}, test);
}

// Whether the test video is the source cut when it was encoded (HandBrake's
// crop, FFmpeg's crop filter), not scaled: a smaller frame whose box is the
// source's box at the same size, at the place the lines cut put it. Only
// lines outside the source's picture were cut, so each box is the same
// picture as detected. Its frame can be a fraction of a percent from the
// source's in shape (1920x1080 cut to 1920x1078), which common_picture would
// otherwise read as the source scaled, cutting rows off the test (issue #18).
// Frames of one size are not: there the boxes' small differences are
// detection's, and the picture both show is cut from both.
bool cropped_when_encoded(int source_width, int source_height, int test_width, int test_height,
                          const CropBox& source_box, const CropBox& test_box) {
    const bool smaller = test_width <= source_width && test_height <= source_height &&
                         (test_width < source_width || test_height < source_height);
    if (!smaller || source_box.w != test_box.w || source_box.h != test_box.h) {
        return false;
    }
    // The lines cut from the source's left and top: at most what was cut in all
    const int cut_left = source_box.x - test_box.x;
    const int cut_top = source_box.y - test_box.y;
    return cut_left >= 0 && cut_left <= source_width - test_width && cut_top >= 0 &&
           cut_top <= source_height - test_height;
}

// How far, in the test video's samples, its frame is from a picture of
// `width` x `height` scaled to it: the least, over every scale factor k, of
// the larger of |test_width - width k| and |test_height - height k| (which
// is where the two are equal). A picture scaled with its size rounded to even
// samples, each side off by at most one, is at most 1.
double shape_error(double width, double height, int test_width, int test_height) {
    return std::abs(test_width * height - test_height * width) / (width + height);
}

// The source's box grown by the thin edges an encode cut to the bars kept,
// where the test's frame is that box at the same size (issue #20: a 1-column
// dark edge at the source's right, found on most previews and so cut 2
// columns wide, which the encoder did not cut: a 1920x816 encode beside a
// 1918x816 box). On each axis the test's frame is the box's size, or larger
// by at most 2 samples on each side the box grows by, placed one way only:
// the test as wide as the source's frame, or the box at one edge of it so
// that it grows to the other. Nothing where no axis grows, or one could grow
// either way (a box between bars on both sides).
std::optional<CropBox> box_grown_to_the_test(int source_width, int source_height, int test_width, int test_height,
                                             const CropBox& source_box) {
    constexpr int thin_edge = 2;  // a 1-sample dark edge, rounded up to even
    struct Span {
        int start, size;
    };
    // One axis: the box's span grown to `test_size`, or nothing.
    const auto grown = [](int frame_size, int test_size, int start, int size) -> std::optional<Span> {
        if (test_size == size) {
            return Span{start, size};
        }
        if (test_size < size || test_size > frame_size) {
            return std::nullopt;
        }
        const int end = start + size;
        int new_start = -1;
        if (test_size == frame_size || start == 0) {
            new_start = 0;
        } else if (end == frame_size) {
            new_start = frame_size - test_size;
        }
        if (new_start < 0 || start - new_start > thin_edge || new_start + test_size - end > thin_edge) {
            return std::nullopt;
        }
        return Span{new_start, test_size};
    };
    const std::optional<Span> across = grown(source_width, test_width, source_box.x, source_box.w);
    const std::optional<Span> down = grown(source_height, test_height, source_box.y, source_box.h);
    const bool grows = across && down && (across->size != source_box.w || down->size != source_box.h);
    if (!grows || (test_width == source_width && test_height == source_height)) {
        return std::nullopt;
    }
    return CropBox{across->size, down->size, across->start, down->start};
}

// The source's box, when the test video's frame shows it whole: an encode
// cut to the source's bars when it was made, then scaled or not (3840x2160
// with 2-row bars to 1920x1078; 1920x800 out of a letterboxed 1920x1080).
// Its frame is then within even rounding of the box's shape (with a sample
// to spare) and nearer it than the source frame's. Nothing where the
// source's frame explains the test's to within a sample, as every
// whole-frame scale rounded to even does: bars so thin that cutting them
// moves the frame by less than a sample cannot be told from no cut (720x404
// is 1920x1080 or 1920x1076 at 720 wide), and are read as the whole frame
// scaled, as before.
std::optional<CropBox> source_box_the_test_shows(int source_width, int source_height, int test_width,
                                                 int test_height, const CropBox& source_box) {
    if (source_box.is_noop(source_width, source_height) || source_box.w <= 0 || source_box.h <= 0) {
        return std::nullopt;
    }
    if (const std::optional<CropBox> grown =
            box_grown_to_the_test(source_width, source_height, test_width, test_height, source_box)) {
        return grown;
    }
    constexpr double whole_frame_rounding = 1.0;  // a whole frame scaled, its size rounded to even samples
    constexpr double box_rounding = 2.0;          // the box scaled and rounded so, with a sample to spare
    const double frame_error = shape_error(source_width, source_height, test_width, test_height);
    const double box_error = shape_error(source_box.w, source_box.h, test_width, test_height);
    if (frame_error <= whole_frame_rounding || box_error > box_rounding || box_error >= frame_error) {
        return std::nullopt;
    }
    return source_box;
}

// The shared picture (edges as fractions of each frame) as one box in each
// video, when the sizes are not one fraction apart (shared_on_one_grid): on
// each axis, the video with fewer samples has it rounded inward to even
// samples, and the other's edges are those scaled to its size, to the
// nearest even sample, kept inside the picture. Both boxes then show the
// same rows, to within a sample of the larger video. Each box rounded inward
// on its own could leave the smaller video's picture several of the larger's
// rows short (1920x1080 against 854x480: the test's rows 64-416 are the
// source's 144-936, against its box's 140-940, a shape 1 % off).
std::pair<CropBox, CropBox> shared_by_rounding(int source_width, int source_height, int test_width,
                                               int test_height, double left, double top, double right,
                                               double bottom) {
    struct Span {
        int start, end;
    };
    // The picture's samples from `from` to `to` (fractions) of `size`, rounded inward to even samples (4:2:0).
    const auto inward = [](double from, double to, int size) {
        int start = static_cast<int>(std::ceil(from * size - 1e-6));
        int end = static_cast<int>(std::floor(to * size + 1e-6));
        start += start & 1;
        end -= end & 1;
        return Span{start, end};
    };
    // One axis: the source's span and the test video's.
    const auto axis = [&inward](int source_size, int test_size, double from, double to) {
        const bool test_smaller = test_size <= source_size;
        const int smaller = test_smaller ? test_size : source_size;
        const int larger = test_smaller ? source_size : test_size;
        const Span rounded = inward(from, to, smaller);
        const Span bounds = inward(from, to, larger);
        const auto scaled = [smaller, larger](int edge) {
            return 2 * static_cast<int>(std::lround(static_cast<double>(edge) * larger / smaller / 2));
        };
        const Span derived{std::max(scaled(rounded.start), bounds.start), std::min(scaled(rounded.end), bounds.end)};
        return test_smaller ? std::pair(derived, rounded) : std::pair(rounded, derived);
    };
    const auto [source_x, test_x] = axis(source_width, test_width, left, right);
    const auto [source_y, test_y] = axis(source_height, test_height, top, bottom);
    return {CropBox{source_x.end - source_x.start, source_y.end - source_y.start, source_x.start, source_y.start},
            CropBox{test_x.end - test_x.start, test_y.end - test_y.start, test_x.start, test_y.start}};
}

}  // namespace

bool samples_through_the_app(const VideoInfo& info) { return windows_from_the_app(info); }

void set_crop_store(std::function<std::filesystem::path()> store) { answers().set_store(std::move(store)); }

void forget_crop_answers() { answers().forget(); }

std::optional<CropAnswer> kept_crop_answer(const VideoInfo& info) {
    // (detect_crop's key: a video whose size changes keeps none)
    const std::optional<AnswerKey> key = info.scaled_to ? std::nullopt : answer_key(info, info.duration);
    return key ? answers().kept(*key) : std::nullopt;
}

void forget_crop_answer(const VideoInfo& info) {
    // (a video whose size changes keeps no answer, but one found before it was seen to change may be kept)
    if (const std::optional<AnswerKey> key = answer_key(info, info.duration)) {
        answers().drop(*key);
    }
}

CropBox detect_crop(const std::string& ffmpeg, const VideoInfo& info, const CancelToken& cancel,
                    const std::optional<std::string>& hwaccel, const std::string& scale_algorithm) {
    const double sampled_duration = info.duration;  // the whole video: the runs give no limit
    // A video whose size changes is sampled scaled, and its box is not kept:
    // the key knows the file, not how its frames are scaled.
    const std::optional<AnswerKey> key =
        info.scaled_to ? std::nullopt : answer_key(info, sampled_duration);
    if (key) {
        if (const std::optional<CropBox> kept = answers().claim(*key, cancel)) {
            return *kept;
        }
    }
    std::optional<CropBox> box;
    BarFindings findings;
    const auto began = std::chrono::steady_clock::now();
    try {
        box = detect_uncached(findings, ffmpeg, info, sampled_duration, cancel, hwaccel, scale_algorithm);
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
        const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - began).count();
        const CropAnswer answer{*box, findings, utc_text(std::chrono::system_clock::now()), seconds};
        answers().settle(*key, spoiled ? std::nullopt : std::optional(answer));
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
    if (cropped_when_encoded(source_width, source_height, distorted_width, distorted_height, *source_crop,
                             *distorted_crop)) {
        return kept;
    }
    // The part of the source the test video's frame shows: the source's box
    // where the test was cut to it when it was made, else its whole frame,
    // when the sizes are one factor apart, across and down, to within half a
    // percent (the same picture at one size or two).
    const std::optional<CropBox> box_shown = source_box_the_test_shows(source_width, source_height, distorted_width,
                                                                       distorted_height, *source_crop);
    if (!box_shown) {
        const double across = static_cast<double>(source_width) / distorted_width;
        const double down = static_cast<double>(source_height) / distorted_height;
        if (std::abs(across - down) > 0.005 * std::max(across, down)) {
            return kept;
        }
    }
    const CropBox shown = box_shown.value_or(CropBox{source_width, source_height, 0, 0});
    // Each box's edges as fractions of what its video shows: the source's
    // within the part the test's frame shows, the test's within its frame.
    struct Edges {
        double left, top, right, bottom;
    };
    const auto edges = [](const CropBox& box, const CropBox& within) {
        return Edges{static_cast<double>(box.x - within.x) / within.w, static_cast<double>(box.y - within.y) / within.h,
                     static_cast<double>(box.x - within.x + box.w) / within.w,
                     static_cast<double>(box.y - within.y + box.h) / within.h};
    };
    const Edges s = edges(*source_crop, shown);
    const Edges d = edges(*distorted_crop, CropBox{distorted_width, distorted_height, 0, 0});
    const double left = std::max(s.left, d.left);
    const double top = std::max(s.top, d.top);
    const double right = std::min(s.right, d.right);
    const double bottom = std::min(s.bottom, d.bottom);
    if (right <= left || bottom <= top) {
        return kept;
    }
    // The shared picture in each video's samples, never a row of the other's
    // bars, its edges on even samples (4:2:0); the source's within the part
    // shown, then moved to its place in the source's frame.
    const auto positive = [](const CropBox& box) { return box.w > 0 && box.h > 0; };
    const std::optional<std::pair<CropBox, CropBox>> on_grid =
        shared_on_one_grid(shown.w, shown.h, distorted_width, distorted_height, left, top, right, bottom);
    auto [shared_source, shared_distorted] =
        on_grid && positive(on_grid->first) && positive(on_grid->second)
            ? *on_grid
            : shared_by_rounding(shown.w, shown.h, distorted_width, distorted_height, left, top, right, bottom);
    if (!positive(shared_source) || !positive(shared_distorted)) {
        return kept;
    }
    shared_source.x += shown.x;
    shared_source.y += shown.y;
    return {shared_source, shared_distorted};
}

SavedBoxesNow rule_2_boxes_now(const VideoInfo& source, const VideoInfo& test, const CropBox& source_box,
                               const CropBox& test_box) {
    if (source.scaled_to || test.scaled_to) {
        return SavedBoxesNow::unknown;
    }
    const auto [source_width, source_height] = source.frame_size();
    const auto [test_width, test_height] = test.frame_size();
    if (std::min({source_width, source_height, test_width, test_height}) <= 0) {
        return SavedBoxesNow::unknown;
    }
    // Frames whose shapes are over half a percent apart: rule 2 kept the boxes
    // as detected, so they are what common_picture is given now.
    const double across = static_cast<double>(source_width) / test_width;
    const double down = static_cast<double>(source_height) / test_height;
    if (std::abs(across - down) > 0.005 * std::max(across, down)) {
        return common_picture(source, test, source_box, test_box) == std::pair(std::optional(source_box),
                                                                                std::optional(test_box))
                   ? SavedBoxesNow::same
                   : SavedBoxesNow::changed;
    }
    // Frames of the same shape, the test's frame one fraction of the source's
    // (largest_grid_denominator), and saved boxes that are that grid's
    // (shared_on_one_grid's: the test's edges on multiples of 2q, the source's
    // the test's scaled by p/q). Such a test box lies within both detected
    // boxes' overlap, which then holds a whole cell of the grid: both rules
    // took the grid's boxes, from the same edges, unless rule 3 read the test
    // as the source cut when it was encoded or as the source's box grown by a
    // thin edge. Both need the source's detected box to fit in the test's
    // frame, and it holds the saved one: a test frame as large as the
    // source's, or narrower or shorter than the saved source box, rules them out.
    if (static_cast<long long>(source_width) * test_height != static_cast<long long>(source_height) * test_width) {
        return SavedBoxesNow::unknown;
    }
    const int common = std::gcd(source_width, test_width);
    const int p = source_width / common;
    const int q = test_width / common;
    const int step = 2 * q;
    const bool on_the_grid = q <= largest_grid_denominator && test_box.w > 0 && test_box.h > 0 &&
                             test_box.w % step == 0 && test_box.h % step == 0 && test_box.x % step == 0 &&
                             test_box.y % step == 0 &&
                             source_box == CropBox{test_box.w / q * p, test_box.h / q * p, test_box.x / q * p,
                                                   test_box.y / q * p};
    const bool source_box_fits_the_test = test_width < source_width && source_box.w <= test_width &&
                                          source_box.h <= test_height;
    return on_the_grid && !source_box_fits_the_test ? SavedBoxesNow::same : SavedBoxesNow::unknown;
}

std::optional<CropBox> test_crop_beside_source_box(const CropBox& source_box, const VideoInfo& test) {
    const auto [width, height] = test.frame_size();
    const int extra_width = width - source_box.w;
    const int extra_height = height - source_box.h;
    const bool only_the_odd_line = (extra_width == 0 || extra_width == 1) && (extra_height == 0 || extra_height == 1);
    if (!only_the_odd_line || (extra_width == 0 && extra_height == 0)) {
        return std::nullopt;
    }
    return CropBox{source_box.w, source_box.h, 0, 0};
}

std::optional<CropBox> box_from_edges(const CropEdges& edges, const VideoInfo& video, std::string_view which) {
    if (edges.none()) {
        return std::nullopt;
    }
    const std::string entered = crop_edges_text(edges);
    for (const int lines : {edges.top, edges.bottom, edges.left, edges.right}) {
        if (lines < 0 || lines % 2 != 0) {
            throw CropEntryError("The crop entered for " + std::string(which) + " (top:bottom:left:right " + entered +
                                 ") must be even numbers of lines: 4:2:0 video keeps its colour in pairs of lines.");
        }
    }
    const auto [width, height] = video.frame_size();
    const int w = (width - edges.left - edges.right) & ~1;
    const int h = (height - edges.top - edges.bottom) & ~1;
    constexpr int smallest_side = 16;
    if (w < smallest_side || h < smallest_side) {
        throw CropEntryError("The crop entered for " + std::string(which) + " (top:bottom:left:right " + entered +
                             ") leaves less than 16x16 of its " + std::to_string(width) + "x" +
                             std::to_string(height) + " picture.");
    }
    return CropBox{w, h, edges.left, edges.top};
}

CropEdges edges_of_box(const CropBox& box, const VideoInfo& video) {
    const auto [width, height] = video.frame_size();
    const auto even_down = [](int lines) { return std::max(0, lines) & ~1; };
    return {even_down(box.y), even_down(height - box.y - box.h), even_down(box.x), even_down(width - box.x - box.w)};
}

std::string crop_edges_text(const CropEdges& edges) {
    return std::to_string(edges.top) + ":" + std::to_string(edges.bottom) + ":" + std::to_string(edges.left) + ":" +
           std::to_string(edges.right);
}

std::optional<CropEdges> parse_crop_edges(std::string_view text) {
    std::vector<int> lines;
    while (true) {
        const std::size_t colon = text.find(':');
        const std::optional<long long> number = python_int(text.substr(0, colon));
        if (!number || *number < 0 || *number > 1'000'000) {
            return std::nullopt;
        }
        lines.push_back(static_cast<int>(*number));
        if (colon == std::string_view::npos) {
            break;
        }
        text.remove_prefix(colon + 1);
    }
    if (lines.size() != 4) {
        return std::nullopt;
    }
    return CropEdges{lines[0], lines[1], lines[2], lines[3]};
}

std::optional<CropBox> source_own_crop(const CropChoice& choice, const std::string& ffmpeg, const VideoInfo& source,
                                       const CancelToken& cancel, const std::optional<std::string>& hwaccel,
                                       const std::string& scale_algorithm) {
    if (choice.mode == CropMode::manual) {
        return box_from_edges(choice.source, source, "the source");
    }
    if (!detects_black_bars(choice.mode)) {
        return std::nullopt;
    }
    return detect_crop(ffmpeg, source, cancel, hwaccel, scale_algorithm);
}

std::pair<std::optional<CropBox>, std::optional<CropBox>> crops_for(const CropChoice& choice, const std::string& ffmpeg,
                                                                    const VideoInfo& source,
                                                                    const VideoInfo& distorted,
                                                                    const CancelToken& cancel,
                                                                    const std::optional<std::string>& source_hwaccel,
                                                                    const std::optional<std::string>& distorted_hwaccel,
                                                                    const std::string& scale_algorithm) {
    switch (choice.mode) {
    case CropMode::none:
        return {};
    case CropMode::manual:
        return {box_from_edges(choice.source, source, "the source"),
                box_from_edges(choice.test, distorted, "the test video")};
    case CropMode::source_only: {  // (the run's settings line says so: "black bars source")
        const CropBox source_box = detect_crop(ffmpeg, source, cancel, source_hwaccel, scale_algorithm);
        return crops_from_detected(choice.mode, false, source, distorted, source_box, std::nullopt);
    }
    case CropMode::automatic:
        break;
    }
    const auto [source_box, distorted_box] =
        detect_pair(ffmpeg, source, distorted, cancel, source_hwaccel, distorted_hwaccel, scale_algorithm);
    const auto crops = crops_from_detected(choice.mode, false, source, distorted, source_box, distorted_box);
    if (crops.first != source_box || crops.second != distorted_box) {
        logger.info("Black bars: the picture both videos show is " + box_text(*crops.first) + " of the source and " +
                    box_text(*crops.second) + " of the test video (detected: " + box_text(source_box) + " and " +
                    box_text(distorted_box) + ")");
    }
    return crops;
}

std::pair<std::optional<CropBox>, std::optional<CropBox>> crops_from_detected(CropMode mode, bool resolution_test,
                                                                              const VideoInfo& source,
                                                                              const VideoInfo& test,
                                                                              const CropBox& source_box,
                                                                              const std::optional<CropBox>& test_box) {
    if (resolution_test) {
        return {source_box, source_box};
    }
    if (mode == CropMode::source_only) {
        return {source_box, test_crop_beside_source_box(source_box, test)};
    }
    return common_picture(source, test, source_box, test_box);
}

}  // namespace vml
