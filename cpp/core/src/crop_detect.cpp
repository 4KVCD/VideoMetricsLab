#include "vml/crop_detect.h"

#include <nlohmann/json.hpp>

#include <chrono>
#include <condition_variable>
#include <fstream>
#include <list>
#include <map>
#include <mutex>
#include <regex>
#include <semaphore>
#include <set>
#include <sstream>
#include <thread>
#include <tuple>
#include <vector>

#include "vml/ffmpeg_tools.h"
#include "vml/hardware_decoding.h"
#include "vml/number_format.h"
#include "vml/platform/process.h"
#include "vml/platform/system.h"

namespace vml {

namespace {

using namespace std::chrono_literals;

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
                return;
            }
        }
        std::error_code ignored;
        std::filesystem::rename(partial, *path, ignored);
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

std::vector<std::string> window_command(const std::string& ffmpeg, const std::string& path, double start,
                                        double window, const std::optional<std::string>& hwaccel,
                                        const std::string& download_format) {
    std::string filters = "cropdetect=limit=" + python_repr(black_limit) + ":round=2:reset=0";
    if (hwaccel) {
        // hwdownload gives the surface's own format, which cropdetect cannot
        // take, so the run's own conversion sits between them.
        filters = "hwdownload,format=" + download_format + "," + filters;
    }
    std::vector<std::string> command{ffmpeg, "-nostdin", "-hide_banner"};
    const std::vector<std::string> decoding = hwaccel_args(hwaccel);
    command.insert(command.end(), decoding.begin(), decoding.end());
    command.insert(command.end(), {"-ss", format_fixed(start, 3), "-i", path, "-map", "0:" + std::string(video_stream),
                                   "-t", format_fixed(window, 3), "-vf", filters, "-f", "null", "-"});
    return command;
}

// Runs one window to its end, holding a decoder slot. Returns its exit code
// and error output.
std::pair<int, std::string> launch_window(const std::vector<std::string>& command, const CancelToken& cancel) {
    while (!decoder_slots.try_acquire_for(100ms)) {
        cancel.throw_if_cancelled();
    }
    struct SlotRelease {
        ~SlotRelease() { decoder_slots.release(); }
    } release_slot;
    try {
        platform::ChildProcess window = platform::ChildProcess::start(command);
        auto running = std::chrono::steady_clock::duration::zero();
        while (!window.wait_for(200ms)) {
            if (cancel.cancelled()) {
                window.terminate();
            }
            running += 200ms;
            if (running >= 60s) {
                window.terminate();
                window.wait();
                throw CropDetectError("Crop detection timed out");
            }
        }
        const int code = window.wait();
        cancel.throw_if_cancelled();
        return {code, window.standard_error()};
    } catch (const platform::ProcessError& error) {
        throw CropDetectError(std::string("Could not run crop detection: ") + error.what());
    }
}

// Runs one window, on the GPU decoder and else in software, and returns the
// box it found, or nothing.
std::optional<CropBox> run_single_window(const std::string& ffmpeg, const std::string& path, double start,
                                         double window, const CancelToken& cancel,
                                         const std::optional<std::string>& hwaccel,
                                         const std::string& download_format) {
    cancel.throw_if_cancelled();
    std::vector<std::optional<std::string>> attempts{hwaccel};
    if (hwaccel) {
        attempts.emplace_back(std::nullopt);
    }
    int code = 0;
    std::string errors;
    for (const std::optional<std::string>& attempt : attempts) {
        try {
            std::tie(code, errors) = launch_window(window_command(ffmpeg, path, start, window, attempt, download_format),
                                                   cancel);
        } catch (const CropDetectError& error) {
            throw CropDetectError(std::string(error.what()) + " (" + path + ")");
        }
        // A hardware attempt that fails is tried again in software; the
        // pixels, and so the box, are the same.
        if (code == 0) {
            break;
        }
    }
    if (code != 0) {
        // The last line of FFmpeg's error output (the output as a whole
        // stripped of spaces at both ends first, as Python strips it).
        std::string tail = "ffmpeg exited with code " + std::to_string(code);
        const std::size_t first = errors.find_first_not_of(" \t\r\n\f\v");
        if (first != std::string::npos) {
            const std::string stripped = errors.substr(first, errors.find_last_not_of(" \t\r\n\f\v") - first + 1);
            const std::size_t line_break = stripped.find_last_of("\r\n");
            tail = line_break == std::string::npos ? stripped : stripped.substr(line_break + 1);
        }
        throw CropDetectError("Crop detection failed for " + path + ": " + tail);
    }
    static const std::regex crop(R"(crop=(\d+):(\d+):(\d+):(\d+))");
    std::optional<CropBox> box;
    for (auto match = std::sregex_iterator(errors.begin(), errors.end(), crop); match != std::sregex_iterator();
         ++match) {
        box = CropBox{std::stoi((*match)[1].str()), std::stoi((*match)[2].str()), std::stoi((*match)[3].str()),
                      std::stoi((*match)[4].str())};
    }
    return box;
}

// detect_crop's work, without the kept answers.
CropBox detect_uncached(const std::string& ffmpeg, const VideoInfo& info, double sampled_duration,
                        const CancelToken& cancel, const std::optional<std::string>& hwaccel) {
    const std::string path = platform::path_text(info.path);
    const double window = sample_window(sampled_duration);
    const std::string download_format = hw_native_format(info.pix_fmt);
    cancel.throw_if_cancelled();
    // One window at a time: each is its own decoder, and five at once held
    // 6.2 GB of GPU memory for a 4K source.
    std::vector<CropBox> boxes;
    std::vector<std::string> failures;
    for (const double start : sample_offsets(sampled_duration, window)) {
        try {
            if (const std::optional<CropBox> box =
                    run_single_window(ffmpeg, path, start, window, cancel, hwaccel, download_format)) {
                boxes.push_back(*box);
            }
        } catch (const CropDetectError& error) {
            failures.emplace_back(error.what());
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
    std::thread reference(detect, 0, std::cref(source), std::cref(source_hwaccel));
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
    return {shared_source, shared_distorted};
}

}  // namespace vml
