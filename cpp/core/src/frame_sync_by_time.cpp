#include "vml/frame_sync_by_time.h"

#include <chrono>
#include <limits>

#include "vml/gpu_frames.h"

namespace vml {

namespace {

// The end time of an input that ended before giving any frame (INT64_MAX).
constexpr long long end_of_input = std::numeric_limits<long long>::max();

}  // namespace

long long rescale(long long value, Rational from, Rational to) {
    const auto b = static_cast<unsigned long long>(from.numerator) * static_cast<unsigned long long>(to.denominator);
    const auto c = static_cast<unsigned long long>(to.numerator) * static_cast<unsigned long long>(from.denominator);
    if (value >= 0) {
        return static_cast<long long>(multiply_add_divide(static_cast<unsigned long long>(value), b, c / 2, c));
    }
    // The magnitude's, negated: halves away from zero on both sides.
    const unsigned long long magnitude = static_cast<unsigned long long>(-(value + 1)) + 1;
    return -static_cast<long long>(multiply_add_divide(magnitude, b, c / 2, c));
}

long long duration_in(const std::string& text, Rational time_base) {
    return rescale(duration_microseconds(text), Rational{1, 1'000'000}, time_base);
}

// ---------------------------------------------------------------- the inputs

// Fetches the input's next frame, or its end, and stamps it as framesync does
// with after=EXT_STOP (framesync's consume_from_fifos, for one input).
void TimedPairs::Input::fill() {
    if (has_next || done) {
        return;
    }
    const std::optional<Frame> item = ended ? std::nullopt : pull();
    if (!item) {
        ended = true;
        next.reset();
        next_pts = pts ? *pts + 1 : end_of_input;
    } else {
        if (!first) {
            first = item->pts;
        }
        next = item->frame;
        next_pts = to_sync_time(item->pts - *first);
    }
    has_next = true;
}

// Makes the next frame the current one, and releases the frame it replaces.
void TimedPairs::Input::take() {
    const std::optional<int> old = frame;
    frame = next;
    pts = next_pts;
    next.reset();
    has_next = false;
    if (!frame) {
        done = true;
    }
    if (old) {
        release(*old);
    }
}

// Releases the frames the input still holds.
void TimedPairs::Input::drop() {
    for (const std::optional<int>& held : {frame, next}) {
        if (held) {
            release(*held);
        }
    }
    frame.reset();
    next.reset();
}

// ----------------------------------------------------------------- the pairs

TimedPairs::TimedPairs(Pull test, Pull source, Rational test_time_base, Rational source_time_base,
                       Release release_test, Release release_source) {
    test_.pull = std::move(test);
    test_.release = std::move(release_test);
    test_.to_sync_time = [](long long pts) { return pts; };
    source_.pull = std::move(source);
    source_.release = std::move(release_source);
    source_.to_sync_time = [source_time_base, test_time_base](long long pts) {
        return rescale(pts, source_time_base, test_time_base);
    };
    source_.nearest = true;
}

TimedPairs::~TimedPairs() { close(); }

void TimedPairs::close() {
    if (!over_) {
        over_ = true;
        test_.drop();
        source_.drop();
    }
}

std::optional<TimedPairs::Pair> TimedPairs::next() {
    if (over_) {
        return std::nullopt;
    }
    bool ready = false;
    bool ended = false;
    while (!(ready || ended)) {
        test_.fill();
        source_.fill();
        if (test_.ended) {
            // The test video's end drops framesync's sync level to 0, which
            // ends the comparison (framesync_sync_level_update).
            close();
            return std::nullopt;
        }
        long long now = end_of_input;
        for (const Input* input : {&test_, &source_}) {
            if (input->has_next && input->next_pts < now) {
                now = input->next_pts;
            }
        }
        if (now == end_of_input) {
            close();
            return std::nullopt;
        }
        for (Input* input : {&test_, &source_}) {
            if (!input->has_next) {
                continue;
            }
            if (input->next_pts == now || (input->nearest && input->next_pts != end_of_input && input->pts &&
                                           input->next_pts - now < now - *input->pts)) {
                input->take();
                if (input == &test_ && test_.frame) {
                    ready = true;
                }
                if (input->done) {  // with after=EXT_STOP (shortest=1), an input's end ends the comparison
                    ended = true;
                    ready = false;
                }
            }
        }
    }
    if (ended) {
        close();
        return std::nullopt;
    }
    return Pair{*test_.frame, source_.frame, *test_.pts};
}

// ------------------------------------------------- videos decoded in the app

void score_timed_pairs(GpuFrameStream& test, GpuFrameStream& reference,
                       const std::optional<std::string>& duration_limit, long long total_frames,
                       const std::function<void(int, int)>& score,
                       const std::function<void(long long, long long, double)>& on_progress,
                       const CancelToken& cancel) {
    const Rational test_base = test.time_base(cancel);
    const Rational reference_base = reference.time_base(cancel);
    const std::optional<long long> stop =
        duration_limit && !duration_limit->empty() ? std::optional(duration_in(*duration_limit, test_base))
                                                   : std::nullopt;
    const auto puller = [&cancel](GpuFrameStream& stream) {
        return [&stream, &cancel]() -> std::optional<TimedPairs::Frame> {
            const std::optional<GpuFrameStream::Picture> picture = stream.next(cancel);
            if (!picture) {
                return std::nullopt;
            }
            return TimedPairs::Frame{picture->slot, picture->pts};
        };
    };
    TimedPairs pairs(puller(test), puller(reference), test_base, reference_base,
                     [&test](int slot) { test.release(slot); }, [&reference](int slot) { reference.release(slot); });
    const auto started = std::chrono::steady_clock::now();
    auto reported = started;
    long long count = 0;
    while (const std::optional<TimedPairs::Pair> pair = pairs.next()) {
        if (stop && pair->time >= *stop) {
            break;
        }
        if (!pair->source) {
            throw GpuDecodeFailedError("the source has no frame for the test video's first");
        }
        score(pair->test, *pair->source);
        ++count;
        const auto now = std::chrono::steady_clock::now();
        if (on_progress && now - reported >= std::chrono::milliseconds(250)) {
            reported = now;
            const double seconds = std::chrono::duration<double>(now - started).count();
            on_progress(count, total_frames, static_cast<double>(count) / seconds);
        }
    }
}

}  // namespace vml
