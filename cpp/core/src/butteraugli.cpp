// Butteraugli computed by VideoMetricsLab itself (vml/butteraugli.h). The
// scalar parts below are ported from libjxl 0.12 (lib/jxl/butteraugli/
// butteraugli.cc, lib/jxl/convolve_slow.cc; BSD licence,
// docs/THIRD_PARTY.md) with the same operations in the same order; the
// vector kernels are in butteraugli_kernels.cpp.
//
// libjxl's comparator works a whole plane at a time: each step reads its
// input planes and writes whole output planes, about 50 full-size planes
// for a 4K pair (over a gigabyte), each going out to memory and back. This
// engine makes the same map a few rows at a time (butteraugli_image.h): it
// runs in steps, and in each step every stage makes its next band of rows
// from rows that earlier steps made. A stage's band lags the stage before
// it by as many rows as its filter reaches below a row, so a step's bands
// read nothing being written in the same step and the threads take its rows
// in any order. A stage keeps only the rows its consumers still need, so the
// whole comparison stays in tens of megabytes, mostly in the cache.
#include "vml/butteraugli.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <mutex>
#include <new>
#include <stdexcept>
#include <utility>

#include "butteraugli_image.h"
#include "row_pool.h"

namespace vml {
namespace butteraugli {
namespace {

// libjxl's ComputeKernel: a Gaussian of `sigma` truncated at 2.25 sigma.
std::vector<float> compute_kernel(float sigma) {
    const float m = 2.25;
    const double scaler = -1.0 / (2.0 * sigma * sigma);
    const int diff = std::max<int>(1, static_cast<int>(m * std::fabs(sigma)));
    std::vector<float> kernel(2 * diff + 1);
    for (int i = -diff; i <= diff; ++i) {
        kernel[i + diff] = static_cast<float>(std::exp(scaler * i * i));
    }
    return kernel;
}

// A blur of `sigma`, which must have `taps` taps, and its taps scaled by
// 1 / their sum (libjxl's weights away from the borders).
BlurKernel blur_kernel(float sigma, std::size_t taps) {
    const std::vector<float> kernel = compute_kernel(sigma);
    if (kernel.size() != taps) throw std::logic_error("butteraugli: unexpected blur kernel length");
    BlurKernel blur{};
    float weight_no_border = 0.0f;
    for (std::size_t j = 0; j < taps; ++j) {
        blur.taps[j] = kernel[j];
        weight_no_border += kernel[j];
    }
    const float scale_no_border = 1.0f / weight_no_border;
    for (std::size_t i = 0; i <= taps / 2; ++i) blur.sk[i] = kernel[i] * scale_no_border;
    return blur;
}

// libjxl's Blur of sigma 1.2 (5 taps), which goes to its Separable5.
WeightsSeparable5 opsin_blur_weights() {
    const std::vector<float> kernel = compute_kernel(static_cast<float>(1.2));
    float sum_weights = 0.0f;
    for (const float w : kernel) sum_weights += w;
    const float scale = 1.0f / sum_weights;
    const float w0 = kernel[2] * scale;
    const float w1 = kernel[1] * scale;
    const float w2 = kernel[0] * scale;
    return {
        {w0, w0, w0, w0, w1, w1, w1, w1, w2, w2, w2, w2},
        {w0, w0, w0, w0, w1, w1, w1, w1, w2, w2, w2, w2},
    };
}

// ---- The schedule ----

// The engine's stages: kStageRgb to kStageCompare at the full resolution,
// the same at the half, then the map's mixing of the two.
constexpr std::size_t stage_id(std::size_t level, StageKind kind) { return level * kStageKinds + kind; }
constexpr std::size_t kSupersample = 2 * kStageKinds;
constexpr std::size_t kStages = kSupersample + 1;
constexpr std::size_t kNoGroup = kGroups;

// How a consumer's row reaches into its producer's rows.
enum class Reach : std::uint8_t {
    kWindow,       // rows y - radius to y + radius (clipped to the picture)
    kMaltaIn,      // Malta's row p reads the bands' row p - kMaltaMargin
    kMaltaOut,     // the comparison's row y reads Malta's rows y to y + 2 * kMaltaMargin
    kSubsample,    // the half resolution's row y reads the full one's rows 2y and 2y + 1
    kSupersample,  // the full resolution's row y reads the half one's row y / 2
};

struct Edge {
    std::size_t consumer;
    std::size_t producer;
    Reach reach;
    std::int64_t radius;      // kWindow's
    std::size_t group;        // the producer's ring it reads, or kNoGroup
    std::size_t group_level;  // the ring's resolution
};

// The last producer row that consumer row y reads.
std::int64_t last_read(const Edge& e, std::int64_t y, std::int64_t producer_rows) {
    switch (e.reach) {
    case Reach::kWindow: return std::min(y + e.radius, producer_rows - 1);
    case Reach::kMaltaIn:
        return std::clamp<std::int64_t>(y - static_cast<std::int64_t>(kMaltaMargin), 0, producer_rows - 1);
    case Reach::kMaltaOut: return y + 2 * static_cast<std::int64_t>(kMaltaMargin);
    case Reach::kSubsample: return std::min(2 * y + 1, producer_rows - 1);
    case Reach::kSupersample: return y / 2;
    }
    return y;
}

// The last consumer row that reads producer row j.
std::int64_t last_reader(const Edge& e, std::int64_t j, std::int64_t consumer_rows) {
    switch (e.reach) {
    case Reach::kWindow: return std::min(j + e.radius, consumer_rows - 1);
    case Reach::kMaltaIn: return std::min(j + static_cast<std::int64_t>(kMaltaMargin), consumer_rows - 1);
    case Reach::kMaltaOut: return std::min(j, consumer_rows - 1);
    case Reach::kSubsample: return std::min(j / 2, consumer_rows - 1);
    case Reach::kSupersample: return std::min(2 * j + 1, consumer_rows - 1);
    }
    return j;
}

// The rows a consumer row reads at once, which a ring must hold in slots of
// their own.
std::int64_t window_rows(const Edge& e) {
    switch (e.reach) {
    case Reach::kWindow: return 2 * e.radius + 1;
    case Reach::kMaltaOut: return 2 * static_cast<std::int64_t>(kMaltaMargin) + 1;
    case Reach::kSubsample: return 2;
    default: return 1;
    }
}

// When each stage makes each row: in step k, stage s makes its rows
// [k * band - lag, (k + 1) * band - lag).
struct Schedule {
    bool used[kStages] = {};
    std::int64_t rows[kStages] = {};
    std::int64_t band[kStages] = {};
    std::int64_t lag[kStages] = {};
    std::int64_t steps = 0;
    std::size_t ring_rows[2][kGroups] = {};

    std::int64_t step_of(std::size_t stage, std::int64_t row) const { return (row + lag[stage]) / band[stage]; }
};

std::vector<Edge> make_edges(bool has_half) {
    std::vector<Edge> edges;
    const auto add = [&](std::size_t level, StageKind consumer, Group group, Reach reach, std::int64_t radius) {
        edges.push_back({stage_id(level, consumer), stage_id(level, kGroupInfo[group].producer), reach, radius,
                         static_cast<std::size_t>(group), level});
    };
    for (std::size_t level = 0; level < (has_half ? 2u : 1u); ++level) {
        add(level, kStageOpsin, kRgb, Reach::kWindow, 2);
        add(level, kStageLf, kLfh, Reach::kWindow, 16);
        add(level, kStageLf, kXyb, Reach::kWindow, 0);
        add(level, kStageMf, kMfh, Reach::kWindow, 7);
        add(level, kStageMf, kMfRaw, Reach::kWindow, 0);
        add(level, kStageHf, kHfh, Reach::kWindow, 3);
        add(level, kStageHf, kHfRaw, Reach::kWindow, 0);
        add(level, kStageMaskBlur, kMh, Reach::kWindow, 6);
        add(level, kStageMaltaDiffs, kMf, Reach::kMaltaIn, 0);
        add(level, kStageMaltaDiffs, kHf, Reach::kMaltaIn, 0);
        add(level, kStageMaltaDiffs, kUhf, Reach::kMaltaIn, 0);
        add(level, kStageCompare, kMalta, Reach::kMaltaOut, 0);
        add(level, kStageCompare, kLf, Reach::kWindow, 0);
        add(level, kStageCompare, kMf, Reach::kWindow, 0);
        add(level, kStageCompare, kHf, Reach::kWindow, 0);
        add(level, kStageCompare, kBm, Reach::kWindow, 3);  // FuzzyErosion's rows y - 3 to y + 3
    }
    if (has_half) {
        edges.push_back({stage_id(1, kStageRgb), stage_id(0, kStageRgb), Reach::kSubsample, 0, kRgb, 0});
        edges.push_back({kSupersample, stage_id(1, kStageCompare), Reach::kSupersample, 0, kMap, 1});
        // The full resolution's map row, which its comparison writes.
        edges.push_back({kSupersample, stage_id(0, kStageCompare), Reach::kWindow, 0, kNoGroup, 0});
    }
    return edges;
}

Schedule make_schedule(std::size_t ysize, std::size_t half_ysize, bool has_half, std::int64_t band) {
    Schedule s;
    const std::vector<Edge> edges = make_edges(has_half);
    const std::size_t levels = has_half ? 2 : 1;
    for (std::size_t level = 0; level < levels; ++level) {
        const auto rows = static_cast<std::int64_t>(level == 0 ? ysize : half_ysize);
        for (std::size_t kind = 0; kind < kStageKinds; ++kind) {
            const std::size_t stage = stage_id(level, static_cast<StageKind>(kind));
            s.used[stage] = true;
            s.rows[stage] = kind == kStageMaltaDiffs ? rows + 2 * static_cast<std::int64_t>(kMaltaMargin) : rows;
            // The half resolution makes half the rows per step, to keep pace.
            s.band[stage] = level == 0 ? band : std::max<std::int64_t>(1, band / 2);
        }
    }
    if (has_half) {
        s.used[kSupersample] = true;
        s.rows[kSupersample] = static_cast<std::int64_t>(ysize);
        s.band[kSupersample] = band;
    }

    // Lags, in the stages' order (every edge's producer comes before its
    // consumer): the least that puts every row a consumer row reads in an
    // earlier step.
    for (std::size_t c = 0; c < kStages; ++c) {
        if (!s.used[c]) continue;
        std::int64_t lag = 0;
        for (const Edge& e : edges) {
            if (e.consumer != c) continue;
            for (std::int64_t y = 0; y < s.rows[c]; ++y) {
                const std::int64_t made = s.step_of(e.producer, last_read(e, y, s.rows[e.producer]));
                lag = std::max(lag, (made + 1) * s.band[c] - y);
            }
        }
        s.lag[c] = lag;
    }
    for (std::size_t stage = 0; stage < kStages; ++stage) {
        if (s.used[stage]) s.steps = std::max(s.steps, s.step_of(stage, s.rows[stage] - 1) + 1);
    }

    // Ring sizes: a producer's row j may be overwritten (by row j + rows)
    // only in a step after the last one that reads it, and the rows a step
    // makes or a consumer row reads need slots of their own.
    for (std::size_t level = 0; level < levels; ++level) {
        for (std::size_t g = 0; g < kGroups; ++g) {
            if (g == kMap && level == 0) continue;  // the full resolution's map is whole
            const std::size_t p = stage_id(level, kGroupInfo[g].producer);
            std::int64_t ring = s.band[p];
            for (const Edge& e : edges) {
                if (e.group != g || e.group_level != level) continue;
                ring = std::max(ring, window_rows(e));
                for (std::int64_t j = 0; j < s.rows[p]; ++j) {
                    const std::int64_t read = s.step_of(e.consumer, last_reader(e, j, s.rows[e.consumer]));
                    ring = std::max(ring, (read + 1) * s.band[p] - s.lag[p] - j);
                }
            }
            s.ring_rows[level][g] = static_cast<std::size_t>(std::min(ring, s.rows[p]));
        }
    }
    return s;
}

// The rows each stage makes per step (its band; half of it at half
// resolution). Each step ends when its slowest row does, so small bands make
// the threads wait more often: at 4K with both metrics (4 pairs at a time,
// Butteraugli on 5 threads each beside SSIMULACRA2's), 16 rows were 8% slower
// than the whole-plane engine, 32 rows 3%, 64 and 128 rows as fast, 128 with
// 1 GB more rings. A test build with VML_BUTTERAUGLI_BAND_FROM_ENV takes the
// band from VML_BUTTERAUGLI_BAND, to show that every band gives the same map.
std::int64_t band_rows() {
    constexpr std::int64_t kBandRows = 64;
#if defined(VML_BUTTERAUGLI_BAND_FROM_ENV) && defined(_WIN32)
    char* text = nullptr;
    std::size_t length = 0;
    if (_dupenv_s(&text, &length, "VML_BUTTERAUGLI_BAND") == 0 && text != nullptr) {
        const long value = std::strtol(text, nullptr, 10);
        std::free(text);
        if (value >= 2 && value <= 1024) return value;
    }
#endif
    return kBandRows;
}

// Calls fn(level, group, picture, plane, stride, rows, doubled) for each ring
// of `levels` levels of a picture `width` wide, in the order of their block:
// each ring's row stride in floats (padded, butteraugli_image.h), its rows,
// and whether its rows are held twice (a windowed group whose ring is
// shorter than its plane).
template <class Fn>
void for_each_ring_shape(const Schedule& s, std::size_t width, std::size_t levels, Fn&& fn) {
    const std::size_t xsize[2] = {width, (width + 1) / 2};
    for (std::size_t level = 0; level < levels; ++level) {
        for (std::size_t g = 0; g < kGroups; ++g) {
            if (g == kMap && level == 0) continue;  // written straight into the map
            const GroupInfo& info = kGroupInfo[g];
            const std::size_t rows = s.ring_rows[level][g];
            const auto plane_rows = static_cast<std::size_t>(s.rows[stage_id(level, info.producer)]);
            const std::size_t stride =
                g == kMalta ? kMaltaLeft + row_floats(xsize[level] + kMaltaMargin) : row_floats(xsize[level]);
            const bool doubled = info.windowed && rows < plane_rows;
            for (std::size_t picture = 0; picture < (info.per_picture ? 2u : 1u); ++picture) {
                for (std::size_t plane = 0; plane < info.planes; ++plane) {
                    fn(level, g, picture, plane, stride, rows, doubled);
                }
            }
        }
    }
}

// The floats of every ring of a picture of this size, with its schedule.
std::size_t ring_floats(const Schedule& s, std::size_t width, std::size_t levels) {
    std::size_t total = 0;
    for_each_ring_shape(s, width, levels,
                        [&](std::size_t, std::size_t, std::size_t, std::size_t, std::size_t stride, std::size_t rows,
                            bool doubled) { total += stride * rows * (doubled ? 2 : 1); });
    return total;
}

}  // namespace

// One row of one stage, for one picture (0 for the pair's stages).
struct Task {
    std::uint8_t stage;
    std::uint8_t picture;
    std::uint32_t row;
};

struct AlignedFree {
    void operator()(float* p) const { ::operator delete[](p, std::align_val_t{64}); }
};

// convolve_slow.cc's SlowSeparable5 for one row (its SlowSeparablePixel for
// a radius of 2, mirrored borders; rows[] are the mirrored rows y - 2 to
// y + 2).
void slow_separable5_row(const float* const rows[5], std::size_t xsize, const WeightsSeparable5& weights,
                         float* row_out) {
    const auto width = static_cast<std::int64_t>(xsize);
    for (std::int64_t x = 0; x < width; ++x) {
        float mul = 0.0f;
        for (int dy = -2; dy <= 2; ++dy) {
            const float wy = weights.vert[std::abs(dy) * 4];
            const float* row = rows[dy + 2];
            for (int dx = -2; dx <= 2; ++dx) {
                const float wx = weights.horz[std::abs(dx) * 4];
                mul += row[Mirror(x + dx, width)] * wx * wy;
            }
        }
        row_out[x] = mul;
    }
}

}  // namespace butteraugli

class Butteraugli::Engine {
public:
    Engine(const JxlLinearRgb& source_reading, const JxlLinearRgb& test_reading, std::size_t width,
           std::size_t height, float intensity_target, std::size_t threads)
        : pool_(threads),
          readings_{source_reading, test_reading},
          test_scale_(test_scale_of(source_reading, test_reading)),
          width_(width),
          height_(height),
          // libjxl's comparator gives pictures under 8x8 an all-zero map, and
          // compares at half resolution too when that is 8x8 or more.
          compares_(width >= 8 && height >= 8),
          has_half_((width + 1) / 2 >= 8 && (height + 1) / 2 >= 8),
          map_(width * height, 0.0f) {
        if (!compares_) return;
        using namespace butteraugli;
        frame_.params.intensity_target = intensity_target;
        frame_.opsin_blur = opsin_blur_weights();
        frame_.lf_blur = blur_kernel(static_cast<float>(7.15593339443), 33);
        frame_.mf_blur = blur_kernel(static_cast<float>(3.22489901262), 15);
        frame_.hf_blur = blur_kernel(static_cast<float>(1.56416327805), 7);
        frame_.mask_blur = blur_kernel(2.7f, 13);
        frame_.map = map_.data();
        frame_.level[0].xsize = width;
        frame_.level[0].ysize = height;
        frame_.level[1].xsize = (width + 1) / 2;
        frame_.level[1].ysize = (height + 1) / 2;
        const Schedule schedule = make_schedule(height, frame_.level[1].ysize, has_half_, band_rows());
        allocate_rings(schedule, has_half_ ? 2 : 1);
        make_tasks(schedule);
    }

    // libjxl's comparator brings the test picture to the source's scale:
    // each is made linear with its own intensity target (1.0 is 10000 nits
    // for PQ, 1000 for HLG, 255 otherwise), then the test is multiplied by
    // its target over the source's (JxlButteraugliComparator::CompareWith).
    static float test_scale_of(const JxlLinearRgb& source_reading, const JxlLinearRgb& test_reading) {
        return test_reading.intensity_target() / source_reading.intensity_target();
    }

    const std::vector<float>& distortion_map(const std::uint16_t* source, const std::uint16_t* test,
                                             std::size_t stride) {
        if (!compares_) return map_;  // all zero
        pictures_[0] = source;
        pictures_[1] = test;
        stride_ = stride;
        // A row that throws (a thread's first scratch rows not allocated)
        // must not leave the pool, whose threads cannot throw: the step's
        // other rows are skipped and the error is thrown here. The next
        // picture is whole again, as every row is written before it is read.
        std::mutex failure_mutex;
        std::exception_ptr failure;
        std::atomic<bool> failed{false};
        for (std::size_t k = 0; k + 1 < step_begin_.size(); ++k) {
            const std::size_t begin = step_begin_[k];
            pool_.run_each(step_begin_[k + 1] - begin, [&](std::size_t i) {
                if (failed.load(std::memory_order_relaxed)) return;
                try {
                    run_task(tasks_[begin + i]);
                } catch (...) {
                    const std::lock_guard lock(failure_mutex);
                    if (!failure) failure = std::current_exception();
                    failed.store(true, std::memory_order_relaxed);
                }
            });
            if (failed.load(std::memory_order_relaxed)) std::rethrow_exception(failure);
        }
        return map_;
    }

    std::size_t width() const { return width_; }
    std::size_t height() const { return height_; }

private:
    // One block for every ring, 64-byte aligned, each ring's rows padded
    // (butteraugli_image.h). Zeros; NaNs in a build with
    // VML_BUTTERAUGLI_POISON, to show that no pixel reads a row before it is
    // written, or the padding past a row's pixels.
    void allocate_rings(const butteraugli::Schedule& schedule, std::size_t levels) {
        using namespace butteraugli;
        const std::size_t total = ring_floats(schedule, width_, levels);
        rings_.reset(static_cast<float*>(::operator new[](total * sizeof(float), std::align_val_t{64})));
#ifdef VML_BUTTERAUGLI_POISON
        std::fill_n(rings_.get(), total, std::numeric_limits<float>::quiet_NaN());
#else
        std::fill_n(rings_.get(), total, 0.0f);
#endif
        float* next = rings_.get();
        for_each_ring_shape(schedule, width_, levels,
                            [&](std::size_t level, std::size_t g, std::size_t picture, std::size_t plane,
                                std::size_t stride, std::size_t rows, bool doubled) {
            Ring& ring = frame_.level[level].ring[g][picture][plane];
            ring.data = next;
            ring.stride = stride;
            ring.rows = rows;
            ring.doubled = doubled;
            next += stride * rows * (doubled ? 2 : 1);
        });
    }

    // Every step's rows, the heaviest stages first so that the threads that
    // end a step have light rows left to share.
    void make_tasks(const butteraugli::Schedule& schedule) {
        using namespace butteraugli;
        static constexpr StageKind kOrder[kStageKinds] = {
            kStageCompare, kStageOpsin, kStageMaltaDiffs, kStageLf, kStageMf, kStageHf, kStageMaskBlur, kStageRgb,
        };
        std::vector<std::size_t> stages;
        for (std::size_t level = 0; level < 2; ++level) {
            for (const StageKind kind : kOrder) stages.push_back(stage_id(level, kind));
        }
        stages.push_back(kSupersample);
        for (std::int64_t k = 0; k < schedule.steps; ++k) {
            step_begin_.push_back(tasks_.size());
            for (const std::size_t stage : stages) {
                if (!schedule.used[stage]) continue;
                const std::int64_t first = std::max<std::int64_t>(0, k * schedule.band[stage] - schedule.lag[stage]);
                const std::int64_t end =
                    std::min(schedule.rows[stage], (k + 1) * schedule.band[stage] - schedule.lag[stage]);
                const auto kind = static_cast<StageKind>(stage % kStageKinds);
                const bool per_picture = stage != kSupersample && kind != kStageMaltaDiffs && kind != kStageCompare;
                for (std::int64_t row = first; row < end; ++row) {
                    for (std::size_t picture = 0; picture < (per_picture ? 2u : 1u); ++picture) {
                        tasks_.push_back({static_cast<std::uint8_t>(stage), static_cast<std::uint8_t>(picture),
                                          static_cast<std::uint32_t>(row)});
                    }
                }
            }
        }
        step_begin_.push_back(tasks_.size());
    }

    void run_task(const butteraugli::Task& task) {
        using namespace butteraugli;
        if (task.stage == kSupersample) {
            supersample_row(task.row);
            return;
        }
        const std::size_t level = task.stage / kStageKinds;
        const auto kind = static_cast<StageKind>(task.stage % kStageKinds);
        if (kind != kStageRgb) {
            run_stage_row(frame_, kind, level, task.picture, task.row);
        } else if (level == 0) {
            rgb_row(task.picture, task.row);
        } else {
            subsample_row(task.picture, task.row);
        }
    }

    // Row y of a picture in linear rgb, the test's brought to the source's
    // scale.
    void rgb_row(std::size_t picture, std::size_t y) {
        using namespace butteraugli;
        const LevelRings& rings = frame_.level[0];
        float* const rgb[3] = {rings.at(kRgb, picture, 0).row(y), rings.at(kRgb, picture, 1).row(y),
                               rings.at(kRgb, picture, 2).row(y)};
        readings_[picture].convert_row(pictures_[picture] + y * stride_, width_, rgb[0], rgb[1], rgb[2]);
        if (picture == 1 && test_scale_ != 1.0f) {
            for (std::size_t c = 0; c < 3; ++c) {
                for (std::size_t x = 0; x < width_; ++x) rgb[c][x] = test_scale_ * rgb[c][x];
            }
        }
    }

    // libjxl's SubSample2x for row oy: 2x2 means (an odd last column or row
    // counts twice), each pixel's sum and doublings in libjxl's order (its
    // two input rows in turn, then the last column's, then the last row's
    // doubling).
    void subsample_row(std::size_t picture, std::size_t oy) {
        using namespace butteraugli;
        const LevelRings& full = frame_.level[0];
        const LevelRings& half = frame_.level[1];
        const std::size_t xs = half.xsize;
        const std::size_t ys = half.ysize;
        for (std::size_t c = 0; c < 3; ++c) {
            const Ring& out = half.at(kRgb, picture, c);
            float* row_out = out.row(oy);
            std::fill(row_out, row_out + out.stride, 0.0f);
            for (std::size_t y = 2 * oy; y < std::min(2 * oy + 2, full.ysize); ++y) {
                const float* row_in = full.at(kRgb, picture, c).row(y);
                for (std::size_t x = 0; x < full.xsize; ++x) {
                    row_out[x / 2] += 0.25f * row_in[x];
                }
            }
            if ((full.xsize & 1) != 0) {
                row_out[xs - 1] *= 2.0f;
            }
            if ((full.ysize & 1) != 0 && oy == ys - 1) {
                for (std::size_t x = 0; x < xs; ++x) {
                    row_out[x] *= 2.0f;
                }
            }
        }
    }

    // libjxl's AddSupersampled2x for row y of the map: scaled by 1 - 0.3 w
    // (in double), plus w times the half resolution's pixel, w = 0.5.
    void supersample_row(std::size_t y) {
        using namespace butteraugli;
        constexpr double kHeuristicMixingValue = 0.3;
        const float w = 0.5f;
        float* row = map_.data() + y * width_;
        const float* row_src = frame_.level[1].at(kMap, 0, 0).row(y / 2);
        for (std::size_t x = 0; x < width_; ++x) {
            row[x] = static_cast<float>(row[x] * (1.0 - kHeuristicMixingValue * w));
            row[x] += w * row_src[x / 2];
        }
    }

    RowPool pool_;
    std::array<JxlLinearRgb, 2> readings_;  // source's, test's
    float test_scale_;                      // test_scale_of; 1 for pictures of the same target
    std::size_t width_;
    std::size_t height_;
    bool compares_;
    bool has_half_;
    std::vector<float> map_;  // the full resolution's map, which its comparison writes
    butteraugli::StreamFrame frame_;
    std::unique_ptr<float[], butteraugli::AlignedFree> rings_;
    std::vector<butteraugli::Task> tasks_;
    std::vector<std::size_t> step_begin_;  // each step's first task in tasks_, then tasks_'s size
    const std::uint16_t* pictures_[2] = {};
    std::size_t stride_ = 0;
};

Butteraugli::Butteraugli(const JxlLinearRgb& source_reading, const JxlLinearRgb& test_reading, std::size_t width,
                         std::size_t height, float intensity_target, std::size_t threads)
    : engine_(std::make_unique<Engine>(source_reading, test_reading, width, height, intensity_target, threads)) {}
Butteraugli::~Butteraugli() = default;
Butteraugli::Butteraugli(Butteraugli&&) noexcept = default;
Butteraugli& Butteraugli::operator=(Butteraugli&&) noexcept = default;

std::size_t Butteraugli::memory_bytes(std::size_t width, std::size_t height, std::size_t threads) {
    using namespace butteraugli;
    const std::size_t map = width * height * sizeof(float);
    if (width < 8 || height < 8) return map;  // as Engine: no comparison, no rings
    const bool has_half = (width + 1) / 2 >= 8 && (height + 1) / 2 >= 8;
    const Schedule schedule = make_schedule(height, (height + 1) / 2, has_half, band_rows());
    // A stage takes at most 7 scratch rows (ScratchRows in butteraugli_kernels.cpp).
    const std::size_t scratch = threads * 7 * row_floats(width);
    return map + (ring_floats(schedule, width, has_half ? 2 : 1) + scratch) * sizeof(float);
}

std::size_t Butteraugli::width() const { return engine_->width(); }
std::size_t Butteraugli::height() const { return engine_->height(); }

const std::vector<float>& Butteraugli::distortion_map(const std::uint16_t* source, const std::uint16_t* test,
                                                      std::size_t stride) {
    return engine_->distortion_map(source, test, stride);
}

}  // namespace vml
