#include "vml/picture_placement.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>

namespace vml {

namespace {

// One axis of unscaled_places: where the test's picture (`size` long, at
// `test_start` in a test frame `test_frame` long) may start in the source's
// frame (`source_frame` long), whose box is `box_size` long at `box_start`.
std::vector<int> picture_starts(int source_frame, int test_frame, int box_start, int box_size, int test_start,
                                int size) {
    int first = 0;
    int last = 0;
    if (size == box_size) {
        first = last = box_start;
    } else if (size > box_size && size - box_size <= thin_edge) {
        first = box_start + box_size - size;
        last = box_start;
    } else if (size < box_size && box_size - size <= deepest_cut) {
        first = box_start;
        last = box_start + box_size - size;
    } else {
        return {};
    }
    // The test's whole frame within the source's: the encode was cut from it
    first = std::max(first, test_start);
    last = std::min(last, source_frame - test_frame + test_start);
    // The lines cut from the source when the encode was made are even
    // (FFmpeg's crop and HandBrake's round them so for 4:2:0)
    std::vector<int> starts;
    for (int start = first + ((first - test_start) & 1); start <= last; start += 2) {
        starts.push_back(start);
    }
    return starts;
}

// What one picture's samples are weighed by for one output sample, on one
// axis: from `first` on, one weight each.
struct Taps {
    int first = 0;
    std::vector<float> weights;
};

// The taps that scale `size` samples from `start` to `scaled` samples: each
// output the average of the input it covers, partial samples weighed by the
// part covered (an area filter), or, where the input is the smaller, the
// input interpolated at its centre.
std::vector<Taps> scaling_taps(int start, int size, int scaled) {
    std::vector<Taps> taps(static_cast<std::size_t>(scaled));
    const double step = static_cast<double>(size) / scaled;
    for (int out = 0; out < scaled; ++out) {
        Taps& tap = taps[static_cast<std::size_t>(out)];
        if (step >= 1.0) {
            const double from = out * step;
            const double to = from + step;
            tap.first = static_cast<int>(std::floor(from));
            const int end = std::min(size, static_cast<int>(std::ceil(to)));
            for (int sample = tap.first; sample < end; ++sample) {
                const double covered = std::min<double>(sample + 1, to) - std::max<double>(sample, from);
                tap.weights.push_back(static_cast<float>(covered / step));
            }
        } else {
            const double centre = std::clamp((out + 0.5) * step - 0.5, 0.0, size - 1.0);
            tap.first = size > 1 ? std::min(static_cast<int>(centre), size - 2) : 0;
            const double part = centre - tap.first;
            tap.weights = {static_cast<float>(1.0 - part)};
            if (size > 1) {
                tap.weights.push_back(static_cast<float>(part));
            }
        }
        tap.first += start;
    }
    return taps;
}

// `place` of `picture` scaled to `width` x `height`: rows first, then columns.
std::vector<float> scaled_part(const LumaPicture& picture, const CropBox& place, int width, int height) {
    const std::vector<Taps> across = scaling_taps(place.x, place.w, width);
    const std::vector<Taps> down = scaling_taps(place.y, place.h, height);
    std::vector<float> rows(static_cast<std::size_t>(width) * place.h);
    for (int y = 0; y < place.h; ++y) {
        const std::uint8_t* line = &picture.luma[static_cast<std::size_t>(place.y + y) * picture.width];
        for (int x = 0; x < width; ++x) {
            const Taps& tap = across[static_cast<std::size_t>(x)];
            float sum = 0;
            for (std::size_t k = 0; k < tap.weights.size(); ++k) {
                sum += tap.weights[k] * line[tap.first + static_cast<int>(k)];
            }
            rows[static_cast<std::size_t>(y) * width + x] = sum;
        }
    }
    std::vector<float> scaled(static_cast<std::size_t>(width) * height);
    for (int y = 0; y < height; ++y) {
        const Taps& tap = down[static_cast<std::size_t>(y)];
        for (int x = 0; x < width; ++x) {
            float sum = 0;
            for (std::size_t k = 0; k < tap.weights.size(); ++k) {
                const int row = tap.first - place.y + static_cast<int>(k);
                sum += tap.weights[k] * rows[static_cast<std::size_t>(row) * width + x];
            }
            scaled[static_cast<std::size_t>(y) * width + x] = sum;
        }
    }
    return scaled;
}

// The mean absolute difference between `place` of `source` and `crop` of
// `test`: every second sample of each second line where they are one size
// (enough to tell places a line apart), else every sample of the crop against
// the place scaled to it.
double difference(const LumaPicture& source, const CropBox& place, const LumaPicture& test, const CropBox& crop) {
    const auto test_sample = [&test, &crop](int x, int y) {
        return test.luma[static_cast<std::size_t>(crop.y + y) * test.width + crop.x + x];
    };
    double sum = 0;
    long long count = 0;
    if (place.w == crop.w && place.h == crop.h) {
        for (int y = 0; y < crop.h; y += 2) {
            const std::uint8_t* line = &source.luma[static_cast<std::size_t>(place.y + y) * source.width + place.x];
            long long row_sum = 0;
            for (int x = 0; x < crop.w; x += 2) {
                row_sum += std::abs(static_cast<int>(line[x]) - static_cast<int>(test_sample(x, y)));
                ++count;
            }
            sum += static_cast<double>(row_sum);
        }
        return sum / static_cast<double>(count);
    }
    const std::vector<float> scaled = scaled_part(source, place, crop.w, crop.h);
    for (int y = 0; y < crop.h; ++y) {
        for (int x = 0; x < crop.w; ++x) {
            sum += std::abs(scaled[static_cast<std::size_t>(y) * crop.w + x] - test_sample(x, y));
        }
    }
    return sum / (static_cast<double>(crop.w) * crop.h);
}

// Whether `crop` of `picture` has detail enough to tell places apart: its
// luma's standard deviation (every second sample of each second line) at
// least 2 levels. A dark or flat picture fits every place alike.
bool has_detail(const LumaPicture& picture, const CropBox& crop) {
    constexpr double least_deviation = 2.0;
    double sum = 0;
    double squares = 0;
    long long count = 0;
    for (int y = 0; y < crop.h; y += 2) {
        for (int x = 0; x < crop.w; x += 2) {
            const double sample = picture.luma[static_cast<std::size_t>(crop.y + y) * picture.width + crop.x + x];
            sum += sample;
            squares += sample * sample;
            ++count;
        }
    }
    const double mean = sum / static_cast<double>(count);
    return squares / static_cast<double>(count) - mean * mean >= least_deviation * least_deviation;
}

bool inside(const CropBox& box, const LumaPicture& picture) {
    return box.w > 0 && box.h > 0 && box.x >= 0 && box.y >= 0 && box.x + box.w <= picture.width &&
           box.y + box.h <= picture.height &&
           picture.luma.size() >= static_cast<std::size_t>(picture.width) * picture.height;
}

}  // namespace

double shape_error(double width, double height, int test_width, int test_height) {
    return std::abs(test_width * height - test_height * width) / (width + height);
}

std::vector<CropBox> unscaled_places(int source_width, int source_height, int test_width, int test_height,
                                     const CropBox& source_box, const CropBox& test_picture) {
    const bool smaller = test_width <= source_width && test_height <= source_height &&
                         (test_width < source_width || test_height < source_height);
    if (!smaller || source_box.w <= 0 || source_box.h <= 0 || test_picture.w <= 0 || test_picture.h <= 0) {
        return {};
    }
    const std::vector<int> across =
        picture_starts(source_width, test_width, source_box.x, source_box.w, test_picture.x, test_picture.w);
    const std::vector<int> down =
        picture_starts(source_height, test_height, source_box.y, source_box.h, test_picture.y, test_picture.h);
    std::vector<CropBox> places;
    for (const int y : down) {
        for (const int x : across) {
            places.push_back(CropBox{test_picture.w, test_picture.h, x, y});
        }
    }
    return places;
}

std::vector<CropBox> scaled_places(int source_width, int source_height, int test_width, int test_height,
                                   const CropBox& source_box) {
    if (source_box.is_noop(source_width, source_height) || source_box.w <= 0 || source_box.h <= 0 ||
        (test_width == source_width && test_height == source_height)) {
        return {};
    }
    constexpr double whole_frame_rounding = 1.0;  // a whole frame scaled, its size rounded to even samples
    constexpr double box_rounding = 2.0;          // a box scaled and rounded so, with a sample to spare
    const double frame_error = shape_error(source_width, source_height, test_width, test_height);
    const double box_error = shape_error(source_box.w, source_box.h, test_width, test_height);
    if (box_error > box_rounding) {
        return {};
    }
    if (frame_error <= whole_frame_rounding) {  // bars too thin to tell from the frame's shape alone
        return {CropBox{source_width, source_height, 0, 0}, source_box};
    }
    if (box_error >= frame_error) {
        return {};
    }
    // The box grown by a thin edge to one side, on either axis or both
    struct Span {
        int start, size;
    };
    const auto grown = [](int frame, int start, int size) {
        std::vector<Span> spans{{start, size}};
        if (start >= thin_edge) {
            spans.push_back({start - thin_edge, size + thin_edge});
        }
        if (start + size + thin_edge <= frame) {
            spans.push_back({start, size + thin_edge});
        }
        return spans;
    };
    std::vector<CropBox> places{source_box};
    for (const Span down : grown(source_height, source_box.y, source_box.h)) {
        for (const Span across : grown(source_width, source_box.x, source_box.w)) {
            const CropBox place{across.size, down.size, across.start, down.start};
            if (place != source_box && shape_error(place.w, place.h, test_width, test_height) < box_error) {
                places.push_back(place);
            }
        }
    }
    return places.size() > 1 ? places : std::vector<CropBox>{};
}

std::optional<std::size_t> place_shown(const std::vector<LumaPicture>& source, const std::vector<LumaPicture>& test,
                                       const CropBox& test_crop, const std::vector<CropBox>& places) {
    constexpr double clearly_least = 0.75;  // the least a quarter below the next
    constexpr double clear_of_most = 0.5;   // and half the median, among more than two
    if (places.empty() || source.size() != test.size()) {
        return std::nullopt;
    }
    std::vector<double> differences(places.size(), 0.0);
    int pictures = 0;
    for (std::size_t index = 0; index < test.size(); ++index) {
        if (!inside(test_crop, test[index]) || !has_detail(test[index], test_crop)) {
            continue;
        }
        for (std::size_t place = 0; place < places.size(); ++place) {
            if (!inside(places[place], source[index])) {
                return std::nullopt;
            }
            differences[place] += difference(source[index], places[place], test[index], test_crop);
        }
        ++pictures;
    }
    if (pictures == 0) {
        return std::nullopt;
    }
    const auto least = std::min_element(differences.begin(), differences.end());
    if (places.size() == 1) {
        return 0;
    }
    std::vector<double> sorted = differences;
    std::sort(sorted.begin(), sorted.end());
    const bool clear = sorted[0] <= clearly_least * sorted[1] &&
                       (sorted.size() <= 2 || sorted[0] <= clear_of_most * sorted[sorted.size() / 2]);
    return clear ? std::optional(static_cast<std::size_t>(least - differences.begin())) : std::nullopt;
}

}  // namespace vml
