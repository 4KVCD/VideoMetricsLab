#include "vml/range_seek.h"

#include <cmath>

#include "vml/number_format.h"

namespace vml {

namespace {

constexpr double shortest_seek_seconds = 30;  // nearer the start, decoding the pictures costs less than the checks
constexpr double early_seconds = 2;
// How far from a whole number a picture's position on the grid may be: a
// container's millisecond timestamps are at most half a millisecond off, 0.06 of
// a picture at 120 fps.
constexpr double grid_tolerance = 0.25;
// Reading a file through against decoding pictures: a 1.6 GB film was read in 7.3 s
// (220 MB/s, a hard disk); NVIDIA's decoder decodes about 300 pictures of 4K a
// second alone. Above this many pictures per MB, decoding them costs more.
constexpr double pictures_per_megabyte_to_count = 1.5;

double as_number(Rational rational) { return static_cast<double>(rational.numerator) / rational.denominator; }

// The timestamp's position on the grid, in pictures.
double grid_position(long long pts, long long first_pts, Rational time_base, Rational frame_rate) {
    return static_cast<double>(pts - first_pts) * as_number(time_base) * as_number(frame_rate);
}

}  // namespace

bool seek_is_worthwhile(long long decode_start, Rational frame_rate) {
    return static_cast<double>(decode_start) >= shortest_seek_seconds * as_number(frame_rate);
}

double seek_seconds(long long decode_start, Rational frame_rate) {
    return static_cast<double>(decode_start) / as_number(frame_rate) - early_seconds;
}

bool counting_packets_pays(long long file_bytes, long long decode_start) {
    const double megabytes = static_cast<double>(file_bytes) / (1024 * 1024);
    return static_cast<double>(decode_start) >= megabytes * pictures_per_megabyte_to_count;
}

long long picture_number_at(long long pts, long long first_pts, Rational time_base, Rational frame_rate) {
    const double position = grid_position(pts, first_pts, time_base, frame_rate);
    const double nearest = std::round(position);
    return std::abs(position - nearest) <= grid_tolerance && nearest >= 0 ? static_cast<long long>(nearest) : -1;
}

bool pictures_fill_grid(long long first_pts, long long last_pts, long long pictures, Rational time_base,
                        Rational frame_rate) {
    const double last = grid_position(last_pts, first_pts, time_base, frame_rate);
    return pictures > 0 && std::abs(last - static_cast<double>(pictures - 1)) <= grid_tolerance;
}

std::vector<std::string> with_seek(std::vector<std::string> input_options, const std::optional<RangeSeek>& seek) {
    if (seek) {
        input_options.insert(input_options.end(),
                             {"-copyts", "-noaccurate_seek", "-ss", format_fixed(seek->start, 9)});
    }
    return input_options;
}

double seconds_before_picture(const RangeSeek& seek, long long number) {
    return static_cast<double>(seek.first_pts) * as_number(seek.time_base) +
           (static_cast<double>(number) - 0.5) / as_number(seek.frame_rate);
}

}  // namespace vml
