#include "vml/black_bars_rule.h"

#include <algorithm>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace vml {

namespace {

// HandBrake's numbers: luma below black (16) counts as black; a dark line's
// mean is below 32 and each of its samples within 16 of that mean.
constexpr int black = 16;
constexpr int dark_mean = 32;
constexpr int even_within = 16;

// Whether the `count` samples from `first`, `step` apart, make a dark line.
bool dark_line(const std::uint8_t* first, std::ptrdiff_t step, int count) {
    long long sum = 0;
    for (int i = 0; i < count; ++i) {
        sum += std::max<int>(first[i * step], black);
    }
    const int mean = static_cast<int>(sum / count);  // the integer mean, as HandBrake's
    if (mean >= dark_mean) {
        return false;
    }
    for (int i = 0; i < count; ++i) {
        if (std::abs(std::max<int>(first[i * step], black) - mean) > even_within) {
            return false;
        }
    }
    return true;
}

// The bar's lines from one edge: `is_dark(i)` for the i-th line in from it,
// at most `limit`, from `border` in; where no bar begins past the border,
// the lines before it decide (all dark: no bar, so that a thin dark line at
// the edge is not cut).
int bar_lines(const std::function<bool(int)>& is_dark, int limit, int border) {
    int lines = border;
    while (lines < limit && is_dark(lines)) {
        ++lines;
    }
    if (lines > border) {
        return lines;
    }
    lines = 0;
    while (lines < border && is_dark(lines)) {
        ++lines;
    }
    return lines >= border ? 0 : lines;
}

}  // namespace

std::optional<BarEdges> picture_edges(const LumaPicture& picture) {
    const int width = picture.width;
    const int height = picture.height;
    if (width <= 0 || height <= 0 || picture.luma.size() < static_cast<std::size_t>(width) * height) {
        return std::nullopt;
    }
    const std::uint8_t* luma = picture.luma.data();
    const int quarter_height = height / 4;
    const int quarter_width = width / 4;
    const int border = height / 100;
    BarEdges edges;
    edges.top = bar_lines([&](int row) { return dark_line(luma + static_cast<std::ptrdiff_t>(row) * width, 1, width); },
                          quarter_height, border);
    edges.bottom = bar_lines(
        [&](int row) { return dark_line(luma + static_cast<std::ptrdiff_t>(height - 1 - row) * width, 1, width); },
        quarter_height, border);
    // The columns between the bars found above and below
    const int rows = height - edges.top - edges.bottom;
    const std::uint8_t* first_row = luma + static_cast<std::ptrdiff_t>(edges.top) * width;
    if (rows > 0) {
        edges.left =
            bar_lines([&](int column) { return dark_line(first_row + column, width, rows); }, quarter_width, 0);
        edges.right = bar_lines([&](int column) { return dark_line(first_row + (width - 1 - column), width, rows); },
                                quarter_width, 0);
    }
    if (edges.top >= quarter_height || edges.bottom >= quarter_height || edges.left >= quarter_width ||
        edges.right >= quarter_width) {
        return std::nullopt;
    }
    return edges;
}

CropBox handbrake_box(const std::vector<BarEdges>& counted, std::size_t decoded, int width, int height) {
    const CropBox whole{width, height, 0, 0};
    if (counted.empty() || counted.size() < std::min(fewest_judged, decoded)) {
        return whole;
    }
    std::vector<int> top, bottom, left, right;
    for (const BarEdges& edges : counted) {
        top.push_back(edges.top);
        bottom.push_back(edges.bottom);
        left.push_back(edges.left);
        right.push_back(edges.right);
    }
    for (std::vector<int>* edge : {&top, &bottom, &left, &right}) {
        std::sort(edge->begin(), edge->end());
    }
    std::size_t index = counted.size() / 2;  // the median
    // Pictures with ten or more lines more on an edge than its median: four
    // or more, and the video changes shape (HandBrake's 9-pixel threshold and
    // switch count for fewer than 30 previews): the smallest bars, cutting
    // no picture.
    const int median[4] = {top[index], bottom[index], left[index], right[index]};
    const auto shows_more = [&median](const BarEdges& edges) {
        return edges.top < median[0] - 9 || edges.bottom < median[1] - 9 || edges.left < median[2] - 9 ||
               edges.right < median[3] - 9;
    };
    if (std::count_if(counted.begin(), counted.end(), shows_more) >= 4) {
        index = 0;
    }
    const auto even = [](int lines) { return lines & ~1; };
    const int t = even(top[index]);
    const int b = even(bottom[index]);
    const int l = even(left[index]);
    const int r = even(right[index]);
    return CropBox{width - l - r, height - t - b, l, t};
}

Y4mLumaReader::Y4mLumaReader(std::function<std::size_t(void*, std::size_t)> read) : read_(std::move(read)) {}

// A line of the stream without its "\n"; empty at the stream's end where
// `at_start` (no byte read yet), else an error.
std::string Y4mLumaReader::line(bool at_start) {
    std::string text;
    char c = 0;
    while (true) {
        if (read_(&c, 1) == 0) {
            if (at_start && text.empty()) {
                return {};
            }
            throw std::runtime_error("the Y4M stream ended inside a header");
        }
        if (c == '\n') {
            return text;
        }
        text.push_back(c);
        if (text.size() > 4096) {
            throw std::runtime_error("a Y4M header line is too long");
        }
    }
}

std::optional<LumaPicture> Y4mLumaReader::next() {
    if (!header_read_) {
        const std::string header = line(true);
        if (header.empty()) {
            return std::nullopt;
        }
        if (!header.starts_with("YUV4MPEG2 ") || header.find(" Cmono16") == std::string::npos) {
            throw std::runtime_error("not a Y4M stream of 16-bit luma: " + header.substr(0, 80));
        }
        for (std::size_t at = header.find(' '); at != std::string::npos; at = header.find(' ', at + 1)) {
            if (header[at + 1] == 'W') {
                width_ = std::atoi(header.c_str() + at + 2);
            } else if (header[at + 1] == 'H') {
                height_ = std::atoi(header.c_str() + at + 2);
            }
        }
        if (width_ <= 0 || height_ <= 0) {
            throw std::runtime_error("a Y4M header without a size: " + header.substr(0, 80));
        }
        header_read_ = true;
    }
    const std::string frame = line(true);
    if (frame.empty()) {
        return std::nullopt;
    }
    if (!frame.starts_with("FRAME")) {
        throw std::runtime_error("a Y4M frame without its FRAME header");
    }
    const std::size_t samples = static_cast<std::size_t>(width_) * height_;
    std::vector<std::uint8_t> words(samples * 2);
    for (std::size_t got = 0; got < words.size();) {
        const std::size_t n = read_(words.data() + got, words.size() - got);
        if (n == 0) {
            throw std::runtime_error("the Y4M stream ended inside a picture");
        }
        got += n;
    }
    LumaPicture picture{width_, height_, std::vector<std::uint8_t>(samples)};
    for (std::size_t i = 0; i < samples; ++i) {
        picture.luma[i] = words[2 * i + 1];  // little-endian: the high byte
    }
    return picture;
}

}  // namespace vml
