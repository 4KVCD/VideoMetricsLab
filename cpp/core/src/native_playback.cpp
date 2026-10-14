#include "vml/native_playback.h"

#include <cmath>
#include <stdexcept>
#include <utility>

#include "vml/python_text.h"

namespace vml {

namespace {

using Matrix = std::array<std::array<double, 3>, 3>;
using Chromaticity = std::pair<double, double>;

// The CIE xy of the red, green and blue of each set of primaries the shader
// converts from, by FFmpeg's names, all with D65 white. Others (SMPTE 431,
// with DCI-P3's own white) are left to FFmpeg's converter.
struct Primaries {
    std::string_view name;
    std::array<Chromaticity, 3> rgb;
};
constexpr std::array<Primaries, 3> primaries_table{{
    {"bt709", {{{0.640, 0.330}, {0.300, 0.600}, {0.150, 0.060}}}},
    {"bt2020", {{{0.708, 0.292}, {0.170, 0.797}, {0.131, 0.046}}}},
    {"smpte432", {{{0.680, 0.320}, {0.265, 0.690}, {0.150, 0.060}}}},  // Display P3
}};
constexpr Chromaticity d65{0.3127, 0.3290};
// In nits: the brightness mapped to SDR white, which is also HLG's display
// peak; and SDR white.
constexpr double peak_nits = 1000.0;
constexpr double white_nits = 100.0;

const Primaries* find_primaries(std::string_view name) {
    for (const Primaries& item : primaries_table) {
        if (item.name == name) {
            return &item;
        }
    }
    return nullptr;
}

// x = a^-1 b, by Gaussian elimination with partial pivoting.
std::array<double, 3> solve(Matrix a, std::array<double, 3> b) {
    for (int column = 0; column < 3; ++column) {
        int pivot = column;
        for (int row = column + 1; row < 3; ++row) {
            if (std::abs(a[row][column]) > std::abs(a[pivot][column])) {
                pivot = row;
            }
        }
        std::swap(a[column], a[pivot]);
        std::swap(b[column], b[pivot]);
        for (int row = column + 1; row < 3; ++row) {
            const double factor = a[row][column] / a[column][column];
            for (int k = column; k < 3; ++k) {
                a[row][k] -= factor * a[column][k];
            }
            b[row] -= factor * b[column];
        }
    }
    std::array<double, 3> x{};
    for (int row = 2; row >= 0; --row) {
        double sum = b[row];
        for (int k = row + 1; k < 3; ++k) {
            sum -= a[row][k] * x[k];
        }
        x[row] = sum / a[row][row];
    }
    return x;
}

Matrix inverse(const Matrix& a) {
    Matrix result{};
    for (int column = 0; column < 3; ++column) {
        std::array<double, 3> unit{};
        unit[column] = 1.0;
        const std::array<double, 3> x = solve(a, unit);
        for (int row = 0; row < 3; ++row) {
            result[row][column] = x[row];
        }
    }
    return result;
}

// The matrix from linear RGB in `primaries` to CIE XYZ: each primary's XYZ
// (Y = 1) as a column, scaled so that RGB 1, 1, 1 is D65 white.
Matrix to_xyz(const Primaries& primaries) {
    Matrix columns{};
    for (int column = 0; column < 3; ++column) {
        const auto [x, y] = primaries.rgb[static_cast<std::size_t>(column)];
        columns[0][column] = x / y;
        columns[1][column] = 1.0;
        columns[2][column] = (1 - x - y) / y;
    }
    const auto [wx, wy] = d65;
    const std::array<double, 3> scale = solve(columns, {wx / wy, 1.0, (1 - wx - wy) / wy});
    for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column) {
            columns[row][column] *= scale[static_cast<std::size_t>(column)];
        }
    }
    return columns;
}

}  // namespace

bool needs_sdr_tonemap(const PreviewColorSettings& colors) {
    return colors.mode == PreviewColorMode::hdr_to_sdr ||
           (colors.mode == PreviewColorMode::display_aware && colors.display_hdr_enabled != true);
}

std::string hdr_primaries(std::string_view tag) {
    const std::string folded = ascii_lower(tag);
    return folded.empty() || folded == "unknown" || folded == "unspecified" || folded == "bt.2020" ? "bt2020" : folded;
}

bool shader_supports(std::string_view primaries) { return find_primaries(hdr_primaries(primaries)) != nullptr; }

NativeShading native_shading_mode(const VideoInfo& info, const PreviewColorSettings& colors) {
    const std::optional<std::string> kind = hdr_kind(info);
    if (!kind) {
        return NativeShading::none;
    }
    if (needs_sdr_tonemap(colors)) {
        return NativeShading::sdr;
    }
    if (*kind == "HDR10 / PQ" && hdr_primaries(info.color_primaries) == "bt2020") {
        return NativeShading::none;
    }
    return NativeShading::hdr;
}

std::optional<std::string> native_playback_problem(const FrameComparison& comparison,
                                                   const PreviewColorSettings& colors) {
    if (colors.mode == PreviewColorMode::unmanaged) {
        return "unmanaged preview uses FFmpeg to bypass native playback's automatic color handling";
    }
    const auto untagged = [](const VideoInfo& info) {
        return info.color_transfer.empty() || info.color_transfer == "unknown" ||
               info.color_transfer == "unspecified";
    };
    if (colors.mode == PreviewColorMode::hdr_to_sdr &&
        (untagged(comparison.source_info) || untagged(comparison.distorted_info))) {
        return "forced HDR interpretation of untagged video uses FFmpeg";
    }
    if (!hdr_kind(comparison.source_info) && !hdr_kind(comparison.distorted_info)) {
        return std::nullopt;
    }
    for (const VideoInfo* info : {&comparison.source_info, &comparison.distorted_info}) {
        if (native_shading_mode(*info, colors) != NativeShading::none && !shader_supports(info->color_primaries)) {
            return info->color_primaries + " HDR primaries use the FFmpeg color converter";
        }
    }
    return std::nullopt;
}

std::array<float, 16> shading_parameters(std::string_view kind, std::string_view primaries, bool hdr) {
    const Primaries* source = find_primaries(hdr_primaries(primaries));
    const Primaries* target = find_primaries(hdr ? "bt2020" : "bt709");
    if (source == nullptr || target == nullptr) {
        throw std::invalid_argument("the shader does not convert " + std::string(primaries) + " primaries");
    }
    const Matrix from = to_xyz(*source);
    const Matrix back = inverse(to_xyz(*target));
    std::array<float, 16> parameters{};
    parameters[0] = hdr ? 2.0f : 1.0f;
    parameters[1] = kind == "HDR10 / PQ" ? 1.0f : 2.0f;
    for (int column = 0; column < 3; ++column) {
        parameters[2 + static_cast<std::size_t>(column)] = static_cast<float>(from[1][column]);
    }
    for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column) {
            double sum = 0.0;
            for (int k = 0; k < 3; ++k) {
                sum += back[row][k] * from[k][column];
            }
            parameters[5 + static_cast<std::size_t>(3 * row + column)] = static_cast<float>(sum);
        }
    }
    parameters[14] = static_cast<float>(peak_nits);
    parameters[15] = static_cast<float>(white_nits);
    return parameters;
}

std::array<float, 12> yuv_to_rgb(std::string_view matrix, bool full_range, bool ten_bit) {
    double kr = 0.2126;
    double kb = 0.0722;
    if (matrix == "bt601") {
        kr = 0.299;
        kb = 0.114;
    } else if (matrix == "bt2020") {
        kr = 0.2627;
        kb = 0.0593;
    } else if (matrix == "smpte240m") {
        kr = 0.212;
        kb = 0.087;
    } else if (matrix == "fcc") {
        kr = 0.30;
        kb = 0.11;
    }
    const double kg = 1 - kr - kb;
    const int bits = ten_bit ? 10 : 8;
    const double code = ten_bit ? 65535.0 / 64 : 255.0;  // codes in a sampled value of 1
    const int step = 1 << (bits - 8);
    double y0 = 0;
    double y_range = 0;
    double c0 = 0;
    double c_range = 0;
    if (full_range) {
        y0 = 0;
        y_range = (1 << bits) - 1;
        c0 = 1 << (bits - 1);
        c_range = (1 << bits) - 1;
    } else {
        y0 = 16 * step;
        y_range = 219 * step;
        c0 = 128 * step;
        c_range = 224 * step;
    }
    const double ay = code / y_range;
    const double by = -y0 / y_range;
    const double ac = code / c_range;
    const double bc = -c0 / c_range;
    const double gu = -2 * kb * (1 - kb) / kg;
    const double gv = -2 * kr * (1 - kr) / kg;
    const std::array<double, 12> rows{ay, 0.0, 2 * (1 - kr) * ac, by + 2 * (1 - kr) * bc,
                                      ay, gu * ac, gv * ac, by + (gu + gv) * bc,
                                      ay, 2 * (1 - kb) * ac, 0.0, by + 2 * (1 - kb) * bc};
    std::array<float, 12> result{};
    for (std::size_t index = 0; index < rows.size(); ++index) {
        result[index] = static_cast<float>(rows[index]);
    }
    return result;
}

std::string_view yuv_matrix_name(std::string_view ffmpeg_space, int height) {
    if (ffmpeg_space == "bt709") {
        return "bt709";
    }
    if (ffmpeg_space == "bt470bg" || ffmpeg_space == "smpte170m") {
        return "bt601";
    }
    if (ffmpeg_space == "bt2020nc" || ffmpeg_space == "bt2020c") {
        return "bt2020";
    }
    if (ffmpeg_space == "smpte240m") {
        return "smpte240m";
    }
    if (ffmpeg_space == "fcc") {
        return "fcc";
    }
    return height > 576 ? "bt709" : "bt601";
}

}  // namespace vml
