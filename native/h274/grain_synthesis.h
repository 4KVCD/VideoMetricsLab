// H.274 film grain synthesis (SMPTE RDD 5's frequency filtering model) on
// the CPU, sample for sample as native/d3d11_film_grain.cpp's shaders draw
// it on the GPU (that file says how the grain is made, and how it compares
// with FFmpeg's and VTM's): the pattern database and the blocks' random
// offsets, which both use, and the grain of a plane's rows of 8x8 blocks.
// Header-only and C++17, because the film grain library (built with MinGW)
// and the app (MSVC, C++20) both build it, and the decoder libraries link
// nothing of the app's.
#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include "grain_params.h"
#include "h274_tables.h"

namespace h274 {

// RDD 5's pseudo-random number generator: a 31-bit linear feedback shift
// register (primitive polynomial x^31 + x^3 + 1).
inline void advance_random(uint32_t* state) {
    const uint32_t x = *state;
    *state = (x << 1) | ((1u ^ (x >> 2) ^ (x >> 30)) & 1u);
}

// The pattern database: 13 x 13 patterns of 64x64 samples, pattern (h, v)
// at x = 64 h, y = 64 v of one square of database_size samples a side.
constexpr int pattern_size = 64;
constexpr int cutoff_count = 13;
constexpr int database_size = cutoff_count * pattern_size;

// Builds RDD 5's 13 x 13 grain patterns, one for each pair of horizontal and
// vertical cutoff frequencies. Each pattern is Gaussian noise in the
// frequencies up to its cutoffs, taken through RDD 5's 64-point inverse
// transform; the horizontal edges of each 8x8 block are then attenuated.
inline std::vector<int8_t> build_pattern_database() {
    // RDD 5's integer transform matrix, worked out from its formula.
    const double pi = 3.14159265358979323846;
    int transform[64][64];
    for (int y = 0; y < 64; y++)
        for (int k = 0; k < 64; k++)
            transform[y][k] = k == 0 ? 32 : (int)std::lround(32 * std::sqrt(2.0) * std::cos(pi * (2 * y + 1) * k / 128));

    std::vector<int8_t> database((size_t)database_size * database_size, 0);
    int8_t coefficients[64][64];  // [vertical frequency][horizontal frequency]
    int16_t half_transformed[64][64];
    for (int h = 0; h < cutoff_count; h++) {
        for (int v = 0; v < cutoff_count; v++) {
            const int last_horizontal = ((h + 3) << 2) - 1, last_vertical = ((v + 3) << 2) - 1;
            uint32_t random = h274_seeds[h + v * 13];
            std::memset(coefficients, 0, sizeof coefficients);
            for (int l = 0; l <= last_vertical; l++) {
                for (int k = 0; k <= last_horizontal; k += 4) {
                    const uint32_t start = random % 2048;
                    for (int i = 0; i < 4; i++)
                        coefficients[l][k + i] = h274_gaussian[(start + i) % 2048];
                    advance_random(&random);
                }
            }
            coefficients[0][0] = 0;  // no constant offset
            for (int y = 0; y < 64; y++) {
                for (int x = 0; x <= last_vertical; x++) {
                    int32_t sum = 0;
                    for (int p = 0; p <= last_horizontal; p++)
                        sum += transform[y][p] * coefficients[x][p];
                    half_transformed[y][x] = (int16_t)((sum + 128) >> 8);
                }
            }
            for (int y = 0; y < 64; y++) {
                for (int x = 0; x < 64; x++) {
                    int32_t sum = 0;
                    for (int p = 0; p <= last_vertical; p++)
                        sum += half_transformed[x][p] * transform[y][p];
                    int value = (sum + 128) >> 8;
                    value = value < -127 ? -127 : value > 127 ? 127 : value;
                    if ((y & 7) == 0 || (y & 7) == 7)
                        value = (value * h274_deblock[v]) >> 7;
                    database[(size_t)(v * pattern_size + y) * database_size + h * pattern_size + x] = (int8_t)value;
                }
            }
        }
    }
    return database;
}

// Each 16x16 block's place in its pattern and its sign, in raster order, as
// x | y << 8 | sign << 16 (FFmpeg's ff_h274_apply_film_grain), into
// `offsets` (its memory kept from frame to frame). Each block's numbers
// follow from the block before it, so they are worked out for the whole
// plane at once, before its grain is drawn.
inline void block_offsets(std::vector<uint32_t>& offsets, int poc, int component, int blocks_across, int blocks_down) {
    static const int color_offset[3] = {0, 85, 170};
    uint32_t random = h274_seeds[(poc + color_offset[component]) & 0xFF];
    offsets.resize((size_t)blocks_across * blocks_down);
    for (uint32_t& offset : offsets) {
        const uint32_t x = ((random >> 16) % 52) & 0xFFFC;
        const uint32_t y = ((random & 0xFFFF) % 56) & 0xFFF8;
        offset = x | y << 8 | (random & 1) << 16;
        advance_random(&random);
    }
}

// How one 8x8 block's grain is drawn (the draw shader's
// block_grain_settings): whether it has grain, its signed scale, and where
// its 8x8 window starts in the database.
struct BlockGrain {
    bool present = false;
    int scale = 0;
    int pattern_x = 0, pattern_y = 0;
};

// One plane's grain, as both shaders see it: `in` holds its samples as codes
// (uint8_t at 8 bits, uint16_t above), `in_stride` samples a row apart.
template <typename Sample>
class PlaneDrawer {
public:
    // `offsets` are block_offsets' for this plane, `database`
    // build_pattern_database's; `plane`, `offsets` and `database` must
    // outlive the drawer.
    PlaneDrawer(const Sample* in, std::ptrdiff_t in_stride, const GrainPlane& plane, int depth, int log2_scale_factor,
                const std::vector<uint32_t>& offsets, const int8_t* database)
        : in_(in),
          in_stride_(in_stride),
          plane_(plane),
          depth_(depth),
          shift_(log2_scale_factor + 6),
          offsets_(offsets),
          database_(database),
          blocks8_across_((plane.width + 7) / 8),
          blocks16_across_((plane.width + 15) / 16) {}

    // Rows of 8x8 blocks the plane has.
    int block_rows() const { return (plane_.height + 7) / 8; }

    // Writes the grained samples of block rows [first, end) to `out`,
    // `out_stride` samples a row apart. Rows of blocks are independent of
    // each other, so they can be drawn in any order, on any thread.
    void draw_rows(Sample* out, std::ptrdiff_t out_stride, int first, int end) const {
        std::vector<BlockGrain> row((size_t)blocks8_across_);
        for (int block_y = first; block_y < end; block_y++) {
            for (int block_x = 0; block_x < blocks8_across_; block_x++)
                row[(size_t)block_x] = block_grain(block_x, block_y);
            draw_block_row(row, block_y, out, out_stride);
        }
    }

private:
    // The classify shader: the block's average as an 8-bit code, and the
    // first interval it falls in, or -1 for none. Samples past the plane's
    // edge repeat the edge.
    int block_interval(int block_x, int block_y) const {
        uint32_t sum = 0;
        for (int y = 0; y < 8; y++) {
            const int row = block_y * 8 + y < plane_.height ? block_y * 8 + y : plane_.height - 1;
            for (int x = 0; x < 8; x++) {
                const int column = block_x * 8 + x < plane_.width ? block_x * 8 + x : plane_.width - 1;
                sum += in_[(std::ptrdiff_t)row * in_stride_ + column];
            }
        }
        const uint32_t average = sum >> (depth_ - 2);  // divide by 64 samples, and scale to 8 bits
        for (int i = 0; i < plane_.intervals; i++)
            if (average >= plane_.lower[i] && average <= plane_.upper[i])
                return i;
        return -1;
    }

    // The draw shader's block_grain_settings.
    BlockGrain block_grain(int block_x, int block_y) const {
        BlockGrain grain;
        const int interval = block_interval(block_x, block_y);
        if (interval < 0)
            return grain;
        // The 16x16 block's place in the pattern, plus this 8x8 block's place within it.
        const uint32_t offset = offsets_[(size_t)(block_y >> 1) * blocks16_across_ + (size_t)(block_x >> 1)];
        grain.present = true;
        grain.scale = plane_.scale[interval];
        if (offset & 0x10000)
            grain.scale = -grain.scale;
        grain.pattern_x = plane_.h[interval] * pattern_size + (int)(offset & 0xFF) + 8 * (block_x & 1);
        grain.pattern_y = plane_.v[interval] * pattern_size + (int)((offset >> 8) & 0xFF) + 8 * (block_y & 1);
        return grain;
    }

    // The grain of the sample at (x, y) in a block with these settings.
    int grain_at(const BlockGrain& block, int x, int y) const {
        if (!block.present)
            return 0;
        const int pattern = database_[(std::ptrdiff_t)(block.pattern_y + y) * database_size + block.pattern_x + x];
        return (block.scale * pattern) >> shift_;
    }

    // The draw shader, for one row of blocks.
    void draw_block_row(const std::vector<BlockGrain>& row, int block_y, Sample* out, std::ptrdiff_t out_stride) const {
        const BlockGrain none;
        const int max_code = (1 << depth_) - 1;
        const int to_depth = 1 << (depth_ - 8);  // the grain is made in 8-bit units
        for (int y = 0; y < 8; y++) {
            const int position_y = block_y * 8 + y;
            if (position_y >= plane_.height)
                return;
            const Sample* in_row = in_ + (std::ptrdiff_t)position_y * in_stride_;
            Sample* out_row = out + (std::ptrdiff_t)position_y * out_stride;
            for (int block_x = 0; block_x < blocks8_across_; block_x++) {
                const BlockGrain& left = block_x > 0 ? row[(size_t)block_x - 1] : none;
                const BlockGrain& own = row[(size_t)block_x];
                const BlockGrain& right = block_x + 1 < blocks8_across_ ? row[(size_t)block_x + 1] : none;
                for (int x = 0; x < 8; x++) {
                    const int position_x = block_x * 8 + x;
                    if (position_x >= plane_.width)
                        break;
                    // Deblocking: a block that has grain smooths its left
                    // edge with the block before it, changing its own first
                    // column and the last column of the block before.
                    int g = grain_at(own, x, y);
                    if (x == 0 && block_x > 0 && own.present)
                        g = (grain_at(left, 7, y) + 2 * g + grain_at(own, 1, y)) >> 2;
                    else if (x == 7 && right.present)
                        g = (grain_at(right, 0, y) + 2 * g + grain_at(own, 6, y)) >> 2;
                    const int code = in_row[position_x] + g * to_depth;
                    out_row[position_x] = (Sample)(code < 0 ? 0 : code > max_code ? max_code : code);
                }
            }
        }
    }

    const Sample* in_;
    std::ptrdiff_t in_stride_;
    const GrainPlane& plane_;
    int depth_;
    int shift_;
    const std::vector<uint32_t>& offsets_;
    const int8_t* database_;
    int blocks8_across_;
    int blocks16_across_;
};

}  // namespace h274
