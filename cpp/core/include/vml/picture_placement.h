// Where a test video's picture sits in the source's frame when the two
// videos' black-bar boxes alone cannot say (issues #18 and #20): an encode
// cut from the source when it was made that kept a thin dark edge the
// source's detection cut, between bars on both sides (it could be either
// side's), or that was cut a few lines into the picture, or scaled with such
// an edge or such thin bars. Each place the boxes leave is a candidate, and
// pictures of both videos at the same times decide (place_shown).
#pragma once

#include <cstddef>
#include <optional>
#include <vector>

#include "vml/black_bars_rule.h"
#include "vml/video.h"

namespace vml {

// The widest dark edge the source's detection may have cut that an encode
// kept: a 1-sample edge, cut 2 wide (even, for 4:2:0).
inline constexpr int thin_edge = 2;

// The most lines an encode may have cut into the picture, on each axis, for
// its place in the source's frame to be looked for.
inline constexpr int deepest_cut = 32;

// How far, in the test video's samples, its frame is from a picture of
// `width` x `height` scaled to it: the least, over every scale factor k, of
// the larger of |test_width - width k| and |test_height - height k| (which
// is where the two are equal). A picture scaled with its size rounded to even
// samples, each side off by at most one, is at most 1.
double shape_error(double width, double height, int test_width, int test_height);

// The places (source boxes of its size, unscaled) the test video's picture
// (`test_picture`, in its frame) may show, where its frame is the source's
// cut smaller when it was made: on each axis, the source's box (`source_box`)
// itself where the picture is its size; where the picture is larger by up to
// thin_edge, the box grown by that to either side; where it is smaller by up
// to deepest_cut, each place inside the box; every place on even samples,
// with the test's whole frame inside the source's. Nothing where the test's
// frame is larger on an axis, the same size on both, or the picture fits
// none of these on an axis.
std::vector<CropBox> unscaled_places(int source_width, int source_height, int test_width, int test_height,
                                     const CropBox& source_box, const CropBox& test_picture);

// The parts of the source a scaled test frame (its whole frame its picture)
// may show, where the source has bars (`source_box`): the box grown by a thin
// edge to one side on an axis or both, where that grown box's shape is nearer
// the test frame's than the box's, with the box itself; or, where the source's
// frame and the box are both within rounding of the test frame's shape (bars
// thinner than a sample once scaled), the whole frame and the box. The first
// is what the boxes alone would compare (the box, or the whole frame). Nothing
// for other pairs.
std::vector<CropBox> scaled_places(int source_width, int source_height, int test_width, int test_height,
                                   const CropBox& source_box);

// Which of `places` (boxes of the source's frame) the test video's `test_crop`
// shows, from pictures of the two videos at the same times (`source`, `test`,
// pairs at one index): the place whose picture, scaled to the crop's size
// where it is another size, differs least from the crop's on average, over
// the pictures whose crop has detail. Nothing where no picture has detail, a
// box falls outside its picture, or the least is not clearly below the next
// (a quarter less, and half the median where there are more than two).
std::optional<std::size_t> place_shown(const std::vector<LumaPicture>& source, const std::vector<LumaPicture>& test,
                                       const CropBox& test_crop, const std::vector<CropBox>& places);

}  // namespace vml
