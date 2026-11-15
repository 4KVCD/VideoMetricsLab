// Where a raw VVC or HEVC stream can be started to reach a frame
// (raw_keyframes.h), on streams of NAL headers made here.
#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include "vml/raw_keyframes.h"

using namespace vml;
using Bytes = std::vector<std::uint8_t>;

namespace {

// A NAL unit: a start code, its header, and a payload whose first bit says
// whether a picture starts with it (VVC's sh_picture_header_in_slice_header,
// HEVC's first_slice_segment_in_pic); the payload's 00 00 is no start code.
void add_nal(Bytes& stream, RawCodec codec, int type, bool starts_picture = true) {
    stream.insert(stream.end(), {0, 0, 1});
    if (codec == RawCodec::vvc) {
        stream.insert(stream.end(), {0, static_cast<std::uint8_t>(type << 3 | 1)});
    } else {
        stream.insert(stream.end(), {static_cast<std::uint8_t>(type << 1), 1});
    }
    stream.insert(stream.end(), {static_cast<std::uint8_t>(starts_picture ? 0x80 : 0x00), 0, 0, 2, 0x55});
}

std::vector<std::pair<std::int64_t, std::int64_t>> keyframes_of(const Bytes& stream, RawCodec codec, std::size_t piece) {
    RawKeyframes map("", codec);
    for (std::size_t at = 0; at < stream.size(); at += piece) {
        map.scan(stream.data() + at, std::min(piece, stream.size() - at));
    }
    std::vector<std::pair<std::int64_t, std::int64_t>> found;
    for (const RawKeyframe& key : map.keyframes()) {
        found.emplace_back(key.offset, key.first_output);
    }
    return found;
}

}  // namespace

TEST(RawKeyframesTest, KeyframesAreNumberedAsADecoderStartingAtThemGivesFrames) {
    // VVC as VVenC writes it: an IDR whose RADL pictures follow it, then CRAs
    // with their parameter sets, each with RASL pictures, which a decoder
    // starting at the CRA drops.
    Bytes vvc;
    add_nal(vvc, RawCodec::vvc, 15);  // SPS
    add_nal(vvc, RawCodec::vvc, 16);  // PPS
    add_nal(vvc, RawCodec::vvc, 7);   // IDR_W_RADL: shown third
    add_nal(vvc, RawCodec::vvc, 2);   // RADL, shown first
    add_nal(vvc, RawCodec::vvc, 2);
    for (int trailing = 0; trailing < 3; ++trailing) {
        add_nal(vvc, RawCodec::vvc, 0);
    }
    const auto second_unit = static_cast<std::int64_t>(vvc.size());
    add_nal(vvc, RawCodec::vvc, 15);
    add_nal(vvc, RawCodec::vvc, 16);
    add_nal(vvc, RawCodec::vvc, 17);  // APS
    add_nal(vvc, RawCodec::vvc, 9);   // CRA after 6 pictures, its 2 RASL shown before it: frame 8
    add_nal(vvc, RawCodec::vvc, 9, false);  // a second slice of it: no new picture
    add_nal(vvc, RawCodec::vvc, 3);
    add_nal(vvc, RawCodec::vvc, 3);
    add_nal(vvc, RawCodec::vvc, 1);
    add_nal(vvc, RawCodec::vvc, 24);  // suffix SEI: not the next unit's
    const auto third_unit = static_cast<std::int64_t>(vvc.size());
    add_nal(vvc, RawCodec::vvc, 19);  // a picture header, then the CRA's slice: after 10, 1 RASL: frame 11
    add_nal(vvc, RawCodec::vvc, 9, false);
    add_nal(vvc, RawCodec::vvc, 3);
    add_nal(vvc, RawCodec::vvc, 0);
    const std::vector<std::pair<std::int64_t, std::int64_t>> expected{{0, 0}, {second_unit, 8}, {third_unit, 11}};
    // Whole, and in pieces that split start codes and headers.
    for (const std::size_t piece : {vvc.size(), std::size_t{1}, std::size_t{4}, std::size_t{7}}) {
        EXPECT_EQ(keyframes_of(vvc, RawCodec::vvc, piece), expected) << "pieces of " << piece;
    }

    // HEVC: a stream starting at a CRA, whose RASL pictures no decoder shows,
    // so the frames are counted without them.
    Bytes hevc;
    add_nal(hevc, RawCodec::hevc, 32);  // VPS
    add_nal(hevc, RawCodec::hevc, 21);  // CRA
    add_nal(hevc, RawCodec::hevc, 8);   // RASL, dropped
    add_nal(hevc, RawCodec::hevc, 1);
    const auto cra = static_cast<std::int64_t>(hevc.size());
    add_nal(hevc, RawCodec::hevc, 21);  // after 2 shown pictures, its RASL shown first: frame 3
    add_nal(hevc, RawCodec::hevc, 9);
    add_nal(hevc, RawCodec::hevc, 1);
    // A CRA right after an end of sequence, and a BLA: their RASL pictures
    // are dropped, from the start too, so they come after 5 and 7 pictures.
    add_nal(hevc, RawCodec::hevc, 36);  // end of sequence
    const auto after_end = static_cast<std::int64_t>(hevc.size());
    add_nal(hevc, RawCodec::hevc, 21);
    add_nal(hevc, RawCodec::hevc, 9);
    add_nal(hevc, RawCodec::hevc, 1);
    const auto bla = static_cast<std::int64_t>(hevc.size());
    add_nal(hevc, RawCodec::hevc, 16);  // BLA_W_LP
    add_nal(hevc, RawCodec::hevc, 8);
    add_nal(hevc, RawCodec::hevc, 1);
    EXPECT_EQ(keyframes_of(hevc, RawCodec::hevc, 3),
              (std::vector<std::pair<std::int64_t, std::int64_t>>{{0, 0}, {cra, 3}, {after_end, 5}, {bla, 7}}));
}
