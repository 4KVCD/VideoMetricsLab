// The AV2 reader (av2_stream.h) on streams made here: the OBUs of four
// temporal units, in AOM's IVF and in WebM (two of them in one frame or
// block, as avmenc writes them) and as a Section 5 stream; and the display
// times of dav2d's pictures (dav2d/display_order.h).
#include <gtest/gtest.h>

#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

#include "dav2d/display_order.h"
#include "test_videos.h"
#include "vml/av2_stream.h"

using namespace vml;
using Bytes = std::vector<std::uint8_t>;

namespace {

// An OBU: its size (one LEB128 byte: the header and payload), its header
// (the type in bits 6-2), its payload.
Bytes obu(av2::ObuType type, Bytes payload = {}) {
    Bytes bytes{static_cast<std::uint8_t>(payload.size() + 1), static_cast<std::uint8_t>(static_cast<int>(type) << 2)};
    bytes.insert(bytes.end(), payload.begin(), payload.end());
    return bytes;
}

Bytes joined(std::initializer_list<Bytes> parts) {
    Bytes all;
    for (const Bytes& part : parts) {
        all.insert(all.end(), part.begin(), part.end());
    }
    return all;
}

void append_le(Bytes& bytes, std::uint64_t value, int count) {
    for (int i = 0; i < count; i++) {
        bytes.push_back(static_cast<std::uint8_t>(value >> (8 * i)));
    }
}

// An IVF file of `frames` at 1/50 s, 64x48, each frame's time 3 ticks apart.
Bytes ivf(const std::vector<Bytes>& frames) {
    Bytes file{'D', 'K', 'I', 'F', 0, 0, 32, 0, 'A', 'V', '0', '2'};
    append_le(file, 64, 2);
    append_le(file, 48, 2);
    append_le(file, 50, 4);  // the time base's denominator, then its numerator
    append_le(file, 1, 4);
    append_le(file, frames.size(), 4);
    append_le(file, 0, 4);
    long long pts = 0;
    for (const Bytes& frame : frames) {
        append_le(file, frame.size(), 4);
        append_le(file, static_cast<std::uint64_t>(pts), 8);
        file.insert(file.end(), frame.begin(), frame.end());
        pts += 3;
    }
    return file;
}

void write(const std::filesystem::path& path, const Bytes& bytes) {
    std::ofstream(path, std::ios::binary).write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
}

// How a Matroska element's size is written: in one byte, two or eight (as
// libwebm writes them), or unknown (all ones, as a live stream writes it).
enum class Size { one, two, eight, unknown };

// A Matroska element: its ID's bytes, its size, its contents.
Bytes element(std::uint32_t id, const Bytes& contents, Size size = Size::one) {
    Bytes bytes;
    for (int shift = 24; shift >= 0; shift -= 8) {
        if ((id >> shift) != 0) {
            bytes.push_back(static_cast<std::uint8_t>(id >> shift));
        }
    }
    const std::uint64_t length = contents.size();
    switch (size) {
        case Size::one:
            bytes.push_back(static_cast<std::uint8_t>(0x80 | length));
            break;
        case Size::two:
            bytes.insert(bytes.end(), {static_cast<std::uint8_t>(0x40 | (length >> 8)), static_cast<std::uint8_t>(length)});
            break;
        case Size::eight:
            bytes.push_back(0x01);
            for (int shift = 48; shift >= 0; shift -= 8) {
                bytes.push_back(static_cast<std::uint8_t>(length >> shift));
            }
            break;
        case Size::unknown:
            bytes.insert(bytes.end(), {0x01, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff});
            break;
    }
    bytes.insert(bytes.end(), contents.begin(), contents.end());
    return bytes;
}

Bytes text(const std::string& value) { return Bytes(value.begin(), value.end()); }

// A block of track `track` (one byte), `relative` ticks after its cluster,
// with `flags` (0x80: a keyframe; 0x02: Xiph lacing).
Bytes block(std::uint32_t id, int track, int relative, const Bytes& frame, std::uint8_t flags = 0x80) {
    Bytes contents{static_cast<std::uint8_t>(0x80 | track), static_cast<std::uint8_t>(relative >> 8),
                   static_cast<std::uint8_t>(relative), flags};
    contents.insert(contents.end(), frame.begin(), frame.end());
    return element(id, contents, Size::two);
}

const Bytes key_unit = joined({obu(av2::ObuType::temporal_delimiter), obu(av2::ObuType::sequence_header, {1, 2}),
                               obu(av2::ObuType::closed_loop_key, {0xaa, 0xbb})});
const Bytes inter_unit = joined({obu(av2::ObuType::temporal_delimiter), obu(static_cast<av2::ObuType>(7), {0x00})});
const Bytes other_inter = joined({obu(av2::ObuType::temporal_delimiter), obu(static_cast<av2::ObuType>(7), {0x05})});
const Bytes open_unit = joined({obu(av2::ObuType::temporal_delimiter), obu(av2::ObuType::open_loop_key, {0x01}),
                                obu(av2::ObuType::leading_tile_group, {0x02})});

// What a test WebM file holds beyond the four units.
enum class Variant { plain, encoded, laced };

// A WebM file with the four units at 1 ms ticks, as libwebm lays it out: a
// SeekHead, a Void, its Info, its Tracks and its Tags before the first
// cluster. A soundtrack (track 2) comes before the AV2 track (1, of `codec`;
// with ContentEncodings in Variant::encoded). The AV2 track's blocks are a
// simple block in a first cluster (8-byte size, with a CRC-32, the
// soundtrack's block and a Void element) and, in a second cluster of unknown
// size, as a live stream writes, a block group with two units and a simple
// block 40 ticks after it (laced in Variant::laced).
Bytes webm(const std::string& codec, Variant variant = Variant::plain) {
    Bytes av2_track = joined({element(0xD7, {1}), element(0x86, text(codec)),
                              element(0xE0, joined({element(0xB0, {64}), element(0xBA, {48})}))});
    if (variant == Variant::encoded) {
        av2_track = joined({av2_track, element(0x6D80, element(0x6240, element(0x5034, {0})))});
    }
    const Bytes tracks = element(0x1654AE6B,
                                 joined({element(0xAE, joined({element(0xD7, {2}), element(0x86, text("A_OPUS"))})),
                                         element(0xAE, av2_track)}),
                                 Size::two);
    const Bytes first = element(0x1F43B675,
                                joined({element(0xBF, {1, 2, 3, 4}), element(0xE7, {0}), block(0xA3, 1, 0, key_unit),
                                        block(0xA3, 2, 0, {1, 2, 3}), element(0xEC, {0, 0})}),
                                Size::eight);
    const Bytes second =
        element(0x1F43B675,
                joined({element(0xE7, {40}), element(0xA0, block(0xA1, 1, 0, joined({inter_unit, other_inter}))),
                        block(0xA3, 1, 40, open_unit, variant == Variant::laced ? 0x82 : 0x80)}),
                Size::unknown);
    const Bytes headers = joined({element(0x114D9B74, element(0x4DBB, {0x53, 0xAB, 0x81, 0x01})), element(0xEC, {0}),
                                  element(0x1549A966, element(0x2AD7B1, {0x0F, 0x42, 0x40})), tracks,
                                  element(0x1254C367, element(0x7373, {}))});
    return joined({element(0x1A45DFA3, element(0x4282, text("webm"))),
                   element(0x18538067, joined({headers, first, second}), Size::unknown)});
}

}  // namespace

// A WebM file's AV2 units come from its AV2 track's blocks alone, numbered,
// with their block's time in its ticks; one sought again comes again, after
// the end of the file too, and those after it are timed by their cluster
// still. A laced block, or one cut short, ends the file before it. A
// Matroska file whose frames are encoded, or with no AV2 track, is no AV2
// stream.
TEST(Av2Stream, ReadsAndSeeksTheUnitsOfAWebmFile) {
    const test::Folder folder;
    write(folder / "a.webm", webm("V_AV2"));
    av2::Reader reader(folder / "a.webm");
    EXPECT_EQ(reader.container(), av2::Container::matroska);
    EXPECT_EQ(reader.width(), 64);
    EXPECT_EQ(reader.height(), 48);
    EXPECT_EQ(reader.time_base(), (Rational{1, 1000}));
    const std::vector<Bytes> units{key_unit, inter_unit, other_inter, open_unit};
    const std::vector<long long> pts{0, 40, 40, 80};
    std::optional<av2::TemporalUnit> third;
    for (long long number = 0; number < 4; number++) {
        const std::optional<av2::TemporalUnit> unit = reader.next();
        ASSERT_TRUE(unit);
        EXPECT_EQ(unit->data, units[number]);
        EXPECT_EQ(unit->number, number);
        EXPECT_EQ(unit->container_pts, pts[number]);
        EXPECT_EQ(unit->kind.closed_key, number == 0);
        if (number == 2) {
            third = unit;
        }
    }
    EXPECT_FALSE(reader.next());
    reader.seek(*third);
    const std::optional<av2::TemporalUnit> again = reader.next();
    ASSERT_TRUE(again);
    EXPECT_EQ(again->number, 2);
    EXPECT_EQ(again->data, other_inter);
    const std::optional<av2::TemporalUnit> last = reader.next();
    ASSERT_TRUE(last);
    EXPECT_EQ(last->container_pts, 80);

    Bytes cut = webm("V_AV2");
    cut.resize(cut.size() - 1);
    write(folder / "cut.webm", cut);
    write(folder / "laced.webm", webm("V_AV2", Variant::laced));
    for (const char* name : {"cut.webm", "laced.webm"}) {
        av2::Reader ended(folder / name);
        for (int count = 0; count < 3; count++) {
            EXPECT_TRUE(ended.next()) << name;
        }
        EXPECT_FALSE(ended.next()) << name;
        EXPECT_FALSE(ended.next()) << name;  // and stays ended
    }
    write(folder / "encoded.webm", webm("V_AV2", Variant::encoded));
    write(folder / "av1.webm", webm("V_AV1"));
    for (const char* name : {"encoded.webm", "av1.webm"}) {
        EXPECT_EQ(av2::container_of(folder / name), std::nullopt) << name;
    }
}

// A scan of the OBUs' headers alone gives each unit as a full read does,
// but for its payloads: its number, time, kind, and where a seek goes; and a
// file's keyframes are scanned for once.
TEST(Av2Stream, AScanOfTheHeadersGivesTheUnitsOfAFullRead) {
    const test::Folder folder;
    write(folder / "a.ivf", ivf({key_unit, joined({inter_unit, other_inter}), open_unit}));
    write(folder / "a.obu", joined({key_unit, inter_unit, other_inter, open_unit}));
    write(folder / "a.webm", webm("V_AV2"));
    for (const char* name : {"a.ivf", "a.obu", "a.webm"}) {
        av2::Reader full(folder / name);
        av2::Reader scan(folder / name, av2::Payloads::skipped);
        while (const std::optional<av2::TemporalUnit> unit = full.next()) {
            const std::optional<av2::TemporalUnit> scanned = scan.next();
            ASSERT_TRUE(scanned) << name;
            EXPECT_EQ(scanned->number, unit->number) << name;
            EXPECT_EQ(scanned->container_pts, unit->container_pts) << name;
            EXPECT_EQ(scanned->offset, unit->offset) << name;
            EXPECT_EQ(scanned->index, unit->index) << name;
            EXPECT_EQ(scanned->kind.closed_key, unit->kind.closed_key) << name;
            EXPECT_EQ(scanned->kind.open_key, unit->kind.open_key) << name;
            EXPECT_EQ(scanned->kind.leading, unit->kind.leading) << name;
            EXPECT_LT(scanned->data.size(), unit->data.size()) << name;
        }
        EXPECT_FALSE(scan.next()) << name;
    }

    // Its closed-loop keyframes, kept: the same list again, until the file is
    // written again.
    const std::shared_ptr<const std::vector<av2::TemporalUnit>> keyframes = av2::closed_keyframes(folder / "a.ivf");
    ASSERT_EQ(keyframes->size(), 1u);
    EXPECT_EQ(keyframes->front().number, 0);
    EXPECT_EQ(av2::closed_keyframes(folder / "a.ivf"), keyframes);
    write(folder / "a.ivf", ivf({key_unit, inter_unit, key_unit}));
    EXPECT_EQ(av2::closed_keyframes(folder / "a.ivf")->size(), 2u);
}

// Each unit comes whole, in order, numbered, with its frame's time and what
// its OBUs say: a closed-loop keyframe, two pictures that need the ones
// before (one IVF frame), an open-loop keyframe with its leading picture.
TEST(Av2Stream, ReadsTheUnitsOfAnIvfFileAndASection5Stream) {
    const test::Folder folder;
    write(folder / "a.ivf", ivf({key_unit, joined({inter_unit, other_inter}), open_unit}));
    write(folder / "a.obu", joined({key_unit, inter_unit, other_inter, open_unit}));
    for (const char* name : {"a.ivf", "a.obu"}) {
        av2::Reader reader(folder / name);
        const bool is_ivf = reader.container() == av2::Container::ivf;
        EXPECT_EQ(is_ivf, std::string(name) == "a.ivf");
        EXPECT_EQ(reader.width(), is_ivf ? 64 : 0);
        EXPECT_EQ(reader.time_base(), (is_ivf ? Rational{1, 50} : Rational{1, 1}));
        const std::vector<Bytes> units{key_unit, inter_unit, other_inter, open_unit};
        const std::vector<long long> ivf_pts{0, 3, 3, 6};
        const std::vector<int> ivf_index{0, 0, 1, 0};
        for (long long number = 0; number < 4; number++) {
            const std::optional<av2::TemporalUnit> unit = reader.next();
            ASSERT_TRUE(unit) << name;
            EXPECT_EQ(unit->data, units[number]) << name;
            EXPECT_EQ(unit->number, number) << name;
            EXPECT_EQ(unit->container_pts, is_ivf ? ivf_pts[number] : number) << name;
            EXPECT_EQ(unit->index, is_ivf ? ivf_index[number] : 0) << name;
            EXPECT_EQ(unit->kind.closed_key, number == 0) << name;
            EXPECT_EQ(unit->kind.open_key, number == 3) << name;
            EXPECT_EQ(unit->kind.leading, number == 3) << name;
        }
        EXPECT_FALSE(reader.next()) << name;
    }
}

// A unit sought again comes again, numbered as before, from any reader of
// the file: the second unit of an IVF frame, a Section 5 unit.
TEST(Av2Stream, ASeekGivesTheUnitAgain) {
    const test::Folder folder;
    write(folder / "a.ivf", ivf({key_unit, joined({inter_unit, other_inter}), open_unit}));
    write(folder / "a.obu", joined({key_unit, inter_unit, other_inter, open_unit}));
    for (const char* name : {"a.ivf", "a.obu"}) {
        av2::Reader first(folder / name);
        std::optional<av2::TemporalUnit> third;
        while (std::optional<av2::TemporalUnit> unit = first.next()) {
            if (unit->number == 2) {
                third = unit;
            }
        }
        ASSERT_TRUE(third) << name;
        av2::Reader second(folder / name);
        second.seek(*third);
        const std::optional<av2::TemporalUnit> again = second.next();
        ASSERT_TRUE(again) << name;
        EXPECT_EQ(again->number, 2) << name;
        EXPECT_EQ(again->data, other_inter) << name;
        first.seek(*third);  // after the end of the file too
        EXPECT_EQ(first.next()->number, 2) << name;
        EXPECT_EQ(first.next()->number, 3) << name;
    }
}

// A file cut inside a frame ends at the frame before; a file of another form,
// IVF of another codec or AV1's raw streams, is no AV2 stream.
TEST(Av2Stream, ACutFileEndsAtItsLastWholeFrameAndOtherFilesAreRefused) {
    const test::Folder folder;
    Bytes cut = ivf({key_unit, inter_unit});
    cut.resize(cut.size() - 1);
    write(folder / "cut.ivf", cut);
    av2::Reader reader(folder / "cut.ivf");
    EXPECT_TRUE(reader.next());
    EXPECT_FALSE(reader.next());

    Bytes av1 = ivf({key_unit});
    av1[11] = '1';  // "AV01"
    write(folder / "av1.ivf", av1);
    write(folder / "hevc.bin", {0, 0, 0, 1, 0x40, 0x01, 0x0c});
    // AV1's raw streams: Section 5 (its temporal delimiter's header 0x12,
    // then its size, then the sequence header's 0x0A) and Annex B (the sizes
    // of a temporal unit, a frame and an OBU, then the delimiter).
    write(folder / "av1.obu", {0x12, 0x00, 0x0a, 0x0b, 0x00, 0x00, 0x00, 0x24, 0xc4, 0xff, 0xdf, 0x00});
    write(folder / "av1_annexb.obu", {0x10, 0x0e, 0x01, 0x10, 0x0a, 0x0a, 0x0b, 0x00, 0x00, 0x00, 0x24, 0xc4});
    for (const char* name : {"av1.ivf", "hevc.bin", "av1.obu", "av1_annexb.obu"}) {
        EXPECT_EQ(av2::container_of(folder / name), std::nullopt) << name;
        EXPECT_THROW(av2::Reader(folder / name), av2::Av2StreamError) << name;
    }
}

// dav2d's pictures come in display order, each with the stamp of the unit
// that decoded it, as it gave them for avmenc's 20-picture test stream: the
// first unit's picture at once, nothing for the next four (decoded ahead),
// then two pictures after the sixth, stamped 5 and 4. Each is due at the
// smallest stamp not yet given; after a seek, at the new units' stamps.
TEST(Av2Stream, Dav2dsPicturesAreTimedInDisplayOrder) {
    Dav2dDisplayOrder order;
    order.fed(0);
    EXPECT_EQ(order.shown(), 0);
    for (long long unit = 1; unit <= 5; unit++) {
        order.fed(unit);
    }
    EXPECT_EQ(order.shown(), 1);  // dav2d's stamp 5
    EXPECT_EQ(order.shown(), 2);  // dav2d's stamp 4
    order.clear();
    EXPECT_EQ(order.shown(), std::nullopt);  // more pictures than units
    order.fed(8);
    EXPECT_EQ(order.shown(), 8);
}
