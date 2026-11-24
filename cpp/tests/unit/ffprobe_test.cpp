#include "vml/ffprobe.h"

#include <gtest/gtest.h>

#include <chrono>
#include <filesystem>
#include <string>
#include <thread>

#include "reference.h"
#include "test_videos.h"
#include "vml/geometry.h"
#include "vml/platform/system.h"
#include "vml/run_json.h"

namespace vml {
namespace {

// ffprobe's output is read as the Python app reads it, field for field: on
// the golden clips' real output, and on written output for Matroska's tags,
// soundtracks, cover art, odd and missing values, no video and no JSON.
TEST(Ffprobe, ReadsTheVideosDetailsAsThePythonAppDoes) {
    const nlohmann::json reference = test::load_reference("probe.json");
    ASSERT_FALSE(reference["cases"].empty());
    for (const nlohmann::json& item : reference["cases"]) {
        SCOPED_TRACE(item["name"].get<std::string>());
        const std::filesystem::path path = platform::path_from_text(item["path"].get<std::string>());
        const nlohmann::json& expected = item["expected"];
        if (expected.contains("error")) {
            try {
                video_info_from_probe(path, item["output"].get<std::string>());
                ADD_FAILURE() << "no error";
            } catch (const ProbeError& error) {
                EXPECT_EQ(std::string(error.what()), expected["error"].get<std::string>());
            }
            continue;
        }
        const VideoInfo info = video_info_from_probe(path, item["output"].get<std::string>());
        EXPECT_EQ(platform::path_text(info.path), expected["path"].get<std::string>());
        EXPECT_EQ(info.width, expected["width"].get<int>());
        EXPECT_EQ(info.height, expected["height"].get<int>());
        EXPECT_EQ(info.fps, *test::reference_number(expected["fps"]));
        EXPECT_EQ(info.duration, *test::reference_number(expected["duration"]));
        EXPECT_EQ(info.nb_frames, expected["nb_frames"].get<long long>());
        EXPECT_EQ(info.codec_name, expected["codec_name"].get<std::string>());
        EXPECT_EQ(info.sar, expected["sar"].get<std::string>());
        EXPECT_EQ(info.pix_fmt, expected["pix_fmt"].get<std::string>());
        EXPECT_EQ(info.bit_rate, expected["bit_rate"].get<long long>());
        EXPECT_EQ(info.bit_rate_whole_file, expected["bit_rate_whole_file"].get<bool>());
        EXPECT_EQ(info.bit_rate_estimated, expected.value("bit_rate_estimated", false));
        EXPECT_EQ(info.nominal_fps, *test::reference_number(expected["nominal_fps"]));
        EXPECT_EQ(info.average_fps, *test::reference_number(expected["average_fps"]));
        EXPECT_EQ(info.color_range, expected["color_range"].get<std::string>());
        EXPECT_EQ(info.color_space, expected["color_space"].get<std::string>());
        EXPECT_EQ(info.color_transfer, expected["color_transfer"].get<std::string>());
        EXPECT_EQ(info.color_primaries, expected["color_primaries"].get<std::string>());
        EXPECT_EQ(info.chroma_location, expected["chroma_location"].get<std::string>());
        EXPECT_EQ(info.format_name, expected["format_name"].get<std::string>());
        EXPECT_EQ(info.start_offset, *test::reference_number(expected["start_offset"]));
    }
}

// A rotation tag as ffprobe gives it, and a pair stored at different rotations
// refused: compared as stored, one would be on its side against the other.
// An HDR video's frames list a column for their side data after the size.
TEST(Ffprobe, FrameSizesAreReadWithoutTheirSideData) {
    EXPECT_FALSE(listed_sizes_differ("3840,2160,\n3840,2160,\n", 3840, 2160));
    EXPECT_FALSE(listed_sizes_differ("3840,2160\n", 3840, 2160));
    EXPECT_TRUE(listed_sizes_differ("3840,2160,\n2560,1440,\n", 3840, 2160));
    EXPECT_TRUE(listed_sizes_differ("1920,1080\n", 3840, 2160));
    EXPECT_FALSE(listed_sizes_differ("", 3840, 2160));
}

// A raw VVC stream copied into MP4 by FFmpeg (-c copy) is stamped as decoded:
// its packets' stamps only rise, and its pictures, shown, go back. A file
// stamped right stores its reordered pictures out of order.
TEST(Ffprobe, TimestampsInDecodingOrderAreToldFromRightOnes) {
    const auto stamps = [](std::string_view listed) { return listed_timestamps(listed).value(); };
    EXPECT_EQ(stamps("-512\n-256\n0\n\n256\n"), (std::vector<long long>{-512, -256, 0, 256}));
    EXPECT_FALSE(listed_timestamps("0\nN/A\n512\n"));  // a raw stream's: no stamps to judge

    std::vector<long long> as_decoded;
    for (long long stamp = -512; stamp < 32 * 256 - 512; stamp += 256) {
        as_decoded.push_back(stamp);
    }
    EXPECT_TRUE(stamps_rise_as_stored(as_decoded));
    EXPECT_FALSE(stamps_rise_as_stored({0, 256, 512}));  // too few to say
    EXPECT_FALSE(stamps_rise_as_stored(stamps("626\n292\n125\n42\n0\n83\n209\n167\n250\n459\n375\n334\n417\n542\n"
                                              "501\n584\n")));  // an MKV stamped right
    EXPECT_FALSE(stamps_rise_as_stored({0, 20, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280}));

    EXPECT_TRUE(stamps_go_back_as_shown({768, 512, 1024, 256, 1280, 0}));
    EXPECT_FALSE(stamps_go_back_as_shown({0, 42, 83, 125, 167}));
}

TEST(Ffprobe, ReadsARotationTagAndAPairMustShareIt) {
    const auto video_turned = [](const std::string& side_data) {
        return video_info_from_probe(
            "a.mp4", R"({"streams":[{"codec_type":"video","codec_name":"h264","width":640,"height":268,)"
                     R"("pix_fmt":"yuv420p","r_frame_rate":"24/1","avg_frame_rate":"24/1","nb_frames":"24",)"
                     R"("duration":"1.0")" + side_data + R"(}],"format":{"duration":"1.0"}})");
    };
    const VideoInfo upright = video_turned("");
    const VideoInfo upside_down = video_turned(R"(,"side_data_list":[{"side_data_type":"Display Matrix","rotation":-180}])");
    const VideoInfo sideways = video_turned(R"(,"side_data_list":[{"side_data_type":"Display Matrix","rotation":90}])");
    EXPECT_EQ(upright.rotation, 0);
    EXPECT_EQ(upside_down.rotation, 180);
    EXPECT_EQ(sideways.rotation, 90);
    EXPECT_EQ(pair_problem(upside_down, upside_down, 0, 0), std::nullopt);
    EXPECT_NE(pair_problem(upside_down, upright, 0, 0), std::nullopt);
}

// Matroska's statistics tags tell a stream's rate, unless FFmpeg wrote the
// file: then they are its input's, even through a re-encode.
TEST(Ffprobe, StatisticsTagsCountUnlessFfmpegCopiedThem) {
    const auto rate_when_written_by = [](const std::string& writer) {
        const std::string probe = R"({"streams":[{"codec_type":"video","codec_name":"hevc","width":1280,"height":720,)"
                                  R"("pix_fmt":"yuv420p","r_frame_rate":"24/1","avg_frame_rate":"24/1",)"
                                  R"("tags":{"BPS":"100000"}}],)"
                                  R"("format":{"duration":"10.0","size":"789000","bit_rate":"631200",)"
                                  R"("format_name":"matroska,webm","tags":{"ENCODER":")" +
                                  writer + R"("}}})";
        return video_info_from_probe("a.mkv", probe).bit_rate;
    };
    EXPECT_EQ(rate_when_written_by("libebml v1.4.5 + libmatroska v1.7.1"), 100000);
    EXPECT_EQ(rate_when_written_by("Lavf63.1.101"), 631200);
}

// mkvmerge before v40 gave a track with a language its statistics tags with
// it: DURATION-eng, BPS-eng. They are the track's as the plain ones are: a
// Blu-ray remux whose audio runs 0.8 s past its picture is as long as its
// picture (133,697 frames), not as the file. A plain tag comes first.
TEST(Ffprobe, StatisticsTagsWithALanguageAreTheTracks) {
    const auto probe = [](const std::string& video_tags) {
        return R"({"streams":[{"index":0,"codec_type":"video","codec_name":"h264","width":1920,"height":1080,)"
               R"("pix_fmt":"yuv420p","r_frame_rate":"24000/1001","avg_frame_rate":"24000/1001","tags":{)" +
               video_tags +
               R"(}},{"index":1,"codec_type":"audio","codec_name":"dts"}],)"
               R"("format":{"duration":"5577.056000","size":"22000000000","format_name":"matroska,webm",)"
               R"("tags":{"ENCODER":"libebml v1.3.6 + libmatroska v1.5.0"}}})";
    };
    const VideoInfo remux = video_info_from_probe(
        "remux.mkv", probe(R"("BPS-eng":"30600748","DURATION-eng":"01:32:56.279000000",)"
                           R"("NUMBER_OF_FRAMES-eng":"133697","NUMBER_OF_BYTES-eng":"21329788989")"));
    EXPECT_DOUBLE_EQ(remux.duration, 5576.279);
    EXPECT_EQ(remux.bit_rate, 30600748);
    EXPECT_FALSE(remux.bit_rate_estimated || remux.bit_rate_whole_file);
    EXPECT_EQ(remux.stream_bytes, 21329788989);
    EXPECT_EQ(remux.estimated_frame_count(), 133697);

    const VideoInfo both = video_info_from_probe(
        "both.mkv", probe(R"("DURATION-eng":"01:00:00.000000000","DURATION":"01:32:56.279000000")"));
    EXPECT_DOUBLE_EQ(both.duration, 5576.279);
    // Not a language: another tag that starts so.
    const VideoInfo other = video_info_from_probe("other.mkv", probe(R"("DURATION-IN-FRAMES-2":"01:00:00.0")"));
    EXPECT_DOUBLE_EQ(other.duration, 5577.056);
}

// A file FFmpeg wrote, of 1,000,000 bytes over 10 s (800 kb/s), with a video
// and an Opus track whose sizes the tags give as `video_bytes` and
// `audio_bytes` ("" for no tags); with `cover_art`, a cover as FFmpeg lists
// one (a picture stream, with no tags); and with `subtitle_bytes`, a PGS
// track of that size.
VideoInfo ffmpeg_written(const std::string& video_bytes, const std::string& audio_bytes, bool cover_art = false,
                         const std::string& subtitle_bytes = "") {
    const auto tags = [](const std::string& bytes) {
        return bytes.empty() ? std::string() : R"(,"tags":{"NUMBER_OF_BYTES":")" + bytes + R"("})";
    };
    std::string extra = cover_art ? R"(,{"index":2,"codec_type":"video","codec_name":"mjpeg",)"
                                    R"("disposition":{"attached_pic":1}})"
                                  : "";
    if (!subtitle_bytes.empty()) {
        extra += R"(,{"index":3,"codec_type":"subtitle","codec_name":"hdmv_pgs_subtitle")" + tags(subtitle_bytes) +
                 "}";
    }
    const std::string probe = R"({"streams":[{"index":0,"codec_type":"video","codec_name":"hevc","width":1280,)"
                              R"("height":720,"pix_fmt":"yuv420p","r_frame_rate":"24/1","avg_frame_rate":"24/1")" +
                              tags(video_bytes) + R"(},{"index":1,"codec_type":"audio","codec_name":"opus")" +
                              tags(audio_bytes) + "}" + extra +
                              R"(],"format":{"duration":"10.0","size":"1000000","bit_rate":"800000",)"
                              R"("format_name":"matroska,webm","tags":{"ENCODER":"Lavf63.1.101"}}})";
    return video_info_from_probe("a.mkv", probe);
}

// In a file FFmpeg wrote, the tags count where they add up to the file;
// otherwise, without a sample of the file's packets and with an audio track
// that states no rate, the video's rate is the whole file's.
TEST(Ffprobe, TheVideosRateIsItsOwnWhereTheTagsAddUp) {
    const VideoInfo tagged = ffmpeg_written("900000", "97000");  // 99.7 % of the file
    EXPECT_EQ(tagged.bit_rate, 720000);
    EXPECT_FALSE(tagged.bit_rate_estimated || tagged.bit_rate_whole_file);
    EXPECT_EQ(tagged.stream_bytes, 900000);

    const VideoInfo stale = ffmpeg_written("900000", "50000");  // 95 %: the tags are not trusted
    EXPECT_EQ(stale.bit_rate, 800000);
    EXPECT_TRUE(stale.bit_rate_whole_file);
    EXPECT_EQ(stale.stream_bytes, 0);

    // Cover art has no statistics, as an attachment has none: the tags still
    // count (it used to make them all untrusted, and the rate the file's).
    const VideoInfo with_cover = ffmpeg_written("900000", "97000", true);
    EXPECT_EQ(with_cover.bit_rate, 720000);
    EXPECT_FALSE(with_cover.bit_rate_estimated || with_cover.bit_rate_whole_file);

    // mkvmerge, writing a file FFmpeg made (its ENCODER tag kept), stores PGS
    // subtitles compressed and counts them uncompressed: with them, the tags
    // add up to more than the file (102.7 %), and still count.
    const VideoInfo compressed_subtitles = ffmpeg_written("900000", "97000", true, "30000");
    EXPECT_EQ(compressed_subtitles.bit_rate, 720000);
    EXPECT_FALSE(compressed_subtitles.bit_rate_estimated || compressed_subtitles.bit_rate_whole_file);
    // The video and audio alone more than the file (102 %): stale, not counted.
    const VideoInfo too_big = ffmpeg_written("900000", "120000", true, "30000");
    EXPECT_EQ(too_big.bit_rate, 800000);
    EXPECT_TRUE(too_big.bit_rate_whole_file);
}

// A 100 s Matroska file of 800 kb/s (10,000,000 bytes) whose video states no
// rate, with Opus audio, two subtitle tracks, a font of 100,000 bytes and
// cover art.
VideoInfo with_everything(const std::optional<PacketSample>& sample) {
    const std::string probe =
        R"({"streams":[{"index":0,"codec_type":"video","codec_name":"hevc","width":1280,"height":720,)"
        R"("r_frame_rate":"24/1","avg_frame_rate":"24/1"},)"
        R"({"index":1,"codec_type":"audio","codec_name":"opus"},)"
        R"({"index":2,"codec_type":"subtitle","codec_name":"ass"},)"
        R"({"index":3,"codec_type":"subtitle","codec_name":"ass"},)"
        R"({"index":4,"codec_type":"attachment","codec_name":"ttf","extradata_size":100000},)"
        R"({"index":5,"codec_type":"video","codec_name":"mjpeg","disposition":{"attached_pic":1}}],)"
        R"("format":{"duration":"100.0","size":"10000000","bit_rate":"800000","format_name":"matroska,webm",)"
        R"("tags":{"ENCODER":"Lavf63.1.101"}}})";
    return video_info_from_probe("a.mkv", probe, sample);
}

// The estimate takes off the font, then the container's share, then every
// other stream at its rate in the sample; cover art is no stream to take off.
TEST(Ffprobe, TheVideosEstimateIsTheFilesLessEverythingElse) {
    PacketSample sample;
    sample.bytes = {{0, 9'000'000}, {1, 1'000'000}, {2, 5000}, {3, 5000}, {5, 50000}};
    sample.overhead = 0.01;
    sample.seconds = 120;  // longer than the file: its 100 s are read
    const VideoInfo info = with_everything(sample);
    EXPECT_EQ(info.bit_rate, 703280);  // (800,000 - 100,000 x 8 / 100) x 0.99 - 80,000 - 400 - 400
    EXPECT_TRUE(info.bit_rate_estimated);
    EXPECT_FALSE(info.bit_rate_whole_file);

    sample.bytes[1] = 100'000'000;  // more than the file: no estimate
    const VideoInfo too_much = with_everything(sample);
    EXPECT_EQ(too_much.bit_rate, 800000);
    EXPECT_TRUE(too_much.bit_rate_whole_file);

    const VideoInfo unsampled = with_everything(std::nullopt);  // the Opus track's rate is not known
    EXPECT_EQ(unsampled.bit_rate, 800000);
    EXPECT_TRUE(unsampled.bit_rate_whole_file);
}

// A container with no audio holds the container's own bytes at least, so its
// rate is an estimate; a raw stream is the video and nothing else.
TEST(Ffprobe, OnlyARawStreamsFileRateIsTheVideos) {
    const auto alone_in = [](const std::string& format_name) {
        return video_info_from_probe(
            "a", R"({"streams":[{"index":0,"codec_type":"video","codec_name":"h264","width":64,"height":64,)"
                 R"("r_frame_rate":"24/1"}],"format":{"duration":"10.0","size":"125000","bit_rate":"100000",)"
                 R"("format_name":")" + format_name + R"("}})");
    };
    const VideoInfo contained = alone_in("matroska,webm");
    EXPECT_EQ(contained.bit_rate, 100000);
    EXPECT_TRUE(contained.bit_rate_estimated);
    const VideoInfo raw = alone_in("h264");
    EXPECT_EQ(raw.bit_rate, 100000);
    EXPECT_FALSE(raw.bit_rate_estimated || raw.bit_rate_whole_file);
}

// ffprobe lists a packet's index, size and place in the file, whatever order
// it is asked for, then perhaps more; the container's share is what the
// placed packets' span holds beyond all the packets' bytes, a packet without
// a place (N/A) counting in the bytes.
TEST(Ffprobe, APacketSampleCountsEachStreamsBytesAndTheContainersShare) {
    const PacketSample sample =
        packet_sample_from_csv("0,1000,0\r\n1,200,1010\n1,40,N/A\n0,500,1300,\nnot,a,packet\n", 30);
    EXPECT_EQ(sample.bytes, (std::map<int, long long>{{0, 1500}, {1, 240}}));
    EXPECT_DOUBLE_EQ(sample.overhead, 1 - 1740.0 / 1800);
    EXPECT_EQ(sample.seconds, 30);

    EXPECT_EQ(packet_sample_from_csv("0,10,0\n0,10,1000\n", 30).overhead, 0.5);  // a gap, not the container's
    EXPECT_EQ(packet_sample_from_csv("0,10,0\n0,10,N/A\n", 30).overhead, 0.0);   // one placed packet: no span
}

// About 16 MB of a file is read, at least 5 s and at most 2 minutes of it.
TEST(Ffprobe, APacketSampleReadsAbout16MB) {
    EXPECT_EQ(packet_sample_seconds(1'400'000), 96);
    EXPECT_EQ(packet_sample_seconds(8'000'000), 17);
    EXPECT_EQ(packet_sample_seconds(60'000'000), 5);
    EXPECT_EQ(packet_sample_seconds(100'000), 120);
    EXPECT_EQ(packet_sample_seconds(0), 120);  // not known
}

// As measured on streams made at known rates: what ffprobe reports of a raw
// stream told to read it at a rate no stream has.
TEST(Ffprobe, ARawStreamsFrameRateIsOnlyWhatItsHeadersStore) {
    EXPECT_DOUBLE_EQ(*stored_frame_rate("hevc", 24000.0 / 1001), 24000.0 / 1001);
    EXPECT_DOUBLE_EQ(*stored_frame_rate("vvc", 60), 60);
    EXPECT_DOUBLE_EQ(*stored_frame_rate("mpeg2video", 25), 25);  // a real 25 is not taken for FFmpeg's guess
    // Reported doubled.
    EXPECT_DOUBLE_EQ(*stored_frame_rate("h264", 60), 30);
    EXPECT_DOUBLE_EQ(*stored_frame_rate("mpeg1video", 48000.0 / 1001), 24000.0 / 1001);
    // Stored by none: the rate it was told to read at, or the demuxer's ticks.
    EXPECT_FALSE(stored_frame_rate("av1", 1200000));
    EXPECT_FALSE(stored_frame_rate("hevc", 1200000));
    EXPECT_FALSE(stored_frame_rate("hevc", 1999));
    EXPECT_FALSE(stored_frame_rate("hevc", 0));
    // Not the stream's: H.263's clock, whatever rate it was made at; or a codec not shown to store one.
    EXPECT_FALSE(stored_frame_rate("h263", 30000.0 / 1001));
    EXPECT_FALSE(stored_frame_rate("mjpeg", 25));
}

// A run saved before the rate was looked for holds FFmpeg's guess at 25, which
// must not come back as the video's rate. One whose rate was known keeps it.
TEST(Ffprobe, ASavedRawStreamKeepsItsRateButNotFfmpegsGuess) {
    VideoInfo raw;
    raw.path = "clip.hevc";
    raw.format_name = "hevc";
    raw.nb_frames = 100;
    raw.fps = raw.nominal_fps = raw.average_fps = 25;
    raw.duration = 4;
    const VideoInfo guessed = video_from(video_json(raw));
    EXPECT_TRUE(guessed.frame_rate_unknown());
    EXPECT_EQ(guessed.fps, 0);
    EXPECT_EQ(guessed.duration, 0);

    raw.set_frame_rate(50);
    const VideoInfo known = video_from(video_json(raw));
    EXPECT_FALSE(known.frame_rate_unknown());
    EXPECT_EQ(known.fps, 50);
    EXPECT_EQ(known.duration, 2);
}

}  // namespace
}  // namespace vml
