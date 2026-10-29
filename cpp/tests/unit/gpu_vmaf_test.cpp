// VMAF on the GPU's own parts: which runs it takes, and the reading of
// FFmpeg's frames from a pipe as YUV4MPEG.
#include "vml/gpu_vmaf.h"

#include <gtest/gtest.h>

#include <fstream>
#include <future>
#include <string>
#include <thread>
#include <vector>

namespace vml {
namespace {

// Writes `bytes` to the pipe at `path` as FFmpeg does, opening it as a file.
std::thread write_to(const std::string& path, std::string bytes) {
    return std::thread([path, bytes = std::move(bytes)] {
        std::ofstream pipe(path, std::ios::binary);
        ASSERT_TRUE(pipe.is_open());
        pipe.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    });
}

// A YUV4MPEG stream as FFmpeg's yuv4mpegpipe writes it, of 4x2 8-bit 4:2:0
// frames (12 bytes each), the first byte of frame N being N.
std::string stream_of(int frames) {
    std::string stream = "YUV4MPEG2 W4 H2 F120:1 Ip A1:1 C420jpeg XYSCSS=420JPEG\n";
    for (int n = 0; n < frames; ++n) {
        stream += "FRAME\n";
        stream += static_cast<char>(n);
        stream += std::string(11, '\x7f');
    }
    return stream;
}

std::string read_error(StreamReader& reader) {
    try {
        while (std::vector<std::uint8_t>* frame = reader.pull()) {
            reader.release(frame);
        }
    } catch (const LibvmafError& error) {
        return error.what();
    }
    return "";
}

// The frames come out in order, each whole; then the end. A pipe FFmpeg
// never opens (an output that got no frame) ends when FFmpeg has.
TEST(GpuVmaf, ReadsFfmpegsFramesFromAPipe) {
    StreamReader reader("test", 12);
    reader.start();
    std::thread writer = write_to(reader.path(), stream_of(5));
    for (int n = 0; n < 5; ++n) {
        std::vector<std::uint8_t>* frame = reader.pull();
        ASSERT_NE(frame, nullptr) << "frame " << n;
        ASSERT_EQ(frame->size(), 12u);
        EXPECT_EQ((*frame)[0], n);
        EXPECT_EQ((*frame)[11], 0x7f);
        reader.release(frame);
    }
    writer.join();
    reader.ffmpeg_ended();
    EXPECT_EQ(reader.pull(), nullptr);
    EXPECT_EQ(reader.pull(), nullptr);  // and stays at the end
    reader.drop_rest();
    EXPECT_NO_THROW(reader.join());

    StreamReader unopened("unopened", 12);
    unopened.start();
    unopened.ffmpeg_ended();
    EXPECT_EQ(unopened.pull(), nullptr);
    EXPECT_NO_THROW(unopened.join());
}

// A stream that is not what FFmpeg writes is an error, not frames.
TEST(GpuVmaf, AStreamThatIsNotFfmpegsIsAnError) {
    const std::vector<std::pair<std::string, std::string>> cases{
        {"YUV4MPEG3 W4 H2\nFRAME\n", "FFmpeg's frames are not a YUV4MPEG stream"},
        {stream_of(1) + "FRAMES", "FFmpeg's frames are not where they were expected"},
        {stream_of(1) + "FRAME\n" + std::string(5, 'x'), "FFmpeg's frames ended inside one (5 of 12 bytes)"},
        {stream_of(2) + "FRAME\n", "FFmpeg's frames ended after a frame's start"},
    };
    for (const auto& [stream, message] : cases) {
        StreamReader reader("test", 12);
        reader.start();
        // As a run does once FFmpeg has gone: a frame cut short is then
        // known to stay short, not waited for.
        write_to(reader.path(), stream).join();
        reader.ffmpeg_ended();
        EXPECT_EQ(read_error(reader), message);
        reader.stop();
    }
}

// A pipe stopped while FFmpeg could still write into it was closed by the
// app, whose closing then fails that FFmpeg; one whose FFmpeg had finished
// was not (run_ffmpeg_pair takes the cause of a failure by it).
TEST(GpuVmaf, KnowsWhichPipesTheAppClosed) {
    StreamReader finished("finished", 12);
    finished.start();
    write_to(finished.path(), stream_of(1)).join();
    finished.ffmpeg_ended();
    EXPECT_EQ(read_error(finished), "");  // read to its end
    finished.stop();
    EXPECT_FALSE(finished.stopped_while_reading());

    StreamReader writing("writing", 12);
    writing.start();
    std::promise<void> closed;
    std::thread writer([path = writing.path(), done = closed.get_future()] {
        std::ofstream pipe(path, std::ios::binary);
        const std::string frame = stream_of(1);
        pipe.write(frame.data(), static_cast<std::streamsize>(frame.size()));
        pipe.flush();
        done.wait();  // holds the pipe open, as an FFmpeg still decoding does
    });
    EXPECT_NE(writing.pull(), nullptr);  // the frame written: the pipe is being read
    writing.stop();
    EXPECT_TRUE(writing.stopped_while_reading());
    closed.set_value();
    writer.join();
    EXPECT_NO_THROW(writing.join());
}

// VMAF and NEG go to the GPU with the app's own models; a custom model keeps
// both on the CPU, as do a comparison deeper than 10 bits, an odd size, and
// VMAF set to the CPU.
TEST(GpuVmaf, ScoresOnTheGpuWhatThePythonAppDoes) {
    EXPECT_EQ(gpu_models(true, true, "version=vmaf_v0.6.1"),
              (GpuModels{{"vmaf", "vmaf_v0.6.1"}, {"vmaf_neg", "vmaf_v0.6.1neg"}}));
    EXPECT_EQ(gpu_models(true, false, "version=vmaf_4k_v0.6.1"), (GpuModels{{"vmaf", "vmaf_4k_v0.6.1"}}));
    EXPECT_EQ(gpu_models(false, true, ""), (GpuModels{{"vmaf_neg", "vmaf_v0.6.1neg"}}));
    EXPECT_TRUE(gpu_models(true, true, "path=C:/models/mine.json").empty());
    EXPECT_TRUE(gpu_models(false, false, "version=vmaf_v0.6.1").empty());

    const std::string model = "version=vmaf_v0.6.1";
    EXPECT_TRUE(scores_on_gpu(true, true, model, false, 8, std::pair(1920, 1080), CancelToken()).empty());
    EXPECT_TRUE(scores_on_gpu(true, true, model, true, 12, std::pair(1920, 1080), CancelToken()).empty());
    EXPECT_TRUE(scores_on_gpu(true, true, model, true, 10, std::pair(1919, 1080), CancelToken()).empty());
    EXPECT_TRUE(scores_on_gpu(true, true, model, true, 10, std::pair(1920, 1079), CancelToken()).empty());
    EXPECT_TRUE(scores_on_gpu(true, true, "path=C:/models/mine.json", true, 8, std::nullopt, CancelToken()).empty());
}

}  // namespace
}  // namespace vml
