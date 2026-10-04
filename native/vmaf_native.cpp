// SPDX-License-Identifier: MIT
// GPU-native VMAF: shared, unmodified FFmpeg libraries decode and synchronize
// tiny identity tags; the corresponding AVFrames never enter a pixel pipe.
// This executable is disposable: Python owns cancellation/crash isolation.
#define NOMINMAX
#include <windows.h>
#include <cuda.h>
// The old context-creation ABI also works on drivers older than Toolkit 13.
extern "C" CUresult CUDAAPI cuCtxCreate_v2(CUcontext *, unsigned int, CUdevice);
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavfilter/avfilter.h>
#include <libavfilter/buffersrc.h>
#include <libavfilter/buffersink.h>
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_cuda.h>
#include <libavutil/pixdesc.h>
#include <libvmaf/libvmaf.h>
#include <libvmaf/libvmaf_cuda.h>
}
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

static void avcheck(int err, const char *what) {
    if (err >= 0) return;
    char message[AV_ERROR_MAX_STRING_SIZE]; av_strerror(err, message, sizeof(message));
    throw std::runtime_error(std::string(what) + ": " + message);
}
static void vmcheck(int err, const char *what) {
    if (err) throw std::runtime_error(std::string(what) + ": libvmaf " + std::to_string(err));
}
static void cucheck(CUresult err, const char *what) {
    if (err == CUDA_SUCCESS) return;
    const char *message = nullptr; cuGetErrorString(err, &message);
    throw std::runtime_error(std::string(what) + ": " + (message ? message : "CUDA error"));
}
struct FrameFree { void operator()(AVFrame *f) const { av_frame_free(&f); } };
using Frame = std::unique_ptr<AVFrame, FrameFree>;
static Frame frame() {
    Frame f(av_frame_alloc()); if (!f) throw std::bad_alloc(); return f;
}
struct Roi { int w, h, x, y; };
struct Args {
    std::string ref, test, output, ptx;
    Roi ref_roi{}, test_roi{};
    int depth = 0, subsample = 1, frames = 0;
    double start = 0, duration = 0;
    bool ref_gpu = false, test_gpu = false, blocking = false;
    std::vector<std::pair<std::string, std::string>> models;
};
static Roi roi(const std::string &s) {
    Roi r{}; char c;
    std::istringstream in(s);
    if (!(in >> r.w >> c) || c != ':' || !(in >> r.h >> c) || c != ':' ||
        !(in >> r.x >> c) || c != ':' || !(in >> r.y) || !in.eof() ||
        r.w < 32 || r.h < 32 || r.x < 0 || r.y < 0 || (r.w|r.h|r.x|r.y)&1)
        throw std::runtime_error("Invalid even-aligned crop rectangle");
    return r;
}
static Args parse(int argc, char **argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        if (i+1 == argc) throw std::runtime_error("Missing value for " + k);
        std::string v = argv[++i];
        if (k == "--reference") a.ref = v;
        else if (k == "--test") a.test = v;
        else if (k == "--output") a.output = v;
        else if (k == "--ptx") a.ptx = v;
        else if (k == "--reference-crop") a.ref_roi = roi(v);
        else if (k == "--test-crop") a.test_roi = roi(v);
        else if (k == "--depth") a.depth = std::stoi(v);
        else if (k == "--subsample") a.subsample = std::stoi(v);
        else if (k == "--frames") a.frames = std::stoi(v);
        else if (k == "--start") a.start = std::stod(v);
        else if (k == "--duration") a.duration = std::stod(v);
        else if (k == "--reference-decode") a.ref_gpu = v == "cuda";
        else if (k == "--test-decode") a.test_gpu = v == "cuda";
        else if (k == "--wait") {
            if (v != "auto" && v != "blocking") throw std::runtime_error("Invalid wait mode");
            a.blocking = v == "blocking";
        } else if (k == "--model") {
            auto p = v.find('=');
            if (p == std::string::npos) throw std::runtime_error("Model needs name=version");
            auto name = v.substr(0, p), version = v.substr(p+1);
            if ((name != "vmaf" && name != "vmaf_neg") ||
                (version != "vmaf_v0.6.1" && version != "vmaf_4k_v0.6.1" && version != "vmaf_v0.6.1neg"))
                throw std::runtime_error("Unsupported GPU model");
            a.models.emplace_back(name, version);
        } else throw std::runtime_error("Unknown argument " + k);
    }
    if (a.ref.empty() || a.test.empty() || a.output.empty() || a.ptx.empty() || a.models.empty() ||
        a.ref_roi.w != a.test_roi.w || a.ref_roi.h != a.test_roi.h || !a.ref_roi.w ||
        (a.depth != 8 && a.depth != 10) || a.subsample < 1 || a.frames < 0 ||
        !std::isfinite(a.start) || !std::isfinite(a.duration) || a.start < 0 || a.duration < 0)
        throw std::runtime_error("Invalid comparison arguments");
    return a;
}

class Cuda {
public:
    CUcontext context = nullptr;
    AVBufferRef *device = nullptr;
    CUmodule module = nullptr;
    CUfunction prepare = nullptr;
    explicit Cuda(const Args &a) {
        try {
            cucheck(cuInit(0), "Initializing CUDA"); CUdevice gpu;
            cucheck(cuDeviceGet(&gpu, 0), "Selecting GPU");
            cucheck(cuCtxCreate_v2(&context, a.blocking ? CU_CTX_SCHED_BLOCKING_SYNC : CU_CTX_SCHED_AUTO, gpu),
                    "Creating CUDA context");
            // Both decoders and libvmaf use this exact context. No download,
            // graphics interop or second device upload on a hardware input.
            avcheck(av_hwdevice_ctx_create(&device, AV_HWDEVICE_TYPE_CUDA, "0", nullptr,
                                           AV_CUDA_USE_CURRENT_CONTEXT), "Sharing CUDA context");
            // cuModuleLoad's narrow filename is not reliably UTF-8 on
            // Windows. Read through a wide filesystem path, then load data.
            std::ifstream kernel(std::filesystem::u8path(a.ptx), std::ios::binary);
            if (!kernel) throw std::runtime_error("Cannot open luma kernel");
            std::string ptx((std::istreambuf_iterator<char>(kernel)), std::istreambuf_iterator<char>());
            cucheck(cuModuleLoadData(&module, ptx.c_str()), "Loading luma kernel");
            cucheck(cuModuleGetFunction(&prepare, module, "prepare_luma"), "Loading luma entry point");
        } catch (...) { close(); throw; }
    }
    void close() {
        if (module) { cuModuleUnload(module); module = nullptr; }
        av_buffer_unref(&device);
        if (context) { cuCtxDestroy(context); context = nullptr; }
    }
    ~Cuda() { close(); }
};

class Decoder {
    AVFormatContext *format = nullptr;
    AVCodecContext *codec = nullptr;
    AVPacket *packet = nullptr;
    bool draining = false, gpu;
    int index = -1;
    int64_t first_pts = AV_NOPTS_VALUE, minimum_pts = INT64_MIN;
public:
    AVRational time_base{}, fps{};
    explicit Decoder(const std::string &path, bool hardware, Cuda &cuda, double start) : gpu(hardware) {
        try {
            avcheck(avformat_open_input(&format, path.c_str(), nullptr, nullptr), "Opening video");
            avcheck(avformat_find_stream_info(format, nullptr), "Reading stream info");
            const AVCodec *impl = nullptr;
            index = av_find_best_stream(format, AVMEDIA_TYPE_VIDEO, -1, -1, &impl, 0);
            avcheck(index, "Finding video stream");
            AVStream *stream = format->streams[index]; time_base = stream->time_base;
            if (av_packet_side_data_get(stream->codecpar->coded_side_data, stream->codecpar->nb_coded_side_data,
                                        AV_PKT_DATA_DISPLAYMATRIX))
                throw std::runtime_error("Display-matrix inputs require the existing auto-rotation path");
            fps = av_guess_frame_rate(format, stream, nullptr);
            if (fps.num <= 0 || fps.den <= 0) throw std::runtime_error("Missing frame rate");
            codec = avcodec_alloc_context3(impl); if (!codec) throw std::bad_alloc();
            avcheck(avcodec_parameters_to_context(codec, stream->codecpar), "Configuring decoder");
            codec->pkt_timebase = time_base;
            // CPU VVC needs automatic threading. Hardware decode does not
            // need a CPU frame-thread pool: that retains many extra 4K
            // surfaces/contexts for only a small throughput improvement.
            codec->thread_count = gpu ? 1 : 0;
            if (gpu) {
                codec->hw_device_ctx = av_buffer_ref(cuda.device);
                codec->get_format = [](AVCodecContext *, const AVPixelFormat *formats) {
                    for (auto p = formats; *p != AV_PIX_FMT_NONE; ++p)
                        if (*p == AV_PIX_FMT_CUDA) return *p;
                    return AV_PIX_FMT_NONE;  // Never silently advertise software as GPU.
                };
            }
            avcheck(avcodec_open2(codec, impl, nullptr), "Opening decoder");
            packet = av_packet_alloc(); if (!packet) throw std::bad_alloc();
            if (start > 0) {
                // CLI input -ss is relative to the container's start, not
                // the video stream's start. VVC reorder delay can put the
                // latter hundreds of milliseconds after the audio stream.
                const int64_t origin = format->start_time == AV_NOPTS_VALUE ? 0 :
                    av_rescale_q(format->start_time, AV_TIME_BASE_Q, time_base);
                minimum_pts = origin + av_rescale_q(static_cast<int64_t>(start * AV_TIME_BASE), AV_TIME_BASE_Q, time_base);
                avcheck(avformat_seek_file(format, index, INT64_MIN, minimum_pts, minimum_pts, 0), "Seeking video");
                avcodec_flush_buffers(codec);
            }
            std::cerr << (hardware ? "GPU" : "CPU") << " decode: " << impl->name << "\n";
        } catch (...) { close(); throw; }
    }
    void close() { av_packet_free(&packet); avcodec_free_context(&codec); avformat_close_input(&format); }
    ~Decoder() { close(); }
    Frame next() {
        auto f = frame();
        while (true) {
            int e = avcodec_receive_frame(codec, f.get());
            if (!e) {
                int64_t pts = f->best_effort_timestamp;
                if (pts == AV_NOPTS_VALUE) throw std::runtime_error("Missing frame timestamp");
                if (pts < minimum_pts) { av_frame_unref(f.get()); continue; }
                if (first_pts == AV_NOPTS_VALUE) {
                    first_pts = pts;
                    std::cerr << "First retained PTS: " << pts << " (" << pts * av_q2d(time_base) << " s)\n";
                }
                f->pts = pts - first_pts;
                if (gpu && f->format != AV_PIX_FMT_CUDA) throw std::runtime_error("Hardware decoder returned CPU frame");
                return f;
            }
            if (e == AVERROR_EOF) return {};
            avcheck(e == AVERROR(EAGAIN) ? 0 : e, "Decoding video");
            if (draining) throw std::runtime_error("Decoder requested packets after flush");
            do {
                av_packet_unref(packet);
                e = av_read_frame(format, packet);
            } while (e >= 0 && packet->stream_index != index);
            if (e == AVERROR_EOF) { draining = true; avcheck(avcodec_send_packet(codec, nullptr), "Flushing decoder"); }
            else { avcheck(e, "Reading video packet"); avcheck(avcodec_send_packet(codec, packet), "Submitting packet"); }
        }
    }
};

class Pairer {
    AVFilterGraph *graph = nullptr;
    AVFilterContext *inputs[2]{}, *sink = nullptr;
    Decoder &test, &ref;
    std::map<uint64_t, Frame> held[2];
    uint64_t serial[2]{1, 1};
    bool ended[2]{};
    void feed(int i) {
        auto &decoder = i ? ref : test;
        auto f = decoder.next();
        if (!f) { ended[i] = true; avcheck(av_buffersrc_add_frame_flags(inputs[i], nullptr, 0), "Ending frame tags"); return; }
        if (held[i].size() >= 64) throw std::runtime_error("Frame matching exceeded bounded lookahead");
        auto tag = frame(); tag->format = AV_PIX_FMT_YUV444P; tag->width = 8; tag->height = 2;
        tag->pts = f->pts; tag->sample_aspect_ratio = {1,1};
        avcheck(av_frame_get_buffer(tag.get(), 32), "Allocating identity tag");
        for (int p = 0; p < 3; ++p)
            for (int y = 0; y < 2; ++y) memset(tag->data[p] + y * tag->linesize[p], p ? 128 : 0, 8);
        uint64_t id = serial[i]++; memcpy(tag->data[0], &id, sizeof(id));
        held[i].emplace(id, std::move(f));
        avcheck(av_buffersrc_add_frame_flags(inputs[i], tag.get(), AV_BUFFERSRC_FLAG_KEEP_REF), "Submitting identity tag");
    }
public:
    Pairer(Decoder &t, Decoder &r) : test(t), ref(r) {
        try {
            graph = avfilter_graph_alloc(); if (!graph) throw std::bad_alloc();
            graph->nb_threads = 1;
            for (int i = 0; i < 2; ++i) {
                auto &d = i ? r : t; std::ostringstream config;
                config << "video_size=8x2:pix_fmt=" << AV_PIX_FMT_YUV444P << ":time_base="
                       << d.time_base.num << '/' << d.time_base.den << ":pixel_aspect=1/1:frame_rate="
                       << d.fps.num << '/' << d.fps.den;
                avcheck(avfilter_graph_create_filter(&inputs[i], avfilter_get_by_name("buffer"),
                         i ? "ref" : "test", config.str().c_str(), nullptr, graph), "Creating tag source");
            }
            avcheck(avfilter_graph_create_filter(&sink, avfilter_get_by_name("buffersink"), "out", nullptr,
                                                 nullptr, graph), "Creating tag sink");
            AVFilterInOut *in = avfilter_inout_alloc(), *out = avfilter_inout_alloc();
            if (!in || !out) { avfilter_inout_free(&in); avfilter_inout_free(&out); throw std::bad_alloc(); }
            in->name = av_strdup("out"); in->filter_ctx = sink;
            out->name = av_strdup("test"); out->filter_ctx = inputs[0];
            out->next = avfilter_inout_alloc();
            if (!out->next) { avfilter_inout_free(&in); avfilter_inout_free(&out); throw std::bad_alloc(); }
            out->next->name = av_strdup("ref"); out->next->filter_ctx = inputs[1];
            int e = avfilter_graph_parse_ptr(graph,
                "[test]pad=16:2:0:0[canvas];[canvas][ref]overlay=x=8:y=0:format=yuv444:"
                "shortest=1:repeatlast=0:ts_sync_mode=nearest[out]", &in, &out, nullptr);
            avfilter_inout_free(&in); avfilter_inout_free(&out);
            avcheck(e, "Building timestamp matcher");
            avcheck(avfilter_graph_config(graph, nullptr), "Configuring timestamp matcher");
        } catch (...) { avfilter_graph_free(&graph); throw; }
    }
    ~Pairer() { avfilter_graph_free(&graph); }
    bool next(AVFrame *&t, AVFrame *&r, double &seconds) {
        auto tag = frame();
        while (true) {
            int e = av_buffersink_get_frame(sink, tag.get());
            if (e == AVERROR_EOF) return false;
            if (!e) break;
            avcheck(e == AVERROR(EAGAIN) ? 0 : e, "Matching timestamps");
            bool fed = false;
            for (int i = 0; i < 2; ++i) {
                if (!ended[i] && av_buffersrc_get_nb_failed_requests(inputs[i])) { feed(i); fed = true; }
            }
            if (!fed) throw std::runtime_error("Timestamp matcher made no progress");
        }
        uint64_t ids[2]; memcpy(&ids[0], tag->data[0], 8); memcpy(&ids[1], tag->data[0] + 8, 8);
        for (int i = 0; i < 2; ++i) {
            if (!held[i].count(ids[i])) throw std::runtime_error("Frame tag was altered during matching");
            held[i].erase(held[i].begin(), held[i].lower_bound(ids[i]));
        }
        t = held[0].at(ids[0]).get(); r = held[1].at(ids[1]).get();
        seconds = tag->pts * av_q2d(av_buffersink_get_time_base(sink));
        return true;
    }
};

class Scorer {
    VmafContext *vmaf = nullptr;
    std::vector<VmafModel *> models;
    Cuda &cuda;
    const Args &args;
    void fill(VmafPicture &pic, AVFrame *f, Roi crop) {
        if (crop.x + crop.w > f->width || crop.y + crop.h > f->height)
            throw std::runtime_error("Crop exceeds decoded frame");
        AVPixelFormat fmt = static_cast<AVPixelFormat>(f->format);
        const bool gpu = fmt == AV_PIX_FMT_CUDA;
        if (gpu) {
            if (!f->hw_frames_ctx) throw std::runtime_error("Missing GPU frame context");
            fmt = static_cast<AVHWFramesContext *>(reinterpret_cast<void *>(f->hw_frames_ctx->data))->sw_format;
        }
        const AVPixFmtDescriptor *desc = av_pix_fmt_desc_get(fmt);
        const int bytes = args.depth == 8 ? 1 : 2;
        if (!desc || desc->comp[0].depth != args.depth || desc->comp[0].plane != 0 ||
            desc->comp[0].step != bytes || desc->comp[0].offset || (desc->flags & (AV_PIX_FMT_FLAG_RGB | AV_PIX_FMT_FLAG_BE)))
            throw std::runtime_error("Unsupported decoded luma layout");
        const int shift = desc->comp[0].shift;
        if (f->linesize[0] <= 0) throw std::runtime_error("Negative decoded stride is unsupported");
        auto src = f->data[0] + static_cast<size_t>(crop.y) * f->linesize[0] + static_cast<size_t>(crop.x) * bytes;
        if (gpu) {
            CUdeviceptr s = reinterpret_cast<CUdeviceptr>(src), d = reinterpret_cast<CUdeviceptr>(pic.data[0]);
            unsigned long long sp = f->linesize[0], dp = pic.stride[0];
            void *params[] = {&s, &sp, &d, &dp, &crop.w, &crop.h, const_cast<int *>(&bytes), const_cast<int *>(&shift)};
            cucheck(cuLaunchKernel(cuda.prepare, (crop.w+31)/32, (crop.h+7)/8, 1, 32, 8, 1, 0, nullptr,
                                   params, nullptr), "Preparing GPU luma");
            // Source AVFrame must stay alive until the kernel has read it;
            // libvmaf then owns and schedules the destination device picture.
            cucheck(cuStreamSynchronize(nullptr), "Waiting for luma kernel");
        } else {
            if (shift) throw std::runtime_error("Shifted software luma layout is unsupported");
            CUDA_MEMCPY2D copy{}; copy.srcMemoryType = CU_MEMORYTYPE_HOST; copy.srcHost = src;
            copy.srcPitch = f->linesize[0]; copy.dstMemoryType = CU_MEMORYTYPE_DEVICE;
            copy.dstDevice = reinterpret_cast<CUdeviceptr>(pic.data[0]); copy.dstPitch = pic.stride[0];
            copy.WidthInBytes = crop.w * bytes; copy.Height = crop.h;
            cucheck(cuMemcpy2D(&copy), "Uploading software-decoded luma");
        }
    }
public:
    unsigned count = 0;
    Scorer(Cuda &c, const Args &a) : cuda(c), args(a) {
        try {
            VmafConfiguration cfg{}; cfg.log_level = VMAF_LOG_LEVEL_ERROR; cfg.n_subsample = a.subsample;
            vmcheck(vmaf_init(&vmaf, cfg), "Starting libvmaf");
            VmafCudaState *state = nullptr; VmafCudaConfiguration ccfg{}; ccfg.cu_ctx = c.context;
            vmcheck(vmaf_cuda_state_init(&state, ccfg), "Sharing libvmaf CUDA context");
            vmcheck(vmaf_cuda_import_state(vmaf, state), "Importing CUDA state");
            for (const auto &m : a.models) {
                VmafModel *model = nullptr; VmafModelConfig mc{}; mc.name = m.first.c_str();
                vmcheck(vmaf_model_load(&model, &mc, m.second.c_str()), "Loading model"); models.push_back(model);
                vmcheck(vmaf_use_features_from_model(vmaf, model), "Selecting GPU features");
            }
            VmafCudaPictureConfiguration pictures{};
            pictures.pic_params = {static_cast<unsigned>(a.ref_roi.w), static_cast<unsigned>(a.ref_roi.h),
                                   static_cast<unsigned>(a.depth), VMAF_PIX_FMT_YUV420P};
            pictures.pic_prealloc_method = VMAF_CUDA_PICTURE_PREALLOCATION_METHOD_DEVICE;
            vmcheck(vmaf_cuda_preallocate_pictures(vmaf, pictures), "Allocating device picture pool");
        } catch (...) { close(); throw; }
    }
    void close() {
        for (auto m : models) vmaf_model_destroy(m); models.clear();
        if (vmaf) { vmaf_close(vmaf); vmaf = nullptr; }
    }
    ~Scorer() { close(); }
    void add(AVFrame *r, AVFrame *t) {
        VmafPicture ref{}, test{};
        vmcheck(vmaf_cuda_fetch_preallocated_picture(vmaf, &ref), "Taking reference picture");
        try {
            vmcheck(vmaf_cuda_fetch_preallocated_picture(vmaf, &test), "Taking test picture");
            fill(ref, r, args.ref_roi); fill(test, t, args.test_roi);
        } catch (...) {
            vmaf_picture_unref(&ref); if (test.ref) vmaf_picture_unref(&test); throw;
        }
        // Patched libvmaf takes ownership on success AND failure (PR 1652).
        vmcheck(vmaf_read_pictures(vmaf, &ref, &test, count), "Scoring pictures"); ++count;
    }
    void finish() {
        if (!count) throw std::runtime_error("No frame pairs scored");
        vmcheck(vmaf_read_pictures(vmaf, nullptr, nullptr, 0), "Flushing GPU features");
        // Force predictions into the collector before writing its JSON log.
        for (auto m : models) for (unsigned i = 0; i < count; i += args.subsample) {
            double score; vmcheck(vmaf_score_at_index(vmaf, m, &score, i), "Reading model score");
        }
        vmcheck(vmaf_write_output(vmaf, args.output.c_str(), VMAF_OUTPUT_FORMAT_JSON), "Writing score log");
    }
};

static int run(int argc, char **argv) {
    if (argc == 2 && std::string(argv[1]) == "--version") {
        std::cout << "Native VMAF helper 1; FFmpeg " << av_version_info() << "; libvmaf " << vmaf_version() << "\n";
        return 0;
    }
    auto a = parse(argc, argv); av_log_set_level(AV_LOG_ERROR);
    Cuda cuda(a); Decoder test(a.test, a.test_gpu, cuda, a.start), ref(a.ref, a.ref_gpu, cuda, a.start);
    Pairer pairer(test, ref); Scorer scorer(cuda, a);
    auto begin = std::chrono::steady_clock::now(), last = begin;
    AVFrame *t = nullptr, *r = nullptr; double timestamp = 0;
    auto progress = [&]() {
        auto now = std::chrono::steady_clock::now();
        double seconds = std::chrono::duration<double>(now - begin).count();
        std::cout << "fps=" << scorer.count / seconds << "\nframe=" << scorer.count << "\n" << std::flush;
        last = now;
    };
    while (pairer.next(t, r, timestamp)) {
        // Match the old rawvideo -t boundary (libvmaf's graph emits a pair
        // at the duration boundary, while ffmpeg's raw output excludes it).
        if (a.duration > 0 && timestamp >= a.duration) break;
        scorer.add(r, t);
        if (scorer.count == 1 || std::chrono::steady_clock::now() - last > std::chrono::milliseconds(200)) progress();
        if (a.frames && scorer.count >= static_cast<unsigned>(a.frames)) break;
    }
    scorer.finish(); progress(); std::cout << "progress=end\n" << std::flush;
    return 0;
}
// Python passes Windows UTF-16 arguments; FFmpeg's filenames require UTF-8.
int wmain(int argc, wchar_t **wide) {
    try {
        std::vector<std::string> strings; std::vector<char *> argv;
        for (int i = 0; i < argc; ++i) {
            int n = WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, wide[i], -1, nullptr, 0, nullptr, nullptr);
            if (!n) throw std::runtime_error("Invalid Unicode argument");
            std::string s(n, '\0'); WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, wide[i], -1, s.data(), n, nullptr, nullptr);
            s.pop_back(); strings.push_back(std::move(s));
        }
        for (auto &s : strings) argv.push_back(s.data());
        return run(argc, argv.data());
    } catch (const std::exception &e) { std::cerr << "Native GPU VMAF: " << e.what() << "\n"; return 1; }
}
