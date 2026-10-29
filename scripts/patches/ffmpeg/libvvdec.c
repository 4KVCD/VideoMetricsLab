/*
 * H.266/VVC decoding with Fraunhofer HHI's VVdeC (github.com/fraunhoferhhi/vvdec)
 *
 * VideoMetricsLab's addition to its own FFmpeg build (scripts/build_ffmpeg_decoders.ps1
 * copies this file into libavcodec and patches configure, allcodecs.c and the Makefile
 * with scripts/patches/ffmpeg/libvvdec.patch). FFmpeg's own VVC decoder stays in the
 * build: this one is asked for by name ("libvvdec").
 *
 * VVdeC adds the film grain an H.274 film grain characteristics SEI describes (both of
 * its models, at 8 and 10 bits: all it decodes), which FFmpeg's own VVC decoder adds
 * only at 8 bits and for model 0; setting AV_CODEC_EXPORT_DATA_FILM_GRAIN turns it off.
 * VVdeC does not add AOM's AFGS1 grain.
 *
 * This file is part of FFmpeg.
 *
 * FFmpeg is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * FFmpeg is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with FFmpeg; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
 */

#include <stdarg.h>
#include <string.h>

#include <vvdec/vvdec.h>

#include "libavutil/imgutils.h"
#include "libavutil/opt.h"
#include "libavutil/pixdesc.h"

#include "avcodec.h"
#include "codec_internal.h"
#include "decode.h"

// VVdeC's version and commit, which the build script writes (vvdec_get_version() says
// "3.3.0-dev" for every commit of master).
#include "libvvdec_build.h"

typedef struct LibvvdecContext {
    const AVClass *class;
    vvdecDecoder *decoder;
    vvdecAccessUnit *access_unit;
    AVPacket *packet;
    int draining;      // the input has ended: the pictures still inside are taken out
    int threads;       // AVOption: VVdeC's threads (0: as many as the CPU has)
    int parse_delay;   // AVOption: access units parsed ahead (-1: VVdeC's default)
} LibvvdecContext;

// VVdeC's messages go to FFmpeg's log. Only an error is logged as one: the app's decoder
// library takes any error FFmpeg logs for a failed decode. VVdeC keeps one callback for
// the whole process (each decoder sets this same one), so it logs without a context.
static void libvvdec_log(void *opaque, int level, const char *format, va_list arguments)
{
    const int av_level = level <= VVDEC_ERROR   ? AV_LOG_ERROR
                         : level == VVDEC_WARNING ? AV_LOG_WARNING
                         : level == VVDEC_INFO    ? AV_LOG_VERBOSE
                                                  : AV_LOG_DEBUG;
    av_vlog(NULL, av_level, format, arguments);
}

static int libvvdec_open_decoder(AVCodecContext *avctx)
{
    LibvvdecContext *s = avctx->priv_data;
    vvdecParams params;

    vvdec_params_default(&params);
    params.threads = s->threads > 0 ? s->threads : avctx->thread_count > 0 ? avctx->thread_count : -1;
    params.parseDelay = s->parse_delay;
    params.logLevel = VVDEC_WARNING;
    params.verifyPictureHash = false;
    params.filmGrainSynthesis = !(avctx->export_side_data & AV_CODEC_EXPORT_DATA_FILM_GRAIN);

    s->decoder = vvdec_decoder_open(&params);
    if (!s->decoder) {
        av_log(avctx, AV_LOG_ERROR, "VVdeC could not start\n");
        return AVERROR_EXTERNAL;
    }
    vvdec_set_logging_callback(s->decoder, libvvdec_log);
    s->draining = 0;
    return 0;
}

static av_cold int libvvdec_init(AVCodecContext *avctx)
{
    LibvvdecContext *s = avctx->priv_data;

    av_log(avctx, AV_LOG_VERBOSE, "VVdeC %s (%s)\n", vvdec_get_version(), LIBVVDEC_BUILD);
    s->packet = av_packet_alloc();
    s->access_unit = vvdec_accessUnit_alloc();
    if (!s->packet || !s->access_unit)
        return AVERROR(ENOMEM);
    return libvvdec_open_decoder(avctx);
}

static void libvvdec_close_decoder(LibvvdecContext *s)
{
    if (s->decoder) {
        vvdec_decoder_close(s->decoder);
        s->decoder = NULL;
    }
}

static av_cold int libvvdec_close(AVCodecContext *avctx)
{
    LibvvdecContext *s = avctx->priv_data;

    libvvdec_close_decoder(s);
    if (s->access_unit) {
        s->access_unit->payload = NULL;  // the packet's memory, not VVdeC's
        vvdec_accessUnit_free(s->access_unit);
        s->access_unit = NULL;
    }
    av_packet_free(&s->packet);
    return 0;
}

// A seek: VVdeC cannot drop what it holds without decoding it (vvdec_flush decodes every
// access unit still parsed ahead), so a new decoder is started.
static void libvvdec_flush(AVCodecContext *avctx)
{
    LibvvdecContext *s = avctx->priv_data;

    libvvdec_close_decoder(s);
    av_packet_unref(s->packet);
    if (libvvdec_open_decoder(avctx) < 0)
        av_log(avctx, AV_LOG_ERROR, "VVdeC could not start again after a seek\n");
}

static enum AVPixelFormat libvvdec_pixel_format(const vvdecFrame *picture)
{
    static const enum AVPixelFormat formats[4][2] = {
        { AV_PIX_FMT_GRAY8,   AV_PIX_FMT_GRAY10     },
        { AV_PIX_FMT_YUV420P, AV_PIX_FMT_YUV420P10 },
        { AV_PIX_FMT_YUV422P, AV_PIX_FMT_YUV422P10 },
        { AV_PIX_FMT_YUV444P, AV_PIX_FMT_YUV444P10 },
    };

    if (picture->colorFormat < VVDEC_CF_YUV400_PLANAR || picture->colorFormat > VVDEC_CF_YUV444_PLANAR ||
        (picture->bitDepth != 8 && picture->bitDepth != 10))
        return AV_PIX_FMT_NONE;
    return formats[picture->colorFormat][picture->bitDepth == 10];
}

// The sample aspect ratios aspect_ratio_idc 1 to 16 name (ITU-T H.273, Table 7).
static const AVRational libvvdec_aspect_ratios[17] = {
    { 0, 1 }, { 1, 1 }, { 12, 11 }, { 10, 11 }, { 16, 11 }, { 40, 33 }, { 24, 11 }, { 20, 11 }, { 32, 11 },
    { 80, 33 }, { 18, 11 }, { 15, 11 }, { 64, 33 }, { 160, 99 }, { 4, 3 }, { 3, 2 }, { 2, 1 },
};

// The stream's colour description and sample aspect ratio, from the picture's VUI.
static void libvvdec_take_vui(AVCodecContext *avctx, const vvdecVui *vui)
{
    if (vui->colourDescriptionPresentFlag) {
        avctx->color_primaries = vui->colourPrimaries;
        avctx->color_trc = vui->transferCharacteristics;
        avctx->colorspace = vui->matrixCoefficients;
    }
    if (vui->videoSignalTypePresentFlag)
        avctx->color_range = vui->videoFullRangeFlag ? AVCOL_RANGE_JPEG : AVCOL_RANGE_MPEG;
    if (vui->chromaLocInfoPresentFlag)
        avctx->chroma_sample_location = vui->chromaSampleLocType + 1;
    if (vui->aspectRatioInfoPresentFlag) {
        AVRational sar = { 0, 1 };
        if (vui->aspectRatioIdc == 255)
            sar = (AVRational){ vui->sarWidth, vui->sarHeight };
        else if (vui->aspectRatioIdc > 0 && vui->aspectRatioIdc < 17)
            sar = libvvdec_aspect_ratios[vui->aspectRatioIdc];
        ff_set_sar(avctx, sar);
    }
}

// Copies VVdeC's picture into `frame`, from FFmpeg's buffer pool.
static int libvvdec_output(AVCodecContext *avctx, const vvdecFrame *picture, AVFrame *frame)
{
    const enum AVPixelFormat format = libvvdec_pixel_format(picture);
    const int bytes_per_sample = picture->bitDepth > 8 ? 2 : 1;
    int ret;

    if (format == AV_PIX_FMT_NONE) {
        av_log(avctx, AV_LOG_ERROR, "VVdeC gave a picture of %u bits, colour format %d\n", picture->bitDepth,
               picture->colorFormat);
        return AVERROR_PATCHWELCOME;
    }
    avctx->pix_fmt = format;
    if (picture->picAttributes && picture->picAttributes->vui)
        libvvdec_take_vui(avctx, picture->picAttributes->vui);
    if (picture->picAttributes && picture->picAttributes->seqInfo)
        avctx->has_b_frames = picture->picAttributes->seqInfo->maxNumReorderPics;
    if (avctx->width != (int)picture->width || avctx->height != (int)picture->height) {
        ret = ff_set_dimensions(avctx, picture->width, picture->height);
        if (ret < 0)
            return ret;
    }

    ret = ff_get_buffer(avctx, frame, 0);
    if (ret < 0)
        return ret;
    for (uint32_t plane = 0; plane < picture->numPlanes && plane < 3; plane++) {
        const vvdecPlane *in = &picture->planes[plane];
        if (in->bytesPerSample != (uint32_t)bytes_per_sample)
            return AVERROR_BUG;
        av_image_copy_plane(frame->data[plane], frame->linesize[plane], in->ptr, in->stride,
                            in->width * bytes_per_sample, in->height);
    }

    frame->pts = picture->ctsValid ? (int64_t)picture->cts : AV_NOPTS_VALUE;
    frame->pkt_dts = AV_NOPTS_VALUE;
    frame->color_primaries = avctx->color_primaries;
    frame->color_trc = avctx->color_trc;
    frame->colorspace = avctx->colorspace;
    frame->color_range = avctx->color_range;
    frame->chroma_location = avctx->chroma_sample_location;
    frame->sample_aspect_ratio = avctx->sample_aspect_ratio;
    if (picture->picAttributes) {
        const vvdecPicAttributes *attributes = picture->picAttributes;
        // As FFmpeg's own VVC decoder: an IDR or CRA picture is a key frame.
        if (attributes->nalType >= VVC_NAL_UNIT_CODED_SLICE_IDR_W_RADL &&
            attributes->nalType <= VVC_NAL_UNIT_CODED_SLICE_CRA)
            frame->flags |= AV_FRAME_FLAG_KEY;
        frame->pict_type = attributes->sliceType == VVDEC_SLICETYPE_I   ? AV_PICTURE_TYPE_I
                           : attributes->sliceType == VVDEC_SLICETYPE_P ? AV_PICTURE_TYPE_P
                           : attributes->sliceType == VVDEC_SLICETYPE_B ? AV_PICTURE_TYPE_B
                                                                        : AV_PICTURE_TYPE_NONE;
    }
    return 0;
}

// What VVdeC's answer means for the caller: 1 for a picture, 0 to go on, or an error
// (AVERROR_EOF once a flush has given every picture). A picture can come with any answer but
// an error (with VVDEC_EOF too, for an access unit with no data), and is taken whatever came
// with it, as VVdeC's own player takes it.
static int libvvdec_result(AVCodecContext *avctx, int result, const vvdecFrame *picture)
{
    LibvvdecContext *s = avctx->priv_data;

    switch (result) {
    case VVDEC_OK:
    case VVDEC_TRY_AGAIN:
    case VVDEC_EOF:
        break;
    case VVDEC_ERR_DEC_INPUT:
        // As VVdeC's own player does: an access unit it cannot use yet (before the first
        // random access point, say) is skipped.
        av_log(avctx, AV_LOG_WARNING, "VVdeC skipped an access unit: %s\n", vvdec_get_last_error(s->decoder));
        break;
    default:
        av_log(avctx, AV_LOG_ERROR, "VVdeC failed: %s (%s)\n", vvdec_get_error_msg(result),
               vvdec_get_last_error(s->decoder));
        return AVERROR_EXTERNAL;
    }
    if (picture)
        return 1;
    return result == VVDEC_EOF ? AVERROR_EOF : 0;
}

static int libvvdec_receive_frame(AVCodecContext *avctx, AVFrame *frame)
{
    LibvvdecContext *s = avctx->priv_data;
    vvdecFrame *picture = NULL;
    int ret;

    if (!s->decoder)
        return AVERROR_EXTERNAL;  // it could not start again after a seek (logged then)

    // VVdeC's answer and its picture are read in two statements: the call sets `picture`,
    // and C does not say whether an argument beside the call is read before or after it.
    for (;;) {
        if (s->draining) {
            ret = vvdec_flush(s->decoder, &picture);
            ret = libvvdec_result(avctx, ret, picture);
            if (ret == 0)
                return AVERROR_EOF;  // flushed with nothing left
        } else {
            // One access unit in, at most one picture out, as VVdeC's own player feeds it (an
            // access unit with no data would mean the end of the stream to VVdeC).
            ret = ff_decode_get_packet(avctx, s->packet);
            if (ret == AVERROR_EOF) {
                s->draining = 1;
                continue;
            }
            if (ret < 0)
                return ret;
            if (!s->packet->size) {
                av_packet_unref(s->packet);
                continue;
            }
            // The packet's bytes, which VVdeC copies; its time stamp's bits, so that
            // AV_NOPTS_VALUE comes back as itself.
            s->access_unit->payload = s->packet->data;
            s->access_unit->payloadSize = s->packet->size;
            s->access_unit->payloadUsedSize = s->packet->size;
            s->access_unit->cts = (uint64_t)s->packet->pts;
            s->access_unit->ctsValid = true;
            s->access_unit->dts = (uint64_t)s->packet->dts;
            s->access_unit->dtsValid = s->packet->dts != AV_NOPTS_VALUE;
            s->access_unit->rap = !!(s->packet->flags & AV_PKT_FLAG_KEY);
            ret = vvdec_decode(s->decoder, s->access_unit, &picture);
            ret = libvvdec_result(avctx, ret, picture);
            s->access_unit->payload = NULL;
            av_packet_unref(s->packet);
        }
        if (ret < 0)
            return ret;
        if (ret > 0)
            break;
    }

    ret = libvvdec_output(avctx, picture, frame);
    vvdec_frame_unref(s->decoder, picture);  // on this thread: VVdeC's picture list is not locked
    if (ret < 0)
        av_frame_unref(frame);
    return ret;
}

#define OFFSET(x) offsetof(LibvvdecContext, x)
#define VD AV_OPT_FLAG_VIDEO_PARAM | AV_OPT_FLAG_DECODING_PARAM
static const AVOption libvvdec_options[] = {
    { "vvdec_threads", "VVdeC's threads (0: the decoder's thread count, else as many as the CPU has)",
      OFFSET(threads), AV_OPT_TYPE_INT, { .i64 = 0 }, 0, 256, VD },
    { "parse_delay", "access units VVdeC parses ahead (-1: VVdeC's default)",
      OFFSET(parse_delay), AV_OPT_TYPE_INT, { .i64 = -1 }, -1, 256, VD },
    { NULL }
};

static const AVClass libvvdec_class = {
    .class_name = "libvvdec decoder",
    .item_name  = av_default_item_name,
    .option     = libvvdec_options,
    .version    = LIBAVUTIL_VERSION_INT,
};

const FFCodec ff_libvvdec_decoder = {
    .p.name         = "libvvdec",
    CODEC_LONG_NAME("VVdeC H.266 / VVC decoder (" LIBVVDEC_BUILD ")"),
    .p.type         = AVMEDIA_TYPE_VIDEO,
    .p.id           = AV_CODEC_ID_VVC,
    .priv_data_size = sizeof(LibvvdecContext),
    .init           = libvvdec_init,
    .close          = libvvdec_close,
    .flush          = libvvdec_flush,
    FF_CODEC_RECEIVE_FRAME_CB(libvvdec_receive_frame),
    .bsfs           = "vvc_mp4toannexb",
    .p.capabilities = AV_CODEC_CAP_DELAY | AV_CODEC_CAP_OTHER_THREADS,
    .caps_internal  = FF_CODEC_CAP_SETS_FRAME_PROPS | FF_CODEC_CAP_AUTO_THREADS,
    .p.priv_class   = &libvvdec_class,
    .p.wrapper_name = "libvvdec",
};
