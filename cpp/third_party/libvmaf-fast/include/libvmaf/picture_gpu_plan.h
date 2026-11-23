/**
 *
 *  Copyright 2016-2020 Netflix, Inc.
 *
 *     Licensed under the BSD+Patent License (the "License");
 *     you may not use this file except in compliance with the License.
 *     You may obtain a copy of the License at
 *
 *         https://opensource.org/licenses/BSDplusPatent
 *
 *     Unless required by applicable law or agreed to in writing, software
 *     distributed under the License is distributed on an "AS IS" BASIS,
 *     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *     See the License for the specific language governing permissions and
 *     limitations under the License.
 *
 */

/*
 * A picture conversion (vmaf_picture_convert) as a list of steps another
 * engine can run, a GPU for one, giving the same samples bit for bit.
 *
 * vmaf_picture_convert_gpu_plan() builds the list from the very graph zimg
 * builds for the conversion: each resize, depth change and colour operation
 * zimg would run, with its filter coefficients and lookup tables. The plan
 * is a block of 32-bit words; floats are stored as their bit patterns.
 *
 * Layout (word offsets):
 *   header   VMAF_GPU_PLAN_HEADER_WORDS words, VMAF_GPU_PLAN_H_* below
 *   buffers  buffer_count records of VMAF_GPU_PLAN_BUFFER_WORDS words
 *   steps    step_count records of VMAF_GPU_PLAN_STEP_WORDS words, in the
 *            order they run; a step reads only buffers earlier steps (or the
 *            source) wrote
 *   tables   filters and colour operations the steps point into
 *
 * Buffers hold one plane each, rows packed (a plane w samples wide has its
 * rows w samples apart):
 *   SOURCE   a source plane, samples as 16-bit words (their numeric value)
 *   TARGET   a target plane, 16-bit words
 *   SCRATCH  32-bit elements, a float or a word, `elements` of them; one
 *            scratch buffer is reused by later steps once its last reader ran.
 *            A COLOUR or CONVERT step may write a buffer it reads (each
 *            sample is read, all planes, before that sample is written); a
 *            resize never does
 *
 * Arithmetic (all of it IEEE single precision, no contraction into fma
 * other than where written, round to nearest even):
 *   read  FLOAT  the element as a float
 *         WORD   the word as an integer (the int16 resizes)
 *         W2F    fma(scale, word, offset)
 *   write FLOAT  the float as it is
 *         WORD   the word as it is (the int16 resizes)
 *         F2W    v = fma(scale, x, offset); 0 if v is NaN or |v| >= 2^31,
 *                else min(clamp(round_even(v), 0, 65535), max)
 *   RESIZE_FLOAT_H/V  out[i] = acc0 + acc1, tap k of the filter row added to
 *                accumulator k % 2: acc = fma(c[k], in[left[i] + k], acc);
 *                the first tap of each accumulator starts it: in the H step
 *                acc = fma(c, x, +0.0), in the V step acc = c * x (zimg's
 *                AVX2 kernels; the two differ only in the sign of a zero).
 *                A one-tap filter is acc0 alone.
 *   RESIZE_INT16_H/V  s = sum of c[k] * (in[left[i] + k] - 32768) (int32,
 *                exact); v = (s + 8192) >> 14 (arithmetic), saturated to
 *                int16, min(v, limit - 32768) + 32768
 *   COLOUR       the three planes, sample by sample, through the operations
 *                in order (each on all three channels):
 *                MATRIX     out_r = (m[r][0] * a + m[r][1] * b) + m[r][2] * c,
 *                           every product and sum rounded (no fma)
 *                TO_LINEAR  lut[i], 65,537 entries, i = round_even(fma(x,
 *                           32768, 16384)); 0 if NaN or |.| >= 2^31, else
 *                           clamped to 0..65536
 *                TO_GAMMA   lut[h], 65,536 entries, h = x as a half float
 *                           (round to nearest even), its 16 bits
 *   CONVERT      sample by sample: the read, then the write
 */

#ifndef __VMAF_PICTURE_GPU_PLAN_H__
#define __VMAF_PICTURE_GPU_PLAN_H__

#include <stddef.h>
#include <stdint.h>

#include "picture.h"

#ifdef __cplusplus
extern "C" {
#endif

#define VMAF_GPU_PLAN_MAGIC 0x31435056u /* "VPC1" */
#define VMAF_GPU_PLAN_VERSION 1u

enum VmafGpuPlanHeader {
    VMAF_GPU_PLAN_H_MAGIC,
    VMAF_GPU_PLAN_H_VERSION,
    VMAF_GPU_PLAN_H_WORDS,          /* the whole plan */
    VMAF_GPU_PLAN_H_BUFFER_COUNT,
    VMAF_GPU_PLAN_H_BUFFERS,        /* offset of the first buffer record */
    VMAF_GPU_PLAN_H_STEP_COUNT,
    VMAF_GPU_PLAN_H_STEPS,          /* offset of the first step record */
    VMAF_GPU_PLAN_H_SRC_W,
    VMAF_GPU_PLAN_H_SRC_H,
    VMAF_GPU_PLAN_H_SRC_BPC,
    VMAF_GPU_PLAN_H_DST_W,
    VMAF_GPU_PLAN_H_DST_H,
    VMAF_GPU_PLAN_H_DST_BPC,
    VMAF_GPU_PLAN_HEADER_WORDS = 16,
};

enum VmafGpuPlanBufferKind {
    VMAF_GPU_PLAN_SOURCE,
    VMAF_GPU_PLAN_TARGET,
    VMAF_GPU_PLAN_SCRATCH,
};

/* buffer record: kind, plane (source and target: 0 to 2), elements */
enum VmafGpuPlanBuffer {
    VMAF_GPU_PLAN_B_KIND,
    VMAF_GPU_PLAN_B_PLANE,
    VMAF_GPU_PLAN_B_ELEMENTS,
    VMAF_GPU_PLAN_BUFFER_WORDS = 4,
};

enum VmafGpuPlanStepKind {
    VMAF_GPU_PLAN_RESIZE_FLOAT_H = 1,
    VMAF_GPU_PLAN_RESIZE_FLOAT_V,
    VMAF_GPU_PLAN_RESIZE_INT16_H,
    VMAF_GPU_PLAN_RESIZE_INT16_V,
    VMAF_GPU_PLAN_COLOUR,
    VMAF_GPU_PLAN_CONVERT,
};

enum VmafGpuPlanRead {
    VMAF_GPU_PLAN_READ_FLOAT,
    VMAF_GPU_PLAN_READ_WORD,
    VMAF_GPU_PLAN_READ_W2F,
};

enum VmafGpuPlanWrite {
    VMAF_GPU_PLAN_WRITE_FLOAT,
    VMAF_GPU_PLAN_WRITE_WORD,
    VMAF_GPU_PLAN_WRITE_F2W,
};

/*
 * Step record. Per-plane fields are indexed by the step's plane 0 to 2
 * (one plane for a resize or CONVERT, at index 0; three for COLOUR).
 * TABLE: a resize's filter, left[out_dim] then out_dim rows of FILTER_WIDTH
 * coefficients (floats, or int16 values sign-extended to a word); COLOUR's
 * operation list, FILTER_WIDTH (the count) pairs of (kind, table offset).
 */
enum VmafGpuPlanStep {
    VMAF_GPU_PLAN_S_KIND,
    VMAF_GPU_PLAN_S_PLANES,
    VMAF_GPU_PLAN_S_IN,             /* 3: buffer indices */
    VMAF_GPU_PLAN_S_OUT = VMAF_GPU_PLAN_S_IN + 3,
    VMAF_GPU_PLAN_S_IN_W = VMAF_GPU_PLAN_S_OUT + 3,
    VMAF_GPU_PLAN_S_IN_H,
    VMAF_GPU_PLAN_S_OUT_W,
    VMAF_GPU_PLAN_S_OUT_H,
    VMAF_GPU_PLAN_S_READ,           /* 3 x (mode, scale, offset, 0) */
    VMAF_GPU_PLAN_S_WRITE = VMAF_GPU_PLAN_S_READ + 12, /* 3 x (mode, scale, offset, max) */
    VMAF_GPU_PLAN_S_TABLE = VMAF_GPU_PLAN_S_WRITE + 12,
    VMAF_GPU_PLAN_S_FILTER_WIDTH,   /* COLOUR: the operation count */
    VMAF_GPU_PLAN_S_LIMIT,          /* the int16 resizes' largest sample */
    VMAF_GPU_PLAN_STEP_WORDS = 40,
};

enum VmafGpuPlanColourOp {
    VMAF_GPU_PLAN_OP_MATRIX = 1,    /* 9 floats, row by row */
    VMAF_GPU_PLAN_OP_TO_LINEAR,     /* 65,537 floats */
    VMAF_GPU_PLAN_OP_TO_GAMMA,      /* 65,536 floats */
};

/**
 * Builds the plan of the conversion vmaf_picture_convert_context_init()
 * would set up for `src` (its format, size and colour; its samples are not
 * read) and `target`. -ENOTSUP when this build has no plan builder (libvmaf
 * built without zimg's sources), or zimg's graph has a step the plan does not
 * express: only 10- and 12-bit 4:2:0 sources are taken. `*plan` is freed with
 * vmaf_picture_convert_gpu_plan_free().
 */
int vmaf_picture_convert_gpu_plan(const VmafPicture *src,
                                  const VmafPictureConvertTarget *target,
                                  uint32_t **plan, size_t *words);

void vmaf_picture_convert_gpu_plan_free(uint32_t *plan);

#ifdef __cplusplus
}
#endif

#endif /* __VMAF_PICTURE_GPU_PLAN_H__ */
