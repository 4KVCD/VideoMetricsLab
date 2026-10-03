// SPDX-License-Identifier: MIT
// Only luma is used by libvmaf's CUDA VIF, ADM and motion features. P010
// stores its ten bits at the high end; libvmaf expects low-bit planar values.
extern "C" __global__ void prepare_luma(const unsigned char *src, unsigned long long src_pitch,
                                      unsigned char *dst, unsigned long long dst_pitch,
                                      int width, int height, int bytes, int shift) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    if (bytes == 1) dst[y * dst_pitch + x] = src[y * src_pitch + x];
    else reinterpret_cast<unsigned short *>(dst + y * dst_pitch)[x] =
        reinterpret_cast<const unsigned short *>(src + y * src_pitch)[x] >> shift;
}
