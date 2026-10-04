// Derived from libvmaf (https://github.com/Netflix/vmaf), whose CUDA feature
// extractors are Copyright 2016-2023 Netflix, Inc. and Copyright 2021 NVIDIA
// Corporation, licensed under the BSD+Patent License
// (https://opensource.org/licenses/BSDplusPatent); see
// vmaf_app/tools/vmaf_vulkan/licenses/LICENSE.libvmaf.txt.
//
// Shared by every VMAF compute shader (see ../vmaf_vulkan.cpp).
//
// The shaders are a port of libvmaf's CUDA kernels (libvmaf/src/feature/cuda,
// with the pull requests scripts/build_libvmaf_cuda.ps1 lists) and are written
// to give the same integers as those kernels on any GPU: all arithmetic is
// integer, also where the CUDA code uses float and double, because Vulkan
// leaves the rounding of floating point to the driver and Intel's GPUs have no
// double at all.

// A 64-bit sum kept as two 32-bit words, added to with 32-bit atomics (64-bit
// atomics are optional in Vulkan). The carry out of the low word is added to
// the high word, so the pair is the exact sum modulo 2^64 whatever the order.
void acc_add64(RWStructuredBuffer<uint> acc, uint slot, uint64_t value)
{
    uint lo = uint(value), hi = uint(value >> 32), old;
    InterlockedAdd(acc[2 * slot], lo, old);
    uint high = hi + ((old > ~lo) ? 1u : 0u);
    if (high != 0)
        InterlockedAdd(acc[2 * slot + 1], high);
}

// A picture's luma sample: rows of bytes (or 16-bit little-endian samples)
// packed into 32-bit words, each row starting on a word.
uint pixel(StructuredBuffer<uint> picture, int x, int y, int strideWords)
{
#if BPC16
    uint word = picture[y * strideWords + (x >> 1)];
    return (word >> ((x & 1) * 16)) & 0xFFFFu;
#else
    uint word = picture[y * strideWords + (x >> 2)];
    return (word >> ((x & 3) * 8)) & 0xFFu;
#endif
}

int sext16(int v) { return (v << 16) >> 16; }

int msb32(uint v) { return int(firstbithigh(v)); }  // v != 0

int msb64(uint64_t v)  // v != 0
{
    uint hi = uint(v >> 32);
    return hi != 0 ? 32 + msb32(hi) : msb32(uint(v));
}

struct U128 { uint64_t hi, lo; };

U128 mul64(uint64_t a, uint64_t b)
{
    uint64_t a0 = a & 0xFFFFFFFFull, a1 = a >> 32, b0 = b & 0xFFFFFFFFull, b1 = b >> 32;
    uint64_t p00 = a0 * b0, p01 = a0 * b1, p10 = a1 * b0, p11 = a1 * b1;
    uint64_t mid = (p00 >> 32) + (p01 & 0xFFFFFFFFull) + (p10 & 0xFFFFFFFFull);
    U128 r;
    r.lo = (p00 & 0xFFFFFFFFull) | (mid << 32);
    r.hi = p11 + (p01 >> 32) + (p10 >> 32) + (mid >> 32);
    return r;
}

// A positive 128-bit integer rounded to a double's 53 bits, to nearest and
// ties to even: v ~ m * 2^e with m in [2^52, 2^53). v != 0.
void round53(U128 v, out uint64_t m, out int e)
{
    int top = v.hi != 0 ? 64 + msb64(v.hi) : msb64(v.lo);
    if (top <= 52) {
        m = v.lo << (52 - top);
        e = top - 52;
        return;
    }
    int sh = top - 52;
    uint64_t q;
    bool above, tie;
    if (sh < 64) {
        q = (v.lo >> sh) | (v.hi << (64 - sh));
        uint64_t rem = v.lo & ((1ull << sh) - 1), half = 1ull << (sh - 1);
        above = rem > half;
        tie = rem == half;
    } else if (sh == 64) {
        q = v.hi;
        above = v.lo > (1ull << 63);
        tie = v.lo == (1ull << 63);
    } else {
        int s = sh - 64;
        q = v.hi >> s;
        uint64_t rem = v.hi & ((1ull << s) - 1), half = 1ull << (s - 1);
        above = rem > half || (rem == half && v.lo != 0);
        tie = rem == half && v.lo == 0;
    }
    if (above || (tie && (q & 1) != 0)) {
        q++;
        if (q == (1ull << 53)) {
            q >>= 1;
            sh++;
        }
    }
    m = q;
    e = sh;
}
