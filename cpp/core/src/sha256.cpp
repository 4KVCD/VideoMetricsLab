// SHA-256 as FIPS 180-4 defines it.
#include "vml/sha256.h"

#include <algorithm>
#include <cstring>

namespace vml {

namespace {

// The first 32 bits of the fractional parts of the cube roots of the first
// 64 primes (FIPS 180-4, 4.2.2).
constexpr std::array<std::uint32_t, 64> round_constants{
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
};

std::uint32_t rotate_right(std::uint32_t value, int bits) { return (value >> bits) | (value << (32 - bits)); }

}  // namespace

void Sha256::compress(const std::uint8_t* block) {
    std::array<std::uint32_t, 64> schedule{};
    for (int i = 0; i < 16; ++i) {
        schedule[i] = (std::uint32_t{block[4 * i]} << 24) | (std::uint32_t{block[4 * i + 1]} << 16) |
                      (std::uint32_t{block[4 * i + 2]} << 8) | std::uint32_t{block[4 * i + 3]};
    }
    for (int i = 16; i < 64; ++i) {
        const std::uint32_t s0 =
            rotate_right(schedule[i - 15], 7) ^ rotate_right(schedule[i - 15], 18) ^ (schedule[i - 15] >> 3);
        const std::uint32_t s1 =
            rotate_right(schedule[i - 2], 17) ^ rotate_right(schedule[i - 2], 19) ^ (schedule[i - 2] >> 10);
        schedule[i] = schedule[i - 16] + s0 + schedule[i - 7] + s1;
    }
    std::uint32_t a = state_[0], b = state_[1], c = state_[2], d = state_[3];
    std::uint32_t e = state_[4], f = state_[5], g = state_[6], h = state_[7];
    for (int i = 0; i < 64; ++i) {
        const std::uint32_t s1 = rotate_right(e, 6) ^ rotate_right(e, 11) ^ rotate_right(e, 25);
        const std::uint32_t choice = (e & f) ^ (~e & g);
        const std::uint32_t first = h + s1 + choice + round_constants[i] + schedule[i];
        const std::uint32_t s0 = rotate_right(a, 2) ^ rotate_right(a, 13) ^ rotate_right(a, 22);
        const std::uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
        const std::uint32_t second = s0 + majority;
        h = g;
        g = f;
        f = e;
        e = d + first;
        d = c;
        c = b;
        b = a;
        a = first + second;
    }
    state_[0] += a;
    state_[1] += b;
    state_[2] += c;
    state_[3] += d;
    state_[4] += e;
    state_[5] += f;
    state_[6] += g;
    state_[7] += h;
}

void Sha256::update(const void* data, std::size_t bytes) {
    const auto* next = static_cast<const std::uint8_t*>(data);
    length_ += bytes;
    while (bytes > 0) {
        const std::size_t taken = std::min(bytes, block_.size() - used_);
        std::memcpy(block_.data() + used_, next, taken);
        used_ += taken;
        next += taken;
        bytes -= taken;
        if (used_ == block_.size()) {
            compress(block_.data());
            used_ = 0;
        }
    }
}

std::string Sha256::hexdigest() {
    // The padding: a 1 bit, zeros up to 8 bytes short of a block's end,
    // then the message's length in bits, big-endian.
    const std::uint64_t bits = length_ * 8;
    const std::uint8_t one = 0x80;
    update(&one, 1);
    const std::uint8_t zero = 0;
    while (used_ != block_.size() - 8) {
        update(&zero, 1);
    }
    std::uint8_t length[8];
    for (int i = 0; i < 8; ++i) {
        length[i] = static_cast<std::uint8_t>(bits >> (56 - 8 * i));
    }
    update(length, sizeof length);
    static constexpr char digits[] = "0123456789abcdef";
    std::string text;
    text.reserve(64);
    for (const std::uint32_t word : state_) {
        for (int shift = 28; shift >= 0; shift -= 4) {
            text += digits[(word >> shift) & 0xF];
        }
    }
    return text;
}

}  // namespace vml
