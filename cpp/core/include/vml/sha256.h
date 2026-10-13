// SHA-256, the digest Python's hashlib.sha256 gives. The Vulkan engine's
// self-test compares what a GPU calculates with digests the Python app
// keeps (vmaf_vulkan._PROBE_SUMS).
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>

namespace vml {

class Sha256 {
public:
    // Adds `bytes` bytes at `data` to what is hashed.
    void update(const void* data, std::size_t bytes);

    // The digest of everything added, in lowercase hexadecimal, as
    // hexdigest() gives it. Nothing can be added after it.
    std::string hexdigest();

private:
    void compress(const std::uint8_t* block);

    // The first 32 bits of the fractional parts of the square roots of the
    // first 8 primes (FIPS 180-4, 5.3.3).
    std::array<std::uint32_t, 8> state_{0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                                        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    std::array<std::uint8_t, 64> block_{};
    std::size_t used_ = 0;     // bytes waiting in block_
    std::uint64_t length_ = 0;  // bytes added in all
};

}  // namespace vml
