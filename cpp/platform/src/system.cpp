// What every system's part of system.h shares.
#include "vml/platform/system.h"

#include <cmath>
#include <fstream>
#include <initializer_list>
#include <system_error>
#include <utility>

#if defined(_MSC_VER) && defined(_M_X64)
#include <intrin.h>
#endif

namespace vml::platform {

bool cpu_runs_x86_64_v3() {
#if defined(_MSC_VER) && defined(_M_X64)
    // CPUID's feature bits (Intel's SDM, volume 2A, CPUID): register indices
    // 1 = EBX, 2 = ECX.
    int regs[4];
    __cpuid(regs, 0);
    if (regs[0] < 7) return false;
    __cpuid(regs, 1);
    const unsigned leaf1 = static_cast<unsigned>(regs[2]);
    __cpuidex(regs, 7, 0);
    const unsigned leaf7 = static_cast<unsigned>(regs[1]);
    __cpuid(regs, static_cast<int>(0x80000001));
    const unsigned extended = static_cast<unsigned>(regs[2]);
    const auto has = [](unsigned value, std::initializer_list<int> bits) {
        for (const int bit : bits) {
            if ((value >> bit & 1u) == 0) return false;
        }
        return true;
    };
    // FMA, MOVBE, OSXSAVE, AVX, F16C; BMI1, AVX2, BMI2; LZCNT.
    if (!has(leaf1, {12, 22, 27, 28, 29}) || !has(leaf7, {3, 5, 8}) || !has(extended, {5})) return false;
    return (_xgetbv(0) & 6) == 6;  // the system saves the SSE and AVX registers
#elif (defined(__GNUC__) || defined(__clang__)) && defined(__x86_64__)
    return __builtin_cpu_supports("x86-64-v3");
#else
    return false;
#endif
}

std::optional<std::string> read_file(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    std::error_code error;
    const std::uintmax_t size = std::filesystem::file_size(path, error);
    if (!file.is_open() || error) {
        return std::nullopt;
    }
    std::string data(static_cast<std::size_t>(size), '\0');
    file.read(data.data(), static_cast<std::streamsize>(data.size()));
    data.resize(static_cast<std::size_t>(file.gcount()));  // a file cut short since its size was read
    return data;
}

DisplayHdrInfo display_hdr_from_edr(double potential_headroom, std::string device_name) {
    DisplayHdrInfo info;
    info.device_name = std::move(device_name);
    if (!std::isfinite(potential_headroom) || potential_headroom <= 0) {
        return info;
    }
    info.hdr_enabled = potential_headroom > 1.0;
    if (*info.hdr_enabled) {
        info.hdr_supported = true;
    }
    return info;
}

}  // namespace vml::platform
