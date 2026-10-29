// What every system's InboundPipe shares.
#include "vml/platform/pipe.h"

#include <cstdint>

namespace vml::platform {

std::size_t InboundPipe::read_full(void* into, std::size_t bytes, const std::atomic<bool>& stop,
                                   const std::atomic<bool>& writer_gone) {
    auto* const start = static_cast<std::uint8_t*>(into);
    std::size_t filled = 0;
    while (filled < bytes) {
        gather(bytes - filled, stop, writer_gone);
        const std::size_t got = read(start + filled, bytes - filled);
        if (got == 0) {
            break;
        }
        filled += got;
    }
    return filled;
}

}  // namespace vml::platform
