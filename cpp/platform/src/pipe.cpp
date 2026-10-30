// What every system's pipes share.
#include "vml/platform/pipe.h"

#include <cstdint>

namespace vml::platform {

PipeWriteError pipe_write_error(unsigned long code) {
    // Windows' codes, here by number so that every system can test them.
    constexpr unsigned long broken_pipe = 109;           // ERROR_BROKEN_PIPE
    constexpr unsigned long no_data = 232;               // ERROR_NO_DATA: the pipe is being closed
    constexpr unsigned long no_system_resources = 1450;  // ERROR_NO_SYSTEM_RESOURCES
    constexpr unsigned long working_set_quota = 1453;    // ERROR_WORKING_SET_QUOTA
    constexpr unsigned long not_enough_quota = 1816;     // ERROR_NOT_ENOUGH_QUOTA
    if (code == broken_pipe || code == no_data) {
        return PipeWriteError::reader_gone;
    }
    if (code == no_system_resources || code == working_set_quota || code == not_enough_quota) {
        return PipeWriteError::try_again;
    }
    return PipeWriteError::failed;
}

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
