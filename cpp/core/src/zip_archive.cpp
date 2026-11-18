#include "vml/zip_archive.h"

#include <miniz.h>

#include <fstream>
#include <stdexcept>

#include "vml/platform/system.h"

namespace vml {

void write_zip(const std::filesystem::path& destination, const std::vector<std::pair<std::string, std::string>>& files) {
    mz_zip_archive zip{};
    if (!mz_zip_writer_init_heap(&zip, 0, 0)) {
        throw std::runtime_error("the zip file could not be made");
    }
    std::string failure;
    for (const auto& [name, bytes] : files) {
        if (!mz_zip_writer_add_mem(&zip, name.c_str(), bytes.data(), bytes.size(), MZ_DEFAULT_LEVEL)) {
            failure = name + " could not be added to the zip file";
            break;
        }
    }
    void* archive = nullptr;
    std::size_t size = 0;
    if (failure.empty() && !mz_zip_writer_finalize_heap_archive(&zip, &archive, &size)) {
        failure = "the zip file could not be made";
    }
    if (failure.empty()) {
        std::ofstream out(destination, std::ios::binary | std::ios::trunc);
        out.write(static_cast<const char*>(archive), static_cast<std::streamsize>(size));
        out.close();  // the last of it may fail only now
        if (!out) {
            failure = platform::path_text(destination) + " could not be written";
        }
    }
    mz_zip_writer_end(&zip);
    mz_free(archive);
    if (!failure.empty()) {
        throw std::runtime_error(failure);
    }
}

}  // namespace vml
