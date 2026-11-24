#include "vml/matroska.h"

#include <algorithm>
#include <fstream>

#include "matroska_ebml.h"

namespace vml {

std::optional<std::string> matroska_writing_app(std::istream& in) {
    using namespace matroska;
    const std::optional<Element> header = read_element(in);
    if (!header || header->id != mkv::ebml || header->size == mkv::unknown_size) {
        return std::nullopt;
    }
    in.seekg(end_of(in, *header));
    const std::optional<Element> segment = read_element(in);
    if (!segment || segment->id != mkv::segment) {
        return std::nullopt;
    }
    // The segment's elements up to its first cluster: a SeekHead, a Void,
    // the Info, the Tracks... A file whose Info comes later, found only
    // through its SeekHead, is not read here.
    constexpr int most_elements = 64;
    for (int count = 0; count < most_elements; ++count) {
        const std::optional<Element> element = read_element(in);
        if (!element || element->id == mkv::cluster || element->size == mkv::unknown_size) {
            return std::nullopt;
        }
        const std::int64_t end = end_of(in, *element);
        if (element->id == mkv::info) {
            std::optional<std::string> app;
            constexpr std::uint64_t longest = 1024;
            each_child(in, end, [&](const Element& field) {
                if (field.id == mkv::writing_app && field.size <= longest) {
                    std::string text(static_cast<std::size_t>(field.size), '\0');
                    in.read(text.data(), static_cast<std::streamsize>(field.size));
                    text.erase(std::find(text.begin(), text.end(), '\0'), text.end());  // padding
                    if (in && !text.empty()) {
                        app = std::move(text);
                    }
                }
                return true;
            });
            return app;
        }
        in.clear();
        in.seekg(end);
        if (!in) {
            return std::nullopt;
        }
    }
    return std::nullopt;
}

std::optional<std::string> matroska_writing_app(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return std::nullopt;
    }
    return matroska_writing_app(in);
}

}  // namespace vml
