#include "vml/bitrate_store.h"

#include <ostream>
#include <stdexcept>
#include <utility>

#include <nlohmann/json.hpp>

#include "vml/build_info.h"
#include "vml/npz.h"
#include "vml/platform/system.h"
#include "vml/python_json.h"
#include "vml/result_json.h"
#include "vml/temp_files.h"

namespace vml {

namespace {

// A version in each file: one of another layout is not read.
constexpr int bitrate_format_version = 1;

std::vector<int> as_ints(const std::vector<bool>& flags) { return {flags.begin(), flags.end()}; }

// Byte counts as 64-bit floats: exact to 2^53 bytes, far past any packet.
std::vector<double> as_doubles(const std::vector<long long>& counts) {
    std::vector<double> doubles;
    doubles.reserve(counts.size());
    for (const long long count : counts) {
        doubles.push_back(static_cast<double>(count));
    }
    return doubles;
}

std::vector<int> as_ints(const std::vector<std::uint8_t>& bytes) { return {bytes.begin(), bytes.end()}; }

}  // namespace

std::filesystem::path bitrate_store_folder(const std::filesystem::path& base) { return base / "bitrate"; }

std::optional<SavedBitrate> load_saved_bitrate(const std::filesystem::path& base, const std::string& sampled_hash,
                                               const std::filesystem::path& path) {
    if (sampled_hash.empty()) {
        return std::nullopt;
    }
    const std::optional<std::map<std::string, NpyArray>> arrays =
        read_npz(bitrate_store_folder(base) / (sampled_hash + ".npz"));
    if (!arrays || !arrays->contains("metadata")) {
        return std::nullopt;
    }
    try {
        const std::optional<std::string> text = npy_text_value(arrays->at("metadata"));
        const nlohmann::json metadata = nlohmann::json::parse(python_json_readable(text.value_or("")), nullptr, false);
        if (!metadata.is_object() || metadata.value("format_version", 0) != bitrate_format_version) {
            return std::nullopt;
        }
        const auto numbers = [&arrays](const char* name) {
            return arrays->contains(name) ? npy_numbers(arrays->at(name)) : std::nullopt;
        };
        const auto times = numbers("times");
        const auto durations = numbers("durations");
        const auto sizes = numbers("sizes");
        const auto keyframes = numbers("keyframes");
        const auto types = numbers("frame_types");
        if (!times || !durations || !sizes || !keyframes) {
            return std::nullopt;
        }
        SavedBitrate saved;
        saved.info = video_info_from_dict(metadata.at("info"));
        saved.info.path = path;
        std::vector<long long> byte_counts;
        byte_counts.reserve(sizes->size());
        for (const double size : *sizes) {
            byte_counts.push_back(static_cast<long long>(size));
        }
        std::vector<bool> keys;
        for (const double key : *keyframes) {
            keys.push_back(key != 0.0);
        }
        saved.data = bitrate_data(path, *times, *durations, std::move(byte_counts), std::move(keys));
        // No types saved by a version that saved a reading cut short as none
        // (frame_types_whole missing): read again, as not known.
        const bool none_known = types && types->empty() && !saved.data.sizes.empty() &&
                                !metadata.value("frame_types_whole", false);
        if (types && metadata.value("frame_types_read", false) && !none_known) {
            saved.frame_types.emplace();
            for (const double type : *types) {
                saved.frame_types->push_back(static_cast<std::uint8_t>(type));
            }
        }
        return saved;
    } catch (const std::exception&) {
        return std::nullopt;  // another layout, or columns not as long: read the file again
    }
}

void save_bitrate(const std::filesystem::path& base, const std::string& sampled_hash, const SavedBitrate& bitrate) {
    if (sampled_hash.empty()) {
        return;
    }
    const nlohmann::json metadata = {{"format_version", bitrate_format_version},
                                     {"path", platform::path_text(bitrate.info.path)},
                                     {"info", video_info_to_dict(bitrate.info)},
                                     {"frame_types_read", bitrate.frame_types.has_value()},
                                     // Types saved only from a file read to its end (frame_types.h).
                                     {"frame_types_whole", bitrate.frame_types.has_value()},
                                     {"saved_by", build_stamp()}};
    std::vector<std::pair<std::string, NpyArray>> arrays{
        {"metadata", npy_text(python_json_compact(metadata))},
        {"times", npy_float64(bitrate.data.times)},
        {"durations", npy_float64(bitrate.data.durations)},
        {"sizes", npy_float64(as_doubles(bitrate.data.sizes))},
        {"keyframes", npy_int32(as_ints(bitrate.data.keyframes))},
        {"frame_types", npy_int32(as_ints(bitrate.frame_types.value_or(std::vector<std::uint8_t>())))}};
    const std::string bytes = npz_bytes(arrays);
    std::filesystem::create_directories(bitrate_store_folder(base));
    write_file_whole(bitrate_store_folder(base) / (sampled_hash + ".npz"),
                     [&bytes](std::ostream& out) { out.write(bytes.data(), static_cast<std::streamsize>(bytes.size())); });
}

}  // namespace vml
