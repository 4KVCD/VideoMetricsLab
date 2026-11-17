#include "vml/ffmpeg_download.h"

#include <miniz.h>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cctype>
#include <charconv>
#include <fstream>
#include <regex>
#include <sstream>
#include <system_error>
#include <vector>

#include "vml/ffmpeg_tools.h"
#include "vml/platform/system.h"

namespace vml {

namespace {

using Json = nlohmann::json;

// A SHA-256 in hexadecimal, lowercased; nothing when `text` is not one.
std::optional<std::string> sha256_text(std::string_view text) {
    if (text.size() != 64) {
        return std::nullopt;
    }
    std::string lower;
    for (const char c : text) {
        if (std::isxdigit(static_cast<unsigned char>(c)) == 0) {
            return std::nullopt;
        }
        lower.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
    }
    return lower;
}

std::string text_field(const Json& object, const char* key) {
    const auto found = object.find(key);
    return found != object.end() && found->is_string() ? found->get<std::string>() : std::string();
}

// A number of decimal digits; nothing where it does not fit an int (an
// asset name GitHub lists must not throw out of a Qt slot, as std::stoi
// would).
std::optional<int> small_number(const std::string& digits) {
    int value = 0;
    const auto [end, error] = std::from_chars(digits.data(), digits.data() + digits.size(), value);
    return error == std::errc() && end == digits.data() + digits.size() ? std::optional(value) : std::nullopt;
}

// The two programs the app needs, as a build's zip holds them.
constexpr std::array<const char*, 2> wanted_programs{"ffmpeg.exe", "ffprobe.exe"};

// Where a zip entry named "<top>/bin/<program>" goes: `program`, when that
// is one wanted and `top` one plain folder (no "..", no deeper path); else
// nothing.
std::optional<std::string> wanted_entry(std::string_view entry) {
    const std::size_t slash = entry.find('/');
    if (slash == std::string_view::npos || slash == 0) {
        return std::nullopt;
    }
    const std::string_view top = entry.substr(0, slash);
    if (top == "." || top == ".." || top.find('\\') != std::string_view::npos) {
        return std::nullopt;
    }
    for (const char* program : wanted_programs) {
        if (entry.substr(slash) == std::string("/bin/") + program) {
            return std::string(program);
        }
    }
    return std::nullopt;
}

// miniz reading the zip through a stream that takes a wide Windows path
// (miniz is built without its own file reading).
size_t read_zip(void* opaque, mz_uint64 offset, void* buffer, size_t bytes) {
    auto& stream = *static_cast<std::ifstream*>(opaque);
    stream.clear();
    stream.seekg(static_cast<std::streamoff>(offset));
    stream.read(static_cast<char*>(buffer), static_cast<std::streamsize>(bytes));
    return static_cast<size_t>(stream.gcount());
}

// An entry being unpacked: where it is written, and the Cancel that stops
// it between chunks.
struct Unpacking {
    std::ofstream* out;
    const CancelToken* cancel;
};

size_t write_entry(void* opaque, mz_uint64 /*offset*/, const void* buffer, size_t bytes) {
    auto& unpacking = *static_cast<Unpacking*>(opaque);
    if (unpacking.cancel->cancelled()) {
        return 0;  // miniz stops; the caller sees the Cancel
    }
    unpacking.out->write(static_cast<const char*>(buffer), static_cast<std::streamsize>(bytes));
    return *unpacking.out ? bytes : 0;
}

// The open zip, ended however unpacking ends.
struct ZipReader {
    mz_zip_archive zip{};
    bool open = false;
    ~ZipReader() {
        if (open) {
            mz_zip_reader_end(&zip);
        }
    }
};

}  // namespace

std::optional<std::string> sha256_from_digest(std::string_view digest) {
    constexpr std::string_view kind = "sha256:";
    if (digest.substr(0, kind.size()) != kind) {
        return std::nullopt;
    }
    return sha256_text(digest.substr(kind.size()));
}

std::optional<FfmpegBuild> choose_ffmpeg_build(const std::string& release_json) {
    const Json release = Json::parse(release_json, nullptr, false);
    if (!release.is_object() || !release.contains("assets") || !release["assets"].is_array()) {
        return std::nullopt;
    }
    // "ffmpeg-n9.0-latest-win64-gpl-9.0.zip": the stable 9.0 build, its
    // programs linked in (not "-shared"), GPL (x264, x265...) rather than LGPL.
    static const std::regex stable_build(R"(ffmpeg-n(\d+)\.(\d+)-latest-win64-gpl-[\d.]+\.zip)");
    std::optional<FfmpegBuild> chosen;
    std::vector<int> chosen_version;
    std::string checksums_url;
    for (const Json& asset : release["assets"]) {
        if (!asset.is_object()) {
            continue;
        }
        const std::string name = text_field(asset, "name");
        if (name == "checksums.sha256") {
            checksums_url = text_field(asset, "browser_download_url");
        }
        std::smatch match;
        if (!std::regex_match(name, match, stable_build)) {
            continue;
        }
        const std::optional<int> major = small_number(match[1]);
        const std::optional<int> minor = small_number(match[2]);
        if (!major || !minor) {
            continue;
        }
        const std::vector<int> version{*major, *minor};
        const std::string url = text_field(asset, "browser_download_url");
        const auto size = asset.find("size");
        if (version < minimum_ffmpeg_version || url.empty() || size == asset.end() || !size->is_number_integer() ||
            (chosen && version <= chosen_version)) {
            continue;
        }
        chosen = FfmpegBuild{name, match[1].str() + "." + match[2].str(), url, size->get<long long>(),
                             sha256_from_digest(text_field(asset, "digest")).value_or(""), ""};
        chosen_version = version;
    }
    if (chosen) {
        chosen->checksums_url = checksums_url;
    }
    return chosen;
}

std::optional<std::string> checksum_from_list(const std::string& text, std::string_view name) {
    std::istringstream lines(text);
    std::string line;
    while (std::getline(lines, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        // "<hex>  <name>", or "<hex> *<name>" for a file read as binary.
        const std::size_t space = line.find(' ');
        if (space == std::string::npos) {
            continue;
        }
        std::string_view listed = std::string_view(line).substr(space);
        listed.remove_prefix(std::min(listed.find_first_not_of(" *"), listed.size()));
        if (listed == name) {
            return sha256_text(std::string_view(line).substr(0, space));
        }
    }
    return std::nullopt;
}

void unpack_ffmpeg(const std::filesystem::path& zip_path, const std::filesystem::path& folder,
                   const CancelToken& cancel) {
    std::error_code error;
    const auto zip_size = std::filesystem::file_size(zip_path, error);
    std::ifstream stream(zip_path, std::ios::binary);
    if (error || !stream) {
        throw FfmpegUnpackError("Could not read " + platform::path_text(zip_path));
    }
    ZipReader reader;
    reader.zip.m_pRead = read_zip;
    reader.zip.m_pIO_opaque = &stream;
    if (!mz_zip_reader_init(&reader.zip, zip_size, 0)) {
        throw FfmpegUnpackError("Not a zip file: " + platform::path_text(zip_path));
    }
    reader.open = true;

    // Each wanted program's entry.
    std::array<std::optional<mz_uint>, wanted_programs.size()> entries;
    for (mz_uint index = 0; index < mz_zip_reader_get_num_files(&reader.zip); ++index) {
        mz_zip_archive_file_stat stat{};
        if (!mz_zip_reader_file_stat(&reader.zip, index, &stat) || stat.m_is_directory) {
            continue;
        }
        if (const std::optional<std::string> program = wanted_entry(stat.m_filename)) {
            const auto slot = std::find(wanted_programs.begin(), wanted_programs.end(), *program);
            entries[static_cast<std::size_t>(slot - wanted_programs.begin())] = index;
        }
    }
    for (std::size_t slot = 0; slot < wanted_programs.size(); ++slot) {
        if (!entries[slot]) {
            throw FfmpegUnpackError(std::string("The download has no ") + wanted_programs[slot]);
        }
    }

    std::filesystem::create_directories(folder, error);
    std::vector<std::filesystem::path> parts;
    const auto remove_parts = [&parts] {
        std::error_code ignored;
        for (const std::filesystem::path& part : parts) {
            std::filesystem::remove(part, ignored);
        }
    };
    try {
        for (std::size_t slot = 0; slot < wanted_programs.size(); ++slot) {
            const std::filesystem::path part = folder / (std::string(wanted_programs[slot]) + ".part");
            parts.push_back(part);
            std::ofstream out(part, std::ios::binary | std::ios::trunc);
            Unpacking unpacking{&out, &cancel};
            // miniz checks the entry's size and CRC-32 as it unpacks it.
            const bool unpacked = out && mz_zip_reader_extract_to_callback(&reader.zip, *entries[slot], write_entry,
                                                                           &unpacking, 0) != 0;
            out.close();
            cancel.throw_if_cancelled();
            if (!unpacked || !out) {
                throw FfmpegUnpackError("Could not unpack " + std::string(wanted_programs[slot]) + " to " +
                                        platform::path_text(part));
            }
        }
        // Both whole: now in place of the old ones.
        for (std::size_t slot = 0; slot < wanted_programs.size(); ++slot) {
            std::filesystem::rename(parts[slot], folder / wanted_programs[slot], error);
            if (error) {
                throw FfmpegUnpackError("Could not replace " + platform::path_text(folder / wanted_programs[slot]) +
                                        " (" + error.message() + "): is it running?");
            }
        }
    } catch (...) {
        remove_parts();
        throw;
    }
}

}  // namespace vml
