#include "vml/npz.h"

#include <miniz.h>

#include <algorithm>
#include <bit>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <stdexcept>

#include "vml/platform/system.h"

namespace vml {

namespace {

// Where numpy's .npy files align their data (numpy.lib.format.ARRAY_ALIGN),
// and the room numpy leaves in a header for the growing axis's length to
// grow (GROWTH_AXIS_MAX_DIGITS).
constexpr std::size_t array_align = 64;
constexpr std::size_t growth_axis_max_digits = 21;

// The UTF-8 encoding of one code point.
void append_utf8(std::string& out, std::uint32_t code) {
    if (code < 0x80) {
        out += static_cast<char>(code);
    } else if (code < 0x800) {
        out += static_cast<char>(0xC0 | (code >> 6));
        out += static_cast<char>(0x80 | (code & 0x3F));
    } else if (code < 0x10000) {
        out += static_cast<char>(0xE0 | (code >> 12));
        out += static_cast<char>(0x80 | ((code >> 6) & 0x3F));
        out += static_cast<char>(0x80 | (code & 0x3F));
    } else {
        out += static_cast<char>(0xF0 | (code >> 18));
        out += static_cast<char>(0x80 | ((code >> 12) & 0x3F));
        out += static_cast<char>(0x80 | ((code >> 6) & 0x3F));
        out += static_cast<char>(0x80 | (code & 0x3F));
    }
}

// The code points of UTF-8 text. A byte that does not start a well-formed
// sequence stands for itself (the text the app writes is ASCII: its JSON
// escapes every other character).
std::vector<std::uint32_t> code_points(std::string_view utf8) {
    std::vector<std::uint32_t> codes;
    for (std::size_t i = 0; i < utf8.size();) {
        const auto lead = static_cast<unsigned char>(utf8[i]);
        const std::size_t length = lead < 0x80 ? 1 : (lead & 0xE0) == 0xC0 ? 2 : (lead & 0xF0) == 0xE0 ? 3
                                                     : (lead & 0xF8) == 0xF0  ? 4
                                                                              : 0;
        if (length <= 1 || i + length > utf8.size()) {
            codes.push_back(lead);
            ++i;
            continue;
        }
        std::uint32_t code = lead & (length == 2 ? 0x1F : length == 3 ? 0x0F : 0x07);
        for (std::size_t k = 1; k < length; ++k) {
            code = (code << 6) | (static_cast<unsigned char>(utf8[i + k]) & 0x3F);
        }
        codes.push_back(code);
        i += length;
    }
    return codes;
}

template <typename T>
NpyArray little_endian_array(std::string descr, std::vector<std::size_t> shape, const T* values, std::size_t count) {
    NpyArray array{std::move(descr), std::move(shape), std::string(count * sizeof(T), '\0')};
    for (std::size_t i = 0; i < count; ++i) {
        unsigned char bytes[sizeof(T)];
        std::memcpy(bytes, &values[i], sizeof(T));
        if constexpr (std::endian::native == std::endian::big) {
            std::reverse(std::begin(bytes), std::end(bytes));
        }
        std::memcpy(array.data.data() + i * sizeof(T), bytes, sizeof(T));
    }
    return array;
}

// The value of `bytes` little-endian bytes at `at`, as the type `T`.
template <typename T>
T little_endian_value(const char* at) {
    unsigned char bytes[sizeof(T)];
    std::memcpy(bytes, at, sizeof(T));
    if constexpr (std::endian::native == std::endian::big) {
        std::reverse(std::begin(bytes), std::end(bytes));
    }
    T value;
    std::memcpy(&value, bytes, sizeof(T));
    return value;
}

std::size_t item_count(const std::vector<std::size_t>& shape) {
    std::size_t count = 1;
    for (const std::size_t length : shape) {
        count *= length;
    }
    return count;
}

// An .npy header's text, as numpy's _write_array_header makes it: the keys
// in order, each value as Python's repr() writes it, and room for the
// growing axis's length.
std::string header_text(const NpyArray& array) {
    std::string shape = "(";
    for (std::size_t i = 0; i < array.shape.size(); ++i) {
        shape += std::to_string(array.shape[i]) + (array.shape.size() == 1 ? "," : i + 1 < array.shape.size() ? ", " : "");
    }
    shape += ")";
    std::string header = "{'descr': '" + array.descr + "', 'fortran_order': False, 'shape': " + shape + ", }";
    if (!array.shape.empty()) {
        const std::size_t digits = std::to_string(array.shape.front()).size();
        header.append(growth_axis_max_digits > digits ? growth_axis_max_digits - digits : 0, ' ');
    }
    return header;
}

// An array as an .npy file of format version 1.0, as numpy writes one: the
// header padded with spaces and ended with a newline, so that the data
// starts at a multiple of 64 bytes.
std::string npy_bytes(const NpyArray& array) {
    const std::string header = header_text(array);
    const std::size_t length = header.size() + 1;  // with the newline
    const std::size_t padding = array_align - ((8 + 2 + length) % array_align);
    const std::size_t stored = length + padding;
    std::string bytes = "\x93NUMPY";
    bytes += static_cast<char>(1);
    bytes += static_cast<char>(0);
    bytes += static_cast<char>(stored & 0xFF);
    bytes += static_cast<char>((stored >> 8) & 0xFF);
    bytes += header;
    bytes.append(padding, ' ');
    bytes += '\n';
    bytes += array.data;
    return bytes;
}

// Reads the Python literal of an .npy header: a dict of 'descr' (a string),
// 'fortran_order' (a bool) and 'shape' (a tuple of integers).
class HeaderReader {
public:
    explicit HeaderReader(std::string_view text) : text_(text) {}

    std::optional<NpyArray> read() {
        NpyArray array;
        bool has_descr = false;
        bool has_shape = false;
        if (!take('{')) {
            return std::nullopt;
        }
        while (!take('}')) {
            const std::optional<std::string> key = string();
            if (!key || !take(':')) {
                return std::nullopt;
            }
            if (*key == "descr") {
                const std::optional<std::string> descr = string();
                if (!descr) {
                    return std::nullopt;
                }
                array.descr = *descr;
                has_descr = true;
            } else if (*key == "fortran_order") {
                const std::optional<bool> fortran = boolean();
                // C and Fortran order are the same for the one-dimensional
                // arrays the app saves; a matrix in Fortran order is not one.
                if (!fortran || (*fortran && array.shape.size() > 1)) {
                    return std::nullopt;
                }
            } else if (*key == "shape") {
                const std::optional<std::vector<std::size_t>> shape = tuple();
                if (!shape) {
                    return std::nullopt;
                }
                array.shape = *shape;
                has_shape = true;
            } else {
                return std::nullopt;
            }
            if (!take(',') && !peek('}')) {
                return std::nullopt;
            }
        }
        if (!has_descr || !has_shape) {
            return std::nullopt;
        }
        return array;
    }

private:
    void skip_space() {
        while (at_ < text_.size() && std::isspace(static_cast<unsigned char>(text_[at_]))) {
            ++at_;
        }
    }
    bool peek(char wanted) {
        skip_space();
        return at_ < text_.size() && text_[at_] == wanted;
    }
    bool take(char wanted) {
        if (!peek(wanted)) {
            return false;
        }
        ++at_;
        return true;
    }
    std::optional<std::string> string() {
        skip_space();
        if (at_ >= text_.size() || (text_[at_] != '\'' && text_[at_] != '"')) {
            return std::nullopt;
        }
        const char quote = text_[at_++];
        const std::size_t end = text_.find(quote, at_);
        if (end == std::string_view::npos) {
            return std::nullopt;
        }
        std::string value(text_.substr(at_, end - at_));
        at_ = end + 1;
        return value;
    }
    std::optional<bool> boolean() {
        skip_space();
        for (const auto& [word, value] : {std::pair<std::string_view, bool>{"True", true}, {"False", false}}) {
            if (text_.substr(at_, word.size()) == word) {
                at_ += word.size();
                return value;
            }
        }
        return std::nullopt;
    }
    std::optional<std::vector<std::size_t>> tuple() {
        if (!take('(')) {
            return std::nullopt;
        }
        std::vector<std::size_t> values;
        while (!take(')')) {
            skip_space();
            const std::size_t start = at_;
            std::size_t value = 0;
            while (at_ < text_.size() && std::isdigit(static_cast<unsigned char>(text_[at_]))) {
                value = value * 10 + static_cast<std::size_t>(text_[at_] - '0');
                ++at_;
            }
            if (at_ == start) {
                return std::nullopt;
            }
            values.push_back(value);
            if (!take(',') && !peek(')')) {
                return std::nullopt;
            }
        }
        return values;
    }

    std::string_view text_;
    std::size_t at_ = 0;
};

// The size of one item of a type the app reads, or 0 for another type.
std::size_t item_size(const std::string& descr) {
    if (descr.size() < 3 || (descr[0] != '<' && descr[0] != '|' && descr[0] != '=')) {
        return 0;
    }
    if (descr[1] == 'U') {
        std::size_t characters = 0;
        for (std::size_t i = 2; i < descr.size(); ++i) {
            if (!std::isdigit(static_cast<unsigned char>(descr[i]))) {
                return 0;
            }
            characters = characters * 10 + static_cast<std::size_t>(descr[i] - '0');
        }
        return characters * 4;
    }
    if (descr.size() != 3 || (descr[1] != 'i' && descr[1] != 'u' && descr[1] != 'f')) {
        return 0;
    }
    const std::size_t bytes = static_cast<std::size_t>(descr[2] - '0');
    const bool known = descr[1] == 'f' ? (bytes == 4 || bytes == 8) : (bytes == 1 || bytes == 2 || bytes == 4 || bytes == 8);
    return known ? bytes : 0;
}

// An .npy file's array, or nothing if it is not one the app can read.
std::optional<NpyArray> parse_npy(std::string_view bytes) {
    if (bytes.size() < 10 || bytes.substr(0, 6) != "\x93NUMPY") {
        return std::nullopt;
    }
    const auto major = static_cast<unsigned char>(bytes[6]);
    std::size_t length = 0;
    std::size_t start = 0;
    if (major == 1) {
        length = static_cast<unsigned char>(bytes[8]) | (static_cast<std::size_t>(static_cast<unsigned char>(bytes[9])) << 8);
        start = 10;
    } else if (major == 2 || major == 3) {
        if (bytes.size() < 12) {
            return std::nullopt;
        }
        for (int k = 3; k >= 0; --k) {
            length = (length << 8) | static_cast<unsigned char>(bytes[8 + static_cast<std::size_t>(k)]);
        }
        start = 12;
    } else {
        return std::nullopt;
    }
    if (bytes.size() < start + length) {
        return std::nullopt;
    }
    std::optional<NpyArray> array = HeaderReader(bytes.substr(start, length)).read();
    if (!array) {
        return std::nullopt;
    }
    const std::size_t size = item_size(array->descr);
    const std::string_view data = bytes.substr(start + length);
    if (size == 0 || data.size() != item_count(array->shape) * size) {
        return std::nullopt;
    }
    array->data = std::string(data);
    return array;
}

}  // namespace

NpyArray npy_int32(const std::vector<int>& values) {
    std::vector<std::int32_t> items(values.begin(), values.end());
    return little_endian_array("<i4", {items.size()}, items.data(), items.size());
}

NpyArray npy_float64(const std::vector<double>& values) {
    return little_endian_array("<f8", {values.size()}, values.data(), values.size());
}

NpyArray npy_float32(const std::vector<float>& values) {
    return little_endian_array("<f4", {values.size()}, values.data(), values.size());
}

NpyArray npy_float64_scalar(double value) { return little_endian_array("<f8", {}, &value, 1); }

NpyArray npy_text(std::string_view utf8) {
    std::vector<std::uint32_t> codes = code_points(utf8);
    if (codes.empty()) {
        codes.push_back(0);  // numpy's '<U1' of an empty string
    }
    return little_endian_array("<U" + std::to_string(codes.size()), {}, codes.data(), codes.size());
}

std::optional<std::vector<double>> npy_numbers(const NpyArray& array) {
    const std::size_t size = item_size(array.descr);
    if (size == 0 || array.descr[1] == 'U') {
        return std::nullopt;
    }
    const char kind = array.descr[1];
    std::vector<double> values;
    values.reserve(array.data.size() / size);
    for (std::size_t at = 0; at + size <= array.data.size(); at += size) {
        const char* item = array.data.data() + at;
        double value = 0.0;
        if (kind == 'f') {
            value = size == 4 ? static_cast<double>(little_endian_value<float>(item)) : little_endian_value<double>(item);
        } else if (kind == 'i') {
            value = size == 1   ? little_endian_value<std::int8_t>(item)
                    : size == 2 ? little_endian_value<std::int16_t>(item)
                    : size == 4 ? little_endian_value<std::int32_t>(item)
                                : static_cast<double>(little_endian_value<std::int64_t>(item));
        } else {
            value = size == 1   ? little_endian_value<std::uint8_t>(item)
                    : size == 2 ? little_endian_value<std::uint16_t>(item)
                    : size == 4 ? little_endian_value<std::uint32_t>(item)
                                : static_cast<double>(little_endian_value<std::uint64_t>(item));
        }
        values.push_back(value);
    }
    return values;
}

std::optional<std::string> npy_text_value(const NpyArray& array) {
    if (item_size(array.descr) == 0 || array.descr[1] != 'U' || !array.shape.empty()) {
        return std::nullopt;
    }
    std::vector<std::uint32_t> codes;
    for (std::size_t at = 0; at + 4 <= array.data.size(); at += 4) {
        codes.push_back(little_endian_value<std::uint32_t>(array.data.data() + at));
    }
    while (!codes.empty() && codes.back() == 0) {
        codes.pop_back();
    }
    std::string text;
    for (const std::uint32_t code : codes) {
        append_utf8(text, code);
    }
    return text;
}

std::optional<std::map<std::string, NpyArray>> read_npz(const std::filesystem::path& path) {
    const std::optional<std::string> file = platform::read_file(path);
    if (!file) {
        return std::nullopt;
    }
    mz_zip_archive zip{};
    if (!mz_zip_reader_init_mem(&zip, file->data(), file->size(), 0)) {
        return std::nullopt;
    }
    std::map<std::string, NpyArray> arrays;
    bool good = true;
    const mz_uint count = mz_zip_reader_get_num_files(&zip);
    for (mz_uint index = 0; good && index < count; ++index) {
        mz_zip_archive_file_stat stat{};
        if (!mz_zip_reader_file_stat(&zip, index, &stat) || stat.m_is_directory) {
            continue;
        }
        const std::string name = stat.m_filename;
        if (name.size() <= 4 || name.compare(name.size() - 4, 4, ".npy") != 0) {
            continue;  // np.load would keep it as bytes, which no saved score has
        }
        std::size_t size = 0;
        void* bytes = mz_zip_reader_extract_to_heap(&zip, index, &size, 0);
        if (bytes == nullptr) {
            good = false;
            break;
        }
        std::optional<NpyArray> array = parse_npy(std::string_view(static_cast<const char*>(bytes), size));
        mz_free(bytes);
        if (!array) {
            good = false;
            break;
        }
        arrays[name.substr(0, name.size() - 4)] = std::move(*array);
    }
    mz_zip_reader_end(&zip);
    if (!good) {
        return std::nullopt;
    }
    return arrays;
}

std::string npz_bytes(const std::vector<std::pair<std::string, NpyArray>>& arrays) {
    mz_zip_archive zip{};
    if (!mz_zip_writer_init_heap(&zip, 0, 0)) {
        throw std::runtime_error("Could not pack the scores into a .npz archive");
    }
    bool good = true;
    for (const auto& [name, array] : arrays) {
        const std::string bytes = npy_bytes(array);
        if (!mz_zip_writer_add_mem(&zip, (name + ".npy").c_str(), bytes.data(), bytes.size(), MZ_DEFAULT_LEVEL)) {
            good = false;
            break;
        }
    }
    void* archive = nullptr;
    std::size_t size = 0;
    good = good && mz_zip_writer_finalize_heap_archive(&zip, &archive, &size);
    std::string bytes = good ? std::string(static_cast<const char*>(archive), size) : std::string();
    mz_zip_writer_end(&zip);
    mz_free(archive);
    if (!good) {
        throw std::runtime_error("Could not pack the scores into a .npz archive");
    }
    return bytes;
}

}  // namespace vml
