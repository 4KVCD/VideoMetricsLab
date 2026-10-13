// numpy's .npy arrays and .npz files (zip archives of them), as the Python
// app's saved scores hold them (metric_cache.py writes them with
// np.savez_compressed and reads them with np.load). Only what those files
// hold: one-dimensional and scalar arrays of little-endian integers and
// floats, and Unicode text.
#pragma once

#include <cstddef>
#include <filesystem>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace vml {

// One array: numpy's type of its items ("<i4", "<f8", "<U400"), its shape
// (empty for a scalar) and its bytes, in C order.
struct NpyArray {
    std::string descr;
    std::vector<std::size_t> shape;
    std::string data;
};

// Arrays of the types the Python app saves scores in: frame numbers as
// 32-bit integers, times as 64-bit floats, scores as 32-bit floats, a
// whole-video score as a 64-bit float scalar, and text as a Unicode scalar
// (numpy's '<U', 4 bytes a character).
NpyArray npy_int32(const std::vector<int>& values);
NpyArray npy_float64(const std::vector<double>& values);
NpyArray npy_float32(const std::vector<float>& values);
NpyArray npy_float64_scalar(double value);
NpyArray npy_text(std::string_view utf8);

// An array's items as numbers, whatever integer or float type it was saved
// in (numpy converts them the same way); nothing for another type.
std::optional<std::vector<double>> npy_numbers(const NpyArray& array);

// A Unicode scalar's text as UTF-8, as numpy's item() gives it (the padding
// of NUL characters at its end dropped); nothing for another kind of array.
std::optional<std::string> npy_text_value(const NpyArray& array);

// Every array of an .npz file, by name ("frame" for frame.npy), as np.load
// reads it; nothing if the file cannot be read or holds anything else.
std::optional<std::map<std::string, NpyArray>> read_npz(const std::filesystem::path& path);

// The bytes of an .npz file of `arrays`, in their order, as
// np.savez_compressed writes one: a zip archive of deflated .npy files.
std::string npz_bytes(const std::vector<std::pair<std::string, NpyArray>>& arrays);

}  // namespace vml
