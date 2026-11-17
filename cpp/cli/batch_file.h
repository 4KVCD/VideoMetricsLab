// compare --batch's list: which test videos to compare with which reference,
// a row each, read from a CSV file, so that one process compares them all.
#pragma once

#include <cstddef>
#include <filesystem>
#include <map>
#include <string>
#include <string_view>
#include <vector>

namespace vml::cli {

// A row of the list: a test video, or a pattern such as encodes\*.mkv, and
// its reference.
struct BatchRow {
    std::size_t line = 0;  // where it is in the file, from 1
    std::filesystem::path reference;
    std::string test;
    std::string sequence;  // its sequence cell; empty where the file has none
    // The other columns' cells, by column name (in lower case), for the
    // --scores-csv file.
    std::map<std::string, std::string> text;
};

// The rows of one reference, in the file's order.
struct BatchGroup {
    std::filesystem::path reference;
    std::vector<BatchRow> rows;
};

// The list in `text`, the contents of the CSV file `name` in `folder`: a
// header row with the columns reference and test (in any case) and any
// others, then a row per test video. A relative path is relative to
// `folder`. Rows of the same reference (the same file, however written)
// make one group; the groups come in the order their references first come.
// Throws std::runtime_error saying what is wrong, `name` naming the file.
std::vector<BatchGroup> read_batch_file(std::string_view text, const std::filesystem::path& folder,
                                        const std::string& name);

}  // namespace vml::cli
