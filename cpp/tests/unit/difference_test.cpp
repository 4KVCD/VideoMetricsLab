// Two runs' difference (vml/difference.h): paired by frame, or CVVDP's by
// second, summarized, and written as CSV beside export_csv's files.
#include "vml/difference.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "test_videos.h"
#include "vml/export_csv.h"
#include "vml/platform/system.h"
#include "vml/temp_files.h"

namespace vml {
namespace {

// A CVVDP score with no per-second values still reaches the CSV.
TEST(ExportCsv, ACvvdpScoreWithoutSecondsIsWritten) {
    const test::Folder folder;
    ComparisonResult result;
    result.metric_results.add(SequenceMetricResult{"cvvdp", 8.5, {}, {}, {}, {}});
    const std::filesystem::path out = folder / "run.csv";
    export_csv(result, out);
    std::ifstream file(out);
    std::stringstream text;
    text << file.rdbuf();
    EXPECT_NE(text.str().find(",8.5"), std::string::npos) << text.str();
}

// Two labels that name one file on this system, still to be written, get two
// files; and a write that fails leaves the file that was there, and no other.
TEST(ExportCsv, FilesAreKeptApartAndReplacedWhole) {
    const test::Folder folder;
    std::set<std::filesystem::path> reserved;
    const std::filesystem::path first = unique_output_path(folder / "", "Clip", ".csv", reserved);
    const std::filesystem::path second = unique_output_path(folder / "", "clip", ".csv", reserved);
    EXPECT_NE(platform::name_as_compared(platform::path_text(first.filename())),
              platform::name_as_compared(platform::path_text(second.filename())));

    write_file_whole(first, [](std::ostream& file) { file << "kept"; });
    EXPECT_THROW(write_file_whole(first, [](std::ostream& file) { file.setstate(std::ios::failbit); }),
                 std::runtime_error);
    std::ifstream kept(first);
    std::string text;
    kept >> text;
    EXPECT_EQ(text, "kept");
    EXPECT_EQ(std::distance(std::filesystem::directory_iterator(first.parent_path()),
                            std::filesystem::directory_iterator()),
              1);
}

constexpr float inf = std::numeric_limits<float>::infinity();
constexpr float nan = std::numeric_limits<float>::quiet_NaN();

// NaN compares unequal to itself: "nan" stands for it here.
std::vector<std::string> spelled(const std::vector<double>& values) {
    std::vector<std::string> out;
    for (const double value : values) {
        out.push_back(std::isnan(value) ? "nan" : std::to_string(value));
    }
    return out;
}

TEST(Difference, PairsFramesByNumber) {
    // Frame 0 has no pair, 2 is XPSNR's infinity in both (no difference),
    // 4 has no score in the second, 5 is infinite in the first alone.
    const std::vector<int> first_frames{0, 1, 2, 4, 5};
    const std::vector<float> first_values{90, 80, inf, 70, inf};
    const std::vector<int> second_frames{1, 2, 3, 4, 5};
    const std::vector<float> second_values{75, inf, 50, nan, 40};
    EXPECT_EQ(spelled(frame_difference(first_frames, first_values, second_frames, second_values)),
              spelled({NAN, 5.0, 0.0, NAN, NAN}));
}

TEST(Difference, PairsSecondsByTheSecondTheyStartIn) {
    // An encode at another frame rate starts its seconds a little later.
    const std::vector<double> first_starts{0.0, 1.0, 2.0};
    const std::vector<float> first_values{8.0f, 9.0f, 7.0f};
    const std::vector<double> second_starts{1.001, 2.002};
    const std::vector<float> second_values{8.5f, 6.0f};
    EXPECT_EQ(spelled(second_difference(first_starts, first_values, second_starts, second_values)),
              spelled({NAN, 0.5, 1.0}));
}

TEST(Difference, SummarizesThePairedPointsOnly) {
    const std::vector<double> differences{NAN, 5.0, 0.0, -2.0};
    const std::optional<DifferenceSummary> summary = summarize_difference(differences);
    ASSERT_TRUE(summary.has_value());
    EXPECT_DOUBLE_EQ(summary->mean, 1.0);
    EXPECT_EQ(summary->minimum, -2.0);
    EXPECT_EQ(summary->maximum, 5.0);
    EXPECT_EQ(summary->paired, 3);
    const std::vector<double> none{NAN, NAN};
    EXPECT_FALSE(summarize_difference(none).has_value());
}

TEST(Difference, LabelCutsEachNameAtACharacter) {
    EXPECT_EQ(difference_label("a", "b"), "a_minus_b");
    EXPECT_EQ(difference_label(std::string(61, 'a'), "b"), std::string(60, 'a') + "_minus_b");
    // "\xC3\xA9" is one character (é) in two bytes: kept whole or not at all.
    EXPECT_EQ(difference_label(std::string(59, 'a') + "\xC3\xA9\xC3\xA9", "b"),
              std::string(59, 'a') + "\xC3\xA9_minus_b");
}

// A CSV file's lines, each split at its commas.
std::vector<std::vector<std::string>> read_csv(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    std::vector<std::vector<std::string>> rows;
    std::string line;
    while (std::getline(file, line)) {
        EXPECT_TRUE(!line.empty() && line.back() == '\r');  // CR LF, as export_csv's
        line.pop_back();
        std::vector<std::string> cells;
        std::stringstream cells_text(line);
        for (std::string cell; std::getline(cells_text, cell, ',');) {
            cells.push_back(cell);
        }
        if (!line.empty() && line.back() == ',') {
            cells.emplace_back();  // the last cell, blank
        }
        rows.push_back(cells);
    }
    return rows;
}

TEST(Difference, CsvHasTheFirstsRowsAndColumnsAndTheDifferenceInThem) {
    ComparisonResult first;
    first.metric_results.add(FrameMetricResult{"vmaf", {0, 1, 2}, {0.0, 0.5, 1.0}, {90, 80, 70}, {}});
    first.metric_results.add(FrameMetricResult{"psnr", {0, 1, 2}, {0.0, 0.5, 1.0}, {40, 41, 42}, {}});
    first.metric_results.add(SequenceMetricResult{"cvvdp", 8.5, {}, {0, 2}, {0.0, 1.0}, {8.0f, 9.0f}});
    ComparisonResult second;  // no PSNR; VMAF from frame 1 on
    second.metric_results.add(FrameMetricResult{"vmaf", {1, 2, 3}, {0.5, 1.0, 1.5}, {75, 72, 60}, {}});
    second.metric_results.add(SequenceMetricResult{"cvvdp", 8.0, {}, {0, 2}, {0.0, 1.0}, {7.5f, 9.5f}});

    test::Folder folder;
    export_csv(first, folder / "first.csv");
    export_difference_csv(first, second, folder / "difference.csv");
    const std::vector<std::vector<std::string>> scores = read_csv(folder / "first.csv");
    const std::vector<std::vector<std::string>> rows = read_csv(folder / "difference.csv");
    ASSERT_EQ(rows.size(), 4u);
    EXPECT_EQ(rows[0], scores[0]);  // the same header, so scripts read both alike
    const auto column = [&rows](const std::string& name) {
        const auto found = std::find(rows[0].begin(), rows[0].end(), name);
        EXPECT_NE(found, rows[0].end()) << name;
        const auto at = static_cast<std::size_t>(found - rows[0].begin());
        std::vector<std::string> cells;
        for (std::size_t row = 1; row < rows.size(); ++row) {
            cells.push_back(rows[row].at(at));
        }
        return cells;
    };
    using Cells = std::vector<std::string>;
    EXPECT_EQ(column("frame"), (Cells{"0", "1", "2"}));
    EXPECT_EQ(column("time_s"), (Cells{"0.000000", "0.500000", "1.000000"}));
    EXPECT_EQ(column("vmaf"), (Cells{"", "5.0", "-2.0"}));
    EXPECT_EQ(column("psnr"), (Cells{"", "", ""}));  // the second has none
    EXPECT_EQ(column("cvvdp_second_jod"), (Cells{"0.5", "", "-0.5"}));
    EXPECT_EQ(column("cvvdp_video_jod"), (Cells{"0.5", "", "0.5"}));
}

}  // namespace
}  // namespace vml
