// The graph's chart, held to the Python app's own (reference/chart.json,
// from its ChartWidget at 800x400), and to what drawing must keep: a narrow
// dip in a long line, and one drawing per size the window settles on.
#include "chart.h"

#include <QColor>
#include <QImage>
#include <QTimer>

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "reference.h"

namespace vml::app {

// The chart's parts, for its tests: the one class ChartWidget lets in.
class ChartTest : public ::testing::Test {
protected:
    // How many times the chart has been drawn into its kept pixmap.
    static int renders(const ChartWidget& chart) { return chart.renders_; }
    static int drawn_width(const ChartWidget& chart) { return chart.cache_ ? chart.cache_->width() : 0; }
    // The window's size has settled: the wait for it is over.
    static void settle(ChartWidget& chart) { chart.resize_settle_->stop(); }
};

namespace {

// A line whose arrays the series keeps alive.
ChartSeries line_of(std::vector<double> times, std::vector<float> values) {
    auto arrays = std::make_shared<std::pair<std::vector<double>, std::vector<float>>>(std::move(times),
                                                                                       std::move(values));
    return {arrays->first, arrays->second, arrays, QColor("#4C72B0"), true};
}

// Every frame's time at `per_second`, or the reference's own list.
std::vector<double> times_from(const nlohmann::json& data) {
    if (data.is_object()) {
        std::vector<double> times(data.at("count").get<std::size_t>());
        for (std::size_t i = 0; i < times.size(); ++i) {
            times[i] = static_cast<double>(i) / data.at("per_second").get<double>();
        }
        return times;
    }
    return test::reference_numbers(data);
}

std::vector<float> values_from(const nlohmann::json& data) {
    if (data.is_object()) {
        return std::vector<float>(data.at("count").get<std::size_t>(), data.at("value").get<float>());
    }
    return test::floats_from(data);
}

TEST_F(ChartTest, DrawsItsAxesAsThePythonApp) {
    const nlohmann::json reference = test::load_reference("chart.json");
    ASSERT_FALSE(reference.at("steps").empty());
    for (const nlohmann::json& step : reference.at("steps")) {
        const double span = *test::reference_number(step.at("span"));
        SCOPED_TRACE(span);
        EXPECT_TRUE(test::same_number(*test::reference_number(step.at("time")), nice_time_step(span)));
        EXPECT_TRUE(test::same_number(*test::reference_number(step.at("value")), nice_value_step(span)));
    }
    ASSERT_FALSE(reference.at("charts").empty());
    for (const nlohmann::json& item : reference.at("charts")) {
        SCOPED_TRACE(item.at("name").get<std::string>());
        const nlohmann::json& fixed = item.at("fixed_y_max");
        ChartWidget chart(QStringLiteral("axis"), fixed.is_null() ? std::nullopt : std::optional(fixed.get<double>()),
                          item.at("inverted").get<bool>());
        chart.resize(800, 400);
        chart.set_series(1, line_of(times_from(item.at("times")), values_from(item.at("values"))));

        const QRect rect = chart.plot_rect();
        EXPECT_EQ((std::vector<int>{rect.left(), rect.top(), rect.width(), rect.height()}),
                  item.at("plot_rect").get<std::vector<int>>());
        const auto number = [](const nlohmann::json& text) { return *test::reference_number(text); };
        EXPECT_EQ(chart.y_range(), std::pair(number(item.at("y_range")[0]), number(item.at("y_range")[1])));
        EXPECT_EQ(chart.x_range(), std::pair(number(item.at("x_range")[0]), number(item.at("x_range")[1])));
        QStringList labels;
        for (const nlohmann::json& label : item.at("time_labels")) {
            labels.append(QString::fromStdString(label.get<std::string>()));
        }
        EXPECT_EQ(chart.time_tick_labels(), labels);
        std::vector<std::pair<int, QString>> ticks;
        for (const nlohmann::json& tick : item.at("value_ticks")) {
            ticks.emplace_back(tick[0].get<int>(), QString::fromStdString(tick[1].get<std::string>()));
        }
        EXPECT_EQ(chart.value_ticks(), ticks);
        for (const nlohmann::json& pair : item.at("pixels_for_times")) {
            EXPECT_EQ(chart.pixel_for_time(number(pair[0])), pair[1].get<int>());
        }
        for (const nlohmann::json& pair : item.at("times_at")) {
            EXPECT_TRUE(test::same_number(number(pair[1]), chart.time_at(pair[0].get<int>())));
        }
        for (const nlohmann::json& pair : item.at("values_at")) {
            EXPECT_TRUE(test::same_number(number(pair[1]), chart.value_at(pair[0].get<int>())));
        }
        EXPECT_TRUE(test::same_number(number(item.at("seconds_per_pixel")), chart.seconds_per_pixel()));
    }
}

TEST_F(ChartTest, ANarrowDipSurvivesAndASettledSizeIsDrawnOnce) {
    // A 3-frame dip in 200,000 frames: drawn as one min/max stroke per pixel
    // column, it is still there, not averaged away or skipped. It is off its
    // column's first frame, which merely picking a frame per column would
    // keep, and off the axis's bottom, where the frame's own line is: the
    // first frame scores 0.
    constexpr std::size_t count = 200'000;
    constexpr std::size_t dip = 123'457;
    std::vector<double> times(count);
    std::vector<float> values(count, 95.0F);
    for (std::size_t i = 0; i < count; ++i) {
        times[i] = static_cast<double>(i) / 30.0;
    }
    values[0] = 0.0F;
    values[dip] = values[dip + 1] = values[dip + 2] = 10.0F;
    ChartWidget chart(QStringLiteral("VMAF"), 100.0);
    chart.resize(800, 400);
    chart.set_series(0, line_of(std::move(times), std::move(values)));
    const QImage image = chart.render_to_pixmap().toImage();
    const QRect rect = chart.plot_rect();
    ASSERT_EQ(chart.y_range(), std::pair(0.0, 100.0));
    // The dip's pixel column, and its row, between the 95s and the bottom.
    const int dip_x = rect.left() + static_cast<int>(dip * rect.width() / count);
    const int dip_row = rect.bottom() - static_cast<int>(0.10 * rect.height());
    bool inked = false;
    for (int x = dip_x - 1; x <= dip_x + 1; ++x) {
        for (int y = dip_row - 1; y <= dip_row + 1; ++y) {
            inked = inked || image.pixelColor(x, y).value() < 200;
        }
    }
    EXPECT_TRUE(inked);

    // Resized again and again (the window dragged by an edge), it is drawn
    // once, at the size it settles on; meanwhile it is shown stretched.
    chart.show();
    chart.grab();
    ASSERT_EQ(renders(chart), 1);
    for (int width = 810; width < 850; width += 10) {
        chart.resize(width, 400);
        chart.grab();
    }
    EXPECT_EQ(renders(chart), 1);
    settle(chart);
    chart.grab();
    EXPECT_EQ(renders(chart), 2);
    EXPECT_EQ(drawn_width(chart), 840);
    // Resized hidden (the window, on another tab): drawn afresh when shown,
    // not stretched from the old size.
    chart.hide();
    chart.resize(900, 400);
    chart.show();
    chart.grab();
    EXPECT_EQ(renders(chart), 3);
    EXPECT_EQ(drawn_width(chart), 900);
}

}  // namespace
}  // namespace vml::app
