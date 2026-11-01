// The graph's chart, held to the Python app's own (reference/chart.json,
// from its ChartWidget at 800x400), and to what drawing must keep: a narrow
// dip in a long line, and one drawing per size the window settles on.
#include "chart.h"

#include <QColor>
#include <QCoreApplication>
#include <QImage>
#include <QMouseEvent>
#include <QPainter>
#include <QStringList>
#include <QTimer>

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <cmath>
#include <cstring>
#include <limits>
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
    static void zoom_to(ChartWidget& chart, double start, double span) { chart.view_ = ChartWidget::View{start, span}; }
    // The chart's lines alone, on white, where the chart draws them, as a
    // screen at `ratio` gets them.
    static QImage lines(const ChartWidget& chart, double ratio) {
        QImage image(static_cast<int>(std::ceil(chart.width() * ratio)),
                     static_cast<int>(std::ceil(chart.height() * ratio)), QImage::Format_RGB32);
        image.setDevicePixelRatio(ratio);
        image.fill(Qt::white);
        QPainter painter(&image);
        for (const ChartSeries* series : chart.visible_series()) {
            chart.draw_series(painter, chart.plot_rect(), *series);
        }
        return image;
    }
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
        QStringList drawn;
        for (const auto& [px, label] : chart.time_ticks()) {
            drawn.append(label);
        }
        EXPECT_EQ(drawn, labels);
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

// Drags the chart `dx` pixels to the left, as the mouse does.
void drag_left(ChartWidget& chart, int dx) {
    const QPointF from = chart.plot_rect().center();
    const QPointF to = from - QPointF(dx, 0);
    QMouseEvent press(QEvent::MouseButtonPress, from, from, Qt::LeftButton, Qt::LeftButton, Qt::NoModifier);
    QMouseEvent move(QEvent::MouseMove, to, to, Qt::NoButton, Qt::LeftButton, Qt::NoModifier);
    QMouseEvent release(QEvent::MouseButtonRelease, to, to, Qt::LeftButton, Qt::NoButton, Qt::NoModifier);
    QCoreApplication::sendEvent(&chart, &press);
    QCoreApplication::sendEvent(&chart, &move);
    QCoreApplication::sendEvent(&chart, &release);
}

// The screen's pixel columns inside the plot, away from its edges (where
// lines come into view and leave it).
std::pair<int, int> inner_columns(const QRect& plot, double ratio) {
    constexpr int edge = 3;
    return {static_cast<int>(std::ceil(plot.left() * ratio)) + edge,
            static_cast<int>(std::floor((plot.right() + 1) * ratio)) - edge};
}

// Whether, within `columns`, `after` is `before` moved `shift` pixels to the
// left.
bool moved_left(const QImage& before, const QImage& after, std::pair<int, int> columns, int shift) {
    const int old_start = columns.first + std::max(0, shift);
    const int new_start = columns.first + std::max(0, -shift);
    const int width = columns.second - columns.first - std::abs(shift);
    for (int y = 0; y < before.height(); ++y) {
        const auto* old_row = reinterpret_cast<const QRgb*>(before.constScanLine(y));
        const auto* new_row = reinterpret_cast<const QRgb*>(after.constScanLine(y));
        if (std::memcmp(old_row + old_start, new_row + new_start, sizeof(QRgb) * width) != 0) {
            return false;
        }
    }
    return true;
}

// The middle of the inked pixels within two rows of `row`, or -1.
double ink_middle(const QImage& image, int row) {
    int first = -1;
    int last = -1;
    for (int y = std::max(0, row - 2); y <= std::min(image.height() - 1, row + 2); ++y) {
        for (int x = 0; x < image.width(); ++x) {
            if (image.pixel(x, y) != qRgb(255, 255, 255)) {
                first = first < 0 ? x : std::min(first, x);
                last = std::max(last, x);
            }
        }
    }
    return first < 0 ? -1.0 : (first + last + 1) / 2.0;
}

TEST_F(ChartTest, DraggingMovesTheLinesWithoutChangingThem) {
    // Dragged left and right, the lines move as they are: the frames a
    // pixel column sums up, and where a point falls within a pixel, do not
    // depend on where the view starts. Zoomed to many frames a column, to
    // about as many frames as a line is drawn point by point, and to few;
    // on a screen at 100%, 125% and 175% (where a pixel of the mouse is 1.75
    // of the screen's). A line with a gap; one low frame, at the axis's
    // bottom, and one at VMAF's 100, drawn on the row above the plot, each
    // within a pixel or two of where its time is.
    constexpr int count = 20'000;
    constexpr int low_frame = 3000;   // 125 s: in every view below
    constexpr int high_frame = 3050;  // 127 s
    std::vector<double> times(count);
    std::vector<float> values(count);
    unsigned noise = 12345;
    for (int i = 0; i < count; ++i) {
        times[i] = i / 24.0;
        noise = noise * 1103515245U + 12345U;
        values[i] = static_cast<float>(70.0 + 20.0 * std::sin(i * 0.05) + (noise >> 16) % 17 - 8.0);
    }
    values[low_frame] = 20.0F;  // the others are 42 to 98
    values[high_frame] = 100.0F;
    for (int i = 3100; i < 3105; ++i) {
        values[i] = std::numeric_limits<float>::quiet_NaN();
    }
    ChartWidget chart(QStringLiteral("VMAF"), 100.0);
    chart.resize(800, 400);
    chart.set_series(0, line_of(std::move(times), std::move(values)));
    const QRect plot = chart.plot_rect();
    ASSERT_EQ(chart.y_range(), std::pair(20.0, 100.0));
    for (const double frames_in_view : {3000.0, 2.0 * plot.width() - 1, 300.0}) {
        for (const double ratio : {1.0, 1.25, 1.75}) {
            const std::pair<int, int> columns = inner_columns(plot, ratio);
            const std::string where = std::to_string(frames_in_view) + " frames in view at " + std::to_string(ratio);
            const auto expect_frame_at = [&](const QImage& image, int frame, double row) {
                const auto [x0, x1] = chart.x_range();
                const double x = (plot.left() + (frame / 24.0 - x0) / (x1 - x0) * plot.width()) * ratio;
                EXPECT_NEAR(ink_middle(image, static_cast<int>(row * ratio)), x, 2.0) << where << ", frame " << frame;
            };
            zoom_to(chart, 120.123457, frames_in_view / 24.0);
            QImage before = lines(chart, ratio);
            for (const int dx : {1, 3, 10, -7}) {
                drag_left(chart, dx);
                const QImage after = lines(chart, ratio);
                // The drawing moves by the whole screen pixels the drag's
                // 1.75-pixel steps add up to: down or up.
                const auto shift_down = static_cast<int>(std::floor(dx * ratio));
                const auto shift_up = static_cast<int>(std::ceil(dx * ratio));
                EXPECT_TRUE(moved_left(before, after, columns, shift_down) ||
                            moved_left(before, after, columns, shift_up))
                    << where << ", dragged " << dx;
                expect_frame_at(after, low_frame, plot.bottom());
                expect_frame_at(after, high_frame, plot.top() - 1);
                before = after;
            }
        }
    }
}

}  // namespace
}  // namespace vml::app
