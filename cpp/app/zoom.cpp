#include "zoom.h"

#include <algorithm>
#include <cmath>
#include <string>

#include "vml/number_format.h"
#include "vml/python_text.h"

namespace vml::app {

namespace {

// `centre` (a fraction of the frame) kept so that a frame larger than the
// view covers all of it; a smaller one is centred.
double centred(double centre, double view, double size) {
    if (size <= view) {
        return 0.5;
    }
    const double margin = view / (2 * size);
    return std::min(std::max(centre, margin), 1 - margin);
}

}  // namespace

Placement Zoom::placement(Extent view, Extent frame, double ratio) const {
    const auto [view_width, view_height] = view;
    const auto [frame_width, frame_height] = frame;
    const double scale =
        factor ? *factor / ratio : std::min(view_width / frame_width, view_height / frame_height);
    const double width = frame_width * scale;
    const double height = frame_height * scale;
    return {view_width / 2 - centred(centre.first, view_width, width) * width,
            view_height / 2 - centred(centre.second, view_height, height) * height, width, height};
}

bool Zoom::can_drag(Extent view, Extent frame, double ratio) const {
    const Placement place = placement(view, frame, ratio);
    return place.width > view.first + 0.5 || place.height > view.second + 0.5;
}

bool Zoom::drag(double dx, double dy, Extent view, Extent frame, double ratio) {
    if (!factor) {
        return false;
    }
    const Placement place = placement(view, frame, ratio);
    const std::pair<double, double> current{(view.first / 2 - place.x) / place.width,
                                            (view.second / 2 - place.y) / place.height};
    const std::pair<double, double> moved{centred(current.first - dx / place.width, view.first, place.width),
                                          centred(current.second - dy / place.height, view.second, place.height)};
    if (moved == current) {
        return false;
    }
    centre = moved;
    return true;
}

std::optional<double> parse_percent(const QString& text) {
    // As Python's float() reads what is left: surrounding spaces allowed.
    QString cleaned = text;
    cleaned.remove('%').replace(',', '.');
    const std::optional<double> percent = python_float(cleaned.toStdString());
    if (!percent || !std::isfinite(*percent)) {
        return std::nullopt;
    }
    return std::min(std::max(*percent / 100, minimum_zoom), maximum_zoom);
}

QString percent_text(double factor) {
    // As Python's f"{round(factor * 100, 1):g}%": round() gives the double
    // nearest the correctly rounded tenth, as reading that tenth back does.
    const double rounded = python_float(format_fixed(factor * 100, 1)).value_or(factor * 100);
    return QString::fromStdString(format_general(rounded) + "%");
}

}  // namespace vml::app
