// The videos' colours on the graphs (the Python app's
// vmaf_app/ui/series_colors.py): one per video, the same on Metric Graphs
// and the Bitrate tab, for this session only. A video keeps its colour after
// its row is removed, so adding it again brings the same colour back.
#pragma once

#include <QColor>
#include <QObject>
#include <QString>
#include <QStyledItemDelegate>

#include <array>
#include <filesystem>
#include <map>

class QWidget;

namespace vml::app {

// The colours given out in turn.
inline constexpr std::array<const char*, 10> series_palette{"#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2",
                                                            "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD"};

// The window's video colours, changed on the window's thread only.
class VideoColors : public QObject {
    Q_OBJECT

public:
    explicit VideoColors(QObject* parent = nullptr);

    // A video's key: its full path, in one case on Windows, whose paths are
    // either.
    static QString path_key(const std::filesystem::path& path);
    // The video's colour: given the next from the palette the first time.
    QColor for_path(const std::filesystem::path& path);
    // Sets a video's colour; says so (changed) when it is a new one.
    void set_color(const std::filesystem::path& path, const QColor& color);
    // Asks the user for a video's colour, with a button back to its first:
    // the colour is shown in the dialog as it is chosen, and set on OK.
    void choose(const std::filesystem::path& path, QWidget* parent);

signals:
    void changed(const QString& path_key, const QColor& color);

private:
    std::map<QString, QColor> colors_;
    std::map<QString, QColor> defaults_;  // the colour each was first given
};

// Draws a table's colour squares, and says when one is clicked (only the
// square, not the tick box or the name). The square's place is the style's
// own, so it follows the style, the display's scale and the scrolling.
class ColorSwatchDelegate : public QStyledItemDelegate {
    Q_OBJECT

public:
    using QStyledItemDelegate::QStyledItemDelegate;

signals:
    void clicked(int row);

protected:
    bool editorEvent(QEvent* event, QAbstractItemModel* model, const QStyleOptionViewItem& option,
                     const QModelIndex& index) override;
};

}  // namespace vml::app
