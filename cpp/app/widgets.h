// Reusable Qt widgets that know nothing of the app's data or columns (the
// Python app's vmaf_app/ui/widgets.py). Each exists because Qt's own widget
// is slightly wrong for how the app uses it.
#pragma once

#include <QFont>
#include <QHeaderView>
#include <QLabel>
#include <QList>
#include <QMap>
#include <QScrollArea>
#include <QSize>
#include <QStringList>
#include <QTabWidget>
#include <QTableWidget>

#include <optional>
#include <tuple>
#include <utility>
#include <vector>

class QMimeData;

namespace vml::app {

// A horizontal header where chosen sections have a checkbox, as FFMetrics'
// PSNR, SSIM and XPSNR columns do, so that which metrics are calculated is
// set right above the column the results go in. `checkable` maps a section's
// index to its first checked state; other sections are ordinary headers.
class CheckableHeaderView : public QHeaderView {
    Q_OBJECT

public:
    CheckableHeaderView(const QMap<int, bool>& checkable, QWidget* parent = nullptr);

    // Whether a section's checkbox is checked.
    bool is_checked(int section) const;
    // Sets a checkbox without sending sectionToggled, for bringing the header
    // in line with a change made elsewhere, which must not loop back.
    void set_checked(int section, bool value);
    // Where this section's checkbox is drawn, in viewport coordinates: the
    // geometry paintSection uses, so what can be clicked is what is seen.
    QRect section_indicator_rect(int section) const;

signals:
    void sectionToggled(int section, bool checked);

protected:
    void paintSection(QPainter* painter, const QRect& rect, int logical_index) const override;
    QSize sectionSizeFromContents(int logical_index) const override;
    void mousePressEvent(QMouseEvent* event) override;

private:
    QRect indicator_rect(const QRect& rect) const;

    QMap<int, bool> checked_;
};

// A QTableWidget where one column (`fill_column`) always grows to fill the
// space the others leave, while the user can still resize it by dragging.
// Qt's own Stretch mode fills the space too, but stops the column being
// dragged at all. It also takes files dropped from outside (filesDropped) and
// its own rows dragged within it (rowsMoved), moving nothing itself: what a
// row is lives with its owner.
class FillColumnTable : public QTableWidget {
    Q_OBJECT

public:
    FillColumnTable(int rows, int columns, int fill_column, std::vector<int> other_columns, QWidget* parent = nullptr);

    void setHorizontalHeader(QHeaderView* header);

    // The local files (not folders) a drag carries.
    static QStringList dropped_files(const QMimeData* mime);

    // Whether drops are taken (not during a run).
    bool drops_allowed = true;

signals:
    // Files dropped on the table from outside it: their paths.
    void filesDropped(const QStringList& paths);
    // Rows dragged within the table: the rows moved, in order, and the row
    // they go before (rowCount() for the end).
    void rowsMoved(const QList<int>& rows, int target);

protected:
    void resizeEvent(QResizeEvent* event) override;
    void dragEnterEvent(QDragEnterEvent* event) override;
    void dragMoveEvent(QDragMoveEvent* event) override;
    void dragLeaveEvent(QDragLeaveEvent* event) override;
    void dropEvent(QDropEvent* event) override;
    void paintEvent(QPaintEvent* event) override;

private:
    enum class DropKind { none, rows, files };

    DropKind drop_kind(QDropEvent* event) const;
    int insertion_row(QDropEvent* event) const;
    void set_drop_line(std::optional<int> row);
    void on_section_resized(int logical_index, int old_size, int new_size);
    void recalculate_fill_column();

    std::optional<int> drop_line_;  // the row a dragged row would go before, while dragging
    // What the drag over the table carries, found once when it enters: a
    // drag move comes with every mouse move, and finding it again then would
    // look at every dragged file on disk each time.
    DropKind drag_kind_ = DropKind::none;
    int fill_column_;
    std::vector<int> other_columns_;
    bool recalculating_ = false;
    // A manual drag of the fill column past the width that just fills the
    // space: without it, any later resize would set the column straight back.
    std::optional<int> fill_column_user_width_;
};

// One line of text cut to the width it is given, ending in "…", with the
// whole text in the tooltip when it is cut. A plain QLabel asks for its
// text's full width, so one long line would set the window's minimum width;
// this one asks for no width at all.
class ElidedLabel : public QLabel {
    Q_OBJECT

public:
    explicit ElidedLabel(QWidget* parent = nullptr);

    // Sets the text, and a tooltip to show with it.
    void set_text(const QString& text, const QString& tooltip = QString());
    // The whole text, before cutting.
    QString full_text() const { return full_; }
    // Clears the text and the tooltip.
    void clear_text();

    QSize minimumSizeHint() const override;

protected:
    void resizeEvent(QResizeEvent* event) override;
    void showEvent(QShowEvent* event) override;

private:
    void fit();

    QString full_;
    QString tooltip_;
};

// Lines of plain text that change as the pointer moves over a chart (its
// readout), drawn as a QLabel draws them. A QLabel suits text that is set
// now and then: at every new text it lays its parent out again, and, being
// see-through, has every widget behind it painted again, which at each move
// of the pointer was most of what the move cost. This one is opaque, in the
// colour of the tab it sits on, a new text repaints the text alone, and
// nothing is laid out again unless the text has outgrown it.
class ReadoutLabel : public QWidget {
    Q_OBJECT

public:
    explicit ReadoutLabel(const QString& text = QString(), QWidget* parent = nullptr);

    void set_text(const QString& text);
    const QString& text() const { return text_; }
    // How far the text is set in, inside the margins, from the top and the
    // side it starts on (the right for right-to-left text), and the room
    // kept after it, on the side it ends on.
    void set_indent(int indent, int trailing = 0);
    // Never less tall than `lines` lines of text, so that a text of fewer
    // lines (a hint, before the pointer comes) does not resize what is
    // around it.
    void reserve_lines(int lines);

    QSize sizeHint() const override;
    QSize minimumSizeHint() const override;

protected:
    void paintEvent(QPaintEvent* event) override;
    void changeEvent(QEvent* event) override;

private:
    QSize measure(const QString& text) const;
    int alignment() const;
    QRect text_rect() const;
    void remeasure();

    QString text_;
    bool right_to_left_ = false;
    // The text's size as QLabel measures it, worked out when the layout
    // asks, and the line count and longest line of the text it was for.
    mutable std::optional<QSize> text_size_;
    mutable std::pair<int, qsizetype> sized_for_{0, 0};
    int indent_ = 0;
    int trailing_ = 0;
    int reserved_lines_ = 0;
};

// A QTabWidget that lays itself out again, and repaints, only when that can
// change something. Qt's QTabWidget lays itself out again at every
// LayoutRequest, and repaints the whole of itself with it, so a label's text
// changing on one of its pages (a run line, a playback time) would repaint
// every control on the tab.
class QuietTabWidget : public QTabWidget {
    Q_OBJECT

public:
    explicit QuietTabWidget(QWidget* parent = nullptr);

protected:
    bool event(QEvent* event) override;

private:
    using Shape = std::tuple<QSize, QSize, QSize, std::optional<QSize>, std::optional<QSize>>;
    std::optional<Shape> laid_out_for_;
};

// A page in a scroll area, to be a tab of its own. While the window is large
// enough, the page fills the tab exactly as it would without the scroll
// area; when the window is smaller than the page needs, the tab scrolls.
class ScrollingPage : public QScrollArea {
    Q_OBJECT

public:
    explicit ScrollingPage(QWidget* page);

private:
    void show_focus(QWidget* old, QWidget* now);
};

// The font for text with a symbol the window's font lacks (Video Compare's
// "▶ Play" and "❚❚ Pause", Metric Graphs' "✕"): on Windows Segoe UI Symbol,
// which has them, at `font`'s size; elsewhere `font`. A character Segoe UI
// lacks makes Qt load its fallback fonts, Chinese, Japanese and Korean ones
// among them: some 22 MB, for as long as the app runs.
QFont symbol_font(const QFont& font);

}  // namespace vml::app
