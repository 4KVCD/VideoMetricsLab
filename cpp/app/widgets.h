// Reusable Qt widgets that know nothing of the app's data or columns (the
// Python app's vmaf_app/ui/widgets.py). Each exists because Qt's own widget
// is slightly wrong for how the app uses it.
#pragma once

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

}  // namespace vml::app
