// Reusable Qt widgets that know nothing of the app's data or columns (the
// Python app's vmaf_app/ui/widgets.py). Each exists because Qt's own widget
// is slightly wrong for how the app uses it.
#pragma once

#include <QFont>
#include <QHeaderView>
#include <QLabel>
#include <QList>
#include <QPushButton>
#include <QMap>
#include <QScrollArea>
#include <QSlider>
#include <QSize>
#include <QStringList>
#include <QStyledItemDelegate>
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

// Draws a selected cell that has a background of its own (the Videos tab's
// best and worst scores, green and red) in that background, with the
// selection as bands along its edges; any other cell as Qt does. VML
// Modern's style sheet gives tables a selection-background-color, which a
// style sheet paints over the cell's own background: with Highlight
// best/worst results ticked, the selected video's best and worst could not
// be seen. In VML Classic the mark showed with no sign of the selection.
class MarkedCellDelegate : public QStyledItemDelegate {
public:
    using QStyledItemDelegate::QStyledItemDelegate;

    void paint(QPainter* painter, const QStyleOptionViewItem& option, const QModelIndex& index) const override;
};

// A QTableWidget where one column (`fill_column`) always grows to fill the
// space the others leave, while the user can still resize it by dragging.
// Qt's own Stretch mode fills the space too, but stops the column being
// dragged at all. It also takes files dropped from outside (filesDropped) and
// its own rows dragged within it (rowsMoved), moving nothing itself: what a
// row is lives with its owner. Its leading columns can stay in place while
// the others scroll sideways (freeze_leading_columns).
class FillColumnTable : public QTableWidget {
    Q_OBJECT

public:
    FillColumnTable(int rows, int columns, int fill_column, std::vector<int> other_columns, QWidget* parent = nullptr);

    void setHorizontalHeader(QHeaderView* header);

    // The first `count` columns as shown stay in place while the others
    // scroll sideways under them: a view of those columns alone (frozen_)
    // lies over the table's leading edge, sharing its rows, selection and
    // delegate. Called once the table is set up (header, delegate, drags).
    void freeze_leading_columns(int count);

    // The cell at a point, and where a cell is, as the user sees them: a
    // frozen column's cells where frozen_ draws them.
    QModelIndex indexAt(const QPoint& position) const override;
    QRect visualRect(const QModelIndex& index) const override;
    // A frozen cell scrolls only the rows; another one is brought out past
    // the frozen columns, not left under them.
    void scrollTo(const QModelIndex& index, ScrollHint hint = EnsureVisible) override;

    // The local files (not folders) a drag carries.
    static QStringList dropped_files(const QMimeData* mime);

    // Whether drops are taken (not during a run).
    bool drops_allowed = true;

    // Selects `row` when no row is selected, as new rows are added: so that
    // what is shown for the selection (the Options panel) is there without a
    // click first. A selection the user made is left as it is.
    void select_row_if_none_is(int row);

    // While the table has rows, one of them stays selected (what is shown for
    // the selection, the Options panel, is then always there): a click where
    // there is no row does nothing, and whatever else would leave none
    // selected (Ctrl+click on the only one, a drag over no row, removing the
    // rows selected) selects the last row selected again, or the nearest.
    void keep_a_row_selected();

signals:
    // Files dropped on the table from outside it: their paths, once the
    // drop has returned (issue #202).
    void filesDropped(const QStringList& paths);
    // Rows dragged within the table: the rows moved, in order, and the row
    // they go before (rowCount() for the end).
    void rowsMoved(const QList<int>& rows, int target);

protected:
    void mousePressEvent(QMouseEvent* event) override;
    void resizeEvent(QResizeEvent* event) override;
    void dragEnterEvent(QDragEnterEvent* event) override;
    void dragMoveEvent(QDragMoveEvent* event) override;
    void dragLeaveEvent(QDragLeaveEvent* event) override;
    void dropEvent(QDropEvent* event) override;
    void paintEvent(QPaintEvent* event) override;
    void focusInEvent(QFocusEvent* event) override;
    void focusOutEvent(QFocusEvent* event) override;
    void updateGeometries() override;
    bool eventFilter(QObject* watched, QEvent* event) override;

private:
    friend class FrozenColumnsView;  // paints the drop line
    enum class DropKind { none, rows, files };

    DropKind drop_kind(QDropEvent* event) const;
    int insertion_row(QDropEvent* event) const;
    void set_drop_line(std::optional<int> row);
    // Draws the drop line across `port` (the table's viewport or frozen_'s).
    void paint_drop_line(QWidget* port) const;
    void on_section_resized(int logical_index, int old_size, int new_size);
    void recalculate_fill_column();
    bool is_frozen(int logical_column) const;
    // Where frozen_'s columns are seen, in the viewport's coordinates.
    QRect frozen_strip() const;
    void place_frozen_view();

    std::optional<int> drop_line_;  // the row a dragged row would go before, while dragging
    QTableView* frozen_ = nullptr;   // freeze_leading_columns' view, over the table's leading edge
    int frozen_count_ = 0;
    // The header's sort arrow as frozen_ last drew it: (shown, column, order).
    std::tuple<bool, int, Qt::SortOrder> frozen_sort_{false, -1, Qt::AscendingOrder};
    bool keeps_a_row_ = false;      // keep_a_row_selected
    int last_selected_row_ = 0;     // the row selected last, for keep_a_row_selected
    bool reselect_pending_ = false; // keep_a_row_selected will select one
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

// A slider that shades a band of its values, under its handle: Video
// Compare's timeline, the frames a loop repeats (issue #201). The band is
// drawn at least min_band_px wide, so that a short one still shows on a
// long range.
class BandSlider : public QSlider {
    Q_OBJECT

public:
    static constexpr int min_band_px = 6;
    explicit BandSlider(Qt::Orientation orientation, QWidget* parent = nullptr);
    // The values shaded, first and last included; none for no band.
    void set_band(std::optional<std::pair<int, int>> band);
    std::optional<std::pair<int, int>> band() const { return band_; }
    // Where the band is drawn, in the slider's pixels; empty for none.
    QRect band_rect() const;

protected:
    void paintEvent(QPaintEvent* event) override;

private:
    std::optional<std::pair<int, int>> band_;
};

// One line of text cut to the width it is given, ending in "…", with the
// whole text in the tooltip when it is cut (or a shorter text is shown in
// its place). A plain QLabel asks for its text's full width, so one long
// line would set the window's minimum width; this one asks for no width at
// all.
class ElidedLabel : public QLabel {
    Q_OBJECT

public:
    explicit ElidedLabel(QWidget* parent = nullptr);

    // Sets the text, and a tooltip to show with it.
    void set_text(const QString& text, const QString& tooltip = QString());
    // Sets the text, and a shorter one shown whole where it does not fit
    // and the shorter one does; cut where neither fits.
    void set_text_or_shorter(const QString& text, const QString& shorter);
    // The whole text, before cutting.
    QString full_text() const { return full_; }
    // Clears the text and the tooltip.
    void clear_text();
    // Where the text is cut: at its end (the default), or in its middle (a
    // file name, whose end tells encodes apart).
    void set_elide_mode(Qt::TextElideMode mode);

    QSize minimumSizeHint() const override;

protected:
    void resizeEvent(QResizeEvent* event) override;
    void showEvent(QShowEvent* event) override;

private:
    void fit();

    QString full_;
    QString shorter_;  // none: only full_
    QString tooltip_;
    Qt::TextElideMode mode_ = Qt::ElideRight;
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

// A click on one of `tabs`' tabs leaves the focus where it was outside the
// pages, and nowhere in them: Qt moves the focus from the page it leaves to
// the first control of the one it shows (QStackedLayout), so once a box on a
// page had been clicked, every tab clicked then had its first box focused
// (Video Compare's Zoom, Videos' reference, BD-rate's Name). Switching tabs
// from the keyboard (Ctrl+Tab) still takes the focus along, as Qt has it.
void keep_focus_on_tab_click(QTabWidget* tabs);

// A QTabWidget that lays itself out again, and repaints, only when that can
// change something. Qt's QTabWidget lays itself out again at every
// LayoutRequest, and repaints the whole of itself with it, so a label's text
// changing on one of its pages (a run line, a playback time) would repaint
// every control on the tab.
//
// On the Mac, Control+Tab and Control+Shift+Tab switch its tabs, as Ctrl+Tab
// does on Windows (Qt's own is Command+Tab there, the system's app switcher).
class QuietTabWidget : public QTabWidget {
    Q_OBJECT

public:
    explicit QuietTabWidget(QWidget* parent = nullptr);

protected:
    bool event(QEvent* event) override;
#ifdef Q_OS_MACOS
    bool eventFilter(QObject* watched, QEvent* event) override;
#endif

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

// The font for text with a symbol the window's font lacks (Metric Graphs'
// "✕"): on Windows Segoe UI Symbol, which has them, at `font`'s size;
// elsewhere `font`. A character Segoe UI lacks makes Qt load its fallback
// fonts, Chinese, Japanese and Korean ones among them: some 22 MB, for as long
// as the app runs.
QFont symbol_font(const QFont& font);

// Video Compare's Play / Pause: "Play" or "Pause" in the button's own font,
// beside its symbol drawn (a triangle, two bars) in the button's text colour,
// and greyed as the text is when it is disabled. Text symbols ("▶ Play",
// "❚❚ Pause") needed Segoe UI Symbol, which has no bold: VML Modern's primary
// buttons are semibold, and drawn bold the triangle came out chipped; and the
// two bars were two letters, set apart as letters are.
class PlayPauseButton : public QPushButton {
public:
    explicit PlayPauseButton(QWidget* parent = nullptr);

    // Shows Pause while playing, Play otherwise.
    void set_playing(bool playing);
    bool playing() const { return playing_; }

protected:
    void changeEvent(QEvent* event) override;
    bool event(QEvent* event) override;

private:
    // The text and the symbol, in the colours the button has now.
    void refresh();

    bool playing_ = false;
};

// Turns the mouse wheel's value changes off in the spin boxes and combo boxes
// on `page` (the Settings tab): the wheel scrolls the page under them
// instead, so scrolling it never changes a setting the pointer happens to
// pass over, nor gives one the focus for the arrow keys to change. Their open
// lists still scroll; the keyboard and clicks change values as before. Boxes
// anywhere else keep the wheel. Made before the page is first shown, it
// installs itself on the application and goes with the page. (The filter
// sees a wheel event at each widget it goes up through; at a box of the
// page's it drops it unread, ignored, so Qt passes it on to the box's
// parent, the page's scroll area at last.)
class WheelPastValueBoxes : public QObject {
public:
    explicit WheelPastValueBoxes(QWidget* page);

protected:
    bool eventFilter(QObject* watched, QEvent* event) override;

private:
    QWidget* page_;
};

// A table sized to its rows, up to `rows` of them: the tab scrolls, not the
// table, for a usual number of rows. Made again once the style is set: the
// header's and rows' heights follow it.
void fit_table_height(QTableWidget* table, int rows);

// `text` as a tooltip that wraps: Qt shows a plain-text tooltip as one line,
// however long (some ran the width of the screen), but wraps rich text. A
// plain text longer than a short line becomes rich text, its line breaks
// kept; a short one, and one already rich text, are left as they are.
QString wrapped_tool_tip(const QString& text);

// Every long tooltip of the app wrapped (wrapped_tool_tip), made once at the
// start and installed on the application: a widget's as it is set, and an
// item's (a table cell's, a list's or a dropdown's entry) as it is shown.
class WrapLongToolTips : public QObject {
public:
    explicit WrapLongToolTips(QObject* parent);

protected:
    bool eventFilter(QObject* watched, QEvent* event) override;
};

}  // namespace vml::app
