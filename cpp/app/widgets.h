// Reusable Qt widgets that know nothing of the app's data or columns (the
// Python app's vmaf_app/ui/widgets.py). Each exists because Qt's own widget
// is slightly wrong for how the app uses it.
#pragma once

#include <QColor>
#include <QFont>
#include <QFontMetrics>
#include <QIcon>
#include <QHeaderView>
#include <QItemSelectionModel>
#include <QLabel>
#include <QList>
#include <QPushButton>
#include <QMap>
#include <QScrollArea>
#include <QSlider>
#include <QSize>
#include <QStringList>
#include <QStyledItemDelegate>
#include <QTabBar>
#include <QTabWidget>
#include <QTableWidget>

#include <functional>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

class QMimeData;
class QToolButton;

namespace vml::app {

// A column heading on two lines where that makes it narrower: `text` split at
// the space that makes its wider line the narrowest, in `metrics` ("PSNR YUV
// (6:1:1)" -> "PSNR YUV\n(6:1:1)"); unchanged without a
// space. Qt's header measures and draws the two lines itself.
QString two_line_heading(const QString& text, const QFontMetrics& metrics);

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
    // The size a section's heading and its checkbox need, hidden sections
    // included (Qt's sectionSizeHint is 0 for those).
    QSize content_size(int section) const { return sectionSizeFromContents(section); }

signals:
    void sectionToggled(int section, bool checked);

protected:
    void paintSection(QPainter* painter, const QRect& rect, int logical_index) const override;
    QSize sectionSizeFromContents(int logical_index) const override;
    void mousePressEvent(QMouseEvent* event) override;
    void mouseDoubleClickEvent(QMouseEvent* event) override;

private:
    QRect indicator_rect(const QRect& rect) const;
    // Toggles the checkbox at `position`, if there is one there.
    bool toggle_at(const QPoint& position);

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
// A row's mark, a click on which acts on that row, drawn rather than a
// character (U+2BBE and the like are in few fonts) in the text's colour, its
// pen heavier and its shape tinted under the pointer:
// - `remove`: a circled cross that removes the row (Metric Graphs' lines, the
//   Videos table's videos). Not the danger colour: a row removed deletes
//   nothing (docs/ARCHITECTURE.md, "Removing and deleting").
// - `send`: an arrow out of a box, which sends the row's line back to
//   Metric Graphs (the Videos table's).
// The cell's text is not drawn. `tooltip`, where set, says what a click on
// the cell does (else the cell's own tooltip); where `off` says so, the mark
// is drawn disabled, and the caller ignores its clicks.
class RemoveMarkDelegate : public QStyledItemDelegate {
public:
    enum class Shape { remove, send };
    explicit RemoveMarkDelegate(QObject* parent = nullptr, Shape shape = Shape::remove)
        : QStyledItemDelegate(parent), shape_(shape) {}

    static constexpr int mark = 14;  // the shape's size, in pixels
    std::function<QString(const QModelIndex&)> tooltip;
    std::function<bool(const QModelIndex&)> off;

    void paint(QPainter* painter, const QStyleOptionViewItem& option, const QModelIndex& index) const override;
    QSize sizeHint(const QStyleOptionViewItem& option, const QModelIndex& index) const override;
    bool helpEvent(QHelpEvent* event, QAbstractItemView* view, const QStyleOptionViewItem& option,
                   const QModelIndex& index) override;

    // Hovering drawn on `view`: its viewport told of the pointer's moves
    // (Qt::WA_Hover), which an item view's viewport is not by default.
    static void track_hover(QAbstractItemView* view);

private:
    Shape shape_;
};

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
    // Columns whose cells a press doesn't select (a row mark's): the
    // selection stays as it was, and only the click acts.
    std::vector<int> unselecting_columns;

signals:
    // Files dropped on the table from outside it: their paths, once the
    // drop has returned (issue #202).
    void filesDropped(const QStringList& paths);
    // Rows dragged within the table: the rows moved, in order, and the row
    // they go before (rowCount() for the end).
    void rowsMoved(const QList<int>& rows, int target);

protected:
    void mousePressEvent(QMouseEvent* event) override;
    QItemSelectionModel::SelectionFlags selectionCommand(const QModelIndex& index,
                                                         const QEvent* event = nullptr) const override;
    void resizeEvent(QResizeEvent* event) override;
    void dragEnterEvent(QDragEnterEvent* event) override;
    void dragMoveEvent(QDragMoveEvent* event) override;
    void dragLeaveEvent(QDragLeaveEvent* event) override;
    void dropEvent(QDropEvent* event) override;
    void paintEvent(QPaintEvent* event) override;
    void focusInEvent(QFocusEvent* event) override;
    void focusOutEvent(QFocusEvent* event) override;
    void updateGeometries() override;
    void scrollContentsBy(int dx, int dy) override;
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
    // Puts frozen_ on the strip and its heading over the header.
    void place_frozen_view();
    // Where frozen_'s viewport starts in the table's.
    QPoint frozen_offset() const;

    std::optional<int> drop_line_;  // the row a dragged row would go before, while dragging
    QTableView* frozen_ = nullptr;   // freeze_leading_columns' view, over the table's leading edge
    QHeaderView* frozen_header_ = nullptr;  // frozen_'s heading, over the table's header
    int frozen_count_ = 0;
    // The header's sort arrow as frozen_ last drew it: (shown, column, order).
    std::tuple<bool, int, Qt::SortOrder> frozen_sort_{false, -1, Qt::AscendingOrder};
    // Redraws frozen_header_ if the header's sort arrow is not what it drew.
    void follow_sort_arrow();
    // follow_sort_arrow as `header`'s arrow changes.
    void watch_sort_arrow(QHeaderView* header);
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
// A line of text that goes on to more lines where it does not fit, never
// cut: Video Compare's frame line and the Videos tab's run lines, which list
// every metric. Its height is the lines its text needs at its width; it grows
// when a text needs more and stays while texts need fewer, so a line rewritten
// at each frame or progress tick does not move what is below it each time it
// changes. A new width, or clear_text(), fits it to its text again. It takes
// the width it is given (never widens the window).
class WrappingLabel : public QLabel {
    Q_OBJECT

public:
    explicit WrappingLabel(QWidget* parent = nullptr);

    // Sets the text, and a tooltip to show with it.
    void set_text(const QString& text, const QString& tooltip = QString());
    // `text` kept on one line where the label wraps: no-break spaces, and
    // a word joiner after each hyphen ("PSNR-HVS-M" is not broken at them).
    static QString unbroken(QString text);
    // `text` that may wrap at its spaces but not at its hyphens.
    static QString unhyphenated(QString text);
    // The text as given, without what unbroken() put in.
    QString full_text() const;
    // Clears the text and the tooltip, the height back to one line.
    void clear_text();

protected:
    void resizeEvent(QResizeEvent* event) override;
    void showEvent(QShowEvent* event) override;
    void changeEvent(QEvent* event) override;

private:
    // The height for the text at this width: grown to it, or set to it
    // when `may_shrink`.
    void fit_height(bool may_shrink);

    int fitted_width_ = -1;
};

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

// A tab widget inside one of the window's tabs (Settings' sections, Metric
// Graphs' metrics): Ctrl+Tab and Ctrl+Shift+Tab pass it by, to the window's
// tabs, wherever the focus is; Ctrl+PgDn and Ctrl+PgUp step through its own.
void pass_ctrl_tab_up(QTabWidget* tabs);

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

// A QTabBar that says when a tab is added or removed (QTabBar has no signal
// for it), for a TabPills showing its tabs. QuietTabWidget's bar is one.
class WatchedTabBar : public QTabBar {
    Q_OBJECT

public:
    using QTabBar::QTabBar;

signals:
    void tabs_changed();

protected:
    void tabInserted(int index) override;
    void tabRemoved(int index) override;
};

// A tab bar's tabs as a row of pills that wraps onto more rows where they do
// not fit, every name whole. A QTabBar scrolls instead, with arrows that
// leave a tab cut at one end (Metric Graphs' metrics, Worst frames'), and a
// sliver of the tab before when scrolled back. The bar stays the model:
// hidden, it keeps the tabs, their order and the current one; a pill clicked
// makes its tab current, as a click on the tab would, and the bar's own
// signals (currentChanged, tabBarClicked) still come.
//
// The pills follow the bar's current tab and, through a WatchedTabBar, its
// tabs. Where the bar's signals are blocked while it changes, rebuild() after.
// The widget takes the keyboard focus (Tab): Left, Right, Home and End move
// the current tab, as on a tab bar.
class TabPills : public QWidget {
    Q_OBJECT

public:
    explicit TabPills(QTabBar* bar, QWidget* parent = nullptr);

    // The pills made again from the bar's tabs, its current one checked.
    void rebuild();
    // The pills, in the bar's order (for tests).
    const std::vector<QToolButton*>& pills() const { return pills_; }

    bool hasHeightForWidth() const override { return true; }
    int heightForWidth(int width) const override;
    QSize sizeHint() const override;
    QSize minimumSizeHint() const override;

protected:
    bool event(QEvent* event) override;
    void resizeEvent(QResizeEvent* event) override;
    void keyPressEvent(QKeyEvent* event) override;
    void paintEvent(QPaintEvent* event) override;

private:
    // Places the pills in rows `width` wide (when `apply`) and returns the
    // height they take.
    int place(int width, bool apply) const;
    void schedule_rebuild();
    void check_current();

    QTabBar* bar_;
    std::vector<QToolButton*> pills_;
    bool rebuild_pending_ = false;
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

// An ⓘ, a circled "i" in `colour` (U+24D8 is not in every font), 16 px at
// `ratio`: a button that opens what something shows (Worst Frames' graph
// legend, Saved Results' details).
QIcon info_icon(const QColor& colour, qreal ratio);

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
