// A new test video's row is selected when no row is, so that the Options
// panel shows its settings at once (FillColumnTable::select_row_if_none_is,
// called as the window adds test videos).
#include <QColor>
#include <QCoreApplication>
#include <QImage>
#include <QDropEvent>
#include <QFileInfo>
#include <QHeaderView>
#include <QItemSelectionModel>
#include <QMimeData>
#include <QPaintEvent>
#include <QScrollBar>
#include <QSignalSpy>
#include <QTableWidget>
#include <QTest>
#include <QUrl>

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdlib>
#include <vector>

#include "widgets.h"

namespace vml::app {
namespace {

std::vector<int> selected_rows(const FillColumnTable& table) {
    std::vector<int> rows;
    for (const QModelIndex& index : table.selectionModel()->selectedRows()) {
        rows.push_back(index.row());
    }
    std::sort(rows.begin(), rows.end());
    return rows;
}

TEST(TableSelection, ANewRowIsSelectedOnlyWhenNoRowIs) {
    // As the Videos table: whole rows, several at a time, its first column
    // (the tick box) narrow and the second filling the space.
    FillColumnTable table(0, 3, 1, {0, 2});
    table.setSelectionBehavior(QAbstractItemView::SelectRows);
    table.setSelectionMode(QAbstractItemView::ExtendedSelection);
    table.setColumnHidden(0, true);
    QSignalSpy changed(&table, &QTableWidget::itemSelectionChanged);

    // The first video added: its row is selected, the Options panel told.
    table.setRowCount(1);
    table.select_row_if_none_is(0);
    EXPECT_EQ(selected_rows(table), std::vector<int>{0});
    EXPECT_EQ(changed.count(), 1);
    // Its current cell is in the first column shown.
    EXPECT_EQ(table.currentIndex(), table.model()->index(0, 1));

    // More videos added: the row already selected stays the one selected.
    table.setRowCount(3);
    table.select_row_if_none_is(1);
    EXPECT_EQ(selected_rows(table), std::vector<int>{0});

    // The user's own selection of several rows is left as it is.
    table.selectionModel()->select(QItemSelection(table.model()->index(1, 0), table.model()->index(2, 2)),
                                   QItemSelectionModel::ClearAndSelect);
    table.setRowCount(4);
    table.select_row_if_none_is(3);
    EXPECT_EQ(selected_rows(table), (std::vector<int>{1, 2}));

    // With nothing selected again, the next video added is.
    table.clearSelection();
    table.setRowCount(5);
    table.select_row_if_none_is(4);
    EXPECT_EQ(selected_rows(table), std::vector<int>{4});

    // A row that isn't there selects nothing.
    table.clearSelection();
    table.select_row_if_none_is(5);
    table.select_row_if_none_is(-1);
    EXPECT_TRUE(selected_rows(table).empty());
}

// While the Videos table has rows, one is selected, so that the Options panel
// is never blank (FillColumnTable::keep_a_row_selected): a click where there
// is no row does nothing, and whatever else leaves none selected selects the
// last row selected again, or the nearest.
TEST(TableSelection, ARowStaysSelectedWhileThereAreRows) {
    FillColumnTable table(0, 3, 1, {0, 2});
    table.setSelectionBehavior(QAbstractItemView::SelectRows);
    table.setSelectionMode(QAbstractItemView::ExtendedSelection);
    table.setColumnHidden(0, true);
    table.keep_a_row_selected();
    table.resize(400, 300);
    table.show();
    ASSERT_TRUE(QTest::qWaitForWindowExposed(&table));

    // With no rows, nothing to keep: no row is made up.
    table.clearSelection();
    QCoreApplication::processEvents();
    EXPECT_TRUE(selected_rows(table).empty());

    table.setRowCount(3);
    table.select_row_if_none_is(1);
    ASSERT_EQ(selected_rows(table), std::vector<int>{1});

    // A click below the last row: no change.
    const QPoint below(50, table.rowViewportPosition(2) + table.rowHeight(2) + 40);
    ASSERT_FALSE(table.indexAt(below).isValid());
    QTest::mouseClick(table.viewport(), Qt::LeftButton, {}, below);
    QCoreApplication::processEvents();
    EXPECT_EQ(selected_rows(table), std::vector<int>{1});

    // Ctrl+click on the only row selected: it is selected again.
    const QRect second = table.visualRect(table.model()->index(1, 1));
    QTest::mouseClick(table.viewport(), Qt::LeftButton, Qt::ControlModifier, second.center());
    QCoreApplication::processEvents();
    EXPECT_EQ(selected_rows(table), std::vector<int>{1});

    // A click on a row still selects it, and only it.
    QTest::mouseClick(table.viewport(), Qt::LeftButton, {}, table.visualRect(table.model()->index(2, 1)).center());
    QCoreApplication::processEvents();
    EXPECT_EQ(selected_rows(table), std::vector<int>{2});

    // The row selected removed, the last: the one now last is selected, once
    // control is back in the event loop (not while rows are being removed).
    table.removeRow(2);
    QCoreApplication::processEvents();
    EXPECT_EQ(selected_rows(table), std::vector<int>{1});

    // Cleared in code, as when the window clears it: selected again.
    table.clearSelection();
    QCoreApplication::processEvents();
    EXPECT_EQ(selected_rows(table), std::vector<int>{1});

    // Every row removed: none to select.
    table.setRowCount(0);
    QCoreApplication::processEvents();
    EXPECT_TRUE(selected_rows(table).empty());
}

// With Highlight best/worst results ticked, a selected row's best and worst
// cells keep their green and red, the selection a band along their edges; a cell
// with no colour is selected as before. Without the delegate the selection
// filled every cell of the row, hiding the marks.
TEST(TableSelection, ASelectedRowKeepsItsBestAndWorstColours) {
    QTableWidget table(1, 2);
    table.setSelectionBehavior(QAbstractItemView::SelectRows);
    table.setItemDelegate(new MarkedCellDelegate(&table));
    const QColor green(0, 160, 0);
    table.setItem(0, 0, new QTableWidgetItem);
    table.setItem(0, 1, new QTableWidgetItem);
    table.item(0, 0)->setData(Qt::BackgroundRole, green);
    table.item(0, 1)->setData(Qt::BackgroundRole, QColor(0, 0, 0, 0));  // as apply_best_worst clears one
    table.resize(400, 200);
    table.show();
    // The marked cell current, in a table focused from the keyboard: the
    // style draws its focus frame over it, as macOS's does in this test
    // (#216). The frame's tint once put the cell's band 22 off the selection's colour.
    table.activateWindow();
    table.setFocus();
    table.setAttribute(Qt::WA_KeyboardFocusChange);
    const QModelIndex marked = table.model()->index(0, 0);
    table.selectionModel()->setCurrentIndex(marked, QItemSelectionModel::NoUpdate);
    QCoreApplication::processEvents();
    const QRect cell = table.visualRect(marked);
    const QPoint middle_at(cell.right() - 3, cell.center().y());
    const QColor unselected = table.viewport()->grab().toImage().pixelColor(middle_at);
    table.selectRow(0);
    QCoreApplication::processEvents();
    const QImage shot = table.viewport()->grab().toImage();
    const QColor selected = shot.pixelColor(table.visualRect(table.model()->index(0, 1)).center());
    // The marked cell's middle is as it is unselected (the style's own
    // drawing of its green and its focus frame); its top edge is the
    // selection's colour, as the plain selected cell shows it.
    const QColor middle = shot.pixelColor(middle_at);
    EXPECT_EQ(middle.rgb(), unselected.rgb()) << middle.name().toStdString() << " " << unselected.name().toStdString();
    EXPECT_GT(middle.green(), std::max(middle.red(), middle.blue()) + 100) << middle.name().toStdString();
    const QColor edge = shot.pixelColor(cell.right() - 3, cell.top());
    EXPECT_EQ(edge.rgb(), selected.rgb()) << edge.name().toStdString() << " " << selected.name().toStdString();
}

// Issue #202: files dropped from outside are said once the drop has returned,
// not inside it: a question the files start ("Already added") opened while
// the system's drag and drop still ran was placed at the window's corner.
// The drop handed to the table as the system's drag and drop hands it.
class DropTable : public FillColumnTable {
public:
    DropTable() : FillColumnTable(0, 3, 1, {0, 2}) {}
    using FillColumnTable::dropEvent;
};

TEST(TableSelection, DroppedFilesAreSaidAfterTheDrop) {
    DropTable table;
    const QString file = QCoreApplication::applicationFilePath();  // a file that is there
    QMimeData mime;
    mime.setUrls({QUrl::fromLocalFile(file)});
    QSignalSpy dropped(&table, &FillColumnTable::filesDropped);
    QDropEvent drop(QPointF(5, 5), Qt::CopyAction, &mime, Qt::LeftButton, Qt::NoModifier);
    table.dropEvent(&drop);
    EXPECT_TRUE(drop.isAccepted());
    EXPECT_EQ(dropped.count(), 0);  // not yet: the drop is still running
    QCoreApplication::processEvents();
    ASSERT_EQ(dropped.count(), 1);
    EXPECT_EQ(QFileInfo(dropped.front().front().toStringList().front()), QFileInfo(file));
}

// As the Videos table: the tick box and File name (the fill column), then
// metric columns, 20 in all; the first two are frozen once it is shown.
constexpr int frozen_test_columns = 20;

std::vector<int> all_but_the_file_name() {
    std::vector<int> others;
    for (int column = 0; column < frozen_test_columns; ++column) {
        if (column != 1) {
            others.push_back(column);
        }
    }
    return others;
}

void set_up_like_the_videos_table(FillColumnTable& table, QSize size) {
    table.setSelectionBehavior(QAbstractItemView::SelectRows);
    table.setSelectionMode(QAbstractItemView::ExtendedSelection);
    table.verticalHeader()->setVisible(false);
    for (int row = 0; row < table.rowCount(); ++row) {
        auto* tick = new QTableWidgetItem();
        tick->setFlags(Qt::ItemIsEnabled | Qt::ItemIsSelectable | Qt::ItemIsUserCheckable);
        tick->setCheckState(Qt::Unchecked);
        table.setItem(row, 0, tick);
        table.setItem(row, 1, new QTableWidgetItem(QStringLiteral("video %1.mkv").arg(row)));
    }
    for (int column = 2; column < frozen_test_columns; ++column) {
        table.setColumnWidth(column, 80);
    }
    table.setColumnWidth(0, 28);
    table.resize(size);
    table.freeze_leading_columns(2);
    table.show();
}

// The area a widget's paint events cover, from its making on.
class PaintedArea : public QObject {
public:
    explicit PaintedArea(QWidget& widget) { widget.installEventFilter(this); }
    long long area = 0;

protected:
    bool eventFilter(QObject*, QEvent* event) override {
        if (event->type() == QEvent::Paint) {
            for (const QRect& rect : static_cast<QPaintEvent*>(event)->region()) {
                area += static_cast<long long>(rect.width()) * rect.height();
            }
        }
        return false;
    }
};

// Paints what is waiting to be painted in `window` now: Qt paints on an
// update request it posts for later, not in processEvents.
void paint_now(QWidget& window) {
    QCoreApplication::processEvents();
    QEvent update(QEvent::UpdateRequest);
    QCoreApplication::sendEvent(&window, &update);
}

TEST(TableSelection, FrozenColumnsStayWhileTheOthersScroll) {
    FillColumnTable table(4, frozen_test_columns, 1, all_but_the_file_name());
    set_up_like_the_videos_table(table, QSize(500, 200));
    ASSERT_TRUE(QTest::qWaitForWindowExposed(&table));
    auto* frozen = table.findChild<QTableView*>(QStringLiteral("frozenColumns"));
    ASSERT_NE(frozen, nullptr);

    // Scrolled fully sideways, the File name cell is drawn where it was, and
    // a point there is its cell.
    const QRect name_cell = table.visualRect(table.model()->index(1, 1));
    QScrollBar* bar = table.horizontalScrollBar();
    ASSERT_GT(bar->maximum(), 0);
    bar->setValue(bar->maximum());
    EXPECT_EQ(table.visualRect(table.model()->index(1, 1)), name_cell);
    EXPECT_EQ(table.indexAt(name_cell.center()), table.model()->index(1, 1));
    // Left to right, frozen_ starts at the corner of the table's viewport.
    EXPECT_EQ(frozen->viewport()->mapToGlobal(QPoint()), table.viewport()->mapToGlobal(QPoint()));

    // Going to a cell scrolled under the frozen columns brings it out past
    // them; going to a frozen cell leaves the sideways scrolling alone.
    table.scrollTo(table.model()->index(0, 5));
    EXPECT_GE(table.columnViewportPosition(5), table.columnWidth(0) + table.columnWidth(1));
    const int scrolled = bar->value();
    table.scrollTo(table.model()->index(3, 1));
    EXPECT_EQ(bar->value(), scrolled);

    // A click on a frozen tick box ticks it once; a shift-click on another
    // row's File name selects the rows between, in the table's selection.
    QSignalSpy changed(&table, &QTableWidget::itemChanged);
    const QRect tick_cell = table.visualRect(table.model()->index(0, 0));
    QTest::mouseClick(frozen->viewport(), Qt::LeftButton, {}, tick_cell.center());
    EXPECT_EQ(changed.count(), 1);
    EXPECT_EQ(table.item(0, 0)->checkState(), Qt::Checked);
    QTest::mouseClick(frozen->viewport(), Qt::LeftButton, Qt::ShiftModifier,
                      table.visualRect(table.model()->index(2, 1)).center());
    EXPECT_EQ(selected_rows(table), (std::vector<int>{0, 1, 2}));

    // Right to left, the frozen columns are on the right, the tick box first.
    table.setLayoutDirection(Qt::RightToLeft);
    QCoreApplication::processEvents();
    const QRect name_rtl = table.visualRect(table.model()->index(1, 1));
    EXPECT_GT(name_rtl.center().x(), table.viewport()->width() / 2);
    EXPECT_GT(table.visualRect(table.model()->index(1, 0)).left(), name_rtl.right());
    EXPECT_EQ(table.indexAt(name_rtl.center()), table.model()->index(1, 1));
    // A click there, on frozen_ (which no longer starts where the table's
    // viewport does), reaches the table at the point clicked.
    const QRect tick_rtl = table.visualRect(table.model()->index(3, 0));
    QTest::mouseClick(frozen->viewport(), Qt::LeftButton, {},
                      frozen->viewport()->mapFrom(table.viewport(), tick_rtl.center()));
    EXPECT_EQ(table.item(3, 0)->checkState(), Qt::Checked);
}

TEST(TableSelection, FrozenColumnsLetTheTableScrollWithoutRepaintingIt) {
    // A step of scrolling, sideways or down, paints what comes into view, in
    // the table and in frozen_, and not the rest: Qt moves what is drawn
    // already only for a viewport no sibling widget lies over, and with one
    // over it (frozen_ as it was) it painted nothing at all, leaving the
    // cells where they were before the scroll. Left to right and right to
    // left; frozen_ and its heading stay at the leading edge.
    FillColumnTable table(40, frozen_test_columns, 1, all_but_the_file_name());
    set_up_like_the_videos_table(table, QSize(600, 300));
    ASSERT_TRUE(QTest::qWaitForWindowExposed(&table));
    auto* frozen = table.findChild<QTableView*>(QStringLiteral("frozenColumns"));
    ASSERT_NE(frozen, nullptr);
    QWidget* port = table.viewport();
    const long long port_area = static_cast<long long>(port->width()) * port->height();
    for (const Qt::LayoutDirection direction : {Qt::LeftToRight, Qt::RightToLeft}) {
        table.setLayoutDirection(direction);
        for (QScrollBar* bar : {table.horizontalScrollBar(), table.verticalScrollBar()}) {
            paint_now(table);  // what the direction's change painted
            const int x_before = table.horizontalHeader()->offset();
            const int y_before = table.verticalHeader()->offset();
            PaintedArea painted(*port);
            PaintedArea painted_frozen(*frozen->viewport());
            bar->setValue(bar->value() + bar->singleStep());
            paint_now(table);
            // The band that came into view, all along the viewport.
            const long long in_view =
                static_cast<long long>(std::abs(table.horizontalHeader()->offset() - x_before)) * port->height() +
                static_cast<long long>(std::abs(table.verticalHeader()->offset() - y_before)) * port->width();
            ASSERT_GT(in_view, 0);
            const long long all_painted = painted.area + painted_frozen.area;
            const char* step = bar == table.horizontalScrollBar() ? "sideways" : "down";
            const char* way = direction == Qt::LeftToRight ? "left to right" : "right to left";
            EXPECT_GE(all_painted, in_view) << way << ", " << step;
            EXPECT_LT(all_painted, port_area / 2) << way << ", " << step;
            const int leading_x = direction == Qt::LeftToRight ? 0 : port->width() - frozen->width();
            EXPECT_EQ(frozen->geometry().topLeft(), QPoint(leading_x, 0)) << way << ", " << step;
        }
    }
    // The frozen heading lies over the header, above frozen_.
    auto* heading = table.findChild<QHeaderView*>(QStringLiteral("frozenHeader"));
    ASSERT_NE(heading, nullptr);
    const QHeaderView* header = table.horizontalHeader();
    EXPECT_EQ(heading->mapToGlobal(QPoint()),
              QPoint(frozen->mapToGlobal(QPoint()).x(), header->mapToGlobal(QPoint()).y()));
    EXPECT_EQ(heading->size(), QSize(frozen->width(), header->height()));
}

}  // namespace
}  // namespace vml::app
