// A new test video's row is selected when no row is, so that the Options
// panel shows its settings at once (FillColumnTable::select_row_if_none_is,
// called as the window adds test videos).
#include <QColor>
#include <QCoreApplication>
#include <QImage>
#include <QDropEvent>
#include <QFileInfo>
#include <QItemSelectionModel>
#include <QMimeData>
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
// cells keep their green and red, the selection a tint over them; a cell
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
    table.selectRow(0);
    QCoreApplication::processEvents();
    const QImage shot = table.viewport()->grab().toImage();
    const QRect cell = table.visualRect(table.model()->index(0, 0));
    const QRect plain = table.visualRect(table.model()->index(0, 1));
    const QColor selected = shot.pixelColor(plain.center());
    // The marked cell's middle is its green, as it is unselected; its top
    // edge is the selection's colour, as the plain selected cell shows it.
    // (The style blends an item's background a little.)
    const QColor middle = shot.pixelColor(cell.right() - 3, cell.center().y());
    EXPECT_LE(std::abs(middle.red() - green.red()) + std::abs(middle.green() - green.green()) +
                  std::abs(middle.blue() - green.blue()),
              40);
    const QColor edge = shot.pixelColor(cell.right() - 3, cell.top());
    EXPECT_LE(std::abs(edge.red() - selected.red()) + std::abs(edge.green() - selected.green()) +
                  std::abs(edge.blue() - selected.blue()),
              6);
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

TEST(TableSelection, FrozenColumnsStayWhileTheOthersScroll) {
    // As the Videos table: the tick box and File name (the fill column)
    // frozen, many metric columns beside them in a narrow table.
    constexpr int columns = 20;
    std::vector<int> others;
    for (int column = 0; column < columns; ++column) {
        if (column != 1) {
            others.push_back(column);
        }
    }
    FillColumnTable table(4, columns, 1, others);
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
    for (int column = 2; column < columns; ++column) {
        table.setColumnWidth(column, 80);
    }
    table.setColumnWidth(0, 28);
    table.resize(500, 200);
    table.freeze_leading_columns(2);
    table.show();
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
    // frozen_'s viewport lies exactly over the table's.
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
}

}  // namespace
}  // namespace vml::app
