// A new test video's row is selected when no row is, so that the Options
// panel shows its settings at once (FillColumnTable::select_row_if_none_is,
// called as the window adds test videos).
#include <QCoreApplication>
#include <QItemSelectionModel>
#include <QSignalSpy>
#include <QTest>

#include <gtest/gtest.h>

#include <algorithm>
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

}  // namespace
}  // namespace vml::app
