// A new test video's row is selected when no row is, so that the Options
// panel shows its settings at once (FillColumnTable::select_row_if_none_is,
// called as the window adds test videos).
#include <QItemSelectionModel>
#include <QSignalSpy>

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

}  // namespace
}  // namespace vml::app
