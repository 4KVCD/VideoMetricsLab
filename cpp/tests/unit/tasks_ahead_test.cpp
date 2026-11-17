// TasksAhead (tasks_ahead.h): the command line reads every video at once and
// takes each read in the command's order.
#include "vml/tasks_ahead.h"

#include <gtest/gtest.h>

#include <stdexcept>

namespace vml {
namespace {

TEST(TasksAhead, EachIndexGetsItsOwnResultOrWhatItThrew) {
    TasksAhead<int> tasks(20, 3, [](std::size_t index) {
        if (index == 7) {
            throw std::runtime_error("unreadable");
        }
        return static_cast<int>(index * index);
    });
    EXPECT_EQ(tasks.get(19), 361);  // taken in any order
    for (std::size_t index = 0; index < 19; ++index) {
        if (index == 7) {
            EXPECT_THROW(tasks.get(index), std::runtime_error);  // the others still run
        } else {
            EXPECT_EQ(tasks.get(index), static_cast<int>(index * index));
        }
    }
}

TEST(TasksAhead, NoTasksAndTasksNeverTakenEndCleanly) {
    { TasksAhead<int> none(0, 8, [](std::size_t) { return 0; }); }
    TasksAhead<int> untaken(5, 2, [](std::size_t index) { return static_cast<int>(index); });
    EXPECT_EQ(untaken.get(0), 0);  // the rest finish or are not started as it ends
}

}  // namespace
}  // namespace vml
