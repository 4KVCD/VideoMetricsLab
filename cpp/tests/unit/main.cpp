// The engine's and the command line's tests, in a home folder of their own
// (private_home.h).
#include <gtest/gtest.h>

#include "private_home.h"

int main(int argc, char** argv) {
    const vml::testing::PrivateHome home;
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
