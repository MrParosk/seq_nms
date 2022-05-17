#include <gtest/gtest.h>
#include "test_box_utils.h"
#include "test_seq_nms.h"
#include "test_sequence_utils.h"

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
