#include <gtest/gtest.h>
#include "../src/seq_nms.h"


TEST(calculate_area, test_calculate_area_single_box) {
    auto boxes = torch::tensor({1, 2, 3, 4}, {torch::kFloat32});
    boxes = boxes.view({1, 1, 4});
    auto areas = calculate_area(boxes);

    std::cout << areas.sizes();

    auto expected_areas = torch::tensor({4}, {torch::kFloat32});
    expected_areas = expected_areas.view({1, 1});
    ASSERT_TRUE(torch::equal(expected_areas, areas));
}


int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}