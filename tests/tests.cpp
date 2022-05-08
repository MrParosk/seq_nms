#include <gtest/gtest.h>
#include "../src/seq_nms.h"
#include "utils.cpp"

TEST(calculate_area, area_single_box) {
    auto boxes = torch::tensor({1, 2, 3, 4}, {torch::kFloat32});
    boxes = boxes.view({1, 1, 4});
    auto areas = calculate_area(boxes);

    auto expected_areas = torch::tensor({4}, {torch::kFloat32});
    expected_areas = expected_areas.view({1, 1});
    ASSERT_TRUE(torch::equal(expected_areas, areas));
}

TEST(calculate_area, area_multiple_boxes_first_idx) {
    auto boxes = torch::tensor({1, 2, 3, 4, 5, 6, 8, 9}, {torch::kFloat32});
    boxes = boxes.view({2, 1, 4});
    torch::Tensor areas = calculate_area(boxes);

    auto expected_areas = torch::tensor({4, 9}, {torch::kFloat32});
    expected_areas = expected_areas.view({2, 1});
    ASSERT_TRUE(torch::equal(expected_areas, areas));
}

TEST(calculate_area, area_multiple_boxes_second_idx) {
    auto boxes = torch::tensor({1, 2, 3, 4, 5, 6, 8, 9}, {torch::kFloat32});
    boxes = boxes.view({1, 2, 4});
    torch::Tensor areas = calculate_area(boxes);

    auto expected_areas = torch::tensor({4, 9}, {torch::kFloat32});
    expected_areas = expected_areas.view({1, 2});
    ASSERT_TRUE(torch::equal(expected_areas, areas));
}

TEST(calculate_iou_given_area, iou_overlap) {
    auto boxes_a = torch::tensor({1, 2, 3, 4, 10, 10, 20, 20}, {torch::kFloat32});
    boxes_a = boxes_a.view({2, 4});
    auto boxes_b = torch::tensor({1, 2, 2, 3, 20, 20, 30, 30}, {torch::kFloat32});
    boxes_b = boxes_b.view({2, 4});
    auto areas_a = torch::tensor({4, 100}, {torch::kFloat32});
    auto areas_b = torch::tensor({1, 100}, {torch::kFloat32});

    torch::Tensor ious = calculate_iou_given_area(boxes_a, boxes_b, areas_a, areas_b);

    auto expected_ious = torch::tensor({0.25, 0.0, 0.0, 0.0}, {torch::kFloat32});
    expected_ious = expected_ious.view({2, 2});

    ASSERT_TRUE(torch::equal(expected_ious, ious));
}

TEST(build_box_sequences, one_overlap) {
    double linkage_threshold = 0.1;

    auto boxes = torch::tensor({1, 2, 3, 4, 10, 10, 20, 20, 1, 2, 2, 3, 20, 20, 30, 30}, {torch::kFloat32});
    boxes = boxes.view({2, 2, 4});

    auto classes = torch::tensor({0, 0, 0, 0}, {torch::kInt32});
    classes = classes.view({2, 2});

    auto graph_sequences = build_box_sequences(boxes, classes, linkage_threshold);
    box_seq_t expected_sequence{{{0}, {}}};

    ASSERT_TRUE(graph_sequence_equal(graph_sequences, expected_sequence));
}

TEST(build_box_sequences, two_overlap) {
    double linkage_threshold = 0.1;

    auto boxes = torch::tensor({1, 2, 3, 4, 10, 10, 20, 20, 1, 2, 2, 3, 1, 2, 2, 3}, {torch::kFloat32});
    boxes = boxes.view({2, 2, 4});

    auto classes = torch::tensor({0, 0, 0, 0}, {torch::kInt32});
    classes = classes.view({2, 2});

    auto graph_sequences = build_box_sequences(boxes, classes, linkage_threshold);
    box_seq_t expected_sequence{{{0, 1}, {}}};

    ASSERT_TRUE(graph_sequence_equal(graph_sequences, expected_sequence));
}

TEST(build_box_sequences, test_threshold_filter) {
    double linkage_threshold = 0.5;

    auto boxes = torch::tensor({1, 2, 3, 4, 10, 10, 20, 20, 1, 2, 2, 3, 1, 2, 2, 3}, {torch::kFloat32});
    boxes = boxes.view({2, 2, 4});

    auto classes = torch::tensor({0, 0, 0, 0}, {torch::kInt32});
    classes = classes.view({2, 2});

    auto graph_sequences = build_box_sequences(boxes, classes, linkage_threshold);
    box_seq_t expected_sequence{{{}, {}}};

    ASSERT_TRUE(graph_sequence_equal(graph_sequences, expected_sequence));
}

TEST(build_box_sequences, test_class_filter) {
    double linkage_threshold = 0.1;

    auto boxes = torch::tensor({1, 2, 3, 4, 10, 10, 20, 20, 1, 2, 2, 3, 1, 2, 2, 3}, {torch::kFloat32});
    boxes = boxes.view({2, 2, 4});

    auto classes = torch::tensor({0, 0, 1, 1}, {torch::kInt32});
    classes = classes.view({2, 2});

    auto graph_sequences = build_box_sequences(boxes, classes, linkage_threshold);
    box_seq_t expected_sequence{{{}, {}}};

    ASSERT_TRUE(graph_sequence_equal(graph_sequences, expected_sequence));
}

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
