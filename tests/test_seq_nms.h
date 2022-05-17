#include <gtest/gtest.h>
#include "../src/seq_nms.h"

TEST(build_box_sequences, one_overlap) {
    double linkage_threshold = 0.1;

    auto boxes = torch::tensor({1, 2, 3, 4, 10, 10, 20, 20, 1, 2, 2, 3, 20, 20, 30, 30}, {torch::kFloat32});
    boxes = boxes.view({2, 2, 4});

    auto classes = torch::tensor({0, 0, 0, 0}, {torch::kInt32});
    classes = classes.view({2, 2});

    auto graph_sequences = build_box_sequences(boxes, classes, linkage_threshold);
    box_seq_t expected_sequence{{{0}, {}}};

    ASSERT_EQ(graph_sequences, expected_sequence);
}

TEST(build_box_sequences, two_overlap) {
    double linkage_threshold = 0.1;

    auto boxes = torch::tensor({1, 2, 3, 4, 10, 10, 20, 20, 1, 2, 2, 3, 1, 2, 2, 3}, {torch::kFloat32});
    boxes = boxes.view({2, 2, 4});

    auto classes = torch::tensor({0, 0, 0, 0}, {torch::kInt32});
    classes = classes.view({2, 2});

    auto graph_sequences = build_box_sequences(boxes, classes, linkage_threshold);
    box_seq_t expected_sequence{{{0, 1}, {}}};

    ASSERT_EQ(graph_sequences, expected_sequence);
}

TEST(build_box_sequences, test_threshold_filter) {
    double linkage_threshold = 0.5;

    auto boxes = torch::tensor({1, 2, 3, 4, 10, 10, 20, 20, 1, 2, 2, 3, 1, 2, 2, 3}, {torch::kFloat32});
    boxes = boxes.view({2, 2, 4});

    auto classes = torch::tensor({0, 0, 0, 0}, {torch::kInt32});
    classes = classes.view({2, 2});

    auto graph_sequences = build_box_sequences(boxes, classes, linkage_threshold);
    box_seq_t expected_sequence{{{}, {}}};

    ASSERT_EQ(graph_sequences, expected_sequence);
}

TEST(build_box_sequences, test_class_filter) {
    double linkage_threshold = 0.1;

    auto boxes = torch::tensor({1, 2, 3, 4, 10, 10, 20, 20, 1, 2, 2, 3, 1, 2, 2, 3}, {torch::kFloat32});
    boxes = boxes.view({2, 2, 4});

    auto classes = torch::tensor({0, 0, 1, 1}, {torch::kInt32});
    classes = classes.view({2, 2});

    auto graph_sequences = build_box_sequences(boxes, classes, linkage_threshold);
    box_seq_t expected_sequence{{{}, {}}};

    ASSERT_EQ(graph_sequences, expected_sequence);
}
