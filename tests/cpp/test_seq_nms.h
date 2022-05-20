#include <gtest/gtest.h>
#include "seq_nms.h"

using namespace torch::indexing;

TEST(build_box_sequences, one_overlap) {
    double linkage_threshold = 0.1;

    auto boxes = torch::tensor({1, 2, 3, 4, 10, 10, 20, 20, 1, 2, 2, 3, 20, 20, 30, 30}, {torch::kFloat32});
    boxes = boxes.view({2, 2, 4});

    auto areas = torch::tensor({4, 100, 1, 100});
    areas = areas.view({2, 2});

    auto classes = torch::tensor({0, 0, 0, 0}, {torch::kInt32});
    classes = classes.view({2, 2});

    auto graph_sequences = build_box_sequences(boxes, areas, classes, linkage_threshold);
    box_seq_t expected_sequence{{{0}, {}}};

    ASSERT_EQ(graph_sequences, expected_sequence);
}

TEST(build_box_sequences, two_overlap) {
    double linkage_threshold = 0.1;

    auto boxes = torch::tensor({1, 2, 3, 4, 10, 10, 20, 20, 1, 2, 2, 3, 1, 2, 2, 3}, {torch::kFloat32});
    boxes = boxes.view({2, 2, 4});

    auto areas = torch::tensor({4, 100, 1, 1});
    areas = areas.view({2, 2});

    auto classes = torch::tensor({0, 0, 0, 0}, {torch::kInt32});
    classes = classes.view({2, 2});

    auto graph_sequences = build_box_sequences(boxes, areas, classes, linkage_threshold);
    box_seq_t expected_sequence{{{0, 1}, {}}};

    ASSERT_EQ(graph_sequences, expected_sequence);
}

TEST(build_box_sequences, test_threshold_filter) {
    double linkage_threshold = 0.5;

    auto boxes = torch::tensor({1, 2, 3, 4, 10, 10, 20, 20, 1, 2, 2, 3, 1, 2, 2, 3}, {torch::kFloat32});
    boxes = boxes.view({2, 2, 4});

    auto areas = torch::tensor({4, 100, 1, 1});
    areas = areas.view({2, 2});

    auto classes = torch::tensor({0, 0, 0, 0}, {torch::kInt32});
    classes = classes.view({2, 2});

    auto graph_sequences = build_box_sequences(boxes, areas, classes, linkage_threshold);
    box_seq_t expected_sequence{{{}, {}}};

    ASSERT_EQ(graph_sequences, expected_sequence);
}

TEST(build_box_sequences, test_class_filter) {
    double linkage_threshold = 0.1;

    auto boxes = torch::tensor({1, 2, 3, 4, 10, 10, 20, 20, 1, 2, 2, 3, 1, 2, 2, 3}, {torch::kFloat32});
    boxes = boxes.view({2, 2, 4});

    auto areas = torch::tensor({4, 100, 1, 1});
    areas = areas.view({2, 2});

    auto classes = torch::tensor({0, 0, 1, 1}, {torch::kInt32});
    classes = classes.view({2, 2});

    auto graph_sequences = build_box_sequences(boxes, areas, classes, linkage_threshold);
    box_seq_t expected_sequence{{{}, {}}};

    ASSERT_EQ(graph_sequences, expected_sequence);
}

TEST(seq_nms, no_errors) {
    // This test simply checks that we can run the function without errors, but doesn't validate the results
    torch::manual_seed(42);

    int NUM_FRAMES = 100;

    auto width = 50.0 * torch::rand({NUM_FRAMES, 20});
    auto height = 50.0 * torch::rand({NUM_FRAMES, 20});
    auto x1 = 50.0 * torch::rand({NUM_FRAMES, 20});
    auto y1 = 50.0 * torch::rand({NUM_FRAMES, 20});

    auto boxes = torch::empty({NUM_FRAMES, 20, 4});
    boxes.index({Slice(), Slice(), 0}) = x1;
    boxes.index({Slice(), Slice(), 1}) = y1;
    boxes.index({Slice(), Slice(), 2}) = x1 + width;
    boxes.index({Slice(), Slice(), 3}) = y1 + height;

    auto scores = torch::rand({NUM_FRAMES, 20});

    auto classes = torch::randint(0, 10, {NUM_FRAMES, 20}, {torch::kInt32});

    float linkage_theshold = 0.3;
    float iou_threshold = 0.2;
    std::string metric = "avg";

    torch::Tensor scores_update = seq_nms(boxes, scores, classes, linkage_theshold, iou_threshold, metric);
    std::vector<int64_t> expected_size = {NUM_FRAMES, 20};
    ASSERT_EQ(scores_update.sizes(), expected_size);
}
