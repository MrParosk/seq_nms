#include <gtest/gtest.h>
#include "../src/custom_types.h"
#include "../src/sequence_utils.h"

TEST(find_highest_score_sequence, find_highest) {
    std::vector<score_indicies_list> sequences = {
        {std::make_tuple(0.3, std::vector<int>{3, 2}), std::make_tuple(0.2, std::vector<int>{4, 3})},
        {std::make_tuple(0.4, std::vector<int>{2, 1}), std::make_tuple(0.5, std::vector<int>{1, 0})},
        {}};

    auto best_tuple = find_highest_score_sequence(sequences);

    int expected_index = 1;
    EXPECT_FLOAT_EQ(std::get<0>(best_tuple), expected_index);

    std::vector<int> expected_indicies = {0, 1};
    EXPECT_EQ(std::get<1>(best_tuple), expected_indicies);

    float expected_score = 0.5;
    EXPECT_FLOAT_EQ(std::get<2>(best_tuple), expected_score);
}

TEST(find_best_sequence, full_length) {
    box_seq_t box_sequence = {{{0, 1}, {}}, {{0}, {}}};
    auto scores = torch::tensor({0.1, 0.15, 0.2, 0.05, 0.07, 0.08}, {torch::kFloat32});
    scores = scores.view({3, 2});

    auto best_tuple = find_best_sequence(box_sequence, scores);

    std::vector<int> expected_indicies = {0, 0, 0};
    EXPECT_EQ(std::get<1>(best_tuple), expected_indicies);

    float expected_score = 0.37;
    EXPECT_FLOAT_EQ(std::get<2>(best_tuple), expected_score);
}

TEST(find_best_sequence, subset) {
    box_seq_t box_sequence = {{{0, 1}, {}}, {{}, {}}};
    auto scores = torch::tensor({0.1, 0.15, 0.05, 0.2, 0.07, 0.08}, {torch::kFloat32});
    scores = scores.view({3, 2});

    auto best_tuple = find_best_sequence(box_sequence, scores);

    std::vector<int> expected_indicies = {0, 1};
    EXPECT_EQ(std::get<1>(best_tuple), expected_indicies);

    float expected_score = 0.3;
    EXPECT_FLOAT_EQ(std::get<2>(best_tuple), expected_score);
}
