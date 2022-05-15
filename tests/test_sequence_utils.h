#include <gtest/gtest.h>
#include "../src/seq_nms.h"
#include "../src/sequence_utils.h"

TEST(find_best_sequence, full_length) {
    box_seq_t box_sequence = {{{0, 1}, {}}, {{0}, {}}};
    auto scores = torch::tensor({0.1, 0.15, 0.2, 0.05, 0.07, 0.08}, {torch::kFloat32});
    scores = scores.view({3, 2});

    auto best_tuple = find_best_sequence(box_sequence, scores);

    EXPECT_FLOAT_EQ(std::get<2>(best_tuple), 0.37);
    // ASSERT_THAT(std::get<1>(best_tuple), ElementsAre(0, 0, 0));
}

// TEST(find_best_sequence, subset) {
//     box_seq_t box_sequence = {{{0, 1}, {}}, {{}, {}}};
//     auto scores = torch::tensor({0.1, 0.15, 0.2, 0.05, 0.07, 0.08}, {torch::kFloat32});
//     scores = scores.view({3, 2});

//     auto best_tuple = find_best_sequence(box_sequence, scores);

//     EXPECT_FLOAT_EQ(std::get<2>(best_tuple), 0.3);
// }
