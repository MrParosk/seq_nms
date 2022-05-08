#include "../src/seq_nms.h"

template <typename T>
bool check_same_size(std::vector<T> v1, std::vector<T> v2) {
    if (v1.size() != v2.size()) {
        return false;
    } else {
        return true;
    }
}

bool graph_sequence_equal(box_seq_t box_seq, box_seq_t expected_box_seq) {
    // TODO: refactor this function

    if (!check_same_size(box_seq, expected_box_seq)) {
        return false;
    }

    for (int f_idx = 0; f_idx < box_seq.size(); f_idx++) {
        auto frame_box_seq = box_seq[f_idx];
        auto frame_expected_seq_box = expected_box_seq[f_idx];

        if (!check_same_size(frame_box_seq, frame_expected_seq_box)) {
            return false;
        }

        for (int b_idx = 0; b_idx < frame_box_seq.size(); b_idx++) {
            auto box_box_seq = frame_box_seq[b_idx];
            auto box_expected_seq_box = frame_expected_seq_box[b_idx];

            if (!check_same_size(box_box_seq, box_expected_seq_box)) {
                return false;
            }

            for (int e_idx = 0; e_idx < box_box_seq.size(); e_idx++) {
                if (box_box_seq[e_idx] != box_expected_seq_box[e_idx]) {
                    return false;
                }
            }
        }
    }
    return true;
}
