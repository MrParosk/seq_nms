#include "sequence_utils.h"

std::tuple<int, std::vector<int>, float> find_highest_score_sequence(const std::vector<score_indicies_list>& sequence_roots) {
    float best_score = 0.0;
    std::vector<int> best_sequence;
    int sequence_frame_index = 0;

    for (int s_idx = 0; s_idx < sequence_roots.size(); s_idx++) {
        score_indicies_list frame_sequences = sequence_roots[s_idx];

        if (frame_sequences.size() == 0) {
            continue;
        }

        std::vector<float> score_list;
        for (score_indicies s : frame_sequences) {
            score_list.push_back(std::get<0>(s));
        }

        int max_idx = torch::argmax(torch::tensor(score_list)).item<int>();

        if (std::get<0>(frame_sequences[max_idx]) > best_score) {
            best_score = std::get<0>(frame_sequences[max_idx]);

            std::vector<int> best_sequence = std::get<1>(frame_sequences[max_idx]);
            std::reverse(best_sequence.begin(), best_sequence.end());

            sequence_frame_index = s_idx;
        }
    }

    return std::make_tuple(sequence_frame_index, best_sequence, best_score);
}

std::tuple<int, std::vector<int>, float> find_best_sequence(const box_seq_t& box_graph, torch::Tensor& scores) {
    // list of tuples storing (score up to current frame, path up to current frame)
    // we dynamically build up best paths through graph starting from the end frame
    // s.t we can determine the beginning of sequences i.e. if there are no links
    // to a box from previous frames, then it is a candidate for starting a sequence
    std::vector<score_indicies_list> max_scores_paths;

    // list of all independent sequences where a given row corresponds to starting frame
    std::vector<score_indicies_list> sequence_roots;

    score_indicies_list max_scores;
    for (int idx = 0; idx < scores.size(1); idx++) {
        float score = scores.index({scores.size(0) - 1, idx}).item<float>();
        auto index_list = std::vector<int>{idx};
        max_scores.push_back(std::make_tuple(score, index_list));
    }

    max_scores_paths.push_back(max_scores);

    for (int reverse_idx = box_graph.size() - 1; reverse_idx >= 0; reverse_idx--) {
        auto frame_edges = box_graph[reverse_idx];
        int frame_idx = box_graph.size() - reverse_idx - 1;

        auto used_in_sequence = torch::zeros(static_cast<long>(max_scores_paths.back().size()), {torch::kInt32});
        score_indicies_list max_path_frame;

        for (int box_idx = 0; box_idx < frame_edges.size(); box_idx++) {
            auto box_edges = frame_edges[box_idx];

            if (box_edges.size() == 0) {
                // no edges for current box so consider it a max path consisting of a single node
                float score = scores.index({frame_idx, box_idx}).item<float>();

                std::vector<int> indicies = {box_idx};
                max_path_frame.push_back(std::make_tuple(score, indicies));
            } else {
                // extend previous max paths
                // here we use box_edges list to index used_in_sequence list and mark boxes in corresponding frame t+1
                // as part of a sequence since we have links to them and can always make a better max path by making it longer
                // (no negative scores)
                std::vector<float> score_list;
                for (int e_idx : box_edges) {
                    used_in_sequence.index({e_idx}) = 1;
                    score_list.push_back(std::get<0>(max_scores_paths.back()[e_idx]));
                }

                int prev_idx = torch::argmax(torch::tensor(score_list)).item<int>();
                float score_so_far = std::get<0>(max_scores_paths.back()[box_edges[prev_idx]]);
                std::vector<int> path_so_far = std::get<1>(max_scores_paths.back()[box_edges[prev_idx]]);
                path_so_far.push_back(box_idx);

                max_path_frame.push_back(
                    std::make_tuple(scores.index({frame_idx, box_idx}).item<float>() + score_so_far, path_so_far));
            }
        }

        // create new sequence roots for boxes in frame at frame_idx + 1 that did not have links from boxes in frame_idx
        score_indicies_list new_sequence_roots;
        for (int idx = 0; idx < used_in_sequence.size(0); idx++) {
            if (used_in_sequence.index({idx}).item<int>() == 0) {
                new_sequence_roots.push_back(max_scores_paths.back()[idx]);
            }
        }

        sequence_roots.push_back(new_sequence_roots);
        max_scores_paths.push_back(max_path_frame);
    }

    // add sequences starting in begining frame as roots
    sequence_roots.push_back(max_scores_paths.back());

    // reverse sequence roots since built sequences from back to front
    std::reverse(sequence_roots.begin(), sequence_roots.end());

    return find_highest_score_sequence(sequence_roots);
}
