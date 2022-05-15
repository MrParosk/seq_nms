#include "sequence_utils.h"

std::tuple<int, std::vector<int>, float> find_highest_score_sequence(const std::vector<score_indicies_list>& sequences) {
    /*
    Find the element that has the highest score.

    sequences are expected to have a length of number of frames.
        For each element in sequences are expected to have a length number of boxes.
    */

    float best_score = 0.0;
    std::vector<int> best_sequence;
    int sequence_frame_index = 0;

    for (int f_idx = 0; f_idx < sequences.size(); f_idx++) {
        score_indicies_list frame_sequences = sequences[f_idx];

        if (frame_sequences.size() == 0) {continue;}

        auto score_tensor = torch::empty(frame_sequences.size(), {torch::kFloat32});
        for (int i = 0; i < frame_sequences.size(); i++) {
            score_tensor.index({i}) = std::get<0>(frame_sequences[i]);
        }

        int max_idx = torch::argmax(score_tensor).item<int>();

        if (std::get<0>(frame_sequences[max_idx]) > best_score) {
            best_score = std::get<0>(frame_sequences[max_idx]);

            best_sequence = std::get<1>(frame_sequences[max_idx]);
            std::reverse(best_sequence.begin(), best_sequence.end());

            sequence_frame_index = f_idx;
        }
    }

    return std::make_tuple(sequence_frame_index, best_sequence, best_score);
}

std::tuple<int, std::vector<int>, float> find_best_sequence(const box_seq_t& box_graph, const torch::Tensor& scores) {
    /*
    A function for finding the best path through the graph @box_graph.
    The best path is the one that has the highest cumulative sum.
    We dynamically build up best paths through graph starting from the end frame such that we can determine the beginning of
    sequences. For example if there are no links to a box from previous frames, then it is a candidate for starting a sequence.
    */

    std::vector<score_indicies_list> max_scores_paths;
    std::vector<score_indicies_list> sequence_roots;

    score_indicies_list last_scores;
    for (int idx = 0; idx < scores.size(1); idx++) {
        float score = scores.index({scores.size(0) - 1, idx}).item<float>();
        auto index_list = std::vector<int>{idx};
        last_scores.push_back(std::make_tuple(score, index_list));
    }
    max_scores_paths.push_back(last_scores);

    for (int frame_idx = box_graph.size() - 1; frame_idx >= 0; frame_idx--) {
        auto frame_edges = box_graph[frame_idx];

        auto used_in_sequence = torch::zeros(static_cast<long>(max_scores_paths.back().size()), {torch::kBool});
        score_indicies_list max_path_frame;

        for (int box_idx = 0; box_idx < frame_edges.size(); box_idx++) {
            std::vector<int> box_edges = frame_edges[box_idx];

            if (box_edges.size() == 0) {
                // no edges for current box so consider it a max path consisting of a single node
                float score = scores.index({frame_idx, box_idx}).item<float>();

                std::vector<int> indicies = {box_idx};
                max_path_frame.push_back(std::make_tuple(score, indicies));
            } else {
                // extend previous max paths
                // here we use box_edges to index used_in_sequence and mark boxes in corresponding frame t+1
                // as part of a sequence since we have links to them and can always make a better max path by making it longer
                // (score >= 0.0)

                auto score_tensor = torch::empty(box_edges.size(), {torch::kFloat32});
                for (int i = 0; i < box_edges.size(); i++) {
                    int e_idx = box_edges[i];
                    used_in_sequence.index({e_idx}) = true;
                    score_tensor.index({i}) = std::get<0>(max_scores_paths.back()[e_idx]);
                }

                int prev_idx = torch::argmax(score_tensor).item<int>();
                float score_so_far = std::get<0>(max_scores_paths.back()[box_edges[prev_idx]]);
                std::vector<int> path_so_far = std::get<1>(max_scores_paths.back()[box_edges[prev_idx]]);
                path_so_far.push_back(box_idx);

                max_path_frame.push_back(
                    std::make_tuple(scores.index({frame_idx, box_idx}).item<float>() + score_so_far, path_so_far));
            }
        }

        // create new sequence roots for boxes in frame at frame_idx + 1 that did not have links from boxes in frame_idx
        score_indicies_list new_sequence_root;
        for (int idx = 0; idx < used_in_sequence.size(0); idx++) {
            if (!used_in_sequence.index({idx}).item<bool>()) {
                new_sequence_root.push_back(max_scores_paths.back()[idx]);
            }
        }

        sequence_roots.push_back(new_sequence_root);
        max_scores_paths.push_back(max_path_frame);
    }

    sequence_roots.push_back(max_scores_paths.back());

    // reverse sequence roots since built sequences from back to front
    std::reverse(sequence_roots.begin(), sequence_roots.end());

    return find_highest_score_sequence(sequence_roots);
}
