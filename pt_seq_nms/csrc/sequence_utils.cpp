#include "sequence_utils.h"
#include <algorithm>
#include "box_utils.h"

using namespace torch::indexing;

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

        if (frame_sequences.size() == 0) {
            continue;
        }

        auto score_tensor = torch::empty(frame_sequences.size(), {torch::kFloat32});
        auto score_acc = score_tensor.accessor<float, 1>();
        for (int i = 0; i < frame_sequences.size(); i++) {
            score_acc[i] = std::get<0>(frame_sequences[i]);
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

    auto scores_acc = scores.accessor<float, 2>();

    score_indicies_list last_scores;
    for (int idx = 0; idx < scores.size(1); idx++) {
        float score = scores_acc[scores.size(0) - 1][idx];
        auto index_list = std::vector<int>{idx};
        last_scores.push_back(std::make_tuple(score, index_list));
    }
    max_scores_paths.push_back(last_scores);

    for (int frame_idx = box_graph.size() - 1; frame_idx >= 0; frame_idx--) {
        auto frame_edges = box_graph[frame_idx];

        auto used_in_sequence = torch::zeros(static_cast<long>(max_scores_paths.back().size()), {torch::kBool});
        auto used_in_sequence_acc = used_in_sequence.accessor<bool, 1>();

        score_indicies_list max_path_frame;
        for (int box_idx = 0; box_idx < frame_edges.size(); box_idx++) {
            std::vector<int> box_edges = frame_edges[box_idx];

            if (box_edges.size() == 0) {
                // no edges for current box so consider it a max path consisting of a single node
                float score = scores_acc[frame_idx][box_idx]; // scores.index({frame_idx, box_idx}).item<float>();

                std::vector<int> indicies = {box_idx};
                max_path_frame.push_back(std::make_tuple(score, indicies));
            } else {
                // extend previous max paths
                // here we use box_edges to index used_in_sequence and mark boxes in corresponding frame t+1
                // as part of a sequence since we have links to them and can always make a better max path by making it longer
                // (score >= 0.0)

                auto score_tensor = torch::empty(box_edges.size(), {torch::kFloat32});
                auto score_acc = score_tensor.accessor<float, 1>();

                for (int i = 0; i < box_edges.size(); i++) {
                    int e_idx = box_edges[i];
                    used_in_sequence_acc[e_idx] = true;
                    score_acc[i] = std::get<0>(max_scores_paths.back()[e_idx]);
                }

                int prev_idx = torch::argmax(score_tensor).item<int>();
                float score_so_far = std::get<0>(max_scores_paths.back()[box_edges[prev_idx]]);
                std::vector<int> path_so_far = std::get<1>(max_scores_paths.back()[box_edges[prev_idx]]);
                path_so_far.push_back(box_idx);

                max_path_frame.push_back(std::make_tuple(scores_acc[frame_idx][box_idx] + score_so_far, path_so_far));
            }
        }

        // create new sequence roots for boxes in frame at frame_idx + 1 that did not have links from boxes in frame_idx
        score_indicies_list new_sequence_root;
        for (int idx = 0; idx < used_in_sequence.size(0); idx++) {
            if (!used_in_sequence_acc[idx]) {
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

void rescore_sequence(
    const std::vector<int>& sequence,
    torch::Tensor& scores,
    const int& sequence_frame_index,
    const float& max_sum,
    const ScoreMetric& metric) {
    /*
    Given a sequence, rescore the scores either by:
        - Average max_sum amongs sequence's elements (ScoreMetric::avg).
        - Find the max value amongs the sequence's elements, and set all values to that (ScoreMetric::max).
    */

    auto scores_acc = scores.accessor<float, 2>();

    if (metric == ScoreMetric::avg) {
        float avg_score = max_sum / static_cast<float>(sequence.size());

        for (int i = 0; i < sequence.size(); i++) {
            int box_idx = sequence[i];
            scores_acc[sequence_frame_index + i][box_idx] = avg_score;
        }
    } else {
        // metric == ScoreMetric::max
        float max_score = 0.0;

        for (int i = 0; i < sequence.size(); i++) {
            int box_idx = sequence[i];
            if (scores_acc[sequence_frame_index + i][box_idx] > max_score) {
                max_score = scores_acc[sequence_frame_index + i][box_idx];
            }
        }

        for (int i = 0; i < sequence.size(); i++) {
            int box_idx = sequence[i];
            scores_acc[sequence_frame_index + i][box_idx] = max_score;
        }
    }
}

void delete_sequence(
    const std::vector<int>& sequence,
    const int& sequence_frame_index,
    const torch::Tensor& scores,
    const torch::Tensor& boxes,
    const torch::Tensor& box_areas,
    box_seq_t& box_graph,
    const float& iou_threshold) {
    /*
    Given a sequence, remove connections in @box_graph which have iou higher than @iou_threshold
        with index @sequence_frame_index.
    */

    for (int s_idx = 0; s_idx < sequence.size(); s_idx++) {
        int box_idx = sequence[s_idx];

        torch::Tensor other_boxes = boxes.index({sequence_frame_index + s_idx, Slice(), Slice()});
        torch::Tensor other_areas = box_areas.index({sequence_frame_index + s_idx, Slice()});

        torch::Tensor seq_box = boxes.index({sequence_frame_index + s_idx, box_idx, Slice()});
        seq_box = seq_box.unsqueeze(0);
        torch::Tensor seq_box_area = box_areas.index({sequence_frame_index + s_idx, box_idx});
        seq_box_area = seq_box_area.unsqueeze(0);

        torch::Tensor iou_tensor = calculate_iou_given_area(other_boxes, seq_box, other_areas, seq_box_area);
        auto iou_acc = iou_tensor.accessor<float, 2>();

        std::vector<int> delete_indicies;
        for (int i = 0; i < iou_tensor.size(0); i++) {
            if (iou_acc[0][i] >= iou_threshold) {
                delete_indicies.push_back(i);
            }
        }

        if (sequence_frame_index + s_idx < box_graph.size()) {
            for (int delete_idx : delete_indicies) {
                box_graph[sequence_frame_index + s_idx][delete_idx].clear();
            }
        }

        if ((s_idx > 0) || (sequence_frame_index > 0)) {
            // remove connections to current sequence node from previous frame
            for (auto& prior_box : box_graph[sequence_frame_index + s_idx - 1]) {
                for (int delete_idx : delete_indicies) {
                    prior_box.erase(std::remove(prior_box.begin(), prior_box.end(), delete_idx), prior_box.end());
                }
            }
        }
    }
}
