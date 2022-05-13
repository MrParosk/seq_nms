#include "seq_nms.h"

using namespace torch::indexing;

torch::Tensor calculate_area(const torch::Tensor& boxes) {
    /*
    Computes the area of the boxes.

    boxes are expected to have the shape [N, M, 4] and of the format [x_min, y_min, x_max, y_max].
    */

    auto x1 = boxes.index({Slice(), Slice(), 0});
    auto y1 = boxes.index({Slice(), Slice(), 1});
    auto x2 = boxes.index({Slice(), Slice(), 2});
    auto y2 = boxes.index({Slice(), Slice(), 3});
    auto areas = (x2 - x1) * (y2 - y1);
    return areas;
}

torch::Tensor calculate_iou_given_area(
    const torch::Tensor& boxes_a,
    const torch::Tensor& boxes_b,
    const torch::Tensor& areas_a,
    const torch::Tensor& areas_b) {
    /*
    Computes the IOU between boxes_a and boxes_b.

    boxes_a are expected to have the shape [N, 4] and of the format [x_min, y_min, x_max, y_max].
    boxes_b are expected to have the shape [M, 4] and of the format [x_min, y_min, x_max, y_max].
    areas_a are expected to have the shape [N].
    areas_b are expected to have the shape [M].

    The returned IOU tensor has the shape [N, M].
    */

    auto max_xy = torch::min(boxes_a.index({Slice(), None, Slice(2, None)}), boxes_b.index({None, Slice(), Slice(2, None)}));

    auto min_xy = torch::max(boxes_a.index({Slice(), None, Slice(None, 2)}), boxes_b.index({None, Slice(), Slice(None, 2)}));

    auto inter = torch::clamp(max_xy - min_xy, /*min=*/0.0);
    auto intersection = inter.index({Slice(), Slice(), 0}) * inter.index({Slice(), Slice(), 1});

    auto union_ = areas_a.unsqueeze(1) + areas_b.unsqueeze(0) - intersection;
    auto iou = torch::div(intersection, union_);
    return iou;
}

box_seq_t build_box_sequences(const torch::Tensor& boxes, const torch::Tensor& classes, const double& linkage_threshold) {
    /*
    Creates a graph where vericies are object at a given frame and the edges are if they have an IOU higher than
    @linkage_threshold (two consecutive frames).

    boxes are expected to have the shape [F, N, 4] and of the format [x_min, y_min, x_max, y_max].
        F is the number of frames and N is the number of objects per frame.
    classes are expected to have the shape [F, N].
    linkage_threshold is the threshold for linking two objects in consecutive frames.
    */

    auto areas = calculate_area(boxes);

    box_seq_t box_graph;
    for (int f_idx = 0; f_idx < boxes.size(0) - 1; f_idx++) {
        auto current_box = boxes.index({f_idx, Slice(), Slice()});
        auto current_area = areas.index({f_idx, Slice()});

        auto next_box = boxes.index({f_idx + 1, Slice(), Slice()});
        auto next_area = areas.index({f_idx + 1, Slice()});

        std::vector<std::vector<int>> adjacency_matrix;
        for (int b_idx = 0; b_idx < current_box.size(0); b_idx++) {
            auto box = current_box.index({b_idx, Slice()});
            box = box.unsqueeze(0);
            auto box_area = current_area.index({b_idx}).view({1});

            // overlap has shape [1, M]
            auto overlaps = calculate_iou_given_area(box, next_box, box_area, next_area);

            std::vector<int> edges;
            for (int ovr_idx = 0; ovr_idx < overlaps.size(1); ovr_idx++) {
                auto iou = overlaps.index({0, ovr_idx}).item<double>();
                bool same_class = torch::equal(classes.index({f_idx, b_idx}), classes.index({f_idx + 1, ovr_idx}));

                if ((iou >= linkage_threshold) && same_class) {
                    edges.push_back(ovr_idx);
                }
            }

            adjacency_matrix.push_back(edges);
        }
        box_graph.push_back(adjacency_matrix);
    }

    return box_graph;
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

    float best_score = 0.0;
    std::vector<int> best_sequence;
    int sequence_frame_index = 0;

    for (int s_idx = 0; s_idx < sequence_roots.size(); s_idx++) {
        score_indicies_list frame_sequences = sequence_roots[s_idx];

        if (frame_sequences.size() == 0) {
            continue;
        }

        std::vector<int> index_list;
        for (score_indicies s : frame_sequences) {
            index_list.push_back(std::get<0>(s));
        }

        int max_idx = torch::argmax(torch::tensor(index_list)).item<int>();

        if (std::get<0>(frame_sequences[max_idx]) > best_score) {
            best_score = std::get<0>(frame_sequences[max_idx]);

            std::vector<int> best_sequence = std::get<1>(frame_sequences[max_idx]);
            std::reverse(best_sequence.begin(), best_sequence.end());

            sequence_frame_index = s_idx;
        }
    }

    return std::make_tuple(sequence_frame_index, best_sequence, best_score);
}
