#include <tuple>
#include <vector>
#include "seq_nms.h"
#include "box_utils.h"
#include "sequence_utils.h"

using namespace torch::indexing;

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

void seq_nms(
    const torch::Tensor& boxes,
    const torch::Tensor& scores,
    const torch::Tensor& classes,
    const float& linkage_threshold,
    const float& iou_threshold,
    const ScoreMetric& metric) {
    box_seq_t box_graph = build_box_sequences(boxes, classes, linkage_threshold);
    torch::Tensor local_scores = scores.clone();

    while (true) {
        auto best_tuple = find_best_sequence(box_graph, local_scores);
        int sequence_frame_index = std::get<0>(best_tuple);
        std::vector<int> best_sequence = std::get<1>(best_tuple);
        float best_score = std::get<2>(best_tuple);

        if (best_sequence.size() <= 1) {
            break;
        }

        rescore_sequence(best_sequence, local_scores, sequence_frame_index, best_score, metric);
        delete_sequence(best_sequence, sequence_frame_index, local_scores, boxes, box_graph, iou_threshold);
    }
}
