#include "seq_nms.h"
#include <exception>
#include <tuple>
#include <vector>
#include "box_utils.h"
#include "sequence_utils.h"

using namespace torch::indexing;

box_seq_t build_box_sequences(
    const torch::Tensor& boxes,
    const torch::Tensor& box_areas,
    const torch::Tensor& classes,
    const double& linkage_threshold) {
    /*
    Creates a graph where vericies are object at a given frame and the edges are if they have an IOU higher than
    @linkage_threshold (two consecutive frames).

    boxes are expected to have the shape [F, N, 4] and of the format [x_min, y_min, x_max, y_max].
        F is the number of frames and N is the number of objects per frame.
    classes are expected to have the shape [F, N].
    linkage_threshold is the threshold for linking two objects in consecutive frames.
    */

    auto classes_acc = classes.accessor<int, 2>();

    box_seq_t box_graph;
    for (int f_idx = 0; f_idx < boxes.size(0) - 1; f_idx++) {
        auto current_box = boxes.index({f_idx, Slice(), Slice()});
        auto current_area = box_areas.index({f_idx, Slice()});

        auto next_box = boxes.index({f_idx + 1, Slice(), Slice()});
        auto next_area = box_areas.index({f_idx + 1, Slice()});

        // overlap has shape [N, N]
        auto overlaps = calculate_iou_given_area(current_box, next_box, current_area, next_area);
        auto overlaps_acc = overlaps.accessor<float, 2>();

        std::vector<std::vector<int>> adjacency_matrix;
        for (int b_idx = 0; b_idx < current_box.size(0); b_idx++) {
            std::vector<int> edges;
            for (int ovr_idx = 0; ovr_idx < overlaps.size(1); ovr_idx++) {
                float iou = overlaps_acc[b_idx][ovr_idx];
                bool same_class = (classes_acc[f_idx][b_idx] == classes_acc[f_idx + 1][ovr_idx]);

                // class idx < 0 are considered skip idxs
                bool accepted_class = (classes_acc[f_idx][b_idx] >= 0);

                if ((iou >= linkage_threshold) && same_class && accepted_class) {
                    edges.push_back(ovr_idx);
                }
            }

            adjacency_matrix.push_back(edges);
        }
        box_graph.push_back(adjacency_matrix);
    }

    return box_graph;
}

ScoreMetric get_score_enum_from_string(const std::string& metric_string) {
    /*
    Converts @metric_string to the enum "ScoreMetric".
    */

    if (metric_string == "avg") {
        return ScoreMetric::avg;
    } else if (metric_string == "max") {
        return ScoreMetric::max;
    } else {
        throw std::invalid_argument("Unsupported metric_string");
    }
}

torch::Tensor seq_nms(
    const torch::Tensor& boxes,
    const torch::Tensor& scores,
    const torch::Tensor& classes,
    const double& linkage_threshold,
    const double& iou_threshold,
    const std::string& metric) {
    /*
    Applies the seq-nms algorithm to the input boxes.

    boxes are expected to have the shape [F, N, 4] and of the format [x_min, y_min, x_max, y_max].
        F is the number of frames and N is the number of objects per frame.
    scores are expected to have the shape [F, N].
    classes are expected to have the shape [F, N].
    linkage_threshold is the threshold for linking two objects in consecutive frames.
    iou_threshold is the threshold for considering two boxes to be overlapping.
    metric is the metric type, currently "avg" and "max" is supported.
    */

    ScoreMetric metric_enum = get_score_enum_from_string(metric);
    float linkage_threshold_float = static_cast<float>(linkage_threshold);
    float iou_threshold_float = static_cast<float>(iou_threshold);

    torch::Tensor box_areas = calculate_area(boxes);
    box_areas = box_areas.to(torch::kCPU);

    const auto boxes_cpu = boxes.to(torch::kCPU);
    const auto classes_cpu = classes.to(torch::kCPU);

    box_seq_t box_graph = build_box_sequences(boxes_cpu, box_areas, classes_cpu, linkage_threshold_float);
    torch::Tensor local_scores = scores.to(torch::kCPU).clone();

    while (true) {
        auto best_tuple = find_best_sequence(box_graph, local_scores);

        int sequence_frame_index = std::get<0>(best_tuple);
        std::vector<int> best_sequence = std::get<1>(best_tuple);
        float best_score = std::get<2>(best_tuple);

        if (best_sequence.size() <= 1) {
            break;
        }

        rescore_sequence(best_sequence, local_scores, sequence_frame_index, best_score, metric_enum);
        delete_sequence(
            best_sequence, sequence_frame_index, local_scores, boxes_cpu, box_areas, box_graph, iou_threshold_float);
    }

    local_scores = local_scores.to(scores.device());
    return local_scores;
}
