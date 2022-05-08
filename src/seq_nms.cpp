#include "seq_nms.h"


using namespace torch::indexing;


torch::Tensor calculate_area(const torch::Tensor& boxes)
{
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
    const torch::Tensor& areas_b)
{

    auto max_xy = torch::min(
        boxes_a.index({Slice(), None, Slice(2, None)}),
        boxes_b.index({None, Slice(), Slice(2, None)})
    );

    auto min_xy = torch::max(
        boxes_a.index({Slice(), None, Slice(None, 2)}),
        boxes_b.index({None, Slice(), Slice(None, 2)})
    );

    auto inter = torch::clamp(max_xy - min_xy, /*min=*/ 0.0);
    auto intersection = inter.index({Slice(), Slice(), 0}) * inter.index({Slice(), Slice(), 1});

    auto union_ = areas_a.unsqueeze(1) + areas_b.unsqueeze(0) - intersection;    
    auto iou = torch::div(intersection, union_);
    return iou;
}

box_seq_t build_box_sequences(
    const torch::Tensor& boxes,
    const torch::Tensor& classes,
    const double& linkage_threshold)
{
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
