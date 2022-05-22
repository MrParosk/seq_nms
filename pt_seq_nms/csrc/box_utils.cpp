#include "box_utils.h"
#include "custom_types.h"

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
    auto iou = torch::div(intersection, union_ + EPS);
    return iou;
}
