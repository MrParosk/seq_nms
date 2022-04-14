#include "seq_nms.h"

torch::Tensor calculate_area(const torch::Tensor& boxes)
{
    auto x1 = boxes.index({torch::indexing::Slice(), torch::indexing::Slice(), 0});
    auto y1 = boxes.index({torch::indexing::Slice(), torch::indexing::Slice(), 1});
    auto x2 = boxes.index({torch::indexing::Slice(), torch::indexing::Slice(), 2});
    auto y2 = boxes.index({torch::indexing::Slice(), torch::indexing::Slice(), 3});
    auto areas = (x2 - x1) * (y2 - y1);
    return areas;
}
