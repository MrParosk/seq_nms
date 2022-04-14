#pragma once
#include <torch/torch.h>


torch::Tensor calculate_area(const torch::Tensor &boxes);

torch::Tensor calculate_iou_given_area(
    const torch::Tensor& boxes_a,
    const torch::Tensor& boxes_b,
    const torch::Tensor& aread_a,
    const torch::Tensor& areas_b
);
