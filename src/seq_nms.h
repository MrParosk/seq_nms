#pragma once
#include <torch/torch.h>

typedef std::vector<std::vector<std::vector<int>>> box_seq_t;

torch::Tensor calculate_area(const torch::Tensor& boxes);

torch::Tensor calculate_iou_given_area(
    const torch::Tensor& boxes_a,
    const torch::Tensor& boxes_b,
    const torch::Tensor& aread_a,
    const torch::Tensor& areas_b);

box_seq_t build_box_sequences(const torch::Tensor& boxes, const torch::Tensor& classes, const double& linkage_threshold);
