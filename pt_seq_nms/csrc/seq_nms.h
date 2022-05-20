#pragma once
#include <torch/torch.h>
#include "custom_types.h"

box_seq_t build_box_sequences(const torch::Tensor& boxes, const torch::Tensor& classes, const double& linkage_threshold);

void seq_nms(
    const torch::Tensor& boxes,
    const torch::Tensor& scores,
    const torch::Tensor& classes,
    const float& linkage_threshold,
    const float& iou_threshold,
    const ScoreMetric& metric);
