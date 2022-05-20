#pragma once
#include <torch/torch.h>
#include "custom_types.h"

box_seq_t build_box_sequences(const torch::Tensor& boxes, const torch::Tensor& classes, const double& linkage_threshold);

ScoreMetric get_score_enum_from_string(const std::string& metric_string);

void seq_nms(
    const torch::Tensor& boxes,
    const torch::Tensor& scores,
    const torch::Tensor& classes,
    const double& linkage_threshold,
    const double& iou_threshold,
    const std::string& metric);
