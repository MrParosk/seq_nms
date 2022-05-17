#pragma once
#include <torch/torch.h>
#include <tuple>
#include <vector>
#include "custom_types.h"

box_seq_t build_box_sequences(const torch::Tensor& boxes, const torch::Tensor& classes, const double& linkage_threshold);
