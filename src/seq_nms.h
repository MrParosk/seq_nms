#pragma once
#include <torch/torch.h>

torch::Tensor calculate_area(const torch::Tensor &boxes);