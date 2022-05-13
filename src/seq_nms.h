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

typedef std::tuple<float, std::vector<int>> score_indicies;
typedef std::vector<score_indicies> score_indicies_list;

std::tuple<int, std::vector<int>, float> find_best_sequence(const box_seq_t& box_graph, torch::Tensor& scores);
