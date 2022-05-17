#pragma once
#include <torch/torch.h>
#include <tuple>
#include <vector>
#include "custom_types.h"

std::tuple<int, std::vector<int>, float> find_highest_score_sequence(const std::vector<score_indicies_list>& sequence_roots);

std::tuple<int, std::vector<int>, float> find_best_sequence(const box_seq_t& box_graph, const torch::Tensor& scores);

void rescore_sequence(
    const std::vector<int>& sequence,
    torch::Tensor& scores,
    const int& sequence_frame_index,
    const float& max_sum,
    const ScoreMetric& metric);

void delete_sequence(
    const std::vector<int>& sequence,
    const int& sequence_frame_index,
    torch::Tensor& scores,
    const torch::Tensor& boxes,
    const box_seq_t& box_graph,
    const float& iou_threshold);
