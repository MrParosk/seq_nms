#pragma once
#include <torch/torch.h>
#include <tuple>
#include <vector>
#include "seq_nms.h"

typedef std::tuple<float, std::vector<int>> score_indicies;
typedef std::vector<score_indicies> score_indicies_list;

std::tuple<int, std::vector<int>, float> find_highest_score_sequence(const std::vector<score_indicies_list>& sequence_roots);

std::tuple<int, std::vector<int>, float> find_best_sequence(const box_seq_t& box_graph, torch::Tensor& scores);
