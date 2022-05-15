#pragma once
#include <tuple>
#include <vector>

typedef std::vector<std::vector<std::vector<int>>> box_seq_t;

typedef std::tuple<float, std::vector<int>> score_indicies;

typedef std::vector<score_indicies> score_indicies_list;
