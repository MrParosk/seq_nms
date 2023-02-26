# Seq-nms in PyTorch

![main](https://github.com/MrParosk/seq_nms/workflows/main/badge.svg?branch=main) [![codecov](https://codecov.io/gh/MrParosk/seq_nms/branch/main/graph/badge.svg?token=7DYQ1CHZQS)](https://codecov.io/gh/MrParosk/seq_nms)

Implementation of the seq-nms algorithm described in the paper: [Seq-NMS for Video Object Detection](https://arxiv.org/abs/1602.08465)

The algorithm is implemented in PyTorch's C++ frontend for better performance.

It can be exported with both PyTorch's scripting and tracing.

## Install

Make sure that you have installed PyTorch, version 1.7 or higher. Install the package by

```Shell
pip install git+https://github.com/MrParosk/seq_nms.git
```

Note that if you are using Windows, you need MSVC installed.

## Example usage

```python
import torch

from pt_seq_nms import seq_nms, seq_nms_from_list

linkage_threshold = 0.5
iou_threshold = 0.5

boxes = torch.tensor([[[20, 20, 40, 40], [10, 10, 20, 20]], [[100, 100, 120, 120], [20, 20, 35, 35]]], dtype=torch.float, device="cpu")
scores = torch.tensor([[0.9, 0.7], [0.7, 0.7]], dtype=torch.float, device="cpu")
classes = torch.tensor([[0, 1], [0, 0]], dtype=torch.int, device="cpu")

updated_scores = seq_nms(boxes, scores, classes, linkage_threshold, iou_threshold)
# updated_scores=tensor([[0.8, 0.7],[0.7, 0.8]])


# Using seq_nms_from_list allows for variable-number of boxes per frame
boxes_list = [
    torch.tensor([[20, 20, 40, 40], [10, 10, 20, 20]], dtype=torch.float, device="cpu"),
    torch.tensor([[20, 20, 35, 35]], dtype=torch.float, device="cpu")
]

scores_list = [
    torch.tensor([[0.9, 0.7]], dtype=torch.float, device="cpu"),
    torch.tensor([[0.7]], dtype=torch.float, device="cpu")
]

classes_list = [
    torch.tensor([[0, 1]], dtype=torch.int, device="cpu"),
    torch.tensor([[0]], dtype=torch.int, device="cpu")
]

updated_scores_list = seq_nms_from_list(boxes_list, scores_list, classes_list, linkage_threshold, iou_threshold)
# updated_scores_list=tensor([[0.8, 0.7],[0.8, 0.0]])
```
