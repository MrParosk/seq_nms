import torch

def seq_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    classes: torch.Tensor,
    linkage_threshold: float,
    iou_threshold: float,
    metrics: str = "avg",
) -> torch.Tensor:
    return torch.ops.seq_nms.seq_nms(boxes, scores, classes, linkage_threshold, iou_threshold, metrics)
