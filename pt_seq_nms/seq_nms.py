import torch


def seq_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    classes: torch.Tensor,
    linkage_threshold: float,
    iou_threshold: float,
    metrics: str = "avg",
) -> torch.Tensor:

    assert len(boxes.shape) == 3 and boxes.shape[-1] == 4, f"boxes has wrong shape, expected (F, N, 4), got {boxes.shape}"
    assert len(scores.shape) == 2, f"scores has wrong shape, expected (F, N) got {scores.shape}"
    assert len(classes.shape) == 2, f"classes has wrong shape, expected (F, N) got {classes.shape}"

    assert 0.0 <= linkage_threshold <= 1.0, f"linkage_threshold should be between 0.0 and 1.0; got {linkage_threshold}"
    assert 0.0 <= iou_threshold <= 1.0, f"iou_threshold should be between 0.0 and 1.0; got {iou_threshold}"

    assert metrics in ("avg", "max"), f"Expected metrics to be in {('avg', 'max')}, got {metrics}"

    # only support cpu ATM
    updated_scores = torch.ops.seq_nms.seq_nms(
        boxes.cpu(), scores.cpu(), classes.cpu(), linkage_threshold, iou_threshold, metrics
    )
    updated_scores = updated_scores.to(scores.device)
    return updated_scores
