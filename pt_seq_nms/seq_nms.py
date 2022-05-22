from typing import List, Tuple

import torch


def _validate_tensor_types(boxes: torch.Tensor, scores: torch.Tensor, classes: torch.Tensor) -> None:
    assert boxes.dtype == torch.float32, f"boxes are expected to have dtype float32, got {boxes.dtype}"
    assert scores.dtype == torch.float32, f"scores are expected to have dtype float32, got {scores.dtype}"
    assert classes.dtype == torch.int32, f"classes are expected to have dtype float32, got {classes.dtype}"


def _validate_auxiliary_params(linkage_threshold: float, iou_threshold: float, metrics: str = "avg") -> None:
    assert 0.0 <= linkage_threshold <= 1.0, f"linkage_threshold should be between 0.0 and 1.0; got {linkage_threshold}"
    assert 0.0 <= iou_threshold <= 1.0, f"iou_threshold should be between 0.0 and 1.0; got {iou_threshold}"
    assert metrics in ("avg", "max"), f"Expected metrics to be in {('avg', 'max')}, got {metrics}"


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

    _validate_tensor_types(boxes, scores, classes)
    _validate_auxiliary_params(linkage_threshold, iou_threshold, metrics)

    # only support cpu ATM
    updated_scores = torch.ops.seq_nms.seq_nms(
        boxes.cpu(), scores.cpu(), classes.cpu(), linkage_threshold, iou_threshold, metrics
    )
    updated_scores = updated_scores.to(scores.device)
    return updated_scores


def _from_list_to_tensor(
    boxes_list: List[torch.Tensor],
    scores_list: List[torch.Tensor],
    classes_list: List[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    max_length = max([len(b) for b in boxes_list])
    num_element = len(boxes_list)

    boxes = torch.zeros((num_element, max_length, 4), dtype=torch.float32)
    scores = torch.zeros((num_element, max_length), dtype=torch.float32)
    classes = -1 * torch.ones((num_element, max_length), dtype=torch.int32)

    for idx in range(len(boxes_list)):
        len_idx = len(boxes_list[idx])

        boxes_idx = boxes_list[idx]
        scores_idx = scores_list[idx]
        classes_idx = classes_list[idx]

        _validate_tensor_types(boxes_idx, scores_idx, classes_idx)

        # only support cpu ATM
        boxes[idx, 0:len_idx, :] = boxes_idx.cpu()
        scores[idx, 0:len_idx] = scores_idx.cpu()
        classes[idx, 0:len_idx] = classes_idx.cpu()

    return boxes, scores, classes


def seq_nms_from_list(
    boxes_list: List[torch.Tensor],
    scores_list: List[torch.Tensor],
    classes_list: List[torch.Tensor],
    linkage_threshold: float,
    iou_threshold: float,
    metrics: str = "avg",
) -> torch.Tensor:

    _validate_auxiliary_params(linkage_threshold, iou_threshold, metrics)

    boxes, scores, classes = _from_list_to_tensor(boxes_list, scores_list, classes_list)

    updated_scores = torch.ops.seq_nms.seq_nms(boxes, scores, classes, linkage_threshold, iou_threshold, metrics)

    updated_scores = updated_scores.to(scores.device)
    return updated_scores
