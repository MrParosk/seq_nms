from typing import List, Tuple

import torch


def _all_same_types(types: List[str]) -> bool:
    # Can't use set(types) == 1 since set() is not supported by torchscript at the moment
    found: List[str] = []

    for t in types:
        if t not in found:
            found.append(t)

    return len(found) == 1


def _validate_tensor_types(boxes: torch.Tensor, scores: torch.Tensor, classes: torch.Tensor) -> None:
    """
    Utility function for validating that the boxes, scores and classes have the same type.

    Args:
        boxes (Tensor[F, N, 4]) Boxes to perform seq-nms on. They are expected to be in
           (x_min, y_min, x_max, y_max) format.
        scores (Tensor[F, N]): Scores for each one of the boxes.
        classes (Tensor[F, N]): Class for each one of the boxes.
    Returns:
    """

    assert boxes.dtype == torch.float32, f"boxes are expected to have dtype float32, got {boxes.dtype}"
    assert scores.dtype == torch.float32, f"scores are expected to have dtype float32, got {scores.dtype}"
    assert classes.dtype == torch.int32, f"classes are expected to have dtype float32, got {classes.dtype}"
    assert _all_same_types(
        [boxes.device.type, scores.device.type, classes.device.type]
    ), "Expected all of the tensors to be on the same device"


def _validate_auxiliary_params(linkage_threshold: float, iou_threshold: float, metrics: str = "avg") -> None:
    """
    Utility function for validating that the parameters are in the valid range.

    Args:
        linkage_threshold (float): The threshold for linking two objects in consecutive frames.
        iou_threshold (float): the threshold for considering two boxes to be overlapping.
        metric (str): the metric type, currently "avg" and "max" is supported.
    Returns:
    """

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
    """
    Applies the seq-nms algorithm to the input boxes.

    Below F is the number of frames and N is the number of objects per frame.

    Args:
        boxes (Tensor[F, N, 4]) Boxes to perform seq-nms on. They are expected to be in
           (x_min, y_min, x_max, y_max) format.
        scores (Tensor[F, N]): Scores for each one of the boxes.
        classes (Tensor[F, N]): Class for each one of the boxes.
        linkage_threshold (float): The threshold for linking two objects in consecutive frames.
        iou_threshold (float): the threshold for considering two boxes to be overlapping.
        metric (str): the metric type, currently "avg" and "max" is supported.
    Returns:
        updated_scores (Tensor): tensor with the updated scores, i.e. the scores after they have been
            updated according to seq-nms.
    """

    assert len(boxes.shape) == 3 and boxes.shape[-1] == 4, f"boxes has wrong shape, expected (F, N, 4), got {boxes.shape}"
    assert len(scores.shape) == 2, f"scores has wrong shape, expected (F, N) got {scores.shape}"
    assert len(classes.shape) == 2, f"classes has wrong shape, expected (F, N) got {classes.shape}"

    _validate_tensor_types(boxes, scores, classes)
    _validate_auxiliary_params(linkage_threshold, iou_threshold, metrics)

    updated_scores = torch.ops.seq_nms.seq_nms(boxes, scores, classes, linkage_threshold, iou_threshold, metrics)
    return updated_scores


def _from_list_to_tensor(
    boxes_list: List[torch.Tensor],
    scores_list: List[torch.Tensor],
    classes_list: List[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Converts a list of tensors of different shapes to one tensor by taking the max-length
        and filling the remaining values with zeros.

    For example, if the input is [[1], [2, 3]] the output will be [[1, 0], [2, 3]].

    Below F is the number of frames and N_f is the number of objects per frame (which can vary between frames).
    M is the maximum length of each box.

    Args:
        boxes_list (List[F]) Boxes to perform seq-nms on. Each element in the list is a Tensor[N_f, 4].
            They are expected to be in (x_min, y_min, x_max, y_max) format.
        scores_list (List[F]) Scores for each one of the boxes. Each element in the list is a Tensor[N_f].
        classes_list (List[F]): Class for each one of the boxes. Each element in the list is a Tensor[N_f].
    Returns:
        boxes (Tensor[F, M, 4]): Concatenated boxes.
        scores (Tensor[F, M]): Concatenated scores.
        classes (Tensor[F, M]): Concatenated classes.
    """

    max_length = max([len(b) for b in boxes_list])
    num_element = len(boxes_list)
    device = boxes_list[0].device.type if len(boxes_list) > 0 else "cpu"

    boxes = torch.zeros((num_element, max_length, 4), dtype=torch.float32, device=device)
    scores = torch.zeros((num_element, max_length), dtype=torch.float32, device=device)
    classes = -1 * torch.ones((num_element, max_length), dtype=torch.int32, device=device)

    for idx in range(len(boxes_list)):
        len_idx = len(boxes_list[idx])

        boxes_idx = boxes_list[idx]
        scores_idx = scores_list[idx]
        classes_idx = classes_list[idx]

        _validate_tensor_types(boxes_idx, scores_idx, classes_idx)

        boxes[idx, 0:len_idx, :] = boxes_idx
        scores[idx, 0:len_idx] = scores_idx
        classes[idx, 0:len_idx] = classes_idx

    return boxes, scores, classes


def seq_nms_from_list(
    boxes_list: List[torch.Tensor],
    scores_list: List[torch.Tensor],
    classes_list: List[torch.Tensor],
    linkage_threshold: float,
    iou_threshold: float,
    metrics: str = "avg",
) -> torch.Tensor:
    """
    Applies the seq-nms algorithm to a list of input boxes, which can have different shapes.

    Below F is the number of frames and N_f is the number of objects per frame (which can vary between frames).

    Args:
        boxes (List[F]) Boxes to perform seq-nms on. Each element in the list is a Tensor[N_f, 4].
            They are expected to be in (x_min, y_min, x_max, y_max) format.
        scores (List[F]) Scores for each one of the boxes. Each element in the list is a Tensor[N_f].
        classes (List[F]): Class for each one of the boxes. Each element in the list is a Tensor[N_f].
        linkage_threshold (float): The threshold for linking two objects in consecutive frames.
        iou_threshold (float): the threshold for considering two boxes to be overlapping.
        metric (str): the metric type, currently "avg" and "max" is supported.
    Returns:
        updated_scores (Tensor): tensor with the updated scores, i.e. the scores after they have been
            updated according to seq-nms.
    """

    _validate_auxiliary_params(linkage_threshold, iou_threshold, metrics)

    boxes, scores, classes = _from_list_to_tensor(boxes_list, scores_list, classes_list)

    updated_scores = torch.ops.seq_nms.seq_nms(boxes, scores, classes, linkage_threshold, iou_threshold, metrics)
    return updated_scores
