import time

import torch

from pt_seq_nms import seq_nms as pt_seq_nms

NUM_ITERS = 10
NUM_CLASSES = 10
NUM_FRAMES = 100
NUM_BOXES = 20


def generate_data(num_frames, num_boxes, num_classes):
    boxes = torch.empty((num_frames, num_boxes, 4))
    width = 50.0 * torch.rand((num_frames, num_boxes))
    height = 50.0 * torch.rand((num_frames, num_boxes))
    x1 = 50.0 * torch.rand((num_frames, num_boxes))
    y1 = 50.0 * torch.rand((num_frames, num_boxes))

    boxes[:, :, 0] = x1
    boxes[:, :, 1] = y1
    boxes[:, :, 2] = x1 + width
    boxes[:, :, 3] = y1 + height

    scores = torch.rand((num_frames, num_boxes))
    classes = torch.randint(0, num_classes, (num_frames, num_boxes)).to(torch.int32)

    return boxes, scores, classes


def main():
    linkage_threshold = 0.3
    iou_threshold = 0.2

    boxes, scores, classes = generate_data(NUM_FRAMES, NUM_BOXES, NUM_CLASSES)

    start_time = time.time()
    for _ in range(NUM_ITERS):
        _ = pt_seq_nms(boxes, scores, classes, linkage_threshold, iou_threshold, "avg")
    end_time = time.time()

    avg_time = (end_time - start_time) / NUM_ITERS
    avg_time = round(avg_time, 4)
    print(f"Took on average: {avg_time} seconds")


if __name__ == "__main__":
    main()
