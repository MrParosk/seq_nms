import unittest

import torch

from pt_seq_nms.seq_nms import _from_list_to_tensor, seq_nms, seq_nms_from_list


class TestE2ESeqNMS(unittest.TestCase):
    def setUp(self):
        torch.random.manual_seed(42)
        NUM_FRAMES = 100

        boxes = torch.empty((NUM_FRAMES, 20, 4), dtype=torch.float32)
        width = 50.0 * torch.rand((NUM_FRAMES, 20), dtype=torch.float32)
        height = 50.0 * torch.rand((NUM_FRAMES, 20), dtype=torch.float32)
        x1 = 50.0 * torch.rand((NUM_FRAMES, 20), dtype=torch.float32)
        y1 = 50.0 * torch.rand((NUM_FRAMES, 20), dtype=torch.float32)

        boxes[:, :, 0] = x1
        boxes[:, :, 1] = y1
        boxes[:, :, 2] = x1 + width
        boxes[:, :, 3] = y1 + height

        self.boxes = boxes
        self.scores = torch.rand((NUM_FRAMES, 20), dtype=torch.float32)
        self.classes = torch.randint(0, 10, (NUM_FRAMES, 20), dtype=torch.int32)

        self.linkage_threshold = 0.3
        self.iou_threshold = 0.2

    def test_cpu(self):
        updated_scores = seq_nms(self.boxes, self.scores, self.classes, self.linkage_threshold, self.iou_threshold)

        self.assertEqual(updated_scores.shape, self.scores.shape)
        self.assertTrue(not torch.equal(updated_scores, self.scores))

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_cuda(self):
        updated_scores = seq_nms(
            self.boxes.cuda(), self.scores.cuda(), self.classes.cuda(), self.linkage_threshold, self.iou_threshold
        )

        self.assertEqual(updated_scores.shape, self.scores.shape)
        self.assertTrue(not torch.equal(updated_scores, self.scores.cuda()))


class TestFromListToTensor(unittest.TestCase):
    def test_fill(self):
        boxes_list = [
            torch.tensor([1, 2, 3, 4, 5, 6, 7, 8], dtype=torch.float32).view((2, 4)),
            torch.tensor([9, 10, 11, 12], dtype=torch.float32).view((1, 4)),
            torch.tensor([], dtype=torch.float32).view((0, 4)),
        ]

        scores_list = [
            torch.tensor([0.5, 0.7], dtype=torch.float32),
            torch.tensor([0.9], dtype=torch.float32),
            torch.tensor([], dtype=torch.float32),
        ]

        classes_list = [
            torch.tensor([1, 2], dtype=torch.int32),
            torch.tensor([3], dtype=torch.int32),
            torch.tensor([], dtype=torch.int32),
        ]

        boxes, scores, classes = _from_list_to_tensor(boxes_list, scores_list, classes_list)

        expected_boxes = torch.tensor(
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.float32
        ).view((3, 2, 4))

        expected_scores = torch.tensor([0.5, 0.7, 0.9, 0.0, 0.0, 0.0], dtype=torch.float32).view((3, 2))

        expected_classes = torch.tensor([1, 2, 3, -1, -1, -1], dtype=torch.int32).view((3, 2))

        self.assertTrue(torch.allclose(boxes, expected_boxes))
        self.assertTrue(torch.allclose(scores, expected_scores))
        self.assertTrue(torch.allclose(classes, expected_classes))


class TestE2ESeqNMSList(unittest.TestCase):
    def setUp(self) -> None:
        NUM_FRAMES = 100
        MAX_OBJECT_FRAMES = 20

        boxes_list, scores_list, classes_list = [], [], []

        import random

        random.seed(42)

        for _ in range(NUM_FRAMES):
            num_objects = random.randint(0, MAX_OBJECT_FRAMES)

            boxes = torch.empty((num_objects, 4), dtype=torch.float32)
            width = 50.0 * torch.rand((num_objects), dtype=torch.float32)
            height = 50.0 * torch.rand((num_objects), dtype=torch.float32)
            x1 = 50.0 * torch.rand((num_objects), dtype=torch.float32)
            y1 = 50.0 * torch.rand((num_objects), dtype=torch.float32)

            boxes[:, 0] = x1
            boxes[:, 1] = y1
            boxes[:, 2] = x1 + width
            boxes[:, 3] = y1 + height

            scores = torch.rand((num_objects,))
            classes = torch.randint(0, 10, (num_objects,), dtype=torch.int32)

            boxes_list.append(boxes)
            scores_list.append(scores)
            classes_list.append(classes)

        self.boxes_list = boxes_list
        self.scores_list = scores_list
        self.classes_list = classes_list

        self.linkage_threshold = 0.3
        self.iou_threshold = 0.2

    def test(self):
        _ = seq_nms_from_list(self.boxes_list, self.scores_list, self.classes_list, self.linkage_threshold, self.iou_threshold)


if __name__ == "__main__":
    unittest.main()
