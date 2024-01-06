import os
import unittest

import torch
import torch._dynamo

from pt_seq_nms.seq_nms import seq_nms, seq_nms_from_list


def _under_version_two():
    return torch.__version__ < (2, 0)


if not _under_version_two():
    # Needed ATM to fall back to eager for torch.compile
    torch._dynamo.config.suppress_errors = True


class TestCompileSeqNMS(unittest.TestCase):
    def setUp(self):
        torch.random.manual_seed(42)
        self.boxes = 100 * torch.rand((10, 2, 4)).float().cpu()
        self.scores = torch.rand((10, 2)).float().cpu()
        self.classes = torch.randint(0, 10, (10, 2)).int().cpu()
        self.linkage_threshold = 0.2
        self.iou_threshold = 0.2

    @unittest.skipIf(_under_version_two() or os.name == "nt", "Torch version < 2.0 or running on Windows")
    def test_compile_cpu(self):
        compiled_seq_nms = torch.compile(seq_nms)
        _ = compiled_seq_nms(self.boxes, self.scores, self.classes, self.linkage_threshold, self.iou_threshold)

    @unittest.skipIf(
        _under_version_two() or not torch.cuda.is_available() or os.name == "nt",
        "Torch version < 2.0 or CUDA not available or running on Windows",
    )
    def test_compile_cuda(self):
        compiled_seq_nms = torch.compile(seq_nms)
        _ = compiled_seq_nms(
            self.boxes.cuda(), self.scores.cuda(), self.classes.cuda(), self.linkage_threshold, self.iou_threshold
        )


class TestCompileSeqNMSList(unittest.TestCase):
    def setUp(self):
        torch.random.manual_seed(42)
        self.boxes = [100 * torch.rand((2, 4)).float().cpu(), 100 * torch.rand((2, 4)).float().cpu()]
        self.scores = [torch.rand((2,)).float().cpu(), torch.rand((2,)).float().cpu()]
        self.classes = [torch.randint(0, 10, (2,)).int().cpu(), torch.randint(0, 10, (2,)).int().cpu()]

        self.linkage_threshold = 0.2
        self.iou_threshold = 0.2

    @unittest.skipIf(_under_version_two() or os.name == "nt", "Torch version < 2.0 or running on Windows")
    def test_compile_cpu(self):
        compiled_soft_nms = torch.compile(seq_nms_from_list)
        _ = compiled_soft_nms(self.boxes, self.scores, self.classes, self.linkage_threshold, self.iou_threshold)

    @unittest.skipIf(
        _under_version_two() or not torch.cuda.is_available() or os.name == "nt",
        "Torch version < 2.0 or CUDA not available or running on Windows",
    )
    def test_compile_cuda(self):
        boxes = [b.cuda() for b in self.boxes]
        scores = [s.cuda() for s in self.scores]
        classes = [c.cuda() for c in self.classes]

        compiled_soft_nms = torch.compile(seq_nms_from_list)
        _ = compiled_soft_nms(boxes, scores, classes, self.linkage_threshold, self.iou_threshold)


if __name__ == "__main__":
    unittest.main()
