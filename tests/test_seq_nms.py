import os
import unittest
from copy import deepcopy
from tempfile import TemporaryDirectory

import torch

from pt_seq_nms import seq_nms


class TestingModule(torch.nn.Module):
    def forward(self, boxes, scores, classes):
        return seq_nms(boxes, scores, classes, 0.2, 0.2)


class TestScritable(unittest.TestCase):
    def setUp(self):
        self.module = TestingModule()
        torch.random.manual_seed(42)
        self.boxes = 100 * torch.rand((10, 2, 4), dtype=torch.float32)
        self.scores = torch.rand((10, 2), dtype=torch.float32)
        self.classes = torch.randint(0, 10, (10, 2), dtype=torch.int32)

    def _call_module(self, module):
        _ = module(self.boxes, self.scores, self.classes)

    def _jit_save_load(self, m):
        jit_module = torch.jit.script(m)
        self._call_module(jit_module)

        with TemporaryDirectory() as dir:
            file = os.path.join(dir, "module.pt")
            torch.jit.save(jit_module, file)
            loaded_module = torch.jit.load(file)

        self._call_module(loaded_module)

    def test_scriptable_cpu(self):
        m = deepcopy(self.module).cpu()
        self._jit_save_load(m)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_scriptable_gpu(self):
        m = deepcopy(self.module).cuda()
        self._jit_save_load(m)


class TestTracing(unittest.TestCase):
    def setUp(self):
        self.module = TestingModule()
        torch.random.manual_seed(42)
        self.boxes = 100 * torch.rand((10, 2, 4), dtype=torch.float32)
        self.scores = torch.rand((10, 2), dtype=torch.float32)
        self.classes = torch.randint(0, 10, (10, 2), dtype=torch.int32)

    def _call_module(self, module):
        _ = module(self.boxes, self.scores, self.classes)

    def _trace_save_load(self, m):
        trace_module = torch.jit.trace(m, (self.boxes, self.scores, self.classes))
        self._call_module(trace_module)

        with TemporaryDirectory() as dir:
            file = os.path.join(dir, "module.pt")
            torch.jit.save(trace_module, file)
            loaded_module = torch.jit.load(file)

        self._call_module(loaded_module)

    def test_tracing_cpu(self):
        m = deepcopy(self.module).cpu()
        self._trace_save_load(m)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_tracing_gpu(self):
        m = deepcopy(self.module).cuda()
        self._trace_save_load(m)


class TestE2E(unittest.TestCase):
    def test(self):
        torch.random.manual_seed(42)
        NUM_FRAMES = 100

        boxes = torch.empty((NUM_FRAMES, 20, 4))
        width = 50.0 * torch.rand((NUM_FRAMES, 20))
        height = 50.0 * torch.rand((NUM_FRAMES, 20))
        x1 = 50.0 * torch.rand((NUM_FRAMES, 20))
        y1 = 50.0 * torch.rand((NUM_FRAMES, 20))

        boxes[:, :, 0] = x1
        boxes[:, :, 1] = y1
        boxes[:, :, 2] = x1 + width
        boxes[:, :, 3] = y1 + height

        scores = torch.rand((NUM_FRAMES, 20))
        classes = torch.randint(0, 10, (NUM_FRAMES, 20)).to(torch.int32)

        linkage_theshold = 0.3
        iou_threshold = 0.2

        updated_scores = seq_nms(boxes, scores, classes, linkage_theshold, iou_threshold)

        self.assertEqual(updated_scores.shape, scores.shape)
        self.assertTrue(not torch.equal(updated_scores, scores))
