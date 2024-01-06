import os
import unittest
from copy import deepcopy
from tempfile import TemporaryDirectory
from typing import List

import torch

from pt_seq_nms.seq_nms import seq_nms, seq_nms_from_list


def _save_load_module(jit_module):
    with TemporaryDirectory() as dir:
        file = os.path.join(dir, "module.pt")
        torch.jit.save(jit_module, file)
        loaded_module = torch.jit.load(file)
    return loaded_module


class TestSeqNMSModule(torch.nn.Module):
    def forward(self, boxes, scores, classes):
        return seq_nms(boxes, scores, classes, 0.2, 0.2)


class TestSeqNMSListModule(torch.nn.Module):
    def forward(self, boxes: List[torch.Tensor], scores: List[torch.Tensor], classes: List[torch.Tensor]) -> torch.Tensor:
        return seq_nms_from_list(boxes, scores, classes, 0.2, 0.2)


class TestSeqNMSScriptable(unittest.TestCase):
    def setUp(self):
        self.module = TestSeqNMSModule()
        torch.random.manual_seed(42)
        self.boxes = 100 * torch.rand((10, 2, 4)).float().cpu()
        self.scores = torch.rand((10, 2)).float().cpu()
        self.classes = torch.randint(0, 10, (10, 2)).int().cpu()

    def _call_module(self, module):
        _ = module(self.boxes, self.scores, self.classes)

    def _jit_save_load(self, m):
        jit_module = torch.jit.script(m)
        self._call_module(jit_module)

        loaded_module = _save_load_module(jit_module)
        self._call_module(loaded_module)

    def test_scriptable_cpu(self):
        m = deepcopy(self.module).cpu()
        self._jit_save_load(m)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_scriptable_gpu(self):
        m = deepcopy(self.module).cuda()
        self._jit_save_load(m)


class TestSeqNMSTracing(unittest.TestCase):
    def setUp(self):
        self.module = TestSeqNMSModule()
        torch.random.manual_seed(42)
        self.boxes = 100 * torch.rand((10, 2, 4)).float().cpu()
        self.scores = torch.rand((10, 2)).float().cpu()
        self.classes = torch.randint(0, 10, (10, 2)).int().cpu()

    def _call_module(self, module):
        _ = module(self.boxes, self.scores, self.classes)

    def _trace_save_load(self, m):
        trace_module = torch.jit.trace(m, (self.boxes, self.scores, self.classes))
        self._call_module(trace_module)

        loaded_module = _save_load_module(trace_module)
        self._call_module(loaded_module)

    def test_tracing_cpu(self):
        m = deepcopy(self.module).cpu()
        self._trace_save_load(m)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_tracing_gpu(self):
        m = deepcopy(self.module).cuda()
        self._trace_save_load(m)


class TestSeqNMSListScriptable(unittest.TestCase):
    def setUp(self):
        self.module = TestSeqNMSListModule()
        torch.random.manual_seed(42)
        self.boxes = [100 * torch.rand((2, 4)).float().cpu(), 100 * torch.rand((2, 4)).float().cpu()]
        self.scores = [torch.rand((2,)).float().cpu(), torch.rand((2,)).float().cpu()]
        self.classes = [torch.randint(0, 10, (2,)).int().cpu(), torch.randint(0, 10, (2,)).int().cpu()]

    def _call_module(self, module):
        _ = module(self.boxes, self.scores, self.classes)

    def _jit_save_load(self, m):
        jit_module = torch.jit.script(m)
        self._call_module(jit_module)

        loaded_module = _save_load_module(jit_module)
        self._call_module(loaded_module)

    def test_scriptable_cpu(self):
        m = deepcopy(self.module).cpu()
        self._jit_save_load(m)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_scriptable_gpu(self):
        m = deepcopy(self.module).cuda()
        self._jit_save_load(m)


class TestSeqNMSListTracing(unittest.TestCase):
    def setUp(self):
        self.module = TestSeqNMSListModule()
        torch.random.manual_seed(42)
        self.boxes = [100 * torch.rand((2, 4)).float().cpu(), 100 * torch.rand((2, 4)).float().cpu()]
        self.scores = [torch.rand((2,)).float().cpu(), torch.rand((2,)).float().cpu()]
        self.classes = [torch.randint(0, 10, (2,)).int().cpu(), torch.randint(0, 10, (2,)).int().cpu()]

    def test_scriptable_cpu(self):
        m = deepcopy(self.module).cpu()
        jit_module = torch.jit.trace(m, (self.boxes, self.scores, self.classes))

        _ = m(self.boxes, self.scores, self.classes)
        loaded_module = _save_load_module(jit_module)
        _ = loaded_module(self.boxes, self.scores, self.classes)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_scriptable_gpu(self):
        boxes = [b.cuda() for b in self.boxes]
        scores = [s.cuda() for s in self.scores]
        classes = [c.cuda() for c in self.classes]

        m = deepcopy(self.module).cuda()
        jit_module = torch.jit.trace(m, (boxes, scores, classes))
        _ = m(boxes, scores, classes)

        loaded_module = _save_load_module(jit_module)
        _ = loaded_module(boxes, scores, classes)


if __name__ == "__main__":
    unittest.main()
