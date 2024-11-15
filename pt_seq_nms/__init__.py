import os

import torch

from .seq_nms import seq_nms, seq_nms_from_list  # noqa: F401

if os.name == "nt":
    file = "seq_nms.pyd"
else:
    file = "seq_nms.so"

# TODO: see why the .so files is installed below our package folder
this_dir = os.path.dirname(__file__)
torch.ops.load_library(os.path.join(this_dir, "..", file))  # type: ignore
