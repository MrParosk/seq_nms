# Seq-nms in PyTorch

![main](https://github.com/MrParosk/seq_nms/workflows/main/badge.svg?branch=main) [![codecov](https://codecov.io/gh/MrParosk/seq_nms/branch/main/graph/badge.svg?token=7DYQ1CHZQS)](https://codecov.io/gh/MrParosk/seq_nms)

Implementation of the seq-nms algorithm described in the paper: [Seq-NMS for Video Object Detection](https://arxiv.org/abs/1602.08465)

The algorithm is implemented in PyTorch's C++ frontend for better performance.

It can be exported with both PyTorch's scripting and tracing.

## Install

Make sure that you have installed PyTorch, version 1.7 or higher. Install the package by

```Shell
pip install git+https://github.com/MrParosk/seq_nms.git
```

Note that if you are using Windows, you need MSVC installed.
