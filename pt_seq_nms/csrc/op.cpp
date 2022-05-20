#include <torch/torch.h>
#include "seq_nms.h"

#ifdef _WIN32
#include <Python.h>
PyMODINIT_FUNC PyInit_seq_nms(void) {
    return NULL;
}
#endif

TORCH_LIBRARY(seq_nms, m) {
    m.def("seq_nms", &seq_nms);
}
