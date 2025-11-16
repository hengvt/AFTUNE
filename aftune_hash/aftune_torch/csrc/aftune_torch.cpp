#include <Python.h>
#include <torch/library.h>

extern "C" {
  PyObject* PyInit__C(void) {
      static struct PyModuleDef module_def = {
          PyModuleDef_HEAD_INIT,
          "_C",
          NULL,
          -1,
          NULL,
      };
      return PyModule_Create(&module_def);
  }
}

namespace aftune_torch {

TORCH_LIBRARY(aftune_torch, m) {
  m.def("sha256(Tensor a, int chunk_size=128) -> Tensor");
  m.def("blaze3(Tensor a, int chunk_size=128) -> Tensor");
}

}
