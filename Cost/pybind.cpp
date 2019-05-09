#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include "cost_compute.h"
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("Intrinsic_Cost",&Intrinsic_Cost);
}
