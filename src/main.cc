#include "fast_woe_encoder.h"
#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(_core, m) {
  m.doc() = "pybind11 example plugin";

  // Binding for WoEEncoderOptions struct
  py::class_<fast_woe_encoder::WoEEncoderOptions>(m, "WoEEncoderOptions")
      .def(py::init<>())
      .def_readwrite("epsilon", &fast_woe_encoder::WoEEncoderOptions::epsilon)
      .def_readwrite("default_woe",
                     &fast_woe_encoder::WoEEncoderOptions::default_woe)
      .def_readwrite("verbose", &fast_woe_encoder::WoEEncoderOptions::verbose);

  // Binding for WoEEncoder class - Corrected Constructor Binding
  py::class_<fast_woe_encoder::WoEEncoder>(m, "WoEEncoder")
      .def(py::init<fast_woe_encoder::WoEEncoderOptions>(),
           py::arg("options") = fast_woe_encoder::WoEEncoderOptions())
      .def("fit", &fast_woe_encoder::WoEEncoder::Fit)
      .def("transform", &fast_woe_encoder::WoEEncoder::Transform);
}