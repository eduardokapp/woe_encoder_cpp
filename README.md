# Weight of Evidence (WoE) Encoder

A C++ implementation of WoE encoder exposed as Python package with scikit-learn compatible interface.

## Requirements
- CMake >= 3.12
- pybind11
- Python 3.7+
- scikit-learn
- C++17 compatible compiler

## Installation
1. Clone this repository
2. Build the project:
```bash
mkdir build
cd build
cmake ..
make
python ./build/setup.py bdist_wheel
popd
```# woe_encoder_cpp
