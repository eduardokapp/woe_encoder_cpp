# Weight of Evidence (WoE) Encoder

A C++ implementation of WoE encoder exposed as Python package with scikit-learn compatible interface.

For now the implementation is very simple:

* A python wrapper class takes in a Pandas dataframe and applies ordinal encoder
  on the selected columns (or all if none are selected)
* The transformed, ordinal-encoded columns are passed as a column-major vector of vector int to the C++ class.

# Usage
Should be very similar to [category_encoders.WoEEncoder](https://contrib.scikit-learn.org/category_encoders/woe.html).

For now:

```python
from woe_encoder_cpp import WoEEncoder

# data = Your data here..

enc = WoEEncoder()
data = enc.fit_transform()
```

## TODO: Features to be added
### Behavioral
- [x] Basic implementation that calculates WoE values correctly and is able
to fit/transform/fit_transform the data.
- [ ] Make the Python wrapper class equivalent (as possible) with category_encoders.WoEEncoder
- [ ] Make the wrapper class inherit from scikit transformer

### Infra, CI/CD
- [x] Build system that does not make the user need to know anything about C++ or CMake, just `pip install` and that's it.
- [ ] Basic unit testing
- [ ] CI pipeline that runs unit tests and also applies linting
- [ ] Push lib to pypi? CD pipeline

### Optimization
- [ ] Make the C++ class accept contiguous, row-major order arrays instead of fortran style column-major.
- [ ] Reduce number of copies made in the python wrapper
- [ ] Implement ordinal encoder in C++ as well
- [ ] Use NOPYTHON mode so that no wrapper is necessary anymore.

## Requirements
- CMake >= 3.12
- pybind11
- Python 3.7+
- scikit-learn
- C++17 compatible compiler

These will be taken care of by the scikit-build if you don't have them installed.

## Installation
1. Clone this repository
2. Install using `pip install .`
