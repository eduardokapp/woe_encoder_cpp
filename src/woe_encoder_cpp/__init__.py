# woe_encoder_cpp/__init__.py
"""woe_encoder_cpp package initializer."""
__version__ = "0.1.1"  # Updated version

import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from typing import List

from . import _core  # Import the C++ extension


class WoEEncoder(_core.WoEEncoder):
    def __init__(self, cols: List[str] = None,
                 epsilon: float = None,
                 default_woe: float = None,
                 verbose: int = None):
        cpp_options = _core.WoEEncoderOptions()

        if epsilon is not None:
            cpp_options.epsilon = epsilon
        if default_woe is not None:
            cpp_options.default_woe = default_woe
        if verbose is not None:
            cpp_options.verbose = verbose

        super().__init__(cpp_options)
        self.ordinal_encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=-1)
        self.cols = cols

    def fit(self, features: pd.DataFrame, target: np.ndarray):
        X_encoded = self._encode_categorical(features, fit=True)

        # Convert to column-major Fortran contiguous array for efficiency
        super().fit(X_encoded[self.cols].astype(np.int32).to_numpy().T, target)
        return self

    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        X_encoded = self._encode_categorical(features, fit=False)

        # Get the transformed data from the C++ extension and convert to numpy
        transformed_list = super().transform(X_encoded[self.cols].astype(np.int32).to_numpy().T)
        transformed_np = np.array(transformed_list).T  # Transpose during conversion

        X_encoded[self.cols] = transformed_np
        return X_encoded

    def fit_tranform(self, features: pd.DataFrame, target: np.ndarray):
        X_encoded = self._encode_categorical(features, fit=True)

        # Convert to column-major Fortran contiguous array for efficiency
        array_data = X_encoded[self.cols].astype(np.int32).to_numpy().T
        super().fit(array_data, target)

        transformed_list = super().transform(array_data)
        transformed_np = np.array(transformed_list).T

        X_encoded[self.cols] = transformed_np

        return X_encoded

    def _encode_categorical(self, X: pd.DataFrame, fit: bool) -> pd.DataFrame:
        """Encodes categorical columns using OrdinalEncoder if needed."""
        # Do a copy of the original df
        X = X.copy()
        # Encode all columns if no specific columns are provided.
        if self.cols is None or len(self.cols) == 0:
            self.cols = X.select_dtypes(
                include=['object', 'category']).columns.tolist()

        if self.cols and fit:
            X[self.cols] = self.ordinal_encoder.fit_transform(X[self.cols])
        elif self.cols:
            X[self.cols] = self.ordinal_encoder.transform(X[self.cols])

        return X
