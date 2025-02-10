#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 21:02:13 2025

@author: eduardokapp
"""

import time
import pandas as pd
import numpy as np

from woe_encoder_cpp import WoEEncoder as woe1
from category_encoders import WOEEncoder as woe2
from sklearn.datasets import fetch_openml

# Number of rows and columns
n_rows = 1000
n_cols = 100

# Number of unique categories per column
n_categories = 20

# Create a list of column names
column_names = [f'col_{i}' for i in range(n_cols)]

# Generate random category indices using numpy (more efficient)
category_indices = np.random.randint(0, n_categories, size=(n_rows, n_cols))

# Create the DataFrame directly from the numpy array
df = pd.DataFrame(category_indices, columns=column_names)

# # Convert all columns to categorical type
for col in df.columns:
    df[col] = df[col].astype('category')

X1 = df.copy()
X2 = df.copy()
y = np.random.choice([True, False], size=n_rows)


# Lame timing for custom version

start_time = time.time()

enc1 = woe1(verbose=0).fit(X1, y)
transformed1 = enc1.transform(X1)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"fast_woe_encoder elapsed time: {elapsed_time:.4f} seconds")

# Lame timing for cat encoders version

start_time = time.time() # Time the entire process

enc2 = woe2(regularization=0).fit(X2, y)
transformed2 = enc2.transform(X2)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"category_encoders elapsed time: {elapsed_time:.4f} seconds")

# Sanity check
print(transformed1)
print(transformed2)
