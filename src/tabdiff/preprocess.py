"""Turning an Adult DataFrame into a float array for the model, and back."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer

from tabdiff.data import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS


class TabularEncoder:
    """Encode numeric columns with a normal quantile transform and
    categorical columns (including the target) with one-hot encoding.

    The output array has the numeric columns first, then one block of
    one-hot columns per categorical column. ``column_slices`` says which
    array columns belong to which original column.
    """

    def __init__(
        self,
        numeric_columns: list[str] = NUMERIC_COLUMNS,
        categorical_columns: list[str] = CATEGORICAL_COLUMNS,
        n_quantiles: int = 1000,
        seed: int = 0,
    ):
        self.numeric_columns = list(numeric_columns)
        self.categorical_columns = list(categorical_columns)
        self.n_quantiles = n_quantiles
        self.seed = seed

    def fit(self, df: pd.DataFrame) -> "TabularEncoder":
        encoded = set(self.numeric_columns + self.categorical_columns)
        # Remember the original column order and dtypes for inverse_transform.
        self.columns_ = [col for col in df.columns if col in encoded]
        self.dtypes_ = df[self.columns_].dtypes

        # QuantileTransformer cannot use more quantiles than there are rows.
        self.quantile_ = QuantileTransformer(
            n_quantiles=min(self.n_quantiles, len(df)),
            output_distribution="normal",
            random_state=self.seed,
        )
        self.quantile_.fit(df[self.numeric_columns].to_numpy(dtype=float))

        # A category not seen in fit becomes an all-zero block, not an error.
        self.onehot_ = OneHotEncoder(
            handle_unknown="ignore", sparse_output=False, dtype=np.float64
        )
        self.onehot_.fit(df[self.categorical_columns])
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        numeric = self.quantile_.transform(df[self.numeric_columns].to_numpy(dtype=float))
        categorical = self.onehot_.transform(df[self.categorical_columns])
        return np.hstack([numeric, categorical]).astype(np.float64)

    def inverse_transform(self, array: np.ndarray) -> pd.DataFrame:
        slices = self.column_slices()
        out = {}

        n_numeric = len(self.numeric_columns)
        numeric = self.quantile_.inverse_transform(array[:, :n_numeric])
        for i, col in enumerate(self.numeric_columns):
            values = numeric[:, i]
            if pd.api.types.is_integer_dtype(self.dtypes_[col]):
                values = np.rint(values)
            out[col] = values

        for col, categories in zip(self.categorical_columns, self.onehot_.categories_):
            # The category with the highest value in its one-hot block wins.
            out[col] = categories[array[:, slices[col]].argmax(axis=1)]

        df = pd.DataFrame(out)[self.columns_]
        return df.astype(self.dtypes_.to_dict())

    def column_slices(self) -> dict[str, slice]:
        """Map each original column to the array columns that hold it."""
        slices = {}
        start = 0
        for col in self.numeric_columns:
            slices[col] = slice(start, start + 1)
            start += 1
        for col, categories in zip(self.categorical_columns, self.onehot_.categories_):
            slices[col] = slice(start, start + len(categories))
            start += len(categories)
        return slices
