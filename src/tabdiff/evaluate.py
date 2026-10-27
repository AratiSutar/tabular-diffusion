"""Comparing synthetic rows with real rows."""

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from tabdiff.data import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS

# In the Adult data every education level has exactly one education-num.
EDUCATION_NUM = {
    "Preschool": 1,
    "1st-4th": 2,
    "5th-6th": 3,
    "7th-8th": 4,
    "9th": 5,
    "10th": 6,
    "11th": 7,
    "12th": 8,
    "HS-grad": 9,
    "Some-college": 10,
    "Assoc-voc": 11,
    "Assoc-acdm": 12,
    "Bachelors": 13,
    "Masters": 14,
    "Prof-school": 15,
    "Doctorate": 16,
}


def compare(
    real_df: pd.DataFrame,
    fake_df: pd.DataFrame,
    numeric_columns: list[str] = NUMERIC_COLUMNS,
    categorical_columns: list[str] = CATEGORICAL_COLUMNS,
) -> pd.DataFrame:
    """Return one row per column saying how far apart real and fake are.

    ``distance`` is the Kolmogorov-Smirnov statistic for numeric columns and
    the total variation distance between category shares for categorical
    columns. Both go from 0 (identical) to 1 (completely different).
    ``real_mean`` and ``fake_mean`` are filled in for numeric columns only.
    """
    rows = []
    for col in numeric_columns:
        rows.append(
            {
                "column": col,
                "kind": "numeric",
                "distance": ks_2samp(real_df[col], fake_df[col]).statistic,
                "real_mean": real_df[col].mean(),
                "fake_mean": fake_df[col].mean(),
            }
        )
    for col in categorical_columns:
        real_shares = real_df[col].value_counts(normalize=True)
        fake_shares = fake_df[col].value_counts(normalize=True)
        # A category missing on one side counts as share 0 there.
        diff = real_shares.sub(fake_shares, fill_value=0).abs()
        rows.append(
            {
                "column": col,
                "kind": "categorical",
                "distance": 0.5 * diff.sum(),
                "real_mean": np.nan,
                "fake_mean": np.nan,
            }
        )
    return pd.DataFrame(rows).set_index("column")


def education_consistency(df: pd.DataFrame) -> float:
    """Return the share of rows whose education-num matches their education
    level, as given by ``EDUCATION_NUM``."""
    expected = df["education"].map(EDUCATION_NUM)
    return float((df["education-num"] == expected).mean())
