"""Loading and splitting the UCI Adult dataset."""

from pathlib import Path

import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

NUMERIC_COLUMNS = [
    "age",
    "fnlwgt",
    "education-num",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
]

# "class" is the income label (<=50K / >50K). It is listed here because a
# generative model synthesises it along with the other columns.
CATEGORICAL_COLUMNS = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
    "class",
]

TARGET_COLUMN = "class"

MISSING_CATEGORY = "Missing"


def load_adult(cache_dir: str | Path = "data") -> pd.DataFrame:
    """Return the Adult dataset, downloading it from OpenML on first use.

    The data is cached as ``<cache_dir>/adult.csv``. Both the first call and
    later calls read from that CSV, so the returned dtypes are always the same.

    Missing values (``?`` in the original data) in the categorical columns
    are replaced with the category ``"Missing"``, so a generative model can
    learn how often they occur.
    """
    cache_path = Path(cache_dir) / "adult.csv"
    if not cache_path.exists():
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        bunch = fetch_openml(name="adult", version=2, as_frame=True)
        bunch.frame.to_csv(cache_path, index=False)

    df = pd.read_csv(cache_path)
    df = df[NUMERIC_COLUMNS + CATEGORICAL_COLUMNS].copy()
    df[CATEGORICAL_COLUMNS] = df[CATEGORICAL_COLUMNS].fillna(MISSING_CATEGORY)
    return df


def train_test_split_fixed(
    df: pd.DataFrame, test_size: float = 0.2, seed: int = 0
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split ``df`` into train and test sets, reproducibly for a given seed."""
    train, test = train_test_split(df, test_size=test_size, random_state=seed)
    return train, test
