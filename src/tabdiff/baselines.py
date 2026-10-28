"""Simple ways to make synthetic rows, to compare the diffusion model with."""

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from tabdiff.data import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS, TARGET_COLUMN
from tabdiff.preprocess import TabularEncoder


def smote_sample(
    train_df: pd.DataFrame,
    n_rows: int,
    k: int = 5,
    numeric_columns: list[str] = NUMERIC_COLUMNS,
    categorical_columns: list[str] = CATEGORICAL_COLUMNS,
    target: str = TARGET_COLUMN,
    seed: int = 0,
) -> pd.DataFrame:
    """Make ``n_rows`` synthetic rows in the style of SMOTE.

    For each new row it picks a random ``train_df`` row and one of its ``k``
    nearest neighbours with the same ``target`` value. Numeric columns take
    a random point on the line between the two rows (one random position
    per new row, shared by all numeric columns). Each categorical column
    takes the value of one of the two rows, chosen at random.

    Neighbours are found in the space of a ``TabularEncoder`` fitted on
    ``train_df``. A class with ``k`` rows or fewer uses all its other rows
    as neighbours, and a row that is alone in its class is paired with
    itself. Integer columns are rounded and all columns keep the dtypes of
    ``train_df``. ``target`` must be one of ``categorical_columns``.
    """
    rng = np.random.default_rng(seed)
    train_df = train_df.reset_index(drop=True)
    encoder = TabularEncoder(numeric_columns, categorical_columns, seed=seed)
    encoded = encoder.fit(train_df).transform(train_df)

    labels = train_df[target].to_numpy()
    base = rng.integers(0, len(train_df), size=n_rows)
    partner = np.empty(n_rows, dtype=int)
    for label in np.unique(labels[base]):
        members = np.flatnonzero(labels == label)
        picked = np.flatnonzero(labels[base] == label)
        k_class = min(k, len(members) - 1)
        if k_class == 0:
            partner[picked] = base[picked]
            continue
        # With no rows passed in, kneighbors leaves each row out of its own
        # neighbours. The result holds positions within ``members``.
        neighbours = (
            NearestNeighbors(n_neighbors=k_class)
            .fit(encoded[members])
            .kneighbors(return_distance=False)
        )
        position = np.searchsorted(members, base[picked])
        choice = rng.integers(0, k_class, size=len(picked))
        partner[picked] = members[neighbours[position, choice]]

    rows_a = train_df.iloc[base].reset_index(drop=True)
    rows_b = train_df.iloc[partner].reset_index(drop=True)
    out = {}

    step = rng.random(n_rows)
    for col in numeric_columns:
        values = rows_a[col] + step * (rows_b[col] - rows_a[col])
        if pd.api.types.is_integer_dtype(train_df[col]):
            values = np.rint(values)
        out[col] = values

    for col in categorical_columns:
        from_b = rng.random(n_rows) < 0.5
        out[col] = np.where(from_b, rows_b[col], rows_a[col])

    columns = [col for col in train_df.columns if col in out]
    return pd.DataFrame(out)[columns].astype(train_df[columns].dtypes.to_dict())
