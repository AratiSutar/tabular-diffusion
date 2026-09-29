import numpy as np
import pandas as pd
import pytest

from tabdiff.preprocess import TabularEncoder

NUMERIC = ["age", "hours-per-week", "score"]
CATEGORICAL = ["workclass", "sex", "class"]


@pytest.fixture
def fake_df():
    rng = np.random.default_rng(0)
    n = 200
    return pd.DataFrame(
        {
            "age": rng.integers(18, 90, size=n),
            "workclass": rng.choice(["Private", "State-gov", "Missing"], size=n),
            "hours-per-week": rng.integers(1, 99, size=n),
            # A float column, so both the rounded and unrounded paths run.
            "score": rng.normal(50.0, 10.0, size=n),
            "sex": rng.choice(["Male", "Female"], size=n),
            "class": rng.choice(["<=50K", ">50K"], size=n),
        }
    )


@pytest.fixture
def encoder(fake_df):
    return TabularEncoder(NUMERIC, CATEGORICAL).fit(fake_df)


def test_transform_returns_float_array_without_nan(encoder, fake_df):
    array = encoder.transform(fake_df)
    assert isinstance(array, np.ndarray)
    assert array.dtype == np.float64
    assert not np.isnan(array).any()


def test_round_trip_gives_same_categories(encoder, fake_df):
    restored = encoder.inverse_transform(encoder.transform(fake_df))
    pd.testing.assert_frame_equal(restored[CATEGORICAL], fake_df[CATEGORICAL])


def test_round_trip_numeric_values_are_close(encoder, fake_df):
    restored = encoder.inverse_transform(encoder.transform(fake_df))
    for col in NUMERIC:
        np.testing.assert_allclose(restored[col], fake_df[col], rtol=1e-6)


def test_round_trip_keeps_columns_and_dtypes(encoder, fake_df):
    restored = encoder.inverse_transform(encoder.transform(fake_df))
    assert list(restored.columns) == list(fake_df.columns)
    pd.testing.assert_series_equal(restored.dtypes, fake_df.dtypes)
    assert not restored.isna().any().any()


def test_unseen_category_becomes_all_zero_block(encoder, fake_df):
    row = fake_df.iloc[[0]].copy()
    row["workclass"] = "Never-seen"

    array = encoder.transform(row)

    slices = encoder.column_slices()
    np.testing.assert_array_equal(array[:, slices["workclass"]], 0.0)
    # The other categorical blocks are still normal one-hot.
    for col in ["sex", "class"]:
        np.testing.assert_array_equal(array[:, slices[col]].sum(axis=1), 1.0)


def test_column_slices(encoder, fake_df):
    slices = encoder.column_slices()
    array = encoder.transform(fake_df)

    # Numeric columns first, then categorical, with no gaps or overlaps.
    assert list(slices) == NUMERIC + CATEGORICAL
    start = 0
    for col, s in slices.items():
        assert s.start == start
        start = s.stop
    assert start == array.shape[1]

    for col in NUMERIC:
        assert slices[col].stop - slices[col].start == 1
    for col in CATEGORICAL:
        block = array[:, slices[col]]
        assert block.shape[1] == fake_df[col].nunique()
        # Each row has exactly one 1 in the block for its column.
        np.testing.assert_array_equal(block.sum(axis=1), 1.0)
