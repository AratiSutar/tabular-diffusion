import numpy as np
import pandas as pd
import pytest

from tabdiff.data import (
    CATEGORICAL_COLUMNS,
    MISSING_CATEGORY,
    NUMERIC_COLUMNS,
    TARGET_COLUMN,
    load_adult,
    train_test_split_fixed,
)


@pytest.fixture
def fake_df():
    rng = np.random.default_rng(123)
    n = 100
    return pd.DataFrame(
        {
            # Unique ids make every row distinct, so overlap checks are exact.
            "id": np.arange(n),
            "age": rng.integers(18, 90, size=n),
            "hours-per-week": rng.integers(1, 99, size=n),
            "sex": rng.choice(["Male", "Female"], size=n),
            "class": rng.choice(["<=50K", ">50K"], size=n),
        }
    )


def test_split_sizes(fake_df):
    train, test = train_test_split_fixed(fake_df, test_size=0.2, seed=0)
    assert len(train) == 80
    assert len(test) == 20


def test_same_seed_same_split(fake_df):
    train_a, test_a = train_test_split_fixed(fake_df, seed=42)
    train_b, test_b = train_test_split_fixed(fake_df, seed=42)
    pd.testing.assert_frame_equal(train_a, train_b)
    pd.testing.assert_frame_equal(test_a, test_b)


def test_different_seed_different_split(fake_df):
    _, test_a = train_test_split_fixed(fake_df, seed=0)
    _, test_b = train_test_split_fixed(fake_df, seed=1)
    assert set(test_a["id"]) != set(test_b["id"])


def test_train_and_test_do_not_share_rows(fake_df):
    train, test = train_test_split_fixed(fake_df, seed=0)
    assert set(train.index).isdisjoint(test.index)
    assert set(train["id"]).isdisjoint(test["id"])
    assert len(train) + len(test) == len(fake_df)


def test_column_lists():
    assert set(NUMERIC_COLUMNS).isdisjoint(CATEGORICAL_COLUMNS)
    assert TARGET_COLUMN in CATEGORICAL_COLUMNS


def test_load_adult_fills_missing_categories(tmp_path):
    n = 4
    df = pd.DataFrame({col: np.arange(n) for col in NUMERIC_COLUMNS})
    for col in CATEGORICAL_COLUMNS:
        df[col] = ["a", "b", "c", "d"]
    # The same columns that have gaps in the real data.
    df.loc[0, "workclass"] = np.nan
    df.loc[1, "occupation"] = np.nan
    df.loc[[1, 3], "native-country"] = np.nan
    # Writing the file into tmp_path means load_adult reads it from there
    # and never downloads anything.
    df.to_csv(tmp_path / "adult.csv", index=False)

    loaded = load_adult(cache_dir=tmp_path)

    assert not loaded.isna().any().any()
    assert loaded.loc[0, "workclass"] == MISSING_CATEGORY
    assert loaded.loc[1, "occupation"] == MISSING_CATEGORY
    assert list(loaded["native-country"]) == ["a", MISSING_CATEGORY, "c", MISSING_CATEGORY]
    # Values that were present are left alone.
    assert loaded.loc[2, "workclass"] == "c"
    pd.testing.assert_frame_equal(loaded[NUMERIC_COLUMNS], df[NUMERIC_COLUMNS])
