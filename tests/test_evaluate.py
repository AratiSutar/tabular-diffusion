import numpy as np
import pandas as pd
import pytest

from tabdiff.evaluate import EDUCATION_NUM, compare, education_consistency

NUMERIC = ["age"]
CATEGORICAL = ["sex"]


@pytest.fixture
def real_df():
    rng = np.random.default_rng(0)
    n = 200
    return pd.DataFrame(
        {
            "age": rng.integers(18, 60, size=n),
            "sex": rng.choice(["Male", "Female"], size=n),
        }
    )


def test_identical_data_gives_zero(real_df):
    table = compare(real_df, real_df.copy(), NUMERIC, CATEGORICAL)
    assert list(table.index) == ["age", "sex"]
    assert table.loc["age", "distance"] == 0
    assert table.loc["sex", "distance"] == 0
    assert table.loc["age", "real_mean"] == table.loc["age", "fake_mean"]


def test_very_different_data_gives_large_value(real_df):
    # Ages that never overlap, and a category the real data does not have.
    fake_df = pd.DataFrame({"age": real_df["age"] + 100, "sex": "Other"})
    table = compare(real_df, fake_df, NUMERIC, CATEGORICAL)
    assert table.loc["age", "distance"] == pytest.approx(1.0)
    assert table.loc["sex", "distance"] == pytest.approx(1.0)


def test_total_variation_distance_value():
    real = pd.DataFrame({"sex": ["Male"] * 50 + ["Female"] * 50})
    fake = pd.DataFrame({"sex": ["Male"] * 80 + ["Female"] * 20})
    table = compare(real, fake, numeric_columns=[], categorical_columns=["sex"])
    assert table.loc["sex", "distance"] == pytest.approx(0.3)


def test_consistency_is_one_on_real_style_rows():
    df = pd.DataFrame(
        {
            "education": list(EDUCATION_NUM),
            "education-num": list(EDUCATION_NUM.values()),
        }
    )
    assert education_consistency(df) == 1.0


def test_consistency_counts_mismatches():
    df = pd.DataFrame(
        {
            "education": ["HS-grad", "HS-grad", "Bachelors", "Masters"],
            "education-num": [9, 8, 13, 5],
        }
    )
    assert education_consistency(df) == 0.5
