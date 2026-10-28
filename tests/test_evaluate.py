import numpy as np
import pandas as pd
import pytest

from tabdiff.evaluate import EDUCATION_NUM, compare, education_consistency, utility

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


def make_labelled(n, seed):
    """Small Adult-like rows where the label depends on age and sex."""
    rng = np.random.default_rng(seed)
    age = rng.integers(18, 70, size=n)
    sex = rng.choice(["Male", "Female"], size=n)
    score = (age - 40) / 10 + (sex == "Male") + rng.normal(size=n)
    label = np.where(score > 0.5, ">50K", "<=50K")
    return pd.DataFrame({"age": age, "sex": sex, "class": label})


def test_utility_gap_is_zero_when_fake_equals_real():
    train, test = make_labelled(300, seed=0), make_labelled(200, seed=1)
    result = utility(train, test, train.copy())
    assert result["message"] is None
    assert result["gap"] == pytest.approx(0.0, abs=1e-9)
    assert result["real_accuracy"] == pytest.approx(result["fake_accuracy"])


def test_utility_scores_are_in_range():
    train, test = make_labelled(300, seed=0), make_labelled(200, seed=1)
    fake = make_labelled(300, seed=2)
    result = utility(train, test, fake)
    for key in ["real_auc", "fake_auc", "real_accuracy", "fake_accuracy"]:
        assert 0 <= result[key] <= 1


def test_utility_one_class_fake_data_gives_message():
    train, test = make_labelled(300, seed=0), make_labelled(200, seed=1)
    fake = train.assign(**{"class": "<=50K"})
    result = utility(train, test, fake)
    assert "only one value" in result["message"]
    assert np.isnan(result["fake_auc"])
    assert np.isnan(result["gap"])
    assert 0 <= result["real_auc"] <= 1
