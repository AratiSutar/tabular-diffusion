import numpy as np
import pandas as pd
import pytest

from tabdiff.baselines import smote_sample

NUMERIC = ["age", "score"]
CATEGORICAL = ["sex", "class"]


@pytest.fixture
def real_df():
    rng = np.random.default_rng(0)
    n = 200
    return pd.DataFrame(
        {
            "age": rng.integers(18, 90, size=n),
            "sex": rng.choice(["Male", "Female"], size=n),
            # A float column, so both the rounded and unrounded paths run.
            "score": rng.normal(50.0, 10.0, size=n),
            "class": rng.choice(["<=50K", ">50K"], size=n),
        }
    )


def smote(df, n_rows=300, **kwargs):
    return smote_sample(
        df, n_rows, numeric_columns=NUMERIC, categorical_columns=CATEGORICAL, **kwargs
    )


def test_shape_columns_and_dtypes(real_df):
    fake = smote(real_df)
    assert len(fake) == 300
    assert list(fake.columns) == list(real_df.columns)
    assert (fake.dtypes == real_df.dtypes).all()


def test_values_stay_in_real_range(real_df):
    fake = smote(real_df)
    for col in NUMERIC:
        assert fake[col].between(real_df[col].min(), real_df[col].max()).all()
    for col in CATEGORICAL:
        assert set(fake[col]) <= set(real_df[col])


def test_same_seed_gives_same_rows(real_df):
    pd.testing.assert_frame_equal(smote(real_df, seed=1), smote(real_df, seed=1))
    assert not smote(real_df, seed=1).equals(smote(real_df, seed=2))


def test_partner_has_the_same_class():
    # Each class has its own age range and its own sex. A partner from the
    # other class would give an age or a sex outside the class's own.
    df = pd.DataFrame(
        {
            "age": list(range(20, 30)) + list(range(60, 70)),
            "score": np.linspace(0.0, 1.0, 20),
            "sex": ["Male"] * 10 + ["Female"] * 10,
            "class": ["<=50K"] * 10 + [">50K"] * 10,
        }
    )
    fake = smote(df)
    low = fake[fake["class"] == "<=50K"]
    high = fake[fake["class"] == ">50K"]
    assert len(low) > 0 and len(high) > 0
    assert low["age"].between(20, 29).all() and (low["sex"] == "Male").all()
    assert high["age"].between(60, 69).all() and (high["sex"] == "Female").all()


def test_partner_is_a_near_neighbour():
    # One class, two groups of 6 rows far apart: the 5 nearest neighbours of
    # a row are the other rows of its group, so new rows stay in a group.
    df = pd.DataFrame(
        {
            "age": list(range(20, 26)) + list(range(80, 86)),
            "score": [0.0] * 6 + [100.0] * 6,
            "sex": ["Male"] * 6 + ["Female"] * 6,
            "class": "<=50K",
        }
    )
    fake = smote(df)
    young = fake["age"] <= 25
    assert (young | (fake["age"] >= 80)).all()
    assert (fake.loc[young, "sex"] == "Male").all()
    assert (fake.loc[~young, "sex"] == "Female").all()
    assert (fake.loc[young, "score"] == 0.0).all()


def test_mixes_values_of_two_rows():
    df = pd.DataFrame(
        {
            "age": [20, 30],
            "score": [0.0, 10.0],
            "sex": ["Male", "Female"],
            "class": "<=50K",
        }
    )
    fake = smote(df)
    # The score is a point between the two rows, not always one of them.
    assert ((fake["score"] > 0) & (fake["score"] < 10)).any()
    # The age and score come from the same point on the line.
    assert np.allclose(fake["age"], 20 + fake["score"], atol=0.5)
    # Each category is taken from either row.
    assert set(fake["sex"]) == {"Male", "Female"}


def test_class_with_one_row_is_copied():
    df = pd.DataFrame(
        {
            "age": [20, 21, 22, 50],
            "score": [1.0, 2.0, 3.0, 9.0],
            "sex": ["Male", "Male", "Female", "Female"],
            "class": ["<=50K", "<=50K", "<=50K", ">50K"],
        }
    )
    fake = smote(df, n_rows=200)
    alone = fake[fake["class"] == ">50K"]
    assert len(alone) > 0
    assert (alone[["age", "score", "sex"]] == [50, 9.0, "Female"]).all().all()
