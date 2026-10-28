"""Comparing synthetic rows with real rows."""

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from tabdiff.data import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS, TARGET_COLUMN

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


def _classifier(
    categorical: list[str], numeric: list[str], seed: int
) -> Pipeline:
    """One-hot encode ``categorical``, pass ``numeric`` through, then fit
    a gradient boosting classifier."""
    columns = ColumnTransformer(
        [
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical,
            ),
            ("numeric", "passthrough", numeric),
        ]
    )
    return Pipeline(
        [
            ("columns", columns),
            ("model", HistGradientBoostingClassifier(random_state=seed)),
        ]
    )


def _fit_and_score(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target: str,
    categorical: list[str],
    numeric: list[str],
    seed: int,
) -> tuple[float, float]:
    """Train on ``train_df`` and return (ROC AUC, accuracy) on ``test_df``."""
    pipeline = _classifier(categorical, numeric, seed)
    pipeline.fit(train_df.drop(columns=target), train_df[target])
    x_test, y_test = test_df.drop(columns=target), test_df[target]
    # predict_proba's second column is the probability of classes_[1].
    positive = pipeline.classes_[1]
    proba = pipeline.predict_proba(x_test)[:, 1]
    auc = roc_auc_score(y_test == positive, proba)
    accuracy = accuracy_score(y_test, pipeline.predict(x_test))
    return float(auc), float(accuracy)


def utility(
    real_train: pd.DataFrame,
    real_test: pd.DataFrame,
    fake_train: pd.DataFrame,
    target: str = TARGET_COLUMN,
    seed: int = 0,
) -> dict:
    """Train one classifier on real rows and one on fake rows, and test both
    on ``real_test``.

    Both classifiers use the same settings and the same number of rows: the
    larger of ``real_train`` and ``fake_train`` is cut down to the size of
    the smaller one by random sampling with ``seed``. Columns that are not
    numbers are one-hot encoded.

    Returns a dict with ``real_auc``, ``real_accuracy``, ``fake_auc``,
    ``fake_accuracy``, ``gap`` (real AUC minus fake AUC) and ``message``.
    ``message`` is None when everything worked. If ``fake_train`` has only
    one value of ``target``, no classifier can be trained on it: the fake
    numbers and the gap are NaN and ``message`` says why.
    """
    n_rows = min(len(real_train), len(fake_train))
    if len(real_train) > n_rows:
        real_train = real_train.sample(n_rows, random_state=seed)
    if len(fake_train) > n_rows:
        fake_train = fake_train.sample(n_rows, random_state=seed)

    features = real_train.drop(columns=target)
    categorical = list(features.select_dtypes(exclude="number").columns)
    numeric = list(features.select_dtypes(include="number").columns)

    real_auc, real_accuracy = _fit_and_score(
        real_train, real_test, target, categorical, numeric, seed
    )
    result = {
        "real_auc": real_auc,
        "real_accuracy": real_accuracy,
        "fake_auc": np.nan,
        "fake_accuracy": np.nan,
        "gap": np.nan,
        "message": None,
    }

    fake_classes = fake_train[target].unique()
    if len(fake_classes) < 2:
        result["message"] = (
            f"fake_train has only one value of {target!r} "
            f"({list(fake_classes)}), so no classifier can be trained on it."
        )
        return result

    fake_auc, fake_accuracy = _fit_and_score(
        fake_train, real_test, target, categorical, numeric, seed
    )
    result["fake_auc"] = fake_auc
    result["fake_accuracy"] = fake_accuracy
    result["gap"] = real_auc - fake_auc
    return result
