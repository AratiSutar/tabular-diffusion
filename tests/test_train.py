import json
import math

import numpy as np
import pandas as pd
import pytest
import torch

from tabdiff.data import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS
from tabdiff.train import CONFIG_FILE, ENCODER_FILE, MODEL_FILE, load_run, train


@pytest.fixture
def fake_df():
    # Every Adult column, filled with random values.
    rng = np.random.default_rng(0)
    n = 100
    df = pd.DataFrame({col: rng.integers(0, 100, size=n) for col in NUMERIC_COLUMNS})
    for col in CATEGORICAL_COLUMNS:
        df[col] = rng.choice(["a", "b", "Missing"], size=n)
    return df


def test_train_saves_files_and_load_run_works(fake_df, tmp_path):
    out_dir = tmp_path / "run"
    model, encoder = train(fake_df, num_steps=20, batch_size=32, out_dir=out_dir)

    for name in [MODEL_FILE, ENCODER_FILE, CONFIG_FILE]:
        assert (out_dir / name).exists()

    config = json.loads((out_dir / CONFIG_FILE).read_text())
    assert config["num_steps"] == 20
    assert math.isfinite(config["final_loss"])

    loaded_model, loaded_encoder = load_run(out_dir)

    # Same weights, so the same output for the same input.
    x = torch.tensor(encoder.transform(fake_df[:8]), dtype=torch.float32)
    t = torch.arange(8)
    with torch.no_grad():
        torch.testing.assert_close(loaded_model(x, t), model(x, t))

    # The loaded encoder gives the same array as the original one.
    np.testing.assert_array_equal(
        loaded_encoder.transform(fake_df), encoder.transform(fake_df)
    )
