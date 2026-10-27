import numpy as np
import pandas as pd
import pytest
import torch

from tabdiff.diffusion import cosine_schedule, sample
from tabdiff.model import Denoiser, diffusion_loss
from tabdiff.preprocess import TabularEncoder

NUMERIC = ["age", "score"]
CATEGORICAL = ["sex", "class"]
NUM_STEPS = 50


@pytest.fixture
def fake_df():
    rng = np.random.default_rng(0)
    n = 100
    return pd.DataFrame(
        {
            "age": rng.integers(18, 90, size=n),
            "sex": rng.choice(["Male", "Female"], size=n),
            # A float column, so both the rounded and unrounded paths run.
            "score": rng.normal(50.0, 10.0, size=n),
            "class": rng.choice(["<=50K", ">50K", "Missing"], size=n),
        }
    )


@pytest.fixture
def trained(fake_df):
    torch.manual_seed(0)
    encoder = TabularEncoder(NUMERIC, CATEGORICAL).fit(fake_df)
    data = torch.tensor(encoder.transform(fake_df), dtype=torch.float32)
    _, alphas_cumprod = cosine_schedule(NUM_STEPS)
    model = Denoiser(data.shape[1], hidden_dim=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(5):
        optimizer.zero_grad()
        diffusion_loss(model, data, alphas_cumprod).backward()
        optimizer.step()
    return model, encoder


def test_sample_has_right_rows_columns_and_dtypes(trained, fake_df):
    model, encoder = trained
    fake = sample(model, encoder, n_rows=30, num_steps=NUM_STEPS)
    assert len(fake) == 30
    assert list(fake.columns) == list(fake_df.columns)
    pd.testing.assert_series_equal(fake.dtypes, fake_df.dtypes)


def test_sample_categories_are_known(trained, fake_df):
    model, encoder = trained
    fake = sample(model, encoder, n_rows=30, num_steps=NUM_STEPS)
    for col in CATEGORICAL:
        assert set(fake[col]) <= set(fake_df[col])


class HugeNoiseModel(torch.nn.Module):
    """Always predicts very large noise, and records the largest |x_t| it sees."""

    def __init__(self):
        super().__init__()
        self.max_abs_input = 0.0

    def forward(self, x_t, t):
        self.max_abs_input = max(self.max_abs_input, x_t.abs().max().item())
        return torch.full_like(x_t, 1e6)


def test_clamping_keeps_sampling_bounded(trained):
    _, encoder = trained
    model = HugeNoiseModel()
    fake = sample(model, encoder, n_rows=30, num_steps=NUM_STEPS)
    # Without clamping, x_t would grow to around 1e6 or more.
    assert model.max_abs_input < 20
    assert not fake.isna().any().any()


def test_same_seed_same_result(trained):
    model, encoder = trained
    a = sample(model, encoder, n_rows=30, num_steps=NUM_STEPS, seed=7)
    b = sample(model, encoder, n_rows=30, num_steps=NUM_STEPS, seed=7)
    pd.testing.assert_frame_equal(a, b)
