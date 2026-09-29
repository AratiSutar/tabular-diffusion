import pytest
import torch

from tabdiff.diffusion import cosine_schedule, q_sample

NUM_STEPS = 1000


@pytest.fixture
def schedule():
    return cosine_schedule(NUM_STEPS)


@pytest.fixture
def x0_and_noise():
    gen = torch.Generator().manual_seed(0)
    x0 = torch.randn(64, 10, generator=gen)
    noise = torch.randn(64, 10, generator=gen)
    return x0, noise


def test_alphas_cumprod_between_0_and_1_and_decreasing(schedule):
    _, alphas_cumprod = schedule
    assert alphas_cumprod.shape == (NUM_STEPS,)
    assert torch.all(alphas_cumprod > 0)
    assert torch.all(alphas_cumprod < 1)
    assert torch.all(alphas_cumprod[1:] < alphas_cumprod[:-1])


def test_first_step_is_close_to_x0(schedule, x0_and_noise):
    _, alphas_cumprod = schedule
    x0, noise = x0_and_noise
    t = torch.zeros(len(x0), dtype=torch.long)
    x_t = q_sample(x0, t, noise, alphas_cumprod)
    torch.testing.assert_close(x_t, x0, atol=0.05, rtol=0)


def test_last_step_is_close_to_noise(schedule, x0_and_noise):
    _, alphas_cumprod = schedule
    x0, noise = x0_and_noise
    t = torch.full((len(x0),), NUM_STEPS - 1, dtype=torch.long)
    x_t = q_sample(x0, t, noise, alphas_cumprod)
    torch.testing.assert_close(x_t, noise, atol=1e-3, rtol=0)


def test_q_sample_keeps_shape(schedule, x0_and_noise):
    _, alphas_cumprod = schedule
    x0, noise = x0_and_noise
    # A different step for every row.
    t = torch.randint(0, NUM_STEPS, (len(x0),), generator=torch.Generator().manual_seed(1))
    x_t = q_sample(x0, t, noise, alphas_cumprod)
    assert x_t.shape == x0.shape
