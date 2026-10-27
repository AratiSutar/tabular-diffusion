import math

import pytest
import torch

from tabdiff.diffusion import cosine_schedule
from tabdiff.model import Denoiser, diffusion_loss, timestep_embedding

NUM_STEPS = 100
N_FEATURES = 5


@pytest.fixture
def alphas_cumprod():
    _, alphas_cumprod = cosine_schedule(NUM_STEPS)
    return alphas_cumprod


@pytest.fixture
def x0():
    # Two clusters, so there is structure for the model to learn.
    gen = torch.Generator().manual_seed(0)
    centres = torch.tensor([[-2.0] * N_FEATURES, [2.0] * N_FEATURES])
    labels = torch.randint(0, 2, (128,), generator=gen)
    return centres[labels] + 0.1 * torch.randn(128, N_FEATURES, generator=gen)


def test_timestep_embedding_shape_and_values():
    t = torch.tensor([0, 1, 50])
    emb = timestep_embedding(t, 16)
    assert emb.shape == (3, 16)
    # At step 0 every sine is 0 and every cosine is 1.
    torch.testing.assert_close(emb[0], torch.cat([torch.zeros(8), torch.ones(8)]))


def test_timestep_embedding_odd_dim():
    emb = timestep_embedding(torch.tensor([3, 7]), 7)
    assert emb.shape == (2, 7)
    assert torch.all(emb[:, -1] == 0)


def test_denoiser_output_shape_equals_input_shape():
    torch.manual_seed(0)
    model = Denoiser(N_FEATURES)
    x_t = torch.randn(8, N_FEATURES)
    t = torch.randint(0, NUM_STEPS, (8,))
    assert model(x_t, t).shape == x_t.shape


def test_loss_is_finite(alphas_cumprod, x0):
    torch.manual_seed(0)
    loss = diffusion_loss(Denoiser(N_FEATURES), x0, alphas_cumprod)
    assert loss.dim() == 0
    assert math.isfinite(loss.item())


def test_training_makes_loss_go_down(alphas_cumprod, x0):
    torch.manual_seed(0)
    model = Denoiser(N_FEATURES)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    def eval_loss():
        # The same seed gives the same steps and noise, so the two
        # evaluations are comparable.
        torch.manual_seed(123)
        with torch.no_grad():
            return diffusion_loss(model, x0, alphas_cumprod).item()

    before = eval_loss()
    torch.manual_seed(1)
    for _ in range(200):
        optimizer.zero_grad()
        loss = diffusion_loss(model, x0, alphas_cumprod)
        loss.backward()
        optimizer.step()
    after = eval_loss()

    assert after < 0.5 * before
