"""The denoising network and its training loss."""

import math

import torch
from torch import nn

from tabdiff.diffusion import q_sample


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Return the sinusoidal embedding of the step numbers ``t``, as in DDPM.

    ``t`` has shape ``(batch,)``. The result has shape ``(batch, dim)``: the
    first half holds sines and the second half cosines, at frequencies that
    go from 1 down to 1/10000. If ``dim`` is odd, the last column is zero.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, dtype=torch.float32, device=t.device) / half
    )
    angles = t.float()[:, None] * freqs[None, :]
    embedding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)
    if dim % 2 == 1:
        embedding = nn.functional.pad(embedding, (0, 1))
    return embedding


class Denoiser(nn.Module):
    """An MLP that predicts the noise in ``x_t`` at step ``t``.

    The time embedding is joined to ``x_t`` and passed through 3 hidden
    layers with SiLU activations. The output has the same shape as ``x_t``.
    """

    def __init__(self, n_features: int, hidden_dim: int = 256, time_dim: int = 128):
        super().__init__()
        self.time_dim = time_dim
        self.net = nn.Sequential(
            nn.Linear(n_features + time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, n_features),
        )

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        emb = timestep_embedding(t, self.time_dim).to(x_t.dtype)
        return self.net(torch.cat([x_t, emb], dim=1))


def diffusion_loss(
    model: nn.Module, x0: torch.Tensor, alphas_cumprod: torch.Tensor
) -> torch.Tensor:
    """Return the mean squared error between the real and predicted noise.

    Each row of ``x0`` gets its own random step ``t`` and its own noise.
    """
    t = torch.randint(0, len(alphas_cumprod), (len(x0),), device=x0.device)
    noise = torch.randn_like(x0)
    x_t = q_sample(x0, t, noise, alphas_cumprod)
    return nn.functional.mse_loss(model(x_t, t), noise)
