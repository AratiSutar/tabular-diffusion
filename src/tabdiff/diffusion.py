"""Noise schedule and forward (noising) process for Gaussian diffusion."""

import math

import torch


def cosine_schedule(
    num_steps: int = 1000, s: float = 0.008, max_beta: float = 0.999
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(betas, alphas_cumprod)`` for the cosine schedule of
    Nichol & Dhariwal (2021), "Improved Denoising Diffusion Probabilistic Models".

    Both tensors have length ``num_steps`` and dtype float64. Betas are
    clipped to at most ``max_beta`` so the last steps stay stable, and
    ``alphas_cumprod`` is computed from the clipped betas.
    """

    def alpha_bar(step: torch.Tensor) -> torch.Tensor:
        return torch.cos((step / num_steps + s) / (1 + s) * math.pi / 2) ** 2

    steps = torch.arange(num_steps + 1, dtype=torch.float64)
    alpha_bars = alpha_bar(steps) / alpha_bar(torch.zeros(1, dtype=torch.float64))
    betas = (1 - alpha_bars[1:] / alpha_bars[:-1]).clamp(max=max_beta)
    alphas_cumprod = torch.cumprod(1 - betas, dim=0)
    return betas, alphas_cumprod


def q_sample(
    x0: torch.Tensor,
    t: torch.Tensor,
    noise: torch.Tensor,
    alphas_cumprod: torch.Tensor,
) -> torch.Tensor:
    """Return x_t = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * noise.

    ``t`` holds one step number per row of ``x0``. The result has the same
    shape and dtype as ``x0``.
    """
    alpha_bar_t = alphas_cumprod.to(x0.device)[t].to(x0.dtype)
    # Shape (batch, 1, ..., 1) so it broadcasts over the other dimensions of x0.
    alpha_bar_t = alpha_bar_t.view(-1, *([1] * (x0.dim() - 1)))
    return alpha_bar_t.sqrt() * x0 + (1 - alpha_bar_t).sqrt() * noise
