"""Noise schedule and forward (noising) process for Gaussian diffusion."""

import math

import pandas as pd
import torch
from torch import nn

# A normal quantile transform outputs at most about +-5.2 (sklearn clips
# there), so a predicted clean numeric value outside this range is an error.
NUMERIC_CLAMP = 5.2


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


def sample(
    model: nn.Module, encoder, n_rows: int, num_steps: int = 1000, seed: int = 0
) -> pd.DataFrame:
    """Generate ``n_rows`` synthetic rows with DDPM sampling.

    Starts from pure noise x_T ~ N(0, I). For t from T-1 down to 0 it:

    1. turns the model's predicted noise ``eps`` into a predicted clean row
       x0 = (x_t - sqrt(1 - alpha_bar_t) * eps) / sqrt(alpha_bar_t),
    2. clamps x0 to the range the encoder can produce (``NUMERIC_CLAMP``
       for numeric columns, [0, 1] for one-hot columns),
    3. sets x_{t-1} to the DDPM posterior mean given x_t and the clamped x0,
       plus sqrt(beta_t) * z with fresh noise z (zero at the last step).

    Without the clamp, step 3 is the same as the usual
    x_{t-1} = 1/sqrt(alpha_t) * (x_t - beta_t / sqrt(1 - alpha_bar_t) * eps)
    update. The clamp stops errors in ``eps`` from growing: at the first
    step the cosine schedule has beta close to 1, and that update would
    multiply them by about 30.

    ``num_steps`` must match the schedule the model was trained with.
    ``encoder`` is the fitted ``TabularEncoder`` that turns the final array
    back into a DataFrame.
    """
    betas, alphas_cumprod = cosine_schedule(num_steps)
    betas, alphas_cumprod = betas.float(), alphas_cumprod.float()
    alphas = 1 - betas
    alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

    slices = encoder.column_slices()
    n_features = max(s.stop for s in slices.values())
    lower, upper = torch.zeros(n_features), torch.ones(n_features)
    for col in encoder.numeric_columns:
        lower[slices[col]], upper[slices[col]] = -NUMERIC_CLAMP, NUMERIC_CLAMP

    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(n_rows, n_features, generator=gen)

    with torch.no_grad():
        for t in reversed(range(num_steps)):
            predicted_noise = model(x, torch.full((n_rows,), t, dtype=torch.long))
            x0 = (
                x - (1 - alphas_cumprod[t]).sqrt() * predicted_noise
            ) / alphas_cumprod[t].sqrt()
            x0 = torch.clamp(x0, lower, upper)

            # Posterior mean of x_{t-1} given x_t and x0 (DDPM, eq. 7).
            x0_coef = alphas_cumprod_prev[t].sqrt() * betas[t] / (1 - alphas_cumprod[t])
            xt_coef = alphas[t].sqrt() * (1 - alphas_cumprod_prev[t]) / (1 - alphas_cumprod[t])
            x = x0_coef * x0 + xt_coef * x
            if t > 0:
                x = x + betas[t].sqrt() * torch.randn(n_rows, n_features, generator=gen)

    return encoder.inverse_transform(x.double().numpy())
