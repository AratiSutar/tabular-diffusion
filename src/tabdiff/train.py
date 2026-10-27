"""Training the denoiser on a DataFrame, and saving and loading the result."""

import json
from pathlib import Path

import joblib
import pandas as pd
import torch

from tabdiff.diffusion import cosine_schedule
from tabdiff.model import Denoiser, diffusion_loss
from tabdiff.preprocess import TabularEncoder

LOG_EVERY = 500

MODEL_FILE = "model.pt"
ENCODER_FILE = "encoder.joblib"
CONFIG_FILE = "config.json"


def train(
    train_df: pd.DataFrame,
    num_steps: int = 5000,
    batch_size: int = 256,
    lr: float = 1e-3,
    seed: int = 0,
    out_dir: str | Path = "runs/default",
) -> tuple[Denoiser, TabularEncoder]:
    """Train a Denoiser on ``train_df`` and save it into ``out_dir``.

    ``num_steps`` is the number of optimizer steps. Each step uses a batch of
    ``batch_size`` rows drawn at random (with replacement) from ``train_df``.
    The encoder is fitted on ``train_df`` only, so no test data leaks in.

    ``out_dir`` gets the model weights, the fitted encoder and a JSON file
    with the settings and the final loss (the average over the last
    logging window).
    """
    torch.manual_seed(seed)

    encoder = TabularEncoder(seed=seed).fit(train_df)
    data = torch.tensor(encoder.transform(train_df), dtype=torch.float32)
    _, alphas_cumprod = cosine_schedule()

    model = Denoiser(data.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    window_sum, window_count, final_loss = 0.0, 0, float("nan")
    for step in range(1, num_steps + 1):
        batch = data[torch.randint(0, len(data), (batch_size,))]
        loss = diffusion_loss(model, batch, alphas_cumprod)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        window_sum += loss.item()
        window_count += 1
        if step % LOG_EVERY == 0 or step == num_steps:
            final_loss = window_sum / window_count
            if step % LOG_EVERY == 0:
                print(f"step {step:>6}  loss {final_loss:.4f}")
            window_sum, window_count = 0.0, 0

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / MODEL_FILE)
    joblib.dump(encoder, out_dir / ENCODER_FILE)
    config = {
        "num_steps": num_steps,
        "batch_size": batch_size,
        "lr": lr,
        "seed": seed,
        "n_features": data.shape[1],
        "n_train_rows": len(train_df),
        "final_loss": final_loss,
    }
    (out_dir / CONFIG_FILE).write_text(json.dumps(config, indent=2))

    model.eval()
    return model, encoder


def load_run(out_dir: str | Path) -> tuple[Denoiser, TabularEncoder]:
    """Load the model (in eval mode) and encoder saved by ``train``."""
    out_dir = Path(out_dir)
    config = json.loads((out_dir / CONFIG_FILE).read_text())
    model = Denoiser(config["n_features"])
    model.load_state_dict(torch.load(out_dir / MODEL_FILE, weights_only=True))
    model.eval()
    encoder = joblib.load(out_dir / ENCODER_FILE)
    return model, encoder
