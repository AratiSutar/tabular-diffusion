"""Compare the diffusion model in runs/steps20k with the SMOTE baseline.

Makes 10,000 rows with each, runs the utility and privacy checks, and prints
one table. Run from anywhere with:

    .venv\\Scripts\\python scripts\\compare_smote.py
"""

from pathlib import Path

import pandas as pd

from tabdiff.baselines import smote_sample
from tabdiff.data import load_adult, train_test_split_fixed
from tabdiff.diffusion import sample
from tabdiff.evaluate import privacy, utility
from tabdiff.train import load_run

ROOT = Path(__file__).resolve().parent.parent
N_ROWS = 10000


def main() -> None:
    train, test = train_test_split_fixed(load_adult(ROOT / "data"))
    model, encoder = load_run(ROOT / "runs" / "steps20k")
    fakes = {
        "diffusion 20k": sample(model, encoder, N_ROWS, seed=0),
        "SMOTE": smote_sample(train, N_ROWS, seed=0),
    }

    columns = {}
    for name, fake in fakes.items():
        scores = utility(train, test, fake)
        if scores["message"] is not None:
            print(f"{name}: {scores['message']}")
        distances = privacy(train, test, fake, encoder)
        # The real column is the same in both runs, so it is filled in once.
        columns.setdefault(
            "real",
            {
                "AUC": scores["real_auc"],
                "accuracy": scores["real_accuracy"],
                **distances.loc["test"].to_dict(),
            },
        )
        columns[name] = {
            "AUC": scores["fake_auc"],
            "accuracy": scores["fake_accuracy"],
            **distances.loc["fake"].to_dict(),
        }

    print(pd.DataFrame(columns).to_string(float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
