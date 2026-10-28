# tabular-diffusion

![tests](https://github.com/AratiSutar/tabular-diffusion/actions/workflows/tests.yml/badge.svg)

A small diffusion model that creates synthetic (fake but realistic) tabular data. It is trained on the UCI Adult census dataset and runs on a normal laptop CPU.

## Why

Companies often cannot share real data about people because of privacy rules. Synthetic data has the same patterns as the real data, but the rows are new people, so it can be shared and used more safely.

## How it works

1. **Data:** load the Adult dataset, split it into train and test parts with a fixed seed, and fill missing categories with "Missing".
2. **Encode:** turn number columns into a bell-curve shape (quantile transform) and text columns into one-hot vectors.
3. **Train:** a small neural network (MLP) learns to guess the noise that was added to a real row (DDPM with a cosine noise schedule).
4. **Sample:** start from pure random noise and remove it step by step until it becomes a new row. The model's guess is clamped at every step so the numbers stay in a normal range.
5. **Evaluate:** check how similar, how useful and how private the synthetic data is.

## Results (20,000 training steps)

| Check | Real data (baseline) | Synthetic data |
|---|---|---|
| Average column distance (0 = identical) | 0.009 | 0.100 |
| Education consistency | 100% | 45% |
| Classifier AUC on real test data | 0.921 | 0.868 |
| Exact copies of training rows | 0.04% | 0% |

- **Similarity:** the baseline compares real test rows with real train rows. The synthetic data is still about 10 times further from the real data than this baseline.
- **Usefulness:** a classifier trained only on synthetic data reaches 0.868 AUC on real test data, compared with 0.921 when trained on real data (10,000 rows each).
- **Privacy:** no synthetic row is an exact copy of a training row, and synthetic rows are not closer to the training data than real test rows are.

## How to run

    pip install -e ".[dev]"
    pytest

## Limitations

- Results come from one training seed only.
- Tested on one dataset only.
- Small network, trained on CPU only.
- Education and education-num agree in only 45% of synthetic rows.
- Categories are treated as continuous numbers, not with a method made for categories.
