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

## Results

### Usefulness and privacy (10,000 rows each)

Compared with a simple SMOTE-style baseline, which makes new rows between two real neighbours.

| Check | Real data | Diffusion (20k steps) | SMOTE |
|---|---|---|---|
| Classifier AUC on real test data | 0.921 | 0.868 | 0.907 |
| Classifier accuracy on real test data | 0.868 | 0.808 | 0.854 |
| Exact copies of training rows | 0.15% | 0% | 0.13% |
| Median distance to closest training row | 0.850 | 1.460 | 0.211 |

For "Real data", the classifier is trained on real rows, and distances are measured from real test rows that the model never saw.

**The trade-off:** SMOTE is more useful (AUC 0.907) because its rows stay very close to real people. That same closeness is a privacy risk: its rows are about 4 times closer to a training row than unseen real rows are. Diffusion rows are less useful, but they are not close to any real person.

### Similarity (diffusion, 20k steps)

| Check | Real data (baseline) | Diffusion |
|---|---|---|
| Average column distance (0 = identical) | 0.009 | 0.100 |
| Education consistency | 100% | 45% |

The baseline compares real test rows with real train rows. The diffusion data is still about 10 times further from the real data than this baseline.

## How to run

    pip install -e ".[dev]"
    pytest

To repeat the SMOTE comparison:

    python scripts/compare_smote.py

## Limitations

- Results come from one training seed only.
- Tested on one dataset only.
- Small network, trained on CPU only.
- Education and education-num agree in only 45% of synthetic rows.
- Categories are treated as continuous numbers, not with a method made for categories.
