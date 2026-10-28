# tabular-diffusion

![tests](https://github.com/AratiSutar/tabular-diffusion/actions/workflows/tests.yml/badge.svg)

[One sentence: what does this project do?
Example: A small diffusion model that creates synthetic (fake but realistic) tabular data, trained on the UCI Adult census dataset. Runs on a normal CPU.]

## Why
[1–2 sentences: why is synthetic data useful? Hint: share data without sharing real people.]

## How it works
1. [Data: load Adult, split into train and test, fill missing values with "Missing".]
2. [Encode: numbers → bell curve, categories → one-hot.]
3. [Train: a small neural network learns to guess the noise.]
4. [Sample: start from pure noise and remove it step by step.]

## Results (20,000 training steps)

| Check | Real data | Synthetic data |
|---|---|---|
| Average column distance (0 = identical) | 0.009 (baseline) | 0.100 |
| Education consistency | 100% | 45% |
| Classifier AUC on real test data | 0.921 | 0.868 |
| Exact copies of training rows | 0.04% | 0% |

[1–2 sentences: what do these numbers mean? Hint: useful, no copying, but still far from real.]

## How to run
    pip install -e ".[dev]"
    pytest

## Limitations
- [one training seed only]
- [one dataset only]
- [small network, CPU only]
- [education consistency only 45%]
- [categories are treated as continuous numbers]
