# tabdiff

A small diffusion model for synthetic tabular data, trained on the UCI Adult dataset. CPU only.

## Layout

- `src/` layout: code in `src/tabdiff/`, tests in `tests/`.

## Commands

- Always use `.venv\Scripts\python`, never the system Python.
- Run tests: `.venv\Scripts\python -m pytest`

## Rules

- Tests must never download data. Use small fake DataFrames or `tmp_path`.
- Run pytest before suggesting a commit.
- Do one task at a time. Do not write code the user did not ask for.
- Explain changes in simple English.

## Data

- `data/` is the local download cache and is ignored by git.
- Missing values in categorical columns become `"Missing"` (see `load_adult` in `src/tabdiff/data.py`).
