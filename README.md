# Cloud Workload Forecasting – Abhishek Kumar

## Install
python -m venv .venv  (activate)  →  python -m pip install -e ".[dev]"

## Data
Download data-v1.tar.gz from the GitHub release, extract to challenge_data/:
  mkdir challenge_data && tar -xzf data-v1.tar.gz -C challenge_data

## Generate predictions (single command)
python -m cwfc.predict --data challenge_data --output predictions.csv

## Run tests
python -m pytest -q

## Layout
src/cwfc/ (targets, baselines, models, evaluate, predict) · notebooks/ (01 EDA … 06) · reports/final_report.md

## Main decisions
- Targets at the native 5-minute cadence; split 110 train / 24 holdout.
- active_vm_count: constant by dataset construction; same model returns it exactly.
- Validation: expanding-window rolling origin inside the 110 training rows (4 folds, 24-step horizon; 10-fold sensitivity check).
- Final model: mean-reverting AR(1); the selection rule and its deviation are in the report.
- Holdout targets are never read when forecasting.