# favorita-replenishment

Sales forecasting + weekly replenishment policy on Kaggle Favorita data.
Business question: how much to order each Monday, per store and item, for the next 9 days
(7-day period + 2-day lead time), and how much stock-out + overstock cost it saves
vs. the current rule.

## Environment

- Python 3.12 in .venv (never the global Python, never 3.14).
- Raw CSVs live outside the repo, in the folder given by FAVORITA_RAW_DIR (E:\favorita-data\raw).
- DuckDB temp_directory: E:\favorita-data\duckdb_tmp. Drive C: is a small SSD: never write large files there.
- Processed data: data/processed/ (git-ignored).

## Hard rules

- Never use transactions.csv (data leakage).
- Every feature must be known on the order date (Monday): lags >= 14 days, no same-day data.
- Never split data randomly: always by date.
- Never commit data/ (Kaggle rules). Never commit .env or secrets.
- Do not modify files outside the current task. No refactoring that was not asked for.

## How we work

- For every task: show a plan and the list of files first, wait for my OK, then code.
- After coding: run the code and pytest, show the real output.
- Before any git commit: show `git status`, wait for my OK.
- Files I write myself (do not write them, only review if I ask):
  src/features.py, src/backtest.py, src/simulate.py
- The project plan is docs/M1_PLAN.md. Work on one step at a time, in order.
