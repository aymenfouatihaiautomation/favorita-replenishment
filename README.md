# favorita-replenishment

## Setup

Python 3.12 in a local virtual environment (never the global Python).

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

The raw Kaggle CSVs are **not** in the repo (Kaggle rules forbid redistributing them).
Point the code to them with the `FAVORITA_RAW_DIR` environment variable:

```powershell
$env:FAVORITA_RAW_DIR = "E:\favorita-data\raw"
```

Expected files in that folder: `train.csv`, `items.csv`, `stores.csv`, `holidays_events.csv`, `oil.csv`.

Optional: DuckDB spills large intermediate results to disk. Point it to a drive with
free space with `FAVORITA_DUCKDB_TMP` (created if missing). If unset, DuckDB's default is used.

```powershell
$env:FAVORITA_DUCKDB_TMP = "E:\favorita-data\duckdb_tmp"
```

### Data preparation

```powershell
python -m src.prepare_data
python -m pytest
```

Outputs go to `data/processed/` (git-ignored): `sales.parquet`, `holidays_events.parquet`, `oil.parquet`.

## Data notes

- **Selection.** 10 stores (the 2 largest per store type A-E among stores active over the whole
  period) and 200 items: 50 each of DAIRY, BREAD/BAKERY (perishable) and GROCERY I, BEVERAGES
  (non-perishable). Items are ranked per family by number of selling days, then total
  `unit_sales`, then `item_nbr`.
- **Active-series filter.** A store must sell on or before 2015-01-15 and on or after 2017-08-01;
  an item must have its first sale on or before 2015-01-31 and its last sale on or after
  2017-07-15 (in the selected stores). Activity means `unit_sales > 0`. This favors regularly
  sold items: the series are dense, with few long gaps, which is not representative of
  slow movers or recently launched products.
- **PRODUCE is excluded on purpose.** Almost all PRODUCE items first sell *after* 2015-01-31 in
  the selected stores (293 of 306 start during 2015, median first sale 2015-06-01; only 1 meets the
  first-sale rule), so they fail the active-series filter, even though 302 of 306 still sell
  after 2017-07-15. Keeping them would mean relaxing the filter for that family only.
- **Zero is not always zero demand.** Missing `train.csv` rows are filled with 0 (full grid),
  and negative sales (returns) are clipped to 0 (count logged). December 25 is dropped, not zero-filled
  (all stores are closed).
- `transactions.csv` is never used (data leakage).
