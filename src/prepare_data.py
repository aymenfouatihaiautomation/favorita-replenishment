"""Data preparation: raw Favorita CSVs -> data/processed/sales.parquet.

train.csv is ~4.7 GB, so everything is done with DuckDB (never loaded whole in pandas).

Usage:
    FAVORITA_RAW_DIR      folder with the Kaggle CSVs (required)
    FAVORITA_DUCKDB_TMP   DuckDB spill folder (optional)
    python -m src.prepare_data
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import duckdb

log = logging.getLogger("prepare_data")

# --- Configuration -----------------------------------------------------------
START_DATE = "2015-01-01"
END_DATE = "2017-08-15"

# Active-series rules (activity = at least one row with unit_sales > 0).
STORE_FIRST_SALE_BY = "2015-01-15"
STORE_LAST_SALE_FROM = "2017-08-01"
ITEM_FIRST_SALE_BY = "2015-01-31"
ITEM_LAST_SALE_FROM = "2017-07-15"

STORE_TYPES = ["A", "B", "C", "D", "E"]
STORES_PER_TYPE = 2
# (family, perishable flag, number of items). PRODUCE is deliberately left out: almost all
# PRODUCE items fail the active-series filter (see README, "Data notes").
ITEM_GROUPS = [
    ("DAIRY", 1, 50),
    ("BREAD/BAKERY", 1, 50),
    ("GROCERY I", 0, 50),
    ("BEVERAGES", 0, 50),
]

DUCKDB_TMP_ENV = "FAVORITA_DUCKDB_TMP"  # optional; DuckDB's default temp dir if unset
RAW_DIR_ENV = "FAVORITA_RAW_DIR"
PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"


def _sql_list(values: list[str]) -> str:
    return ", ".join("'" + v.replace("'", "''") + "'" for v in values)


def _raw_dir() -> Path:
    value = os.environ.get(RAW_DIR_ENV)
    if not value:
        raise SystemExit(f"Environment variable {RAW_DIR_ENV} is not set (folder with the Kaggle CSVs).")
    path = Path(value)
    missing = [
        f for f in ("train.csv", "items.csv", "stores.csv", "holidays_events.csv", "oil.csv")
        if not (path / f).is_file()
    ]
    if missing:
        raise SystemExit(f"Missing files in {path}: {', '.join(missing)}")
    return path


def _sql_path(path: Path) -> str:
    return path.as_posix().replace("'", "''")


def _connect() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    tmp = os.environ.get(DUCKDB_TMP_ENV)
    if tmp:
        tmp_dir = Path(tmp)
        tmp_dir.mkdir(parents=True, exist_ok=True)
        con.execute(f"SET temp_directory = '{_sql_path(tmp_dir)}'")
    else:
        log.info("%s not set: using DuckDB's default temp directory", DUCKDB_TMP_ENV)
    con.execute("SET enable_progress_bar = false")
    return con


def select_stores(con: duckdb.DuckDBPyConnection, raw: Path) -> list[int]:
    """Active stores, then the top STORES_PER_TYPE per store type by total unit_sales."""
    train = _sql_path(raw / "train.csv")
    con.execute(f"""
        CREATE TEMP TABLE store_stats AS
        SELECT store_nbr,
               MIN(date) FILTER (WHERE unit_sales > 0) AS first_sale,
               MAX(date) FILTER (WHERE unit_sales > 0) AS last_sale,
               SUM(unit_sales) AS total_sales
        FROM read_csv('{train}', header = true,
                      columns = {{'id': 'BIGINT', 'date': 'DATE', 'store_nbr': 'INTEGER',
                                  'item_nbr': 'INTEGER', 'unit_sales': 'DOUBLE',
                                  'onpromotion': 'BOOLEAN'}})
        WHERE date BETWEEN DATE '{START_DATE}' AND DATE '{END_DATE}'
          AND NOT (month(date) = 12 AND day(date) = 25)
        GROUP BY store_nbr
    """)
    n_all = con.execute("SELECT COUNT(*) FROM store_stats").fetchone()[0]
    con.execute(f"""
        CREATE TEMP TABLE active_stores AS
        SELECT * FROM store_stats
        WHERE first_sale <= DATE '{STORE_FIRST_SALE_BY}' AND last_sale >= DATE '{STORE_LAST_SALE_FROM}'
    """)
    n_active = con.execute("SELECT COUNT(*) FROM active_stores").fetchone()[0]
    log.info("Stores with sales in window: %d | active: %d | excluded (inactive): %d",
             n_all, n_active, n_all - n_active)

    stores_csv = _sql_path(raw / "stores.csv")
    rows = con.execute(f"""
        SELECT store_nbr, type, total_sales FROM (
            SELECT a.store_nbr, s.type, a.total_sales,
                   ROW_NUMBER() OVER (PARTITION BY s.type
                                      ORDER BY a.total_sales DESC, a.store_nbr) AS rk
            FROM active_stores a
            JOIN read_csv('{stores_csv}', header = true) s USING (store_nbr)
            WHERE s.type IN ({_sql_list(STORE_TYPES)})
        ) WHERE rk <= {STORES_PER_TYPE}
        ORDER BY type, total_sales DESC
    """).fetchall()
    for store, stype, total in rows:
        log.info("  selected store %d (type %s, total unit_sales %.0f)", store, stype, total)
    stores = [r[0] for r in rows]
    expected = STORES_PER_TYPE * len(STORE_TYPES)
    if len(stores) != expected:
        raise SystemExit(f"Expected {expected} stores, got {len(stores)}")
    return stores


def select_items(con: duckdb.DuckDBPyConnection, raw: Path, stores: list[int]) -> list[int]:
    """Active items in the selected stores, then top-N per family by selling days
    (ties: total unit_sales descending, then item_nbr)."""
    train = _sql_path(raw / "train.csv")
    items_csv = _sql_path(raw / "items.csv")
    con.execute(f"""
        CREATE TEMP TABLE item_stats AS
        SELECT item_nbr,
               MIN(date) FILTER (WHERE unit_sales > 0) AS first_sale,
               MAX(date) FILTER (WHERE unit_sales > 0) AS last_sale,
               COUNT(DISTINCT date) FILTER (WHERE unit_sales > 0) AS selling_days,
               SUM(unit_sales) AS total_sales
        FROM read_csv('{train}', header = true,
                      columns = {{'id': 'BIGINT', 'date': 'DATE', 'store_nbr': 'INTEGER',
                                  'item_nbr': 'INTEGER', 'unit_sales': 'DOUBLE',
                                  'onpromotion': 'BOOLEAN'}})
        WHERE date BETWEEN DATE '{START_DATE}' AND DATE '{END_DATE}'
          AND NOT (month(date) = 12 AND day(date) = 25)
          AND store_nbr IN ({_sql_list([str(s) for s in stores])})
        GROUP BY item_nbr
    """)
    con.execute(f"""
        CREATE TEMP TABLE candidates AS
        SELECT i.item_nbr, i.family, i.perishable, s.first_sale, s.last_sale, s.selling_days,
               s.total_sales,
               (s.first_sale <= DATE '{ITEM_FIRST_SALE_BY}'
                AND s.last_sale >= DATE '{ITEM_LAST_SALE_FROM}') AS active
        FROM read_csv('{items_csv}', header = true) i
        JOIN item_stats s USING (item_nbr)
        WHERE s.first_sale IS NOT NULL
    """)

    selected: list[int] = []
    for family, flag, n in ITEM_GROUPS:
        where = f"perishable = {flag} AND family = {_sql_list([family])}"
        n_all, n_active = con.execute(
            f"SELECT COUNT(*), COUNT(*) FILTER (WHERE active) FROM candidates WHERE {where}"
        ).fetchone()
        log.info("Items %s (perishable=%d): %d sold in the 10 stores | active: %d | "
                 "excluded (inactive): %d", family, flag, n_all, n_active, n_all - n_active)
        ids = [r[0] for r in con.execute(f"""
            SELECT item_nbr FROM candidates
            WHERE active AND {where}
            ORDER BY selling_days DESC, total_sales DESC, item_nbr
            LIMIT {n}
        """).fetchall()]
        if len(ids) != n:
            raise SystemExit(f"Family {family}: expected {n} items, only {len(ids)} eligible")
        selected += ids
    return selected


def build_sales(con: duckdb.DuckDBPyConnection, raw: Path, stores: list[int],
                items: list[int]) -> int:
    """Create the full store x item x date grid in table `sales`. Returns clipped row count."""
    train = _sql_path(raw / "train.csv")
    stores_csv = _sql_path(raw / "stores.csv")
    items_csv = _sql_path(raw / "items.csv")

    # Sparse sales for the selected series only (small).
    con.execute(f"""
        CREATE TEMP TABLE sparse AS
        SELECT date, store_nbr, item_nbr, unit_sales, onpromotion
        FROM read_csv('{train}', header = true,
                      columns = {{'id': 'BIGINT', 'date': 'DATE', 'store_nbr': 'INTEGER',
                                  'item_nbr': 'INTEGER', 'unit_sales': 'DOUBLE',
                                  'onpromotion': 'BOOLEAN'}})
        WHERE date BETWEEN DATE '{START_DATE}' AND DATE '{END_DATE}'
          AND NOT (month(date) = 12 AND day(date) = 25)
          AND store_nbr IN ({_sql_list([str(s) for s in stores])})
          AND item_nbr IN ({_sql_list([str(i) for i in items])})
    """)
    n_dup = con.execute(
        "SELECT COUNT(*) - COUNT(DISTINCT (date, store_nbr, item_nbr)) FROM sparse"
    ).fetchone()[0]
    if n_dup:
        raise SystemExit(f"{n_dup} duplicate (date, store, item) rows in train.csv")
    n_clipped = con.execute("SELECT COUNT(*) FROM sparse WHERE unit_sales < 0").fetchone()[0]

    con.execute("CREATE TEMP TABLE sel_stores AS SELECT UNNEST([" + ",".join(map(str, stores)) + "]) AS store_nbr")
    con.execute("CREATE TEMP TABLE sel_items AS SELECT UNNEST([" + ",".join(map(str, items)) + "]) AS item_nbr")
    con.execute(f"""
        CREATE TEMP TABLE sales AS
        SELECT g.date, g.store_nbr, g.item_nbr,
               GREATEST(COALESCE(s.unit_sales, 0.0), 0.0) AS unit_sales,
               COALESCE(s.onpromotion, FALSE) AS onpromotion,
               st.city, st.state, st.type AS store_type, st.cluster,
               it.family, it."class" AS item_class, it.perishable
        FROM (
            SELECT CAST(d AS DATE) AS date, store_nbr, item_nbr
            FROM generate_series(DATE '{START_DATE}', DATE '{END_DATE}', INTERVAL 1 DAY) AS t(d)
            CROSS JOIN sel_stores CROSS JOIN sel_items
            WHERE NOT (month(d) = 12 AND day(d) = 25)
        ) g
        LEFT JOIN sparse s USING (date, store_nbr, item_nbr)
        JOIN read_csv('{stores_csv}', header = true) st USING (store_nbr)
        JOIN read_csv('{items_csv}', header = true) it USING (item_nbr)
        ORDER BY g.store_nbr, g.item_nbr, g.date
    """)
    return n_clipped


def export_auxiliary(con: duckdb.DuckDBPyConnection, raw: Path) -> None:
    """Parquet copies of holidays_events and oil (transactions.csv is deliberately not used)."""
    for name in ("holidays_events", "oil"):
        src = _sql_path(raw / f"{name}.csv")
        dst = _sql_path(PROCESSED_DIR / f"{name}.parquet")
        con.execute(f"COPY (SELECT * FROM read_csv('{src}', header = true)) TO '{dst}' (FORMAT PARQUET)")


def print_summary(con: duckdb.DuckDBPyConnection, n_clipped: int) -> None:
    rows, dmin, dmax, ns, ni, zero, promo = con.execute("""
        SELECT COUNT(*), MIN(date), MAX(date), COUNT(DISTINCT store_nbr), COUNT(DISTINCT item_nbr),
               AVG((unit_sales = 0)::INT), AVG(onpromotion::INT)
        FROM sales
    """).fetchone()
    print("=" * 60)
    print("SUMMARY - data/processed/sales.parquet")
    print("=" * 60)
    print(f"rows                  : {rows:,}")
    print(f"date range            : {dmin} -> {dmax}")
    print(f"n_stores              : {ns}")
    print(f"n_items               : {ni}")
    print(f"share of zero sales   : {zero:.2%}")
    print(f"share of promo rows   : {promo:.2%}")
    print(f"clipped (negative) rows: {n_clipped:,}")
    print("items per family:")
    for fam, n in con.execute(
        "SELECT family, COUNT(DISTINCT item_nbr) FROM sales GROUP BY family ORDER BY family"
    ).fetchall():
        print(f"  {fam:<14}: {n}")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    raw = _raw_dir()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    con = _connect()

    log.info("Raw dir: %s", raw)
    stores = select_stores(con, raw)
    items = select_items(con, raw, stores)
    n_clipped = build_sales(con, raw, stores, items)
    log.info("Clipped %d negative unit_sales rows to 0", n_clipped)

    out = _sql_path(PROCESSED_DIR / "sales.parquet")
    con.execute(f"COPY sales TO '{out}' (FORMAT PARQUET)")
    export_auxiliary(con, raw)
    log.info("Wrote parquet files to %s", PROCESSED_DIR)
    print_summary(con, n_clipped)


if __name__ == "__main__":
    sys.exit(main())
