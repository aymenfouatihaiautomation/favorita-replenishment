"""Checks on data/processed/sales.parquet (run `python -m src.prepare_data` first)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

SALES_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "sales.parquet"
START, END = pd.Timestamp("2015-01-01"), pd.Timestamp("2017-08-15")


@pytest.fixture(scope="module")
def sales() -> pd.DataFrame:
    if not SALES_PATH.is_file():
        pytest.skip(f"{SALES_PATH} not found: run `python -m src.prepare_data` first")
    df = pd.read_parquet(SALES_PATH)
    df["date"] = pd.to_datetime(df["date"])
    return df


def test_no_duplicate_keys(sales):
    assert not sales.duplicated(["date", "store_nbr", "item_nbr"]).any()


def test_no_negative_sales(sales):
    assert (sales["unit_sales"] >= 0).all()
    assert sales["unit_sales"].notna().all()


def test_store_and_item_counts(sales):
    assert sales["store_nbr"].nunique() == 10
    assert sales["item_nbr"].nunique() == 200
    # 2 stores per type A-E
    per_type = sales.drop_duplicates("store_nbr").groupby("store_type").size()
    assert per_type.to_dict() == {t: 2 for t in "ABCDE"}


def test_fifty_items_per_family(sales):
    per_family = sales.drop_duplicates("item_nbr").groupby("family").size()
    assert per_family.to_dict() == {
        "BEVERAGES": 50, "BREAD/BAKERY": 50, "DAIRY": 50, "GROCERY I": 50,
    }
    perishable = sales.drop_duplicates("item_nbr").groupby("family")["perishable"].first()
    assert perishable.to_dict() == {
        "BEVERAGES": 0, "BREAD/BAKERY": 1, "DAIRY": 1, "GROCERY I": 0,
    }


def test_no_december_25(sales):
    assert not ((sales["date"].dt.month == 12) & (sales["date"].dt.day == 25)).any()


def test_expected_date_range(sales):
    assert sales["date"].min() == START
    assert sales["date"].max() == END


def test_no_missing_date_for_any_series(sales):
    all_days = pd.date_range(START, END, freq="D")
    expected = all_days[~((all_days.month == 12) & (all_days.day == 25))]
    sizes = sales.groupby(["store_nbr", "item_nbr"])["date"].agg(["size", "nunique"])
    assert len(sizes) == 10 * 200
    assert (sizes["size"] == len(expected)).all()
    assert (sizes["nunique"] == len(expected)).all()
