"""Tests for src/baseline.py on a small hand-made DataFrame (no real data needed)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.baseline import LAGS_DAYS, seasonal_naive

START = pd.Timestamp("2016-11-01")
N_DAYS = 70  # Nov 1 2016 -> Jan 9 2017, crosses the missing December 25
MISSING_DAY = pd.Timestamp("2016-12-25")
STORES = [1, 2]
ITEM = 10
STORE_OFFSET = {1: 0.0, 2: 1000.0}  # store 2 is always +1000: any mixing is visible


def _unit_sales(store: int, date: pd.Timestamp) -> float:
    day = (date - START).days
    return STORE_OFFSET[store] + (day * 7) % 23 + day / 10  # irregular, all values different


@pytest.fixture()
def sales() -> pd.DataFrame:
    rows = [
        {"store_nbr": s, "item_nbr": ITEM, "date": d, "unit_sales": _unit_sales(s, d)}
        for s in STORES
        for d in pd.date_range(START, periods=N_DAYS)
        if d != MISSING_DAY
    ]
    df = pd.DataFrame(rows)
    # Shuffle the rows and keep a non-trivial index: the result must follow the input order.
    df = df.sample(frac=1, random_state=0)
    df.index = df.index * 3 + 5
    return df


def _expected(store: int, date: pd.Timestamp) -> float:
    """Mean of the sales 14, 21, 28, 35 days earlier, looked up by date. NaN if one is missing."""
    values = []
    for k in LAGS_DAYS:
        d = date - pd.Timedelta(days=k)
        if d < START or d == MISSING_DAY:
            return np.nan
        values.append(_unit_sales(store, d))
    return float(np.mean(values))


def test_one_forecast_per_row_same_index(sales):
    forecast = seasonal_naive(sales)
    assert len(forecast) == len(sales)
    assert forecast.index.equals(sales.index)


def test_forecast_is_mean_of_lags_14_21_28_35(sales):
    forecast = seasonal_naive(sales)
    expected = pd.Series(
        [_expected(s, d) for s, d in zip(sales["store_nbr"], sales["date"])],
        index=sales.index,
    )
    assert expected.notna().sum() > 0  # the test must really compare numbers
    pd.testing.assert_series_equal(forecast, expected, check_names=False)


def test_first_35_days_have_no_forecast(sales):
    forecast = seasonal_naive(sales)
    day_number = (sales["date"] - START).dt.days
    assert forecast[day_number < 35].isna().all()
    # Day 35 is the first one with 4 lags available (Dec 6 here, no gap before it).
    assert forecast[day_number == 35].notna().all()


def test_two_stores_never_mix(sales):
    forecast = seasonal_naive(sales)
    by_key = pd.Series(forecast.values, index=pd.MultiIndex.from_frame(sales[["store_nbr", "date"]]))
    store1 = by_key.xs(1, level="store_nbr")
    store2 = by_key.xs(2, level="store_nbr")
    both = store1.index.intersection(store2.index)
    # Store 2 sales = store 1 sales + 1000, so its forecast must be exactly +1000 (NaN stays NaN).
    pd.testing.assert_series_equal(store2[both], store1[both] + 1000.0, check_names=False)


def test_uses_dates_not_row_positions_around_missing_dec_25(sales):
    forecast = seasonal_naive(sales)
    lookup = pd.Series(forecast.values, index=pd.MultiIndex.from_frame(sales[["store_nbr", "date"]]))
    # Jan 1: all four lag dates exist (Dec 18, 11, 4, Nov 27) -> forecast by date, not shifted by the gap.
    assert lookup[(1, pd.Timestamp("2017-01-01"))] == pytest.approx(
        _expected(1, pd.Timestamp("2017-01-01"))
    )
    # Jan 8: its 14-day lag is the missing Dec 25 -> no forecast (NaN), never a shifted value.
    assert np.isnan(lookup[(1, pd.Timestamp("2017-01-08"))])
