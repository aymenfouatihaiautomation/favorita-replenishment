"""Tests for add_calendar_features in src/features.py (hand-picked dates, no real data)."""

from __future__ import annotations

import pandas as pd
import pytest

from src.features import add_calendar_features

# (date, day_of_week, day_of_month, month, is_payday)
CASES = [
    ("2017-08-14", 0, 14, 8, 0),  # Monday, day before payday
    ("2017-08-15", 1, 15, 8, 1),  # 15th
    ("2017-08-16", 2, 16, 8, 0),  # day after the 15th
    ("2017-08-30", 2, 30, 8, 0),  # day before month end (31 days)
    ("2017-08-31", 3, 31, 8, 1),  # last day, 31-day month
    ("2017-04-30", 6, 30, 4, 1),  # last day, 30-day month (Sunday)
    ("2017-04-29", 5, 29, 4, 0),
    ("2017-02-27", 0, 27, 2, 0),  # not the last day of Feb 2017
    ("2017-02-28", 1, 28, 2, 1),  # last day, non-leap February
    ("2016-02-28", 6, 28, 2, 0),  # NOT the last day in a leap year
    ("2016-02-29", 0, 29, 2, 1),  # last day, leap February
    ("2016-12-31", 5, 31, 12, 1),  # year end
    ("2017-01-01", 6, 1, 1, 0),
]


def _frame(date_values):
    return pd.DataFrame({"date": date_values, "store_nbr": range(len(date_values))})


@pytest.fixture(params=["datetime", "string"])
def df(request):
    dates = [c[0] for c in CASES]
    if request.param == "datetime":
        dates = pd.to_datetime(dates)
    return _frame(dates)


def test_day_of_week_monday_is_zero(df):
    out = add_calendar_features(df)
    assert out["day_of_week"].tolist() == [c[1] for c in CASES]
    assert out.loc[0, "day_of_week"] == 0  # 2017-08-14 is a Monday


def test_day_of_month_and_month(df):
    out = add_calendar_features(df)
    assert out["day_of_month"].tolist() == [c[2] for c in CASES]
    assert out["month"].tolist() == [c[3] for c in CASES]


def test_is_payday_15th_and_last_day_of_month(df):
    out = add_calendar_features(df)
    assert out["is_payday"].tolist() == [c[4] for c in CASES]
    assert set(out["is_payday"].unique()) <= {0, 1}


def test_input_is_not_modified(df):
    before = df.copy(deep=True)
    add_calendar_features(df)
    pd.testing.assert_frame_equal(df, before)
    assert list(df.columns) == ["date", "store_nbr"]


def test_one_output_row_per_input_row(df):
    out = add_calendar_features(df)
    assert len(out) == len(df)
    assert out["store_nbr"].tolist() == df["store_nbr"].tolist()  # same order


def test_duplicate_dates_keep_one_row_each():
    df = _frame(["2017-08-15"] * 3)
    out = add_calendar_features(df)
    assert len(out) == 3
    assert out["is_payday"].tolist() == [1, 1, 1]
