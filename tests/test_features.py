"""Tests for add_calendar_features in src/features.py (hand-picked dates, no real data)."""

from __future__ import annotations

import pandas as pd
import pytest

from src.features import add_calendar_features, add_holiday_features

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


# ---------------------------------------------------------------------------
# add_holiday_features: small hand-made holidays table
# ---------------------------------------------------------------------------

# (date, type, locale, locale_name, transferred)
HOLIDAYS = [
    ("2017-01-02", "Holiday", "National", "Ecuador", True),    # moved away: NOT a day off
    ("2017-01-03", "Transfer", "National", "Ecuador", False),  # the real day off
    ("2017-02-01", "Holiday", "Local", "Quito", False),        # city holiday
    ("2017-03-01", "Holiday", "Regional", "Pichincha", False), # state holiday
    ("2017-04-01", "Work Day", "National", "Ecuador", False),  # working day, not a day off
    ("2017-05-01", "Event", "National", "Ecuador", False),     # event, not a holiday
]

STORES = {"quito": ("Quito", "Pichincha"), "cuenca": ("Cuenca", "Azuay")}


def _holidays(transferred_as):
    """Same table, with transferred as bool or as the text 'True'/'False'."""
    h = pd.DataFrame(HOLIDAYS, columns=["date", "type", "locale", "locale_name", "transferred"])
    if transferred_as == "text":
        h["transferred"] = h["transferred"].map({True: "True", False: "False"})
    return h


def _stores_frame(dates):
    rows = [(d, name, city, state) for d in dates for name, (city, state) in STORES.items()]
    return pd.DataFrame(rows, columns=["date", "store", "city", "state"])


def _lookup(out, column):
    """{(date, store): value} to make assertions readable."""
    return {(r.date, r.store): getattr(r, column) for r in out.itertuples()}


@pytest.fixture(params=["bool", "text"])
def holidays(request):
    return _holidays(request.param)


def test_transferred_holiday_is_not_a_day_off(holidays):
    out = add_holiday_features(_stores_frame(["2017-01-02"]), holidays)
    assert out["is_holiday"].tolist() == [0, 0]


def test_transfer_row_is_a_day_off(holidays):
    out = add_holiday_features(_stores_frame(["2017-01-03"]), holidays)
    assert out["is_holiday"].tolist() == [1, 1]  # national: every store


def test_local_holiday_only_in_that_city(holidays):
    got = _lookup(add_holiday_features(_stores_frame(["2017-02-01"]), holidays), "is_holiday")
    assert got[("2017-02-01", "quito")] == 1
    assert got[("2017-02-01", "cuenca")] == 0


def test_regional_holiday_only_in_that_state(holidays):
    got = _lookup(add_holiday_features(_stores_frame(["2017-03-01"]), holidays), "is_holiday")
    assert got[("2017-03-01", "quito")] == 1
    assert got[("2017-03-01", "cuenca")] == 0


def test_work_day_is_not_a_day_off(holidays):
    out = add_holiday_features(_stores_frame(["2017-04-01"]), holidays)
    assert out["is_holiday"].tolist() == [0, 0]
    assert out["is_event"].tolist() == [0, 0]


def test_national_event_sets_is_event_not_is_holiday(holidays):
    out = add_holiday_features(_stores_frame(["2017-05-01"]), holidays)
    assert out["is_event"].tolist() == [1, 1]
    assert out["is_holiday"].tolist() == [0, 0]


def test_ordinary_day_is_all_zero(holidays):
    out = add_holiday_features(_stores_frame(["2017-06-10"]), holidays)
    assert out["is_holiday"].tolist() == [0, 0]
    assert out["is_event"].tolist() == [0, 0]


def test_bool_and_text_transferred_give_same_result():
    dates = [h[0] for h in HOLIDAYS] + ["2017-06-10"]
    a = add_holiday_features(_stores_frame(dates), _holidays("bool"))
    b = add_holiday_features(_stores_frame(dates), _holidays("text"))
    pd.testing.assert_frame_equal(a, b)


def test_holiday_features_keep_rows_and_inputs(holidays):
    df = _stores_frame([h[0] for h in HOLIDAYS])
    df_before, h_before = df.copy(deep=True), holidays.copy(deep=True)
    out = add_holiday_features(df, holidays)
    assert len(out) == len(df)
    assert out["store"].tolist() == df["store"].tolist()
    assert set(out["is_holiday"]) <= {0, 1} and set(out["is_event"]) <= {0, 1}
    pd.testing.assert_frame_equal(df, df_before)
    pd.testing.assert_frame_equal(holidays, h_before)
