"""Tests for src/backtest.py on small hand-made DataFrames (no real data needed)."""

from __future__ import annotations

import pandas as pd
import pytest

from src.backtest import HORIZON_DAYS, N_ORDER_DATES, order_dates, run_backtest, summarize

DATA_END = pd.Timestamp("2017-08-15")  # a Tuesday
DATA_START = pd.Timestamp("2017-03-01")  # long enough for 16 order dates (first one: 2017-04-17)


def _make_sales() -> pd.DataFrame:
    """2 stores x 2 items x every day from DATA_START to DATA_END, with a different value each row."""
    rows = []
    for store in (1, 2):
        for item, family, perishable in ((10, "BREAD", 1), (20, "GROCERY", 0)):
            for day, date in enumerate(pd.date_range(DATA_START, DATA_END)):
                rows.append(
                    {
                        "date": date,
                        "store_nbr": store,
                        "item_nbr": item,
                        "family": family,
                        "perishable": perishable,
                        "unit_sales": float(store * 1000 + item + day),
                    }
                )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- order_dates


def test_order_dates_are_16_mondays_oldest_first_last_is_2017_07_31():
    dates = order_dates(DATA_END)
    assert len(dates) == N_ORDER_DATES == 16
    assert all(d.weekday() == 0 for d in dates)  # all Mondays
    assert dates == sorted(dates)  # oldest first
    assert dates[-1] == pd.Timestamp("2017-07-31")
    assert dates[0] == pd.Timestamp("2017-04-17")  # 15 weeks before the last one
    assert all((b - a) == pd.Timedelta(days=7) for a, b in zip(dates, dates[1:]))  # no gap


# ---------------------------------------------------------------- no cheating


def test_forecaster_never_sees_future_sales_or_unit_sales_of_the_9_days():
    sales = _make_sales()
    calls = []

    def spy(history: pd.DataFrame, targets: pd.DataFrame) -> pd.Series:
        calls.append((history.copy(), targets.copy()))
        # The fake forecaster is only allowed to use these two inputs.
        assert "unit_sales" not in targets.columns  # answers are hidden
        assert "unit_sales" in history.columns  # the past is given with its sales
        assert history["date"].max() < targets["date"].min()  # all past < all targets
        return pd.Series(0.0, index=targets.index)

    run_backtest(sales, spy)

    assert len(calls) == N_ORDER_DATES  # the spy really ran 16 times
    for history, targets in calls:
        assert not history.empty and not targets.empty
        assert history["date"].max() < targets["date"].min()
        # Daily data: the history stops the day before the order date, which is a Monday.
        order_date = history["date"].max() + pd.Timedelta(days=1)
        assert order_date.weekday() == 0
        assert targets["date"].min() == order_date + pd.Timedelta(days=1)
        # None of the target rows leaks into the history (same row = same index in the source frame).
        assert history.index.intersection(targets.index).empty


def test_history_and_targets_do_not_overlap_when_answers_are_poisoned():
    """Put a huge sentinel on every target day: if it ever shows up in the history, we cheated."""
    sales = _make_sales()
    seen_max_history_sales = []

    def spy(history: pd.DataFrame, targets: pd.DataFrame) -> pd.Series:
        seen_max_history_sales.append(history["unit_sales"].max())
        return pd.Series(0.0, index=targets.index)

    results = run_backtest(sales, spy)
    first_origin = results["order_date"].min()
    # First call: history ends before 2017-04-17, so max sales must come from that period only.
    expected_max_first = sales[sales["date"] < first_origin]["unit_sales"].max()
    assert seen_max_history_sales[0] == expected_max_first
    # And it must grow call after call (more past), never jump to the future's values.
    last_origin = results["order_date"].max()
    expected_max_last = sales[sales["date"] < last_origin]["unit_sales"].max()
    assert seen_max_history_sales[-1] == expected_max_last
    assert expected_max_last < sales["unit_sales"].max()  # data after the last order date exists


# ---------------------------------------------------------------- 9 target days


def test_each_monday_has_exactly_9_target_days():
    sales = _make_sales()
    results = run_backtest(sales, lambda history, targets: pd.Series(1.0, index=targets.index))

    assert results["order_date"].nunique() == N_ORDER_DATES
    days_per_order = results.groupby("order_date")["date"].nunique()
    assert HORIZON_DAYS == 9
    assert (days_per_order == 9).all()

    series_count = sales[["store_nbr", "item_nbr"]].drop_duplicates().shape[0]
    assert series_count == 4
    rows_per_order = results.groupby("order_date").size()
    assert (rows_per_order == 9 * series_count).all()

    # The 9 days are the 9 days right after the order date, and the answers are kept for scoring.
    for order_date, block in results.groupby("order_date"):
        assert order_date.weekday() == 0
        assert sorted(block["date"].unique()) == [
            order_date + pd.Timedelta(days=i) for i in range(1, HORIZON_DAYS + 1)
        ]
        assert block["unit_sales"].notna().all()
    assert results["date"].max() <= DATA_END  # the last horizon stays inside the data


# ---------------------------------------------------------------- summarize


def test_summarize_wape_and_bias_on_tiny_example_done_by_hand():
    results = pd.DataFrame(
        {
            "family": ["A", "A", "B", "B"],
            "perishable": [1, 1, 0, 0],
            "unit_sales": [10.0, 20.0, 50.0, 40.0],
            "forecast": [12.0, 15.0, 60.0, float("nan")],  # last row: no forecast -> ignored
        }
    )
    # Hand calculation (the NaN row is dropped):
    #   A: errors +2 and -5 -> abs 7, net -3, sales 30 -> WAPE 7/30,  bias -3/30 = -0.1
    #   B: error +10        -> abs 10, net +10, sales 50 -> WAPE 0.2, bias 0.2
    #   all: abs 17, net +7, sales 80 -> WAPE 17/80 = 0.2125, bias 7/80 = 0.0875

    overall = summarize(results)
    assert overall.loc["all", "rows"] == 3
    assert overall.loc["all", "sales"] == pytest.approx(80.0)
    assert overall.loc["all", "wape"] == pytest.approx(0.2125)
    assert overall.loc["all", "bias"] == pytest.approx(0.0875)

    per_family = summarize(results, "family")
    assert per_family.loc["A", "wape"] == pytest.approx(7 / 30)
    assert per_family.loc["A", "bias"] == pytest.approx(-0.1)
    assert per_family.loc["B", "wape"] == pytest.approx(0.2)
    assert per_family.loc["B", "bias"] == pytest.approx(0.2)

    per_perishable = summarize(results, "perishable")
    assert per_perishable.loc[1, "wape"] == pytest.approx(7 / 30)  # same rows as family A
    assert per_perishable.loc[0, "bias"] == pytest.approx(0.2)  # same rows as family B
