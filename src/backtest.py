"""Rolling backtest: 16 Monday order dates, 9-day horizon, no peeking at the future."""
from collections.abc import Callable
from pathlib import Path

import pandas as pd

from src.baseline import seasonal_naive

HORIZON_DAYS = 9
N_ORDER_DATES = 16
SALES_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "sales.parquet"

# A forecaster receives the past (with sales) and the days to forecast (without sales).
Forecaster = Callable[[pd.DataFrame, pd.DataFrame], pd.Series]


def order_dates(last_date: pd.Timestamp, n: int = N_ORDER_DATES) -> list[pd.Timestamp]:
    """The last n Mondays whose full 9-day horizon is inside the data, oldest first."""
    last_possible = last_date - pd.Timedelta(days=HORIZON_DAYS)
    last_monday = last_possible - pd.Timedelta(days=last_possible.weekday())
    return [last_monday - pd.Timedelta(weeks=i) for i in range(n - 1, -1, -1)]


def run_backtest(sales: pd.DataFrame, forecaster: Forecaster) -> pd.DataFrame:
    """For each order date: give the forecaster only the past, hide the answers, score."""
    sales = sales.copy()
    sales["date"] = pd.to_datetime(sales["date"])
    blocks = []
    for origin in order_dates(sales["date"].max()):
        history = sales[sales["date"] < origin]
        end = origin + pd.Timedelta(days=HORIZON_DAYS)
        targets = sales[(sales["date"] > origin) & (sales["date"] <= end)]
        hidden = targets.drop(columns="unit_sales")  # the forecaster never sees the answers
        forecast = forecaster(history, hidden)

        block = targets[["date", "store_nbr", "item_nbr", "family", "perishable", "unit_sales"]].copy()
        block["forecast"] = forecast
        block["order_date"] = origin
        blocks.append(block)
    return pd.concat(blocks, ignore_index=True)


def baseline_forecaster(history: pd.DataFrame, targets: pd.DataFrame) -> pd.Series:
    """'Ordering as usual': seasonal naive, computed from the past only."""
    combined = pd.concat([history, targets.assign(unit_sales=float("nan"))])
    return seasonal_naive(combined).loc[targets.index]


def summarize(results: pd.DataFrame, by: str | None = None) -> pd.DataFrame:
    """WAPE and bias, overall (by=None) or per group."""
    scored = results.dropna(subset=["forecast"]).copy()
    scored["error"] = scored["forecast"] - scored["unit_sales"]
    scored["abs_error"] = scored["error"].abs()
    if by is None:
        scored["scope"] = "all"
        by = "scope"
    table = scored.groupby(by).agg(
        rows=("unit_sales", "size"),
        sales=("unit_sales", "sum"),
        abs_error=("abs_error", "sum"),
        error=("error", "sum"),
    )
    table["wape"] = table["abs_error"] / table["sales"]
    table["bias"] = table["error"] / table["sales"]
    return table[["rows", "sales", "wape", "bias"]]


def main() -> None:
    sales = pd.read_parquet(SALES_PATH)
    results = run_backtest(sales, baseline_forecaster)
    print("Order dates:", results["order_date"].min().date(), "->", results["order_date"].max().date())
    print("\nOverall\n", summarize(results).round(3))
    print("\nBy family\n", summarize(results, "family").round(3))
    print("\nPerishable (1) vs not (0)\n", summarize(results, "perishable").round(3))
    results.to_parquet(SALES_PATH.parent / "backtest_baseline.parquet")


if __name__ == "__main__":
    main()