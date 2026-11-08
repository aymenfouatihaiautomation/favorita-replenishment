"""Seasonal naive baseline: 'ordering as usual'."""
import pandas as pd

LAGS_DAYS = [14, 21, 28, 35]
KEYS = ["store_nbr", "item_nbr"]


def seasonal_naive(sales: pd.DataFrame) -> pd.Series:
    """Forecast for each row = mean of the sales of the same store and item
    14, 21, 28 and 35 days earlier (same weekday, known on the order date)."""
    df = sales[KEYS + ["date", "unit_sales"]].copy()
    df["date"] = pd.to_datetime(df["date"])
    lag_columns = []

    for k in LAGS_DAYS:
        past = df[KEYS + ["date", "unit_sales"]].copy()
        # The sales of day d become the "lag k" of day d + k.
        past["date"] = past["date"] + pd.Timedelta(days=k)
        past = past.rename(columns={"unit_sales": f"lag_{k}"})
        df = df.merge(past, on=KEYS + ["date"], how="left")
        lag_columns.append(f"lag_{k}")

    # Mean of the 4 past sales. If one is missing, no forecast (NaN).
    forecast = df[lag_columns].mean(axis=1, skipna=False)
    forecast.index = sales.index
    return forecast.rename("baseline_forecast")