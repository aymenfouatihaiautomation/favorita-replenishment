"""Features: the information the model gets for each (store, item, day).
Rule: every feature must be known on the Monday of the order."""
import pandas as pd


def add_calendar_features(df):
    df = df.copy()
    date = pd.to_datetime(df["date"])
    df["day_of_week"] = date.dt.day_of_week
    df["day_of_month"] = date.dt.day
    df["month"] = date.dt.month
    df["is_payday"] = ((date.dt.day == 15) | date.dt.is_month_end).astype(int)
    return df