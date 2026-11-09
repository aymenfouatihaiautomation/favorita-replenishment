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

DAY_OFF_TYPES = ["Holiday", "Transfer", "Additional", "Bridge"]


def add_holiday_features(df, holidays):
    df = df.copy()
    date = pd.to_datetime(df["date"])

    h = holidays.copy()
    h["date"] = pd.to_datetime(h["date"])
    moved_away = h["transferred"].astype(str).str.lower() == "true"
    days_off = h[h["type"].isin(DAY_OFF_TYPES) & ~moved_away]

    national = days_off[days_off["locale"] == "National"]["date"]
    regional = days_off[days_off["locale"] == "Regional"][["date", "locale_name"]]
    local = days_off[days_off["locale"] == "Local"][["date", "locale_name"]]

    is_national = date.isin(national)
    is_regional = pd.MultiIndex.from_arrays([date, df["state"]]).isin(
        pd.MultiIndex.from_frame(regional))
    is_local = pd.MultiIndex.from_arrays([date, df["city"]]).isin(
        pd.MultiIndex.from_frame(local))
    df["is_holiday"] = (is_national | is_regional | is_local).astype(int)

    events = h[(h["type"] == "Event") & (h["locale"] == "National")]["date"]
    df["is_event"] = date.isin(events).astype(int)
    return df