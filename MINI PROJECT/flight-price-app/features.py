"""Cleaning and feature engineering, shared by training (train.py) and the API (app.py).

Using one module for both guarantees the API builds features exactly the way the model
was trained on them.
"""
import re

import numpy as np
import pandas as pd

from festivals import FESTIVAL_WINDOW_DAYS, FESTIVALS, SUMMER_PEAK_START

CATEGORICAL = ["Airline", "Source", "Destination", "Additional_Info"]
NUMERIC = [
    "Total_Stops",
    "Duration_Mins",
    "Journey_Day",
    "Journey_Month",
    "Journey_Weekday",
    "Dep_Hour",
    "Dep_Minute",
    "Is_Weekend",
    "Is_Peak_Hour",
    "Days_To_Festival",
    "Near_Festival",
    "Is_Summer_Peak",
]
# Feature groups, used by the ablation test in train.py
RAW_TIME_FEATURES = ["Journey_Day", "Journey_Month", "Journey_Weekday", "Dep_Hour", "Dep_Minute"]
DEMAND_FEATURES = ["Is_Weekend", "Is_Peak_Hour", "Days_To_Festival", "Near_Festival", "Is_Summer_Peak"]
FEATURES = CATEGORICAL + NUMERIC

CITY_FIXES = {"Banglore": "Bengaluru", "New Delhi": "Delhi"}
STOPS = {"non-stop": 0, "1 stop": 1, "2 stops": 2, "3 stops": 3, "4 stops": 4}
MAIN_EXTRAS = ["No info", "In-flight meal not included", "No check-in baggage included"]
MIN_ROWS_PER_AIRLINE = 20  # airline categories with fewer rows can't be learned
MIN_DURATION_MINS = 30     # anything shorter is a data-entry error
PEAK_HOURS = set(range(6, 10)) | set(range(17, 21))  # 06:00-09:59 and 17:00-20:59


def duration_to_minutes(text: str) -> int:
    """'2h 50m' -> 170, '19h' -> 1140, '5m' -> 5."""
    h = re.search(r"(\d+)\s*h", text)
    m = re.search(r"(\d+)\s*m", text)
    return (int(h.group(1)) * 60 if h else 0) + (int(m.group(1)) if m else 0)


def clean_raw(raw: pd.DataFrame):
    """Turn the raw MachineHack table into a tidy frame. Returns (tidy_df, cleaning_log)."""
    log = {"raw_rows": len(raw)}
    df = raw.copy()

    log["missing_values_before"] = {k: int(v) for k, v in df.isna().sum().items() if v}

    df = df.drop_duplicates()
    log["duplicates_dropped"] = log["raw_rows"] - len(df)

    # Standardise text columns
    df["Source"] = df["Source"].replace(CITY_FIXES)
    df["Destination"] = df["Destination"].replace(CITY_FIXES)
    df["Journey_Date"] = pd.to_datetime(df["Date_of_Journey"], format="%d/%m/%Y")
    df["Duration_Mins"] = df["Duration"].map(duration_to_minutes)
    df["Total_Stops"] = df["Total_Stops"].map(STOPS)

    # Missing Total_Stops: fill with the most common value for the same airline + route
    group = ["Airline", "Source", "Destination"]
    fill = df.groupby(group)["Total_Stops"].transform(lambda s: s.mode().iloc[0] if s.notna().any() else np.nan)
    df["Total_Stops"] = df["Total_Stops"].fillna(fill).fillna(df["Total_Stops"].mode().iloc[0]).astype(int)
    log["total_stops_imputed"] = int(raw["Total_Stops"].isna().sum())

    # "Route" is redundant with Source/Destination/Total_Stops, so it is dropped (its 1 NaN no longer matters)
    # Additional_Info: fix "No Info" vs "No info", pool the rare values into "Other"
    extra = df["Additional_Info"].str.strip()
    extra = extra.where(extra.str.lower() != "no info", "No info")
    df["Additional_Info"] = extra.where(extra.isin(MAIN_EXTRAS), "Other")

    # Rare airline categories (business / premium economy / one-off carriers)
    counts = df["Airline"].value_counts()
    rare = counts[counts < MIN_ROWS_PER_AIRLINE].index
    log["rare_airlines_dropped"] = {a: int(counts[a]) for a in rare}
    df = df[~df["Airline"].isin(rare)]

    # Impossible durations (a 5-minute flight with 2 stops)
    bad = df["Duration_Mins"] < MIN_DURATION_MINS
    log["bad_duration_dropped"] = int(bad.sum())
    df = df[~bad]

    log["final_rows"] = len(df)
    keep = ["Airline", "Source", "Destination", "Additional_Info", "Journey_Date",
            "Dep_Time", "Duration_Mins", "Total_Stops", "Price"]
    return df[keep].reset_index(drop=True), log


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add the date/time, peak-time and festival features.

    Input needs: Airline, Source, Destination, Additional_Info, Journey_Date (datetime),
    Dep_Time ('HH:MM'), Duration_Mins, Total_Stops. Returns a frame with exactly FEATURES.
    """
    out = df.copy()
    d = pd.to_datetime(out["Journey_Date"])
    out["Journey_Day"] = d.dt.day
    out["Journey_Month"] = d.dt.month
    out["Journey_Weekday"] = d.dt.weekday          # Mon=0 ... Sun=6
    out["Is_Weekend"] = (d.dt.weekday >= 5).astype(int)

    dep = out["Dep_Time"].str.split(":", expand=True).astype(int)
    out["Dep_Hour"], out["Dep_Minute"] = dep[0], dep[1]
    out["Is_Peak_Hour"] = out["Dep_Hour"].isin(PEAK_HOURS).astype(int)

    # Days to the nearest festival (capped at 15 so far-away dates don't dominate)
    fest = pd.to_datetime(list(FESTIVALS))
    gap = np.abs((d.values[:, None] - fest.values[None, :]) / np.timedelta64(1, "D")).min(axis=1)
    out["Days_To_Festival"] = np.minimum(gap, 15).astype(int)
    out["Near_Festival"] = (gap <= FESTIVAL_WINDOW_DAYS).astype(int)

    out["Is_Summer_Peak"] = (d >= pd.Timestamp(SUMMER_PEAK_START)).astype(int)
    return out[FEATURES]
