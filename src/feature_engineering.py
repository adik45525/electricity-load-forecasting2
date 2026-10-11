import numpy as np
import pandas as pd
import holidays

SCENARIOS = ["calendar", "history", "weather_past", "weather_oracle"]
SCENARIO_NAMES = {
    "calendar": "Только календарь", "history": "Календарь + история",
    "weather_past": "Календарь + история + прошлая погода",
    "weather_oracle": "Календарь + история + будущая погода (oracle)",
}


def calendar_features(index):
    local = index.tz_convert("Europe/Madrid")
    result = pd.DataFrame(index=index)
    es_holidays = holidays.Spain(years=sorted(set(local.year)))
    for name, values, period in [("hour", local.hour, 24),
                                  ("weekday", local.dayofweek, 7),
                                  ("year_day", local.dayofyear, 365.25)]:
        result[name] = values
        result[name + "_sin"] = np.sin(2 * np.pi * values / period)
        result[name + "_cos"] = np.cos(2 * np.pi * values / period)
    result["month"] = local.month
    result["weekend"] = (local.dayofweek >= 5).astype(int)
    result["holiday"] = [int(date in es_holidays) for date in local.date]
    result["utc_offset"] = [date.utcoffset().total_seconds() / 3600 for date in local]
    result["horizon"] = index.hour + 1
    for hour in range(24):
        result[f"hour_is_{hour}"] = (local.hour == hour).astype(int)
    for day in range(7):
        result[f"weekday_is_{day}"] = (local.dayofweek == day).astype(int)
    for month in range(1, 13):
        result[f"month_is_{month}"] = (local.month == month).astype(int)
    result["working_hour"] = ((local.hour >= 8) & (local.hour <= 20)
                              & (local.dayofweek < 5)).astype(int)
    return result


def make_features(hourly):
    result = calendar_features(hourly.index)
    origin = hourly.index.floor("D")
    load = hourly.load_history
    for lag in [24, 48, 168, 336]:
        result[f"load_lag_{lag}"] = load.shift(lag)
    # Every row in a 24-hour block uses statistics frozen at that block's origin.
    for window in [24, 168]:
        previous = load.shift(1).rolling(window, min_periods=window)
        for stat in ["mean", "std", "min", "max"]:
            values = getattr(previous, stat)()
            result[f"load_recent_{window}_{stat}"] = values.reindex(origin).to_numpy()
    weather_cols = [c for c in hourly if c in ["temp", "pressure", "humidity", "wind_speed", "rain_1h", "clouds_all"]
                    or c.startswith("temp_")]
    weather_features = {}
    for col in weather_cols:
        weather_features["past_" + col] = hourly[col].shift(24)
        weather_features["oracle_" + col] = hourly[col]
    result = pd.concat([result, pd.DataFrame(weather_features, index=hourly.index)], axis=1).copy()
    result["past_heating"] = (18 - result.past_temp).clip(lower=0)
    result["past_cooling"] = (result.past_temp - 22).clip(lower=0)
    result["oracle_heating"] = (18 - result.oracle_temp).clip(lower=0)
    result["oracle_cooling"] = (result.oracle_temp - 22).clip(lower=0)
    result["target"] = hourly.load
    result["origin"] = origin
    return result


def feature_columns(frame, scenario):
    calendar = [c for c in frame if c not in ["target", "origin"]
                and not c.startswith(("load_", "past_", "oracle_"))]
    history = [c for c in frame if c.startswith("load_")]
    past = [c for c in frame if c.startswith("past_")]
    oracle = [c for c in frame if c.startswith("oracle_")]
    return calendar + (history if scenario != "calendar" else []) + (
        past if scenario == "weather_past" else oracle if scenario == "weather_oracle" else [])


def evaluation_frame(frame):
    # Shared sample across experiments: 24 observed targets and complete lag history.
    complete = frame.dropna()
    counts = complete.groupby("origin").size()
    return complete[complete.origin.isin(counts[counts == 24].index)].copy()


def split_masks(frame):
    train_end = pd.Timestamp("2017-01-01", tz="UTC")
    val_end = pd.Timestamp("2018-01-01", tz="UTC")
    return (frame.origin < train_end,
            (frame.origin >= train_end) & (frame.origin < val_end),
            frame.origin >= val_end)
