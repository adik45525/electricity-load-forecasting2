import numpy as np
import pandas as pd


def metrics(actual, predicted):
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    error = actual - predicted
    return {
        "MAE": float(np.mean(np.abs(error))),
        "RMSE": float(np.sqrt(np.mean(error ** 2))),
        "MAPE": float(np.mean(np.abs(error) / np.maximum(np.abs(actual), 1)) * 100),
        "sMAPE": float(np.mean(2 * np.abs(error) /
                            np.maximum(np.abs(actual) + np.abs(predicted), 1)) * 100),
    }


def paired_daily_interval(predictions, candidate, baseline="Seasonal Naive", draws=2000):
    data = predictions[predictions.model.isin([candidate, baseline])].copy()
    data["error"] = (data.actual - data.predicted).abs()
    daily = data.groupby(["origin", "model"]).error.mean().unstack()
    difference = (daily[candidate] - daily[baseline]).dropna().to_numpy()
    rng = np.random.default_rng(42)
    samples = rng.choice(difference, (draws, len(difference)), replace=True).mean(axis=1)
    return {"mae_difference_mw": float(difference.mean()),
            "lower_95": float(np.quantile(samples, .025)),
            "upper_95": float(np.quantile(samples, .975)), "days": len(difference)}


def error_breakdown(predictions):
    local = pd.DatetimeIndex(pd.to_datetime(predictions.time, utc=True)).tz_convert("Europe/Madrid")
    data = predictions.copy()
    data["hour"] = local.hour
    data["month"] = local.month
    data["weekend"] = local.dayofweek >= 5
    data["absolute_error"] = (data.actual - data.predicted).abs()
    frames = []
    for grouping in ["hour", "month", "weekend", "horizon"]:
        grouped = data.groupby(["model", "scenario", grouping]).absolute_error.agg(["mean", "count"]).reset_index()
        grouped = grouped.rename(columns={grouping: "value", "mean": "MAE", "count": "hours"})
        grouped["group"] = grouping
        frames.append(grouped)
    return pd.concat(frames, ignore_index=True)
