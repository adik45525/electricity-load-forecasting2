import json
import pickle
import pandas as pd
from src.data_loader import ROOT
from src.feature_engineering import make_features


def load_model(path=None):
    native = ROOT / "models/best_model.cbm"
    metadata = ROOT / "models/best_model.json"
    portable = json.loads(metadata.read_text()) if metadata.exists() else {}
    if path is None and native.exists() and portable.get("format") == "catboost-native-cbm":
        from catboost import CatBoostRegressor

        bundle = portable
        model = CatBoostRegressor()
        model.load_model(str(native))
        bundle["model"] = model
        return bundle
    # Pickle is retained only for trusted local experiment artifacts.
    path = path or ROOT / "models/best_model.pkl"
    with path.open("rb") as handle:
        return pickle.load(handle)


def forecast_historical(hourly, origin, bundle=None):
    bundle = bundle or load_model()
    origin = pd.Timestamp(origin)
    origin = origin.tz_localize("UTC") if origin.tzinfo is None else origin.tz_convert("UTC")
    if origin.hour or origin.minute or origin.second:
        raise ValueError("Forecast origin must be 00:00 UTC")
    if origin < pd.Timestamp("2018-01-01", tz="UTC"):
        raise ValueError("Historical demo is restricted to the held-out test period")
    times = pd.date_range(origin, periods=24, freq="h")
    if not times.isin(hourly.index).all():
        raise ValueError("No complete 24-hour historical block; fresh history is unavailable")
    history_start = origin - pd.Timedelta(hours=336)
    window = hourly.loc[history_start:times[-1]]
    features = make_features(window).reindex(times)
    X = features[bundle["columns"]]
    if X.isna().any().any():
        raise ValueError("Insufficient historical inputs")
    options = {"thread_count": 2} if bundle["name"] == "CatBoost" else {}
    return pd.DataFrame({"actual": hourly.load.reindex(times),
                         "predicted": bundle["model"].predict(X, **options)}, index=times)
