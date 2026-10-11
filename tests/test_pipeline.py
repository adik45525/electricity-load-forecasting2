import json
import pickle
import numpy as np
import pandas as pd
import pytest
from src.data_loader import ROOT
from src.feature_engineering import make_features, feature_columns, evaluation_frame, split_masks, calendar_features
from src.evaluate import metrics
from src.predict import forecast_historical, load_model


@pytest.fixture(scope="module")
def hourly():
    frame = pd.read_csv(ROOT / "data/processed/hourly.csv", index_col="time", parse_dates=["time"])
    frame.index = pd.to_datetime(frame.index, utc=True)
    return frame


@pytest.fixture(scope="module")
def features(hourly):
    return make_features(hourly)


@pytest.fixture(scope="module")
def predictions():
    return pd.read_csv(ROOT / "reports/predictions.csv", parse_dates=["time", "origin"])


def test_utc_grid_and_missing_targets(hourly):
    assert hourly.index.is_unique
    assert hourly.index.is_monotonic_increasing
    assert str(hourly.index.tz) == "UTC"
    assert (hourly.index.to_series().diff().dropna() == pd.Timedelta(hours=1)).all()
    assert hourly.load.isna().sum() == 36
    assert hourly.load_history.notna().sum() > hourly.load.notna().sum()


@pytest.mark.parametrize("origin", ["2018-03-25", "2018-10-28", "2018-07-01"])
def test_future_load_and_weather_do_not_change_operational_features(hourly, features, origin):
    origin = pd.Timestamp(origin, tz="UTC")
    changed = hourly.copy()
    future = changed.index >= origin
    changed.loc[future, ["load", "load_history"]] = 999999
    for col in changed.columns.difference(["load", "load_history"]):
        changed.loc[future, col] = -999
    changed_features = make_features(changed)
    times = pd.date_range(origin, periods=24, freq="h")
    for scenario in ["calendar", "history", "weather_past"]:
        columns = feature_columns(features, scenario)
        pd.testing.assert_frame_equal(features.loc[times, columns], changed_features.loc[times, columns])
    assert not features.loc[times, "oracle_temp"].equals(changed_features.loc[times, "oracle_temp"])


def test_every_history_timestamp_precedes_origin(hourly, features):
    for lag in [24, 48, 168, 336]:
        assert ((features.index - pd.Timedelta(hours=lag)) < features.origin).all()
        np.testing.assert_allclose(features[f"load_lag_{lag}"].iloc[lag:], hourly.load_history.iloc[:-lag], equal_nan=True)


def test_rolling_is_frozen_at_origin(hourly, features):
    origin = pd.Timestamp("2018-04-12", tz="UTC")
    values = features.loc[pd.date_range(origin, periods=24, freq="h")]
    history = hourly.loc[(hourly.index >= origin - pd.Timedelta(hours=168)) & (hourly.index < origin), "load_history"]
    assert values.load_recent_168_mean.nunique() == 1
    assert values.load_recent_168_mean.iloc[0] == pytest.approx(history.mean())


def test_splits_and_complete_blocks(features):
    frame = evaluation_frame(features)
    tr, va, te = split_masks(frame)
    assert not (tr & va).any() and not (va & te).any()
    assert frame.loc[tr].index.max() < frame.loc[va].index.min()
    assert frame.loc[va].index.max() < frame.loc[te].index.min()
    assert (frame.groupby("origin").size() == 24).all()
    assert frame.target.notna().all()


def test_dst_preserves_repeated_hours():
    fall = pd.date_range("2018-10-28 00:00", periods=4, freq="h", tz="UTC")
    result = calendar_features(fall)
    assert result.hour.iloc[0] == result.hour.iloc[1] == 2
    assert result.utc_offset.iloc[0] == 2
    assert result.utc_offset.iloc[1] == 1
    spring = pd.date_range("2018-03-25 00:00", periods=4, freq="h", tz="UTC")
    assert calendar_features(spring).hour.iloc[:2].tolist() == [1, 3]


def test_national_holiday():
    index = pd.date_range("2018-01-01", periods=24, freq="h", tz="UTC")
    result = calendar_features(index)
    assert result.holiday.iloc[0] == 1


def test_metric_formulas():
    result = metrics([100, 200], [110, 180])
    assert result["MAE"] == 15
    assert result["RMSE"] == pytest.approx(np.sqrt(250))
    assert result["MAPE"] == 10
    assert result["sMAPE"] == pytest.approx(((20 / 210) + (40 / 380)) * 50)


def test_saved_metrics_recalculate(predictions):
    performance = pd.read_csv(ROOT / "reports/metrics.csv")
    for (model, scenario), rows in predictions.groupby(["model", "scenario"]):
        saved = performance[(performance.model == model) & (performance.scenario == scenario) & (performance.split == "test")].iloc[0]
        for name, value in metrics(rows.actual, rows.predicted).items():
            assert saved[name] == pytest.approx(value)
        assert len(rows) == saved.hours


def test_all_experiments_have_identical_targets(predictions):
    reference = predictions[predictions.model == "Seasonal Naive"].sort_values("time")
    for _, rows in predictions.groupby(["model", "scenario"]):
        rows = rows.sort_values("time")
        assert rows.time.tolist() == reference.time.tolist()
        np.testing.assert_array_equal(rows.actual.to_numpy(), reference.actual.to_numpy())
        assert rows.predicted.notna().all()


def test_selection_uses_validation_and_excludes_oracle():
    summary = json.loads((ROOT / "reports/summary.json").read_text())
    results = pd.read_csv(ROOT / "reports/metrics.csv")
    eligible = results[(results.split == "validation") & (results.scenario != "weather_oracle") & (results.model != "Seasonal Naive")]
    best = eligible.loc[eligible.MAE.idxmin()]
    assert best.model == summary["best_model"]
    assert best.scenario == summary["best_scenario"]
    assert best.MAE == pytest.approx(summary["validation_mae"])


@pytest.mark.parametrize("offset", [0, 100, -1])
def test_loaded_model_matches_saved_predictions(hourly, predictions, offset):
    bundle = load_model()
    rows = predictions[(predictions.model == bundle["name"]) & (predictions.scenario == bundle["scenario"])]
    origins = sorted(rows.origin.unique())
    origin = origins[offset]
    actual = forecast_historical(hourly, origin, bundle)
    expected = rows[rows.origin == origin].sort_values("time")
    np.testing.assert_allclose(actual.predicted, expected.predicted, rtol=1e-12)
    np.testing.assert_array_equal(actual.actual, expected.actual)


def test_rejects_current_date_without_history(hourly):
    with pytest.raises(ValueError, match="No complete"):
        forecast_historical(hourly, "2026-10-11")


def test_rejects_non_test_and_non_midnight_origin(hourly):
    with pytest.raises(ValueError, match="held-out"):
        forecast_historical(hourly, "2017-01-01")
    with pytest.raises(ValueError, match="00:00"):
        forecast_historical(hourly, "2018-01-01 12:00")


def test_catboost_shap_additivity(hourly, features):
    summary = json.loads((ROOT / "reports/summary.json").read_text())
    scenario = summary["shap_scenario"]
    if scenario == summary["best_scenario"] and summary["best_model"] == "CatBoost":
        bundle = load_model()
    else:
        with (ROOT / "models" / f"catboost_{scenario}.pkl").open("rb") as handle:
            bundle = pickle.load(handle)
    data = np.load(ROOT / "reports/shap_values.npz")
    predicted = bundle["model"].predict(pd.DataFrame(data["features"], columns=data["names"]))
    np.testing.assert_allclose(data["values"].sum(axis=1), predicted, rtol=1e-6)
