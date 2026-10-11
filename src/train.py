import argparse
import json
import pickle
import time
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from src.data_loader import ROOT
from src.preprocessing import prepare_data
from src.feature_engineering import make_features, evaluation_frame, feature_columns, split_masks, SCENARIOS
from src.evaluate import metrics, paired_daily_interval, error_breakdown

CANDIDATES = {
    "Ridge": [{"alpha": 10.0}, {"alpha": 1000.0}],
    "Random Forest": [{"n_estimators": 100, "max_depth": 14, "min_samples_leaf": 4},
                      {"n_estimators": 100, "max_depth": 22, "min_samples_leaf": 2}],
    "CatBoost": [{"iterations": 650, "depth": 6, "learning_rate": .06},
                 {"iterations": 850, "depth": 8, "learning_rate": .05}],
}


def estimator(name, parameters):
    if name == "Ridge":
        return make_pipeline(StandardScaler(), Ridge(**parameters))
    if name == "Random Forest":
        return RandomForestRegressor(**parameters, random_state=42, n_jobs=4)
    return CatBoostRegressor(**parameters, loss_function="RMSE", random_seed=42,
                             thread_count=4, verbose=False, allow_writing_files=False)


def train():
    hourly, audit = prepare_data()
    frame = evaluation_frame(make_features(hourly))
    tr, va, te = split_masks(frame)
    rows, searches, outputs = [], [], []
    for mask, split in [(va, "validation"), (te, "test")]:
        rows.append({"model": "Seasonal Naive", "scenario": "history", "split": split,
                     **metrics(frame.loc[mask, "target"], frame.loc[mask, "load_lag_168"]),
                     "hours": int(mask.sum()), "fit_seconds": 0})
    baseline_test = frame.loc[te, ["origin", "target", "horizon"]].copy()
    baseline_test["predicted"] = frame.loc[te, "load_lag_168"]
    baseline_test["model"] = "Seasonal Naive"
    baseline_test["scenario"] = "history"
    outputs.append(baseline_test)
    best_validation = float("inf")
    best_bundle = None
    cat_bundle = None
    cat_score = float("inf")
    for scenario in SCENARIOS:
        columns = feature_columns(frame, scenario)
        for name, candidates in CANDIDATES.items():
            selected = None
            selected_score = float("inf")
            started = time.perf_counter()
            for parameters in candidates:
                model = estimator(name, parameters)
                model.fit(frame.loc[tr, columns], frame.loc[tr, "target"])
                score = metrics(frame.loc[va, "target"], model.predict(frame.loc[va, columns]))
                searches.append({"model": name, "scenario": scenario, "parameters": parameters, **score})
                if score["MAE"] < selected_score:
                    selected, selected_score, val_metrics = parameters, score["MAE"], score
            print(f"{scenario} / {name}: validation MAE={selected_score:.1f}", flush=True)
            model = estimator(name, selected)
            model.fit(frame.loc[tr | va, columns], frame.loc[tr | va, "target"])
            predicted = model.predict(frame.loc[te, columns])
            elapsed = time.perf_counter() - started
            for split, values, mask in [("validation", val_metrics, va),
                                        ("test", metrics(frame.loc[te, "target"], predicted), te)]:
                rows.append({"model": name, "scenario": scenario, "split": split, **values,
                             "hours": int(mask.sum()), "fit_seconds": elapsed,
                             "parameters": json.dumps(selected)})
            forecast = frame.loc[te, ["origin", "target", "horizon"]].copy()
            forecast["predicted"] = predicted
            forecast["model"] = name
            forecast["scenario"] = scenario
            outputs.append(forecast)
            bundle = {"model": model, "columns": columns, "scenario": scenario,
                      "name": name, "parameters": selected,
                      "train_end": "2017-12-31 23:00 UTC", "origin_hour_utc": 0}
            if scenario != "weather_oracle" and selected_score < best_validation:
                best_validation, best_bundle = selected_score, bundle
            if name == "CatBoost" and scenario != "weather_oracle" and selected_score < cat_score:
                cat_score, cat_bundle = selected_score, bundle
            with (ROOT / "models" / f"{name.replace(' ', '_').lower()}_{scenario}.pkl").open("wb") as handle:
                pickle.dump(bundle, handle)
    predictions = pd.concat(outputs).rename(columns={"target": "actual"})
    predictions.index.name = "time"
    predictions = predictions.reset_index()
    performance = pd.DataFrame(rows)
    performance.to_csv(ROOT / "reports/metrics.csv", index=False)
    predictions.to_csv(ROOT / "reports/predictions.csv", index=False)
    error_breakdown(predictions).to_csv(ROOT / "reports/error_breakdown.csv", index=False)
    (ROOT / "reports/tuning.json").write_text(json.dumps(searches, indent=2), encoding="utf-8")
    with (ROOT / "models/best_model.pkl").open("wb") as handle:
        pickle.dump(best_bundle, handle)
    metadata = {key: value for key, value in best_bundle.items() if key != "model"}
    metadata["format"] = "catboost-native-cbm" if best_bundle["name"] == "CatBoost" else "pickle"
    if best_bundle["name"] == "CatBoost":
        best_bundle["model"].save_model(str(ROOT / "models/best_model.cbm"))
    (ROOT / "models/best_model.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    with (ROOT / "models/best_model.pkl").open("rb") as handle:
        restored = pickle.load(handle)
    np.testing.assert_allclose(restored["model"].predict(frame.loc[te, restored["columns"]].iloc[:24]),
                               best_bundle["model"].predict(frame.loc[te, restored["columns"]].iloc[:24]))
    best_predictions = predictions[(predictions.model == best_bundle["name"])
                                    & (predictions.scenario == best_bundle["scenario"])]
    interval_frame = pd.concat([best_predictions, predictions[predictions.model == "Seasonal Naive"]])
    daily = []
    for origin, group in best_predictions.groupby("origin"):
        actual_peak = group.loc[group.actual.idxmax()]
        forecast_peak = group.loc[group.predicted.idxmax()]
        daily.append({"origin": origin, "actual_peak": actual_peak.actual,
                      "predicted_peak": forecast_peak.predicted,
                      "peak_magnitude_error": abs(actual_peak.actual - forecast_peak.predicted),
                      "peak_hour_error": abs(actual_peak.horizon - forecast_peak.horizon)})
    pd.DataFrame(daily).to_csv(ROOT / "reports/daily_peaks.csv", index=False)
    X = frame.loc[te, best_bundle["columns"]].iloc[::24]
    importance = permutation_importance(best_bundle["model"], X,
                                        frame.loc[te, "target"].iloc[::24],
                                        n_repeats=3, random_state=42,
                                        scoring="neg_mean_absolute_error", n_jobs=1)
    pd.DataFrame({"feature": best_bundle["columns"], "importance_mw": importance.importances_mean,
                  "std_mw": importance.importances_std}).sort_values("importance_mw", ascending=False).to_csv(
                      ROOT / "reports/feature_importance.csv", index=False)
    shap_X = frame.loc[te, cat_bundle["columns"]].iloc[::24]
    shap_values = cat_bundle["model"].get_feature_importance(Pool(shap_X), type="ShapValues")
    pd.DataFrame({"feature": cat_bundle["columns"], "mean_abs_shap_mw": np.abs(shap_values[:, :-1]).mean(axis=0)}).sort_values(
        "mean_abs_shap_mw", ascending=False).to_csv(ROOT / "reports/shap_importance.csv", index=False)
    np.savez_compressed(ROOT / "reports/shap_values.npz", values=shap_values,
                        features=shap_X.to_numpy(), names=np.array(cat_bundle["columns"]))
    summary = {
        "best_model": best_bundle["name"], "best_scenario": best_bundle["scenario"],
        "selection": "Minimum validation MAE, excluding oracle; final refit on train + validation",
        "validation_mae": best_validation,
        "test": metrics(best_predictions.actual, best_predictions.predicted),
        "bootstrap": paired_daily_interval(interval_frame, best_bundle["name"]),
        "split_hours": {"train": int(tr.sum()), "validation": int(va.sum()), "test": int(te.sum())},
        "split_days": {name: int(frame.loc[mask, "origin"].nunique())
                       for name, mask in [("train", tr), ("validation", va), ("test", te)]},
        "excluded_hours": len(hourly) - len(frame),
        "shap_model": cat_bundle["name"], "shap_scenario": cat_bundle["scenario"],
        "peak_mae_mw": float(pd.DataFrame(daily).peak_magnitude_error.mean()),
        "peak_hour_mae": float(pd.DataFrame(daily).peak_hour_error.mean()),
        "model_roundtrip_verified": True,
    }
    (ROOT / "reports/summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    from src.analysis import create_analysis
    create_analysis(hourly, frame, performance, predictions, summary, audit)


if __name__ == "__main__":
    argparse.ArgumentParser(description="Train all experiments on the chronological holdout").parse_args()
    train()
