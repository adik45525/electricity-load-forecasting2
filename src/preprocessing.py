import hashlib
import json
import numpy as np
import pandas as pd
from src.data_loader import ROOT, load_raw

WEATHER_BOUNDS = {
    "temp": (223.15, 333.15), "pressure": (850, 1100),
    "humidity": (0, 100), "wind_speed": (0, 75),
    "rain_1h": (0, 200), "clouds_all": (0, 100),
}


def prepare_data():
    energy, weather = load_raw()
    audit = {
        "energy_rows": len(energy), "weather_rows": len(weather),
        "energy_duplicate_timestamps": int(energy.duplicated("time").sum()),
        "weather_duplicate_keys": int(weather.duplicated(["dt_iso", "city_name"]).sum()),
        "cities": sorted(weather.city_name.unique().tolist()),
        "weather_invalid": {},
        "sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in (ROOT / "data/raw").glob("*.csv")},
    }
    energy = energy.groupby("time")["total load actual"].mean().sort_index()
    index = pd.date_range(energy.index.min(), energy.index.max(), freq="h")
    audit["missing_energy_hours"] = int(len(index) - len(energy))
    energy = energy.reindex(index)
    invalid_load = (energy <= 0) | (energy > 100000)
    audit["invalid_load"] = int(invalid_load.sum())
    audit["missing_load"] = int(energy.isna().sum())
    energy = energy.mask(invalid_load)
    for col, (low, high) in WEATHER_BOUNDS.items():
        bad = ~weather[col].between(low, high) & weather[col].notna()
        audit["weather_invalid"][col] = int(bad.sum())
        weather.loc[bad, col] = np.nan
    weather["temp"] -= 273.15
    weather = weather.groupby(["dt_iso", "city_name"])[list(WEATHER_BOUNDS)].mean()
    frames = []
    for city in audit["cities"]:
        frame = weather.xs(city, level="city_name").reindex(index)
        audit[f"missing_weather_{city}"] = int(frame.isna().sum().sum())
        # Only past observations fill gaps; no interpolation from the future.
        frames.append(frame.ffill(limit=24).add_suffix("_" + city))
    weather_wide = pd.concat(frames, axis=1)
    result = pd.DataFrame({"load": energy, "load_history": energy.ffill(limit=24)})
    for col in WEATHER_BOUNDS:
        columns = [f"{col}_{city}" for city in audit["cities"]]
        result[col] = weather_wide[columns].mean(axis=1)
    for city in audit["cities"]:
        result[f"temp_{city}"] = weather_wide[f"temp_{city}"]
    result["temp_spread"] = weather_wide.filter(regex="^temp_").std(axis=1)
    result.index.name = "time"
    audit.update({"start_utc": str(index.min()), "end_utc": str(index.max()),
                  "hours": len(result), "target_observed": int(result.load.notna().sum()),
                  "load_min_mw": float(result.load.min()), "load_max_mw": float(result.load.max()),
                  "timezone": "UTC storage; Europe/Madrid calendar",
                  "aggregation": "Equal-weight mean of five cities; not population weighted"})
    (ROOT / "data/processed").mkdir(parents=True, exist_ok=True)
    result.to_csv(ROOT / "data/processed/hourly.csv")
    (ROOT / "reports/data_quality.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return result, audit
