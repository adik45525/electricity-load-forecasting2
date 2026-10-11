from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def load_raw(raw_dir=ROOT / "data/raw"):
    energy = pd.read_csv(raw_dir / "energy_dataset.csv")
    weather = pd.read_csv(raw_dir / "weather_features.csv")
    required = {"time", "total load actual"}
    if not required.issubset(energy):
        raise ValueError(f"Missing energy columns: {required - set(energy)}")
    if not {"dt_iso", "city_name", "temp"}.issubset(weather):
        raise ValueError("Missing weather columns")
    energy["time"] = pd.to_datetime(energy["time"], utc=True, errors="raise")
    weather["dt_iso"] = pd.to_datetime(weather["dt_iso"], utc=True, errors="raise")
    weather["city_name"] = weather["city_name"].str.strip()
    return energy, weather
