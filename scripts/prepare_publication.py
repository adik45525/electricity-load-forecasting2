"""Build a self-contained publication tree without local credentials or training models."""
import hashlib
import json
from pathlib import Path
import re
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "publication"
ROOT_FILES = ["README.md", "DATA_LICENSE.md", "Dockerfile", ".dockerignore", ".gitignore",
              ".python-version", "pytest.ini", "requirements.txt", "requirements-training.txt",
              "requirements.lock.txt"]
EXCLUDED_REPORTS = {"training.log", "tests.log", "kaggle_metadata.json"}
SECRET_PATTERNS = [r"gh[pousr]_[A-Za-z0-9]{30,}", r"github_pat_[A-Za-z0-9_]{30,}",
                   r"AKIA[0-9A-Z]{16}", r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"]


def publication_paths():
    paths = [ROOT / name for name in ROOT_FILES]
    for folder in ["app", "src", "tests", "notebooks", "scripts", "data", ".streamlit"]:
        for path in (ROOT / folder).rglob("*"):
            if path.is_file() and "__pycache__" not in path.parts:
                if path.name == "secrets.toml" or path.name.startswith("secrets."):
                    continue
                if path.suffix not in [".py", ".html", ".toml", ".csv", ".ipynb", ".json", ".md"]:
                    continue
                paths.append(path)
    paths.extend([ROOT / "models/best_model.cbm", ROOT / "models/best_model.json"])
    for path in (ROOT / "reports").rglob("*"):
        if path.is_file() and path.name not in EXCLUDED_REPORTS and path.suffix in [".csv", ".json", ".md", ".png", ".jpg", ".npz"]:
            paths.append(path)
    return sorted(set(paths))


def prepare():
    paths = publication_paths()
    records = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path.relative_to(ROOT))
        size = path.stat().st_size
        if size >= 25 * 1024 * 1024:
            raise ValueError(f"Publication file exceeds 25 MiB: {path.relative_to(ROOT)}")
        content = path.read_bytes()
        if path.suffix in [".py", ".html", ".md", ".json", ".toml", ".ipynb", ".txt"]:
            text = content.decode("utf-8")
            if any(re.search(pattern, text) for pattern in SECRET_PATTERNS):
                raise ValueError(f"Possible secret in {path.relative_to(ROOT)}")
        relative = path.relative_to(ROOT)
        destination = DESTINATION / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        records.append({"path": relative.as_posix(), "bytes": size,
                        "sha256": hashlib.sha256(content).hexdigest()})
    manifest = {"files": records, "total_bytes": sum(row["bytes"] for row in records),
                "model_format": "CatBoost native CBM", "python": "3.12",
                "entrypoint": "app/streamlit_app.py", "training_on_startup": False,
                "dataset_license": "CC0: Public Domain"}
    (DESTINATION / "publication_manifest.json").write_text(json.dumps(manifest, indent=2))
    with zipfile.ZipFile(ROOT / "publication/gridscope-publication.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for record in records:
            archive.write(DESTINATION / record["path"], record["path"])
        archive.write(DESTINATION / "publication_manifest.json", "publication_manifest.json")
    print(json.dumps({"files": len(records), "total_mib": manifest["total_bytes"] / 1024 ** 2,
                      "largest_mib": max(row["bytes"] for row in records) / 1024 ** 2,
                      "output": str(DESTINATION)}, indent=2))


if __name__ == "__main__":
    prepare()
