from __future__ import annotations
from pathlib import Path
import json


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def load_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def load_progress(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {"completed": [], "failed": []}
    return {"completed": [], "failed": []}

def save_progress(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

def load_manifest(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {"repo": "", "runs": []}
    return {"repo": "", "runs": []}

def append_manifest_entry(path: Path, repo: Path, entry: dict) -> None:
    """Append a single run entry to manifest.json using an atomic write."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = load_manifest(path)
    runs = data.get("runs", [])
    runs.append(entry)
    new_data = {"repo": str(repo), "runs": runs}
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(new_data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)