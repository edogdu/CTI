from __future__ import annotations

from pathlib import Path
import json
from dataclasses import asdict, is_dataclass
from typing import Any
import time


# Legacy JSON helpers
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


# IR serialization helpers
def _serialize(obj: Any) -> Any:
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, list):
        return [_serialize(o) for o in obj]
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    return obj


def ensure_stage_dir(stage_name: str) -> Path:
    out = Path("results") / stage_name
    out.mkdir(parents=True, exist_ok=True)
    return out


def save_stage_outputs(stage_name: str, obj: Any, *, jsonl_threshold: int = 200) -> Path | None:
    """
    Save stage outputs under results/<stage_name>/.
    - If obj is a list longer than jsonl_threshold, write JSONL.
    - Otherwise write a JSON array/object.
    Returns the written path or None on failure (logs to stdout).
    """
    stage_dir = ensure_stage_dir(stage_name)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    try:
        if isinstance(obj, list) and len(obj) > jsonl_threshold:
            out_path = stage_dir / f"{timestamp}.jsonl"
            with open(out_path, "w", encoding="utf-8") as f:
                for row in obj:
                    f.write(json.dumps(_serialize(row), ensure_ascii=False) + "\n")
        else:
            out_path = stage_dir / f"{timestamp}.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(_serialize(obj), f, ensure_ascii=False, indent=2)
        print(f"[Save] {stage_name} -> {out_path}")
        return out_path
    except Exception as e:
        print(f"[Save] Failed to write {stage_name}: {e}")
        return None


__all__ = [
    "save_json",
    "load_json",
    "load_progress",
    "save_progress",
    "load_manifest",
    "append_manifest_entry",
    "save_stage_outputs",
    "ensure_stage_dir",
]

