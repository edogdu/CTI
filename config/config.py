import yaml
from pathlib import Path
from types import SimpleNamespace

# Path to YAML config file
CONFIG_YAML_PATH = Path(__file__).resolve().parent / "config.yaml"

def _dict_to_namespace(data):
    if isinstance(data, dict):
        return SimpleNamespace(**{k: _dict_to_namespace(v) for k, v in data.items()})
    elif isinstance(data, list):
        return [_dict_to_namespace(v) for v in data]
    else:
        return data

def load_config(path: Path = CONFIG_YAML_PATH):
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return _dict_to_namespace(data)
