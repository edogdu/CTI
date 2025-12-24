from pathlib import Path
from typing import Dict, Any
import json


def render_reports(metrics: Dict[str, Any], out_dir: Path) -> Path:
    """Write a simple JSON report; extend with plots/CSVs as needed."""
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "pipeline_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    return report_path

