from __future__ import annotations
from pathlib import Path
import subprocess


def _log(msg: str) -> None:
    # Lightweight local logger
    print(msg)


def ensure_repo(repo_url: str, dest: Path) -> None:
    dest_parent = dest.parent
    dest_parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        _log(f"[Dataset] Cloning {repo_url} -> {dest}")
        subprocess.check_call(["git", "clone", "--depth", "1", repo_url, str(dest)])
    else:
        try:
            _log(f"[Dataset] Pulling latest in {dest}")
            subprocess.check_call(["git", "-C", str(dest), "pull", "--ff-only"])
        except subprocess.CalledProcessError as e:
            _log(f"[Dataset] Warning: git pull failed ({e}); continuing with existing checkout")


def enumerate_cti_hal(repo_root: Path):
    """Yield tuples of (pdf_path, group_name, annotator_L_dir, annotator_S_dir)."""
    reports_dir = repo_root / "Reports"
    data_dir = repo_root / "Data"
    candidates = sorted(reports_dir.glob("**/*.pdf"), key=lambda p: str(p).lower())
    for pdf in candidates:
        try:
            group = pdf.relative_to(reports_dir).parts[0]
        except Exception:
            group = "unknown"
        ann_L = data_dir / group / "annotator L"
        ann_S = data_dir / group / "annotator S"
        yield pdf, group, ann_L, ann_S

