import sys
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple


COSINE_THRESHOLD = 0.7
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config.config import load_config


@dataclass
class GroundTruthItem:
    names: Set[str]
    descriptions: Set[str]

    @classmethod
    def empty(cls) -> "GroundTruthItem":
        return cls(names=set(), descriptions=set())

    def merge(self, name: Optional[str], description: Optional[str]) -> None:
        if name:
            cleaned = name.strip()
            if cleaned:
                self.names.add(cleaned)
        if description:
            cleaned = description.strip()
            if cleaned:
                self.descriptions.add(cleaned)


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip())


def normalize_key(value: str) -> str:
    return re.sub(r"\W+", "", value.lower())


def load_manifest(manifest_path: Path) -> List[Dict]:
    with manifest_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload.get("runs", [])


def load_doc_identifier_map(mapping_csv: Path) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    if not mapping_csv.exists():
        return mapping
    with mapping_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            document = row.get("Document")
            identifier = row.get("Identifier")
            if document and identifier:
                mapping[document.strip()] = identifier.strip()
    return mapping


def collect_annotation_paths(
    datasets_dir: Path, group: str, identifier: str
) -> List[Path]:
    paths: List[Path] = []
    base = datasets_dir / "CTI-HAL" / "Data" / group
    for annotator in ("annotator L", "annotator S"):
        candidate = base / annotator / f"{identifier}.json"
        if candidate.exists():
            paths.append(candidate)
    return paths


def _add_gt(
    gt: Dict[str, GroundTruthItem],
    clean_id: Optional[str],
    name: Optional[str],
    description: Optional[str],
) -> None:
    if not clean_id:
        return
    cid = clean_id.strip().upper()
    if not cid:
        return
    item = gt.setdefault(cid, GroundTruthItem.empty())
    item.merge(name, description)


def load_ground_truth(annotation_paths: Iterable[Path]) -> Dict[str, GroundTruthItem]:
    gt: Dict[str, GroundTruthItem] = {}
    for path in annotation_paths:
        try:
            with path.open("r", encoding="utf-8") as f:
                entries = json.load(f)
        except Exception:
            continue
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            meta = entry.get("metadata", {}) or {}
            description = meta.get("description")
            technique = entry.get("technique")
            technique_name = meta.get("technique_name")
            _add_gt(gt, technique, technique_name, description)

            sub_tech = meta.get("sub_technique")
            sub_name = meta.get("sub_technique_name")
            _add_gt(gt, sub_tech, sub_name, description)

            tactic_ids = meta.get("tactic") or []
            tactic_names = meta.get("tactic_name") or []
            for idx, tid in enumerate(tactic_ids):
                name = tactic_names[idx] if idx < len(tactic_names) else None
                _add_gt(gt, tid, name, description)

            tool_ids = meta.get("tool") or []
            tool_names = meta.get("tool_name") or []
            for idx, tool_id in enumerate(tool_ids):
                name = tool_names[idx] if idx < len(tool_names) else None
                _add_gt(gt, tool_id, name, description)
    return gt


def load_chunk_contexts(chunk_path: Path) -> Dict[str, Set[str]]:
    context_map: Dict[str, Set[str]] = {}
    if not chunk_path.exists():
        return context_map

    with chunk_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    data = payload.get("data", []) if isinstance(payload, dict) else []
    for entry in data:
        if not isinstance(entry, dict):
            continue
        ctx = entry.get("context")
        if not ctx:
            continue
        ctx_clean = normalize_text(str(ctx))
        triples = entry.get("triple") or []
        for triple in triples:
            if not isinstance(triple, dict):
                continue
            for role in ("subject", "object"):
                node = triple.get(role) or {}
                if not isinstance(node, dict):
                    continue
                name = node.get("name")
                if not name:
                    continue
                raw_key = name.strip().lower()
                norm_key = normalize_key(name)
                for key in (raw_key, norm_key):
                    if not key:
                        continue
                    context_map.setdefault(key, set()).add(ctx_clean)
    return context_map


def iter_similarity_candidates(similarity_dir: Path) -> Iterable[Tuple[Dict, Dict, Dict]]:
    per_node = similarity_dir / "similarity_per_node.json"
    if not per_node.exists():
        return []
    with per_node.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        return []
    for node_info in payload.values():
        if not isinstance(node_info, dict):
            continue
        base_source = node_info.get("source") or {}
        top_k = node_info.get("top_k") or []
        for candidate in top_k:
            if not isinstance(candidate, dict):
                continue
            target = candidate.get("target") or {}
            scores = candidate.get("scores") or {}
            source = candidate.get("source") or base_source
            yield source, target, scores


def infer_target_type(clean_id: str, target: Dict[str, any]) -> str:
    cid = clean_id.upper()
    if cid.startswith("TA"):
        return "TACTIC"
    if cid.startswith("T"):
        return "TECHNIQUE" if "." not in cid else "SUB_TECHNIQUE"
    if cid.startswith("S"):
        return "SOFTWARE"

    labels = target.get("labels") or []
    for label in labels:
        if isinstance(label, str) and label.startswith("Ucoex"):
            return label.replace("Ucoex", "").upper()

    type_field = target.get("type")
    if isinstance(type_field, str) and type_field.strip():
        return type_field.strip().upper()
    return "UNKNOWN"


def build_dataset_record(
    source: Dict,
    target: Dict,
    scores: Dict,
    context_map: Dict[str, Set[str]],
    ground_truth: Dict[str, GroundTruthItem],
    report_id: str,
    report_name: str,
) -> Optional[Dict]:
    cosine = scores.get("cosine")
    try:
        cosine_val = float(cosine)
    except (TypeError, ValueError):
        return None
    if cosine_val <= COSINE_THRESHOLD:
        return None

    clean_id = target.get("clean_id")
    if not clean_id:
        return None
    clean_id = clean_id.strip().upper()
    if not clean_id:
        return None

    source_name = (source.get("name") or "").strip()
    source_uid = source.get("uid")
    source_uid_str = str(source_uid).strip() if source_uid is not None else ""
    if not source_name:
        return None
    source_key = source_name.lower()
    norm_key = normalize_key(source_name)
    context_values: Set[str] = set()
    for key in filter(None, (source_key, norm_key)):
        context_values |= context_map.get(key, set())
    if not context_values:
        return None

    gt_item = ground_truth.get(clean_id)
    target_name = target.get("name")
    target_description = target.get("description")
    if gt_item:
        if (not target_name or not target_name.strip()) and gt_item.names:
            target_name = next(iter(gt_item.names))
        if (not target_description or not str(target_description).strip()) and gt_item.descriptions:
            target_description = next(iter(gt_item.descriptions))

    target_name = (target_name or clean_id).strip()
    target_description = (target_description or "").strip()
    label = 1 if clean_id in ground_truth else 0
    target_type = infer_target_type(clean_id, target)

    contexts = sorted(context_values)

    source_key_for_filter = source_uid_str or source_name

    return {
        "c_clean_id": clean_id,
        "c_name": target_name,
        "c_description": target_description,
        "c_type": target_type,
        "s_name": source_name,
        "s_type": (source.get("type") or "").strip(),
        "s_contexts": contexts,
        "label": label,
        "s_report_id": report_id,
        "s_report_name": report_name,
        "cosine": round(cosine_val, 6),
        "_source_key": source_key_for_filter,
    }


def main() -> None:
    cfg = load_config()
    output_root = (REPO_ROOT / Path(cfg.paths.output_dir)).resolve()
    manifest_path = output_root / "CTI-HAL" / "manifest.json"
    mapping_csv = (REPO_ROOT / Path(cfg.paths.config_dir)).resolve() / "CTI_HAL_mappings.csv"
    datasets_dir = (REPO_ROOT / Path(cfg.paths.datasets_dir)).resolve()
    dataset_out = (REPO_ROOT / "datasets" / "reranker" / "ce_dataset.jsonl").resolve()

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found at {manifest_path}")

    runs = load_manifest(manifest_path)
    doc_map = load_doc_identifier_map(mapping_csv)

    rows: List[Dict] = []

    for run in runs:
        pdf_path = run.get("pdf")
        similarity_dir_str = run.get("similarity_dir")
        chunk_json_str = run.get("chunk_json")
        report_id = run.get("document_id") or run.get("document_props", {}).get("id")
        group = run.get("group")
        if not all((pdf_path, similarity_dir_str, chunk_json_str, report_id, group)):
            continue
        pdf_name = Path(pdf_path).name
        identifier = doc_map.get(pdf_name)
        if not identifier:
            print(f"[warn] No identifier mapping for {pdf_name}, skipping annotation lookup.")
            annotation_paths: List[Path] = []
        else:
            annotation_paths = collect_annotation_paths(datasets_dir, group, identifier)
        gt = load_ground_truth(annotation_paths)

        chunk_path = Path(chunk_json_str)
        if not chunk_path.is_absolute():
            chunk_path = REPO_ROOT / chunk_path
        context_map = load_chunk_contexts(chunk_path)

        similarity_dir = Path(similarity_dir_str)
        if not similarity_dir.is_absolute():
            similarity_dir = REPO_ROOT / similarity_dir
        if not similarity_dir.exists():
            print(f"[warn] Similarity directory missing: {similarity_dir}")
            continue

        for source, target, scores in iter_similarity_candidates(similarity_dir):
            record = build_dataset_record(
                source, target, scores, context_map, gt, report_id, pdf_name
            )
            if record:
                rows.append(record)

    if rows:
        # track which sources produced at least one hit
        hit_sources: Set[str] = {row["_source_key"] for row in rows if row["label"] == 1}
        rows = [row for row in rows if row["_source_key"] in hit_sources]
        for row in rows:
            row.pop("_source_key", None)

    dataset_out.parent.mkdir(parents=True, exist_ok=True)
    with dataset_out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"[info] Wrote {len(rows)} examples to {dataset_out}")


if __name__ == "__main__":
    main()
