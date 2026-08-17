"""Convert DNRTI gold dataset to multi-task QLoRA training format.

Generates two types of training examples from each document:
  - NER: "extract entities from this text" -> gold entity list
  - RE:  "given these entities, extract relationships" -> gold triple list

Both include negative examples (empty [] outputs) to teach restraint.

Uses iterative multi-label stratification (Sechidis et al., 2011) to split
into train/val/test with balanced predicate AND entity type distributions.
"""
from __future__ import annotations

import json
import logging
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

from cti_analysis.ontology import (
    TYPES,
    ENTITY_EXTRACTION_PROMPT,
    EXTRACTION_PROMPT,
    RELATION_EXTRACTION_PROMPT,
    format_entity_hints,
)

from .config import DataConfig

logger = logging.getLogger(__name__)


# =============================================================================
# GOLD EXTRACTION FROM DNRTI FORMAT
# =============================================================================

def _build_gold_triples(entry: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Convert a single DNRTI entry to the target JSON triple format.

    DNRTI relation format: [predicate, head_idx, tail_idx]
    Per normalize.py convention:
      - tail_idx (rel[2]) = SUBJECT entity
      - head_idx (rel[1]) = OBJECT entity
    """
    entities = entry.get("entities", [])
    relations = entry.get("relations", [])
    triples = []

    for rel in relations:
        if not (isinstance(rel, list) and len(rel) >= 3):
            continue

        predicate = rel[0]
        head_idx = rel[1]  # object
        tail_idx = rel[2]  # subject

        if predicate == "noRelation":
            continue

        if not (0 <= head_idx < len(entities) and 0 <= tail_idx < len(entities)):
            continue

        subj_ent = entities[tail_idx]
        obj_ent = entities[head_idx]

        if len(subj_ent) < 4 or len(obj_ent) < 4:
            continue

        subj_name = subj_ent[2]
        subj_label = subj_ent[3]
        obj_name = obj_ent[2]
        obj_label = obj_ent[3]

        subj_type = subj_label  # use native DNRTI label
        obj_type = obj_label

        if subj_type not in TYPES or obj_type not in TYPES:
            continue

        triples.append({
            "subject": {"name": subj_name, "type": subj_type},
            "predicate": predicate,
            "object": {"name": obj_name, "type": obj_type},
        })

    return triples


def _build_gold_entities(entry: Dict[str, Any]) -> List[Dict[str, str]]:
    """Extract unique STIX-typed entities from a DNRTI entry.

    Returns deduplicated list of {"name": ..., "type": ...} dicts.
    """
    entities = entry.get("entities", [])
    seen: set = set()
    result = []

    for ent in entities:
        if len(ent) < 4:
            continue
        name = ent[2]
        label = ent[3]
        if label not in TYPES:
            continue
        key = (name.lower(), label)
        if key not in seen:
            seen.add(key)
            result.append({"name": name, "type": label})

    return result


def _get_doc_labels(triples: List[Dict[str, Any]], entities: List[Dict[str, str]]) -> Set[str]:
    """Extract multi-hot label set for stratification.

    Labels are prefixed to avoid collisions: 'pred:uses', 'ent:malware'.
    """
    labels = set()
    for t in triples:
        labels.add(f"pred:{t['predicate']}")
        labels.add(f"ent:{t['subject']['type']}")
        labels.add(f"ent:{t['object']['type']}")
    for e in entities:
        labels.add(f"ent:{e['type']}")
    return labels


# =============================================================================
# CONVERSATION BUILDERS (one per task type)
# =============================================================================

def _format_entities_pipe(entities: List[Dict[str, str]]) -> str:
    """Format entity list as pipe-delimited lines."""
    if not entities:
        return "NONE"
    return "\n".join(f"{e['name']} | {e['type']}" for e in entities)


def _format_triples_pipe(triples: List[Dict[str, Any]]) -> str:
    """Format triple list as pipe-delimited lines."""
    if not triples:
        return "NONE"
    lines = []
    for t in triples:
        lines.append(
            f"{t['subject']['name']} | {t['subject']['type']} | "
            f"{t['predicate']} | "
            f"{t['object']['name']} | {t['object']['type']}"
        )
    return "\n".join(lines)


def _build_ner_conversation(
    text: str,
    entities: List[Dict[str, str]],
) -> List[Dict[str, str]]:
    """Build a NER training example: extract entities from text."""
    user_content = ENTITY_EXTRACTION_PROMPT.format(text=text)
    assistant_content = _format_entities_pipe(entities)
    return [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_content},
    ]


def _build_re_conversation(
    text: str,
    entities: List[Dict[str, str]],
    triples: List[Dict[str, Any]],
) -> List[Dict[str, str]]:
    """Build an RE training example: classify relations given entity hints."""
    hint_text = format_entity_hints(entities)
    user_content = RELATION_EXTRACTION_PROMPT.format(
        text=text, entity_hints=hint_text,
    )
    assistant_content = _format_triples_pipe(triples)
    return [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_content},
    ]


def _build_legacy_conversation(
    text: str,
    triples: List[Dict[str, Any]],
) -> List[Dict[str, str]]:
    """Build a legacy single-shot training example (backward compat)."""
    user_content = EXTRACTION_PROMPT.format(text=text)
    assistant_content = _format_triples_pipe(triples)
    return [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_content},
    ]


# =============================================================================
# ITERATIVE MULTI-LABEL STRATIFICATION (Sechidis et al., 2011)
# =============================================================================

def _iterative_stratified_split(
    items: List[Tuple[Any, Set[str], int]],
    split_ratios: List[float],
    seed: int,
) -> List[List[Tuple[Any, int]]]:
    """Split items into N sets with balanced label distributions.

    Each item is (payload, label_set, original_doc_index).
    split_ratios: e.g. [0.80, 0.10, 0.10] for train/val/test.

    Returns N lists of (payload, original_doc_index) tuples.
    """
    rng = random.Random(seed)
    n_splits = len(split_ratios)

    items = list(items)
    rng.shuffle(items)

    splits: List[List[Tuple[Any, int]]] = [[] for _ in range(n_splits)]
    assigned = [False] * len(items)

    label_freq: Counter = Counter()
    for _, labels, _ in items:
        for label in labels:
            label_freq[label] += 1

    split_label_counts: List[Counter] = [Counter() for _ in range(n_splits)]

    sorted_labels = [label for label, _ in label_freq.most_common()]
    sorted_labels.reverse()  # rarest first

    for label in sorted_labels:
        unassigned_with_label = [
            i for i, (_, labels, _) in enumerate(items)
            if not assigned[i] and label in labels
        ]

        for item_idx in unassigned_with_label:
            if assigned[item_idx]:
                continue

            best_split = 0
            best_need = -float("inf")

            for s in range(n_splits):
                current_count = split_label_counts[s][label]
                target_ratio = split_ratios[s]

                current_ratio = current_count / label_freq[label] if label_freq[label] > 0 else 0
                need = target_ratio - current_ratio

                overall_ratio = len(splits[s]) / len(items) if len(items) > 0 else 0
                size_need = target_ratio - overall_ratio

                score = need + 0.5 * size_need

                if score > best_need:
                    best_need = score
                    best_split = s

            _, labels, doc_idx = items[item_idx]
            splits[best_split].append((items[item_idx][0], doc_idx))
            assigned[item_idx] = True

            for lbl in labels:
                split_label_counts[best_split][lbl] += 1

    remaining = [i for i in range(len(items)) if not assigned[i]]
    for k, item_idx in enumerate(remaining):
        s = k % n_splits
        splits[s].append((items[item_idx][0], items[item_idx][2]))

    for s in splits:
        rng.shuffle(s)

    return splits


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def prepare_dataset(
    dataset_path: str | Path,
    data_cfg: DataConfig,
    seed: int = 42,
) -> Tuple[List[List[Dict[str, str]]], List[List[Dict[str, str]]], List[int], Dict[str, Any]]:
    """Convert DNRTI dataset to multi-task training conversations.

    Each document generates up to two training examples:
      - NER example (entity extraction)
      - RE example (relation extraction with entity hints)

    The task_mix_ratio controls the final NER/RE balance.
    Negative examples (empty outputs) are included when configured.

    Returns:
        (train_conversations, val_conversations, test_doc_indices, stats)
    """
    path = Path(dataset_path)
    logger.info("Loading DNRTI dataset from %s", path)

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    logger.info("Loaded %d documents", len(data))

    rng = random.Random(seed)

    # ---- Pass 1: extract gold data from every document ----
    # Store per-doc extracted info for splitting and conversation building
    doc_records: List[Dict[str, Any]] = []  # one per usable doc
    negative_pool: List[Tuple[int, str]] = []  # (doc_idx, text) for docs with 0 triples

    stats = {
        "total_docs": len(data),
        "skipped_docs": 0,
        "positive_docs": 0,
        "negative_pool_size": 0,
        "total_triples": 0,
        "total_entities": 0,
        "predicate_counts": Counter(),
        "entity_type_counts": Counter(),
    }

    for i, entry in enumerate(data):
        text = entry.get("text", "").strip()
        if not text:
            stats["skipped_docs"] += 1
            continue

        triples = _build_gold_triples(entry)
        entities = _build_gold_entities(entry)

        if len(triples) >= max(data_cfg.min_triples_per_doc, 1):
            # Positive document
            labels = _get_doc_labels(triples, entities)

            for t in triples:
                stats["predicate_counts"][t["predicate"]] += 1
                stats["entity_type_counts"][t["subject"]["type"]] += 1
                stats["entity_type_counts"][t["object"]["type"]] += 1
            stats["total_triples"] += len(triples)
            stats["total_entities"] += len(entities)

            doc_records.append({
                "doc_idx": i,
                "text": text,
                "triples": triples,
                "entities": entities,
                "labels": labels,
                "is_negative": False,
            })
            stats["positive_docs"] += 1

        elif data_cfg.include_negatives and len(triples) == 0:
            # Candidate negative (has text but no valid triples)
            negative_pool.append((i, text))

    stats["negative_pool_size"] = len(negative_pool)

    # Sample negatives
    if data_cfg.include_negatives and negative_pool:
        target_neg_count = int(
            len(doc_records) * data_cfg.negative_ratio / (1 - data_cfg.negative_ratio)
        )
        if len(negative_pool) > target_neg_count:
            sampled_negatives = rng.sample(negative_pool, target_neg_count)
        else:
            sampled_negatives = negative_pool

        for doc_idx, text in sampled_negatives:
            entities = _build_gold_entities(data[doc_idx])
            doc_records.append({
                "doc_idx": doc_idx,
                "text": text,
                "triples": [],
                "entities": entities,  # may have entities but no relations
                "labels": {"negative"},
                "is_negative": True,
            })

        logger.info(
            "Added %d negative examples (pool=%d, target=%d)",
            len(sampled_negatives), len(negative_pool), target_neg_count,
        )

    # ---- Pass 2: stratified split on documents ----
    assert data_cfg.val_ratio + data_cfg.test_ratio < 1.0
    train_ratio = 1.0 - data_cfg.val_ratio - data_cfg.test_ratio

    items_for_split = [
        (rec, rec["labels"], rec["doc_idx"])
        for rec in doc_records
    ]

    splits = _iterative_stratified_split(
        items_for_split,
        split_ratios=[train_ratio, data_cfg.val_ratio, data_cfg.test_ratio],
        seed=seed,
    )

    train_recs, val_recs, test_recs = splits

    # ---- Pass 3: generate conversations from split docs ----
    task = getattr(data_cfg, "task", "joint")
    mix = data_cfg.task_mix_ratio  # 0.85 = 85% NER (only used for "joint")

    def _generate_conversations(
        split_items: List[Tuple[Dict, int]],
    ) -> Tuple[List[List[Dict[str, str]]], List[int]]:
        conversations = []
        doc_indices = []

        for rec, doc_idx in split_items:
            text = rec["text"]
            triples = rec["triples"]
            entities = rec["entities"]

            if task == "ner":
                # NER-only: every doc becomes a NER example
                conversations.append(_build_ner_conversation(text, entities))
            elif task == "re":
                # RE-only: every doc becomes a RE example
                conversations.append(_build_re_conversation(text, entities, triples))
            else:
                # Joint: probabilistic mix based on task_mix_ratio
                r = rng.random()
                if r < mix:
                    conversations.append(_build_ner_conversation(text, entities))
                else:
                    conversations.append(_build_re_conversation(text, entities, triples))

            doc_indices.append(doc_idx)

        return conversations, doc_indices

    train_convos, train_indices = _generate_conversations(train_recs)
    val_convos, val_indices = _generate_conversations(val_recs)
    test_indices = [doc_idx for _, doc_idx in test_recs]

    # ---- Stats ----
    stats["train_size"] = len(train_convos)
    stats["val_size"] = len(val_convos)
    stats["test_size"] = len(test_indices)
    stats["task"] = task
    stats["task_mix_ratio"] = mix if task == "joint" else (1.0 if task == "ner" else 0.0)
    stats["predicate_counts"] = dict(stats["predicate_counts"])
    stats["entity_type_counts"] = dict(stats["entity_type_counts"])
    stats["train_indices"] = train_indices
    stats["val_indices"] = val_indices
    stats["test_indices"] = test_indices

    # Count task types in train set for logging
    if task == "ner":
        ner_count, re_count = len(train_convos), 0
    elif task == "re":
        ner_count, re_count = 0, len(train_convos)
    else:
        ner_count = int(len(train_convos) * mix)
        re_count = len(train_convos) - ner_count
    logger.info(
        "Prepared %d train (%d NER, %d RE) [task=%s], %d val, %d test "
        "(%d triples, %d entities, %d docs skipped)",
        len(train_convos), ner_count, re_count, task,
        len(val_convos), len(test_indices),
        stats["total_triples"], stats["total_entities"],
        stats["skipped_docs"],
    )

    return train_convos, val_convos, test_indices, stats


# =============================================================================
# UTILITIES
# =============================================================================

def regenerate_from_manifest(
    dataset_path: str | Path,
    manifest_path: str | Path,
    data_cfg: DataConfig,
    seed: int = 42,
) -> Tuple[List[List[Dict[str, str]]], List[List[Dict[str, str]]], List[int], Dict[str, Any]]:
    """Regenerate conversations from an existing split manifest.

    Reuses the exact same train/val/test doc indices from a previous run,
    but rebuilds conversations with the current task setting (ner/re/joint).
    This ensures the split is identical across task-specific training runs.

    Returns:
        (train_conversations, val_conversations, test_doc_indices, stats)
    """
    path = Path(dataset_path)
    logger.info("Loading DNRTI dataset from %s", path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    logger.info("Loading split manifest from %s", manifest_path)
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    train_indices = manifest["train_indices"]
    val_indices = manifest["val_indices"]
    test_indices = manifest["test_indices"]
    logger.info(
        "Reusing split: %d train, %d val, %d test",
        len(train_indices), len(val_indices), len(test_indices),
    )

    rng = random.Random(seed)
    task = getattr(data_cfg, "task", "joint")
    mix = data_cfg.task_mix_ratio

    def _build_conversations_from_indices(
        indices: List[int],
    ) -> List[List[Dict[str, str]]]:
        conversations = []
        for doc_idx in indices:
            entry = data[doc_idx]
            text = entry.get("text", "").strip()
            if not text:
                continue
            triples = _build_gold_triples(entry)
            entities = _build_gold_entities(entry)

            if task == "ner":
                conversations.append(_build_ner_conversation(text, entities))
            elif task == "re":
                conversations.append(_build_re_conversation(text, entities, triples))
            else:
                r = rng.random()
                if r < mix:
                    conversations.append(_build_ner_conversation(text, entities))
                else:
                    conversations.append(_build_re_conversation(text, entities, triples))
        return conversations

    train_convos = _build_conversations_from_indices(train_indices)
    val_convos = _build_conversations_from_indices(val_indices)

    # Count task types
    if task == "ner":
        ner_count, re_count = len(train_convos), 0
    elif task == "re":
        ner_count, re_count = 0, len(train_convos)
    else:
        ner_count = int(len(train_convos) * mix)
        re_count = len(train_convos) - ner_count

    stats = {
        "train_size": len(train_convos),
        "val_size": len(val_convos),
        "test_size": len(test_indices),
        "task": task,
        "task_mix_ratio": mix if task == "joint" else (1.0 if task == "ner" else 0.0),
        "reused_manifest": str(manifest_path),
    }

    logger.info(
        "Regenerated %d train (%d NER, %d RE) [task=%s], %d val, %d test",
        len(train_convos), ner_count, re_count, task,
        len(val_convos), len(test_indices),
    )

    return train_convos, val_convos, test_indices, stats


def export_test_set_dnrti(
    dataset_path: str | Path,
    test_indices: List[int],
    output_path: str | Path,
) -> Path:
    """Export test split as a standalone DNRTI-format JSON file."""
    path = Path(dataset_path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    test_docs = [data[i] for i in test_indices]

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(test_docs, f, ensure_ascii=False)

    logger.info("Exported %d test docs (DNRTI format) to %s", len(test_docs), out)
    return out


def save_dataset(
    conversations: List[List[Dict[str, str]]],
    output_path: str | Path,
) -> Path:
    """Save conversations as JSONL for caching/inspection."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for convo in conversations:
            f.write(json.dumps(convo, ensure_ascii=False) + "\n")

    logger.info("Saved %d conversations to %s", len(conversations), path)
    return path
