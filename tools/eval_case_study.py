"""Document-level ranked alignment eval for CrashOverride case study.

For each retrieval method, produces a single ranked list of ATT&CK IDs
for the entire document, then evaluates R@k and MRR against ground truth.
"""
import json
import math
from pathlib import Path
from neo4j import GraphDatabase


def extract_id(uri):
    if uri and "#" in uri:
        return uri.split("#")[-1]
    return None


def rollup(ids):
    expanded = set(ids)
    for i in ids:
        if "." in i:
            expanded.add(i.split(".")[0])
    return expanded


def recall_at_k(ranked_ids, gt, k):
    """Recall@k: fraction of GT found in top-k of ranked list."""
    top_k = set(ranked_ids[:k])
    return len(top_k & gt) / len(gt) if gt else 0


def mrr(ranked_ids, gt):
    """Mean Reciprocal Rank across all GT IDs."""
    if not gt:
        return 0
    rr_sum = 0
    for gt_id in gt:
        for rank, rid in enumerate(ranked_ids):
            if rid == gt_id:
                rr_sum += 1.0 / (rank + 1)
                break
    return rr_sum / len(gt)


TTP_TYPES = ["MAL", "TOOL", "ACT", "APT", "VULID", "VULNAME"]


def eval_config(driver, db_name, label, gt, gt_r):
    """Evaluate one pipeline config, returning ranked lists per method."""
    with driver.session(database=db_name) as s:
        doc = s.run(
            "MATCH (d:CTIDocument) WHERE d.id CONTAINS $p RETURN d.id AS id",
            p="CrashOverride",
        ).single()
        if not doc:
            return None
        doc_id = doc["id"]

        # Get all TTP-relevant entities with embeddings
        ent_result = s.run(
            "MATCH (d:CTIDocument {id: $did})-[:MENTIONS]->(e:CTIEntity) "
            "WHERE e.embedding IS NOT NULL AND e.type IN $types "
            "RETURN DISTINCT e.name AS name, e.type AS type, e.embedding AS emb",
            did=doc_id,
            types=TTP_TYPES,
        )
        entities = [(r["name"], r["type"], r["emb"]) for r in ent_result]

        # Get all sentence embeddings
        sent_result = s.run(
            "MATCH (d:CTIDocument {id: $did})-[:CONTAINS]->(sent:CTISentence) "
            "WHERE sent.embedding IS NOT NULL "
            "RETURN sent.id AS sid, sent.embedding AS emb",
            did=doc_id,
        )
        sentences = [(r["sid"], r["emb"]) for r in sent_result]

    # --- 1. Entity-neighbors vector ---
    # Per entity top-10, track max score and hit count per ATT&CK ID
    # final_score = max_score * (1 + log(count))
    vec_max = {}   # cid -> best single score
    vec_count = {} # cid -> number of distinct entities that retrieved it
    for name, etype, emb in entities:
        seen_this_entity = set()
        with driver.session(database=db_name) as s:
            vr = s.run(
                'CALL db.index.vector.queryNodes("attack_vec", 10, $emb) '
                "YIELD node, score WHERE node.uri IS NOT NULL "
                "RETURN node.uri AS uri, score",
                emb=emb,
            )
            for r in vr:
                cid = extract_id(r["uri"])
                if cid:
                    vec_max[cid] = max(vec_max.get(cid, 0), r["score"])
                    if cid not in seen_this_entity:
                        vec_count[cid] = vec_count.get(cid, 0) + 1
                        seen_this_entity.add(cid)

    vec_scores = {
        cid: vec_max[cid] * (1 + math.log(vec_count[cid]))
        for cid in vec_max
    }
    vec_ranked = [cid for cid, _ in sorted(vec_scores.items(), key=lambda x: -x[1])]

    # --- 2. BM25 ---
    bm25_max = {}   # cid -> best single score
    bm25_count = {} # cid -> number of distinct entity names that retrieved it
    entity_names = list(set(n for n, _, _ in entities))
    for name in entity_names:
        cleaned = (
            name.replace("&", "")
            .replace("/", "  ")
            .replace("(", "")
            .replace(")", "")
            .replace('"', "")
            .replace("-", " ")
            .strip()
        )
        if not cleaned or len(cleaned) < 3:
            continue
        seen_this_name = set()
        try:
            with driver.session(database=db_name) as s:
                br = s.run(
                    'CALL db.index.fulltext.queryNodes("attackFulltext", $q) '
                    "YIELD node, score WHERE node.uri IS NOT NULL AND score > 2.0 "
                    "RETURN node.uri AS uri, score ORDER BY score DESC LIMIT 10",
                    q=cleaned,
                )
                for r in br:
                    cid = extract_id(r["uri"])
                    if cid:
                        bm25_max[cid] = max(bm25_max.get(cid, 0), r["score"])
                        if cid not in seen_this_name:
                            bm25_count[cid] = bm25_count.get(cid, 0) + 1
                            seen_this_name.add(cid)
        except Exception:
            pass

    bm25_scores = {
        cid: bm25_max[cid] * (1 + math.log(bm25_count[cid]))
        for cid in bm25_max
    }
    bm25_ranked = [cid for cid, _ in sorted(bm25_scores.items(), key=lambda x: -x[1])]

    # --- 3. Sentence vector ---
    sent_max = {}   # cid -> best single score
    sent_count = {} # cid -> number of distinct sentences that retrieved it
    for sid, emb in sentences:
        seen_this_sent = set()
        with driver.session(database=db_name) as s:
            sr = s.run(
                'CALL db.index.vector.queryNodes("attack_vec", 10, $emb) '
                "YIELD node, score WHERE node.uri IS NOT NULL "
                "RETURN node.uri AS uri, score",
                emb=emb,
            )
            for r in sr:
                cid = extract_id(r["uri"])
                if cid:
                    sent_max[cid] = max(sent_max.get(cid, 0), r["score"])
                    if cid not in seen_this_sent:
                        sent_count[cid] = sent_count.get(cid, 0) + 1
                        seen_this_sent.add(cid)

    sent_scores = {
        cid: sent_max[cid] * (1 + math.log(sent_count[cid]))
        for cid in sent_max
    }
    sent_ranked = [cid for cid, _ in sorted(sent_scores.items(), key=lambda x: -x[1])]

    # --- 4. Hybrid RRF (k=60) ---
    # Build per-method rank maps
    vec_rank = {cid: i + 1 for i, cid in enumerate(vec_ranked)}
    bm25_rank = {cid: i + 1 for i, cid in enumerate(bm25_ranked)}
    sent_rank = {cid: i + 1 for i, cid in enumerate(sent_ranked)}

    all_cids = set(vec_rank) | set(bm25_rank) | set(sent_rank)
    rrf_k = 60
    rrf_scores = {}
    for cid in all_cids:
        score = 0
        if cid in vec_rank:
            score += 1.0 / (rrf_k + vec_rank[cid])
        if cid in bm25_rank:
            score += 1.0 / (rrf_k + bm25_rank[cid])
        if cid in sent_rank:
            score += 1.0 / (rrf_k + sent_rank[cid])
        rrf_scores[cid] = score

    rrf_ranked = [cid for cid, _ in sorted(rrf_scores.items(), key=lambda x: -x[1])]

    return {
        "Entity-Vec": vec_ranked,
        "BM25": bm25_ranked,
        "Sentence-Vec": sent_ranked,
        "Hybrid RRF": rrf_ranked,
    }


def main():
    # Ground truth
    data = json.loads(
        Path("datasets/CTI-HAL/data/sandworm/CrashOverride.json").read_text(
            encoding="utf-8"
        )
    )
    techniques, tactics, software = set(), set(), set()
    for sent in data:
        meta = sent.get("metadata", {}) or {}
        tech = sent.get("technique")
        if tech:
            techniques.add(tech)
        sub = meta.get("sub_technique")
        if sub:
            techniques.add(sub)
        for t in meta.get("tactic") or []:
            if t:
                tactics.add(t)
        for s_item in meta.get("tool") or []:
            if s_item:
                software.add(s_item)

    gt = techniques | tactics | software
    gt_r = rollup(gt)

    print(f"Ground truth: {len(gt)} IDs ({len(gt_r)} with rollup)")
    print(f"  Techniques: {sorted(techniques)}")
    print(f"  Tactics: {sorted(tactics)}")
    print(f"  Software: {sorted(software)}")

    driver = GraphDatabase.driver(
        "bolt://localhost:7687", auth=("neo4j", "abcd90909090")
    )

    configs = [
        ("neo4j", "Raw"),
        ("chunked", "Chunked"),
        ("canonicalized", "Canon"),
        ("chunkedcanon", "Chunk+Canon"),
    ]

    ks = [5, 10, 20]

    # Header
    print(f"\n{'Config':<14} {'Method':<14}", end="")
    for k in ks:
        print(f" {'R@'+str(k):>6} {'R@'+str(k)+'+':>6}", end="")
    print(f" {'MRR':>6} {'MRR+':>6} {'Pool':>5}")
    print("-" * 100)

    for db_name, label in configs:
        results = eval_config(driver, db_name, label, gt, gt_r)
        if not results:
            print(f"{label:<14} NO DOCUMENT FOUND")
            continue

        for method_name, ranked in results.items():
            ranked_r = list(dict.fromkeys(rollup(set(ranked))))
            # Re-rank rollup: keep original order but expand
            # Better: for rollup, insert parent right after child
            ranked_rollup = []
            seen = set()
            for cid in ranked:
                if cid not in seen:
                    ranked_rollup.append(cid)
                    seen.add(cid)
                if "." in cid:
                    parent = cid.split(".")[0]
                    if parent not in seen:
                        ranked_rollup.append(parent)
                        seen.add(parent)

            print(f"{label:<14} {method_name:<14}", end="")
            for k in ks:
                r_at_k = recall_at_k(ranked, gt, k)
                r_at_k_plus = recall_at_k(ranked_rollup, gt_r, k)
                print(f" {r_at_k:>6.3f} {r_at_k_plus:>6.3f}", end="")
            m = mrr(ranked, gt)
            m_r = mrr(ranked_rollup, gt_r)
            pool = len(set(ranked) & gt) / len(gt)
            print(f" {m:>6.3f} {m_r:>6.3f} {pool:>5.3f}")

        print()

    driver.close()


if __name__ == "__main__":
    main()
