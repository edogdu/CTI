"""Paired significance tests for the JISA 2026 graph-quality and retrieval claims.

Tests are paired across four pipeline configurations measured on the same
CTI-HAL documents (N=69: the 71 processed PDFs minus the two lacking ATT&CK
annotations). Wilcoxon signed-rank (Pratt zeros, normal approximation) is the
primary test; a sign-flip permutation test and exact sign test are reported as
robustness checks. Paired BCa bootstrap CIs (fallback: percentile) accompany
corpus-ratio and per-document effect estimates. The clustering-coefficient
"stability" claim is tested via TOST equivalence with a pre-specified margin.

Inputs (read-only):
  experiments/<config>/eval/graph_metrics/quality.json   (per_document)
  results/significance/per_doc_recall/<config>.json      (from eval_reranked.py --per-doc-output)

Outputs (results/significance/):
  per_doc_graph_metrics.csv, per_doc_recall.csv, per_doc_ranked_ids.json,
  paired_tests.csv, equivalence_tests.csv, run_manifest.json

Usage:
    python tools/significance_tests.py
"""
import argparse
import csv
import hashlib
import json
import subprocess
import sys
import re
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy
from scipy.stats import (binomtest, bootstrap, permutation_test, rankdata,
                         wilcoxon)

REPO = Path(__file__).parent.parent

SEED = 20260812
BOOT_B = 10_000
PERM_B = 20_000
EQUIV_DELTA = 0.05  # |CC shift| margin: 10% of the Raw baseline mean (0.525)

CONFIGS = {
    "raw": "ctihal-pipeline",
    "raw_canon": "ctihal-markov",
    "chunked": "ctihal-chunked",
    "chunked_canon": "ctihal-chunked-markov",
}

# (family, baseline, treatment). Families = one asserted claim each.
CONTRASTS = [
    ("F1_chunking", "raw", "chunked"),
    ("F2_canonicalization", "raw", "raw_canon"),
    ("F3_combined", "raw", "chunked_canon"),
]

GRAPH_METRICS = ["n_edges", "n_nodes", "density", "n_components",
                 "largest_component_ratio"]
EQUIV_METRIC = "clustering_coefficient"

RETRIEVAL_METHODS = ["Entity-neighbors vector", "Hybrid RRF"]
RETRIEVAL_RANKING = "max_score"
RETRIEVAL_K = "100"

# The two processed PDFs with no ATT&CK annotation file (graph N=71 -> 69).
UNANNOTATED = {"sandworm_CarbonBlack", "fin6_FIN6-SecurityIntelligence_9"}

# Published aggregates (main.tex Table 4 / quality.json), reproduction gate.
PUBLISHED_71 = {
    "raw": {"density": 0.017135, "components": 12.7, "lcr": 0.5846, "cc": 0.5252, "edges": 19630},
    "raw_canon": {"density": 0.019145, "components": 9.35, "lcr": 0.6803, "cc": 0.5236, "edges": 18301},
    "chunked": {"density": 0.028952, "components": 7.28, "lcr": 0.7111, "cc": 0.5428, "edges": 29711},
    "chunked_canon": {"density": 0.032366, "components": 4.79, "lcr": 0.8131, "cc": 0.5508, "edges": 27975},
}


def sanitize(s: str) -> str:
    """Canonical doc_id sanitizer (mirrors run_ctihal_pipeline.py)."""
    s = re.sub(r"[^\w\-.]", "_", s)
    return re.sub(r"_+", "_", s).strip("_")


def rng_for(key: str) -> np.random.Generator:
    """Order-invariant per-test RNG: stable under adding/reordering tests."""
    h = int.from_bytes(hashlib.blake2b(key.encode(), digest_size=8).digest(), "big")
    return np.random.default_rng([SEED, h])


# --------------------------------------------------------------------------
# Data layer
# --------------------------------------------------------------------------

def load_graph_metrics(config: str) -> dict:
    """{sanitized doc_id: {metric: value}} from quality.json per_document."""
    p = REPO / "experiments" / CONFIGS[config] / "eval" / "graph_metrics" / "quality.json"
    per_doc = json.loads(p.read_text(encoding="utf-8"))["per_document"]
    out = {}
    for raw_id, m in per_doc.items():
        doc = sanitize(raw_id)
        assert doc not in out, f"sanitize() collision in {config}: {doc}"
        out[doc] = {
            "n_edges": m["n_edges"],
            "n_nodes": m["n_nodes"],
            "density": m["density"],
            "n_components": m["components"]["n_components"],
            "largest_component_ratio": m["components"]["largest_component_ratio"],
            "clustering_coefficient": m["clustering_coefficient"],
        }
    return out


def load_recall(config: str, per_doc_dir: Path) -> dict:
    """{method: {sanitized doc_id: {"recall": float, "n_gt": int, ...}}}."""
    p = per_doc_dir / f"{config}.json"
    data = json.loads(p.read_text(encoding="utf-8"))
    out = {}
    for method, strategies in data["methods"].items():
        docs = {}
        for raw_id, rec in strategies[RETRIEVAL_RANKING].items():
            doc = sanitize(raw_id)
            assert doc not in docs, f"sanitize() collision in {config}/{method}: {doc}"
            docs[doc] = rec
        out[method] = docs
    return out, data


def align(a: dict, b: dict, keys=None):
    """Inner join two {doc: value} dicts; returns (x, y, sorted_docs)."""
    docs = sorted(set(a) & set(b) if keys is None else (set(a) & set(b) & set(keys)))
    assert docs, "empty join"
    return docs


# --------------------------------------------------------------------------
# Inference primitives
# --------------------------------------------------------------------------

def wilcoxon_paired(x: np.ndarray, y: np.ndarray) -> dict:
    d = y - x
    n_zero = int((d == 0).sum())
    if np.all(d == 0):
        return {"p": 1.0, "W": 0.0, "n": len(d), "n_zero": n_zero,
                "n_pos": 0, "n_neg": 0}
    res = wilcoxon(y, x, zero_method="pratt", correction=True,
                   alternative="two-sided", method="approx")
    return {"p": float(res.pvalue), "W": float(res.statistic), "n": len(d),
            "n_zero": n_zero, "n_pos": int((d > 0).sum()),
            "n_neg": int((d < 0).sum())}


def signflip_perm(x: np.ndarray, y: np.ndarray, rng) -> float:
    d = y - x
    if np.all(d == 0):
        return 1.0
    res = permutation_test((d,), lambda s, axis=-1: np.mean(s, axis=axis),
                           permutation_type="samples",
                           alternative="two-sided",
                           n_resamples=PERM_B, rng=rng)
    return float(res.pvalue)


def sign_test(x: np.ndarray, y: np.ndarray) -> float:
    d = y - x
    n_pos, n_neg = int((d > 0).sum()), int((d < 0).sum())
    if n_pos + n_neg == 0:
        return 1.0
    return float(binomtest(n_pos, n_pos + n_neg, 0.5).pvalue)


def rank_biserial(x: np.ndarray, y: np.ndarray) -> float:
    """Matched-pairs rank-biserial from Pratt-ranked signed ranks."""
    d = y - x
    r = rankdata(np.abs(d))
    nz = d != 0
    if not nz.any():
        return 0.0
    wp = r[d > 0].sum()
    wm = r[d < 0].sum()
    return float((wp - wm) / r[nz].sum())


def hodges_lehmann(d: np.ndarray) -> float:
    """Median of Walsh averages (i <= j)."""
    i, j = np.triu_indices(len(d))
    return float(np.median((d[i] + d[j]) / 2.0))


def cohens_dz(x: np.ndarray, y: np.ndarray) -> float:
    d = y - x
    sd = d.std(ddof=1)
    return float(d.mean() / sd) if sd > 0 else 0.0


def paired_ci(x, y, stat, rng, cl=0.95):
    """Paired document-resampling bootstrap CI. BCa; percentile fallback."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            r = bootstrap((x, y), stat, paired=True, vectorized=False,
                          n_resamples=BOOT_B, method="BCa",
                          confidence_level=cl, rng=rng)
            lo, hi = r.confidence_interval
            if np.isfinite(lo) and np.isfinite(hi):
                return float(lo), float(hi), "BCa"
        except Exception:
            pass
        r = bootstrap((x, y), stat, paired=True, vectorized=False,
                      n_resamples=BOOT_B, method="percentile",
                      confidence_level=cl, rng=rng)
        lo, hi = r.confidence_interval
        return float(lo), float(hi), "percentile"


def tost_wilcoxon(d: np.ndarray, delta: float):
    """Nonparametric TOST: two one-sided Wilcoxons on shifted differences."""
    p_lo = wilcoxon(d + delta, zero_method="pratt", correction=True,
                    alternative="greater", method="approx").pvalue
    p_hi = wilcoxon(d - delta, zero_method="pratt", correction=True,
                    alternative="less", method="approx").pvalue
    return float(max(p_lo, p_hi)), float(p_lo), float(p_hi)


def holm(pvals):
    """Holm step-down adjusted p-values (monotone, capped at 1)."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adj[idx] = min(running, 1.0)
    return adj.tolist()


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

def run_contrast(family, contrast, metric, x, y, n, extra=None):
    """One tidy row of paired superiority tests + effect sizes + CIs."""
    key = f"{family}|{contrast}|{metric}"
    d = y - x
    w = wilcoxon_paired(x, y)
    corpus_ratio = float(y.sum() / x.sum() - 1.0)
    cr_lo, cr_hi, cr_m = paired_ci(
        x, y, lambda a, b: b.sum() / a.sum() - 1.0, rng_for(key + "|corpus_ratio"))
    if np.all(x > 0):
        med_pct = float(np.median(y / x) - 1.0)
        mp_lo, mp_hi, mp_m = paired_ci(
            x, y, lambda a, b: float(np.median(b / a) - 1.0),
            rng_for(key + "|median_pct"))
    else:  # bounded-[0,1] recall can have zero baselines; ratio undefined
        med_pct = mp_lo = mp_hi = float("nan")
        mp_m = "n/a"
    hl = hodges_lehmann(d)
    hl_lo, hl_hi, hl_m = paired_ci(
        x, y, lambda a, b: hodges_lehmann(b - a), rng_for(key + "|hl"))
    row = {
        "family": family, "contrast": contrast, "metric": metric,
        "n": n, "n_zero": w["n_zero"], "n_pos": w["n_pos"], "n_neg": w["n_neg"],
        "baseline_mean": float(x.mean()), "treatment_mean": float(y.mean()),
        "baseline_median": float(np.median(x)), "treatment_median": float(np.median(y)),
        "corpus_ratio_pct": 100 * corpus_ratio,
        "corpus_ratio_ci_lo": 100 * cr_lo, "corpus_ratio_ci_hi": 100 * cr_hi,
        "median_pct_change": 100 * med_pct,
        "median_pct_change_ci_lo": 100 * mp_lo, "median_pct_change_ci_hi": 100 * mp_hi,
        "hl_shift": hl, "hl_ci_lo": hl_lo, "hl_ci_hi": hl_hi,
        "p_wilcoxon": w["p"],
        "p_permutation": signflip_perm(x, y, rng_for(key + "|perm")),
        "p_sign": sign_test(x, y),
        "r_rank_biserial": rank_biserial(x, y),
        "cohens_dz": cohens_dz(x, y),
        "ci_method": ";".join(sorted({cr_m, mp_m, hl_m})),
        "test_direction": "superiority",
        "zero_method": "pratt", "wilcoxon_method": "approx",
    }
    if extra:
        row.update(extra)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", default=str(REPO / "results" / "significance"))
    parser.add_argument("--per-doc-recall-dir",
                        default=str(REPO / "results" / "significance" / "per_doc_recall"))
    parser.add_argument("--equiv-delta", type=float, default=EQUIV_DELTA)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_doc_dir = Path(args.per_doc_recall_dir)

    # ---- Load graph metrics; join gate ------------------------------------
    graph = {c: load_graph_metrics(c) for c in CONFIGS}
    doc_sets = [set(g) for g in graph.values()]
    common = set.intersection(*doc_sets)
    assert all(len(s) == 71 for s in doc_sets), \
        f"expected 71 docs per config, got {[len(s) for s in doc_sets]}"
    assert len(common) == 71, f"join gate failed: {len(common)}/71 common docs"
    assert UNANNOTATED <= common, "expected unannotated doc_ids not found"
    docs69 = sorted(common - UNANNOTATED)
    assert len(docs69) == 69

    # ---- Reproduction gate: published Table 4 aggregates at N=71 ----------
    for c, pub in PUBLISHED_71.items():
        g = graph[c]
        n = len(g)
        checks = {
            "density": (round(sum(v["density"] for v in g.values()) / n, 6), pub["density"]),
            "components": (round(sum(v["n_components"] for v in g.values()) / n, 2), pub["components"]),
            "lcr": (round(sum(v["largest_component_ratio"] for v in g.values()) / n, 4), pub["lcr"]),
            "cc": (round(sum(v["clustering_coefficient"] for v in g.values()) / n, 4), pub["cc"]),
            "edges": (sum(v["n_edges"] for v in g.values()), pub["edges"]),
        }
        for name, (got, want) in checks.items():
            assert got == want, f"reproduction gate failed: {c}.{name} got {got}, want {want}"
    print("Reproduction gate PASSED: all four configs match published Table 4 at N=71")

    rows = []

    # ---- F1-F3: graph metrics at N=69 --------------------------------------
    for family, base, treat in CONTRASTS:
        contrast = f"{base}__vs__{treat}"
        for metric in GRAPH_METRICS:
            x = np.array([graph[base][d][metric] for d in docs69], dtype=float)
            y = np.array([graph[treat][d][metric] for d in docs69], dtype=float)
            rows.append(run_contrast(family, contrast, metric, x, y, len(docs69)))

    # ---- F4: clustering coefficient TOST equivalence -----------------------
    equiv_rows = []
    for family, base, treat in CONTRASTS:
        contrast = f"{base}__vs__{treat}"
        x = np.array([graph[base][d][EQUIV_METRIC] for d in docs69])
        y = np.array([graph[treat][d][EQUIV_METRIC] for d in docs69])
        d = y - x
        p_tost, p_lo, p_hi = tost_wilcoxon(d, args.equiv_delta)
        hl = hodges_lehmann(d)
        lo90, hi90, ci_m = paired_ci(
            x, y, lambda a, b: hodges_lehmann(b - a),
            rng_for(f"F4_cc_equivalence|{contrast}|{EQUIV_METRIC}|hl90"), cl=0.90)
        w = wilcoxon_paired(x, y)
        equiv_rows.append({
            "family": "F4_cc_equivalence", "contrast": contrast,
            "metric": EQUIV_METRIC, "n": len(docs69),
            "n_zero": w["n_zero"], "n_pos": w["n_pos"], "n_neg": w["n_neg"],
            "baseline_mean": float(x.mean()), "treatment_mean": float(y.mean()),
            "delta": args.equiv_delta,
            "hl_shift": hl, "ci90_lo": lo90, "ci90_hi": hi90, "ci_method": ci_m,
            "p_tost": p_tost, "p_tost_lower": p_lo, "p_tost_upper": p_hi,
            "p_wilcoxon_superiority": w["p"],
            "test_direction": "equivalence",
            "zero_method": "pratt", "wilcoxon_method": "approx",
        })

    # ---- F5: retrieval R@100, per-method pairwise intersection ------------
    recall = {}
    recall_payloads = {}
    for c in CONFIGS:
        recall[c], recall_payloads[c] = load_recall(c, per_doc_dir)
    for family, base, treat in CONTRASTS:
        contrast = f"{base}__vs__{treat}"
        for method in RETRIEVAL_METHODS:
            a, b = recall[base][method], recall[treat][method]
            docs = align(a, b)
            x = np.array([a[d]["recall"][RETRIEVAL_K] for d in docs])
            y = np.array([b[d]["recall"][RETRIEVAL_K] for d in docs])
            rows.append(run_contrast(
                "F5_retrieval", contrast, f"recall@{RETRIEVAL_K}:{method}",
                x, y, len(docs),
                extra={"ranking": RETRIEVAL_RANKING}))

    # ---- F6: method comparison (Hybrid RRF vs Entity-neighbors), per config
    # Backs the paper's bolded claim that Hybrid RRF achieves the best R@100.
    for c in CONFIGS:
        a = recall[c]["Entity-neighbors vector"]
        b = recall[c]["Hybrid RRF"]
        docs = align(a, b)
        x = np.array([a[d]["recall"][RETRIEVAL_K] for d in docs])
        y = np.array([b[d]["recall"][RETRIEVAL_K] for d in docs])
        rows.append(run_contrast(
            "F6_method_comparison", f"entity_vec__vs__hybrid_rrf@{c}",
            f"recall@{RETRIEVAL_K}", x, y, len(docs),
            extra={"ranking": RETRIEVAL_RANKING}))

    # ---- Corrections: Holm within each family ------------------------------
    for family in sorted({r["family"] for r in rows}):
        fam = [r for r in rows if r["family"] == family]
        adj = holm([r["p_wilcoxon"] for r in fam])
        for r, p in zip(fam, adj):
            r["p_adj"] = p
            r["adj_method"] = "holm"
    adj = holm([r["p_tost"] for r in equiv_rows])
    for r, p in zip(equiv_rows, adj):
        r["p_adj"] = p
        r["adj_method"] = "holm"
        r["equivalent"] = bool(p < 0.05)

    # ---- Sanity gate --------------------------------------------------------
    dens = next(r for r in rows if r["family"] == "F1_chunking" and r["metric"] == "density")
    assert dens["n_pos"] == 69 and dens["p_adj"] < 1e-6, "sanity gate: density Raw->Chunked"

    # ---- Outputs ------------------------------------------------------------
    def write_csv(path, rws):
        cols = sorted({k for r in rws for k in r}, key=lambda c: (c != "family", c != "contrast", c != "metric", c))
        with open(path, "w", newline="", encoding="utf-8") as f:
            wtr = csv.DictWriter(f, fieldnames=cols)
            wtr.writeheader()
            wtr.writerows(rws)

    write_csv(out_dir / "paired_tests.csv", rows)
    write_csv(out_dir / "equivalence_tests.csv", equiv_rows)

    # Collaborator file 1: per-document graph metrics, long format
    with open(out_dir / "per_doc_graph_metrics.csv", "w", newline="", encoding="utf-8") as f:
        wtr = csv.writer(f)
        wtr.writerow(["doc_id", "actor", "config", "annotated", "triple_count",
                      "node_count", "edge_count", "density", "n_components",
                      "lcr", "clustering_coeff"])
        for c in CONFIGS:
            for doc in sorted(graph[c]):
                m = graph[c][doc]
                wtr.writerow([doc, doc.split("_", 1)[0], c,
                              doc not in UNANNOTATED,
                              m["n_edges"], m["n_nodes"], m["n_edges"],
                              m["density"], m["n_components"],
                              m["largest_component_ratio"],
                              m["clustering_coefficient"]])

    # Collaborator file 2: per-document recall, long format (all methods/ks)
    with open(out_dir / "per_doc_recall.csv", "w", newline="", encoding="utf-8") as f:
        wtr = csv.writer(f)
        wtr.writerow(["doc_id", "actor", "config", "method", "ranking", "k",
                      "recall", "n_gt"])
        for c in CONFIGS:
            payload = recall_payloads[c]
            for method, strategies in payload["methods"].items():
                for ranking, docs_ in strategies.items():
                    for raw_id, rec in docs_.items():
                        doc = sanitize(raw_id)
                        for k, v in rec["recall"].items():
                            wtr.writerow([doc, doc.split("_", 1)[0], c, method,
                                          ranking, k, v, rec["n_gt"]])

    # Bonus: raw ranked IDs vs gold, for arbitrary cutoffs
    ranked_ids = {}
    for c in CONFIGS:
        payload = recall_payloads[c]
        ranked_ids[c] = {
            "ground_truth": {sanitize(d): g for d, g in payload["ground_truth"].items()},
            "ranked_ids": {
                method: {sanitize(d): rec["ranked_ids"]
                         for d, rec in strategies[RETRIEVAL_RANKING].items()}
                for method, strategies in payload["methods"].items()
            },
        }
    (out_dir / "per_doc_ranked_ids.json").write_text(
        json.dumps(ranked_ids), encoding="utf-8")

    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                             capture_output=True, text=True).stdout.strip()
    except Exception:
        sha = "unknown"
    manifest = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "seed": SEED, "bootstrap_B": BOOT_B, "permutation_B": PERM_B,
        "equiv_delta": args.equiv_delta,
        "scipy": scipy.__version__, "numpy": np.__version__,
        "python": sys.version.split()[0], "git_sha": sha,
        "configs": CONFIGS, "contrasts": CONTRASTS,
        "retrieval": {"methods": RETRIEVAL_METHODS, "ranking": RETRIEVAL_RANKING,
                      "k": RETRIEVAL_K},
        "n_graph": len(docs69), "excluded_unannotated": sorted(UNANNOTATED),
        "doc_ids": docs69,
        "retrieval_n": {r["contrast"] + "|" + r["metric"]: r["n"]
                        for r in rows if r["family"] == "F5_retrieval"},
        "cli": sys.argv,
    }
    (out_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")

    # ---- Console summary ----------------------------------------------------
    print(f"\n{'family':<22s} {'contrast':<28s} {'metric':<38s} "
          f"{'n':>3s} {'ratio%':>8s} {'HL':>9s} {'r_rb':>6s} {'p_adj':>10s}")
    for r in rows:
        print(f"{r['family']:<22s} {r['contrast']:<28s} {r['metric']:<38s} "
              f"{r['n']:3d} {r['corpus_ratio_pct']:8.1f} {r['hl_shift']:9.4f} "
              f"{r['r_rank_biserial']:6.2f} {r['p_adj']:10.2e}")
    print()
    for r in equiv_rows:
        print(f"{r['family']:<22s} {r['contrast']:<28s} TOST delta={r['delta']} "
              f"HL={r['hl_shift']:+.4f} CI90=[{r['ci90_lo']:+.4f},{r['ci90_hi']:+.4f}] "
              f"p_adj={r['p_adj']:.4f} equivalent={r['equivalent']}")
    print(f"\nWrote outputs to {out_dir}")


if __name__ == "__main__":
    main()
