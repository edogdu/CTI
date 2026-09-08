"""Evaluate structural quality of extracted knowledge graphs.

Computes three core metrics per document:
1. Graph density: |E| / (|V| * (|V| - 1))
2. Connected components: count + largest component ratio
3. Triple connectivity: fraction of triples with connected endpoints

Can read from Neo4j or from triple JSON files in experiment folders.

Usage:
    python tools/eval_graph_quality.py                           # all docs in Neo4j
    python tools/eval_graph_quality.py --doc-id apt29_Analysis...  # single doc
    python tools/eval_graph_quality.py --from-json experiments/ctihal-pipeline/eval/triples/
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

_repo = Path(__file__).parent.parent
sys.path.insert(0, str(_repo / "src"))


def build_graph_from_triples(triples: List[Dict]) -> Tuple[Set[str], List[Tuple[str, str, str]]]:
    """Extract nodes and edges from triple dicts.

    Returns (nodes, edges) where nodes are entity identifiers and
    edges are (subject, predicate, object) tuples.
    """
    nodes = set()
    edges = []
    for t in triples:
        if isinstance(t.get("subject"), dict):
            subj = f"{t['subject']['name']}|{t['subject'].get('type', '')}"
        else:
            subj = str(t.get("subject", ""))
        if isinstance(t.get("object"), dict):
            obj = f"{t['object']['name']}|{t['object'].get('type', '')}"
        else:
            obj = str(t.get("object", ""))
        pred = t.get("predicate", "")
        if subj and obj:
            nodes.add(subj)
            nodes.add(obj)
            edges.append((subj, pred, obj))
    return nodes, edges


def compute_density(n_nodes: int, n_edges: int) -> float:
    """Graph density: |E| / (|V| * (|V| - 1)). Returns 0 for < 2 nodes."""
    if n_nodes < 2:
        return 0.0
    return n_edges / (n_nodes * (n_nodes - 1))


def compute_connected_components(nodes: Set[str], edges: List[Tuple[str, str, str]]) -> Dict:
    """Compute connected components using union-find."""
    parent = {n: n for n in nodes}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for subj, _, obj in edges:
        union(subj, obj)

    components = defaultdict(set)
    for n in nodes:
        components[find(n)].add(n)

    comp_sizes = sorted([len(c) for c in components.values()], reverse=True)
    n_components = len(comp_sizes)
    largest = comp_sizes[0] if comp_sizes else 0
    largest_ratio = largest / len(nodes) if nodes else 0

    return {
        "n_components": n_components,
        "largest_component_size": largest,
        "largest_component_ratio": round(largest_ratio, 4),
        "component_sizes": comp_sizes,
    }


def compute_clustering_coefficient(nodes: Set[str], edges: List[Tuple[str, str, str]]) -> float:
    """Average clustering coefficient (undirected projection).

    For each node, measures the fraction of its neighbors that are also
    connected to each other. Averaged across all nodes with degree >= 2.
    Ref: Watts & Strogatz (1998).
    """
    if not nodes or not edges:
        return 0.0

    # Build undirected adjacency
    neighbors = defaultdict(set)
    for subj, _, obj in edges:
        neighbors[subj].add(obj)
        neighbors[obj].add(subj)

    cc_sum = 0.0
    cc_count = 0
    for node in nodes:
        nb = neighbors[node]
        k = len(nb)
        if k < 2:
            continue
        # Count edges among neighbors
        links = 0
        nb_list = list(nb)
        for i in range(len(nb_list)):
            for j in range(i + 1, len(nb_list)):
                if nb_list[j] in neighbors[nb_list[i]]:
                    links += 1
        cc_local = (2.0 * links) / (k * (k - 1))
        cc_sum += cc_local
        cc_count += 1

    return round(cc_sum / cc_count, 4) if cc_count > 0 else 0.0


def evaluate_document(triples: List[Dict]) -> Dict:
    """Compute all metrics for a document's triples."""
    nodes, edges = build_graph_from_triples(triples)
    n_nodes = len(nodes)
    n_edges = len(edges)

    density = compute_density(n_nodes, n_edges)
    components = compute_connected_components(nodes, edges)
    clustering = compute_clustering_coefficient(nodes, edges)

    return {
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "density": round(density, 6),
        "components": components,
        "clustering_coefficient": clustering,
    }


def load_triples_from_neo4j(driver, doc_id: str) -> List[Dict]:
    """Load triples for a document from Neo4j."""
    with driver.session() as session:
        result = session.run(
            """
            MATCH (d:CTIDocument {id: $doc_id})-[:MENTIONS]->(subj:CTIEntity)
            MATCH (subj)-[r]->(obj:CTIEntity)
            WHERE (d)-[:MENTIONS]->(obj)
            RETURN subj.name AS subj_name, subj.type AS subj_type,
                   type(r) AS predicate,
                   obj.name AS obj_name, obj.type AS obj_type
            """,
            doc_id=doc_id,
        )
        return [
            {
                "subject": {"name": rec["subj_name"], "type": rec["subj_type"]},
                "predicate": rec["predicate"],
                "object": {"name": rec["obj_name"], "type": rec["obj_type"]},
            }
            for rec in result
        ]


def load_triples_from_json(json_path: Path) -> List[Dict]:
    """Load triples from a per-document JSON file."""
    data = json.loads(json_path.read_text(encoding="utf-8"))
    return data.get("triples", [])


def main():
    parser = argparse.ArgumentParser(description="Evaluate knowledge graph structural quality")
    parser.add_argument("--doc-id", default=None, help="Evaluate single document (Neo4j)")
    parser.add_argument("--from-json", default=None, help="Load triples from experiment JSON dir")
    parser.add_argument("--output", default=None, help="Save results JSON")
    args = parser.parse_args()

    results = {}

    if args.from_json:
        # Load from experiment folder
        json_dir = Path(args.from_json)
        for json_file in sorted(json_dir.glob("*.json")):
            doc_id = json_file.stem
            triples = load_triples_from_json(json_file)
            if triples:
                results[doc_id] = evaluate_document(triples)
    else:
        # Load from Neo4j
        from neo4j import GraphDatabase
        from cti_analysis.config import load_config

        cfg = load_config()
        driver = GraphDatabase.driver(cfg.neo4j.uri, auth=(cfg.neo4j.user, cfg.neo4j.password))

        if args.doc_id:
            doc_ids = [args.doc_id]
        else:
            with driver.session() as s:
                r = s.run("MATCH (d:CTIDocument) RETURN d.id AS id ORDER BY d.id")
                doc_ids = [rec["id"] for rec in r]

        for doc_id in doc_ids:
            triples = load_triples_from_neo4j(driver, doc_id)
            if triples:
                results[doc_id] = evaluate_document(triples)

        driver.close()

    if not results:
        print("No documents with triples found.")
        return

    # Print summary
    print(f"Graph Quality Metrics ({len(results)} documents)")
    print("=" * 95)
    print(f"{'Document':<55s} {'Nodes':>6s} {'Edges':>6s} {'Density':>8s} {'Comp':>5s} {'LCR':>6s} {'CC':>6s}")
    print("-" * 95)

    all_density = []
    all_components = []
    all_lcr = []
    all_clustering = []

    for doc_id, m in sorted(results.items()):
        short_id = doc_id[:54]
        comp = m["components"]
        cc = m["clustering_coefficient"]
        print(f"{short_id:<55s} {m['n_nodes']:6d} {m['n_edges']:6d} "
              f"{m['density']:8.4f} {comp['n_components']:5d} "
              f"{comp['largest_component_ratio']:6.2f} {cc:6.3f}")

        all_density.append(m["density"])
        all_components.append(comp["n_components"])
        all_lcr.append(comp["largest_component_ratio"])
        all_clustering.append(cc)

    # Aggregates
    n = len(results)
    print("-" * 95)
    print(f"{'MEAN':<55s} {'':>6s} {'':>6s} "
          f"{sum(all_density)/n:8.4f} {sum(all_components)/n:5.1f} "
          f"{sum(all_lcr)/n:6.2f} {sum(all_clustering)/n:6.3f}")

    if args.output:
        out = {
            "n_documents": n,
            "per_document": results,
            "aggregates": {
                "mean_density": round(sum(all_density) / n, 6),
                "mean_components": round(sum(all_components) / n, 2),
                "mean_largest_component_ratio": round(sum(all_lcr) / n, 4),
                "mean_clustering_coefficient": round(sum(all_clustering) / n, 4),
            },
        }
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
