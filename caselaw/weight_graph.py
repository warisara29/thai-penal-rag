"""B — court-weighted provision graph.

Weights the rule-derived APPLIES_TO edges (data/applies_to_edges.jsonl, 13,296 edges) by how
often their two endpoints are actually cited together in Supreme Court judgments, so edges that
real courts never use (much of the auto-materialized ``scope=all_offences`` set) can be pruned.
CITES edges are left untouched (task B). Leakage control: co-citations are counted only over the
GRAPH set (case year ≤ Y0); cases with year > Y0 are reserved for the held-out test set (task D).

Weight (normalized co-citation, per the locked decision):

    w(u,v) = c(u,v) / min(c(u), c(v))   if c(u,v) ≥ min_support, else 0

where c(x) = #graph-cases citing section x, c(u,v) = #graph-cases citing both. w ∈ [0,1];
w = 1 means "whenever the rarer section appears, the other does too". A support floor
(``min_support``) is required because pure normalization otherwise gives w = 1.0 to pairs
co-cited a single time (c_uv = 1) and ranks those noise edges above high-frequency, legally
meaningful ones (e.g. 69→288 cited together 211×). Edges with w ≥ τ are kept.

Output ``data/caselaw/applies_to_weighted.jsonl`` = every APPLIES_TO edge + {w, c_u, c_v, c_uv},
consumed by the extended retrieval.kg_expand.KGExpander (τ filter + ablation revert live there).

Usage:
    python -m caselaw.weight_graph --y0-ce 2019
"""

from __future__ import annotations

import argparse
import itertools
import json
from collections import Counter, defaultdict

# Dika reports begin ~BE 2411; the corpus has a single implausible year. Guard the graph set to
# a sane CE window so OCR-garbage years (e.g. CE 1006) don't slip into the "≤ Y0" side.
_CE_MIN, _CE_MAX = 1900, 2025


def _sid(node: str) -> str:
    return node.split(":", 1)[1] if node.startswith("Section:") else node


def build_counts(mapped_path: str, y0_ce: int) -> tuple[Counter, Counter, int]:
    """c(x) and c(u,v) over the graph set (plausible year ≤ Y0)."""
    c_single: Counter = Counter()
    c_pair: Counter = Counter()
    n_graph = 0
    for line in open(mapped_path, encoding="utf-8"):
        d = json.loads(line)
        y = d.get("year_ce")
        if y is None or not (_CE_MIN <= y <= _CE_MAX) or y > y0_ce:
            continue
        ids = sorted(set(d["section_ids"]))
        if not ids:
            continue
        n_graph += 1
        for x in ids:
            c_single[x] += 1
        for u, v in itertools.combinations(ids, 2):   # unordered, u<v by sort
            c_pair[(u, v)] += 1
    return c_single, c_pair, n_graph


def weight_applies(applies_path: str, c_single: Counter, c_pair: Counter,
                   min_support: int = 5) -> list[dict]:
    out = []
    for line in open(applies_path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        e = json.loads(line)
        if e.get("type") != "APPLIES_TO":
            continue
        u, v = _sid(e["src"]), _sid(e["dst"])
        key = (u, v) if u < v else (v, u)
        c_uv = c_pair.get(key, 0)
        cu, cv = c_single.get(u, 0), c_single.get(v, 0)
        # support floor: a single co-citation is not evidence of a court-used link
        w = (c_uv / min(cu, cv)) if (cu and cv and c_uv >= min_support) else 0.0
        out.append({**e, "w": round(w, 4), "c_u": cu, "c_v": cv, "c_uv": c_uv})
    return out


def run(mapped_path: str, applies_path: str, y0_ce: int, out_path: str,
        min_support: int = 5, taus=(0.0, 0.05, 0.1, 0.2, 0.3)) -> dict:
    c_single, c_pair, n_graph = build_counts(mapped_path, y0_ce)
    edges = weight_applies(applies_path, c_single, c_pair, min_support)
    with open(out_path, "w", encoding="utf-8") as f:
        for e in edges:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    total = len(edges)
    all_off = [e for e in edges if e.get("scope") == "all_offences"]
    kept_by_tau = {}
    for t in taus:
        kept = sum(e["w"] >= t and e["w"] > 0 for e in edges)
        ao_kept = sum((e["w"] >= t and e["w"] > 0) for e in all_off)
        kept_by_tau[t] = {
            "kept": kept, "kept_pct": kept / total if total else 0,
            "all_off_pruned_pct": 1 - (ao_kept / len(all_off)) if all_off else 0,
        }
    top = sorted((e for e in edges if e["w"] > 0), key=lambda e: e["w"], reverse=True)
    return {
        "n_graph_cases": n_graph, "total_edges": total,
        "all_offences_edges": len(all_off),
        "edges_with_any_cocitation": sum(e["w"] > 0 for e in edges),
        "kept_by_tau": kept_by_tau, "top": top[:20], "out": out_path,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mapped", default="data/caselaw/penal_citations.mapped.jsonl")
    ap.add_argument("--applies", default="data/applies_to_edges.jsonl")
    ap.add_argument("--y0-ce", type=int, default=2019, help="graph set = case year ≤ Y0 (CE)")
    ap.add_argument("--min-support", type=int, default=5,
                    help="min co-citation count before an edge gets a weight (else w=0)")
    ap.add_argument("--out", default="data/caselaw/applies_to_weighted.jsonl")
    args = ap.parse_args()
    s = run(args.mapped, args.applies, args.y0_ce, args.out, args.min_support)
    print(f"Y0 = CE {args.y0_ce}  |  min_support = {args.min_support}  "
          f"|  graph-set cases: {s['n_graph_cases']}")
    print(f"APPLIES_TO edges: {s['total_edges']}  "
          f"(scope=all_offences: {s['all_offences_edges']})")
    print(f"edges with ≥1 co-citation (w>0): {s['edges_with_any_cocitation']} "
          f"({s['edges_with_any_cocitation'] / s['total_edges']:.1%})")
    print("\nτ sweep — edges kept / ALL_OFFENCES pruned:")
    print(f"  {'τ':>5} {'kept':>8} {'kept%':>7} {'all_off pruned%':>16}")
    for t, v in s["kept_by_tau"].items():
        print(f"  {t:>5} {v['kept']:>8} {v['kept_pct']:>6.1%} {v['all_off_pruned_pct']:>15.1%}")
    print("\ntop-20 APPLIES_TO edges by w (general → offence):")
    for e in s["top"]:
        print(f"  w={e['w']:.3f}  {_sid(e['src']):>6} → {_sid(e['dst']):<6}  "
              f"(c_uv={e['c_uv']}, min={min(e['c_u'], e['c_v'])})  {e.get('rule','')[:34]}")
    print(f"\nwrote {s['out']}")


if __name__ == "__main__":
    main()
