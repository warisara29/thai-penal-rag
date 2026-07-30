"""Assemble per-item outcomes (paired across arms) from retrieval results and
generation verdicts. Unit of analysis = the eval item (design §7).

A binary outcome is {item_id: 0/1}; a rate outcome is {item_id: float}. Pairwise
tests intersect item ids so the design stays paired.
"""

from __future__ import annotations

import json
from pathlib import Path

from retrieval.metrics import coverage_at_k, hit_at_k

RETRIEVAL_METRICS = {"hit", "coverage"}  # binary per-item, from retrieval/results
VERDICT_BINARY = {"correct"}             # from generation/verdicts correctness.correct
VERDICT_RATE = {"claim_grounding", "hallucination_hard", "hallucination_soft"}


def from_retrieval(path: Path, metric: str, k: int) -> dict[str, float]:
    out = {}
    for line in Path(path).read_text("utf-8").splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        gold, ranked = d["gold"], d["ranked"]
        if not gold:
            continue  # unanswerable: no retrieval target
        if metric == "hit":
            out[d["id"]] = hit_at_k(ranked, gold, k)
        elif metric == "coverage":
            gplus = list(dict.fromkeys(gold + d.get("supporting", [])))
            out[d["id"]] = coverage_at_k(ranked, gplus, k)
        else:
            raise ValueError(f"unknown retrieval metric {metric!r}")
    return out


def from_verdicts(path: Path, metric: str) -> dict[str, float]:
    out = {}
    for line in Path(path).read_text("utf-8").splitlines():
        if not line.strip():
            continue
        v = json.loads(line)
        if metric == "correct" and "correctness" in v:
            out[v["id"]] = 1.0 if v["correctness"]["correct"] else 0.0
        elif metric == "claim_grounding" and "claim_grounding" in v:
            out[v["id"]] = float(v["claim_grounding"]["rate"])
        elif metric == "hallucination_hard":
            out[v["id"]] = float(v["hallucination"]["hard_rate"])
        elif metric == "hallucination_soft":
            out[v["id"]] = float(v["hallucination"]["soft_rate"])
    return out


def load_arm(arm: str, metric: str, k: int, results_dir: Path, verdicts_dir: Path) -> dict[str, float]:
    if metric in RETRIEVAL_METRICS:
        return from_retrieval(results_dir / f"{arm}.jsonl", metric, k)
    return from_verdicts(verdicts_dir / f"{arm}.jsonl", metric)


def paired(a: dict[str, float], b: dict[str, float]) -> tuple[list[float], list[float]]:
    ids = [i for i in a if i in b]
    return [a[i] for i in ids], [b[i] for i in ids]
