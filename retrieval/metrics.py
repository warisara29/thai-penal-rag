"""Retrieval metrics scored against gold_sections. RQ1/RQ2 axis.

recall@k  : fraction of an item's gold sections that appear in the top-k
mrr@k     : reciprocal rank of the FIRST gold section within the top-k (0 if none)
hit@k     : 1 if any gold section is in the top-k
Unanswerable items (empty gold) are excluded from retrieval scoring.
"""

from __future__ import annotations

from statistics import mean


def recall_at_k(ranked: list[str], gold: list[str], k: int) -> float:
    if not gold:
        return float("nan")
    topk = set(ranked[:k])
    return sum(1 for g in gold if g in topk) / len(gold)


def hit_at_k(ranked: list[str], gold: list[str], k: int) -> float:
    if not gold:
        return float("nan")
    topk = set(ranked[:k])
    return 1.0 if any(g in topk for g in gold) else 0.0


def mrr_at_k(ranked: list[str], gold: list[str], k: int) -> float:
    if not gold:
        return float("nan")
    goldset = set(gold)
    for i, sid in enumerate(ranked[:k], 1):
        if sid in goldset:
            return 1.0 / i
    return 0.0


def map_at_k(ranked: list[str], gold: list[str], k: int) -> float:
    """Mean average precision over gold within top-k."""
    if not gold:
        return float("nan")
    goldset, hits, ap = set(gold), 0, 0.0
    for i, sid in enumerate(ranked[:k], 1):
        if sid in goldset:
            hits += 1
            ap += hits / i
    return ap / len(goldset)


def coverage_at_k(ranked: list[str], gold_plus: list[str], k: int) -> float:
    """Multi-hop coverage: 1 iff ALL of gold∪supporting are in top-k (RQ2)."""
    if not gold_plus:
        return float("nan")
    return 1.0 if set(gold_plus) <= set(ranked[:k]) else 0.0


def aggregate(rows: list[dict], k_recall: int, k_mrr: int) -> dict:
    """rows: [{ranked, gold, supporting?}]. Mean metrics over answerable items (§6a)."""
    scored = [r for r in rows if r["gold"]]
    if not scored:
        return {"n": 0}

    def gplus(r):
        return list(dict.fromkeys(r["gold"] + r.get("supporting", [])))
    with_sup = [r for r in scored if r.get("supporting")]
    out = {
        "n": len(scored),
        f"recall@{k_recall}": mean(recall_at_k(r["ranked"], r["gold"], k_recall) for r in scored),
        f"hit@{k_recall}": mean(hit_at_k(r["ranked"], r["gold"], k_recall) for r in scored),
        f"mrr@{k_mrr}": mean(mrr_at_k(r["ranked"], r["gold"], k_mrr) for r in scored),
        f"map@{k_mrr}": mean(map_at_k(r["ranked"], r["gold"], k_mrr) for r in scored),
        f"coverage@{k_recall}": mean(coverage_at_k(r["ranked"], gplus(r), k_recall) for r in scored),
    }
    if with_sup:  # support-recall isolates the KG/general-provision effect (RQ2)
        out[f"support_recall@{k_recall}"] = mean(
            recall_at_k(r["ranked"], r["supporting"], k_recall) for r in with_sup)
        out["n_multihop"] = len(with_sup)
    return out


if __name__ == "__main__":  # tiny self-test
    assert recall_at_k(["a", "b", "c"], ["b"], 5) == 1.0
    assert recall_at_k(["a", "b", "c"], ["x"], 5) == 0.0
    assert recall_at_k(["a", "b"], ["a", "z"], 5) == 0.5
    assert mrr_at_k(["a", "b", "c"], ["b"], 5) == 0.5
    assert mrr_at_k(["a", "b"], ["x"], 5) == 0.0
    assert hit_at_k(["a", "b"], ["b"], 5) == 1.0
    assert coverage_at_k(["a", "b", "c"], ["a", "b"], 5) == 1.0
    assert coverage_at_k(["a", "c"], ["a", "b"], 5) == 0.0
    assert abs(map_at_k(["a", "x", "b"], ["a", "b"], 5) - (1 / 1 + 2 / 3) / 2) < 1e-9
    print("metrics self-test ok")
