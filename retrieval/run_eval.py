"""Harness: run the same questions through every arm at k=5 (design §4c).

Runnable now for arms that need no models (R0, and A2's KG logic once a base
ranker exists). Model arms are constructed but skipped with a clear message
until their backend is configured. Writes per-arm rankings + a metric summary.

Usage:
  python -m retrieval.run_eval --arms R0                 # lexical floor, runs today
  python -m retrieval.run_eval --arms R0,R1,A1,A2,A3,A4  # full sweep once backends set
  python -m retrieval.run_eval --eval eval/eval_set.seed.jsonl --arms R0
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

from . import arms as A
from . import backends as B
from . import metrics as M
from .base import Corpus, EvalItem
from .bm25 import BM25
from .kg_expand import KGExpander
from .pageindex import HeuristicNodeSelector, PageIndexNavigator


def _load_backend(spec):
    """spec: 'module:Class' -> instance, or None -> not-configured stub."""
    if not spec:
        return None
    mod, cls = spec.split(":")
    return getattr(importlib.import_module(mod), cls)()


def build_context(cfg) -> A.Context:
    corpus = Corpus.load(cfg["sections_path"])
    bm25 = BM25(corpus, tokenizer=cfg.get("tokenizer", "char_ngram"))
    expander = KGExpander(cfg.get("cites_edges_path"), cfg.get("applies_to_edges_path"),
                          cfg.get("verified_applies_to_only", False))
    bk = cfg.get("backends", {})
    # navigator defaults to the no-LLM heuristic descent so A3 runs today; override
    # backends.navigator with an LLMNodeSelector-based navigator for the real arm.
    navigator = _load_backend(bk.get("navigator")) or PageIndexNavigator(HeuristicNodeSelector())
    return A.Context(
        corpus=corpus, bm25=bm25, expander=expander,
        embedder=_load_backend(bk.get("embedder")) or B.default_embedder(),
        reranker=_load_backend(bk.get("reranker")) or B.default_reranker(),
        navigator=navigator,
        config=cfg,
    )


def run_arm(name, ctx, items, k):
    arm = A.build_arm(name, ctx)
    rows, out = [], []
    for it in items:
        if it.question_type == "unanswerable":
            continue  # no retrieval target
        res = arm.retrieve(it.question, it.id, k)
        rows.append({"ranked": res.ranked_ids, "gold": it.gold_sections,
                     "supporting": it.supporting_sections})
        out.append({"id": it.id, "ranked": res.ranked_ids, "gold": it.gold_sections,
                    "supporting": it.supporting_sections, "meta": res.meta})
    return rows, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("retrieval/config.json"))
    ap.add_argument("--eval", type=Path, default=Path("eval/eval_set.seed.jsonl"))
    ap.add_argument("--arms", default="R0", help="comma list, or 'all'")
    ap.add_argument("--out-dir", type=Path, default=Path("retrieval/results"))
    args = ap.parse_args()

    cfg = json.loads(args.config.read_text("utf-8"))
    ctx = build_context(cfg)
    items = EvalItem.load_all(args.eval)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    k, kr, km = cfg["k"], cfg["k_recall"], cfg["k_mrr"]

    requested = A.ALL_ARMS if args.arms == "all" else [a.strip() for a in args.arms.split(",")]
    print(f"eval items: {len(items)}  (answerable retrieval-scored: "
          f"{sum(1 for i in items if i.question_type != 'unanswerable')})\n")

    summary = []
    for name in requested:
        if name in A.GEN_ONLY:
            print(f"  {name:4s} generation-only (C0 closed-book / C1 oracle) — "
                  f"needs Generator backend; not retrieval-scored")
            continue
        try:
            rows, out = run_arm(name, ctx, items, k)
        except B.NotConfigured as e:
            print(f"  {name:4s} SKIPPED — {e}")
            continue
        (args.out_dir / f"{name}.jsonl").write_text(
            "\n".join(json.dumps(o, ensure_ascii=False) for o in out), "utf-8")
        agg = M.aggregate(rows, kr, km)
        summary.append((name, agg))
        sup = f"  support_recall@{kr}={agg[f'support_recall@{kr}']:.3f}" if f"support_recall@{kr}" in agg else ""
        print(f"  {name:4s} n={agg['n']:>3}  "
              f"recall@{kr}={agg.get(f'recall@{kr}', float('nan')):.3f}  "
              f"hit@{kr}={agg.get(f'hit@{kr}', float('nan')):.3f}  "
              f"mrr@{km}={agg.get(f'mrr@{km}', float('nan')):.3f}  "
              f"map@{km}={agg.get(f'map@{km}', float('nan')):.3f}  "
              f"cov@{kr}={agg.get(f'coverage@{kr}', float('nan')):.3f}{sup}")

    if summary:
        print(f"\n✓ wrote per-arm rankings -> {args.out_dir}/<arm>.jsonl")


if __name__ == "__main__":
    main()
