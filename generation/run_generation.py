"""Generation harness — same generator, same prompt, temp 0, across all gen arms.

Context assembly per arm:
  C0 CLOSED : no context
  C1 ORACLE : gold ∪ supporting (from the eval item)
  others    : top-k ranked ids from retrieval/results/<arm>.jsonl
Writes generation/answers/<arm>.jsonl with the answer + cost meta (RQ4).

Usage:
  python -m generation.run_generation --arms C1,A1 --eval eval/eval_set.generated.jsonl
Needs a Generator backend (retrieval/backends.py) wired in generation/config.json.
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

from retrieval import arms as A
from retrieval import backends as B
from retrieval.base import Corpus, EvalItem
from . import prompts


def load_backend(spec):
    if not spec:
        return None
    mod, cls = spec.split(":")
    return getattr(importlib.import_module(mod), cls)()


def context_ids_for(arm, item, k, results_dir, ranked_cache):
    if arm == "C0":
        return []
    if arm == "C1":
        return list(dict.fromkeys(item.gold_sections + item.supporting_sections))
    if arm not in ranked_cache:
        path = results_dir / f"{arm}.jsonl"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} missing — run `python -m retrieval.run_eval --arms {arm}` first")
        ranked_cache[arm] = {json.loads(l)["id"]: json.loads(l)["ranked"]
                             for l in path.read_text("utf-8").splitlines() if l.strip()}
    return ranked_cache[arm].get(item.id, [])[:k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("generation/config.json"))
    ap.add_argument("--eval", type=Path, default=Path("eval/eval_set.seed.jsonl"))
    ap.add_argument("--arms", default="C1", help="comma list from A1-A4,R0,R1,C0,C1")
    ap.add_argument("--results-dir", type=Path, default=Path("retrieval/results"))
    ap.add_argument("--out-dir", type=Path, default=Path("generation/answers"))
    args = ap.parse_args()

    cfg = json.loads(args.config.read_text("utf-8"))
    corpus = Corpus.load(cfg["sections_path"])
    items = EvalItem.load_all(args.eval)
    gen = load_backend(cfg.get("backends", {}).get("generator")) or B.default_generator()
    k = cfg["k"]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    ranked_cache: dict = {}

    for arm in [A.resolve(a.strip()) for a in args.arms.split(",")]:
        out = []
        try:
            for it in items:
                ctx_ids = context_ids_for(arm, it, k, args.results_dir, ranked_cache)
                answer, meta = gen.answer(it.question, ctx_ids, corpus)
                out.append({"id": it.id, "arm": arm, "question": it.question,
                            "question_type": it.question_type, "context_ids": ctx_ids,
                            "answer": answer, "gold": it.gold_sections,
                            "supporting": it.supporting_sections, "meta": meta})
        except B.NotConfigured as e:
            print(f"  {arm:4s} SKIPPED — {e}")
            continue
        (args.out_dir / f"{arm}.jsonl").write_text(
            "\n".join(json.dumps(o, ensure_ascii=False) for o in out), "utf-8")
        print(f"  {A.label(arm):8s}({arm}) wrote {len(out)} answers -> {args.out_dir}/{arm}.jsonl")


if __name__ == "__main__":
    main()
