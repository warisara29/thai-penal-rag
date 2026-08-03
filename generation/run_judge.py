"""Judge harness (§6b). Deterministic metrics run now (no LLM); LLM-judge metrics
run when a JudgeBackend is configured.

Per arm, reads generation/answers/<arm>.jsonl, joins the eval item (reference
answer + answer_claims), and reports:
  hallucinated_section_rate  hard/soft   (deterministic)
  refusal precision/recall/F1            (deterministic, over the whole arm)
  claim_grounding_rate                   (LLM judge, per reference claim vs context)
  answer_correctness (mean score, %correct)  (LLM judge)
Writes generation/verdicts/<arm>.jsonl.

Usage:
  python -m generation.run_judge --arms C1,A1 --eval eval/eval_set.generated.jsonl
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from statistics import mean

from retrieval import arms as A
from retrieval.base import Corpus
from . import hallucination as H
from . import judge as J
from . import prompts
from . import refusal as RF


def load_backend(spec):
    if not spec:
        return None
    mod, cls = spec.split(":")
    return getattr(importlib.import_module(mod), cls)()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("generation/config.json"))
    ap.add_argument("--eval", type=Path, default=Path("eval/eval_set.seed.jsonl"))
    ap.add_argument("--arms", default="C1", help="comma list")
    ap.add_argument("--answers-dir", type=Path, default=Path("generation/answers"))
    ap.add_argument("--out-dir", type=Path, default=Path("generation/verdicts"))
    ap.add_argument("--workers", type=int, default=4, help="parallel judge calls per arm")
    args = ap.parse_args()

    cfg = json.loads(args.config.read_text("utf-8"))
    corpus = Corpus.load(cfg["sections_path"])
    corpus_ids = set(corpus.by_id)
    ev = {json.loads(l)["id"]: json.loads(l)
          for l in args.eval.read_text("utf-8").splitlines() if l.strip()}
    judge_spec = cfg.get("backends", {}).get("judge")
    judge = load_backend(judge_spec) or J.default_judge()
    judge_on = judge_spec is not None
    args.out_dir.mkdir(parents=True, exist_ok=True)

    requested = A.ALL_ARMS if args.arms == "all" else [A.resolve(a.strip()) for a in args.arms.split(",")]
    for arm in requested:
        tag = f"{A.label(arm):8s}({arm})"
        path = args.answers_dir / f"{arm}.jsonl"
        if not path.exists():
            print(f"  {tag} no answers file ({path}) — run run_generation first")
            continue
        answers = [json.loads(l) for l in path.read_text("utf-8").splitlines() if l.strip()]

        def judge_one(a):
            item = ev.get(a["id"], {})
            hl = H.flag(a["answer"], corpus_ids, set(a["context_ids"]))
            v = {"id": a["id"], "arm": arm, "hallucination": hl}
            if judge_on:
                ctx = prompts.build_context(a["context_ids"], corpus)
                claims = item.get("answer_claims", []) if a["question_type"] != "unanswerable" else []
                if claims:
                    gv = judge.ground_claims(claims, ctx)
                    v["claim_grounding"] = {"verdicts": gv, "rate": sum(gv) / len(gv)}
                if a["question_type"] != "unanswerable" and item.get("reference_answer"):
                    v["correctness"] = judge.score_correctness(a["answer"], item["reference_answer"])
            return v

        from concurrent.futures import ThreadPoolExecutor, as_completed
        verdicts = []
        if args.workers > 1 and judge_on:
            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                futs = [ex.submit(judge_one, a) for a in answers]
                for f in as_completed(futs):
                    verdicts.append(f.result())
        else:
            verdicts = [judge_one(a) for a in answers]
        verdicts.sort(key=lambda v: v["id"])

        hard = [v["hallucination"]["hard_rate"] for v in verdicts]
        soft = [v["hallucination"]["soft_rate"] for v in verdicts]
        ground_rates = [v["claim_grounding"]["rate"] for v in verdicts if "claim_grounding" in v]
        scores = [v["correctness"]["score"] for v in verdicts if "correctness" in v]
        corrects = [1 if v["correctness"]["correct"] else 0 for v in verdicts if "correctness" in v]

        (args.out_dir / f"{arm}.jsonl").write_text(
            "\n".join(json.dumps(v, ensure_ascii=False) for v in verdicts), "utf-8")

        ref = RF.score([{"question_type": a["question_type"], "answer": a["answer"]} for a in answers])
        line = (f"  {tag} n={len(answers):>3}  "
                f"halluc_hard={mean(hard):.3f} soft={mean(soft):.3f}  "
                f"refusal_F1={ref['refusal_f1']:.3f}")
        if judge_on and ground_rates:
            line += f"  claim_ground={mean(ground_rates):.3f}"
        if judge_on and scores:
            line += f"  correct%={mean(corrects):.3f} score={mean(scores):.2f}"
        elif not judge_on:
            line += "  (LLM-judge metrics: configure backends.judge)"
        print(line)

    print(f"\n✓ verdicts -> {args.out_dir}/<arm>.jsonl")


if __name__ == "__main__":
    main()
