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

    for arm in [a.strip() for a in args.arms.split(",")]:
        path = args.answers_dir / f"{arm}.jsonl"
        if not path.exists():
            print(f"  {arm:4s} no answers file ({path}) — run run_generation first")
            continue
        answers = [json.loads(l) for l in path.read_text("utf-8").splitlines() if l.strip()]

        verdicts, hard, soft, ground_rates, scores, corrects = [], [], [], [], [], []
        for a in answers:
            item = ev.get(a["id"], {})
            hl = H.flag(a["answer"], corpus_ids, set(a["context_ids"]))
            hard.append(hl["hard_rate"]); soft.append(hl["soft_rate"])
            v = {"id": a["id"], "arm": arm, "hallucination": hl}

            if judge_on:
                ctx = prompts.build_context(a["context_ids"], corpus)
                claims = item.get("answer_claims", []) if a["question_type"] != "unanswerable" else []
                if claims:
                    gv = judge.ground_claims(claims, ctx)
                    rate = sum(gv) / len(gv)
                    ground_rates.append(rate)
                    v["claim_grounding"] = {"verdicts": gv, "rate": rate}
                if a["question_type"] != "unanswerable" and item.get("reference_answer"):
                    sc = judge.score_correctness(a["answer"], item["reference_answer"])
                    scores.append(sc["score"]); corrects.append(1 if sc["correct"] else 0)
                    v["correctness"] = sc
            verdicts.append(v)

        (args.out_dir / f"{arm}.jsonl").write_text(
            "\n".join(json.dumps(v, ensure_ascii=False) for v in verdicts), "utf-8")

        ref = RF.score([{"question_type": a["question_type"], "answer": a["answer"]} for a in answers])
        line = (f"  {arm:4s} n={len(answers):>3}  "
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
