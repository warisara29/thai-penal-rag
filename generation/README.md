# generation/ — answer generation + LLM-judge (design §6b)

Same generator, same prompt, temp 0, across A1–A4 + C0 + C1. Then score answer
quality with an LLM judge (≠ generator) + deterministic checks.

## Pipeline
```
retrieval/results/<arm>.jsonl                         eval item (ref answer + claims)
        │  top-k context ids                                  │
        ▼                                                     ▼
run_generation.py ── Generator (Thai model, temp 0) ──▶ answers/<arm>.jsonl
        │                                                     │
        ▼                                                     ▼
run_judge.py ── deterministic + JudgeBackend (≠ generator) ─▶ verdicts/<arm>.jsonl
        │
        ▼
audit_sample.py ── lawyer scores ~20% ──▶ judge–human κ / ρ  (licenses the judge)
```

## Metrics (§6b)
| Metric | Runs now? |
|---|---|
| Hallucinated-section rate (hard: not-in-corpus / soft: not-retrieved) | ✅ deterministic |
| Refusal precision/recall/F1 on `unanswerable` | ✅ deterministic |
| Claim-grounding rate (per `answer_claims` vs context) | ⚙️ needs JudgeBackend |
| Answer correctness (1–5 + binary vs `reference_answer`) | ⚙️ needs JudgeBackend |

## Run
```bash
python -m retrieval.metrics && python -m generation.hallucination && python -m generation.refusal
python -m generation.run_generation --arms C1,A1 --eval eval/eval_set.generated.jsonl
python -m generation.run_judge      --arms C1,A1 --eval eval/eval_set.generated.jsonl
python -m generation.audit_sample --make          # 20% worksheet for the lawyer
python -m generation.audit_sample --agreement     # κ / ρ once filled
```

## Wire the two backends (`generation/config.json`)
- **`generator`** — the pinned Thai open model (Typhoon/SeaLLM/OpenThaiGPT), temp 0.
  Implement `retrieval.backends.Generator` and set `backends.generator`.
- **`judge`** — MUST differ from the generator. `generation.judge:AnthropicJudge`
  (Claude Opus 4.8) works out of the box after `pip install anthropic` + creds.

Cost axis (RQ4): `run_generation` records `meta` per answer — decompose latency /
LLM-calls / tokens there once the generator reports them.
