# NEXT_STEPS — from scaffold to RQ1–RQ5

Every box in the flow is scaffolded and runs on inputs that need no models
(R0, A3-heuristic, KG expansion, all metrics, deterministic judge, §7 stats).
What remains is **wiring 4 model backends, running the pipeline, and writing up**.
Do it in this order — each step unlocks the next.

---

## Step 0 — one-time inputs (no code)
- [ ] `export DEEPINFRA_API_KEY=…`  (the ONLY key needed — all models run on DeepInfra)
- [ ] `python3 -m venv .venv && .venv/bin/pip install openai requests statsmodels pandas pythainlp`
      (run everything with `.venv/bin/python`; homebrew python blocks system installs)
- [ ] Lawyer: verify `kg/applies_to_rules.json` (`verified:true`) + fill
      `data/applies_to_review.csv`; re-run `kg/expand_applies_to.py … --verified-only`.

All models are on DeepInfra (verified working on the seed set):
| Role | Model |
|---|---|
| Generator + PageIndex navigator | `Qwen/Qwen3.6-35B-A3B` |
| Embedder | `BAAI/bge-m3` |
| Reranker | `Qwen/Qwen3-Reranker-4B` |
| Judge + eval drafter (≠ generator) | `deepseek-ai/DeepSeek-V4-Pro` |

## Step 1 — build the eval set
```bash
python eval/generate_eval.py --out eval/eval_set.generated.jsonl --workers 4
python eval/validate_eval.py eval/eval_set.generated.jsonl --schema eval/schema.json --sections data/sections.jsonl
```
→ 255 items. Use `--eval eval/eval_set.generated.jsonl` everywhere below.

## Step 2 — retrieval backends (wired to DeepInfra, verified)
`backends_deepinfra.py` implements everything; the config files already point at it.
B1/B2/B3/B3+PI all confirmed on the seed set. Just run:
```bash
.venv/bin/python -m retrieval.run_eval --arms all --eval eval/eval_set.generated.jsonl
```
→ writes `retrieval/results/<arm>.jsonl` (recall@5, mrr@10, map@10, coverage@5, support_recall@5).
(Set any backend to `null` in `retrieval/config.json` to fall back: navigator=null → the
no-LLM heuristic PageIndex; embedder/reranker=null → skip those arms. R0/B1 needs none.)

## Step 3 — generation + judge (already wired)
Generator = `backends_deepinfra:DeepInfraGenerator` (Qwen, temp 0); judge =
`backends_deepinfra:DeepInfraJudge` (DeepSeek-V4-Pro ≠ Qwen generator, no self-preference bias). Then:
```bash
python -m generation.run_generation --arms A1,A2,A3,A4,R0,R1,C0,C1 --eval eval/eval_set.generated.jsonl
python -m generation.run_judge      --arms A1,A2,A3,A4,R0,R1,C0,C1 --eval eval/eval_set.generated.jsonl
python -m generation.audit_sample --make      # lawyer scores ~20%
python -m generation.audit_sample --agreement # judge–human κ / ρ (license the judge)
```

## Step 4 — analysis (design §7) — already built
```bash
python -m analysis.run_analysis --metric hit      --arms all
python -m analysis.run_analysis --metric coverage --arms all          # RQ2
python -m analysis.run_analysis --metric correct  --arms all --glmm   # RQ5 + 2×2 GLMM
```
→ per-arm CIs, pre-registered contrasts (McNemar/permutation), Holm–Bonferroni,
the KG×base interaction, and `analysis/report.<metric>.json`.

## Step 5 — the last box (not yet scaffolded)
- [ ] **Ablations §8** — k-sweep (k∈{1,3,5,10}), KG-edge-type ablation
      (CITES-only vs +APPLIES_TO), verified-vs-all APPLIES_TO, PI beam width.
- [ ] **Cost axis (RQ4)** — roll up `meta.llm_calls`/`tokens`/latency per arm.
- [ ] **Write RQ1–RQ5** from the `analysis/report.*.json` numbers.
  Ask Claude to scaffold §8 once Step 4 produces real reports.

---

### Fairness invariants to preserve (design §4c) — don't let a backend break these
Chunk = 1 มาตรา everywhere · same generator+prompt+temp 0 for A1–A4/C0/C1 · same k
presented to the generator · PageIndex navigator = the generator's model · one shared
reranker across A1–A4.
```
