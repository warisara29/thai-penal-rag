# Run 2026-08-03 — first full experimental run

Immutable snapshot of the first complete 255-item run (all 8 arms, live on DeepInfra).
Working dirs (`retrieval/results/`, `generation/{answers,verdicts}/`, `eval/eval_set.generated.jsonl`)
are gitignored/regenerable; this folder is the preserved record.

## Setup
- **Corpus:** ประมวลกฎหมายอาญา, PyThaiNLP `criminal-csv-v0.1`, 444 มาตรา
- **Eval set:** 255 items, 6 types (lookup 75 · penalty 55 · multi_hop 45 · unanswerable 35 · exception 25 · definition 20), gold-by-construction, seed=20260729
- **Models (DeepInfra):** generator + PageIndex navigator `Qwen/Qwen3.6-35B-A3B` (temp 0) · embedder `BAAI/bge-m3` · reranker `Qwen/Qwen3-Reranker-4B` · judge + drafter `deepseek-ai/DeepSeek-V4-Pro`
- **k = 5**, chunk = 1 มาตรา, paired across arms

## Results (n≈220 answerable; unanswerable excluded from retrieval)
| Arm | recall@5 | mrr@10 | cov@5 | support_recall@5 | correct% | claim-ground | judge score |
|-----|---------:|-------:|------:|-----------------:|---------:|-------------:|------------:|
| B0 closed-book | — | — | — | — | 0.023 | 0.023 | 1.10 |
| B1 keyword | 0.609 | 0.457 | 0.527 | 0.178 | 0.445 | 0.556 | 2.75 |
| B2 dense | 0.832 | 0.687 | 0.682 | 0.022 | 0.614 | 0.730 | 3.38 |
| B3 hybrid | 0.927 | 0.824 | 0.800 | 0.311 | 0.636 | 0.826 | 3.51 |
| **B3+KG** | **0.936** | **0.829** | **0.826** | **0.422** | **0.655** | **0.838** | **3.61** |
| B3+PI | 0.500 | 0.430 | 0.368 | 0.111 | 0.332 | 0.423 | 2.28 |
| PIKG-RAG | 0.486 | 0.455 | 0.391 | 0.356 | 0.373 | 0.442 | 2.41 |
| ORACLE (ceiling) | — | — | — | — | 0.809 | 0.939 | 4.13 |

Hallucinated-section rate (hard) ≈ 0 across all arms.

## Findings (Holm-Bonferroni FWER-controlled; full stats in `analysis/`)
- **RQ1 — PageIndex vs hybrid: PageIndex loses, significantly.** hit@5 Δ = −0.427
  [−0.50, −0.36], p_adj < 1e-4. H1 rejected in the opposite direction. *Limitation:* the
  Qwen navigator may be under-powered vs strong hybrid dense retrieval.
- **RQ2 — KG expansion: helps multi-hop marginally, not significant after correction.**
  B3+KG is best on every metric; support_recall@5 jumps 0.311→0.422 and coverage@5 +0.027
  (p=0.07). KG×base interaction ≈ 0.
- **RQ5 — retrieval is essential.** Closed-book Qwen answers 2.3% correctly; hybrid RAG 63.6%
  (Δ = +0.614, p_adj < 1e-4); oracle ceiling 80.9%. Ladder B0→B1→B2 all significant on
  correctness; B3−B2 not significant (retrieval gains plateau into generation).

## Caveats / TODO
- `refusal_F1 = nan` — models rarely emit the detected refusal cues on out-of-scope Qs;
  refusal-cue list needs tuning before the refusal-accuracy metric is trustworthy.
- APPLIES_TO edges are unverified (lawyer pass pending) — KG effect uses all candidate edges.
- ~20% human audit of the judge (κ/ρ) not yet done — judge numbers not yet licensed.
- Cost axis (RQ4) and ablations (§8) not yet computed.
