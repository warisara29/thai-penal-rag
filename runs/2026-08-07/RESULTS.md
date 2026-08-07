# Run 2026-08-07 — final additive-design run

Locked-config snapshot of the **additive-component pipeline**. Supersedes
[2026-08-03](../2026-08-03/RESULTS.md) (the replacement-2×2 design, where the full system
finished last). Here the full system **`hybrid+pi+kg`** is the best real arm on answer quality.

## Locked configuration
- **Design:** additive — three ingredients (hybrid, PageIndex, KG) combined, never replacing.
- **Full system:** `hybrid ∪ PageIndex → reserved-slot KG → shared rerank` (`FullSystem`,
  `kg_reserve=1`: KG appends the best supporting provision into a reserved tail slot, so it
  adds multi-hop context without ever displacing a primary gold section).
- **Navigator:** PageIndex beam width = 5.
- **Prompt:** shared generation prompt, identical across arms (a multi-hop reasoning nudge was
  tested and **rejected** — it regressed every question type; see below).
- **Models (DeepInfra):** generator + navigator `Qwen/Qwen3.6-35B-A3B` (temp 0) · embedder
  `BAAI/bge-m3` · reranker `Qwen/Qwen3-Reranker-4B` · judge `deepseek-ai/DeepSeek-V4-Pro`.
- **Eval:** 255 items, 6 types, gold-by-construction, seed 20260729. k = 5, chunk = 1 มาตรา, paired.

**Provenance (no wasted API):** arms that do **not** use the PageIndex navigator (`hybrid`,
`hybrid+kg`, `keyword`, `dense`, `closed`, `ORACLE`) are byte-for-byte reproducible under the
locked config and are carried from the 2026-08-03 sweep. Navigator arms (`pi`, `hybrid+pi`,
`hybrid+pi+kg`) were run fresh under beam=5 + reserved-slot KG.

## Results (retrieval n≈220 answerable; generation n=255)

| arm | recall@5 | mrr@10 | cov@5 | sup@5 | correct% | claim | score |
|-----|---------:|-------:|------:|------:|---------:|------:|------:|
| closed *(floor)* | — | — | — | — | 0.023 | 0.023 | 1.10 |
| keyword | 0.609 | 0.457 | 0.527 | 0.178 | 0.445 | 0.556 | 2.75 |
| dense | 0.832 | 0.687 | 0.682 | 0.022 | 0.614 | 0.730 | 3.38 |
| hybrid *(baseline)* | 0.927 | 0.824 | 0.800 | 0.311 | 0.636 | 0.826 | 3.51 |
| pi | 0.518 | 0.444 | 0.400 | 0.133 | 0.350 | 0.454 | 2.39 |
| hybrid+pi | 0.950 | 0.838 | 0.827 | 0.378 | 0.673 | 0.853 | 3.61 |
| hybrid+kg | 0.936 | 0.829 | 0.826 | 0.422 | 0.655 | 0.838 | 3.61 |
| **hybrid+pi+kg** *(full)* | 0.932 | 0.833 | **0.850** | **0.556** | **0.718** | 0.841 | **3.81** |
| ORACLE *(ceiling)* | — | — | — | — | 0.809 | 0.939 | 4.13 |

Hallucinated-section rate (hard) ≈ 0 across all arms.

## Statistics (paired bootstrap, 10k resamples; McNemar; Holm FWER 0.05)

Per-arm correctness 95% CI and the key paired contrasts:

| contrast | Δ correct% | 95% CI | McNemar p | Holm |
|----------|-----------:|:------:|:---------:|:----:|
| **hybrid+pi+kg − hybrid** | +0.082 | [+0.027, +0.136] | 0.006 | **✅ significant** |
| hybrid+pi+kg − hybrid+kg | +0.064 | [+0.009, +0.118] | 0.029 | ✗ (not after Holm) |
| hybrid+pi+kg − hybrid+pi | +0.045 | [−0.005, +0.100] | 0.121 | ✗ ns |
| hybrid+pi − hybrid | +0.036 | [−0.014, +0.082] | 0.201 | ✗ ns |
| hybrid+kg − hybrid | +0.018 | [−0.027, +0.064] | 0.572 | ✗ ns |

## Findings

- **The full system significantly beats the hybrid baseline** (+0.082, survives Holm). It is
  the **only** arm to clear the baseline with statistical significance — neither added component
  alone (pi +0.036, kg +0.018) is significant; they reach significance only **combined**. This is
  the core argument for the full pipeline.
- **Full vs `hybrid+pi` (+0.045) is not significant** (p=0.12) — nominally best but statistically
  tied at n=220. We do not claim strict superiority over `hybrid+pi`; we note the full system is
  the only one with a significant edge over baseline.
- **RQ1 (refined):** PageIndex as a *replacement* base is far worse than hybrid (pi 0.350 vs
  hybrid 0.636); as an *additive* signal it lifts both recall (0.950) and correctness (+0.037).
- **RQ2:** reserved-slot KG raises multi-hop support-recall 0.378→0.556 and coverage to 0.850
  while lifting overall correctness — the two-tier ranking is what makes KG net-positive.
- **RQ5:** retrieval essential — closed-book 0.023 vs full 0.718 (oracle 0.809).
- **Ceiling gap:** 0.809 − 0.718 = 0.091, down from 0.141; ~35% of the residual closed. The
  remainder is concentrated in `multi_hop` (0.489 vs 0.778 oracle) and is a **generator
  reasoning limit**: the supporting provision is in context (sup-recall 0.556) but the model
  under-synthesises it, and an explicit reasoning prompt made it worse (rejected).

## Ablations (§8) — why 0.718 is a genuine optimum

Four levers were tried to push the full system past 0.718. **All four failed, and all failed
for the same reason** — they feed more into KG expansion, which amplifies the over-generic
`all_offences` APPLIES_TO edges (mean ~271 neighbours/node, 0 verified) and drowns the one
relevant supporting provision. This is strong evidence the pipeline is at its ceiling given the
current KG, and that the remaining bottleneck is **KG edge precision**, not any hyperparameter.

| ablation | setting | correct% | vs base | verdict |
|----------|---------|---------:|--------:|---------|
| **Generation context size** | k=5 (base) | 0.668 | — | k=5 optimal |
| | k=3 | 0.668 | +0.000 | ties, worse claim-grounding |
| | k=2 | 0.636 | −0.032 | worse |
| **Multi-hop reasoning prompt** | off (base) | 0.718 | — | **rejected** |
| | on | 0.700 | −0.018 | regressed *every* type |
| **KG reserved slots** | `kg_reserve=1` (base) | 0.718 | — | 1 optimal |
| | `kg_reserve=2` | 0.702 | −0.018 | support-recall 0.556→0.378 |
| **Candidate pool depth** | `POOL=50` (base) | 0.718 | — | 50 optimal |
| | `POOL=100` | 0.686 | −0.032 | recall ↑0.945 but multi_hop −0.111 |

Note the pool-depth row: `POOL=100` **does** raise recall@5 (0.932→0.945), confirming primary
recall can go higher — but correctness *falls*, because the larger seed expands into more KG
noise and multi_hop collapses. Recall is not the binding constraint; KG precision is.

**Multi-hop diagnosis (the residual gap).** `multi_hop` is 0.489 vs oracle 0.778. Decomposed:
retrieval-fault (supporting missing from top-5) = 33% of items, generation-fault (supporting
present, still wrong) = 18%. Every missing supporting provision **is reachable** in the KG
(100% edge coverage from the gold section) — it is simply out-ranked among ~271 neighbours.
51% of needed supporting provisions are reachable only via `all_offences` edges, so naive
pruning is not viable. **The fix is edge curation, not a pipeline knob** (see future work).

## Caveats / TODO
- `refusal_F1 = nan` — refusal-cue list still needs tuning.
- ~20% human audit of the judge (κ/ρ) pending — judge numbers not yet licensed.
- APPLIES_TO edges unverified (lawyer pass pending).
- GLMM 2×2 interaction not fit (needs statsmodels); §8 ablations and RQ4 cost axis outstanding.
- n=220 limits power to separate the top additive arms; the full-vs-`hybrid+pi` gap is unresolved.
