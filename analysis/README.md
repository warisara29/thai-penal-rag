# analysis/ — statistical methodology (design §7)

Unit of analysis = the eval item, paired across arms. Turns per-item outcomes
(from `retrieval/results/` and `generation/verdicts/`) into the thesis's headline
numbers with uncertainty.

| Piece | File | Runs without models? |
|---|---|---|
| Paired bootstrap 95% CI (mean, Δ, interaction DiD) | `bootstrap.py` | ✅ |
| McNemar (binary) + paired permutation (rate) | `significance.py` | ✅ |
| Holm–Bonferroni over the contrast family | `significance.py` | ✅ |
| Pre-registered contrasts (RQ1/RQ2/RQ5) | `contrasts.py` | ✅ |
| 2×2 mixed-effects logistic GLMM `outcome ~ base*kg + (1\|item)` | `mixed_effects.py` | ⚙️ statsmodels + A1–A4 |

The GLMM is the *primary inferential model*; `bootstrap.ci_interaction` is the
pure-Python difference-in-differences the design also sanctions — both test
"does KG help more on PageIndex than on hybrid?" (RQ2 interaction).

## Run
```bash
python -m analysis.bootstrap && python -m analysis.significance   # self-tests
python -m analysis.run_analysis --metric hit --arms R0            # runs today
python -m analysis.run_analysis --metric hit --arms all           # as arms land
python -m analysis.run_analysis --metric correct --arms all --glmm
```
Metrics: `hit`, `coverage` (retrieval, binary → McNemar); `correct`,
`claim_grounding`, `hallucination_hard` (generation → permutation).
Writes `analysis/report.<metric>.json`.

## Pre-registered contrasts (fixed before results → Holm controls FWER)
- **RQ1** A3−A1 (PageIndex vs hybrid), A1−R0, A1−R1
- **RQ2** A2−A1, A4−A3, and the (A4−A3)−(A2−A1) interaction
- **RQ5** each retrieval arm − C0 (closed-book), C1−C0

## Not built (final step)
Ablations (§8) and the RQ1–RQ5 write-up — those consume these reports once all
arms have run. Power note (§7): overall ~150–200 paired items detect ~8–10 pt
Recall@5; the multi_hop subgroup (~75) only powers ~15 pt — don't over-claim there.
