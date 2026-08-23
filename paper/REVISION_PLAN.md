# Revision Plan — advisor review (13 comment groups)

Response to a full reviewer-grade critique of `paper/main.tex` (PIKG-RAG, Gemma-judge version).
Each item lists the fix and where it lands. Grouped by execution phase.

**Decision locked:** Comment #1 (interaction/synergy) → framing option **(ข)** — report the effect as
*additive* (two structural layers together cross the significance threshold; neither alone does at
this sample size), NOT synergy. Also fit the bootstrap difference-in-differences to state "no
significant interaction" explicitly.

---

## Phase 0 — investigate + free (no new runs)

| # | Comment | Action | Status |
|---|---------|--------|:------:|
| 11d | CLOSED grounding = 0.075 with empty context | Inspect how per-claim grounding is computed when context is empty; fix to "—" or explain | ▢ |
| 12a | grounding flat (PIKG-RAG 0.895 = HYBRID) while accuracy highest | Check whether PIKG-RAG actually cites the supporting section on multi-hop; if not, soften the mechanism claim | ▢ |
| 12b | DENSE sup5 0.022 ≪ BM25 0.178 | Add as direct empirical evidence for the motivation (similarity misses low-overlap provisions) | ▢ |
| 9d | No cost numbers | Roll up llm_calls/tokens per query from retrieval `meta`; PageIndex vs Hybrid | ▢ |
| 1 | Interaction untested | Run `analysis.bootstrap.ci_interaction` (DiD); report no-significant-interaction; rewrite abstract/conclusion to additive framing | ▢ |
| 13 | Terminology + prose | PIKG-RAG (not FULL), HYBRID+PI, multi-hop, reserved rank k, unretrieved-citation rate, adaptive gating, AmE (offense/materialized/normalized); rewrite abstract sent. 1–2, III-F split, judge "organization" not "provider", define conditional fan-out, §V roadmap, "Every missing support" + n, NitiBench note, Eq.(1) brackets, [19] reranker cite, "220 spans five types" | ▢ |

## Phase 1 — verify with the controlled re-run (already launched)

| # | Comment | Action |
|---|---------|--------|
| 2 | §IV-C numbers inconsistent (0.786 vs 0.795, different denominators) | Recompute on ONE base = PIKG-RAG's own 4 primaries: context-size = A4B top-5 vs top-4; KG = A4B-4 off vs on; verify from persisted `runs/2026-08-07/controlled/`; Discussion states net KG Δ≈+0.018, p≈0.54 (not significant) plainly |
| 6 | multi-hop +0.200 fragile | Move κ=0.13 warning next to Table VI; add bootstrap CI per type (expect multi-hop CI to straddle 0 at n=45); reduce narrative weight |
| 3b | Sup@5 denominator | State it is computed only on items with supporting sections (report n) |

## Phase 2 — new experiments (batch, background; persist to repo)

| # | Comment | Experiment |
|---|---------|-----------|
| 5 | Asymmetric tuning | PIKG-RAG at α=0.7; report best-vs-best (Hybrid@0.7 vs PIKG-RAG@0.7) |
| 8b/9c | "KG" mostly generic | CITES-only vs APPLIES_TO-only vs both (KGExpander flags exist) |
| 9a | Clean reserved-slot control | k=6 (5 primary + 1 reserved) |
| 9b | Cited [12] Lost-in-the-Middle but never used | supporting at rank 1 vs rank k |
| 10 | Generation variance unmeasured | 3 repeats of HYBRID vs PIKG-RAG; report Δ spread |
| 11e | 14% (unanswerable) dropped vaguely | judge-based refusal metric to recover the 35 items |
| 7b | Construct validity / lexical-overlap bias | paraphrase 30–50 items into lay language; re-measure recall + the PIKG-RAG−Hybrid gap |

## Phase 3 — writing pass (after numbers settle)

- 8a: rename "knowledge graph" → "rule-derived provision graph" in Method (keep PIKG-RAG acronym, explain).
- 7a: Limitations paragraph on construct validity (lexical overlap) + penalty-snapshot staleness.
- GLMM: fit or, if DiD suffices, state additive result in §IV (remove "remains to be fitted").
- Apply all #13 terminology/prose fixes globally.

---

## Cross-links (comments that share a root)
- **#1 ↔ #6 ↔ #12a**: the multi-hop story, the interaction claim, and the flat grounding all bear on
  whether the KG mechanism is what we say it is. Treat together; land on "additive, KG net-effect not
  significant, multi-hop effect exploratory."
- **#2 ↔ #13-Q1**: the 0.786/0.795 confusion is the single hardest-to-verify spot; fix denominators.
- **#5 ↔ #8b**: fairness of the baseline comparison and where the KG value actually comes from.
- **#7 ↔ #12b**: benchmark validity — 12b is supporting evidence, 7b is the stress test.
