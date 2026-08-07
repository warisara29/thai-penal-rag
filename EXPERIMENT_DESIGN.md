# Experiment Design — Structure-aware retrieval for Thai statutory QA

**Thesis working title:** *Does authored legal structure help retrieval? PageIndex
tree navigation and knowledge-graph expansion vs. dense-hybrid retrieval on the Thai
Penal Code (ประมวลกฎหมายอาญา).*

**Status:** design frozen *before* implementation. This document is the contract the
code and the thesis chapters are written against. Change it deliberately, not silently.

---

## 0. One-paragraph summary

We treat the Thai Penal Code as a single shared corpus and ask whether exploiting the
**authored hierarchy** (ภาค→ลักษณะ→หมวด→มาตรา) via PageIndex-style LLM tree navigation,
and **explicit legal links** (CITES, APPLIES_TO) via a knowledge graph, retrieve the
sections a correct legal answer must cite *better than* a strong dense-hybrid vector
baseline. Retrieval quality and its downstream effect on a **fixed Thai open-source LLM
generator** are both measured against one gold eval set, with paired significance testing
and human-validated LLM-as-judge scoring. The deliverable is a per-layer, per-question-type
attribution of value — "how many points of Recall@5 does each structural ingredient buy,
and does it survive to the generated answer?"

---

## 1. Research questions & hypotheses

Each RQ maps to specific arms (§4) and metrics (§6). Hypotheses are directional; the null
in every case is "no difference."

| RQ | Question | Primary metric | Hypothesis |
|----|----------|----------------|-----------|
| **RQ1** | Does PageIndex tree navigation beat the dense-hybrid baseline at finding the *directly answering* section(s)? | Recall@5, MRR@10 vs `gold_sections` | **H1**: PageIndex ≥ baseline overall; gain is largest on `definition`/`exception`/structural questions where the hierarchy carries signal. **Refined (2026-08-04):** H1 rejected for PageIndex as a *replacement* base (loses to hybrid); it holds only *additively* — PageIndex on top of hybrid (A4B) beats hybrid. |
| **RQ2** | Does KG multi-hop expansion recover the *general provisions* needed to reason (Book-1 sections)? | Multi-hop coverage @k vs `gold ∪ supporting` | **H2**: KG expansion raises multi-hop coverage on `multi_hop` items with negligible effect on `lookup`; the `APPLIES_TO` edge is the driver. |
| **RQ3** | Does better retrieval translate into a better *generated answer* from a fixed Thai LLM? | Claim-grounding rate, hallucinated-section rate | **H3**: downstream faithfulness increases monotonically with retrieval quality, holding the generator fixed. |
| **RQ4** | What does each structural layer cost? | LLM calls/query, p95 latency, index build | **H4** (descriptive): PageIndex and KG add answer quality at super-linear latency/cost; we quantify the trade. |
| **RQ5** | How much does retrieval matter *at all*, given the Penal Code is likely in the LLM's pretraining? | Δ(retrieval arm − closed-book) | **H5**: retrieval adds net value over the model's parametric memory, especially on penalty numbers and inserted sections (ทวิ/ตรี). |

RQ5 is the contamination guard and, for a Thai-law thesis, is a contribution in itself.

---

## 2. Design type

- **Within-subjects / paired.** Every eval item passes through *every* arm. This removes
  item-difficulty variance and gives paired tests far more power than a between-groups
  split — essential because the corpus is one statute and N is bounded.
- **2×2 factorial core** over the two structural ingredients, so we get **main effects and
  their interaction**, not just a leaderboard:
  - **Factor A — base retriever:** `hybrid-baseline` vs `pageindex`
  - **Factor B — KG expansion:** `off` vs `on`
- **Reference arms** (not in the factorial) bracket the results: closed-book (lower bound),
  oracle-context (upper bound), BM25-only (lexical floor).
- **Generator held fixed** at one Thai open model for all main comparisons; a second Thai
  model is a robustness replication (§9).

---

## 3. Corpus & artifacts (already scaffolded)

| Artifact | Produced by | Role in experiment |
|----------|-------------|--------------------|
| `penal_tree.json` | `ingest/pageindex_parser.py` | search space for PageIndex arm |
| `sections.jsonl` (1 record = 1 มาตรา) | same parser | chunk unit for baseline; node source for KG; retrieval index |
| `kg_nodes/edges.jsonl`, `kg.cypher` | `kg/build_kg.py` | CITES / HAS_PENALTY / (curated) APPLIES_TO for KG arm |
| `eval/eval_set.jsonl` | annotation (§5) | the single gold set all arms are scored on |

**Corpus versioning (non-retroactivity).** Freeze one amendment version of the Code, record
its effective date, and stamp every eval item's `applicable_version_date`. Penalty amounts
and inserted sections differ across amendments; a mismatched version silently corrupts the
`penalty` question type. This is a validity requirement, not a nicety.

---

## 4. Experimental arms

Chunk unit is **one มาตรา** everywhere, so retrievers differ only in *how they select*
sections, never in granularity (removes a confound).

### 4a. Factorial arms (the four we reason about)

| ID | Base retriever | KG | Description |
|----|----------------|----|-------------|
| **A1** `HYB` | Hybrid baseline | off | **Control.** BGE-M3 dense + sparse (BM25/SPLADE) fusion → cross-encoder reranker over มาตรา chunks. |
| **A2** `HYB+KG` | Hybrid baseline | on | A1's top-k seeds → expand along CITES/APPLIES_TO one hop → re-rank the expanded pool. |
| **A3** `PI` | PageIndex | off | LLM reasons over `penal_tree.json`, descending ภาค→…→มาตรา, returns selected sections. |
| **A4** `PI+KG` | PageIndex | on | PageIndex-selected sections expanded via KG, then re-ranked. |

### 4b. Reference / control arms (bracket the story)

| ID | Arm | Purpose |
|----|-----|---------|
| **R0** `BM25` | Lexical-only top-k | Retrieval floor; shows dense/hybrid's lift. |
| **R1** `DENSE` | BGE-M3 dense-only, no reranker | Isolates the reranker's contribution inside the baseline. |
| **C0** `CLOSED` | Generator, **no context** | Parametric-memory baseline; the denominator for RQ5. |
| **C1** `ORACLE` | Generator fed exactly `gold ∪ supporting` | Generation ceiling; separates "bad retrieval" from "bad generation." |

`CLOSED` and `ORACLE` are generation-only (no retrieval metric). Every other arm is scored
on both retrieval and generation.

### 4d. Full-system arm (added post-hoc, see 2026-08-04 progress)

The 2×2 above treats PageIndex as a *replacement* base retriever (A3/A4). The first full run
showed this loses badly (A4 recall 0.49 vs hybrid 0.93) — PageIndex is too weak to *replace*
dense-hybrid retrieval. The refined full system keeps hybrid and lets structure *augment* it:

| ID | Arm | Base retriever | KG | Description |
|----|-----|----------------|----|-------------|
| **A4B** `PIKG+` | Additive full system | **hybrid ∪ PageIndex** | on | Union the hybrid pool with PageIndex-selected sections → KG one-hop → shared re-rank. Never discards the strong retriever. |

A4B is reported alongside the factorial, not inside it — the 2×2 main-effects/interaction
analysis (§7) is unchanged. A4B is the arm the thesis proposes as the deployable pipeline.

### 4c. Fair-comparison rules (pre-registered)

- **Same generator, same prompt template, temperature 0** across A1–A4, C0, C1.
- **Same k** presented to the generator (default k=5; swept in §8). KG expansion may *rank*
  more candidates but the generator still sees the top-k after re-ranking.
- **PageIndex navigation LLM = the same Thai open model** used for generation (a Claude
  navigator would confound "structure helps" with "a stronger model helps"). Its calls are
  counted in the cost metric (RQ4).
- **One shared reranker** (same cross-encoder) wherever re-ranking occurs, so re-ranking is
  not an accidental advantage of one arm.

---

## 5. Eval set construction

Follows `eval/ANNOTATION_GUIDE.md` and `eval/schema.json`; this section fixes the numbers.

- **Size:** 50-item pilot (all types, end-to-end harness debug) → scale to **≥250**.
- **Type distribution** (per guide): lookup 25% / multi_hop 30% / penalty 20% /
  definition 10% / exception 10% / unanswerable 5%. `multi_hop` is over-sampled on purpose —
  it is where H1/H2 are decided. At N=250 that is ~75 multi-hop items, the limiting subgroup
  for statistical power (§7).
- **Two annotators, 20% overlap.** Report **inter-annotator agreement** on `gold_sections`
  as **set-level Jaccard** and **Cohen's κ** on the "is section X gold?" decision. Target
  κ ≥ 0.7 before trusting the set; adjudicate disagreements.
- **Label integrity:** `eval/validate_eval.py` runs in CI — every `gold`/`supporting` id must
  exist in `sections.jsonl`; `unanswerable` ⇒ empty gold; enums/uniqueness enforced.
- **Seed caution:** items in `eval_set.seed.jsonl` are *examples*; every section number and
  penalty is re-verified against the frozen corpus version before entering the gold set.

---

## 6. Metrics

### 6a. Retrieval (scored per item, then aggregated)

Let `R_k` = the arm's ranked top-k of sections; `G` = `gold_sections`;
`G⁺` = `gold ∪ supporting_sections`.

| Metric | Definition | Answers |
|--------|-----------|---------|
| **Recall@k** | \|R_k ∩ G\| / \|G\|, k∈{1,3,5,10} | did we surface the answering section? (RQ1) |
| **MRR@10** | mean of 1/rank of first item in G | how high is it ranked? (RQ1) |
| **MAP@10** | mean average precision over G | ranking quality with multiple gold |
| **Multi-hop coverage@k** | 1 if `G⁺ ⊆ R_k` else 0, averaged | did we get *all* pieces needed to reason? (RQ2) |
| **Support recall@k** | recall computed on `supporting` only | isolates the KG/general-provision effect (RQ2) |

All reported **overall and broken down by `question_type`** — the breakdown is where H1/H2
live (the aggregate can hide it).

### 6b. Generation (fixed generator; LLM-as-judge + claim grounding)

| Metric | Definition | Answers |
|--------|-----------|---------|
| **Claim-grounding rate** | fraction of `answer_claims` entailed by the retrieved context (LLM judge, per-claim NLI-style verdict) | faithfulness; "right section, wrong penalty" is caught here |
| **Answer correctness** | LLM judge vs `reference_answer`, rubric-scored (0/1 or 1–5) | is the answer right? |
| **Hallucinated-section rate** | fraction of answers citing a มาตรา that is **not in the corpus** (hard, regex-checkable) or **not retrieved** (soft) | invented-authority rate |
| **Refusal accuracy** | on `unanswerable`: correct-refusal precision/recall/F1 | does it decline out-of-scope Qs instead of hallucinating? |

**LLM-as-judge protocol (pre-registered to keep it defensible):**
- Judge model **≠ generator model** (avoid self-preference bias); temperature 0; fixed prompt.
- Claim grounding is **per-claim binary entailment** against the concatenated retrieved
  sections — deterministic to aggregate, and the atomic `answer_claims` are exactly what the
  guide built the set for.
- **Judge validation:** a lawyer scores a **~20% random audit** of (question, answer) pairs;
  report judge–human agreement (κ / Spearman). We only trust the automated judge numbers if
  agreement is high; report the calibration in the thesis regardless. *(This is the one place
  we add lightweight human scoring — not to grade the whole set, but to license the judge.)*

### 6c. System (RQ4)

Latency p50/p95 **decomposed** into retrieval vs. generation; **LLM calls per query**
(PageIndex navigation and KG re-ranking cost real calls — this is where structure gets
expensive); token count; index build time; index storage.

---

## 7. Statistical methodology

- **Unit of analysis:** the eval item (paired across arms).
- **Primary inferential model — mixed-effects logistic regression** for the 2×2 core, on
  per-item binary outcomes (e.g. hit@5, multi-hop-covered@5):
  `outcome ~ base_retriever * kg_expansion + (1 | item)`.
  The item random intercept models the paired design; the interaction term directly tests
  "does KG help *more* on top of PageIndex than on top of hybrid?" For rate outcomes
  (claim-grounding) use a linear/beta mixed model or aggregate-then-bootstrap.
- **Descriptive uncertainty:** **paired bootstrap 95% CIs** (percentile, 10k resamples over
  items) for every headline metric and every pairwise Δ. Report Δ with CI, not just p.
- **Pairwise significance:** paired permutation test (or McNemar for binary hit@k) for the
  key contrasts: A3−A1 (RQ1), A2−A1 and A4−A3 (RQ2), each retrieval arm − C0 (RQ5).
- **Multiple comparisons:** Holm–Bonferroni across the pre-registered family of contrasts.
- **Effect sizes** reported alongside p (Δ points, odds ratios) — a thesis argues magnitude,
  not just significance.
- **Power / N justification:** paired design, target to detect a **~8–10 pt** Recall@5
  difference at 80% power, α=0.05 → ~150–200 paired items suffice overall; the **multi_hop
  subgroup (~75 items)** is the binding constraint, powered only for **~15 pt** effects.
  State this limit up front and avoid over-claiming small subgroup differences.

---

## 8. Ablations (attribute the value)

Each isolates one design choice so a reviewer can't ask "but was it really *that*?"

1. **k sweep:** k ∈ {1,3,5,10} on every arm — retrieval/generation trade-off curve.
2. **KG edge ablation:** CITES-only → +APPLIES_TO → +HAS_PENALTY. Pinpoints *which* edge
   drives multi-hop coverage (H2 predicts APPLIES_TO).
3. **Reranker on/off** in the baseline (A1 vs R1) — how much is BGE-M3 vs. the reranker.
4. **PageIndex depth / nav budget:** cap navigation steps; does more LLM reasoning keep paying?
5. **Generator robustness:** re-run A1–A4 with a *second* Thai open model — do the *rankings*
   of arms hold even if absolute scores shift? (external validity of the conclusion)

---

## 9. Models (Thai open-source, per your choice)

| Role | Candidate models | Notes |
|------|------------------|-------|
| Generator (primary) | Typhoon / SeaLLM / OpenThaiGPT — pick one, pin the exact revision | Thai-tuned, self-hostable, reproducible; the fixed generator for A1–A4/C0/C1. |
| Generator (robustness) | a second of the above | ablation §8.5. |
| PageIndex navigator | **= primary generator** | keeps "structure helps" from becoming "bigger model helps." |
| Embedder | **BGE-M3** (dense+sparse) | baseline backbone; multilingual, strong on Thai. |
| Reranker | a multilingual cross-encoder (e.g. BGE-reranker-v2-m3) | shared by all re-ranking arms. |
| LLM judge | a model **different from the generator** | avoid self-preference; validated vs human audit (§6b). |

*(Exact revisions/quantization are pinned in the reproducibility appendix at implementation
time; the design only requires that they be fixed and reported.)*

---

## 10. Threats to validity & mitigations

| Threat | Mitigation |
|--------|-----------|
| **Pretraining contamination** — the Penal Code is public; the LLM may "know" answers without retrieval. | `CLOSED` arm (C0) measures parametric memory; all retrieval claims are reported *net of* it (RQ5). Penalty/inserted-section (ทวิ) items stress cases memory gets wrong. |
| **Single-statute external validity.** | Framed honestly as a Thai-Penal-Code study; §8.5 generator swap tests conclusion stability; note transferability to other authored statutes as future work. |
| **LLM-judge bias / noise.** | Judge ≠ generator; temp 0; per-claim binary entailment; **human audit calibration** (§6b). |
| **`APPLIES_TO` curated by one lawyer.** | Treat as a labeled resource; spot-check a sample with a second reader; the edge ablation (§8.2) shows exactly how much rides on it. |
| **Seed-label errors / wrong amendment.** | Frozen corpus version + `validate_eval.py` in CI + re-verification of every seed number. |
| **Thai numeral/tokenization** (๓๓๕/๑, ทวิ). | Canonical section-id normalization already in parser/KG; add a unit test asserting round-trip id equality across parser, KG, and eval labels. |
| **Reranker as hidden confound.** | One shared reranker across all re-ranking arms; reranker on/off ablation (§8.3). |

---

## 11. Reproducibility

- Pin: model revisions, quantization, embedding + reranker versions, decoding params
  (temperature 0), k, random seeds, corpus amendment date.
- Version and release: `eval_set.jsonl`, all prompts (navigation, generation, judge),
  the frozen corpus artifacts, and the run configs.
- Every number in the thesis regenerable by one command from pinned configs.
- Log raw per-item outputs (retrieved ids, generated answer, judge verdicts) so metrics can
  be recomputed without re-running models.

---

## 12. Phased execution plan

| Phase | Deliverable | Exit criterion |
|-------|-------------|----------------|
| **P0. Corpus freeze** | full Penal Code parsed; `sections.jsonl` + `penal_tree.json`; version dated | parser round-trips all มาตรา; id normalization test passes |
| **P1. Pilot eval set** | 50 items, all types, validated | `validate_eval.py` green; harness runs one arm end-to-end |
| **P2. Baselines up** | R0/R1/A1 retrieval + C0/C1 generation | metrics + bootstrap CIs reproduce on pilot |
| **P3. Structural arms** | A3 (PageIndex), A2/A4 (KG); `APPLIES_TO` curated | 2×2 runs on pilot; sanity vs baseline |
| **P4. Scale eval set** | ≥250 items; 20% overlap; IAA reported | κ ≥ 0.7; distribution matches target |
| **P5. Full run** | all arms × full set; ablations §8 | all pre-registered contrasts computed |
| **P6. Judge validation** | 20% human audit; judge–human agreement | agreement reported; judge numbers licensed |
| **P7. Analysis & write-up** | mixed-model + bootstrap results, per-type breakdown, cost curves | RQ1–RQ5 answered with effect sizes + CIs |

---

## 13. What "success" looks like (a priori)

The thesis has a clean result **whatever the outcome**, because the design attributes value
per layer:

- If **H1/H2 hold** → structure-aware retrieval measurably beats a strong Thai dense-hybrid
  baseline on statutory QA, with the gain localized to the question types theory predicts.
- If they **don't** → "authored hierarchy adds little once you have a strong multilingual
  hybrid retriever on well-chunked sections" is itself a publishable, defensible negative
  result — and the closed-book/oracle brackets explain *why*.

The pre-registered arms, metrics, and tests here are what make either outcome a contribution
rather than a vibe.
```
