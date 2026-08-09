# Structure Augments, It Does Not Replace: Additive Structure-Aware Retrieval for Thai Statutory Question Answering

**Draft — sections with complete experimental support. Placeholders marked `[TODO]`.**
Numbers are from the locked run `runs/2026-08-07/` (n=255 eval, DeepInfra models).

---

## Abstract

We study whether the hierarchical structure of a legal code and a citation/application
knowledge graph (KG) improve retrieval-augmented question answering (RAG) over the Thai
Penal Code. We compare a strong dense–sparse **hybrid** retriever against two structural
signals — **PageIndex** tree navigation and one-hop **KG** expansion — on a 255-item,
gold-by-construction benchmark spanning six question types, with a fixed Thai open-model
generator and a separate-vendor LLM judge. Our central finding is a design principle:
structural signals help **only when they augment the hybrid retriever, not when they replace
it**. A PageIndex-only pipeline reaches just 0.350 answer accuracy; naively substituting
PageIndex for hybrid in the "full" system makes it the *worst* real configuration. Recast
**additively** — union the hybrid and PageIndex pools, then append KG-selected supporting
provisions into a reserved rank slot — the full system (`HYBRID+PI+KG`) becomes the best real
arm at **0.718** answer accuracy, the only configuration to beat the hybrid baseline with
Holm-corrected significance (Δ = +0.082, p = 0.006), closing 35% of the gap to an oracle-context
ceiling (0.809). Four ablations and a fault decomposition show the residual gap is bounded by
**KG edge precision** — the correct supporting provision is 100% reachable but out-ranked among
~271 over-generic `APPLIES_TO` neighbours — not by candidate depth, context size, or the
generator. We release the benchmark, pipeline, and a reproducible run snapshot.

**Contributions.** (1) An additive structure-aware RAG design with a *reserved-slot* KG
expansion that adds multi-hop context without displacing primary evidence. (2) A controlled
2×(hybrid/PI) × (KG on/off) study on Thai statutory QA with paired significance testing.
(3) A fault decomposition localising the remaining error to KG edge precision, turning a data
limitation into a concrete, actionable direction.

---

## 1. Introduction

Statutory question answering is retrieval-bound: a language model rarely recalls exact section
numbers, penalty figures, or inserted provisions (ทวิ/ตรี) reliably, so answer quality hinges on
surfacing the right มาตรา. The Thai Penal Code adds two structural affordances a flat retriever
ignores: a **hierarchy** (ภาค → ลักษณะ → หมวด → มาตรา) and a web of **cross-references and general
provisions** (limitation, principals/accessories, attempt, aggravating/mitigating factors) that a
correct answer must combine with the operative offence.

We ask whether exploiting this structure improves RAG, via five questions: **RQ1** does PageIndex
tree navigation beat a dense–sparse hybrid at finding the answering section; **RQ2** does one-hop
KG expansion recover the general provisions needed for multi-hop reasoning; **RQ3** does better
retrieval translate into a more faithful generated answer under a fixed generator; **RQ4** what
does each structural layer cost; **RQ5** how much does retrieval matter at all given likely
pre-training exposure.

Our answer to RQ1 is nuanced and, we believe, the paper's main lesson: **PageIndex loses badly as
a replacement for hybrid retrieval, but contributes as an additive signal on top of it.** The
same holds for KG. This reframes "structure-aware retrieval" from *choosing* a structural
retriever to *composing* structural signals with a strong lexical–dense base.

---

## 2. Related Work

`[TODO: cite — RAG (Lewis et al.); dense retrieval (BGE-M3); hierarchical/tree retrieval and
PageIndex-style navigation; KG-augmented retrieval; legal IR and legal QA; LLM-as-judge and its
validation; Thai NLP / Thai legal NLP.]`

---

## 3. Method

### 3.1 Corpus and artifacts
We use the Thai Penal Code (ประมวลกฎหมายอาญา) from PyThaiNLP `criminal-csv-v0.1`, **444 sections**.
An ingest pipeline produces: `sections.jsonl` (one record per มาตรา — the shared chunk unit for
every retriever, removing granularity as a confound); `penal_tree.json` (the ภาค→…→มาตรา hierarchy,
the search space for PageIndex); and a knowledge graph of `CITES` edges (719) and `APPLIES_TO`
edges (13,296) linking general provisions to the offences they govern.

### 3.2 Evaluation set
We construct **255 gold-by-construction items** across six types — lookup (75), penalty (55),
multi_hop (45), unanswerable (35), exception (25), definition (20) — each carrying `gold_sections`
(the directly answering มาตรา), `supporting_sections` (general provisions required to reason), a
reference answer, and answer claims. Items are drafted from the statute so ground truth is
determined by construction rather than post-hoc annotation. `[TODO: brief generation/validation
protocol; note unanswerable is excluded from the answer-accuracy analysis, see §8.]`

### 3.3 Retrieval components
- **Hybrid.** BM25 (lexical) fused with BGE-M3 dense similarity (min–max normalised, α = 0.5),
  surfacing a pool of `POOL = 50` candidates, then a Qwen3 cross-encoder reranker.
- **PageIndex (PI).** The generator LLM descends `penal_tree.json` with beam width 5, selecting
  sections by reasoning over node headings.
- **KG.** One-hop expansion along `CITES` ∪ `APPLIES_TO` from a seed set, growing the candidate
  pool with cited/applicable provisions before reranking.

### 3.4 Additive system design and reserved-slot KG
The key design choice is **additive composition**. Rather than treating PageIndex as an
alternative *base* retriever (a replacement 2×2), we union the hybrid pool with PageIndex-selected
sections and rank the result — hybrid's recall becomes a floor the structural signal can only
raise. KG is then applied as a **reserved-slot** append: we rank the primary (hybrid ∪ PI) pool
cleanly, take its top `k−1`, and append the single best KG-surfaced supporting provision into the
`k`-th slot. This lets KG add multi-hop context *without ever displacing a primary gold section*,
and mirrors the oracle's `gold ∪ supporting` context construction. When KG surfaces nothing new,
the arm falls back to the top-`k` primary ranking.

### 3.5 Experimental arms
An additive design over {hybrid, PI, KG}, plus references bracketing the range:

| Arm | Definition |
|-----|------------|
| `CLOSED` | generator, no context (parametric-memory floor) |
| `BM25` / `DENSE` | lexical-only / dense-only retrieval |
| `HYBRID` | dense–sparse fusion + rerank (**baseline**) |
| `PI` | PageIndex only (structure as a *replacement* base) |
| `HYBRID+PI` | additive union of hybrid and PageIndex |
| `HYBRID+KG` | hybrid + one-hop KG expansion |
| `HYBRID+PI+KG` | full additive system + reserved-slot KG (**recommended config.**) |
| `ORACLE` | generator fed `gold ∪ supporting` (upper bound) |

### 3.6 Generation and judging
A single Thai open model, **Qwen3.6-35B-A3B** at temperature 0, generates all answers from the
top-`k=5` retrieved sections with one shared prompt across arms (no question-type leakage). The
**same** model drives PageIndex navigation (no cross-arm confound). A **different-vendor** judge,
**DeepSeek-V4-Pro**, scores answer correctness (binary + 1–5) and per-claim grounding, avoiding
self-preference bias. Embedder: BGE-M3; reranker: Qwen3-Reranker-4B. All models served on a
single provider (DeepInfra).

### 3.7 Metrics and statistics
Retrieval: recall@5, MRR@10, coverage@5 over `gold ∪ supporting`, and support-recall@5 over
`supporting`. Generation: judge answer-accuracy, claim-grounding rate, hallucinated-section rate.
Every item passes through every arm (**within-subjects/paired**). We report 95% **paired bootstrap**
CIs (10k resamples), **McNemar** exact tests for paired binary contrasts, and **Holm–Bonferroni**
FWER control over the contrast family.

---

## 4. Results

Retrieval is scored on the 220 answerable items; generation on all 255. `HYBRID`, `HYBRID+KG`, and
the reference arms do not use the navigator and are byte-identical under the locked config; the
PageIndex-based arms are run at beam 5.

**Table 1. Main results.**

| Arm | recall@5 | MRR@10 | cov@5 | sup@5 | acc% | claim | score |
|-----|---------:|-------:|------:|------:|-----:|------:|------:|
| CLOSED *(floor)* | — | — | — | — | 0.023 | 0.023 | 1.10 |
| BM25 | 0.609 | 0.457 | 0.527 | 0.178 | 0.445 | 0.556 | 2.75 |
| DENSE | 0.832 | 0.687 | 0.682 | 0.022 | 0.614 | 0.730 | 3.38 |
| HYBRID *(baseline)* | 0.927 | 0.824 | 0.800 | 0.311 | 0.636 | 0.826 | 3.51 |
| PI | 0.518 | 0.444 | 0.400 | 0.133 | 0.350 | 0.454 | 2.39 |
| HYBRID+PI | 0.950 | 0.838 | 0.827 | 0.378 | 0.673 | 0.853 | 3.61 |
| HYBRID+KG | 0.936 | 0.829 | 0.826 | 0.422 | 0.655 | 0.838 | 3.61 |
| **HYBRID+PI+KG** *(full)* | 0.932 | 0.833 | **0.850** | **0.556** | **0.718** | 0.841 | **3.81** |
| ORACLE *(ceiling)* | — | — | — | — | 0.809 | 0.939 | 4.13 |

Hallucinated-section rate ≈ 0 across all arms.

**Replacement fails; additive wins (RQ1).** `PI` alone reaches only 0.350 accuracy and 0.518
recall — far below `HYBRID` (0.636 / 0.927). Structure as a *replacement* base is strictly worse.
Recast additively, `HYBRID+PI` lifts both recall (0.950, the best of any arm) and accuracy (0.673).

**KG helps multi-hop coverage (RQ2).** Reserved-slot KG raises support-recall@5 from 0.378
(`HYBRID+PI`) to **0.556** and coverage@5 to 0.850, the highest of any arm, while lifting accuracy.

**Retrieval propagates to answers (RQ3, RQ5).** Accuracy tracks retrieval quality across the
ladder; `CLOSED` answers 2.3% correctly vs 71.8% for the full system, with an 80.9% oracle ceiling.

**Table 2. Paired contrasts on answer accuracy** (Δ, 95% bootstrap CI, McNemar p, Holm).

| Contrast | Δ | 95% CI | p | Holm |
|----------|---:|:------:|:--:|:----:|
| HYBRID+PI+KG − HYBRID | +0.082 | [+0.027, +0.136] | 0.006 | **sig.** |
| HYBRID+PI+KG − HYBRID+KG | +0.064 | [+0.009, +0.118] | 0.029 | n.s. |
| HYBRID+PI+KG − HYBRID+PI | +0.045 | [−0.005, +0.100] | 0.121 | n.s. |
| HYBRID+PI − HYBRID | +0.036 | [−0.014, +0.082] | 0.201 | n.s. |
| HYBRID+KG − HYBRID | +0.018 | [−0.027, +0.064] | 0.572 | n.s. |

The full system is the **only** arm that beats the hybrid baseline after Holm correction. Neither
structural component alone is significant; significance emerges only when they are **combined** —
direct evidence for the additive thesis. We do *not* claim `HYBRID+PI+KG` strictly dominates
`HYBRID+PI` (Δ = +0.045, n.s.); at n = 220 the two are statistically tied, but only the full system
has a significant edge over baseline.

---

## 5. Ablations

Four interventions attempt to push past 0.718; **all fail, and all for the same reason** — they
feed more into KG expansion, amplifying over-generic `APPLIES_TO` edges (mean ~271 neighbours/node,
0 verified) that drown the one relevant provision.

| Ablation | Setting | acc% | vs. base |
|----------|---------|-----:|---------:|
| Context size | k=5 / k=3 / k=2 | 0.668 / 0.668 / 0.636 | trimming ≤ 0 |
| Multi-hop reasoning prompt | off / on | 0.718 / 0.700 | −0.018 (rejected) |
| KG reserved slots | 1 / 2 | 0.718 / 0.702 | −0.018 |
| Candidate pool depth | 50 / 100 | 0.718 / 0.686 | −0.032 |
| KG slot gating | static / adaptive | 0.718 / 0.686 | −0.032 (recall@5 ↑0.955) |

Notably, `POOL=100` *raises* recall@5 (0.932 → 0.945) yet *lowers* accuracy — the larger seed
expands into more KG noise and `multi_hop` collapses (−0.111). **Recall is not the binding
constraint; KG precision is.**

**A recall–accuracy frontier.** recall@5 and MRR@10 credit only the primary answering section,
whereas the reserved slot deliberately spends one of `k` positions on a KG supporting provision.
The full system is therefore best on coverage@5 (0.850) and support-recall@5 (0.556) but *not* on
recall@5 (0.932 vs 0.950 for `HYBRID+PI`). An **adaptive** gate that reserves the KG slot only
when its candidate out-scores the cut-off primary attains the highest recall@5 of any arm
(**0.955**) — yet accuracy *falls* to 0.686, because it withholds relevant-but-lower-scored
provisions the multi-hop items need (support-recall 0.556 → 0.400). The two objectives are not
jointly maximisable: forcing the supporting provision in is accuracy-optimal, gating it out is
recall-optimal. We report the accuracy-optimal configuration; its recall@5 shortfall is a
deliberate reallocation of retrieval budget from primary to supporting evidence.

**Multi-hop fault decomposition.** `multi_hop` accuracy is 0.489 vs 0.778 oracle. Of the errors,
33% are retrieval faults (supporting provision absent from top-5) and 18% are generation faults
(present but unused). Every missing supporting provision is **100% reachable** in the KG from the
gold section — it is simply out-ranked among the ~271 neighbours; and 51% are reachable only via
generic `all_offences` edges, so naive pruning is not viable. The bottleneck is edge *precision*,
addressable by curation, not by a pipeline hyperparameter.

---

## 6. Discussion

The results recommend a shift in how "structure-aware retrieval" is framed: not as selecting a
structural retriever, but as **composing structural signals over a strong base**. PageIndex and KG
each encode real legal structure, yet each underperforms hybrid alone; their value is realised only
additively, and only when integration preserves primary evidence (the reserved slot). The full
system's significance-over-baseline arising *only* from the combination suggests the structural
signals are complementary rather than redundant, despite pairwise ties at this sample size.

The ceiling analysis is equally instructive: with the recommended system, primary retrieval is nearly
solved (recall 0.932; `POOL=100` shows it can reach 0.945), and the residual gap to oracle is
concentrated in multi-hop reasoning and bounded by KG edge precision. This localises future effort
precisely.

---

## 7. Limitations

- **Judge validation.** Answer-accuracy and claim-grounding derive from an LLM judge. It is a
  different vendor from the generator (no self-preference), but human agreement is `[TODO: report
  Cohen's κ overall and per type from the 150-item blind audit]`.
- **KG edges are unverified** (`verified = 0` of 13,296 `APPLIES_TO`). RQ2 uses automatically
  extracted edges; the over-generic `all_offences` scope is the identified precision bottleneck.
- **Single generator, single corpus, single statute version.** Robustness replication with a
  second Thai model and generalisation beyond the Penal Code are left to future work.
- **Statistical power.** At n = 220 the top additive arms cannot be separated; the full system's
  advantage is established over the baseline, not over `HYBRID+PI`.
- **Scope.** We report answerable QA; the `unanswerable`/refusal analysis is deferred (its
  deterministic refusal metric is not yet reliable).

---

## 8. Future Work

Lawyer-curated `APPLIES_TO` edges (pruning generic scopes, verifying the specific ones) to lift
the multi-hop precision ceiling; a second-annotator (legal) pass for inter-annotator agreement;
GLMM interaction estimation; an RQ4 cost/latency roll-up; a second Thai generator and a second
legal corpus for robustness and generalisation.

---

## 9. Conclusion

On Thai statutory QA, hierarchical and knowledge-graph structure improve retrieval-augmented
answering **only additively**: replacing a strong hybrid retriever with structure is strictly
worse, whereas composing structure over it — with KG appended into a reserved rank slot — yields
the only configuration to beat the hybrid baseline with corrected significance, at 0.718 accuracy
against an 0.809 ceiling. Ablations localise the remaining gap to knowledge-graph edge precision,
a curation problem rather than a modelling one.

---

*Artifacts: pipeline, 255-item benchmark, and the locked run snapshot (`runs/2026-08-07/`) with
per-arm rankings, judge verdicts, bootstrap reports, and figures are released for reproduction.*
