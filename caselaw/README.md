# Case-law extension — Supreme Court citations → court-weighted provision graph

Pipeline that layers Thai Supreme Court (Dika) citation data onto the existing penal-code
corpus, for the thesis extension (tasks A–C). Everything here builds on the existing
`data/sections.jsonl` (444-section Book–Title–Chapter tree); it does not rewrite it.

## Source

- Dataset: **`phoneee/thai-legal-corpus`** (Hugging Face, CC-BY-4.0, built 2026-03-10).
- 176,543 documents total; **127,012** are `supreme_court`. Also ships pre-built metadata:
  - `metadata/deka_citations.jsonl` — per-case `laws_cited` + `sections_cited` (already parsed
    to Arabic-numeral strings like `"ป.อ. มาตรา 269/5"`).
  - `metadata/law_section_index.json` — reverse index (section → citing-case count); the Penal
    Code entry (`ป.อ.`) lists 1,300 distinct cited section-strings, `total_cases` 102,245.
- Raw downloads live in `data/caselaw/raw/` (gitignored, regenerable). Derived JSONL is committed.

## A1 — load & filter (criminal cases)

Criminal cases = those whose citations reference the current Penal Code (`ประมวลกฎหมายอาญา`).
Of 127,477 citation records, **23,862 (18.7%)** cite the Penal Code; 10,630 cite *only* the
Penal Code. Top co-cited codes: ป.พ.พ. (civil), ป.วิ.พ., ป.วิ.อ. (criminal procedure).

**Two conventions to watch (reconciled in code):**
- **Year.** The corpus `year` field is **CE** (`1972`); citation `case_year` is **BE** (`2568`).
  We store both (`year_be`, `year_ce = year_be − 543`).
- **doc_id.** Citations use `deka-<n>-<BE year>` (e.g. `deka-7946-2568`); the full-text corpus
  uses a sequential `deka-<NNNNN>`. They do **not** join on `doc_id`. We join on **`case_id`**
  parsed from the judgment header `คำพิพากษาศาลฎีกาที่ <case_id>`.

**Year distribution (Penal-citing cases).** Heavily skewed to older cases: 73.7% are CE < 1997;
only 1 case has an implausible year. Candidate graph/held-out splits (Y0 = advisor's call):

| Y0 | graph set (≤Y0) | held-out (>Y0) |
|----|----------------:|---------------:|
| BE 2560 / CE 2017 | 22,989 (96%) | 873 (4%) |
| BE 2562 / CE 2019 | 23,196 (97%) | 666 (3%) |
| BE 2564 / CE 2021 | 23,450 (98%) | 412 (2%) |
| BE 2565 / CE 2022 | 23,579 (99%) | 283 (1%) |

Any of these leaves enough post-Y0 cases for the ~50-case pilot test set (task D). **Recommend
Y0 ≈ CE 2019–2021** (balances a large graph set against a non-trivial held-out tail).

## A2 — Penal-Code section extractor (`extract_citations.py`, `sections.py`)

Because `sections_cited` is already parsed, A2 is **normalization + Penal-Code filtering**, not
raw parsing. Per cite string: keep only `ป.อ.` (drops ป.วิ.อ., civil codes, and the pre-1957
`กฎหมายลักษณะอาญา`), convert Thai numerals, drop `วรรค…`/`(1)` to section level, and canonicalize
to the corpus id scheme (plain `288`, slash `269/5`, word `277 ทวิ`).

A context-aware **raw-text extractor** (`extract_from_text`) is also provided for cross-checking
(A3) and future use: it tracks the nearest law marker so a bare `มาตรา 288, 80` run is only kept
under a Penal-Code context.

Output `data/caselaw/penal_citations.jsonl`: **23,573 cases**, **433 distinct section keys**
(down from 1,300 raw noise). `{deka_id, case_id, year_be, year_ce, sections, raw_mentions}`.

## A4 — map to corpus section_ids (`map_sections.py`)

`CorpusSectionIndex` learns the `ทวิ→/2, ตรี→/3` convention from the corpus headings themselves
(`มาตรา ๒๗๗ ทวิ` ↔ `section_id 277/2`), so nothing is hardcoded. Result:

- **402 / 433** distinct keys map (92.8%); **macro per-case coverage 0.999**.
- The 31 unmapped keys are provable noise — civil-code sections mis-tagged `ป.อ.` (726, 849,
  1620…), OCR garbage (`7071`, `131226`, a leaked year `2566`), and non-current sections
  (`59 ทวิ`, `44/1`). The corpus-membership check thus doubles as a noise filter.

Output `data/caselaw/penal_citations.mapped.jsonl` adds `section_ids` + `unmapped` per case.

## A3 — QC (`qc_extract.py`)

Joins cases to judgment text by `case_id`, samples 100 (seeded), and compares the curated
dataset field against the independent full-text regex. The two measure different targets:

- **Dataset field = operative / head-note citations** (what the case turns on).
- **Full-text regex = every Penal mention**, including sentencing/procedural boilerplate
  (§§78, 90, 91 combine-penalties; 32–33 confiscation; 371).

On a seeded 100-case sample (1,193 joinable) the head-note field is well **corroborated** by the
text — **micro 0.956 / macro 0.961** of listed sections are found in the judgment (only 9 not
found) — while the full-text sweep *over-captures* relative to it (precision ≈ 0.49, ~200
sentencing/procedural extras). For the co-citation **offence** graph (task B) the curated field is the right
signal — the full-text extras would pollute weights. Disagreements (incl. a few cases where the
dataset tags *old-code* sections as Penal, where our extractor is arguably more correct) are
written to `data/caselaw/qc_sample.csv` for **lawyer adjudication (task F)** — the authoritative
precision check.

**Lawyer worksheet (`lawyer_sheet.py`).** `qc_sample.csv` is a dev parser-vs-parser diff; the
reviewer-facing artifact is `data/caselaw/lawyer_qc_sheet.xlsx` — one row per case showing the
sections our production path extracted + a focused judgment excerpt around every มาตรา/พิพากษา
mention, with a ✓/✗ dropdown and blank *missed* / *over-included* columns. Cases where the
full-text regex surfaced a substantive (non-boilerplate) section the curated field lacks, or
where a key failed to map, are flagged ★ ต้องดูพิเศษ and sorted first (42/100 in the default
seeded sample). **This is the file to hand the lawyer.**

## B — court-weighted provision graph (`weight_graph.py`)

Weights the 13,296 rule-derived APPLIES_TO edges by how often their endpoints are **co-cited in
real judgments**, then prunes edges courts never use. Leakage control: co-citations are counted
only over the **graph set** (case year ≤ Y0); year > Y0 is reserved for the held-out test (D).

Weight = normalized co-citation with a support floor (locked decisions):
`w(u,v) = c(u,v)/min(c(u),c(v))` if `c(u,v) ≥ min_support` else 0. The floor (default 5) is
essential — without it, pairs co-cited once get w = 1.0 and outrank high-frequency legal links.

Result (Y0 = CE 2019 provisional, min_support 5, over 22,914 graph cases):

| τ | APPLIES_TO kept | % of 13,296 | `all_offences` pruned |
|---|---------------:|------------:|----------------------:|
| 0.00 | 487 | 3.7% | 95.7% |
| 0.05 | 334 | 2.5% | 96.9% |
| 0.10 | 187 | 1.4% | 98.2% |
| 0.20 | 66 | 0.5% | 99.4% |

So the rule-materialized graph is ~96% court-unsupported noise; ~500 general→offence links
carry real co-citation weight. Top edges are legally coherent: `80→288` attempt→murder
(co-cited 833×), `68→288` self-defence→murder, `69→288` necessity→murder, `59→288`
liability→murder, `90/91→…` concurrence. Output `data/caselaw/applies_to_weighted.jsonl` adds
`{w, c_u, c_v, c_uv}` per edge.

`retrieval.kg_expand.KGExpander` consumes it: `weighted_applies_path=…, tau=…` keeps `w ≥ τ`
edges with their weights (CITES untouched at w=1.0); `rule_based=True` reverts to the unweighted
graph for ablation; `weighted_neighbors(sid)` feeds task C. Wired via `retrieval/config.json`
keys `weighted_applies_to_path`, `kg_tau`, `kg_rule_based` (all optional → defaults to old
behaviour). τ is tuned on dev only (task E).

## C — fact-guided traversal (`facts.py` + extended `FullSystem`)

The reserved KG slot now picks a *legally relevant* general provision instead of any co-cited
section. One Qwen call maps the question's facts to the general-part chapters
`F(q) ⊆ {4,5,6,7,8}` (ม.59–94); the chapter→section map is read from the corpus `chapter`
metadata (`caselaw.facts.load_chapter_sections`), not hardcoded. Traversal (in
`retrieval.arms.FullSystem._fact_guided`):

1. seed = reranked primary pool `P(q)` (hybrid ∪ PageIndex), same as PIKG-RAG;
2. one hop → `expander.weighted_neighbors(s)` for each top-k primary;
3. keep only neighbours whose chapter ∈ `F(q)` (the fact filter);
4. rank by court-citation weight `w`; the reserved slot = heaviest `r_g ∉ P(q)` (else keep `r5`).

Ablation switches (config): `use_fact_filter=false` (skip step 3), `random_slot=true` (random
eligible neighbour, seeded), or build the expander `kg_rule_based=true` (unweighted edges). Per
query, `meta` logs `fact_llm_calls / fact_tokens / fact_latency / fact_categories / kg_reserved`.

Enable via `retrieval/config.json`: `"fact_guided": true`, `"weighted_applies_to_path": …`,
`"kg_tau": …`, and `backends.fact_extractor = "backends_deepinfra:DeepInfraFactExtractor"`.
The F(q) prompt (`caselaw.facts.FACT_SYSTEM` + `build_fact_user_prompt`) is kept verbatim for
Appendix A (the 7th prompt). Worked example: offence primaries {288,289,334,335,340} with
`F(q)=[5]` (attempt) → reserved slot **80** (attempt→murder, w=0.59); `F(q)=[8]` → **93**
(recidivism); `F(q)=[4]` → a หมวด-4 liability provision.

## Reproduce

```bash
# 1. fetch metadata (raw, gitignored)
mkdir -p data/caselaw/raw
B=https://huggingface.co/datasets/phoneee/thai-legal-corpus/resolve/main
curl -sL $B/metadata/deka_citations.jsonl -o data/caselaw/raw/deka_citations.jsonl

# 2. A2 extract → A4 map
python -m caselaw.extract_citations
python -m caselaw.map_sections

# 3. A3 QC (needs a data split with judgment text for the join)
curl -sL $B/data/val.jsonl -o data/caselaw/raw/val.jsonl
python -m caselaw.qc_extract --n 100 --seed 20261006
python -m caselaw.lawyer_sheet --n 100 --seed 20261006   # -> lawyer_qc_sheet.xlsx

# 4. B court-weighted graph (Y0 provisional; re-run once the advisor locks Y0)
python -m caselaw.weight_graph --y0-ce 2019 --min-support 5
```
