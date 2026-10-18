# PIKG-RAG — Thai Criminal-Law Question Answering

Code and data for **"PIKG-RAG: Augmented Hybrid Retrieval System of Thai Criminal Law"**
(iSAI-NLP 2026). Corpus: the Thai Penal Code (**ประมวลกฎหมายอาญา**) only. The system combines
three ingredients over one shared corpus — a dense+sparse **hybrid** retriever, **PageIndex**
hierarchy navigation, and a rule-derived **provision graph** — and measures each one's
contribution against the same evaluation set.

> This repository reproduces the results in the paper. For the exact paper state, check out the
> release tag `v1.0-isai-nlp2026`. Development continues on other branches.

## Arms (paper Table I ↔ code id)

The full system is **additive**: structure is added on top of hybrid, not in place of it.
PIKG-RAG joins the HYBRID and PageIndex pools, reranks them, keeps the top k−1 sections, and
fills the last slot with the best new section found one graph step away (reserved slot).

```
PIKG-RAG = (HYBRID ∪ PageIndex) → rerank → top k−1 + reserved-slot KG section
```

| Paper name | code id | description |
|------------|---------|-------------|
| CLOSED | `C0` | closed-book (no retrieval) — floor |
| BM25 | `R0` | lexical only |
| DENSE | `R1` | dense (BGE-M3) only |
| HYBRID | `A1` | dense+sparse fusion → rerank — **baseline** |
| PI | `A3` | PageIndex tree navigation only (replacement setting) |
| HYBRID+PI | `A5` | hybrid ∪ PageIndex → rerank |
| HYBRID+KG | `A2` | hybrid + reserved-slot graph section |
| **PIKG-RAG** | `A4B` | **hybrid ∪ PageIndex + reserved-slot graph section** (full system) |
| ORACLE | `C1` | gold sections in context — ceiling |

These are the nine conditions reported in the paper. The code also includes `A4` (PI+KG:
PageIndex + graph expansion), which is **not reported in the paper**.

Older `B0–B3` CLI aliases still resolve (see `retrieval/arms.py:OLD_ALIASES`), but the paper
uses the names above.

## Models (all via DeepInfra, temperature 0)

Three distinct vendors, so the judge never grades its own drafter or generator:

| role | model |
|------|-------|
| generator + PageIndex navigator | `Qwen/Qwen3.6-35B-A3B` |
| eval-set drafter | `deepseek-ai/DeepSeek-V4-Pro` |
| LLM judge | `google/gemma-3-27b-it` |
| embedder / reranker | `BAAI/bge-m3` / `Qwen/Qwen3-Reranker-4B` |

Set `DEEPINFRA_API_KEY`. Override any model with the matching `DEEPINFRA_*_MODEL` env var.

## Corpus

Pinned to PyThaiNLP thai-law `criminal-csv-v0.1` → `data/raw/criminal-datasets.csv`. Ingested
into **444 section chunks** (one มาตรา each): **398 base sections (มาตรา 1–398) plus 46 inserted
sections** rendered with a slash id, e.g. `269/1`, `277/2` (= มาตรา ๒๗๗ ทวิ). The release does
not state the law's latest amendment date, so penalties refer to this copy.

## Knowledge graph

- `CITES` (719 edges) are extracted from explicit cross-references in the text.
- `APPLIES_TO` (13,296 edges) point from a Book-1 general rule to an offense (Book 2 or 3).
  They are produced by scope rules — mainly `ALL_OFFENCES` (8,932), `by_penalty` (3,080) and
  `book2` (1,104), plus two minor rules (180). 44 pairs are produced by more than one rule.

All `APPLIES_TO` edges are **rule-materialized and currently unverified** — every edge carries
`verified=false`. The code exposes `--verified-only` / `verified_applies_to_only=True`, but
with no lawyer review done yet it selects an empty set. Do **not** read the flag as "edges
have been checked."

## Statistics

The paper reports paired bootstrap 95% CIs, McNemar's exact test, and Holm–Bonferroni
correction. `analysis/` also contains a bootstrap difference-in-differences, which is **not
reported in the paper**. There is no GLMM.

## Pipeline

```
data/raw/criminal-datasets.csv
   │ ingest/csv_to_sections.py (+ data/hierarchy.json)
   ▼
sections.jsonl + penal_tree.json
   ├─► kg/build_kg.py           → CITES + structure edges
   ├─► kg/expand_applies_to.py  → APPLIES_TO candidate edges (unverified)
   └─► retrieval arms ─► generation ─► LLM judge ─► analysis
```

## Quick start

```bash
export DEEPINFRA_API_KEY=...        # pip install openai requests

# 1. ingest CSV -> structure-aware corpus
python ingest/csv_to_sections.py data/raw/criminal-datasets.csv \
       --hierarchy data/hierarchy.json --out-dir data

# 2. build the graph (CITES + unverified APPLIES_TO candidates)
python kg/build_kg.py data/sections.jsonl --out-dir data
python kg/expand_applies_to.py data/sections.jsonl \
       --rules kg/applies_to_rules.json --out-dir data

# 3. draft the gold-by-construction eval set (drafter = DeepSeek-V4-Pro via DeepInfra)
python eval/generate_eval.py --dry-run                      # plan only, no API call
python eval/generate_eval.py --out eval/eval_set.generated.jsonl --workers 4
python eval/validate_eval.py eval/eval_set.generated.jsonl \
       --schema eval/schema.json --sections data/sections.jsonl

# 4. retrieval → generation → judge (use paper names or code ids)
python -m retrieval.run_eval        --arms all --eval eval/eval_set.generated.jsonl
python -m generation.run_generation --arms all --eval eval/eval_set.generated.jsonl
python -m generation.run_judge      --arms all --eval eval/eval_set.generated.jsonl

# 5. statistics: paired bootstrap CIs + McNemar/Holm
python -m analysis.run_analysis --metric correct --arms all
```

## Layout

```
ingest/      CSV → sections.jsonl + penal_tree.json
kg/          CITES + APPLIES_TO graph construction
retrieval/   arms (A1–A5, R0/R1, C0/C1), BM25, PageIndex, KG expansion, metrics
generation/  generator + LLM-judge harness (correctness, grounding, refusal)
analysis/    bootstrap, McNemar/Holm, plots
eval/        schema + validator + generator for the gold eval set
data/        corpus, tree, graph edges
runs/        result snapshots (per-arm rankings, answers, verdicts)
```

## Citation

```
W. Pilasuk and N. Hongwarittorn, "PIKG-RAG: Augmented Hybrid Retrieval System of Thai
Criminal Law," in Proc. iSAI-NLP 2026.
```

## License

Code is released under the **MIT License** (see `LICENSE`). The Thai Penal Code text under
`data/raw/` is Thai statutory text (not a copyright work under Thai Copyright Act B.E. 2537
§7(1)), redistributed via PyThaiNLP `thai-law`; see `data/raw/SOURCE.txt` for attribution.
