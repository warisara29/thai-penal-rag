# PIKG-RAG — Thai Criminal-Law Question Answering

Code and data for **"PIKG-RAG: Augmenting Hybrid Retrieval with Hierarchical and Graph
Structure for Thai Criminal Law Question Answering."** Corpus: the Thai Penal Code
(**ประมวลกฎหมายอาญา**) only. The system combines three ingredients over one shared corpus —
a dense+sparse **hybrid** retriever, **PageIndex** hierarchy navigation, and a rule-derived
**provision graph** — and measures each one's contribution against the same evaluation set.

> This repository reproduces the results in the paper. For the exact paper state, check out the
> release tag (see *Releases*). Development continues on other branches.

## Arms (paper Table I ↔ code id)

The design is **additive**: structure is added on top of hybrid, never replacing it. The full
system **PIKG-RAG = HYBRID ∪ PageIndex → reserved-slot KG → rerank**.

| Paper name | code id | description |
|------------|---------|-------------|
| CLOSED | `C0` | closed-book (no retrieval) — floor |
| BM25 | `R0` | lexical only |
| DENSE | `R1` | dense (BGE-M3) only |
| HYBRID | `A1` | dense+sparse fusion → rerank — **baseline** |
| PI | `A3` | PageIndex tree navigation only |
| HYBRID+PI | `A5` | hybrid ∪ PageIndex (additive) |
| HYBRID+KG | `A2` | hybrid + one-hop provision-graph expansion |
| PI+KG | `A4` | PageIndex + graph expansion |
| **PIKG-RAG** | `A4B` | **hybrid ∪ PageIndex + reserved-slot KG** (full system) |
| ORACLE | `C1` | gold sections in context — ceiling |

Older `B0–B3` CLI aliases still resolve (see `retrieval/arms.py:OLD_ALIASES`) but the paper and
Table I use the names above. PageIndex is an **additive signal**, not a replacement base; the
study is an additive ladder, not a 2×2, and significance uses paired bootstrap + McNemar/Holm +
a bootstrap difference-in-differences (there is **no GLMM** in the paper).

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
sections** rendered with a slash id, e.g. `269/1`, `277/2` (= มาตรา ๒๗๗ ทวิ). See the PyThaiNLP
source for the corpus license before redistributing `data/raw/`.

## Knowledge graph

`CITES` edges are text-derived. `APPLIES_TO` edges (Book-1 general provision → Book-2 offence)
are **rule-materialized and currently unverified** — all 13,296 carry `verified=false`. The code
exposes `--verified-only` / `verified_applies_to_only=True`, but with no lawyer pass done yet it
selects an empty set; a legal-expert review of these edges is noted as a limitation / future work
in the paper. Do **not** read the flag as "edges have been checked."

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
python -m retrieval.run_eval    --arms all --eval eval/eval_set.generated.jsonl
python -m generation.run_generation --arms all --eval eval/eval_set.generated.jsonl
python -m generation.run_judge      --arms all --eval eval/eval_set.generated.jsonl

# 5. statistics: paired bootstrap CIs + McNemar/Holm + difference-in-differences
python -m analysis.run_analysis --metric correct --arms all
```

## Layout

```
ingest/      CSV → sections.jsonl + penal_tree.json
kg/          CITES + APPLIES_TO graph construction
retrieval/   arms (A1–A4B, R0/R1, C0/C1), BM25, PageIndex, KG expansion, metrics
generation/  generator + LLM-judge harness (correctness, grounding, refusal)
analysis/    bootstrap, McNemar/Holm, difference-in-differences, plots
eval/        schema + validator + generator for the gold eval set
data/         corpus, tree, graph edges
runs/         result snapshots (per-arm rankings, answers, verdicts)
```
