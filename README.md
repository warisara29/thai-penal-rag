# Thai Penal Code RAG — scaffold

Corpus: **ประมวลกฎหมายอาญา only**. Three ingredients, one shared corpus.

## Arms (thesis figure ↔ code)
Comparison ladder **B0→B1→B2→B3→PIKG-RAG** and the 2×2 (base × KG). Internal ids are
stable keys; B-labels match the figure and work on every `--arms` flag.

| B-label | id | | B-label | id |
|---|---|---|---|---|
| B0 LLM only | C0 | | B3+KG | A2 |
| B1 LLM + Keyword | R0 | | B3+PI | A3 |
| B2 LLM + RAG | R1 | | PIKG-RAG | A4 |
| B3 LLM + Hybrid RAG | A1 | | ORACLE (ceiling) | C1 |

PageIndex (B3+PI) is a **replacement** base retriever, not an add-on — so the 2×2 factor
is base{hybrid, PageIndex} × KG{off, on}, matching the §7 GLMM.

```
ingest/pageindex_parser.py   raw .txt  ->  penal_tree.json + sections.jsonl
kg/build_kg.py               sections.jsonl -> KG (nodes/edges/cypher)
eval/                        schema + guide + validator + seed set
```

## Pipeline
```
penal_code.txt
      │  ingest/pageindex_parser.py
      ▼
penal_tree.json ──► PageIndex-style tree search / LLM navigation   (primary retriever)
sections.jsonl  ──► kg/build_kg.py ──► KG ──► multi-hop expansion   (secondary)
                └─► [baseline] chunk = 1 มาตรา ──► BGE-M3 hybrid + reranker  (control)
```
Evaluate all three retrievers against the **same** `eval/eval_set.jsonl` so every
layer's contribution is a measured number, not a vibe.

## Quick start
```bash
# 0. corpus is pinned to PyThaiNLP thai-law `criminal-csv-v0.1`
#    -> data/raw/criminal-datasets.csv  (มาตรา 1-398, columns article,text,notes)

# 1. ingest the CSV into structure-aware form (hierarchy from data/hierarchy.json)
python ingest/csv_to_sections.py data/raw/criminal-datasets.csv \
       --hierarchy data/hierarchy.json --out-dir data
#    -> sections.jsonl + penal_tree.json + enabling_act.jsonl + ingest_report.txt
#    (legacy: ingest/pageindex_parser.py parses a raw .txt instead)

# 2. build the knowledge graph
python kg/build_kg.py data/sections.jsonl --out-dir data       # CITES + structure
python kg/expand_applies_to.py data/sections.jsonl \
       --rules kg/applies_to_rules.json --out-dir data
#    -> applies_to_edges.jsonl (13k candidate edges) + applies_to_review.csv
#    lawyer verifies kg/applies_to_rules.json (verified=true) + fills the review CSV,
#    then re-run with --verified-only for the approved subset

# 3. generate the 250+ eval set (LLM-drafted from the corpus)
python eval/generate_eval.py --dry-run            # plan only, no API key
python eval/generate_eval.py --out eval/eval_set.generated.jsonl --workers 4
#    needs `pip install anthropic` + ANTHROPIC_API_KEY (or `ant auth login`)
#    gold_sections are fixed by construction; drafter = Claude Opus 4.8 (!= judge/generator)

# 4. validate the eval set (schema + gold ids exist in corpus)
python eval/validate_eval.py eval/eval_set.generated.jsonl --schema eval/schema.json \
       --sections data/sections.jsonl

# 5. run the eval set through the retrieval arms (design §4: A1-A4, R0/R1/C0/C1)
python -m retrieval.run_eval --arms R0 --eval eval/eval_set.seed.jsonl   # runs today
python -m retrieval.run_eval --arms all --eval eval/eval_set.generated.jsonl
#    R0/C1/KG-expansion run now; R1/A1/A3/A4 need backends (retrieval/backends.py)

# 6. generate answers + judge them (design §6b: quality axis)
python -m generation.run_generation --arms C1,A1 --eval eval/eval_set.generated.jsonl
python -m generation.run_judge      --arms C1,A1 --eval eval/eval_set.generated.jsonl
python -m generation.audit_sample --make          # 20% human-audit worksheet -> κ/ρ
#    hallucination + refusal metrics run now; claim-grounding + correctness need a judge;
#    generation needs the Thai-model Generator backend (generation/config.json)

# 7. analysis (design §7): CIs, significance, Holm-Bonferroni, 2x2 interaction
python -m analysis.run_analysis --metric hit --arms all
python -m analysis.run_analysis --metric correct --arms all --glmm
#    paired bootstrap + McNemar + Holm run now; GLMM needs statsmodels + A1-A4
```

## Notes / next steps
- **HNMFk** is intentionally omitted — the Penal Code's hierarchy is already
  authored, so unsupervised topic discovery adds little. Revisit only if you add
  คำพิพากษาฎีกา (precedents).
- The parser's structure rules live in `MARKERS` / `SECTION_RE`; adapt to your
  source text's exact formatting.
- `APPLIES_TO` (Book-1 general provision → Book-2 offence) can't be auto-derived
  reliably — `build_kg.py` emits a curation stub for a lawyer. This edge is what
  powers multi-hop reasoning, so it's worth the human pass.
- Seed eval items in `eval/eval_set.seed.jsonl` are **examples** — verify every
  section number and penalty against your actual corpus/amendment version before
  trusting them.
