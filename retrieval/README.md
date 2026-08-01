# retrieval/ — experimental arms (design §4)

Same questions, every arm, chunk = 1 มาตรา, k=5, temperature 0.

Arm ids are stable dict/file keys; **B-labels** match the thesis figure (comparison
ladder + 2×2). `--arms` accepts either (`--arms B3+PI` == `--arms A3`).

| id | B-label | What | Status |
|----|---------|------|--------|
| **C0** | `B0` | LLM only (no context) | ⚙️ needs `Generator` (generation-only) |
| **R0** | `B1` | LLM + Keyword Search (BM25) | ✅ runs now (pure Python) |
| **R1** | `B2` | LLM + RAG (BGE-M3 dense) | ⚙️ needs `Embedder` |
| **A1** | `B3` | LLM + Hybrid RAG (dense+sparse→rerank) | ⚙️ needs `Embedder` + `Reranker` |
| **A2** | `B3+KG` | B3 → KG one-hop → rerank | ⚙️ KG logic ✅, needs A1 backends |
| **A3** | `B3+PI` | PageIndex tree navigation (replaces hybrid base) | ✅ heuristic baseline; ⚙️ plug `LLMNodeSelector` |
| **A4** | `PIKG-RAG` | PageIndex → KG one-hop → rerank | ⚙️ needs `Reranker` |
| **C1** | `ORACLE` | generator, gold∪supporting (ceiling) | ⚙️ needs `Generator` (generation-only) |

Ladder: **B0 → B1 → B2 → B3 → PIKG-RAG**. 2×2: base{hybrid, PageIndex} × KG{off, on},
off/off = B3. "B3+PI" uses PageIndex *instead of* hybrid (replacement, not add-on).

## Run
```bash
python -m retrieval.metrics                              # self-test
python -m retrieval.run_eval --arms R0 --eval eval/eval_set.seed.jsonl
python -m retrieval.run_eval --arms all --eval eval/eval_set.generated.jsonl
```
Writes `retrieval/results/<arm>.jsonl` and prints recall@5 / hit@5 / mrr@10.
Model arms skip with an actionable message until their backend is set.

## Plug in the 4 backends (`retrieval/backends.py`)
Implement each protocol, then point `config.json → backends` at `module:Class`:
- **`Embedder`** — BGE-M3 (`sentence-transformers`, `BAAI/bge-m3`) → R1, A1 dense leg
- **`Reranker`** — one shared cross-encoder for A1–A4 (fair-comparison rule §4c)
- **`TreeNavigator`** — A3/A4. The descent algorithm is built (`pageindex.py`); it runs
  today with `HeuristicNodeSelector` (no-LLM, structure-only). For the real arm, fill
  `LLMNodeSelector._choose` with the **same Thai open model** (temp 0) — it picks branches
  at each tree level; calls are counted for RQ4 cost.
- **`Generator`** — the same Thai open model, temp 0, for all generation arms

The KG factor (`kg_expand.py`) already loads CITES (719) + APPLIES_TO (13,296);
set `verified_applies_to_only: true` in config to use only lawyer-approved
APPLIES_TO edges once curation is done.

## Not yet here
Generation + LLM-judge scoring (answer-quality axis) and the cost axis (RQ4)
attach after a `Generator` backend exists — the harness already carries `meta`
for per-arm LLM call/token accounting.
