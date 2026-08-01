"""Pre-registered contrast family (design §7 + the thesis comparison figure) —
fixed before results so Holm-Bonferroni controls FWER over exactly this set.

Nomenclature (internal id / B-label):
  C0/B0  R0/B1  R1/B2  A1/B3  A2/B3+KG  A3/B3+PI  A4/PIKG-RAG  C1/ORACLE

Two stories:
  Ladder     B0 -> B1 -> B2 -> B3 -> PIKG-RAG  (each rung adds a component)
  2x2 factorial   base{hybrid,PageIndex} x KG{off,on}, off/off = B3

test: 'mcnemar' for binary retrieval outcomes (hit/coverage),
      'permutation' for generation outcomes (correct, claim_grounding).
"""

from __future__ import annotations

# --- retrieval contrasts (binary: hit@k / coverage@k) ----------------------
RETRIEVAL_CONTRASTS = [
    # 2x2 core
    {"label": "B3+PI − B3 (RQ1: PageIndex vs hybrid base)", "rq": "RQ1", "a": "A3", "b": "A1", "test": "mcnemar"},
    {"label": "B3+KG − B3 (RQ2: KG on hybrid)", "rq": "RQ2", "a": "A2", "b": "A1", "test": "mcnemar"},
    {"label": "PIKG-RAG − B3+PI (RQ2: KG on PageIndex)", "rq": "RQ2", "a": "A4", "b": "A3", "test": "mcnemar"},
    # ladder rungs (retrieval; B0 has no retrieval so ladder starts at B1)
    {"label": "B2 − B1 (ladder: dense RAG vs keyword)", "rq": "ladder", "a": "R1", "b": "R0", "test": "mcnemar"},
    {"label": "B3 − B2 (ladder: hybrid vs dense)", "rq": "ladder", "a": "A1", "b": "R1", "test": "mcnemar"},
    {"label": "PIKG-RAG − B3 (ladder: full system vs hybrid)", "rq": "ladder", "a": "A4", "b": "A1", "test": "mcnemar"},
]

# --- generation contrasts (binary correctness) -----------------------------
GENERATION_CONTRASTS = [
    # ladder (answer quality) — the headline progression
    {"label": "B1 − B0 (ladder: keyword vs LLM-only)", "rq": "ladder", "a": "R0", "b": "C0", "test": "permutation"},
    {"label": "B2 − B1 (ladder: RAG vs keyword)", "rq": "ladder", "a": "R1", "b": "R0", "test": "permutation"},
    {"label": "B3 − B2 (ladder: hybrid vs dense)", "rq": "ladder", "a": "A1", "b": "R1", "test": "permutation"},
    {"label": "PIKG-RAG − B3 (ladder: full system vs hybrid)", "rq": "ladder", "a": "A4", "b": "A1", "test": "permutation"},
    # RQ5: retrieval arms vs closed-book (B0), plus the generation ceiling
    {"label": "B3 − B0 (RQ5: hybrid RAG vs parametric)", "rq": "RQ5", "a": "A1", "b": "C0", "test": "permutation"},
    {"label": "PIKG-RAG − B0 (RQ5: full system vs parametric)", "rq": "RQ5", "a": "A4", "b": "C0", "test": "permutation"},
    {"label": "ORACLE − B0 (generation ceiling vs floor)", "rq": "RQ5", "a": "C1", "b": "C0", "test": "permutation"},
]

# the 2x2 interaction (RQ2): does KG help MORE on PageIndex than on hybrid?
INTERACTION = {"label": "(PIKG-RAG − B3+PI) − (B3+KG − B3): KG×base interaction",
               "rq": "RQ2", "arms": ["A4", "A3", "A2", "A1"]}


def family_for(metric: str) -> list[dict]:
    if metric in ("hit", "coverage"):
        return RETRIEVAL_CONTRASTS
    return GENERATION_CONTRASTS
