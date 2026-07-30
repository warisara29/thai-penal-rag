"""Pre-registered contrast family (design §7) — fixed before looking at results,
so Holm-Bonferroni controls FWER over exactly this set.

test: 'mcnemar' for binary retrieval outcomes (hit/coverage),
      'permutation' for rate/binary generation outcomes (correct, claim_grounding).
"""

from __future__ import annotations

# retrieval contrasts (binary): default outcome hit@k / coverage@k
RETRIEVAL_CONTRASTS = [
    {"label": "A3-A1 (RQ1: PageIndex vs hybrid)", "rq": "RQ1", "a": "A3", "b": "A1", "test": "mcnemar"},
    {"label": "A1-R0 (dense/hybrid lift over lexical)", "rq": "RQ1", "a": "A1", "b": "R0", "test": "mcnemar"},
    {"label": "A1-R1 (reranker contribution)", "rq": "RQ1", "a": "A1", "b": "R1", "test": "mcnemar"},
    {"label": "A2-A1 (RQ2: KG on hybrid)", "rq": "RQ2", "a": "A2", "b": "A1", "test": "mcnemar"},
    {"label": "A4-A3 (RQ2: KG on PageIndex)", "rq": "RQ2", "a": "A4", "b": "A3", "test": "mcnemar"},
]

# generation contrasts (binary correctness): each retrieval arm vs closed-book
GENERATION_CONTRASTS = [
    {"label": "A1-C0 (RQ5: retrieval vs parametric)", "rq": "RQ5", "a": "A1", "b": "C0", "test": "permutation"},
    {"label": "A2-C0 (RQ5)", "rq": "RQ5", "a": "A2", "b": "C0", "test": "permutation"},
    {"label": "A3-C0 (RQ5)", "rq": "RQ5", "a": "A3", "b": "C0", "test": "permutation"},
    {"label": "A4-C0 (RQ5)", "rq": "RQ5", "a": "A4", "b": "C0", "test": "permutation"},
    {"label": "C1-C0 (generation ceiling vs floor)", "rq": "RQ5", "a": "C1", "b": "C0", "test": "permutation"},
]

# the 2x2 interaction (RQ2): does KG help MORE on PageIndex than on hybrid?
INTERACTION = {"label": "(A4-A3)-(A2-A1) KG×base interaction", "rq": "RQ2",
               "arms": ["A4", "A3", "A2", "A1"]}


def family_for(metric: str) -> list[dict]:
    if metric in ("hit", "coverage"):
        return RETRIEVAL_CONTRASTS
    return GENERATION_CONTRASTS
