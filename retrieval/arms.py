"""Assemble the experimental arms (design §4) from shared components.

Factorial:  A1 HYB | A2 HYB+KG | A3 PI | A4 PI+KG
References:  R0 BM25 | R1 DENSE | C0 CLOSED | C1 ORACLE   (C0/C1 are generation-only)

Runnable now with no models: R0, C1, and the KG-expansion logic in A2/A4.
A1/A3/A4/R1 need their backends (embedder/reranker/navigator) plugged in.
"""

from __future__ import annotations

from dataclasses import dataclass

from .base import Corpus, RetrievalResult
from .bm25 import BM25
from .kg_expand import KGExpander

GEN_ONLY = {"C0", "C1"}          # scored on generation only, not retrieval
FACTORIAL = ["A1", "A2", "A3", "A4"]
REFERENCE = ["R0", "R1", "C0", "C1"]
ALL_ARMS = ["A1", "A2", "A3", "A4", "A4B", "A5", "R0", "R1", "C0", "C1"]

# Additive-component design: three ingredients (hybrid, PageIndex, KG) combined
# additively — descriptive labels are canonical and read literally (no misleading "+").
# Internal ids stay stable as dict/file keys.
ARM_LABELS = {"C0": "closed", "R0": "keyword", "R1": "dense",
              "A1": "hybrid", "A3": "pi", "A5": "hybrid+pi",
              "A2": "hybrid+kg", "A4": "pi+kg", "A4B": "hybrid+pi+kg",
              "C1": "ORACLE"}
# The clean 5-arm additive comparison set the thesis tells its story with:
ADDITIVE = ["A1", "A3", "A5", "A2", "A4B"]  # hybrid · pi · hybrid+pi · hybrid+kg · hybrid+pi+kg
LADDER = ADDITIVE
LABEL_TO_ARM = {v: k for k, v in ARM_LABELS.items()}
# Back-compat: still accept the retired B0-B3 / PIKG-RAG / PIKG+ labels on the CLI.
OLD_ALIASES = {"B0": "C0", "B1": "R0", "B2": "R1", "B3": "A1", "B3+KG": "A2",
               "B3+PI": "A3", "PIKG-RAG": "A4", "PIKG+": "A4B", "ORACLE": "C1"}


def label(arm: str) -> str:
    return ARM_LABELS.get(arm, arm)


def resolve(name: str) -> str:
    """Accept an internal id (A5), a descriptive label (hybrid+pi), or an old B-label (B3)."""
    return LABEL_TO_ARM.get(name, OLD_ALIASES.get(name, name))

import os
POOL = int(os.environ.get("RAG_POOL", "50"))  # candidate pool before rerank/expansion (env-tunable for ablation)


@dataclass
class Context:
    corpus: Corpus
    bm25: BM25
    expander: KGExpander
    embedder: object
    reranker: object
    navigator: object
    config: dict


class DenseRetriever:
    name = "R1"

    def __init__(self, corpus: Corpus, embedder, name: str = "R1"):
        self.name, self.corpus, self.embedder = name, corpus, embedder
        self._mat = None

    def _ensure(self):
        if self._mat is None:  # lazy so an unconfigured embedder only fails on use
            self._mat = self.embedder.encode([s.content for s in self.corpus.sections])

    def retrieve(self, query: str, query_id: str, k: int) -> RetrievalResult:
        self._ensure()
        qv = self.embedder.encode([query])[0]
        def dot(a, b): return sum(x * y for x, y in zip(a, b))
        sc = {s.section_id: dot(qv, self._mat[i]) for i, s in enumerate(self.corpus.sections)}
        ranked = sorted(sc, key=sc.get, reverse=True)[:k]
        return RetrievalResult(query_id, ranked, {r: sc[r] for r in ranked})


class HybridRetriever:
    """A1: dense+sparse fusion -> cross-encoder rerank over มาตรา chunks."""
    name = "A1"

    def __init__(self, corpus, bm25, embedder, reranker, alpha=0.5, name="A1"):
        self.name, self.corpus, self.bm25 = name, corpus, bm25
        self.dense = DenseRetriever(corpus, embedder, name + ".dense")
        self.reranker, self.alpha = reranker, alpha

    @staticmethod
    def _norm(d: dict) -> dict:
        if not d:
            return {}
        lo, hi = min(d.values()), max(d.values())
        rng = (hi - lo) or 1.0
        return {k: (v - lo) / rng for k, v in d.items()}

    def _fused_pool(self, query: str) -> list[str]:
        bm = self._norm(self.bm25.scores(query))
        self.dense._ensure()
        qv = self.dense.embedder.encode([query])[0]
        def dot(a, b): return sum(x * y for x, y in zip(a, b))
        dn = self._norm({s.section_id: dot(qv, self.dense._mat[i])
                         for i, s in enumerate(self.corpus.sections)})
        fused = {sid: self.alpha * dn.get(sid, 0) + (1 - self.alpha) * bm.get(sid, 0)
                 for sid in set(bm) | set(dn)}
        return sorted(fused, key=fused.get, reverse=True)[:POOL]

    def retrieve(self, query: str, query_id: str, k: int) -> RetrievalResult:
        pool = self._fused_pool(query)
        ranked = self.reranker.rerank(query, pool, self.corpus)[:k]
        return RetrievalResult(query_id, ranked)


class PageIndexRetriever:
    """A3: LLM descends penal_tree.json ภาค→…→มาตรา."""
    name = "A3"

    def __init__(self, corpus, navigator, tree_path: str, name="A3"):
        self.name, self.corpus, self.navigator, self.tree_path = name, corpus, navigator, tree_path

    def retrieve(self, query: str, query_id: str, k: int) -> RetrievalResult:
        ids, meta = self.navigator.navigate(query, self.tree_path, k)
        return RetrievalResult(query_id, ids[:k], meta=meta)


class KGArm:
    """A2/A4: base retriever seeds -> one-hop KG expansion -> shared rerank."""

    def __init__(self, base, expander, reranker, corpus, name: str):
        self.base, self.expander, self.reranker, self.corpus, self.name = \
            base, expander, reranker, corpus, name

    def retrieve(self, query: str, query_id: str, k: int) -> RetrievalResult:
        seeds = self.base.retrieve(query, query_id, POOL)
        expanded = self.expander.expand(seeds.ranked_ids)
        ranked = self.reranker.rerank(query, expanded, self.corpus)[:k]
        return RetrievalResult(query_id, ranked, meta=seeds.meta)


class FullSystem:
    """Additive full system: keep the strong hybrid pool AND add PageIndex-selected
    sections, optionally expand along the KG, then shared rerank. Unlike the
    replacement arms (A3/A4), this never throws hybrid away — structure augments it.
    expander=None → hybrid+pi (no KG); an expander → hybrid+pi+kg."""

    def __init__(self, hybrid, pi, expander, reranker, corpus, name="A4B", kg_reserve=1):
        self.hybrid, self.pi = hybrid, pi
        self.expander, self.reranker, self.corpus, self.name = expander, reranker, corpus, name
        self.kg_reserve = kg_reserve   # slots at the tail reserved for KG supporting provisions

    def retrieve(self, query: str, query_id: str, k: int) -> RetrievalResult:
        hpool = self.hybrid._fused_pool(query)                      # strong hybrid candidates
        pi = self.pi.retrieve(query, query_id, POOL)                # PageIndex-selected
        seed = list(dict.fromkeys(hpool + pi.ranked_ids))          # union (hybrid first)
        # Rank the ANSWER candidates cleanly — KG never dilutes the primary ordering.
        primary = self.reranker.rerank(query, seed, self.corpus)
        if not self.expander:
            return RetrievalResult(query_id, primary[:k], meta=pi.meta)
        # KG APPENDS the best supporting provision(s) into reserved tail slots, so it
        # can add multi-hop context without ever displacing a primary gold section.
        expanded = self.expander.expand(seed)
        head = primary[: k - self.kg_reserve]
        kg_new = [s for s in expanded if s not in set(seed)]        # provisions only KG surfaced
        tail = [s for s in self.reranker.rerank(query, kg_new, self.corpus)
                if s not in set(head)][: self.kg_reserve] if kg_new else []
        ranked = (head + tail + primary)                           # backfill if KG found nothing
        ranked = list(dict.fromkeys(ranked))[:k]
        return RetrievalResult(query_id, ranked, meta=pi.meta)


def build_arm(name: str, ctx: Context):
    if name == "A4B":  # additive full system: hybrid ∪ PageIndex → KG → rerank
        hyb = HybridRetriever(ctx.corpus, ctx.bm25, ctx.embedder, ctx.reranker, name="A4B.hyb")
        pi = PageIndexRetriever(ctx.corpus, ctx.navigator, ctx.config["tree_path"], name="A4B.pi")
        return FullSystem(hyb, pi, ctx.expander, ctx.reranker, ctx.corpus, "A4B",
                          kg_reserve=ctx.config.get("kg_reserve", 1))
    if name == "A5":  # additive hybrid+pi, NO KG: hybrid ∪ PageIndex → rerank
        hyb = HybridRetriever(ctx.corpus, ctx.bm25, ctx.embedder, ctx.reranker, name="A5.hyb")
        pi = PageIndexRetriever(ctx.corpus, ctx.navigator, ctx.config["tree_path"], name="A5.pi")
        return FullSystem(hyb, pi, None, ctx.reranker, ctx.corpus, "A5")
    if name == "R0":
        return ctx.bm25
    if name == "R1":
        return DenseRetriever(ctx.corpus, ctx.embedder)
    if name == "A1":
        return HybridRetriever(ctx.corpus, ctx.bm25, ctx.embedder, ctx.reranker)
    if name == "A2":
        base = HybridRetriever(ctx.corpus, ctx.bm25, ctx.embedder, ctx.reranker, name="A2.base")
        return KGArm(base, ctx.expander, ctx.reranker, ctx.corpus, "A2")
    if name == "A3":
        return PageIndexRetriever(ctx.corpus, ctx.navigator, ctx.config["tree_path"])
    if name == "A4":
        base = PageIndexRetriever(ctx.corpus, ctx.navigator, ctx.config["tree_path"], name="A4.base")
        return KGArm(base, ctx.expander, ctx.reranker, ctx.corpus, "A4")
    raise ValueError(f"{name} is generation-only or unknown (see GEN_ONLY / ALL_ARMS)")
