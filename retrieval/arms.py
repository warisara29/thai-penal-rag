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
ALL_ARMS = FACTORIAL + REFERENCE

# Thesis-figure nomenclature (comparison ladder + 2x2). Internal ids stay stable
# as dict/file keys; B-labels are the canonical display + CLI alias.
ARM_LABELS = {"C0": "B0", "R0": "B1", "R1": "B2", "A1": "B3",
              "A2": "B3+KG", "A3": "B3+PI", "A4": "PIKG-RAG", "C1": "ORACLE"}
LABEL_TO_ARM = {v: k for k, v in ARM_LABELS.items()}
LADDER = ["C0", "R0", "R1", "A1", "A4"]  # B0 -> B1 -> B2 -> B3 -> PIKG-RAG


def label(arm: str) -> str:
    return ARM_LABELS.get(arm, arm)


def resolve(name: str) -> str:
    """Accept either an internal id (A1) or a B-label (B3, PIKG-RAG)."""
    return LABEL_TO_ARM.get(name, name)

POOL = 50  # candidate pool a base retriever surfaces before rerank/expansion


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


def build_arm(name: str, ctx: Context):
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
