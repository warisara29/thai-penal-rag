"""BM25 lexical retriever — arm R0 (retrieval floor). Pure Python, no deps.

Also reused as the sparse leg of the A1 hybrid fusion.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict

from .base import Corpus, RetrievalResult
from .tokenize import get_tokenizer


class BM25:
    name = "R0"

    def __init__(self, corpus: Corpus, tokenizer: str = "char_ngram",
                 k1: float = 1.5, b: float = 0.75, name: str = "R0"):
        self.name = name
        self.corpus = corpus
        self.tok = get_tokenizer(tokenizer)
        self.k1, self.b = k1, b
        self.ids = [s.section_id for s in corpus.sections]
        docs = [self.tok(s.content) for s in corpus.sections]
        self.doclen = [len(d) for d in docs]
        self.avgdl = (sum(self.doclen) / len(docs)) if docs else 0.0
        self.tf: list[Counter] = [Counter(d) for d in docs]
        df: Counter = Counter()
        for d in self.tf:
            df.update(d.keys())
        N = len(docs)
        self.idf = {t: math.log(1 + (N - n + 0.5) / (n + 0.5)) for t, n in df.items()}
        self.postings: dict[str, list[int]] = defaultdict(list)
        for i, d in enumerate(self.tf):
            for t in d:
                self.postings[t].append(i)

    def scores(self, query: str) -> dict[str, float]:
        q = self.tok(query)
        out: dict[int, float] = defaultdict(float)
        for t in set(q):
            idf = self.idf.get(t)
            if idf is None:
                continue
            for i in self.postings[t]:
                f = self.tf[i][t]
                denom = f + self.k1 * (1 - self.b + self.b * self.doclen[i] / (self.avgdl or 1))
                out[i] += idf * f * (self.k1 + 1) / denom
        return {self.ids[i]: s for i, s in out.items()}

    def retrieve(self, query: str, query_id: str, k: int) -> RetrievalResult:
        sc = self.scores(query)
        ranked = sorted(sc, key=sc.get, reverse=True)[:k]
        return RetrievalResult(query_id, ranked, {r: sc[r] for r in ranked})
