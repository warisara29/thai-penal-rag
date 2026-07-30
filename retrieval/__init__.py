"""Retrieval arms for the Thai Penal Code RAG benchmark (design §4).

Runnable now (no models): R0 BM25, C1 oracle, KG expansion, metrics.
Plug in backends (retrieval/backends.py) to enable R1/A1/A3/A4 and generation.
"""
from .base import Corpus, EvalItem, RetrievalResult, Section  # noqa: F401
