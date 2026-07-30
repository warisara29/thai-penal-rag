"""Pluggable model backends. These are the pieces that need real models; the
scaffold defines the interfaces and ships 'not configured' stand-ins so the
package imports and the non-model arms (R0, C1, KG expansion) run today.

Plug in:
  Embedder      -> BGE-M3 (sentence-transformers) for R1/A1 dense leg
  Reranker      -> a cross-encoder, shared by A1-A4 (fair-comparison rule §4c)
  TreeNavigator -> the Thai open model doing PageIndex tree descent (A3/A4)
  Generator     -> the SAME Thai open model producing answers (all gen arms)

Each raises NotConfigured until wired, with a clear message.
"""

from __future__ import annotations

from typing import Protocol, Sequence

from .base import Corpus


class NotConfigured(RuntimeError):
    pass


class Embedder(Protocol):
    def encode(self, texts: Sequence[str]) -> list[list[float]]: ...


class Reranker(Protocol):
    def rerank(self, query: str, candidate_ids: list[str], corpus: Corpus) -> list[str]:
        """Return candidate_ids reordered best-first."""


class TreeNavigator(Protocol):
    def navigate(self, query: str, tree_path: str, k: int) -> tuple[list[str], dict]:
        """Return (selected section ids, cost meta e.g. {'llm_calls','tokens'})."""


class Generator(Protocol):
    def answer(self, question: str, context_sections: list[str], corpus: Corpus) -> tuple[str, dict]:
        """Return (answer_text, cost meta). Empty context => closed-book (C0)."""


# --- default 'not configured' stand-ins ------------------------------------
def _missing(what: str, how: str):
    class _Stub:
        def __getattr__(self, _):
            raise NotConfigured(f"{what} is not configured. {how}")
    return _Stub()


def default_embedder():
    return _missing("Embedder (BGE-M3)",
                    "Implement Embedder.encode with sentence-transformers 'BAAI/bge-m3' "
                    "and set backends.embedder in config.json.")


def default_reranker():
    return _missing("Reranker (cross-encoder)",
                    "Implement Reranker.rerank with one shared cross-encoder for A1-A4.")


def default_navigator():
    return _missing("TreeNavigator (PageIndex LLM)",
                    "Implement TreeNavigator.navigate over data/penal_tree.json using the "
                    "SAME Thai open model as the generator (fair-comparison rule §4c).")


def default_generator():
    return _missing("Generator (Thai open model)",
                    "Implement Generator.answer with the pinned Thai open model, temperature 0.")
