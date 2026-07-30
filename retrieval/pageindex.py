"""PageIndex tree navigation (arms A3/A4). The LLM descends penal_tree.json
ภาค→ลักษณะ→หมวด→มาตรา, choosing which branches to enter and which sections to
return. Design §4c: the navigator LLM MUST be the same Thai open model as the
generator, and its calls count toward the cost metric (RQ4).

Structure here:
  NodeSelector          — picks relevant children at one tree level
  PageIndexNavigator    — the descent algorithm (implements TreeNavigator)
  HeuristicNodeSelector — no-LLM lexical selector: makes A3 runnable today AND
                          serves as the "structure-only navigation" ablation
  LLMNodeSelector       — template for the real arm; fill _choose() with the Thai model
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

from .backends import NotConfigured
from .tokenize import get_tokenizer


class NodeSelector(Protocol):
    def select(self, query: str, options: list[str], max_choose: int) -> tuple[list[int], dict]:
        """Return (chosen option indices, cost meta e.g. {'llm_calls','tokens'})."""


def _subtree_snippet(node: dict, budget: int) -> str:
    """Concatenate descendant section headings+text (bounded) so a container is
    judged by what it CONTAINS, not just its short title."""
    buf: list[str] = []
    total = 0
    stack = list(node.get("nodes", []))
    while stack and total < budget:
        c = stack.pop(0)
        if c.get("node_type") == "section":
            piece = f"{c.get('heading', '')} {c.get('text', '')[:80]}"
            buf.append(piece); total += len(piece)
        else:
            stack[:0] = c.get("nodes", [])
    return " ".join(buf)[:budget]


def _display(node: dict, budget: int = 400) -> str:
    h = node.get("heading", "")
    if node.get("node_type") == "section":
        return f"{h} {node.get('text', '')[:120]}"
    return f"{h} :: {_subtree_snippet(node, budget)}"


class PageIndexNavigator:
    """Beam descent over the section tree. Satisfies the TreeNavigator protocol."""

    def __init__(self, selector: NodeSelector, beam_width: int = 3):
        self.selector = selector
        self.beam_width = beam_width
        self._tree_cache: dict[str, dict] = {}

    def _tree(self, path: str) -> dict:
        if path not in self._tree_cache:
            self._tree_cache[path] = json.loads(Path(path).read_text("utf-8"))
        return self._tree_cache[path]

    def navigate(self, query: str, tree_path: str, k: int) -> tuple[list[str], dict]:
        beam = [self._tree(tree_path)]
        candidates: list[dict] = []       # section nodes reached in chosen branches
        seen_nodes: set[str] = set()
        calls = tokens = 0

        # Phase 1: descend, pruning container branches by relevance; collect the
        # sections in the branches we keep (depth-independent — Book 3's shallow
        # sections and Book 2's deep sections compete on equal footing later).
        while beam:
            next_beam = []
            for node in beam:
                for c in node.get("nodes", []):
                    if c.get("node_type") == "section":
                        candidates.append(c)
                conts = [c for c in node.get("nodes", []) if c.get("node_type") != "section"]
                if conts:
                    idxs, meta = self.selector.select(query, [_display(c) for c in conts],
                                                      self.beam_width)
                    calls += meta.get("llm_calls", 0); tokens += meta.get("tokens", 0)
                    for i in idxs:
                        nid = conts[i].get("node_id", id(conts[i]))
                        if nid not in seen_nodes:
                            seen_nodes.add(nid); next_beam.append(conts[i])
            beam = next_beam

        # Phase 2: final top-k selection among the collected candidate sections.
        if not candidates:
            return [], {"llm_calls": calls, "tokens": tokens, "navigator": type(self.selector).__name__}
        idxs, meta = self.selector.select(query, [_display(c) for c in candidates], k)
        calls += meta.get("llm_calls", 0); tokens += meta.get("tokens", 0)
        selected, seen = [], set()
        for i in idxs:
            sid = candidates[i].get("section_id")
            if sid and sid not in seen:
                seen.add(sid); selected.append(sid)
        return selected[:k], {"llm_calls": calls, "tokens": tokens,
                              "navigator": type(self.selector).__name__}


class HeuristicNodeSelector:
    """No-LLM lexical selector (char-ngram overlap). Runs A3 today and doubles as
    the structure-only-navigation ablation. Not the real A3 — swap in LLMNodeSelector."""

    def __init__(self, tokenizer: str = "char_ngram"):
        self.tok = get_tokenizer(tokenizer)

    def select(self, query: str, options: list[str], max_choose: int):
        q = set(self.tok(query))
        scored = []
        for i, opt in enumerate(options):
            ov = self.tok(opt)
            scored.append((sum(1 for t in ov if t in q), i))
        scored.sort(reverse=True)
        chosen = [i for s, i in scored[:max_choose] if s > 0] or [scored[0][1]] if scored else []
        return chosen, {"llm_calls": 0, "tokens": 0}


class LLMNodeSelector:
    """Real A3/A4 selector: the SAME Thai open model as the generator picks branches.
    Fill `_choose` with a temperature-0 call that returns the chosen indices."""

    def __init__(self, llm=None, model: str | None = None):
        self.llm, self.model = llm, model

    def _choose(self, query: str, options: list[str], max_choose: int) -> tuple[list[int], int]:
        """Return (indices, tokens_used). Prompt template below; wire self.llm."""
        raise NotConfigured(
            "LLMNodeSelector._choose not implemented. Call the pinned Thai open model "
            "(temperature 0) with the numbered options and parse the chosen indices. "
            "Suggested prompt: 'จากรายการหัวข้อกฎหมายด้านล่าง เลือกหมายเลขที่เกี่ยวข้องกับคำถามมากที่สุด "
            "ไม่เกิน {max_choose} หมายเลข ตอบเป็นรายการหมายเลข'.")

    def select(self, query: str, options: list[str], max_choose: int):
        idxs, tokens = self._choose(query, options, max_choose)
        idxs = [i for i in idxs if 0 <= i < len(options)][:max_choose]
        return idxs, {"llm_calls": 1, "tokens": tokens}
