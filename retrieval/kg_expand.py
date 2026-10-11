"""One-hop KG expansion — the `+KG` factor in A2 / A4.

Loads CITES edges (kg_edges.jsonl from build_kg.py) and APPLIES_TO edges
(applies_to_edges.jsonl from expand_applies_to.py). Given a set of seed section
ids from a base retriever, expands one hop along those relations to grow the
candidate pool; the shared reranker then re-ranks seeds+expansion.

verified_applies_to_only=True uses only lawyer-approved APPLIES_TO edges.

Court-weighted mode (thesis extension, task B): pass ``weighted_applies_path`` (the output of
caselaw.weight_graph, with a per-edge ``w``) to keep only APPLIES_TO edges with ``w >= tau`` and
retain their weights. CITES edges are always loaded as-is (weight 1.0). ``rule_based=True`` is the
ablation revert — ignore weights and use the raw rule-derived APPLIES_TO graph. Weighted neighbours
(sorted by w) drive the fact-guided traversal's reserved slot in task C.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path


class KGExpander:
    def __init__(self, cites_path: str | Path | None, applies_path: str | Path | None,
                 verified_applies_to_only: bool = False,
                 weighted_applies_path: str | Path | None = None,
                 tau: float = 0.0, rule_based: bool = False):
        self.adj: dict[str, set[str]] = defaultdict(set)
        self.w: dict[tuple[str, str], float] = {}   # (a,b) -> weight, stored both directions
        self.tau = tau
        self.n_cites = self.n_applies = 0
        if cites_path and Path(cites_path).exists():
            self.n_cites = self._load(cites_path, "CITES")
        # APPLIES_TO: court-weighted file (unless reverted to rule-based for ablation)
        if weighted_applies_path and not rule_based and Path(weighted_applies_path).exists():
            self.n_applies = self._load_weighted(weighted_applies_path, tau)
        elif applies_path and Path(applies_path).exists():
            self.n_applies = self._load(applies_path, "APPLIES_TO",
                                        verified_only=verified_applies_to_only)

    def _sid(self, node: str) -> str:
        return node.split(":", 1)[1] if node.startswith("Section:") else node

    def _add(self, a: str, b: str, w: float = 1.0) -> None:
        self.adj[a].add(b)
        self.adj[b].add(a)  # expansion is symmetric for candidate growth
        self.w[(a, b)] = self.w[(b, a)] = w

    def _load(self, path, etype: str, verified_only: bool = False) -> int:
        n = 0
        for line in Path(path).read_text("utf-8").splitlines():
            if not line.strip():
                continue
            e = json.loads(line)
            if e.get("type") != etype:
                continue
            if verified_only and not e.get("verified", False):
                continue
            self._add(self._sid(e["src"]), self._sid(e["dst"]), 1.0)
            n += 1
        return n

    def _load_weighted(self, path, tau: float) -> int:
        """Load APPLIES_TO edges carrying a co-citation weight; keep w >= tau and w > 0."""
        n = 0
        for line in Path(path).read_text("utf-8").splitlines():
            if not line.strip():
                continue
            e = json.loads(line)
            if e.get("type") != "APPLIES_TO":
                continue
            w = float(e.get("w", 0.0))
            if w <= 0 or w < tau:
                continue
            self._add(self._sid(e["src"]), self._sid(e["dst"]), w)
            n += 1
        return n

    def weight(self, a: str, b: str) -> float:
        return self.w.get((a, b), 0.0)

    def weighted_neighbors(self, sid: str) -> list[tuple[str, float]]:
        """One-hop neighbours of `sid` as (neighbour, weight), heaviest first (task C)."""
        return sorted(((nb, self.w.get((sid, nb), 0.0)) for nb in self.adj.get(sid, ())),
                      key=lambda t: t[1], reverse=True)

    def expand(self, seed_ids: list[str], max_neighbors_per_seed: int = 20) -> list[str]:
        """Return seeds + their one-hop neighbours, seeds first, de-duplicated.
        Neighbours are taken heaviest-weight first so a per-seed cap keeps the strongest links."""
        out, seen = list(seed_ids), set(seed_ids)
        for s in seed_ids:
            nbrs = [nb for nb, _ in self.weighted_neighbors(s)] if self.w else list(self.adj.get(s, ()))
            for nb in nbrs[:max_neighbors_per_seed]:
                if nb not in seen:
                    seen.add(nb)
                    out.append(nb)
        return out
