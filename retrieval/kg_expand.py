"""One-hop KG expansion — the `+KG` factor in A2 / A4.

Loads CITES edges (kg_edges.jsonl from build_kg.py) and APPLIES_TO edges
(applies_to_edges.jsonl from expand_applies_to.py). Given a set of seed section
ids from a base retriever, expands one hop along those relations to grow the
candidate pool; the shared reranker then re-ranks seeds+expansion.

verified_applies_to_only=True uses only lawyer-approved APPLIES_TO edges.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path


class KGExpander:
    def __init__(self, cites_path: str | Path | None, applies_path: str | Path | None,
                 verified_applies_to_only: bool = False):
        self.adj: dict[str, set[str]] = defaultdict(set)
        self.n_cites = self.n_applies = 0
        if cites_path and Path(cites_path).exists():
            self.n_cites = self._load(cites_path, "CITES")
        if applies_path and Path(applies_path).exists():
            self.n_applies = self._load(applies_path, "APPLIES_TO",
                                        verified_only=verified_applies_to_only)

    def _sid(self, node: str) -> str:
        return node.split(":", 1)[1] if node.startswith("Section:") else node

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
            src, dst = self._sid(e["src"]), self._sid(e["dst"])
            self.adj[src].add(dst)
            self.adj[dst].add(src)  # expansion is symmetric for candidate growth
            n += 1
        return n

    def expand(self, seed_ids: list[str], max_neighbors_per_seed: int = 20) -> list[str]:
        """Return seeds + their one-hop neighbours, seeds first, de-duplicated."""
        out, seen = list(seed_ids), set(seed_ids)
        for s in seed_ids:
            for nb in list(self.adj.get(s, ()))[:max_neighbors_per_seed]:
                if nb not in seen:
                    seen.add(nb)
                    out.append(nb)
        return out
