"""Core types for the retrieval arms. Chunk unit = one มาตรา everywhere."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol


@dataclass
class Section:
    section_id: str
    heading: str
    text: str
    node_id: str = ""
    book: str | None = None
    title: str | None = None
    chapter: str | None = None

    @property
    def content(self) -> str:
        return f"{self.heading}\n{self.text}"


@dataclass
class Corpus:
    sections: list[Section]
    by_id: dict[str, Section] = field(default_factory=dict)

    def __post_init__(self):
        if not self.by_id:
            self.by_id = {s.section_id: s for s in self.sections}

    @classmethod
    def load(cls, path: str | Path) -> "Corpus":
        secs = []
        for line in Path(path).read_text("utf-8").splitlines():
            if not line.strip():
                continue
            d = json.loads(line)
            secs.append(Section(
                section_id=d["section_id"], heading=d.get("heading", ""),
                text=d.get("text", ""), node_id=d.get("node_id", ""),
                book=(d.get("book") or {}).get("number"),
                title=(d.get("title") or {}).get("number"),
                chapter=(d.get("chapter") or {}).get("number"),
            ))
        return cls(secs)


@dataclass
class EvalItem:
    id: str
    question: str
    question_type: str
    gold_sections: list[str]
    supporting_sections: list[str] = field(default_factory=list)

    @classmethod
    def load_all(cls, path: str | Path) -> list["EvalItem"]:
        out = []
        for line in Path(path).read_text("utf-8").splitlines():
            if not line.strip():
                continue
            d = json.loads(line)
            out.append(cls(d["id"], d["question"], d["question_type"],
                           d.get("gold_sections", []), d.get("supporting_sections", [])))
        return out


@dataclass
class RetrievalResult:
    """Ranked section ids (best first) plus optional per-id scores."""
    query_id: str
    ranked_ids: list[str]
    scores: dict[str, float] = field(default_factory=dict)
    meta: dict = field(default_factory=dict)  # e.g. llm calls/tokens for cost (RQ4)


class Retriever(Protocol):
    name: str  # arm id, e.g. "A1", "R0"

    def retrieve(self, query: str, query_id: str, k: int) -> RetrievalResult:
        ...
