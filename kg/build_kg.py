"""
Knowledge Graph builder for the Thai Penal Code.

Consumes sections.jsonl (from pageindex_parser.py) and emits a legal KG.

Design note: a legal KG is *mostly pre-defined*, not extracted bottom-up.
We get most edges for free from structure, plus one genuinely useful extracted
edge type (cross-references between มาตรา).

NODE types
  Section   : one มาตรา  (id = section_id, e.g. "288", "335/1", "33 ทวิ")
  Chapter/Title/Book : structural containers
  Penalty   : coarse penalty class attached to a section (death/life/imprison/fine)

EDGE types
  CONTAINS      structural: Book->Title->Chapter->Section  (from the tree)
  CITES         section text references another section ("ตามมาตรา ๓๓๔")
  HAS_PENALTY   section -> penalty class (extracted by keyword)
  APPLIES_TO    (manual/curated) Book-1 general provision -> Book-2 offence
                We can't reliably auto-derive this; we scaffold a stub file
                general_provisions.csv for a lawyer to curate. This is the edge
                that powers multi-hop reasoning, so it's worth the human pass.

Outputs (in --out-dir):
  kg_nodes.jsonl, kg_edges.jsonl   generic graph interchange
  kg.cypher                        ready to \\i into Neo4j
  general_provisions.stub.csv      curation template for APPLIES_TO

Usage:
    python build_kg.py ../data/sections.jsonl --out-dir ../data
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

THAI2ARABIC = {ord(t): str(i) for i, t in enumerate("๐๑๒๓๔๕๖๗๘๙")}


def norm(s: str) -> str:
    return s.translate(THAI2ARABIC)


# match a section reference inside body text:  มาตรา ๓๓๔  /  มาตรา 288 ทวิ  /  ๓๓๕/๑
SECTION_REF = re.compile(
    r"มาตรา\s*([๐-๙\d]+(?:/[๐-๙\d]+)?)\s*(ทวิ|ตรี|จัตวา|เบญจ|ฉ|สัตต|อัฏฐ|นว|ทศ)?"
)
# a range like "มาตรา ๓๓๔ ถึงมาตรา ๓๓๖" or "มาตรา ๓๓๔ ถึง ๓๓๖"
RANGE = re.compile(
    r"มาตรา\s*([๐-๙\d]+)\s*ถึง\s*(?:มาตรา\s*)?([๐-๙\d]+)"
)

# coarse penalty classes, most-severe first
PENALTY_KEYWORDS = [
    ("death", "ประหารชีวิต"),
    ("life", "จำคุกตลอดชีวิต"),
    ("imprisonment", "จำคุก"),
    ("detention", "กักขัง"),
    ("fine", "ปรับ"),
]


# The CSV encodes ทวิ/ตรี sub-articles with slash suffixes (มาตรา ๓๓๕ ทวิ -> "335/2").
# Map the Thai ordinal word to that suffix so extracted citations resolve to real nodes.
ORDINAL_SUFFIX = {
    "ทวิ": 2, "ตรี": 3, "จัตวา": 4, "เบญจ": 5, "ฉ": 6,
    "สัตต": 7, "อัฏฐ": 8, "นว": 9, "ทศ": 10, "เอกาทศ": 11, "ทวาทศ": 12,
}


def ref_id(num: str, ordinal: str | None) -> str:
    base = norm(num)
    if ordinal:
        # already-slashed forms (๓๓๖/๒) keep their number; worded ทวิ/ตรี -> /N
        return f"{base}/{ORDINAL_SUFFIX[ordinal]}" if "/" not in base else base
    return base


def extract_citations(text: str, self_id: str) -> set[str]:
    refs: set[str] = set()
    for m in SECTION_REF.finditer(text):
        refs.add(ref_id(m.group(1), m.group(2)))
    # expand simple numeric ranges
    for m in RANGE.finditer(text):
        lo, hi = int(norm(m.group(1))), int(norm(m.group(2)))
        if 0 < hi - lo < 50:               # guard against garbage ranges
            refs.update(str(x) for x in range(lo, hi + 1))
    refs.discard(self_id)                  # no self-loops
    return refs


def extract_penalties(text: str) -> list[str]:
    return [cls for cls, kw in PENALTY_KEYWORDS if kw in text]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sections", type=Path, help="sections.jsonl")
    ap.add_argument("--out-dir", type=Path, default=Path("../data"))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    sections = [json.loads(l) for l in args.sections.read_text("utf-8").splitlines() if l.strip()]
    known_ids = {s["section_id"] for s in sections}

    nodes: list[dict] = []
    edges: list[dict] = []
    seen_struct: set[str] = set()

    def add_struct(kind: str, info: dict | None, parent_key: str | None, child_key: str):
        if not info:
            return
        key = f"{kind}:{info['number']}"
        if key not in seen_struct:
            seen_struct.add(key)
            nodes.append({"id": key, "label": kind.capitalize(),
                          "number": info["number"], "heading": info["heading"]})
        if parent_key:
            edges.append({"src": parent_key, "dst": key, "type": "CONTAINS"})
        return key

    dangling = 0
    for s in sections:
        sid = s["section_id"]
        nodes.append({
            "id": f"Section:{sid}", "label": "Section", "section_id": sid,
            "heading": s["heading"], "text": s["text"],
            "book": (s.get("book") or {}).get("number"),
            "title": (s.get("title") or {}).get("number"),
            "chapter": (s.get("chapter") or {}).get("number"),
        })
        # structural CONTAINS chain
        bk = add_struct("book", s.get("book"), None, "book")
        tl = add_struct("title", s.get("title"), bk, "title")
        ch = add_struct("chapter", s.get("chapter"), tl or bk, "chapter")
        edges.append({"src": (ch or tl or bk or "root"), "dst": f"Section:{sid}", "type": "CONTAINS"})

        # CITES
        for ref in extract_citations(s["text"], sid):
            edges.append({
                "src": f"Section:{sid}", "dst": f"Section:{ref}",
                "type": "CITES", "resolved": ref in known_ids,
            })
            if ref not in known_ids:
                dangling += 1

        # HAS_PENALTY
        for pen in extract_penalties(s["text"]):
            pen_id = f"Penalty:{pen}"
            if pen_id not in seen_struct:
                seen_struct.add(pen_id)
                nodes.append({"id": pen_id, "label": "Penalty", "class": pen})
            edges.append({"src": f"Section:{sid}", "dst": pen_id, "type": "HAS_PENALTY"})

    # write interchange
    (args.out_dir / "kg_nodes.jsonl").write_text(
        "\n".join(json.dumps(n, ensure_ascii=False) for n in nodes), "utf-8")
    (args.out_dir / "kg_edges.jsonl").write_text(
        "\n".join(json.dumps(e, ensure_ascii=False) for e in edges), "utf-8")

    # write Cypher
    write_cypher(args.out_dir / "kg.cypher", nodes, edges)

    # APPLIES_TO curation stub: list Book-1 sections as candidate general provisions
    stub = ["general_section_id,applies_to_section_id,note  # fill by a lawyer"]
    for s in sections:
        if (s.get("book") or {}).get("number") == "1":
            stub.append(f"{s['section_id']},,")
    (args.out_dir / "general_provisions.stub.csv").write_text("\n".join(stub), "utf-8")

    cites = sum(1 for e in edges if e["type"] == "CITES")
    print(f"✓ nodes={len(nodes)} edges={len(edges)}  (CITES={cites}, dangling refs={dangling})")
    print(f"  -> kg_nodes.jsonl / kg_edges.jsonl / kg.cypher")
    print(f"  -> general_provisions.stub.csv  (curate APPLIES_TO for multi-hop)")


def write_cypher(path: Path, nodes: list[dict], edges: list[dict]):
    def esc(v):
        return str(v).replace("\\", "\\\\").replace("'", "\\'") if v is not None else ""
    lines = ["CREATE CONSTRAINT IF NOT EXISTS FOR (n:Section) REQUIRE n.section_id IS UNIQUE;"]
    for n in nodes:
        props = ", ".join(f"{k}: '{esc(v)}'" for k, v in n.items()
                          if k not in ("label",) and v is not None)
        lines.append(f"MERGE (:{n['label']} {{{props}}});")
    for e in edges:
        lines.append(
            f"MATCH (a {{id:'{esc(e['src'])}'}}), (b {{id:'{esc(e['dst'])}'}}) "
            f"MERGE (a)-[:{e['type']}]->(b);")
    path.write_text("\n".join(lines), "utf-8")


if __name__ == "__main__":
    main()
