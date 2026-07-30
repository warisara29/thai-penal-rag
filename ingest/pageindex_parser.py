"""
PageIndex ingestion for the Thai Penal Code (ประมวลกฎหมายอาญา).

Turns the raw Penal Code text into the hierarchical tree PageIndex-style,
reasoning-based retrieval expects:

    ภาค (Book) -> ลักษณะ (Title) -> หมวด (Chapter) -> [ส่วน] -> มาตรา (Section)

Why parse instead of letting PageIndex auto-build the tree from a PDF?
Because the Penal Code's structure is *authored* by the legislature and marked
explicitly in the text (ภาค/ลักษณะ/หมวด/มาตรา). Parsing those markers gives a
clean, correct tree with real section IDs — no OCR/heuristic guesswork.

Outputs:
  - penal_tree.json  : nested tree (feed to PageIndex-style tree search / LLM nav)
  - sections.jsonl   : one flat record per มาตรา (feeds KG builder + eval labels)

Input assumption: a UTF-8 .txt of the Penal Code where each structural marker
starts its own line, with the human-readable heading on the following line, e.g.

    ภาค ๑
    บทบัญญัติทั่วไป

    ลักษณะ ๑
    บทบัญญัติที่ใช้แก่ความผิดทั่วไป

    มาตรา ๑ ในกฎหมายนี้ ...

If your source differs, adjust the regexes in MARKERS below — that is the only
place structure is defined.

Usage:
    python pageindex_parser.py penal_code.txt --out-dir ../data
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path

# --- Thai <-> Arabic numeral handling -------------------------------------
THAI_DIGITS = "๐๑๒๓๔๕๖๗๘๙"
_THAI2ARABIC = {ord(t): str(i) for i, t in enumerate(THAI_DIGITS)}
# ordinal words used for inserted sections: มาตรา ๓๓ ทวิ, ๓๓ ตรี, ...
ORDINAL_WORDS = {
    "ทวิ": 2, "ตรี": 3, "จัตวา": 4, "เบญจ": 5, "ฉ": 6,
    "สัตต": 7, "อัฏฐ": 8, "นว": 9, "ทศ": 10,
}


def thai_to_arabic(s: str) -> str:
    return s.translate(_THAI2ARABIC)


# --- Structure markers ----------------------------------------------------
# Each pattern anchors at line start. `level` orders the hierarchy (small=top).
# `key` is the node type. Section (มาตรา) is handled specially (has body text).
MARKERS = [
    ("book",    1, re.compile(r"^\s*ภาค\s+([๐-๙\d]+)\s*$")),
    ("title",   2, re.compile(r"^\s*ลักษณะ\s+([๐-๙\d]+)\s*$")),
    ("chapter", 3, re.compile(r"^\s*หมวด\s+([๐-๙\d]+)\s*$")),
    ("part",    4, re.compile(r"^\s*ส่วนที่?\s+([๐-๙\d]+)\s*$")),
]

# มาตรา ๓๓๕/๑  or  มาตรา ๓๓ ทวิ  or  มาตรา 288
SECTION_RE = re.compile(
    r"^\s*มาตรา\s+([๐-๙\d]+(?:/[๐-๙\d]+)?)\s*"
    r"(ทวิ|ตรี|จัตวา|เบญจ|ฉ|สัตต|อัฏฐ|นว|ทศ)?"
)


def canonical_section_id(num: str, ordinal: str | None) -> str:
    """'๓๓๕/๑' + 'ทวิ' -> '335/1 ทวิ' (stable, sortable-ish key)."""
    base = thai_to_arabic(num)
    return f"{base} {ordinal}".strip() if ordinal else base


@dataclass
class Node:
    node_type: str                 # book|title|chapter|part|section|root
    number: str = ""               # arabic, canonical
    heading: str = ""              # human-readable name
    text: str = ""                 # section body (มาตรา only)
    node_id: str = ""              # e.g. B1.T1.C1.S288
    section_id: str = ""           # e.g. "288" (มาตรา only)
    nodes: list["Node"] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        # drop empty leaves for a compact tree
        if not d["nodes"]:
            d.pop("nodes")
        for k in ("text", "section_id"):
            if not d[k]:
                d.pop(k, None)
        return d


def _match_marker(line: str):
    for key, level, rx in MARKERS:
        m = rx.match(line)
        if m:
            return key, level, thai_to_arabic(m.group(1))
    return None


def parse(text: str) -> tuple[Node, list[dict]]:
    lines = text.splitlines()
    root = Node(node_type="root", heading="ประมวลกฎหมายอาญา")
    # stack holds (level, node); level: book=1..part=4, section lives under top-of-stack
    stack: list[tuple[int, Node]] = [(0, root)]
    flat: list[dict] = []

    cur_section: Node | None = None
    id_prefix = {"book": "B", "title": "T", "chapter": "C", "part": "P"}

    def path_id() -> str:
        parts = []
        for lvl, n in stack[1:]:
            parts.append(f"{id_prefix.get(n.node_type,'X')}{n.number}")
        return ".".join(parts)

    i = 0
    n = len(lines)
    while i < n:
        raw = lines[i]
        line = raw.strip()

        # 1) structural marker (ภาค/ลักษณะ/หมวด/ส่วน)?
        mk = _match_marker(line)
        if mk:
            cur_section = None
            key, level, number = mk
            # heading = next non-empty line, if it isn't itself a marker/section
            heading = ""
            j = i + 1
            while j < n and not lines[j].strip():
                j += 1
            if j < n and not _match_marker(lines[j].strip()) \
                    and not SECTION_RE.match(lines[j].strip()):
                heading = lines[j].strip()
                i = j  # consume heading line
            node = Node(node_type=key, number=number, heading=heading)
            # pop stack to parent level
            while stack and stack[-1][0] >= level:
                stack.pop()
            node.node_id = (path_id() + f".{id_prefix[key]}{number}").lstrip(".")
            # recompute after we know parent — simpler: build from parent chain
            stack[-1][1].nodes.append(node)
            stack.append((level, node))
            # fix node_id using parent chain now that it's attached
            node.node_id = path_id()
            i += 1
            continue

        # 2) section (มาตรา ...)?
        sm = SECTION_RE.match(line)
        if sm:
            sec_id = canonical_section_id(sm.group(1), sm.group(2))
            body = line[sm.end():].strip()
            parent = stack[-1][1]
            node = Node(
                node_type="section",
                number=sec_id,
                section_id=sec_id,
                heading=f"มาตรา {sm.group(1)}"
                        + (f" {sm.group(2)}" if sm.group(2) else ""),
                text=body,
            )
            node.node_id = f"{path_id()}.S{sec_id}".lstrip(".")
            parent.nodes.append(node)
            cur_section = node
            i += 1
            continue

        # 3) continuation line -> append to current section body
        if cur_section is not None and line:
            cur_section.text += ("\n" + line) if cur_section.text else line
        i += 1

    # build flat section list
    def walk(node: Node, ctx: dict):
        c = dict(ctx)
        if node.node_type in id_prefix:
            c[node.node_type] = {"number": node.number, "heading": node.heading}
        for child in node.nodes:
            if child.node_type == "section":
                flat.append({
                    "section_id": child.section_id,
                    "node_id": child.node_id,
                    "heading": child.heading,
                    "text": child.text,
                    "book": c.get("book"),
                    "title": c.get("title"),
                    "chapter": c.get("chapter"),
                    "part": c.get("part"),
                })
            else:
                walk(child, c)

    walk(root, {})
    return root, flat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", type=Path, help="Penal Code .txt (UTF-8)")
    ap.add_argument("--out-dir", type=Path, default=Path("../data"))
    args = ap.parse_args()

    text = args.input.read_text(encoding="utf-8")
    root, flat = parse(text)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tree_path = args.out_dir / "penal_tree.json"
    sec_path = args.out_dir / "sections.jsonl"

    tree_path.write_text(
        json.dumps(root.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with sec_path.open("w", encoding="utf-8") as f:
        for rec in flat:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"✓ parsed {len(flat)} sections (มาตรา)")
    print(f"  tree     -> {tree_path}")
    print(f"  sections -> {sec_path}")


if __name__ == "__main__":
    main()
