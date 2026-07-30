"""
Ingest the PyThaiNLP hand-labeled Criminal Code CSV into this project's
structure-aware format: sections.jsonl + penal_tree.json.

Source: https://github.com/PyThaiNLP/thai-law/releases/tag/criminal-csv-v0.1
        criminal-datasets.csv  (columns: article, text, notes)

The CSV is FLAT — it has มาตรา 1..398 (+ inserted /n articles) but no
ภาค/ลักษณะ/หมวด/ส่วน tree. We re-attach the hierarchy from an authoritative,
human-verifiable table (hierarchy.json).

Design: each hierarchy unit is anchored by its START section only. A unit's
span runs until the next unit at the same or shallower level begins, in the
CSV's document order. A section belongs to the deepest unit whose span
contains it. This makes inserted articles (269/1, 336/2 = ๓๓๖ ทวิ, 366/4)
land correctly with no end-boundary bookkeeping.

Level ranks:  book(0) > title(1) > chapter(2) > part(3)

Outputs (in --out-dir):
  sections.jsonl     one record per มาตรา, with nested book/title/chapter[/part]
  penal_tree.json    nested tree consumed by the PageIndex retriever
  enabling_act.jsonl the intro-* rows (พ.ร.บ. ให้ใช้ฯ) kept out of the corpus tree
  ingest_report.txt  validation: coverage, unassigned, missing start ids

Usage:
  python ingest/csv_to_sections.py data/raw/criminal-datasets.csv \
         --hierarchy data/hierarchy.json --out-dir data
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

ORDINALS = "ทวิ|ตรี|จัตวา|เบญจ|ฉ|สัตต|อัฏฐ|นว|ทศ|เอกาทศ|ทวาทศ"
HEADING_RE = re.compile(
    r"^\s*(มาตรา\s*[๐-๙\d]+(?:/[๐-๙\d]+)?(?:\s*(?:" + ORDINALS + r"))?)\s*"
)

LEVELS = ["book", "title", "chapter", "part"]
RANK = {lvl: i for i, lvl in enumerate(LEVELS)}


def clean(v):
    return (v or "").strip()


def is_junk(article: str, text: str) -> bool:
    return not article or not text or article.lower() == "null"


def slug(num: str) -> str:
    return str(num).replace("/", "_")


def split_heading(text: str, article: str) -> tuple[str, str]:
    m = HEADING_RE.match(text)
    if m:
        return re.sub(r"\s+", " ", m.group(1)).strip(), text[m.end():].strip()
    return f"มาตรา {article}", text.strip()


def load_articles(csv_path: Path):
    with csv_path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    enabling, articles = [], []
    for r in rows:
        art, txt = clean(r.get("article")), clean(r.get("text"))
        if is_junk(art, txt):
            continue
        rec = {"article": art, "text": txt, "notes": clean(r.get("notes"))}
        (enabling if art.startswith("intro") else articles).append(rec)
    return enabling, articles


def resolve_spans(hier: list[dict], pos: dict[str, int], n: int):
    """Give every unit a [p0,p1] span from its start anchor and doc order."""
    missing = []
    for u in hier:
        sid = str(u["start"])
        if sid not in pos:
            missing.append(f"{u['level']} {u.get('number')} {u.get('heading','')}: "
                           f"start '{sid}' not in CSV")
        u["p0"] = pos.get(sid)
    live = [u for u in hier if u["p0"] is not None]
    for u in live:
        # end = just before the next unit that is same-or-shallower level and
        # starts strictly after this one
        later = [v["p0"] for v in live
                 if v["p0"] > u["p0"] and RANK[v["level"]] <= RANK[u["level"]]]
        u["p1"] = (min(later) - 1) if later else n - 1
    return missing


def unit_at(hier: list[dict], level: str, p: int):
    for u in hier:
        if u["level"] == level and u.get("p0") is not None and u["p0"] <= p <= u["p1"]:
            return u
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", type=Path)
    ap.add_argument("--hierarchy", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("data"))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    enabling, articles = load_articles(args.csv)
    pos = {a["article"]: i for i, a in enumerate(articles)}

    hier = json.loads(args.hierarchy.read_text("utf-8"))
    missing = resolve_spans(hier, pos, len(articles))

    sections = []
    for i, a in enumerate(articles):
        heading, body = split_heading(a["text"], a["article"])
        parents = {lvl: unit_at(hier, lvl, i) for lvl in LEVELS}

        parts_id = []
        for lvl in LEVELS:
            if parents[lvl]:
                parts_id.append(f"{lvl[0].upper()}{slug(parents[lvl]['number'])}")
        parts_id.append(f"S{a['article']}")

        def unit(lvl):
            u = parents[lvl]
            return {"number": str(u["number"]), "heading": u["heading"]} if u else None

        rec = {
            "section_id": a["article"],
            "node_id": ".".join(parts_id),
            "heading": heading,
            "text": body,
            "book": unit("book"),
            "title": unit("title"),
            "chapter": unit("chapter"),
        }
        if parents["part"]:
            rec["part"] = unit("part")
        if a["notes"]:
            rec["amendment_note"] = a["notes"]
        sections.append(rec)

    tree = build_tree(hier, sections)

    (args.out_dir / "sections.jsonl").write_text(
        "\n".join(json.dumps(s, ensure_ascii=False) for s in sections), "utf-8")
    (args.out_dir / "penal_tree.json").write_text(
        json.dumps(tree, ensure_ascii=False, indent=2), "utf-8")
    (args.out_dir / "enabling_act.jsonl").write_text(
        "\n".join(json.dumps(e, ensure_ascii=False) for e in enabling), "utf-8")

    # --- integrity report --------------------------------------------------
    no_book = [s["section_id"] for s in sections if not s["book"]]
    # sections in Books 1-2 must have a title; Book 3 (ลหุโทษ) has none by design
    no_title_12 = [s["section_id"] for s in sections
                   if s["book"] and s["book"]["number"] in ("1", "2") and not s["title"]]
    with_ch = sum(1 for s in sections if s["chapter"])
    report = [
        f"sections written           : {len(sections)}",
        f"enabling-act rows          : {len(enabling)} (kept out of corpus tree)",
        f"with book / title / chapter: "
        f"{sum(1 for s in sections if s['book'])} / "
        f"{sum(1 for s in sections if s['title'])} / {with_ch}",
        f"missing book               : {len(no_book)} {no_book[:20]}",
        f"in Book 1-2 but no title   : {len(no_title_12)} {no_title_12[:20]}",
        f"missing start ids in CSV   : {len(missing)}",
        *[f"  - {m}" for m in missing],
    ]
    (args.out_dir / "ingest_report.txt").write_text("\n".join(report), "utf-8")
    print("\n".join(report))
    print(f"\n✓ wrote sections.jsonl / penal_tree.json / enabling_act.jsonl -> {args.out_dir}")


def build_tree(hier: list[dict], sections: list[dict]) -> dict:
    """Nest book>title>chapter>part>section using resolved spans."""
    root = {"node_type": "root", "number": "", "heading": "ประมวลกฎหมายอาญา",
            "node_id": "", "nodes": []}
    nodes: dict[str, dict] = {}  # node_id -> node

    def node_id_for(u):
        pieces = []
        for lvl in LEVELS[:RANK[u["level"]]]:
            parent = unit_at(hier, lvl, u["p0"])
            if parent:
                pieces.append(f"{lvl[0].upper()}{slug(parent['number'])}")
        pieces.append(f"{u['level'][0].upper()}{slug(u['number'])}")
        return ".".join(pieces)

    # create containers in document order (shallow to deep so parents exist)
    for u in sorted([h for h in hier if h.get("p0") is not None],
                    key=lambda x: (RANK[x["level"]], x["p0"])):
        nid = node_id_for(u)
        node = {"node_type": u["level"], "number": str(u["number"]),
                "heading": u["heading"], "text": "", "node_id": nid,
                "section_id": "", "nodes": []}
        nodes[nid] = node
        parent_id = ".".join(nid.split(".")[:-1])
        (nodes.get(parent_id, root))["nodes"].append(node)

    for s in sections:
        container_id = ".".join(s["node_id"].split(".")[:-1])
        (nodes.get(container_id, root))["nodes"].append({
            "node_type": "section", "number": s["section_id"],
            "heading": s["heading"], "text": s["text"],
            "node_id": s["node_id"], "section_id": s["section_id"], "nodes": [],
        })
    return root


if __name__ == "__main__":
    main()
