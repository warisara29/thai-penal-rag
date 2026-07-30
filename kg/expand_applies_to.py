"""
Expand APPLIES_TO scope rules -> concrete edges (Book-1 general provision -> offence).

Reads kg/applies_to_rules.json (a lawyer-verifiable scope table) + data/sections.jsonl,
materialises one APPLIES_TO edge per (general_section, offence) the rule covers, and
writes:
  applies_to_edges.jsonl   edges tagged {rule, scope, verified} so the retriever can
                           weight global (พยายาม-applies-to-all) vs specific links
  applies_to_review.csv    the `needs_review` provisions, one row per general section,
                           for the lawyer to hand-curate selective targets

The edge set stays SEPARATE from kg_edges.jsonl (build_kg.py is untouched); the KG-
expansion retriever loads both. Re-run after the lawyer edits the rules file.

Usage:
  python kg/expand_applies_to.py data/sections.jsonl --rules kg/applies_to_rules.json --out-dir data
  python kg/expand_applies_to.py data/sections.jsonl --verified-only   # only emit lawyer-approved rules
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def load_sections(path: Path):
    return [json.loads(l) for l in path.read_text("utf-8").splitlines() if l.strip()]


def book(s):
    return (s.get("book") or {}).get("number")


def is_offence(s):
    return "ระวางโทษ" in s["text"] and book(s) in ("2", "3")


def resolve_targets(scope: str, rule: dict, offences_b2, offences_b3, known: set) -> list[str]:
    if scope == "all_offences":
        return [s["section_id"] for s in offences_b2 + offences_b3]
    if scope == "book2":
        return [s["section_id"] for s in offences_b2]
    if scope == "book3_lahu":
        return [s["section_id"] for s in offences_b3]
    if scope == "by_penalty":
        # placeholder: same universe as all_offences, tagged so retriever/lawyer refine by tier
        return [s["section_id"] for s in offences_b2 + offences_b3]
    if scope == "specific":
        return [t for t in rule.get("targets", []) if t in known]
    return []  # needs_review -> no auto edges


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sections", type=Path)
    ap.add_argument("--rules", type=Path, default=Path("kg/applies_to_rules.json"))
    ap.add_argument("--out-dir", type=Path, default=Path("data"))
    ap.add_argument("--verified-only", action="store_true",
                    help="emit edges only for rules the lawyer marked verified=true")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    secs = load_sections(args.sections)
    known = {s["section_id"] for s in secs}
    offences_b2 = [s for s in secs if is_offence(s) and book(s) == "2"]
    offences_b3 = [s for s in secs if is_offence(s) and book(s) == "3"]

    rules = json.loads(args.rules.read_text("utf-8"))["rules"]
    edges, review, stats, bad_targets = [], [], [], []
    for rule in rules:
        scope = rule["scope"]
        verified = bool(rule.get("verified", False))
        if scope == "specific":  # sanity-check hand-listed targets exist
            for t in rule.get("targets", []):
                if t not in known:
                    bad_targets.append(f"{rule['label']}: target '{t}' not in corpus")
        if scope == "needs_review":
            for gid in rule["general_sections"]:
                review.append({"general_section_id": gid, "rule": rule["label"],
                               "applies_to_section_id": "", "note": rule.get("note", "")})
            stats.append((rule["label"], scope, verified, 0))
            continue
        if args.verified_only and not verified:
            stats.append((rule["label"], scope, verified, 0))
            continue
        targets = resolve_targets(scope, rule, offences_b2, offences_b3, known)
        n = 0
        for gid in rule["general_sections"]:
            if gid not in known:
                bad_targets.append(f"{rule['label']}: general section '{gid}' not in corpus")
                continue
            for tid in targets:
                if tid == gid:
                    continue
                edges.append({
                    "src": f"Section:{gid}", "dst": f"Section:{tid}", "type": "APPLIES_TO",
                    "rule": rule["label"], "scope": scope, "verified": verified,
                })
                n += 1
        stats.append((rule["label"], scope, verified, n))

    (args.out_dir / "applies_to_edges.jsonl").write_text(
        "\n".join(json.dumps(e, ensure_ascii=False) for e in edges), "utf-8")
    with (args.out_dir / "applies_to_review.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["general_section_id", "rule",
                                          "applies_to_section_id", "note"])
        w.writeheader()
        w.writerows(review)

    print(f"offences: ภาค2={len(offences_b2)}  ภาค3(ลหุโทษ)={len(offences_b3)}")
    print("\n=== rule -> edges (scope | verified | #edges) ===")
    for label, scope, verified, n in stats:
        v = "✓" if verified else "unverified"
        print(f"  {label:42s} {scope:13s} {v:10s} {n}")
    print(f"\n  TOTAL edges: {len(edges)}   (verified-only={args.verified_only})")
    print(f"  needs-review rows (hand-curate): {len(review)}")
    if bad_targets:
        print("\n  ! target/section id problems:")
        for b in bad_targets:
            print(f"    - {b}")
    print(f"\n✓ wrote applies_to_edges.jsonl / applies_to_review.csv -> {args.out_dir}")
    print("  Lawyer: verify kg/applies_to_rules.json (flip verified), fill applies_to_review.csv, re-run.")


if __name__ == "__main__":
    main()
