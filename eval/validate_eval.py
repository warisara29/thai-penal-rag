"""
Validate an eval_set JSONL against schema.json and cross-check gold_sections
against the parsed corpus (sections.jsonl) so labels can't reference a
มาตรา that doesn't exist.

Usage:
    python validate_eval.py eval_set.jsonl --schema schema.json --sections ../data/sections.jsonl

Exits non-zero if any item is invalid — wire it into CI / a pre-commit hook.
No third-party deps required (minimal built-in checks); if `jsonschema` is
installed it is used for full schema validation.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REQUIRED = ["id", "question", "question_type", "gold_sections", "reference_answer"]
TYPES = {"lookup", "multi_hop", "penalty", "definition", "exception", "unanswerable"}


def load_jsonl(p: Path):
    out = []
    for i, line in enumerate(p.read_text("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            out.append((i, json.loads(line)))
        except json.JSONDecodeError as e:
            print(f"✗ line {i}: invalid JSON: {e}")
            sys.exit(1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("eval_set", type=Path)
    ap.add_argument("--schema", type=Path, default=Path("schema.json"))
    ap.add_argument("--sections", type=Path, help="sections.jsonl to check gold ids exist")
    args = ap.parse_args()

    items = load_jsonl(args.eval_set)
    errors = 0

    # optional: full schema validation if jsonschema present
    try:
        import jsonschema  # type: ignore
        schema = json.loads(args.schema.read_text("utf-8"))
        validator = jsonschema.Draft7Validator(schema)
    except ImportError:
        validator = None
        print("· jsonschema not installed — running built-in checks only "
              "(pip install jsonschema for full validation)")

    known = None
    if args.sections and args.sections.exists():
        known = {json.loads(l)["section_id"]
                 for l in args.sections.read_text("utf-8").splitlines() if l.strip()}

    ids = set()
    type_counts = Counter()
    for lineno, it in items:
        loc = f"line {lineno} (id={it.get('id','?')})"
        if validator:
            for err in validator.iter_errors(it):
                print(f"✗ {loc}: {err.message}")
                errors += 1
        else:
            for k in REQUIRED:
                if k not in it:
                    print(f"✗ {loc}: missing required field '{k}'"); errors += 1
            if it.get("question_type") not in TYPES:
                print(f"✗ {loc}: bad question_type {it.get('question_type')!r}"); errors += 1

        # semantic checks (beyond schema)
        if it.get("id") in ids:
            print(f"✗ {loc}: duplicate id"); errors += 1
        ids.add(it.get("id"))
        type_counts[it.get("question_type")] += 1

        qtype = it.get("question_type")
        gold = it.get("gold_sections", [])
        if qtype == "unanswerable" and gold:
            print(f"✗ {loc}: unanswerable item must have empty gold_sections"); errors += 1
        if qtype != "unanswerable" and not gold:
            print(f"✗ {loc}: {qtype} item needs at least one gold_section"); errors += 1

        if known is not None:
            for sid in gold + it.get("supporting_sections", []):
                if sid not in known:
                    print(f"✗ {loc}: section '{sid}' not found in corpus"); errors += 1

    print("\n--- distribution ---")
    for t, c in type_counts.most_common():
        print(f"  {t:12s} {c}")
    print(f"  total        {len(items)}")

    if errors:
        print(f"\n✗ {errors} error(s)"); sys.exit(1)
    print("\n✓ all items valid")


if __name__ == "__main__":
    main()
