"""A2 — extract Penal-Code section citations per Supreme Court case.

Reads the dataset's pre-parsed ``metadata/deka_citations.jsonl`` (``sections_cited`` is already
Arabic-numeral, so this is normalization + Penal-Code filtering, not raw parsing), and writes
one record per criminal case that cites at least one Penal-Code section:

    {deka_id, case_id, year_be, year_ce, sections, raw_mentions}

``sections`` are canonical cite keys (see caselaw.sections.parse_penal_cite): "288", "269/5",
"277 ทวิ". Mapping to corpus section_ids is A4 (caselaw.map_sections).

Usage:
    python -m caselaw.extract_citations \
        --in  data/caselaw/raw/deka_citations.jsonl \
        --out data/caselaw/penal_citations.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import Counter

from .sections import parse_penal_cite


def run(in_path: str, out_path: str) -> dict:
    n_total = n_penal_cases = 0
    sec_freq: Counter = Counter()
    with open(in_path, encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            n_total += 1
            raw = d.get("sections_cited") or []
            penal_raw, sections, seen = [], [], set()
            for s in raw:
                key = parse_penal_cite(s)
                if key is None:
                    continue
                penal_raw.append(s)
                if key not in seen:       # dedupe, preserve first-seen order
                    seen.add(key)
                    sections.append(key)
            if not sections:
                continue
            n_penal_cases += 1
            for k in sections:
                sec_freq[k] += 1
            try:
                year_be = int(d.get("case_year"))
            except (TypeError, ValueError):
                year_be = None
            rec = {
                "deka_id": d.get("doc_id"),
                "case_id": d.get("case_id"),
                "year_be": year_be,
                "year_ce": (year_be - 543) if year_be else None,
                "sections": sections,
                "raw_mentions": penal_raw,
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return {
        "records_scanned": n_total,
        "penal_cases": n_penal_cases,
        "distinct_section_keys": len(sec_freq),
        "top20": sec_freq.most_common(20),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="in_path", default="data/caselaw/raw/deka_citations.jsonl")
    ap.add_argument("--out", dest="out_path", default="data/caselaw/penal_citations.jsonl")
    args = ap.parse_args()
    stats = run(args.in_path, args.out_path)
    print(f"scanned {stats['records_scanned']} cases; "
          f"{stats['penal_cases']} cite the Penal Code "
          f"({stats['penal_cases'] / stats['records_scanned']:.1%})")
    print(f"distinct Penal section keys cited: {stats['distinct_section_keys']}")
    print("top-20 cited sections:")
    for k, c in stats["top20"]:
        print(f"  {c:6d}  มาตรา {k}")
    print(f"\nwrote {args.out_path}")


if __name__ == "__main__":
    main()
