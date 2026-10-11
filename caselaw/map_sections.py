"""A4 — map extracted Penal-Code cite keys onto corpus section_ids.

Rewrites penal_citations.jsonl with a resolved ``section_ids`` field (corpus ids) alongside the
raw ``sections`` keys, and reports every key that fails to resolve (repealed sections, numbers
absent from the current code, OCR noise). Mapping uses caselaw.sections.CorpusSectionIndex,
which learns the ทวิ/ตรี -> /n convention from the corpus headings.

Usage:
    python -m caselaw.map_sections \
        --citations data/caselaw/penal_citations.jsonl \
        --sections  data/sections.jsonl \
        --out       data/caselaw/penal_citations.mapped.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import Counter

from .sections import CorpusSectionIndex


def run(citations: str, sections: str, out_path: str) -> dict:
    idx = CorpusSectionIndex.from_sections_jsonl(sections)
    n = 0
    mapped_case_cov: list[float] = []   # per-case fraction of keys that resolved
    unmapped_freq: Counter = Counter()
    distinct_keys: set = set()
    distinct_mapped: set = set()
    with open(citations, encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            n += 1
            ids, unmapped = [], []
            for key in d["sections"]:
                distinct_keys.add(key)
                sid = idx.resolve(key)
                if sid is None:
                    unmapped.append(key)
                    unmapped_freq[key] += 1
                else:
                    distinct_mapped.add(key)
                    if sid not in ids:
                        ids.append(sid)
            d["section_ids"] = ids
            d["unmapped"] = unmapped
            keys = d["sections"]
            mapped_case_cov.append(len(ids) / len(keys) if keys else 1.0)
            fout.write(json.dumps(d, ensure_ascii=False) + "\n")
    return {
        "cases": n,
        "distinct_keys": len(distinct_keys),
        "distinct_mapped": len(distinct_mapped),
        "distinct_unmapped": len(distinct_keys - distinct_mapped),
        "macro_case_coverage": sum(mapped_case_cov) / len(mapped_case_cov) if mapped_case_cov else 0,
        "unmapped_top": unmapped_freq.most_common(30),
        "total_unmapped_mentions": sum(unmapped_freq.values()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--citations", default="data/caselaw/penal_citations.jsonl")
    ap.add_argument("--sections", default="data/sections.jsonl")
    ap.add_argument("--out", default="data/caselaw/penal_citations.mapped.jsonl")
    args = ap.parse_args()
    s = run(args.citations, args.sections, args.out)
    print(f"cases: {s['cases']}")
    print(f"distinct cite keys: {s['distinct_keys']}  "
          f"mapped: {s['distinct_mapped']}  unmapped: {s['distinct_unmapped']}")
    print(f"macro per-case coverage: {s['macro_case_coverage']:.3f}")
    print(f"unmapped mentions (token occurrences): {s['total_unmapped_mentions']}")
    print("top unmapped keys (likely repealed / non-current / procedural leakage):")
    for k, c in s["unmapped_top"]:
        print(f"  {c:5d}  มาตรา {k}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
