"""A3 — QC the Penal-Code citation extraction.

Joins penal cases to their judgment text (via the case_id in the "คำพิพากษาศาลฎีกาที่ …"
header, because the citation metadata and the corpus text use different doc_id schemes),
samples N cases, and compares two INDEPENDENT extractors on the raw text:

    reference  = the dataset's pre-parsed ``sections_cited`` (normalized to Penal keys)
    predicted  = caselaw.sections.extract_from_text (our context-aware regex)

Reports micro/macro precision & recall of our extractor against the dataset field, and writes
a disagreement worksheet (data/caselaw/qc_sample.csv) for human (lawyer) adjudication — the
authoritative precision check in task F. This is parser-vs-parser agreement, not vs ground
truth; high agreement validates both, disagreements are the review queue.

Usage:
    python -m caselaw.qc_extract --val data/caselaw/raw/val.full.jsonl --n 100 --seed 20261006
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re

from .sections import extract_from_text

_HDR = re.compile(r"คำพิพากษาศาลฎีกาที่\s*([\d]+(?:\s*[-–]\s*\d+)?/\d+)")


def _load_text_by_caseid(val_path: str) -> dict[str, str]:
    by_id: dict[str, str] = {}
    with open(val_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            try:
                d = json.loads(line)
            except Exception:
                continue
            if d.get("source") != "supreme_court":
                continue
            t = d.get("text", "")
            m = _HDR.match(t.strip())
            if m:
                by_id.setdefault(m.group(1).replace(" ", ""), t)
    return by_id


def run(val_path: str, citations_path: str, n: int, seed: int, out_csv: str) -> dict:
    text_by_id = _load_text_by_caseid(val_path)
    cases = {}
    with open(citations_path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if d["case_id"] in text_by_id and d["case_id"] not in cases:
                cases[d["case_id"]] = d
    joinable = sorted(cases)
    rng = random.Random(seed)
    sample = rng.sample(joinable, min(n, len(joinable)))

    tp = fp = fn = 0
    precs, recs = [], []
    rows = []
    for cid in sample:
        text = text_by_id[cid]
        ref = set(cases[cid]["sections"])          # dataset's parse, already normalized
        pred = set(extract_from_text(text))
        inter = ref & pred
        only_ref = ref - pred     # recall misses
        only_pred = pred - ref    # precision errors
        tp += len(inter); fp += len(only_pred); fn += len(only_ref)
        if pred:
            precs.append(len(inter) / len(pred))
        if ref:
            recs.append(len(inter) / len(ref))
        rows.append({
            "case_id": cid,
            "reference_sections": " ".join(sorted(ref, key=_sortkey)),
            "predicted_sections": " ".join(sorted(pred, key=_sortkey)),
            "only_in_dataset_(recall_miss)": " ".join(sorted(only_ref, key=_sortkey)),
            "only_in_ours_(precision_err)": " ".join(sorted(only_pred, key=_sortkey)),
            "agree": not only_ref and not only_pred,
        })

    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    micro_p = tp / (tp + fp) if (tp + fp) else 1.0
    micro_r = tp / (tp + fn) if (tp + fn) else 1.0
    return {
        "joinable_cases": len(joinable),
        "sampled": len(sample),
        "micro_precision": micro_p,
        "micro_recall": micro_r,
        "macro_precision": sum(precs) / len(precs) if precs else 1.0,
        "macro_recall": sum(recs) / len(recs) if recs else 1.0,
        "exact_agree_cases": sum(r["agree"] for r in rows),
        "tp": tp, "fp": fp, "fn": fn,
        "out_csv": out_csv,
    }


def _sortkey(k: str):
    m = re.match(r"(\d+)", k)
    return (int(m.group(1)) if m else 0, k)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--val", default="data/caselaw/raw/val.full.jsonl")
    ap.add_argument("--citations", default="data/caselaw/penal_citations.jsonl")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=20261006)
    ap.add_argument("--out", default="data/caselaw/qc_sample.csv")
    args = ap.parse_args()
    s = run(args.val, args.citations, args.n, args.seed, args.out)
    print(f"joinable penal cases with text: {s['joinable_cases']}  |  sampled: {s['sampled']}")
    print("Direction 1 — is the curated dataset field faithful to the judgment text?")
    print(f"  dataset-field corroboration (recall of ref): micro={s['micro_recall']:.3f} "
          f"macro={s['macro_recall']:.3f}  (dataset sections NOT found in text: {s['fn']})")
    print("Direction 2 — the full-text sweep also captures non-operative mentions:")
    print(f"  operative fraction of all in-text mentions (precision): micro={s['micro_precision']:.3f} "
          f"macro={s['macro_precision']:.3f}  (sentencing/procedural extras: {s['fp']})")
    print(f"exact-agreement cases: {s['exact_agree_cases']}/{s['sampled']}")
    print(f"disagreement worksheet -> {s['out_csv']} (for lawyer adjudication, task F)")


if __name__ == "__main__":
    main()
