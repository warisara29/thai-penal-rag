"""Judge validation (§6b): sample ~20% of (question, answer) pairs for a lawyer
to score, then compute judge-human agreement. We only trust the automated judge
if agreement is high; the calibration is reported regardless.

  --make       write audit_worksheet.csv (blank human columns) for the lawyer
  --agreement  read the filled worksheet + verdicts/, report Cohen's κ + Spearman ρ

No scipy: κ and ρ implemented inline.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

SEED = 20260729


def _rows(answers_dir: Path):
    for p in sorted(answers_dir.glob("*.jsonl")):
        for l in p.read_text("utf-8").splitlines():
            if l.strip():
                yield json.loads(l)


def make(answers_dir: Path, out: Path, frac: float):
    rows = list(_rows(answers_dir))
    rng = random.Random(SEED)
    rng.shuffle(rows)
    pick = rows[: max(1, round(len(rows) * frac))]
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "arm", "question", "answer",
                    "human_correct_1to5", "human_correct_bool", "notes"])
        for r in pick:
            w.writerow([r["id"], r["arm"], r["question"], r["answer"], "", "", ""])
    print(f"✓ wrote {len(pick)}/{len(rows)} rows ({frac:.0%}) -> {out}")
    print("  lawyer fills human_correct_1to5 (1-5) and human_correct_bool (0/1).")


def _kappa(a: list[int], b: list[int]) -> float:
    n = len(a)
    if not n:
        return float("nan")
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    cats = set(a) | set(b)
    pe = sum((a.count(c) / n) * (b.count(c) / n) for c in cats)
    return (po - pe) / (1 - pe) if pe != 1 else 1.0


def _spearman(a: list[float], b: list[float]) -> float:
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for t in range(i, j + 1):
                r[order[t]] = avg
            i = j + 1
        return r
    ra, rb = rank(a), rank(b)
    n = len(a)
    ma, mb = sum(ra) / n, sum(rb) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = (sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb)) ** 0.5
    return num / den if den else float("nan")


def agreement(worksheet: Path, verdicts_dir: Path):
    vmap = {}
    for p in verdicts_dir.glob("*.jsonl"):
        for l in p.read_text("utf-8").splitlines():
            if l.strip():
                v = json.loads(l)
                vmap[(v["id"], v["arm"])] = v
    hc_bool, jc_bool, hc_score, jc_score = [], [], [], []
    for row in csv.DictReader(worksheet.open(encoding="utf-8")):
        v = vmap.get((row["id"], row["arm"]))
        c = v.get("correctness") if v else None
        if not c:
            continue
        if row.get("human_correct_bool", "").strip() in ("0", "1"):
            hc_bool.append(int(row["human_correct_bool"])); jc_bool.append(1 if c["correct"] else 0)
        if row.get("human_correct_1to5", "").strip().isdigit():
            hc_score.append(int(row["human_correct_1to5"])); jc_score.append(int(c["score"]))
    print(f"paired (binary): {len(hc_bool)}   paired (1-5): {len(hc_score)}")
    if hc_bool:
        print(f"  Cohen's κ (correct/incorrect): {_kappa(hc_bool, jc_bool):.3f}")
    if hc_score:
        print(f"  Spearman ρ (1-5 score):        {_spearman(hc_score, jc_score):.3f}")
    if not hc_bool and not hc_score:
        print("  no filled rows joined a judge verdict — fill the worksheet and run the judge first")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--make", action="store_true")
    ap.add_argument("--agreement", action="store_true")
    ap.add_argument("--answers-dir", type=Path, default=Path("generation/answers"))
    ap.add_argument("--verdicts-dir", type=Path, default=Path("generation/verdicts"))
    ap.add_argument("--worksheet", type=Path, default=Path("generation/audit_worksheet.csv"))
    ap.add_argument("--frac", type=float, default=0.20)
    args = ap.parse_args()
    if args.make:
        make(args.answers_dir, args.worksheet, args.frac)
    elif args.agreement:
        agreement(args.worksheet, args.verdicts_dir)
    else:
        ap.error("choose --make or --agreement")


if __name__ == "__main__":
    main()
