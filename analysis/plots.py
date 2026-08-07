"""Publication figures from a run snapshot. Self-contained: computes per-arm
means from the snapshot files (retrieval rankings + judge verdicts) and reads the
bootstrap CIs from analysis/report.*.json for error bars.

Usage:
  .venv/bin/python -m analysis.plots --run runs/2026-08-03
Outputs PNGs into <run>/figs/.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# additive-design order + descriptive labels + which arms are the additive comparison set
ORDER = ["C0", "R0", "R1", "A1", "A3", "A5", "A2", "A4B", "C1"]
LABEL = {"C0": "closed", "R0": "keyword", "R1": "dense", "A1": "hybrid",
         "A3": "pi", "A5": "hybrid\n+pi", "A2": "hybrid\n+kg", "A4B": "hybrid\n+pi+kg",
         "A4": "pi+kg", "C1": "ORACLE"}
CORE = {"A1", "A3", "A5", "A2", "A4B"}


def _load(path):
    return [json.loads(l) for l in Path(path).read_text("utf-8").splitlines() if l.strip()]


def recall(ranked, gold, k=5):
    return sum(1 for g in gold if g in set(ranked[:k])) / len(gold) if gold else None


def cover(ranked, gplus, k=5):
    return 1.0 if set(gplus) <= set(ranked[:k]) else 0.0


def retrieval_means(run: Path):
    out = {}
    for p in (run / "retrieval").glob("*.jsonl"):
        rows = _load(p)
        rec = [recall(r["ranked"], r["gold"]) for r in rows if r["gold"]]
        cov = [cover(r["ranked"], list(dict.fromkeys(r["gold"] + r.get("supporting", []))))
               for r in rows if r["gold"]]
        sup = [recall(r["ranked"], r["supporting"]) for r in rows if r.get("supporting")]
        out[p.stem] = {"recall": mean(rec), "coverage": mean(cov),
                       "support_recall": mean([s for s in sup if s is not None])}
    return out


def verdict_means(run: Path):
    out = {}
    for p in (run / "verdicts").glob("*.jsonl"):
        rows = _load(p)
        cor = [1 if v["correctness"]["correct"] else 0 for v in rows if "correctness" in v]
        cg = [v["claim_grounding"]["rate"] for v in rows if "claim_grounding" in v]
        out[p.stem] = {"correct": mean(cor) if cor else 0.0,
                       "claim_ground": mean(cg) if cg else 0.0}
    return out


def ci_from_report(run: Path, metric: str):
    f = run / "analysis" / f"report.{metric}.json"
    if not f.exists():
        return {}
    rep = json.loads(f.read_text("utf-8"))["per_arm"]
    return {a: (d["mean"], d["mean"] - d["lo"], d["hi"] - d["mean"]) for a, d in rep.items()}


def bars(ax, arms, values, colors, ylabel, title, yerr=None):
    x = range(len(arms))
    ax.bar(x, values, color=colors, yerr=yerr, capsize=3, edgecolor="white", linewidth=0.5)
    ax.set_xticks(list(x))
    ax.set_xticklabels([LABEL[a] for a in arms], fontsize=8)
    ax.set_ylabel(ylabel); ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_ylim(0, 1); ax.grid(axis="y", alpha=0.3)
    for i, v in enumerate(values):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=7)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, default=Path("runs/2026-08-03"))
    args = ap.parse_args()
    figs = args.run / "figs"; figs.mkdir(exist_ok=True)

    rm, vm = retrieval_means(args.run), verdict_means(args.run)
    col = lambda a: "#c0392b" if a in CORE else "#7f8c8d"

    # Fig 1: retrieval — recall@5 with bootstrap CI (arms that have retrieval)
    arms_r = [a for a in ORDER if a in rm]
    ci = ci_from_report(args.run, "hit")
    # per-arm whiskers: CI where the report has it, zero-length for spliced-in arms (e.g. A4B)
    yerr = ([[ci.get(a, (0, 0, 0))[1] for a in arms_r], [ci.get(a, (0, 0, 0))[2] for a in arms_r]]
            if any(a in ci for a in arms_r) else None)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars(ax, arms_r, [rm[a]["recall"] for a in arms_r], [col(a) for a in arms_r],
         "Recall@5", "Retrieval Recall@5 (95% bootstrap CI)", yerr)
    fig.tight_layout(); fig.savefig(figs / "fig1_recall.png", dpi=150); plt.close(fig)

    # Fig 2: answer correctness ladder with CI + oracle ceiling
    arms_c = [a for a in ORDER if a in vm]
    cic = ci_from_report(args.run, "correct")
    yerr2 = ([[cic.get(a, (0, 0, 0))[1] for a in arms_c], [cic.get(a, (0, 0, 0))[2] for a in arms_c]]
             if any(a in cic for a in arms_c) else None)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars(ax, arms_c, [vm[a]["correct"] for a in arms_c], [col(a) for a in arms_c],
         "Answer correctness", "Answer Correctness (judge, 95% CI)", yerr2)
    if "C1" in vm:
        ax.axhline(vm["C1"]["correct"], ls="--", color="#27ae60", lw=1)
        ax.text(0, vm["C1"]["correct"] + 0.01, "oracle ceiling", color="#27ae60", fontsize=8)
    fig.tight_layout(); fig.savefig(figs / "fig2_correctness.png", dpi=150); plt.close(fig)

    # Fig 3: three axes side by side (retrieval recall, correctness, claim-grounding)
    arms3 = [a for a in ORDER if a in vm]
    import numpy as np
    x = np.arange(len(arms3)); w = 0.27
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    ax.bar(x - w, [rm.get(a, {}).get("recall", 0) for a in arms3], w, label="Recall@5", color="#2980b9")
    ax.bar(x, [vm[a]["correct"] for a in arms3], w, label="Correct%", color="#c0392b")
    ax.bar(x + w, [vm[a]["claim_ground"] for a in arms3], w, label="Claim-grounding", color="#f39c12")
    ax.set_xticks(x); ax.set_xticklabels([LABEL[a] for a in arms3], fontsize=8)
    ax.set_ylim(0, 1); ax.set_ylabel("score"); ax.grid(axis="y", alpha=0.3)
    ax.set_title("Three measurement axes per arm", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(figs / "fig3_three_axes.png", dpi=150); plt.close(fig)

    # Fig 4: KG effect on multi-hop support recall
    arms_s = [a for a in ["A1", "A3", "A5", "A2", "A4B"] if a in rm]
    kgcol = {"A1": "#7f8c8d", "A3": "#7f8c8d", "A5": "#7f8c8d", "A2": "#c0392b", "A4B": "#c0392b"}
    fig, ax = plt.subplots(figsize=(7, 4))
    bars(ax, arms_s, [rm[a]["support_recall"] for a in arms_s], [kgcol[a] for a in arms_s],
         "Support-recall@5", "KG effect on multi-hop (support-recall@5)")
    fig.tight_layout(); fig.savefig(figs / "fig4_kg_support.png", dpi=150); plt.close(fig)

    print(f"✓ wrote 4 figures -> {figs}/")
    for f in sorted(figs.glob("*.png")):
        print(f"  {f}")


if __name__ == "__main__":
    main()
