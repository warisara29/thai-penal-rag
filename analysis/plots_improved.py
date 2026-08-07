"""Progress-update figures for the additive-full-system fix (PIKG+ / A4B).

Reuses the 2026-08-03 snapshot for every arm and splices in the freshly re-run
PIKG+ from the improved scratch dirs. Produces three comparison PNGs:
  figA_fix_effect.png   — PIKG-RAG (before) vs PIKG+ (after), key axes
  figB_ladder.png       — correctness ladder with PIKG+ on top + oracle ceiling
  figC_head_to_head.png — PIKG+ vs old-best B3+KG across six metrics

Usage:
  .venv/bin/python -m analysis.plots_improved
Outputs into analysis/figs_improved/.
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SNAP = Path("runs/2026-08-03")
IMPR_RETR = Path("/tmp/retr_improved/A4B.jsonl")
IMPR_VERD = Path("/tmp/gen_improved/verdicts/A4B.jsonl")
OUT = Path("analysis/figs_improved"); OUT.mkdir(parents=True, exist_ok=True)

LABEL = {"C0": "B0\nclosed", "R0": "B1\nkeyword", "R1": "B2\ndense", "A1": "B3\nhybrid",
         "A2": "B3+KG", "A3": "B3+PI", "A4": "PIKG-RAG", "A4B": "PIKG+", "C1": "ORACLE"}


def _load(p):
    return [json.loads(l) for l in Path(p).read_text("utf-8").splitlines() if l.strip()]


def recall(ranked, gold, k=5):
    return sum(1 for g in gold if g in set(ranked[:k])) / len(gold) if gold else None


def cover(ranked, gplus, k=5):
    return 1.0 if set(gplus) <= set(ranked[:k]) else 0.0


def retr_means(path):
    rows = _load(path)
    rec = [recall(r["ranked"], r["gold"]) for r in rows if r["gold"]]
    cov = [cover(r["ranked"], list(dict.fromkeys(r["gold"] + r.get("supporting", []))))
           for r in rows if r["gold"]]
    sup = [recall(r["ranked"], r["supporting"]) for r in rows if r.get("supporting")]
    return {"recall": mean(rec), "coverage": mean(cov),
            "support_recall": mean([s for s in sup if s is not None])}


def verd_means(path):
    rows = _load(path)
    cor = [1 if v["correctness"]["correct"] else 0 for v in rows if "correctness" in v]
    cg = [v["claim_grounding"]["rate"] for v in rows if "claim_grounding" in v]
    sc = [v["correctness"].get("score", 0) for v in rows if "correctness" in v]
    return {"correct": mean(cor) if cor else 0.0,
            "claim_ground": mean(cg) if cg else 0.0,
            "score": mean(sc) if sc else 0.0}


# ---- gather per-arm means (snapshot arms + improved A4B) ----
RM, VM = {}, {}
for p in (SNAP / "retrieval").glob("*.jsonl"):
    RM[p.stem] = retr_means(p)
for p in (SNAP / "verdicts").glob("*.jsonl"):
    VM[p.stem] = verd_means(p)
RM["A4B"] = retr_means(IMPR_RETR)
VM["A4B"] = verd_means(IMPR_VERD)

RED, GREEN, GREY, BLUE, ORANGE = "#c0392b", "#27ae60", "#95a5a6", "#2980b9", "#f39c12"


# ---- Fig A: the fix effect (before vs after) ----
metrics = ["recall@5", "correct%", "claim-ground", "judge score /5"]
before = [RM["A4"]["recall"], VM["A4"]["correct"], VM["A4"]["claim_ground"], VM["A4"]["score"] / 5]
after = [RM["A4B"]["recall"], VM["A4B"]["correct"], VM["A4B"]["claim_ground"], VM["A4B"]["score"] / 5]
x = np.arange(len(metrics)); w = 0.38
fig, ax = plt.subplots(figsize=(8, 4.8))
b1 = ax.bar(x - w / 2, before, w, label="PIKG-RAG (replacement, before)", color=GREY)
b2 = ax.bar(x + w / 2, after, w, label="PIKG+ (additive, after)", color=RED)
for bars in (b1, b2):
    for bb in bars:
        ax.text(bb.get_x() + bb.get_width() / 2, bb.get_height() + 0.015,
                f"{bb.get_height():.2f}", ha="center", fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(metrics, fontsize=9)
ax.set_ylim(0, 1); ax.set_ylabel("score"); ax.grid(axis="y", alpha=0.3)
ax.set_title("The fix: replacement → additive full system", fontsize=12, fontweight="bold")
ax.legend(fontsize=9)
fig.tight_layout(); fig.savefig(OUT / "figA_fix_effect.png", dpi=150); plt.close(fig)


# ---- Fig B: correctness ladder with PIKG+ on top + oracle ceiling ----
ladder = ["C0", "R0", "R1", "A1", "A2", "A4B"]  # B0..B3, B3+KG, PIKG+
vals = [VM[a]["correct"] for a in ladder]
colors = [GREY, GREY, GREY, GREY, GREY, RED]
fig, ax = plt.subplots(figsize=(8.5, 4.8))
bar = ax.bar(range(len(ladder)), vals, color=colors, edgecolor="white")
for i, v in enumerate(vals):
    ax.text(i, v + 0.012, f"{v:.3f}", ha="center", fontsize=8,
            fontweight="bold" if ladder[i] == "A4B" else "normal")
ax.axhline(VM["C1"]["correct"], ls="--", color=GREEN, lw=1.2)
ax.text(0, VM["C1"]["correct"] + 0.012, f"ORACLE ceiling {VM['C1']['correct']:.3f}",
        color=GREEN, fontsize=9)
ax.set_xticks(range(len(ladder))); ax.set_xticklabels([LABEL[a] for a in ladder], fontsize=8)
ax.set_ylim(0, 0.9); ax.set_ylabel("answer correctness (judge)")
ax.grid(axis="y", alpha=0.3)
ax.set_title("Correctness ladder — PIKG+ now leads the real arms", fontsize=12, fontweight="bold")
fig.tight_layout(); fig.savefig(OUT / "figB_ladder.png", dpi=150); plt.close(fig)


# ---- Fig C: PIKG+ vs old-best B3+KG head-to-head ----
axes = ["recall@5", "coverage@5", "support_recall@5", "correct%", "claim-ground", "score/5"]
kg = [RM["A2"]["recall"], RM["A2"]["coverage"], RM["A2"]["support_recall"],
      VM["A2"]["correct"], VM["A2"]["claim_ground"], VM["A2"]["score"] / 5]
pk = [RM["A4B"]["recall"], RM["A4B"]["coverage"], RM["A4B"]["support_recall"],
      VM["A4B"]["correct"], VM["A4B"]["claim_ground"], VM["A4B"]["score"] / 5]
x = np.arange(len(axes)); w = 0.38
fig, ax = plt.subplots(figsize=(9.5, 4.8))
b1 = ax.bar(x - w / 2, kg, w, label="B3+KG (old best)", color=BLUE)
b2 = ax.bar(x + w / 2, pk, w, label="PIKG+ (new best)", color=RED)
for bars in (b1, b2):
    for bb in bars:
        ax.text(bb.get_x() + bb.get_width() / 2, bb.get_height() + 0.012,
                f"{bb.get_height():.2f}", ha="center", fontsize=7)
ax.set_xticks(x); ax.set_xticklabels(axes, fontsize=8)
ax.set_ylim(0, 1); ax.set_ylabel("score"); ax.grid(axis="y", alpha=0.3)
ax.set_title("PIKG+ vs old-best B3+KG — wins on all six axes", fontsize=12, fontweight="bold")
ax.legend(fontsize=9)
fig.tight_layout(); fig.savefig(OUT / "figC_head_to_head.png", dpi=150); plt.close(fig)

print(f"✓ wrote 3 figures -> {OUT}/")
for f in sorted(OUT.glob("*.png")):
    print(f"  {f}")
