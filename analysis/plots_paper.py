"""Paper-specific figures not in the standard plots.py set:
  fig_kg_per_type.png  — the KG-effect-by-type result from the controlled comparison (§4.4)
  fig_architecture.png — schematic of the additive pipeline with the reserved KG slot (Figure 1)

Usage: .venv/bin/python -m analysis.plots_paper
Outputs into runs/2026-08-07/figs/.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path("runs/2026-08-07/figs")
OUT.mkdir(parents=True, exist_ok=True)
RED, GREEN, GREY, BLUE, DARK = "#c0392b", "#27ae60", "#95a5a6", "#2980b9", "#2c3e50"


# ---- Figure: KG effect per question type (controlled, primaries held fixed) ----
def kg_per_type():
    # from runs/2026-08-07/controlled/ (paper §4.4): KG on − KG off, same 4 primaries
    types = ["multi_hop", "lookup", "definition", "exception", "penalty"]
    off = [0.311, 0.680, 1.000, 0.760, 0.891]
    on = [0.489, 0.707, 1.000, 0.720, 0.818]
    delta = [o - f for o, f in zip(on, off)]
    order = sorted(range(len(types)), key=lambda i: delta[i], reverse=True)
    types = [types[i] for i in order]; delta = [delta[i] for i in order]
    colors = [GREEN if d > 0.001 else GREY if abs(d) <= 0.001 else RED for d in delta]

    fig, ax = plt.subplots(figsize=(7.5, 4))
    y = range(len(types))
    ax.barh(list(y), delta, color=colors, edgecolor="white")
    ax.axvline(0, color=DARK, lw=1)
    for i, d in enumerate(delta):
        ax.text(d + (0.006 if d >= 0 else -0.006), i, f"{d:+.3f}",
                va="center", ha="left" if d >= 0 else "right", fontsize=9, fontweight="bold")
    ax.set_yticks(list(y)); ax.set_yticklabels(types, fontsize=10)
    ax.set_xlabel("Δ answer accuracy  (KG on − KG off, same 4 primary sections)")
    ax.set_title("KG contribution is multi-hop-specific, not uniform", fontsize=12, fontweight="bold")
    ax.set_xlim(-0.12, 0.22); ax.grid(axis="x", alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / "fig_kg_per_type.png", dpi=150); plt.close(fig)


# ---- Figure 1: architecture schematic ----
def architecture():
    fig, ax = plt.subplots(figsize=(11, 3.4))
    ax.set_xlim(0, 100); ax.set_ylim(0, 34); ax.axis("off")

    def box(x, y, w, h, text, fc="#ecf0f1", ec=DARK, fs=9, bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.2",
                                    fc=fc, ec=ec, lw=1.4))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
                fontweight="bold" if bold else "normal", color=DARK)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14,
                                     lw=1.3, color=DARK))

    box(1, 13, 11, 8, "Question", fc="#d6eaf8", bold=True)
    # two structural sources
    box(18, 20, 20, 8, "Hybrid\nBM25 + BGE-M3 → fuse", fc="#fdebd0")
    box(18, 5, 20, 8, "PageIndex\ntree navigation (LLM)", fc="#fdebd0")
    box(44, 12.5, 15, 9, "Union\npool", fc="#e8f8f5", bold=True)
    box(63, 12.5, 17, 9, "Rerank\n→ top-(k−1) primary\n+ 1 reserved KG slot", fc="#fadbd8", bold=True)
    box(85, 13, 13, 8, "Generator\n(fixed)", fc="#d5f5e3", bold=True)

    arrow(12, 17, 18, 24); arrow(12, 17, 18, 9)
    arrow(38, 24, 44, 18.5); arrow(38, 9, 44, 15.5)
    arrow(59, 17, 63, 17)
    arrow(80, 17, 85, 17)
    # KG annotation into the reserved slot
    ax.text(71.5, 24.2, "KG one-hop expansion\n(reserved slot)", ha="center", fontsize=8,
            color=RED, style="italic")
    ax.add_patch(FancyArrowPatch((71.5, 22.5), (71.5, 21.6), arrowstyle="-|>", mutation_scale=11,
                                 lw=1.1, color=RED))
    ax.set_title("Additive pipeline: structure augments hybrid; KG appended into a reserved slot",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(); fig.savefig(OUT / "fig_architecture.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    kg_per_type()
    architecture()
    print(f"✓ wrote fig_kg_per_type.png, fig_architecture.png -> {OUT}/")
