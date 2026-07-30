"""Pairwise significance + multiple-comparison correction (design §7). Pure Python.

  mcnemar_exact  : binary paired hit@k contrasts (exact two-sided binomial on discordants)
  permutation    : paired sign-flip test for a mean difference (rate or binary)
  holm_bonferroni: adjust a family of p-values, control FWER
"""

from __future__ import annotations

import math
import random

SEED = 20260729


def mcnemar_exact(a: list[float], b: list[float]) -> dict:
    """a,b are paired binary (0/1). Tests marginal-homogeneity of the two arms.
    Reports discordant counts, exact two-sided p, and the paired odds ratio c/b."""
    b01 = sum(1 for x, y in zip(a, b) if x == 1 and y == 0)  # a wins
    c01 = sum(1 for x, y in zip(a, b) if x == 0 and y == 1)  # b wins
    n = b01 + c01
    if n == 0:
        return {"a_wins": 0, "b_wins": 0, "p": 1.0, "odds_ratio": float("nan")}
    k = min(b01, c01)
    tail = sum(math.comb(n, i) for i in range(k + 1)) * (0.5 ** n)
    p = min(1.0, 2 * tail)
    orat = (b01 / c01) if c01 else float("inf")
    return {"a_wins": b01, "b_wins": c01, "p": p, "odds_ratio": orat}


def permutation(a: list[float], b: list[float], n_perm: int = 10_000,
                seed: int = SEED) -> dict:
    """Paired sign-flip permutation test on mean(a-b). Exact if 2^n <= n_perm."""
    diffs = [x - y for x, y in zip(a, b)]
    n = len(diffs)
    if n == 0:
        return {"delta": float("nan"), "p": float("nan")}
    obs = sum(diffs) / n
    nz = [d for d in diffs if d != 0]
    m = len(nz)
    count = 0
    total = 0
    if m <= 20 and (1 << m) <= n_perm:  # exact enumeration over sign flips
        for mask in range(1 << m):
            s = sum(nz[i] if (mask >> i) & 1 else -nz[i] for i in range(m))
            if abs(s / n) >= abs(obs) - 1e-12:
                count += 1
            total += 1
    else:
        rng = random.Random(seed)
        for _ in range(n_perm):
            s = sum(d if rng.random() < 0.5 else -d for d in nz)
            if abs(s / n) >= abs(obs) - 1e-12:
                count += 1
            total += 1
    return {"delta": obs, "p": count / total if total else float("nan")}


def holm_bonferroni(pvals: dict[str, float], alpha: float = 0.05) -> dict[str, dict]:
    """pvals: {label: p}. Returns {label: {p, p_adj, reject}} with FWER control."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out, running = {}, 0.0
    for rank, (label, p) in enumerate(items):
        adj = min(1.0, (m - rank) * p)
        running = max(running, adj)  # enforce monotone non-decreasing adjusted p
        out[label] = {"p": p, "p_adj": running, "reject": running <= alpha}
    return out


if __name__ == "__main__":
    a = [1, 1, 1, 0, 1, 1, 0, 1, 1, 1]
    b = [0, 0, 1, 0, 1, 0, 0, 1, 0, 0]
    mc = mcnemar_exact(a, b)
    assert mc["a_wins"] == 5 and mc["b_wins"] == 0, mc
    pm = permutation(a, b)
    holm = holm_bonferroni({"x": 0.01, "y": 0.04, "z": 0.2})
    assert holm["x"]["reject"] and not holm["z"]["reject"], holm
    print("significance self-test ok:", mc["p"], pm["p"], holm["y"])
