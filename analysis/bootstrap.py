"""Paired bootstrap 95% CIs (percentile, 10k resamples over items) — design §7.
Pure Python; deterministic via a fixed seed.
"""

from __future__ import annotations

import random
from statistics import mean

SEED = 20260729
N_BOOT = 10_000


def _resample_indices(n: int, rng: random.Random) -> list[int]:
    return [rng.randrange(n) for _ in range(n)]


def ci_mean(values: list[float], n_boot: int = N_BOOT, alpha: float = 0.05,
            seed: int = SEED) -> dict:
    """Bootstrap CI for a single arm's mean metric."""
    if not values:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    rng = random.Random(seed)
    n = len(values)
    boots = [mean(values[i] for i in _resample_indices(n, rng)) for _ in range(n_boot)]
    boots.sort()
    lo = boots[int((alpha / 2) * n_boot)]
    hi = boots[int((1 - alpha / 2) * n_boot)]
    return {"mean": mean(values), "lo": lo, "hi": hi, "n": n}


def ci_delta(a: list[float], b: list[float], n_boot: int = N_BOOT, alpha: float = 0.05,
             seed: int = SEED) -> dict:
    """Paired bootstrap CI for the mean difference a-b (same items, same resample)."""
    assert len(a) == len(b), "delta must be paired"
    if not a:
        return {"delta": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    rng = random.Random(seed)
    n = len(a)
    diffs = [ai - bi for ai, bi in zip(a, b)]
    boots = []
    for _ in range(n_boot):
        idx = _resample_indices(n, rng)
        boots.append(mean(diffs[i] for i in idx))
    boots.sort()
    lo = boots[int((alpha / 2) * n_boot)]
    hi = boots[int((1 - alpha / 2) * n_boot)]
    return {"delta": mean(diffs), "lo": lo, "hi": hi, "n": n}


def ci_interaction(a4: list[float], a3: list[float], a2: list[float], a1: list[float],
                   n_boot: int = N_BOOT, alpha: float = 0.05, seed: int = SEED) -> dict:
    """Difference-in-differences (A4-A3) - (A2-A1): does KG help MORE on PageIndex
    than on hybrid? Aggregate-then-bootstrap alternative to the GLMM interaction (§7)."""
    n = len(a1)
    assert len(a2) == len(a3) == len(a4) == n, "interaction needs all four arms paired on items"
    if not n:
        return {"did": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    rng = random.Random(seed)
    per = [(a4[i] - a3[i]) - (a2[i] - a1[i]) for i in range(n)]
    boots = []
    for _ in range(n_boot):
        idx = _resample_indices(n, rng)
        boots.append(mean(per[i] for i in idx))
    boots.sort()
    return {"did": mean(per), "lo": boots[int((alpha / 2) * n_boot)],
            "hi": boots[int((1 - alpha / 2) * n_boot)], "n": n}


if __name__ == "__main__":
    a = [1, 1, 1, 0, 1, 1, 0, 1]
    b = [0, 1, 0, 0, 1, 0, 0, 1]
    d = ci_delta(a, b)
    assert d["lo"] <= d["delta"] <= d["hi"], d
    assert abs(d["delta"] - 0.375) < 1e-9, d
    print("bootstrap self-test ok:", d)
