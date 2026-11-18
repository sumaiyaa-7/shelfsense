"""Statistical tools for evaluating AI systems.

Eval sets are small, so a single accuracy number hides a lot of uncertainty.
We report bootstrap confidence intervals and use a paired permutation test to
decide whether one configuration really beats another on the same questions.
"""
from __future__ import annotations

import numpy as np


def bootstrap_ci(values, n_boot: int = 10_000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float, float]:
    """Mean with a percentile bootstrap (1 - alpha) confidence interval."""
    x = np.asarray(values, dtype=float)
    if x.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n_boot, x.size), replace=True).mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(x.mean()), float(lo), float(hi)


def paired_permutation_test(a, b, n_perm: int = 10_000, seed: int = 0) -> float:
    """Two-sided p-value for H0: mean(a - b) = 0, for paired per-question scores.

    Under H0 the sign of each paired difference is exchangeable, so we flip
    signs at random and count how often the permuted mean is as extreme as
    the observed one.
    """
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    observed = abs(d.mean())
    if observed == 0:
        return 1.0
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(n_perm, d.size))
    perm = np.abs((signs * d).mean(axis=1))
    return float((np.sum(perm >= observed) + 1) / (n_perm + 1))


def percentile(values, q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=float), q)) if len(values) else float("nan")
