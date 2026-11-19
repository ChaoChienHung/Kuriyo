"""Statistics for comparing runs (README "實驗設計與統計檢定"); standard library and NumPy only.

Student t probabilities use the regularized incomplete beta function, so no SciPy is needed.
"""

import math

import numpy as np


def _betacf(a, b, x):
    """Continued fraction for the incomplete beta function (modified Lentz)."""
    tiny = 1e-300
    c, d = 1.0, 1.0 - (a + b) * x / (a + 1.0)
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        for numerator in (m * (b - m) * x / ((a + m2 - 1.0) * (a + m2)),
                          -(a + m) * (a + b + m) * x / ((a + m2) * (a + m2 + 1.0))):
            d = 1.0 + numerator * d
            d = 1.0 / (d if abs(d) > tiny else tiny)
            c = 1.0 + numerator / c
            c = c if abs(c) > tiny else tiny
            h *= d * c
        if abs(d * c - 1.0) < 1e-15:
            break
    return h


def betainc(a, b, x):
    """Regularized incomplete beta I_x(a, b)."""
    if not 0.0 <= x <= 1.0:
        raise ValueError("x must be in [0, 1]")
    if x in (0.0, 1.0):
        return x
    front = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                     + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def t_two_sided_p(t, df):
    """Two-sided p-value of a Student t statistic."""
    if df <= 0:
        raise ValueError("degrees of freedom must be positive")
    if math.isinf(t):
        return 0.0
    return betainc(df / 2.0, 0.5, df / (df + t * t))


def t_quantile(probability, df):
    """Upper quantile: t such that P(T <= t) = probability (probability >= 0.5)."""
    if not 0.5 <= probability < 1.0:
        raise ValueError("probability must be in [0.5, 1)")
    low, high = 0.0, 1.0
    while 1.0 - t_two_sided_p(high, df) / 2.0 < probability:
        high *= 2.0
    for _ in range(200):
        middle = (low + high) / 2.0
        if 1.0 - t_two_sided_p(middle, df) / 2.0 < probability:
            low = middle
        else:
            high = middle
    return (low + high) / 2.0


def summary(values, confidence=0.95):
    """Mean, sample standard deviation, and t confidence interval of seed-level values."""
    values = [float(value) for value in values]
    n = len(values)
    if not n:
        raise ValueError("no values")
    mean = sum(values) / n
    if n == 1:
        return {"n": 1, "mean": mean, "sd": None, "ci": None}
    sd = math.sqrt(sum((value - mean) ** 2 for value in values) / (n - 1))
    half = t_quantile(0.5 + confidence / 2.0, n - 1) * sd / math.sqrt(n)
    return {"n": n, "mean": mean, "sd": sd, "ci": [mean - half, mean + half]}


def welch_t(a, b):
    """Welch's two-sample t-test (two-sided); None when either side has fewer than 2 values."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if len(a) < 2 or len(b) < 2:
        return None
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    difference = float(a.mean() - b.mean())
    if va + vb == 0:
        return {"t": math.inf if difference else 0.0, "df": None, "p": 0.0 if difference else 1.0}
    t = difference / math.sqrt(va + vb)
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    return {"t": t, "df": float(df), "p": t_two_sided_p(t, df)}


def paired_t(differences):
    """Paired t-test on per-user differences (two-sided)."""
    d = np.asarray(differences, dtype=float)
    if len(d) < 2:
        raise ValueError("need at least two pairs")
    mean, sd = float(d.mean()), float(d.std(ddof=1))
    if sd == 0:
        return {"t": math.inf if mean else 0.0, "df": len(d) - 1, "p": 0.0 if mean else 1.0}
    t = mean / (sd / math.sqrt(len(d)))
    return {"t": t, "df": len(d) - 1, "p": t_two_sided_p(t, len(d) - 1)}


def paired_bootstrap(differences, resamples=10000, seed=0, chunk=250):
    """Bootstrap over users: 95% percentile CI of the mean difference and a two-sided p-value
    from the null distribution (differences centred on zero)."""
    d = np.asarray(differences, dtype=float)
    if len(d) < 2:
        raise ValueError("need at least two pairs")
    observed = float(d.mean())
    centred = d - observed
    rng = np.random.default_rng(seed)
    means = np.empty(resamples)
    null = np.empty(resamples)
    for start in range(0, resamples, chunk):
        size = min(chunk, resamples - start)
        index = rng.integers(0, len(d), size=(size, len(d)))
        means[start:start + size] = d[index].mean(axis=1)
        null[start:start + size] = centred[index].mean(axis=1)
    extreme = int(np.sum(np.abs(null) >= abs(observed) - 1e-15))
    return {"mean": observed, "ci": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))],
            "p": (extreme + 1) / (resamples + 1), "resamples": resamples, "seed": seed}


def holm(p_values):
    """Holm–Bonferroni adjusted p-values, in input order (None stays None)."""
    indexed = sorted((p, i) for i, p in enumerate(p_values) if p is not None)
    adjusted = [None] * len(p_values)
    running = 0.0
    m = len(indexed)
    for order, (p, i) in enumerate(indexed):
        running = max(running, min(1.0, (m - order) * p))
        adjusted[i] = running
    return adjusted
