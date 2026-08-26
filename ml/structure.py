# -*- coding: utf-8 -*-
"""
Measurement structure of the questionnaire.

This runs *before* any clustering on purpose. If the twelve items do not hang
together, a single stress score is not a valid target and the clustering result
has to be read as "profiles of response patterns", not "levels of one latent
stress trait". Establishing that first is what keeps the interpretation honest.

Provides: Cronbach's alpha (with alpha-if-item-deleted), corrected item-total
correlations, the inter-item correlation matrix, and the two standard
factorability tests (Bartlett's sphericity, Kaiser-Meyer-Olkin).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from . import config as C


def cronbach_alpha(frame):
    """Standard raw-score Cronbach's alpha."""
    k = frame.shape[1]
    if k < 2:
        return float("nan")
    item_var = frame.var(axis=0, ddof=1).sum()
    total_var = frame.sum(axis=1).var(ddof=1)
    return float((k / (k - 1)) * (1 - item_var / total_var))


def corrected_item_total(frame):
    """Correlation of each item with the sum of the *other* items."""
    total = frame.sum(axis=1)
    return {c: float(np.corrcoef(frame[c], total - frame[c])[0, 1]) for c in frame.columns}


def alpha_if_deleted(frame):
    """Alpha recomputed with each item dropped in turn."""
    out = {}
    for c in frame.columns:
        out[c] = round(cronbach_alpha(frame.drop(columns=[c])), 3)
    return out


def bartlett_sphericity(frame):
    """Bartlett's test that the correlation matrix is not an identity matrix.

    A non-significant result would mean the items are mutually uncorrelated and
    no factor/PCA solution is meaningful.
    """
    R = np.corrcoef(frame.to_numpy(), rowvar=False)
    n, p = frame.shape
    det = np.linalg.det(R)
    det = max(det, 1e-12)
    chi2 = -(n - 1 - (2 * p + 5) / 6) * np.log(det)
    dof = p * (p - 1) / 2
    return {"chi2": round(float(chi2), 2), "dof": int(dof),
            "p": float(stats.chi2.sf(chi2, dof)), "determinant": float(det)}


def kmo(frame):
    """Kaiser-Meyer-Olkin sampling adequacy, overall and per item.

    KMO compares correlations against partial correlations. Values below ~0.60
    say the items share too little common variance for a factor solution to be
    trustworthy - a number worth reporting either way.
    """
    R = np.corrcoef(frame.to_numpy(), rowvar=False)
    inv = np.linalg.pinv(R)
    d = np.sqrt(np.diag(inv))
    partial = -inv / np.outer(d, d)
    np.fill_diagonal(partial, 0.0)
    Rc = R.copy()
    np.fill_diagonal(Rc, 0.0)

    r2 = (Rc ** 2).sum()
    p2 = (partial ** 2).sum()
    overall = r2 / (r2 + p2)

    per = {}
    for i, c in enumerate(frame.columns):
        ri = (Rc[i] ** 2).sum()
        pi = (partial[i] ** 2).sum()
        per[c] = round(float(ri / (ri + pi)), 3)
    return {"overall": round(float(overall), 3), "per_item": per}


def analyse(items):
    """Full measurement-structure report over the direction-aligned items."""
    frame = items[C.ITEMS].astype(float)
    R = frame.corr()
    offdiag = R.to_numpy()[~np.eye(len(C.ITEMS), dtype=bool)]

    it_total = corrected_item_total(frame)
    weak = sorted([k for k, v in it_total.items() if v < 0.20], key=lambda k: it_total[k])
    core = [c for c in C.ITEMS if c not in weak]

    pairs = []
    for i, a in enumerate(C.ITEMS):
        for b in C.ITEMS[i + 1:]:
            pairs.append({"a": a, "b": b, "r": round(float(R.loc[a, b]), 3)})
    pairs.sort(key=lambda d: -abs(d["r"]))

    return {
        "cronbach_alpha_12": round(cronbach_alpha(frame), 3),
        "cronbach_alpha_core": round(cronbach_alpha(frame[core]), 3),
        "core_items": core,
        "weak_items": weak,
        "alpha_if_deleted": alpha_if_deleted(frame),
        "corrected_item_total": {k: round(v, 3) for k, v in it_total.items()},
        "mean_abs_interitem_r": round(float(np.abs(offdiag).mean()), 3),
        "mean_interitem_r": round(float(offdiag.mean()), 3),
        "max_interitem_r": round(float(offdiag.max()), 3),
        "min_interitem_r": round(float(offdiag.min()), 3),
        "top_pairs": pairs[:8],
        "bartlett": bartlett_sphericity(frame),
        "kmo": kmo(frame),
        "correlation_matrix": {a: {b: round(float(R.loc[a, b]), 3) for b in C.ITEMS} for a in C.ITEMS},
    }


def verdict(struct):
    """One-sentence, threshold-based reading of the structure numbers.

    Written as an explicit rule rather than prose so the conclusion cannot drift
    from the numbers it is based on.
    """
    a = struct["cronbach_alpha_12"]
    k = struct["kmo"]["overall"]
    r = struct["mean_abs_interitem_r"]
    if a >= 0.70 and k >= 0.70:
        return ("The twelve items behave as one reasonably reliable scale "
                "(alpha=%.2f, KMO=%.2f); a composite stress score is defensible." % (a, k))
    return ("The twelve items do not form a single reliable scale "
            "(alpha=%.2f, KMO=%.2f, mean |inter-item r|=%.2f). They are a checklist of "
            "partly independent stressors, so the analysis reports profiles across items "
            "rather than one stress level, and no composite score is used as a target." % (a, k, r))
