# -*- coding: utf-8 -*-
"""
Dimensionality reduction (methodology report section 5.4).

PCA on the standardised item matrix, reported three ways so the reduction is
justified rather than assumed:
  * the full eigenvalue / explained-variance table with the Kaiser count and
    the number of components needed to reach the 95% variance target;
  * a parallel analysis against random data, which is a stricter and more
    honest component-retention rule than Kaiser on a 12-item instrument;
  * varimax-rotated loadings, because unrotated components on a low-alpha
    instrument are usually a size factor plus uninterpretable contrasts.

t-SNE is included only as a second projection for the cluster-overlap figure;
nothing is ever clustered in t-SNE space, since its distances are not metric.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from . import config as C


def varimax(loadings, gamma=1.0, max_iter=100, tol=1e-6):
    """Kaiser-normalised varimax rotation of a loading matrix."""
    L = np.asarray(loadings, dtype=float).copy()
    p, k = L.shape
    if k < 2:
        return L
    R = np.eye(k)
    d = 0.0
    for _ in range(max_iter):
        d_old = d
        Lam = L @ R
        u, s, vh = np.linalg.svd(
            L.T @ (Lam ** 3 - (gamma / p) * Lam @ np.diag(np.diag(Lam.T @ Lam)))
        )
        R = u @ vh
        d = float(s.sum())
        if d_old != 0 and abs(d - d_old) / d < tol:
            break
    return L @ R


def parallel_analysis(Z, n_iter=500, percentile=95, random_state=C.RANDOM_STATE):
    """Horn's parallel analysis: keep components beating random-data eigenvalues.

    Kaiser's eigenvalue>1 rule over-retains on short instruments. Comparing each
    observed eigenvalue against the 95th percentile of eigenvalues from
    same-shaped random normal data gives a defensible retention count.
    """
    rng = np.random.default_rng(random_state)
    n, p = Z.shape
    sim = np.empty((n_iter, p))
    for i in range(n_iter):
        X = rng.standard_normal((n, p))
        X = (X - X.mean(0)) / X.std(0, ddof=1)
        sim[i] = np.linalg.eigvalsh(np.corrcoef(X, rowvar=False))[::-1]
    threshold = np.percentile(sim, percentile, axis=0)
    observed = np.linalg.eigvalsh(np.corrcoef(Z, rowvar=False))[::-1]
    keep = int((observed > threshold).sum())
    return {
        "observed_eigenvalues": [round(float(v), 3) for v in observed],
        "random_p%d" % percentile: [round(float(v), 3) for v in threshold],
        "n_retain": keep,
    }


def run_pca(Z, item_names=None, var_target=C.PCA_VAR_TARGET):
    """Fit PCA on the standardised matrix and return (scores, model, report)."""
    item_names = list(item_names or C.ITEMS)
    full = PCA(random_state=C.RANDOM_STATE).fit(Z)
    eig = full.explained_variance_
    ratio = full.explained_variance_ratio_
    cum = np.cumsum(ratio)

    n_kaiser = int((eig > 1).sum())
    n_95 = int(np.searchsorted(cum, var_target) + 1)
    par = parallel_analysis(Z)

    scores = full.transform(Z)
    n_rot = max(2, par["n_retain"])
    raw_load = full.components_[:n_rot].T * np.sqrt(eig[:n_rot])
    rot_load = varimax(raw_load)

    load_df = pd.DataFrame(raw_load, index=item_names,
                           columns=["PC%d" % (i + 1) for i in range(n_rot)])
    rot_df = pd.DataFrame(rot_load, index=item_names,
                          columns=["RC%d" % (i + 1) for i in range(n_rot)])

    report = {
        "n_components_total": int(len(eig)),
        "eigenvalues": [round(float(v), 4) for v in eig],
        "explained_variance_pct": [round(float(v) * 100, 2) for v in ratio],
        "cumulative_variance_pct": [round(float(v) * 100, 2) for v in cum],
        "n_kaiser": n_kaiser,
        "n_for_%d_pct" % int(var_target * 100): n_95,
        "variance_at_2_components_pct": round(float(cum[1] * 100), 2),
        "parallel_analysis": par,
        "n_retained": n_rot,
        "loadings_unrotated": {c: {i: round(float(load_df.loc[i, c]), 3) for i in item_names}
                               for c in load_df.columns},
        "loadings_varimax": {c: {i: round(float(rot_df.loc[i, c]), 3) for i in item_names}
                             for c in rot_df.columns},
        "rotated_variance_pct": [round(float((rot_load[:, j] ** 2).sum() / len(item_names) * 100), 2)
                                 for j in range(n_rot)],
    }
    return scores, full, report, load_df, rot_df


def interpret_components(rot_df, threshold=0.40):
    """Name each rotated component by the items that load on it."""
    out = {}
    for c in rot_df.columns:
        s = rot_df[c]
        pos = [(i, round(float(v), 2)) for i, v in s.items() if v >= threshold]
        neg = [(i, round(float(v), 2)) for i, v in s.items() if v <= -threshold]
        pos.sort(key=lambda t: -t[1])
        neg.sort(key=lambda t: t[1])
        out[c] = {"positive": pos, "negative": neg,
                  "n_salient": len(pos) + len(neg)}
    return out


def tsne_projection(Z, random_state=C.RANDOM_STATE, perplexity=30):
    """2-D t-SNE embedding, used for visual overlap inspection only."""
    ts = TSNE(n_components=2, random_state=random_state, perplexity=perplexity,
              init="pca", max_iter=1000)
    return ts.fit_transform(Z)
