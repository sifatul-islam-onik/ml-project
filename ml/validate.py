# -*- coding: utf-8 -*-
"""
Cluster validation (methodology 5.6): stability, consensus and external checks.

Internal indices alone cannot tell you whether a partition is real. Three
independent lines of evidence are added here:

  * bootstrap stability - re-cluster resamples and measure agreement with the
    reference labelling by adjusted Rand index. Unstable clusters are an
    artifact of one sample draw.
  * consensus - how often each pair of students lands together across
    resamples, summarised per cluster as a co-assignment rate.
  * classes-to-clusters - agreement with each held-out background variable
    (year, gender, living arrangement, CGPA band). None of these entered the
    feature set, so above-chance agreement is genuine external validation and
    at-chance agreement is a real, reportable negative.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.metrics import (adjusted_mutual_info_score, adjusted_rand_score,
                             normalized_mutual_info_score, silhouette_samples)

from . import config as C


# --------------------------------------------------------------------------
# Stability
# --------------------------------------------------------------------------
def bootstrap_stability(Z, k, reference_labels, b=C.BOOTSTRAP_B,
                        random_state=C.RANDOM_STATE, sample_frac=0.80):
    """Cluster B subsamples; compare each to the reference on shared rows (ARI).

    Subsampling rather than resampling with replacement keeps duplicate points
    from inflating agreement artificially.
    """
    rng = np.random.default_rng(random_state)
    n = Z.shape[0]
    size = int(round(sample_frac * n))
    aris, amis = [], []

    for i in range(b):
        idx = rng.choice(n, size=size, replace=False)
        km = KMeans(n_clusters=k, n_init=10, random_state=int(rng.integers(1 << 31))).fit(Z[idx])
        aris.append(adjusted_rand_score(reference_labels[idx], km.labels_))
        amis.append(adjusted_mutual_info_score(reference_labels[idx], km.labels_))

    aris = np.asarray(aris)
    return {
        "n_resamples": int(b),
        "sample_fraction": sample_frac,
        "ari_mean": round(float(aris.mean()), 4),
        "ari_sd": round(float(aris.std()), 4),
        "ari_p05": round(float(np.percentile(aris, 5)), 4),
        "ari_median": round(float(np.median(aris)), 4),
        "ari_p95": round(float(np.percentile(aris, 95)), 4),
        "ami_mean": round(float(np.mean(amis)), 4),
        "pct_ari_above_0.5": round(float(100 * (aris > 0.5).mean()), 1),
        "pct_ari_above_0.75": round(float(100 * (aris > 0.75).mean()), 1),
    }


def seed_stability(Z, k, trials=C.SEED_TRIALS):
    """Re-run k-means from different seeds; report pairwise ARI across runs.

    Answers a narrower question than the bootstrap: given this exact sample, is
    the solution an artifact of initialisation?
    """
    labs = []
    for s in range(trials):
        labs.append(KMeans(n_clusters=k, n_init=10, random_state=s).fit_predict(Z))
    pair = [adjusted_rand_score(labs[i], labs[j])
            for i in range(trials) for j in range(i + 1, trials)]
    pair = np.asarray(pair)
    return {
        "n_seeds": int(trials),
        "pairwise_ari_mean": round(float(pair.mean()), 4),
        "pairwise_ari_min": round(float(pair.min()), 4),
        "pct_identical": round(float(100 * (pair > 0.99).mean()), 1),
    }


def consensus(Z, k, b=100, random_state=C.RANDOM_STATE, sample_frac=0.80):
    """Pairwise co-assignment rate across resamples.

    Returns the consensus matrix plus, for the reference labelling, the mean
    within-cluster co-assignment - a per-cluster reliability figure that says
    which of the profiles is trustworthy and which is a residual bucket.
    """
    rng = np.random.default_rng(random_state)
    n = Z.shape[0]
    size = int(round(sample_frac * n))
    together = np.zeros((n, n), dtype=np.float32)
    seen = np.zeros((n, n), dtype=np.float32)

    for _ in range(b):
        idx = rng.choice(n, size=size, replace=False)
        lab = KMeans(n_clusters=k, n_init=10,
                     random_state=int(rng.integers(1 << 31))).fit_predict(Z[idx])
        same = (lab[:, None] == lab[None, :]).astype(np.float32)
        together[np.ix_(idx, idx)] += same
        seen[np.ix_(idx, idx)] += 1.0

    with np.errstate(invalid="ignore", divide="ignore"):
        M = np.where(seen > 0, together / seen, np.nan)
    np.fill_diagonal(M, 1.0)
    return M


def consensus_by_cluster(M, labels):
    """Mean co-assignment within each reference cluster."""
    out = {}
    for c in sorted(set(labels)):
        idx = np.flatnonzero(labels == c)
        if len(idx) < 2:
            out[int(c)] = None
            continue
        block = M[np.ix_(idx, idx)]
        iu = np.triu_indices(len(idx), k=1)
        out[int(c)] = round(float(np.nanmean(block[iu])), 4)
    return out


# --------------------------------------------------------------------------
# External / classes-to-clusters
# --------------------------------------------------------------------------
def cramers_v(table):
    """Bias-corrected Cramer's V for a contingency table."""
    chi2 = stats.chi2_contingency(table)[0]
    n = table.to_numpy().sum()
    r, k = table.shape
    phi2 = chi2 / n
    phi2c = max(0.0, phi2 - (k - 1) * (r - 1) / (n - 1))
    rc = r - (r - 1) ** 2 / (n - 1)
    kc = k - (k - 1) ** 2 / (n - 1)
    denom = min(kc - 1, rc - 1)
    return float(np.sqrt(phi2c / denom)) if denom > 0 else float("nan")


def classes_to_clusters(labels, background, variables=("year", "gender", "living", "cgpa", "department")):
    """Test each held-out background variable against the cluster labelling.

    Reports chi-square, Cramer's V, ARI and NMI. The classes-to-clusters error
    (WEKA's own measure) is included: assign each cluster to its majority class
    and count how many students that misclassifies.
    """
    out = {}
    for var in variables:
        if var not in background.columns:
            continue
        y = background[var].astype(str)
        ct = pd.crosstab(pd.Series(labels, index=y.index, name="cluster"), y)
        chi2, p, dof, _ = stats.chi2_contingency(ct)
        majority = ct.idxmax(axis=1)
        correct = int(sum(ct.loc[c, majority[c]] for c in ct.index))
        codes = pd.Categorical(y).codes
        out[var] = {
            "chi2": round(float(chi2), 2),
            "dof": int(dof),
            "p": float(p),
            "cramers_v": round(cramers_v(ct), 4),
            "adjusted_rand": round(float(adjusted_rand_score(codes, labels)), 4),
            "nmi": round(float(normalized_mutual_info_score(codes, labels)), 4),
            "classes_to_clusters_accuracy": round(correct / len(y), 4),
            "majority_class_baseline": round(float(y.value_counts(normalize=True).max()), 4),
            "cluster_majority_class": {str(c): str(majority[c]) for c in ct.index},
        }
    return out


def silhouette_breakdown(Z, labels):
    """Per-cluster silhouette, plus the share of negatively scored members.

    A cluster whose members mostly score below zero is closer to another cluster
    than to its own and should not be presented as a coherent group.
    """
    s = silhouette_samples(Z, labels)
    out = {}
    for c in sorted(set(labels)):
        m = labels == c
        out[int(c)] = {
            "n": int(m.sum()),
            "mean_silhouette": round(float(s[m].mean()), 4),
            "pct_negative": round(float(100 * (s[m] < 0).mean()), 1),
        }
    return {"overall_mean": round(float(s.mean()), 4),
            "pct_negative_overall": round(float(100 * (s < 0).mean()), 1),
            "by_cluster": out}, s


def compare_algorithms(labelings):
    """Pairwise ARI between the labellings produced by different algorithms.

    High agreement means the partition is a property of the data; low agreement
    means it is a property of the algorithm, which is itself the finding.
    """
    names = list(labelings)
    M = pd.DataFrame(index=names, columns=names, dtype=float)
    for i, a in enumerate(names):
        for b in names:
            M.loc[a, b] = round(float(adjusted_rand_score(labelings[a], labelings[b])), 4)
    return M
