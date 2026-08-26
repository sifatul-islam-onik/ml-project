# -*- coding: utf-8 -*-
"""
Candidate clustering models and the k-selection evidence (methodology 5.5-5.6).

Cluster count is decided from several independent signals rather than one:
  * within-cluster SSE across k = 2..8, read as an elbow (with the Kneedle
    knee-point computed numerically so "the elbow" is not eyeballed);
  * Silhouette, Davies-Bouldin and Calinski-Harabasz internal indices;
  * Gaussian-mixture BIC and cross-validated log-likelihood, which is the
    scikit-learn equivalent of WEKA's EM with -N -1 (automatic k);
  * Ward dendrogram structure.

Four algorithm families are fitted so the choice of algorithm is also evidenced:
k-means (primary), Gaussian mixture, Ward agglomerative, and a Gower-distance
k-medoids run over the mixed item+background matrix. The Gower branch is a
dependency-free implementation, so nothing here needs a package Kaggle does not
already ship.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import cophenet, dendrogram, fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering
from sklearn.metrics import (calinski_harabasz_score, davies_bouldin_score,
                             silhouette_samples, silhouette_score)
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import KFold

from . import config as C


# --------------------------------------------------------------------------
# Internal validity indices
# --------------------------------------------------------------------------
def internal_indices(Z, labels):
    """Silhouette / Davies-Bouldin / Calinski-Harabasz for one labelling."""
    uniq = len(set(labels))
    if uniq < 2 or uniq >= len(labels):
        return {"silhouette": None, "davies_bouldin": None, "calinski_harabasz": None}
    return {
        "silhouette": round(float(silhouette_score(Z, labels)), 4),
        "davies_bouldin": round(float(davies_bouldin_score(Z, labels)), 4),
        "calinski_harabasz": round(float(calinski_harabasz_score(Z, labels)), 2),
    }


def gap_statistic(Z, ks=None, n_refs=25, random_state=C.RANDOM_STATE):
    """Tibshirani's gap statistic, evaluated from k = 1 upward.

    This is the only criterion here that can answer "is there any structure at
    all", because it includes k = 1 as a candidate. Silhouette, Davies-Bouldin
    and Calinski-Harabasz are all undefined at k = 1 and therefore *cannot*
    report the absence of clusters - they only ever rank the partitions you ask
    for. Reference data is drawn uniformly over the bounding box of the
    principal components, which is the standard null of "one homogeneous blob".

    The selection rule is Tibshirani's: the smallest k whose gap is within one
    standard error of the next k's gap.
    """
    ks = list(ks or ([1] + list(C.K_RANGE)))
    rng = np.random.default_rng(random_state)

    def dispersion(X, k):
        if k == 1:
            centre = X.mean(axis=0, keepdims=True)
            return float(((X - centre) ** 2).sum())
        return float(KMeans(n_clusters=k, n_init=10,
                            random_state=random_state).fit(X).inertia_)

    # Reference box aligned to the principal axes of the observed data.
    from sklearn.decomposition import PCA
    p = PCA().fit(Z)
    Zr = Z @ p.components_.T
    lo, hi = Zr.min(axis=0), Zr.max(axis=0)

    obs_log, ref_log_mean, ref_log_sd = [], [], []
    for k in ks:
        obs_log.append(np.log(dispersion(Z, k)))
        refs = []
        for _ in range(n_refs):
            U = rng.uniform(lo, hi, size=Z.shape)
            refs.append(np.log(dispersion(U @ p.components_, k)))
        ref_log_mean.append(float(np.mean(refs)))
        ref_log_sd.append(float(np.std(refs)))

    gap = np.array(ref_log_mean) - np.array(obs_log)
    sk = np.array(ref_log_sd) * np.sqrt(1 + 1.0 / n_refs)

    chosen = ks[-1]
    for i in range(len(ks) - 1):
        if gap[i] >= gap[i + 1] - sk[i + 1]:
            chosen = ks[i]
            break

    return {
        "ks": ks,
        "gap": [round(float(g), 4) for g in gap],
        "s_k": [round(float(s), 4) for s in sk],
        "k_selected": int(chosen),
        "n_references": int(n_refs),
        "supports_no_structure": bool(chosen == 1),
        "interpretation": (
            "the gap statistic selects k = 1, i.e. the data is better described as one "
            "homogeneous group than as any partition tested"
            if chosen == 1 else
            "the gap statistic selects k = %d over the single-group null" % chosen),
    }


def knee_point(ks, sse):
    """Kneedle: the k whose SSE is furthest below the first-to-last chord.

    Turns "look for the elbow" into a reproducible number, which matters because
    on weakly clustered data different readers pick different elbows by eye.
    """
    x = np.asarray(ks, dtype=float)
    y = np.asarray(sse, dtype=float)
    xn = (x - x.min()) / (x.max() - x.min())
    yn = (y - y.min()) / (y.max() - y.min())
    # Perpendicular distance below the straight line joining the endpoints.
    dist = (yn[0] + (yn[-1] - yn[0]) * xn) - yn
    return int(x[int(np.argmax(dist))]), [round(float(d), 4) for d in dist]


# --------------------------------------------------------------------------
# k-means sweep
# --------------------------------------------------------------------------
def kmeans_sweep(Z, ks=C.K_RANGE, n_init=C.N_INIT, random_state=C.RANDOM_STATE):
    """Fit k-means for each k and collect SSE plus all three internal indices."""
    rows, models = [], {}
    for k in ks:
        km = KMeans(n_clusters=k, n_init=n_init, random_state=random_state).fit(Z)
        models[k] = km
        rows.append({"k": k, "sse": round(float(km.inertia_), 2),
                     **internal_indices(Z, km.labels_)})
    table = pd.DataFrame(rows).set_index("k")
    knee, dists = knee_point(list(table.index), table["sse"].to_numpy())
    report = {
        "table": table.reset_index().to_dict("records"),
        "sse_knee_k": knee,
        "knee_distances": dists,
        "best_silhouette_k": int(table["silhouette"].idxmax()),
        "best_davies_bouldin_k": int(table["davies_bouldin"].idxmin()),
        "best_calinski_k": int(table["calinski_harabasz"].idxmax()),
    }
    return models, table, report


# --------------------------------------------------------------------------
# Gaussian mixture (WEKA EM equivalent, including automatic k)
# --------------------------------------------------------------------------
def gmm_sweep(Z, ks=C.K_RANGE, random_state=C.RANDOM_STATE, cv_folds=C.CV_FOLDS):
    """Fit Gaussian mixtures and select k by BIC and by cross-validated log-likelihood.

    WEKA's EM picks k by ten-fold cross-validated log-likelihood; the same rule
    is reproduced here so the two toolchains can be compared directly, with BIC
    reported alongside as the standard penalised-likelihood criterion.
    """
    rows, models = [], {}
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    for k in ks:
        gm = GaussianMixture(n_components=k, covariance_type="full",
                             n_init=5, random_state=random_state).fit(Z)
        models[k] = gm
        folds = []
        for tr, te in kf.split(Z):
            g = GaussianMixture(n_components=k, covariance_type="full",
                                n_init=3, random_state=random_state).fit(Z[tr])
            folds.append(g.score(Z[te]))
        labels = gm.predict(Z)
        rows.append({
            "k": k,
            "bic": round(float(gm.bic(Z)), 1),
            "aic": round(float(gm.aic(Z)), 1),
            "cv_loglik": round(float(np.mean(folds)), 4),
            "cv_loglik_sd": round(float(np.std(folds)), 4),
            **internal_indices(Z, labels),
        })
    table = pd.DataFrame(rows).set_index("k")
    report = {
        "table": table.reset_index().to_dict("records"),
        "best_bic_k": int(table["bic"].idxmin()),
        "best_cv_loglik_k": int(table["cv_loglik"].idxmax()),
    }
    return models, table, report


# --------------------------------------------------------------------------
# Hierarchical
# --------------------------------------------------------------------------
def hierarchical(Z, ks=C.K_RANGE, methods=("ward", "average", "complete")):
    """Linkage matrices, cophenetic correlation, and cut-based labellings.

    Cophenetic correlation says how faithfully each linkage preserves the
    original distances; it is the standard way to justify choosing Ward rather
    than assuming it.
    """
    from scipy.spatial.distance import pdist

    D = pdist(Z, metric="euclidean")
    out, links, labels_by = {}, {}, {}
    for m in methods:
        Lk = linkage(Z, method=m) if m == "ward" else linkage(D, method=m)
        links[m] = Lk
        coph, _ = cophenet(Lk, D)
        rows = []
        labels_by[m] = {}
        for k in ks:
            lab = fcluster(Lk, t=k, criterion="maxclust") - 1
            labels_by[m][k] = lab
            sizes = np.bincount(lab, minlength=k)
            rows.append({"k": k,
                         "largest_cluster_pct": round(float(100 * sizes.max() / len(lab)), 1),
                         "n_singleton_clusters": int((sizes <= 2).sum()),
                         **internal_indices(Z, lab)})
        out[m] = {"cophenetic_r": round(float(coph), 4),
                  "table": rows,
                  "merge_heights_last10": [round(float(h), 3) for h in Lk[-10:, 2]]}

    best_coph = max(out, key=lambda m: out[m]["cophenetic_r"])

    # A linkage that puts almost everyone in one cluster is degenerate, however
    # well it preserves pairwise distances. Average linkage typically wins the
    # cophenetic comparison on continuum-like data precisely because it chains:
    # it hangs a few outliers off one enormous cluster, which reproduces the
    # distance matrix faithfully and segments nothing. So the reported cut uses
    # the best-correlating linkage that also produces a usable split.
    def degenerate(m):
        return any(r["largest_cluster_pct"] > 90 for r in out[m]["table"])

    usable = [m for m in out if not degenerate(m)]
    best_usable = (max(usable, key=lambda m: out[m]["cophenetic_r"])
                   if usable else best_coph)

    return links, labels_by, {
        "by_method": out,
        "best_cophenetic_method": best_coph,
        "degenerate_methods": [m for m in out if degenerate(m)],
        "reported_method": best_usable,
        "note": ("%s has the highest cophenetic correlation (%.3f) but is degenerate - it "
                 "leaves over 90%% of students in a single cluster at every k tested. The "
                 "reported hierarchical cut therefore uses %s."
                 % (best_coph, out[best_coph]["cophenetic_r"], best_usable)
                 if best_coph != best_usable else
                 "%s has the highest cophenetic correlation (%.3f) and produces a usable split."
                 % (best_coph, out[best_coph]["cophenetic_r"])),
    }


def agglomerative_labels(Z, k, linkage_method="ward"):
    """Direct sklearn agglomerative fit, kept for the final chosen k."""
    ac = AgglomerativeClustering(n_clusters=k, linkage=linkage_method)
    return ac.fit_predict(Z), ac


# --------------------------------------------------------------------------
# Mixed-data branch: Gower distance + k-medoids (PAM)
# --------------------------------------------------------------------------
def gower_matrix(df, numeric_cols, categorical_cols):
    """Gower dissimilarity for a mixed numeric / categorical table.

    Numeric contributions are absolute differences scaled by the column range;
    categorical contributions are 0/1 mismatches. Implemented directly because
    the `gower` package is not part of the Kaggle base image, and n is under a
    thousand so the full n-by-n matrix is cheap.
    """
    n = len(df)
    # float32 and in-place accumulation: the full matrix is O(n^2), so a cohort
    # of a few thousand students would otherwise allocate hundreds of MB in
    # temporaries alone. `buf` is reused for every column.
    total = np.zeros((n, n), dtype=np.float32)
    buf = np.empty((n, n), dtype=np.float32)
    weight = 0.0

    for c in numeric_cols:
        v = df[c].to_numpy(dtype=np.float32)
        rng = float(v.max() - v.min())
        if rng <= 0:
            continue
        np.subtract(v[:, None], v[None, :], out=buf)
        np.abs(buf, out=buf)
        buf /= rng
        total += buf
        weight += 1.0

    for c in categorical_cols:
        # Factorised codes compare as integers, which lets the mismatch indicator
        # be written straight into the float buffer with no string temporaries.
        codes = pd.factorize(df[c].astype(str))[0].astype(np.float32)
        np.subtract(codes[:, None], codes[None, :], out=buf)
        np.sign(buf, out=buf)
        np.abs(buf, out=buf)
        total += buf
        weight += 1.0

    if weight == 0:
        raise ValueError("Gower matrix needs at least one usable column.")
    total /= weight
    np.fill_diagonal(total, 0.0)
    return total


def kmedoids(D, k, random_state=C.RANDOM_STATE, max_iter=300, n_init=10):
    """Voronoi-iteration PAM over a precomputed distance matrix."""
    rng = np.random.default_rng(random_state)
    n = D.shape[0]
    best_labels, best_cost, best_medoids = None, np.inf, None

    for _ in range(n_init):
        # k-means++ style seeding on the distance matrix.
        medoids = [int(rng.integers(n))]
        while len(medoids) < k:
            d = D[:, medoids].min(axis=1) ** 2
            s = d.sum()
            medoids.append(int(rng.choice(n, p=d / s)) if s > 0 else int(rng.integers(n)))
        medoids = np.array(medoids)

        for _ in range(max_iter):
            labels = np.argmin(D[:, medoids], axis=1)
            new = medoids.copy()
            for j in range(k):
                members = np.flatnonzero(labels == j)
                if len(members):
                    new[j] = members[int(np.argmin(D[np.ix_(members, members)].sum(axis=1)))]
            if np.array_equal(new, medoids):
                break
            medoids = new

        labels = np.argmin(D[:, medoids], axis=1)
        cost = float(D[np.arange(n), medoids[labels]].sum())
        if cost < best_cost:
            best_labels, best_cost, best_medoids = labels, cost, medoids

    return best_labels, best_medoids, best_cost


def gower_sweep(D, ks=C.K_RANGE, random_state=C.RANDOM_STATE):
    """k-medoids over Gower distance for each k, scored with precomputed silhouette."""
    rows, labels_by = [], {}
    for k in ks:
        lab, med, cost = kmedoids(D, k, random_state=random_state)
        labels_by[k] = lab
        sil = (float(silhouette_score(D, lab, metric="precomputed"))
               if len(set(lab)) > 1 else None)
        rows.append({"k": k, "cost": round(cost, 2),
                     "silhouette_gower": None if sil is None else round(sil, 4),
                     "n_medoids": len(set(med))})
    table = pd.DataFrame(rows).set_index("k")
    return labels_by, table, {"table": table.reset_index().to_dict("records"),
                              "best_silhouette_k": int(table["silhouette_gower"].idxmax())}


# --------------------------------------------------------------------------
# Spectral cross-check
# --------------------------------------------------------------------------
def spectral_labels(Z, k, random_state=C.RANDOM_STATE):
    """Spectral clustering, which unlike k-means does not assume convex clusters."""
    sc = SpectralClustering(n_clusters=k, random_state=random_state,
                            affinity="nearest_neighbors", n_neighbors=15,
                            assign_labels="kmeans")
    return sc.fit_predict(Z)


# --------------------------------------------------------------------------
# Selection
# --------------------------------------------------------------------------
def select_k(km_report, gmm_report, hier_report, gap_report=None, km_table=None):
    """Combine the independent k signals into one decision with an audit trail.

    Each signal casts one vote and the modal vote wins. Ties are broken by
    silhouette, which the project proposal names as the primary internal
    criterion, and only then by parsimony. The full vote record is returned
    because on data this weakly structured the *disagreement* between criteria
    is a finding in its own right, not something to resolve silently.

    The gap statistic is reported separately rather than as a vote: it answers a
    different question (is there any structure at all) and including k = 1 in a
    vote about how many clusters to profile would confuse the two.
    """
    votes = {
        "kmeans_sse_elbow": km_report["sse_knee_k"],
        "kmeans_silhouette": km_report["best_silhouette_k"],
        "kmeans_davies_bouldin": km_report["best_davies_bouldin_k"],
        "kmeans_calinski": km_report["best_calinski_k"],
        "gmm_bic": gmm_report["best_bic_k"],
        "gmm_cv_loglik": gmm_report["best_cv_loglik_k"],
    }
    # Vote with the linkage the hierarchical step reports, not a hardcoded one,
    # so a degenerate linkage can never cast a vote.
    hm = hier_report.get("reported_method", "ward")
    rows = hier_report["by_method"][hm]["table"]
    best_row = max(rows, key=lambda r: (r["silhouette"] if r["silhouette"] is not None else -1))
    votes["hierarchical_%s_silhouette" % hm] = int(best_row["k"])

    tally = {}
    for v in votes.values():
        tally[v] = tally.get(v, 0) + 1
    top = max(tally.values())
    tied = sorted(k for k, c in tally.items() if c == top)

    if len(tied) > 1 and km_table is not None:
        chosen = max(tied, key=lambda k: (float(km_table.loc[k, "silhouette"]), -k))
        tie_rule = "tie between %s broken by silhouette" % tied
    else:
        chosen = tied[0]
        tie_rule = "clear modal vote" if len(tied) == 1 else "tie broken by parsimony"

    out = {"votes": votes, "tally": tally, "k_selected": int(chosen),
           "n_signals": len(votes), "n_agreeing": int(top),
           "tied_candidates": tied, "tie_break": tie_rule,
           "criteria_disagree": bool(len(set(votes.values())) > 2)}
    if gap_report is not None:
        out["gap_statistic"] = gap_report
    return out


def structure_verdict(silhouette):
    """Plain-language reading of how real the cluster structure is.

    Thresholds follow Kaufman and Rousseeuw's conventional bands. Stating them
    up front prevents a weak solution from being written up as a strong one.
    """
    if silhouette is None:
        return "no valid silhouette"
    if silhouette >= 0.50:
        return "reasonable structure"
    if silhouette >= 0.25:
        return "weak structure that could be artificial"
    return ("no substantial structure: the partition is a convenience segmentation of a "
            "continuum, not evidence of naturally separated groups")
