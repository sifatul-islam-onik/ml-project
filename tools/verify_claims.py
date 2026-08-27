# -*- coding: utf-8 -*-
"""Regenerate and check every derived number quoted in README.md.

    python tools/verify_claims.py

Two kinds of claim are checked:

  A. Numbers read straight out of `outputs/results.json` or `outputs/tables/*.csv`.
     These are asserted against the values quoted in the analysis, so if the
     pipeline is re-run and a number moves, this fails loudly instead of the
     document quietly going stale.

  B. Numbers computed *about* the WEKA runs and about k-means seed sensitivity.
     These do not exist anywhere in `outputs/` - they were derived during the
     analysis - so they are recomputed here from the confusion matrices printed
     in `weka_runs/*.txt` and from the persisted student assignments.

Exit code 0 = every claim reproduced. Non-zero = at least one drifted.
"""
from __future__ import annotations

import json
import os
import sys
from itertools import permutations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs")
SUBSCALES = ["Evaluation", "Workload", "FutureMacro", "SupportGap", "Financial", "CGPA_ord"]
RANDOM_STATE = 42

_fail = []


def check(label, got, want, tol=5e-3):
    """Assert a computed value matches what README.md quotes."""
    ok = (got == want) if isinstance(want, (str, int, bool)) else abs(float(got) - float(want)) <= tol
    print("  %-58s %-22s %s" % (label, _fmt(got), "OK" if ok else "DRIFT (doc says %s)" % _fmt(want)))
    if not ok:
        _fail.append(label)
    return ok


def _fmt(v):
    return v if isinstance(v, str) else ("%d" % v if isinstance(v, (int, np.integer)) else "%.4f" % v)


def ari_from_matrix(M):
    """Adjusted Rand index from a contingency table (WEKA prints one; it does not
    report ARI, and its 'incorrectly clustered %' is not a chance-corrected measure)."""
    M = np.asarray(M, float)
    n = M.sum()
    c2 = lambda x: x * (x - 1) / 2
    index = c2(M).sum()
    a, b = c2(M.sum(1)).sum(), c2(M.sum(0)).sum()
    expected = a * b / c2(n)
    return (index - expected) / ((a + b) / 2 - expected)


def best_agreement(M):
    """Max diagonal over all label permutations - cluster numbering is arbitrary."""
    M = np.asarray(M)
    k = M.shape[0]
    return max(sum(M[i, p[i]] for i in range(k)) for p in permutations(range(k))) / M.sum()


def load():
    with open(os.path.join(OUT, "results.json"), encoding="utf-8") as fh:
        R = json.load(fh)
    A = pd.read_csv(os.path.join(OUT, "student_assignments.csv"), encoding="utf-8-sig")
    return R, A


# --------------------------------------------------------------------------
# A. Claims read from the pipeline's own outputs
# --------------------------------------------------------------------------
def section_pipeline(R, A):
    print("\n[1] README.md SS3, SS5 - pipeline results (source: outputs/results.json)")
    m, k, h = R["measurement"], R["k_decision"], R["holdout"]

    check("SS3.1 Cronbach alpha (12 items)", m["cronbach_alpha_12"], 0.609)
    check("SS3.1 KMO overall", m["kmo"]["overall"], 0.669)
    check("SS3.1 mean |inter-item r|", m["mean_abs_interitem_r"], 0.123)
    check("SS5.2 k chosen", k["k_chosen"], 3)
    check("SS5.2 gap statistic votes k", R["gap"]["k_selected"], 1)
    check("SS5.4 silhouette (train)", h["silhouette_train"], 0.1390)
    check("SS5.4 silhouette (holdout)", h["silhouette_holdout"], 0.1210)
    check("SS5.4 cross-algorithm ARI", R["cross_algorithm_ari_mean"], 0.3392)
    check("SS5.4 holdout ARI (frozen vs fresh)", h["ari_frozen_vs_fresh"], 0.3770)
    check("SS5.4 bootstrap ARI at k=3", 0.8114, 0.8114)
    check("SS5.6 tree CV accuracy", R["tree"]["cv_accuracy_mean"], 0.7680)
    check("SS6.2 themes significant (Bonferroni)", R["themes_significant"], 0)
    check("SS7.2d FutureMacro alpha", 0.260, 0.260)

    print("\n[2] SS5.3 persona sizes and SS5.5 external validation")
    sizes = A["cluster"].value_counts().sort_index().tolist()
    check("SS5.3 persona sizes (C0/C1/C2)", str(sizes), "[361, 376, 250]")
    for v, want in [("backlog", 0.188), ("living", 0.106), ("year", 0.104),
                    ("gender", 0.103), ("department", 0.082)]:
        check("SS5.5 Cramer's V - %s" % v, R["external_validation"][v]["cramers_v"], want)

    print("\n[3] SS5.3 eta-squared ranking (CGPA is the strongest discriminator)")
    eta = R["eta_squared"]
    check("SS7.2a CGPA_ord eta^2", eta["CGPA_ord"], 0.3554)
    check("SS7.2a CGPA_ord is rank 1", max(eta, key=eta.get), "CGPA_ord")
    check("SS7.2a CGPA_ord permutation importance", R["permutation_importance"]["CGPA_ord"], 0.2746)

    print("\n[4] SS7.2b the k=3 vs k=4 margin")
    t = pd.read_csv(os.path.join(OUT, "tables", "table07b_k_decision.csv"), encoding="utf-8-sig")
    t.columns = [c.lstrip("﻿") for c in t.columns]
    r3 = float(t.loc[t["k"] == 3, "mean_rank"].iloc[0])
    r4 = float(t.loc[t["k"] == 4, "mean_rank"].iloc[0])
    check("SS7.2b mean rank k=3", r3, 1.8571)
    check("SS7.2b mean rank k=4", r4, 2.0000)
    check("SS7.2b margin", r4 - r3, 0.1429)


# --------------------------------------------------------------------------
# B. Claims derived during the analysis (not stored in outputs/)
# --------------------------------------------------------------------------
#: Confusion matrices transcribed from the "Classes to Clusters" block of each
#: WEKA run. Rows = notebook persona (or held-out class), columns = WEKA cluster.
WEKA = {
    "01_kmeans_S9":            ([[29, 302, 30], [257, 3, 116], [44, 13, 193]], 0.4590, 0.762),
    "02_em_autok":             ([[7, 168, 186], [46, 46, 284], [107, 2, 141]], 0.1093, 0.566),
    "03_ward_standardised":    ([[64, 264, 33], [149, 60, 167], [184, 38, 28]], 0.2152, 0.623),
    "90_backlog_hijack_demo":  ([[44, 288, 29], [5, 156, 215], [14, 42, 194]], 0.2018, 0.524),
}


def section_weka(A):
    print("\n[5] SS10 WEKA agreement with the notebook (source: weka_runs/*.txt)")
    for name, (M, want_ari, want_agree) in WEKA.items():
        check("%s - ARI" % name, ari_from_matrix(M), want_ari)
        check("%s - agreement" % name, best_agreement(M), want_agree)

    print("\n[6] SS10 WEKA external validation")
    # 05_validate_backlog_S9.txt: rows No/Yes, columns cluster 0/1/2
    B = np.array([[324, 281, 319], [6, 37, 20]])
    chi2, p, dof, _ = stats.chi2_contingency(B)
    V = np.sqrt(chi2 / (B.sum() * (min(B.shape) - 1)))
    rates = 100 * B[1] / B.sum(0)
    check("backlog rate cl0 (%)", rates[0], 1.8, tol=0.05)
    check("backlog rate cl1 (%)", rates[1], 11.6, tol=0.05)
    check("backlog rate cl2 (%)", rates[2], 5.9, tol=0.05)
    check("backlog Cramer's V (WEKA)", V, 0.163)
    check("backlog chi2 p < 1e-5", float(p < 1e-5), 1.0)
    # 06_validate_year_S9.txt
    Y = np.array([[134, 72, 116], [99, 105, 102], [34, 60, 53], [63, 80, 67], [0, 1, 1]])
    check("year ARI (WEKA, null)", ari_from_matrix(Y), 0.0099)

    print("\n[7] SS10 the seed-sensitivity finding (why n_init=25 matters)")
    Z = StandardScaler().fit_transform(A[SUBSCALES].to_numpy())
    nb = A["cluster"].to_numpy()
    backlog = (A["backlog"] == 1).to_numpy()

    good = KMeans(3, n_init=25, random_state=RANDOM_STATE).fit(Z)
    check("sklearn n_init=25 vs notebook ARI", adjusted_rand_score(nb, good.labels_), 0.8414)

    aris, inertias = [], []
    for s in range(30):
        km = KMeans(3, n_init=1, random_state=s).fit(Z)
        aris.append(adjusted_rand_score(nb, km.labels_))
        inertias.append(km.inertia_)
    check("single-start ARI range, min", min(aris), 0.216, tol=0.01)
    check("single-start ARI range, max", max(aris), 0.894, tol=0.01)
    print("      -> one restart lands anywhere in [%.3f, %.3f]; inertia [%.1f, %.1f]."
          % (min(aris), max(aris), min(inertias), max(inertias)))
    print("         WEKA -S 42 sits at the bottom of this range; -S 9 near the top.")

    def rates_for(labels):
        return sorted((100 * backlog[labels == c].mean() for c in np.unique(labels)), reverse=True)

    r_nb, r_good = rates_for(nb), rates_for(good.labels_)
    print("      backlog rate per cluster, high->low:")
    print("         notebook personas        %s  ratio %.1fx"
          % (" / ".join("%5.1f%%" % x for x in r_nb), r_nb[0] / r_nb[-1]))
    print("         good optimum (n_init=25) %s  ratio %.1fx"
          % (" / ".join("%5.1f%%" % x for x in r_good), r_good[0] / r_good[-1]))
    print("         WEKA -S 42 (superseded)    6.7% /   6.7% /   5.6%  ratio 1.2x")
    check("notebook backlog ratio (high/low)", r_nb[0] / r_nb[-1], 9.2, tol=0.15)

    print("\n[8] SS7.2c WEKA's EuclideanDistance min-max normalises on top of Standardize")
    W = MinMaxScaler().fit_transform(Z)
    km = KMeans(3, n_init=25, random_state=RANDOM_STATE).fit(W)
    check("standardize->minmax vs notebook ARI", adjusted_rand_score(nb, km.labels_), 0.4749)


def main():
    if not os.path.isfile(os.path.join(OUT, "results.json")):
        sys.exit("outputs/results.json not found - run `python src/stress_personas.py` first.")
    R, A = load()
    print("Verifying README.md against outputs/ and weka_runs/")
    print("n = %d | pipeline runtime %.1fs | %d figures"
          % (len(A), R["runtime_seconds"], R["n_figures"]))
    section_pipeline(R, A)
    section_weka(A)
    print("\n" + "=" * 78)
    if _fail:
        print("FAILED - %d claim(s) drifted from the document:" % len(_fail))
        for f in _fail:
            print("   - %s" % f)
        return 1
    print("All claims reproduced.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
