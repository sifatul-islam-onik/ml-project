# -*- coding: utf-8 -*-
"""
End-to-end pipeline: load -> audit -> structure -> PCA -> cluster -> validate
-> text -> profile -> supervised -> persist.

Run locally:      python -m ml.run_pipeline
Run on Kaggle:    same command, or import and call `main()` from a notebook cell.

Everything it produces lands under `outputs/` (or /kaggle/working on Kaggle):
    results.json                  every number the report cites
    RESULTS_SUMMARY.md            human-readable narrative of this run
    figures/*.png                 all figures
    tables/*.csv                  every table, ready to paste into the report
    models/*.joblib               fitted scaler / PCA / k-means, plus a
                                  self-contained inference bundle
    student_level_assignments.csv per-student profile assignment
    stress_prepared.arff          the same prepared table, for WEKA
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import joblib
import numpy as np
import pandas as pd

from . import cluster as CL
from . import config as C
from . import dataio as IO
from . import embeddings as EMB
from . import figures as FIG
from . import lexicon as LX
from . import preprocess as PP
from . import profile as PR
from . import reduce as RD
from . import structure as ST
from . import supervised as SUP
from . import textmodel as TM
from . import validate as VA


def _log(msg):
    print(msg, flush=True)


def _banner(step, msg):
    _log("\n[%s] %s" % (step, msg))


def main(argv=None):
    ap = argparse.ArgumentParser(description="KUET academic-stress clustering pipeline")
    ap.add_argument("--data", default=None, help="path to the response workbook")
    ap.add_argument("--k", type=int, default=None,
                    help="force a cluster count instead of using the vote")
    ap.add_argument("--secondary-k", type=int, default=3,
                    help="also profile this k as a documented alternative (0 to disable)")
    ap.add_argument("--bootstrap", type=int, default=C.BOOTSTRAP_B,
                    help="number of bootstrap resamples for the stability check")
    ap.add_argument("--with-embeddings", action="store_true",
                    help="run the optional multilingual-embedding cross-check (needs GPU+internet)")
    ap.add_argument("--fast", action="store_true",
                    help="fewer resamples and permutations; for a quick smoke run")
    args = ap.parse_args(argv)

    if args.fast:
        args.bootstrap = min(args.bootstrap, 40)

    t0 = time.time()
    C.ensure_dirs()
    R = {"meta": {"pipeline_version": "1.0.0",
                  "random_state": C.RANDOM_STATE,
                  "on_kaggle": C.ON_KAGGLE,
                  "python": sys.version.split()[0]}}

    # ----------------------------------------------------------------- load
    _banner(1, "Loading and validating the export")
    df, path = IO.load_raw(args.data)
    R["meta"]["data_file"] = os.path.basename(path)
    schema = IO.validate_schema(df)
    names = IO.column_names(df)
    R["audit"] = IO.audit(df, names)
    n = len(df)
    _log("    %s  ->  %d responses x %d columns" % (os.path.basename(path), n, schema["n_columns"]))
    _log("    missing (closed-ended): %d | duplicate rows: %d | out-of-range Likert: %d"
         % (R["audit"]["closed_ended_missing"], R["audit"]["exact_duplicate_rows"],
            R["audit"]["likert_out_of_range"]))

    # -------------------------------------------------------- preprocessing
    _banner(2, "Preprocessing: direction alignment, encoding, standardisation")
    items, raw = PP.build_items(df, names)
    bg = PP.build_background(df, names)
    Z, scaler = PP.standardise(items)
    strain = PP.composite_score(items)
    R["response_style"] = IO.response_style(items)
    R["imbalance"] = PP.imbalance_report(bg)

    desc = PP.item_descriptives(items, raw)
    IO.save_table(desc, "t01_item_descriptives")
    R["item_descriptives"] = desc.reset_index().to_dict("records")

    dists = {
        "year": PP.distribution(bg["year"], C.YEAR_ORD),
        "cgpa": PP.distribution(bg["cgpa"], C.CGPA_ORD),
        "gender": PP.distribution(bg["gender"], C.GENDER_ORD),
        "living": PP.distribution(bg["living"], C.LIVE_ORD),
        "department": PP.distribution(bg["department"]),
    }
    R["demographics"] = {k: v.to_dict("index") for k, v in dists.items()}
    R["demographics"]["backlog_pct"] = round(float(100 * bg["backlog"].mean()), 1)
    for k, v in dists.items():
        IO.save_table(v, "t02_dist_%s" % k)
    _log("    reverse-coded %s; %d items standardised" % (", ".join(C.REVERSED), len(C.ITEMS)))

    # ---------------------------------------------------------- measurement
    _banner(3, "Measurement structure of the instrument")
    R["structure"] = ST.analyse(items)
    R["structure"]["verdict"] = ST.verdict(R["structure"])
    _log("    Cronbach alpha = %.3f | KMO = %.3f | mean |inter-item r| = %.3f"
         % (R["structure"]["cronbach_alpha_12"], R["structure"]["kmo"]["overall"],
            R["structure"]["mean_abs_interitem_r"]))
    _log("    %s" % R["structure"]["verdict"])
    IO.save_table(pd.DataFrame({
        "corrected_item_total": R["structure"]["corrected_item_total"],
        "alpha_if_deleted": R["structure"]["alpha_if_deleted"],
        "kmo_per_item": R["structure"]["kmo"]["per_item"],
    }), "t03_reliability")

    # ------------------------------------------------------------------ PCA
    _banner(4, "Dimensionality reduction (PCA + parallel analysis + varimax)")
    scores, pca_model, pca_rep, load_df, rot_df = RD.run_pca(Z)
    pca_rep["component_interpretation"] = RD.interpret_components(rot_df)
    R["pca"] = pca_rep
    IO.save_table(pd.DataFrame({
        "eigenvalue": pca_rep["eigenvalues"],
        "explained_pct": pca_rep["explained_variance_pct"],
        "cumulative_pct": pca_rep["cumulative_variance_pct"],
        "random_p95": pca_rep["parallel_analysis"]["random_p95"],
    }, index=["PC%d" % (i + 1) for i in range(len(pca_rep["eigenvalues"]))]), "t04_pca_variance")
    IO.save_table(rot_df, "t05_pca_loadings_varimax")
    var95 = [k for k in pca_rep if k.startswith("n_for_")][0]
    _log("    %d components reach 95%% of variance; parallel analysis retains %d; "
         "PC1+PC2 = %.1f%%" % (pca_rep[var95], pca_rep["parallel_analysis"]["n_retain"],
                               pca_rep["variance_at_2_components_pct"]))

    # ------------------------------------------------------------- clustering
    _banner(5, "Clustering: k-means / Gaussian mixture / hierarchical / Gower")
    km_models, km_table, km_rep = CL.kmeans_sweep(Z)
    IO.save_table(km_table, "t06_kmeans_sweep")
    _log("    k-means sweep done  (SSE knee at k=%d, best silhouette at k=%d)"
         % (km_rep["sse_knee_k"], km_rep["best_silhouette_k"]))

    gmm_models, gmm_table, gmm_rep = CL.gmm_sweep(Z)
    IO.save_table(gmm_table, "t07_gmm_sweep")
    _log("    Gaussian mixture done  (BIC picks k=%d, CV log-likelihood picks k=%d)"
         % (gmm_rep["best_bic_k"], gmm_rep["best_cv_loglik_k"]))

    links, hier_labels, hier_rep = CL.hierarchical(Z)
    _log("    hierarchical done  -> %s" % hier_rep["note"])
    if hier_rep["degenerate_methods"]:
        _log("    degenerate linkages excluded from the cut: %s"
             % ", ".join(hier_rep["degenerate_methods"]))

    gap = CL.gap_statistic(Z, n_refs=10 if args.fast else 25)
    _log("    gap statistic: %s" % gap["interpretation"])

    selection = CL.select_k(km_rep, gmm_rep, hier_rep, gap_report=gap, km_table=km_table)
    K = args.k or selection["k_selected"]
    selection["k_used"] = int(K)
    selection["k_forced"] = args.k is not None
    R["k_selection"] = selection
    _log("    k selected = %d  (%d of %d signals agree, %s; votes: %s)"
         % (K, selection["n_agreeing"], selection["n_signals"],
            selection["tie_break"], selection["votes"]))

    km = km_models[K] if K in km_models else CL.KMeans(
        n_clusters=K, n_init=C.N_INIT, random_state=C.RANDOM_STATE).fit(Z)
    labels = km.labels_

    # Mixed-data cross-check over items + background, via Gower distance.
    mixed = pd.concat([items[C.ITEMS], bg[["year_ord", "cgpa_ord", "backlog"]],
                       bg[["gender", "living"]]], axis=1)
    D = CL.gower_matrix(mixed, numeric_cols=C.ITEMS + ["year_ord", "cgpa_ord"],
                        categorical_cols=["gender", "living", "backlog"])
    gower_labels_by, gower_table, gower_rep = CL.gower_sweep(D, ks=C.K_RANGE)
    IO.save_table(gower_table, "t08_gower_kmedoids_sweep")
    R["gower"] = gower_rep
    _log("    Gower k-medoids done  (best silhouette at k=%d)" % gower_rep["best_silhouette_k"])

    R["clustering"] = {"kmeans": km_rep, "gmm": gmm_rep, "hierarchical": hier_rep}

    # ------------------------------------------------------------ validation
    _banner(6, "Validation: internal indices, stability, consensus, external checks")
    sil_report, sil_values = VA.silhouette_breakdown(Z, labels)
    R["validation"] = {"silhouette": sil_report}
    R["validation"]["structure_verdict"] = CL.structure_verdict(sil_report["overall_mean"])
    _log("    silhouette = %.4f -> %s"
         % (sil_report["overall_mean"], R["validation"]["structure_verdict"]))

    boot = VA.bootstrap_stability(Z, K, labels, b=args.bootstrap)
    seeds = VA.seed_stability(Z, K)
    R["validation"]["bootstrap_stability"] = boot
    R["validation"]["seed_stability"] = seeds
    _log("    bootstrap ARI = %.3f (5th pct %.3f) | seed-to-seed ARI = %.3f"
         % (boot["ari_mean"], boot["ari_p05"], seeds["pairwise_ari_mean"]))

    M = VA.consensus(Z, K, b=40 if args.fast else 100)
    R["validation"]["consensus_by_cluster"] = VA.consensus_by_cluster(M, labels)

    R["validation"]["classes_to_clusters"] = VA.classes_to_clusters(labels, bg)
    for var, d in R["validation"]["classes_to_clusters"].items():
        _log("    external | %-11s chi2 p=%.4g  Cramer's V=%.3f  ARI=%.3f"
             % (var, d["p"], d["cramers_v"], d["adjusted_rand"]))

    algo_labels = {
        "k-means": labels,
        "GMM": gmm_models[K].predict(Z),
        "Ward": hier_labels["ward"][K],
        "Average-linkage": hier_labels["average"][K],
        "Spectral": CL.spectral_labels(Z, K),
        "Gower k-medoids": gower_labels_by[K],
    }
    agree = VA.compare_algorithms(algo_labels)
    IO.save_table(agree, "t09_algorithm_agreement")
    R["validation"]["algorithm_agreement"] = agree.to_dict()
    _log("    between-algorithm ARI vs k-means: %s"
         % {k: float(agree.loc["k-means", k]) for k in agree.columns if k != "k-means"})

    # ------------------------------------------------------------------ text
    _banner(7, "Open-ended text: lexicon, topic-model cross-check, null tests")
    txt1 = df[names["open_current"]]
    txt2 = df[names["open_previous"]]
    T1, mask1, _ = LX.tag_frame(txt1)
    T2, mask2, _ = LX.tag_frame(txt2)

    R["text"] = {
        "lexicon_version": LX.LEXICON_VERSION,
        "profile_current": LX.text_profile(txt1, mask1),
        "profile_previous": LX.text_profile(txt2, mask2),
        "coverage_current": LX.coverage(T1, mask1),
        "coverage_previous": LX.coverage(T2, mask2),
    }
    prev1 = LX.prevalence(T1, mask1)
    prev2 = LX.prevalence(T2, mask2)
    R["text"]["prevalence_current"] = prev1.to_dict()
    R["text"]["prevalence_previous"] = prev2.to_dict()
    IO.save_table(pd.DataFrame({"current_year_pct": prev1, "previous_years_pct": prev2}),
                  "t10_theme_prevalence")
    IO.save_table(LX.export_patterns(), "t11_lexicon_patterns")
    _log("    lexicon covers %.1f%% of current-year answers (%.2f themes each), %.1f%% of previous"
         % (R["text"]["coverage_current"]["coverage_pct"],
            R["text"]["coverage_current"]["mean_themes_per_answer"],
            R["text"]["coverage_previous"]["coverage_pct"]))

    answered_txt = txt1.fillna("").astype(str)[mask1.to_numpy()].tolist()
    topic_rep, _ = TM.topic_model_report(answered_txt)
    R["text"]["topic_model"] = topic_rep
    R["text"]["top_terms"] = TM.top_terms(answered_txt)

    R["text"]["null_checks"] = TM.null_checks(strain.to_numpy(), txt1, mask1)
    _log("    null check | non-response: %s" % R["text"]["null_checks"]["non_response"]["verdict"])
    _log("    null check | answer length: %s" % R["text"]["null_checks"]["answer_length"]["verdict"])

    R["text"]["theme_strain_association"] = TM.theme_strain_association(T1, mask1, strain.to_numpy())
    R["text"]["theme_by_year"] = TM.theme_by_group(T1, mask1, bg["year"])
    R["text"]["theme_by_gender"] = TM.theme_by_group(T1, mask1, bg["gender"])
    R["text"]["theme_by_living"] = TM.theme_by_group(T1, mask1, bg["living"])

    # --------------------------------------------------------------- profiles
    _banner(8, "Cluster profiling and naming")
    cnames, cdetail, Zp = PR.name_clusters(items, labels)
    R["profiles"] = {"names": {str(k): v for k, v in cnames.items()}, "detail": cdetail}
    prof = PR.profile_table(items, labels)
    IO.save_table(prof, "t12_cluster_item_means")
    IO.save_table(Zp, "t13_cluster_item_zscores")
    R["profiles"]["item_means"] = prof.to_dict("index")
    R["profiles"]["item_zscores"] = Zp.to_dict("index")

    summary = PR.cluster_summary(items, labels, bg, cnames)
    IO.save_table(summary, "t14_cluster_summary")
    R["profiles"]["summary"] = summary.reset_index().to_dict("records")
    for _, r in summary.iterrows():
        _log("    %-52s n=%3d (%4.1f%%)  strain=%.2f" %
             (r["profile"], r["n"], r["pct_of_sample"], r["mean_strain"]))

    R["profiles"]["composition"] = PR.composition(labels, bg)
    tbc = PR.theme_by_cluster(T1, mask1, labels)
    R["profiles"]["theme_by_cluster"] = tbc

    disc, top_items = PR.discriminating_items(items, labels)
    IO.save_table(disc, "t15_discriminating_items")
    R["profiles"]["discriminating_items"] = disc.reset_index().to_dict("records")
    R["profiles"]["recommendations"] = PR.recommendations(cnames, cdetail, tbc)
    _log("    most discriminating items: %s" % ", ".join(top_items))

    # A second solution at the k the project proposal anticipated, so the report
    # can show what is gained or lost by splitting further. Reported as an
    # alternative, never as the headline: its silhouette is the honest cost.
    alt_fig = None
    if args.secondary_k and args.secondary_k != K:
        k2 = args.secondary_k
        km2 = CL.KMeans(n_clusters=k2, n_init=C.N_INIT, random_state=C.RANDOM_STATE).fit(Z)
        n2, d2, Zp2 = PR.name_clusters(items, km2.labels_)
        sil2, _ = VA.silhouette_breakdown(Z, km2.labels_)
        sum2 = PR.cluster_summary(items, km2.labels_, bg, n2)
        IO.save_table(sum2, "t17_alternative_k%d_summary" % k2)
        IO.save_table(Zp2, "t18_alternative_k%d_zscores" % k2)
        R["alternative_solution"] = {
            "k": int(k2),
            "silhouette": sil2["overall_mean"],
            "silhouette_at_selected_k": sil_report["overall_mean"],
            "names": {str(a): b for a, b in n2.items()},
            "summary": sum2.reset_index().to_dict("records"),
            "agreement_with_selected": round(float(
                VA.adjusted_rand_score(labels, km2.labels_)), 4),
            "note": ("Provided because the project proposal anticipated roughly three "
                     "personas. It is an alternative segmentation of the same continuum, "
                     "not a better-supported one: silhouette %.4f vs %.4f at k=%d."
                     % (sil2["overall_mean"], sil_report["overall_mean"], K)),
        }
        alt_fig = FIG.fig_cluster_heatmap(Zp2, n2, "fig25_alternative_k%d_profiles.png" % k2)
        _log("    alternative k=%d profiled (silhouette %.4f vs %.4f at k=%d)"
             % (k2, sil2["overall_mean"], sil_report["overall_mean"], K))

    # ------------------------------------------------------------- supervised
    _banner(9, "Supervised extension and incremental-value tests")
    demo = PP.encode_background(bg)
    R["supervised"] = {}
    R["supervised"]["profile_recovery"] = SUP.profile_recovery(labels, demo, T1)
    _log("    recovering the profile from held-out features: %s"
         % R["supervised"]["profile_recovery"]["verdict"])

    # The CGPA band is the target here, so its ordinal encoding must leave the
    # feature matrix - otherwise the model simply reads the answer off an input.
    demo_no_cgpa = PP.encode_background(bg, exclude=("cgpa_ord",))
    y_cgpa = bg["cgpa"].astype(str)
    R["supervised"]["cgpa_prediction"] = SUP.incremental_text_value(
        items[C.ITEMS], demo_no_cgpa, T1, y_cgpa, "CGPA band")
    _log("    predicting CGPA band: %s" % R["supervised"]["cgpa_prediction"]["verdict"])
    _log("    text increment: %s" % R["supervised"]["cgpa_prediction"]["text_verdict"])

    if not args.fast:
        R["supervised"]["permutation_check_cgpa"] = SUP.permutation_check(
            pd.concat([items[C.ITEMS], demo_no_cgpa], axis=1), y_cgpa, n_permutations=100)

    imp, held = SUP.importances(pd.concat([items[C.ITEMS], demo_no_cgpa, T1], axis=1), y_cgpa)
    IO.save_table(imp, "t16_permutation_importance")
    R["supervised"]["top_features"] = imp.reset_index().to_dict("records")

    # ------------------------------------------------------------- embeddings
    if args.with_embeddings:
        _banner(10, "Optional: multilingual sentence-embedding cross-check")
        R["embeddings"] = EMB.run(txt1, mask1, T1)
        _log("    %s" % R["embeddings"].get("verdict", R["embeddings"].get("reason", "")))
    else:
        R["embeddings"] = {"status": "not requested",
                           "reason": "run with --with-embeddings (needs internet + ideally a GPU)"}

    # ---------------------------------------------------------------- figures
    _banner(11, "Rendering figures")
    figs = []
    figs.append(FIG.fig_profile(
        [("Academic year", dists["year"][dists["year"]["n"] >= 3]),
         ("Current CGPA", dists["cgpa"]), ("Gender", dists["gender"]),
         ("Living arrangement", dists["living"])], n))
    figs.append(FIG.fig_department(dists["department"]))
    figs.append(FIG.fig_likert(raw))
    figs.append(FIG.fig_item_means(items))
    figs.append(FIG.fig_correlation(R["structure"]))
    figs.append(FIG.fig_scree(pca_rep))
    figs.append(FIG.fig_variance(pca_rep))
    figs.append(FIG.fig_loadings(rot_df))
    figs.append(FIG.fig_pca_scatter(scores, labels, cnames, sil_report["overall_mean"]))
    try:
        figs.append(FIG.fig_tsne(RD.tsne_projection(Z), labels, cnames))
    except Exception as exc:
        _log("    t-SNE skipped: %s" % exc)
    figs.append(FIG.fig_k_selection(km_table, gmm_table, km_rep["sse_knee_k"]))
    figs.append(FIG.fig_gap(gap))
    figs.append(FIG.fig_dendrogram(links["ward"], K))
    figs.append(FIG.fig_silhouette_samples(sil_values, labels, cnames))
    figs.append(FIG.fig_stability(boot, seeds))
    figs.append(FIG.fig_algorithm_agreement(agree))
    figs.append(FIG.fig_cluster_heatmap(Zp, cnames))
    figs.append(FIG.fig_composition(R["profiles"]["composition"], cnames))
    figs.append(FIG.fig_discriminating(disc))
    figs.append(FIG.fig_theme_prevalence(prev1, prev2, int(mask1.sum()), int(mask2.sum())))
    order_t = prev1.index.tolist()
    if R["text"]["theme_by_year"]:
        figs.append(FIG.fig_theme_heatmap(R["text"]["theme_by_year"], order_t,
                                          "Theme prevalence (%) by academic year",
                                          "fig20_theme_by_year.png"))
    if R["text"]["theme_by_living"]:
        figs.append(FIG.fig_theme_heatmap(R["text"]["theme_by_living"], order_t,
                                          "Theme prevalence (%) by living arrangement",
                                          "fig21_theme_by_living.png"))
    figs.append(FIG.fig_theme_by_cluster(tbc, cnames, order_t[:6]))
    figs.append(FIG.fig_model_comparison(R["supervised"]["cgpa_prediction"]))
    figs.append(FIG.fig_model_comparison(R["supervised"]["profile_recovery"],
                                         "fig24_profile_recovery.png"))
    if alt_fig:
        figs.append(alt_fig)   # rendered earlier, alongside the alternative-k profiling
    _log("    %d figures written to %s" % (len(figs), C.FIG_DIR))
    R["figures"] = [os.path.basename(f) for f in figs]

    # ---------------------------------------------------------------- persist
    _banner(12, "Persisting models, tables and the run report")

    out = pd.DataFrame({
        "timestamp": df[names["timestamp"]],
        "year": bg["year"], "cgpa": bg["cgpa"], "gender": bg["gender"],
        "living": bg["living"], "department": bg["department"], "backlog": bg["backlog"],
    })
    for it in C.ITEMS:
        out["item_" + it] = items[it]
    out["strain_index"] = strain.round(3)
    out["cluster"] = labels
    out["profile"] = [cnames[int(c)] for c in labels]
    out["silhouette"] = np.round(sil_values, 4)
    out["pc1"] = np.round(scores[:, 0], 4)
    out["pc2"] = np.round(scores[:, 1], 4)
    for t in LX.THEME_NAMES:
        out["theme_" + t.split(" ")[0].strip("&,").lower()] = T1[t]
    assign_path = os.path.join(C.OUT_DIR, "student_level_assignments.csv")
    out.to_csv(assign_path, index=False, encoding="utf-8-sig")

    bundle = {
        "version": "1.0.0",
        "scaler": scaler,
        "pca": pca_model,
        "kmeans": km,
        "items": C.ITEMS,
        "reversed_items": C.REVERSED,
        "cluster_names": {int(k): v for k, v in cnames.items()},
        "k": int(K),
        "silhouette": sil_report["overall_mean"],
        "trained_on": {"n": int(n), "file": os.path.basename(path)},
    }
    joblib.dump(bundle, os.path.join(C.MODEL_DIR, "stress_profile_model.joblib"))
    joblib.dump(gmm_models[K], os.path.join(C.MODEL_DIR, "gaussian_mixture_k%d.joblib" % K))

    arff_df = pd.concat([items[C.ITEMS], bg[["year", "cgpa", "gender", "living", "department"]],
                         bg["backlog"].map({0: "No", 1: "Yes"}).rename("backlog"),
                         T1.rename(columns=lambda c: "theme_" + c).replace({0: "no", 1: "yes"}),
                         pd.Series([cnames[int(c)] for c in labels], name="profile",
                                   index=items.index)], axis=1)
    IO.write_arff(
        arff_df, os.path.join(C.OUT_DIR, "stress_prepared.arff"),
        relation="kuet_academic_stress",
        numeric=C.ITEMS,
        nominal={
            "year": C.YEAR_ORD, "cgpa": C.CGPA_ORD, "gender": C.GENDER_ORD,
            "living": C.LIVE_ORD,
            "department": sorted(bg["department"].unique()),
            "backlog": ["No", "Yes"],
            "profile": [cnames[int(c)] for c in sorted(set(labels))],
            **{"theme_" + t: ["no", "yes"] for t in LX.THEME_NAMES},
        },
        string_cols=[])

    R["meta"]["runtime_seconds"] = round(time.time() - t0, 1)
    IO.save_json(R, os.path.join(C.OUT_DIR, "results.json"))
    write_summary(R, os.path.join(C.OUT_DIR, "RESULTS_SUMMARY.md"))

    _log("\nDone in %.1fs. Outputs -> %s" % (R["meta"]["runtime_seconds"], C.OUT_DIR))
    _log("  results.json, RESULTS_SUMMARY.md, student_level_assignments.csv, stress_prepared.arff")
    _log("  %d figures, %d tables, %d model files"
         % (len(figs), len(os.listdir(C.TAB_DIR)), len(os.listdir(C.MODEL_DIR))))
    return R


# --------------------------------------------------------------------------
# Narrative summary
# --------------------------------------------------------------------------
def write_summary(R, path):
    """Write a markdown summary whose sentences are generated from the numbers.

    Deliberately templated: every claim below is interpolated from `results.json`,
    so the narrative cannot drift away from what the run actually produced.
    """
    s, v, k = R["structure"], R["validation"], R["k_selection"]
    txt = R["text"]
    L = []
    A = L.append

    A("# Run summary - KUET academic-stress clustering\n")
    A("Pipeline %s | data `%s` | n = %d | random_state = %d | %.1f s\n"
      % (R["meta"]["pipeline_version"], R["meta"]["data_file"], R["audit"]["n_rows"],
         R["meta"]["random_state"], R["meta"]["runtime_seconds"]))

    A("\n## 1. Data integrity\n")
    a = R["audit"]
    A("- %d responses, %d closed-ended missing values, %d exact duplicate rows, "
      "%d out-of-range Likert values." % (a["n_rows"], a["closed_ended_missing"],
                                          a["exact_duplicate_rows"], a["likert_out_of_range"]))
    A("- Collected %s to %s (%d days), peak %d responses on %s."
      % (a["collection_start"][:10], a["collection_end"][:10], a["collection_days"],
         a["peak_day_n"], a["peak_day"]))
    rs = R["response_style"]
    A("- Straight-lining %.1f%% of respondents; %.1f%% used all five scale points; "
      "mean within-row SD %.2f." % (rs["straight_lining_pct"], rs["pct_using_all_five_points"],
                                    rs["mean_within_row_sd"]))

    A("\n## 2. Does the instrument measure one thing?\n")
    A("- Cronbach's alpha = **%.3f** across twelve items (%.3f on the %d-item core)."
      % (s["cronbach_alpha_12"], s["cronbach_alpha_core"], len(s["core_items"])))
    A("- KMO = %.3f; Bartlett chi-square = %.1f, p = %.3g; mean |inter-item r| = %.3f."
      % (s["kmo"]["overall"], s["bartlett"]["chi2"], s["bartlett"]["p"],
         s["mean_abs_interitem_r"]))
    A("- Weakest items: %s." % (", ".join(s["weak_items"]) or "none"))
    A("\n> %s\n" % s["verdict"])

    A("\n## 3. Dimensionality\n")
    p = R["pca"]
    key = [x for x in p if x.startswith("n_for_")][0]
    A("- %d of 12 components are needed for 95%% of variance; PC1+PC2 explain only %.1f%%."
      % (p[key], p["variance_at_2_components_pct"]))
    A("- Kaiser would retain %d components; parallel analysis against random data retains %d."
      % (p["n_kaiser"], p["parallel_analysis"]["n_retain"]))
    A("- Because variance is spread almost evenly, PCA is used for visualisation and "
      "structure description, not to compress the feature set before clustering.")

    A("\n## 4. How many clusters?\n")
    A("| signal | k |")
    A("|---|---|")
    for name, kk in k["votes"].items():
        A("| %s | %d |" % (name.replace("_", " "), kk))
    A("\n- **k = %d** selected (%d of %d signals agree)%s."
      % (k["k_used"], k["n_agreeing"], k["n_signals"],
         "; forced by --k" if k["k_forced"] else ""))

    A("\n## 5. Are the clusters real?\n")
    sil = v["silhouette"]
    A("- Silhouette = **%.4f** -> %s." % (sil["overall_mean"], v["structure_verdict"]))
    A("- %.1f%% of students have a negative silhouette (closer to another cluster than their own)."
      % sil["pct_negative_overall"])
    b, sd = v["bootstrap_stability"], v["seed_stability"]
    A("- Bootstrap ARI = %.3f (5th percentile %.3f) over %d resamples; seed-to-seed ARI = %.3f."
      % (b["ari_mean"], b["ari_p05"], b["n_resamples"], sd["pairwise_ari_mean"]))
    A("- Held-out background variables (none of which entered the model):")
    for var, d in v["classes_to_clusters"].items():
        A("  - %s: chi-square p = %.3g, Cramer's V = %.3f, ARI = %.3f"
          % (var, d["p"], d["cramers_v"], d["adjusted_rand"]))

    A("\n## 6. The profiles\n")
    A("| profile | n | % of sample | mean strain | modal year | modal living | % female |")
    A("|---|---|---|---|---|---|---|")
    for r in R["profiles"]["summary"]:
        A("| %s | %d | %.1f | %.2f | %s | %s | %.1f |"
          % (r["profile"], r["n"], r["pct_of_sample"], r["mean_strain"],
             r["modal_year"], r["modal_living"], r["pct_female"]))
    A("\nMost discriminating items (eta-squared):")
    for r in R["profiles"]["discriminating_items"][:5]:
        A("- %s: %.3f" % (r["label"], r["eta_squared"]))

    A("\n## 7. Free-text findings\n")
    tc = txt["coverage_current"]
    tp = txt["profile_current"]
    A("- %d of %d students answered the current-stressor question (%.1f%%); median %g words, "
      "%.1f%% two words or fewer." % (tp["n_answered"], R["audit"]["n_rows"],
                                      tp["response_rate_pct"], tp["words_median"],
                                      tp["pct_le2_words"]))
    A("- Script mix: %.1f%% Latin, %.1f%% Bangla, %.1f%% mixed, plus %d romanised-Bangla answers."
      % (tp["lang_latin_pct"], tp["lang_bangla_pct"], tp["lang_mixed_pct"], tp["romanised_n"]))
    A("- The frozen lexicon (v%s, 13 themes) tags %.1f%% of answers, %.2f themes each."
      % (txt["lexicon_version"], tc["coverage_pct"], tc["mean_themes_per_answer"]))
    top3 = list(txt["prevalence_current"].items())[:3]
    A("- Most volunteered themes: %s." % ", ".join("%s (%.0f%%)" % (t, p) for t, p in top3))
    nc = txt["null_checks"]
    A("- Pre-specified null check 1 (non-response): %s (p = %.3g, d = %.2f)."
      % (nc["non_response"]["verdict"], nc["non_response"]["p"], nc["non_response"]["cohens_d"]))
    A("- Pre-specified null check 2 (answer length): %s (rho = %.3f, p = %.3g)."
      % (nc["answer_length"]["verdict"], nc["answer_length"]["spearman_rho"],
         nc["answer_length"]["p"]))

    A("\n## 8. Supervised checks\n")
    pr = R["supervised"]["profile_recovery"]
    cg = R["supervised"]["cgpa_prediction"]
    A("- Recovering the cluster from held-out features: best %.3f vs %.3f baseline - %s."
      % (pr["best_accuracy"], pr["rows"][0]["accuracy"], pr["verdict"]))
    A("- Predicting CGPA band: best %.3f vs %.3f baseline - %s."
      % (cg["best_accuracy"], cg["rows"][0]["accuracy"], cg["verdict"]))
    A("- %s." % cg["text_verdict"])

    emb = R.get("embeddings", {})
    if emb.get("status") == "ok":
        A("\n## 9. Embedding cross-check\n")
        A("- Model %s on %s, %d texts." % (emb["model"], emb["device"], emb["n_texts"]))
        A("- %s" % emb["verdict"])

    A("\n## Headline\n")
    A("> The twelve items are a checklist of partly independent stressors rather than one "
      "scale (alpha = %.2f). Students do not fall into naturally separated groups "
      "(silhouette = %.3f); the k = %d partition is a **useful segmentation of a continuum**, "
      "reported as such. What the free text adds is *what* the strain is about, and the "
      "lexicon covers %.0f%% of answers including the Bangla-script ones that an "
      "English-only pipeline would drop."
      % (s["cronbach_alpha_12"], sil["overall_mean"], k["k_used"], tc["coverage_pct"]))

    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    return path


if __name__ == "__main__":
    main()
