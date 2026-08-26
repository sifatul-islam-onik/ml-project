# -*- coding: utf-8 -*-
"""
All plotting for the pipeline.

One module so the visual language stays consistent: same palette, same grid
treatment, same left-aligned bold titles, direct value labels instead of a
legend wherever a legend would be redundant. Diverging colours are reserved for
Likert composition and correlations (which have a real midpoint); sequential
blue is used for magnitude heatmaps; the categorical ramp is only for series
identity.
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from scipy.cluster.hierarchy import dendrogram

from . import config as C
from . import lexicon as LX

plt.rcParams.update({
    "figure.facecolor": C.SURFACE, "axes.facecolor": C.SURFACE, "savefig.facecolor": C.SURFACE,
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
    "font.size": 9, "axes.titlesize": 10.5, "axes.labelsize": 9,
    "axes.edgecolor": C.BASE, "axes.linewidth": 0.8, "axes.labelcolor": C.INK2,
    "text.color": C.INK, "xtick.color": C.MUTED, "ytick.color": C.MUTED,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "grid.color": C.GRID, "grid.linewidth": 0.7,
    "legend.frameon": False, "legend.fontsize": 8.5,
    "figure.dpi": 170, "savefig.dpi": 170, "savefig.bbox": "tight",
})

SEQ_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list("seq", C.SEQ)
DIV_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list(
    "div", ["#b02f2f", "#e88a89", "#f0efec", "#86b6ef", "#1c5cab"])


def style(ax, xgrid=False, ygrid=True):
    ax.set_axisbelow(True)
    if ygrid:
        ax.grid(axis="y", visible=True, alpha=0.9)
    else:
        ax.grid(axis="y", visible=False)
    if xgrid:
        ax.grid(axis="x", visible=True, alpha=0.9)
        ax.grid(axis="y", visible=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(C.BASE)
    ax.tick_params(length=0)
    return ax


def save(fig, name):
    path = os.path.join(C.FIG_DIR, name)
    fig.savefig(path)
    plt.close(fig)
    return path


def _bare(ax):
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)


# --------------------------------------------------------------------------
# Descriptive
# --------------------------------------------------------------------------
def fig_profile(dists, n, name="fig01_respondent_profile.png"):
    fig, axes = plt.subplots(1, 4, figsize=(12.4, 2.9), gridspec_kw={"wspace": 0.62})
    for ax, (title, d) in zip(axes, dists):
        y = np.arange(len(d))[::-1]
        ax.barh(y, d["n"], height=0.62, color=C.CAT[0], edgecolor=C.SURFACE, linewidth=1.4)
        ax.set_yticks(y)
        ax.set_yticklabels(d.index, fontsize=8)
        ax.set_title(title, color=C.INK, pad=8, loc="left", fontweight="bold")
        for yy, (cnt, pct) in zip(y, d[["n", "pct"]].values):
            ax.text(cnt + n * 0.012, yy, "%d - %.0f%%" % (cnt, pct),
                    va="center", fontsize=7.6, color=C.INK2)
        ax.set_xlim(0, d["n"].max() * 1.62)
        ax.set_xticks([])
        style(ax, ygrid=False)
        ax.spines["bottom"].set_visible(False)
    fig.suptitle("Respondent profile  (n = %d)" % n, x=0.008, ha="left",
                 fontsize=11.5, fontweight="bold", color=C.INK, y=1.06)
    return save(fig, name)


def fig_department(d, name="fig02_department.png"):
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    y = np.arange(len(d))[::-1]
    ax.barh(y, d["n"], height=0.66, color=C.CAT[0], edgecolor=C.SURFACE, linewidth=1.4)
    ax.set_yticks(y)
    ax.set_yticklabels(d.index, fontsize=8.2)
    for yy, (cnt, pct) in zip(y, d[["n", "pct"]].values):
        ax.text(cnt + 4, yy, "%d (%.1f%%)" % (cnt, pct), va="center", fontsize=7.6, color=C.INK2)
    ax.set_xlim(0, d["n"].max() * 1.28)
    ax.set_xticks([])
    ax.set_title("Respondents by department", loc="left", color=C.INK, fontweight="bold", pad=8)
    style(ax, ygrid=False)
    ax.spines["bottom"].set_visible(False)
    return save(fig, name)


def fig_likert(raw, name="fig03_likert_composition.png"):
    """100% stacked diverging composition of the twelve items, raw scale."""
    order = raw[C.ITEMS].mean().sort_values().index.tolist()
    labels5 = ["1 Strongly disagree", "2 Disagree", "3 Neutral", "4 Agree", "5 Strongly agree"]
    fig, ax = plt.subplots(figsize=(9.6, 4.8))
    for i, it in enumerate(order):
        counts = raw[it].value_counts().reindex([1, 2, 3, 4, 5], fill_value=0)
        pct = 100 * counts / counts.sum()
        left = 0.0
        for v in range(1, 6):
            w = pct[v]
            ax.barh(i, w, left=left, height=0.62, color=C.DIV5[v - 1],
                    edgecolor=C.SURFACE, linewidth=1.6)
            if w >= 7:
                ax.text(left + w / 2, i, "%.0f" % w, ha="center", va="center", fontsize=7.2,
                        color="#ffffff" if v in (1, 5) else C.INK)
            left += w
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(["%s  %s" % (C.ITEM_QNUM[i], C.ITEM_LABEL[i]) for i in order], fontsize=8.2)
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_title("Raw response composition of the twelve items  (items ordered by raw mean)",
                 loc="left", color=C.INK, fontweight="bold", pad=26)
    ax.legend(handles=[Patch(facecolor=C.DIV5[i], label=labels5[i]) for i in range(5)],
              loc="lower left", bbox_to_anchor=(0, 1.005), ncol=5, handlelength=1.1,
              handleheight=0.9, columnspacing=1.2)
    style(ax, xgrid=True, ygrid=False)
    return save(fig, name)


def fig_item_means(items, name="fig04_item_means.png"):
    means = items[C.ITEMS].mean().sort_values()
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    yy = np.arange(len(means))
    ax.hlines(yy, 1, means.values, color=C.GRID, linewidth=2)
    ax.plot(means.values, yy, "o", markersize=8, color=C.CAT[0],
            markeredgecolor=C.SURFACE, markeredgewidth=1.6, linestyle="none")
    for i, (nm, m) in enumerate(means.items()):
        ax.text(m + 0.07, i, "%.2f" % m, va="center", fontsize=7.8, color=C.INK2)
    ax.set_yticks(yy)
    ax.set_yticklabels([C.ITEM_LABEL[i] for i in means.index], fontsize=8.4)
    ax.set_xlim(1, 5.45)
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_xlabel("Mean after direction alignment  (higher = more strain / less support)")
    ax.set_title("Mean strain per item, all items aligned to a common direction",
                 loc="left", color=C.INK, fontweight="bold", pad=8)
    style(ax, xgrid=True, ygrid=False)
    return save(fig, name)


def fig_correlation(struct, name="fig05_item_correlation.png"):
    Cm = pd.DataFrame(struct["correlation_matrix"]).loc[C.ITEMS, C.ITEMS]
    fig, ax = plt.subplots(figsize=(6.6, 5.6))
    im = ax.imshow(Cm.values, cmap=DIV_CMAP, vmin=-0.55, vmax=0.55)
    ax.set_xticks(range(len(C.ITEMS)))
    ax.set_xticklabels(C.ITEMS, rotation=55, ha="right", fontsize=7.6)
    ax.set_yticks(range(len(C.ITEMS)))
    ax.set_yticklabels(C.ITEMS, fontsize=7.6)
    for i in range(len(C.ITEMS)):
        for j in range(len(C.ITEMS)):
            v = Cm.values[i, j]
            if i != j and abs(v) >= 0.25:
                ax.text(j, i, "%.2f" % v, ha="center", va="center", fontsize=6.4,
                        color="#ffffff" if abs(v) > 0.42 else C.INK)
    ax.set_title("Inter-item correlation after direction alignment  (mean |r| = %.2f)"
                 % struct["mean_abs_interitem_r"],
                 loc="left", color=C.INK, fontweight="bold", pad=8)
    cb = fig.colorbar(im, ax=ax, shrink=0.72, pad=0.02)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=7.5, length=0)
    _bare(ax)
    return save(fig, name)


# --------------------------------------------------------------------------
# PCA
# --------------------------------------------------------------------------
def fig_scree(pca_rep, name="fig06_scree_parallel.png"):
    eig = np.array(pca_rep["eigenvalues"])
    rand = np.array(pca_rep["parallel_analysis"]["random_p95"])
    xs = np.arange(1, len(eig) + 1)
    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    keep = eig > rand
    ax.bar(xs, eig, width=0.6, color=[C.CAT[0] if k else C.BASE for k in keep],
           edgecolor=C.SURFACE, linewidth=1.4, label="Observed eigenvalue")
    ax.plot(xs, rand, "-o", color=C.CAT[1], linewidth=1.8, markersize=5,
            markeredgecolor=C.SURFACE, label="Random data, 95th percentile")
    ax.axhline(1.0, color=C.MUTED, linewidth=1.2, linestyle=":")
    ax.text(len(eig), 1.04, "Kaiser (eigenvalue = 1)", ha="right", fontsize=7.4, color=C.MUTED)
    for x, e in zip(xs, eig):
        ax.text(x, e + 0.05, "%.2f" % e, ha="center", fontsize=7, color=C.INK2)
    ax.set_xticks(xs)
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Eigenvalue")
    ax.set_ylim(0, max(eig) * 1.25)
    ax.set_title("Parallel analysis retains %d components; Kaiser would retain %d"
                 % (pca_rep["parallel_analysis"]["n_retain"], pca_rep["n_kaiser"]),
                 loc="left", color=C.INK, fontweight="bold", pad=22)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.005), ncol=2)
    style(ax)
    return save(fig, name)


def fig_variance(pca_rep, name="fig07_explained_variance.png"):
    cum = np.array(pca_rep["cumulative_variance_pct"])
    xs = np.arange(1, len(cum) + 1)
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    ax.plot(xs, cum, "-o", color=C.CAT[0], linewidth=2, markersize=6,
            markeredgecolor=C.SURFACE, markeredgewidth=1.4)
    ax.axhline(95, color=C.CAT[1], linestyle="--", linewidth=1.4)
    ax.text(len(cum), 96, "95% target", ha="right", fontsize=7.6, color=C.CAT[1])
    for x, v in zip(xs, cum):
        ax.text(x, v - 5, "%.0f" % v, ha="center", fontsize=7, color=C.INK2)
    ax.set_xticks(xs)
    ax.set_xlabel("Components retained")
    ax.set_ylabel("Cumulative variance (%)")
    ax.set_ylim(0, 105)
    key = [k for k in pca_rep if k.startswith("n_for_")][0]
    ax.set_title("%d of 12 components are needed to reach 95%% of variance"
                 % pca_rep[key], loc="left", color=C.INK, fontweight="bold", pad=8)
    style(ax)
    return save(fig, name)


def fig_loadings(rot_df, name="fig08_rotated_loadings.png"):
    fig, ax = plt.subplots(figsize=(5.4 + 0.7 * rot_df.shape[1], 4.8))
    M = rot_df.loc[C.ITEMS].values
    im = ax.imshow(M, cmap=DIV_CMAP, vmin=-0.8, vmax=0.8, aspect="auto")
    ax.set_xticks(range(rot_df.shape[1]))
    ax.set_xticklabels(rot_df.columns, fontsize=8.5)
    ax.set_yticks(range(len(C.ITEMS)))
    ax.set_yticklabels([C.ITEM_LABEL[i] for i in C.ITEMS], fontsize=8)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if abs(M[i, j]) >= 0.30:
                ax.text(j, i, "%.2f" % M[i, j], ha="center", va="center", fontsize=7.2,
                        color="#ffffff" if abs(M[i, j]) > 0.6 else C.INK)
    ax.set_title("Varimax-rotated loadings  (values below |0.30| left blank)",
                 loc="left", color=C.INK, fontweight="bold", pad=8)
    _bare(ax)
    return save(fig, name)


def fig_pca_scatter(scores, labels, names, sil, name="fig09_pca_scatter.png"):
    fig, ax = plt.subplots(figsize=(6.6, 5.0))
    for i, c in enumerate(sorted(set(labels))):
        m = labels == c
        ax.scatter(scores[m, 0], scores[m, 1], s=14, alpha=0.55, linewidths=0,
                   color=C.CAT[i % len(C.CAT)], label=names[int(c)])
    for i, c in enumerate(sorted(set(labels))):
        m = labels == c
        ax.scatter(scores[m, 0].mean(), scores[m, 1].mean(), s=170, marker="X",
                   color=C.CAT[i % len(C.CAT)], edgecolor=C.SURFACE, linewidth=2, zorder=5)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Clusters in PCA space overlap heavily  (silhouette = %.3f)" % sil,
                 loc="left", color=C.INK, fontweight="bold", pad=30)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.005), ncol=1, fontsize=8)
    style(ax, xgrid=True, ygrid=True)
    return save(fig, name)


def fig_tsne(emb, labels, names, name="fig10_tsne.png"):
    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    for i, c in enumerate(sorted(set(labels))):
        m = labels == c
        ax.scatter(emb[m, 0], emb[m, 1], s=13, alpha=0.6, linewidths=0,
                   color=C.CAT[i % len(C.CAT)], label=names[int(c)])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("t-SNE projection - no separated islands appear",
                 loc="left", color=C.INK, fontweight="bold", pad=30)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.005), ncol=1, fontsize=8)
    _bare(ax)
    return save(fig, name)


# --------------------------------------------------------------------------
# Cluster selection & validation
# --------------------------------------------------------------------------
def fig_k_selection(km_table, gmm_table, knee_k, name="fig11_k_selection.png"):
    fig, axes = plt.subplots(1, 4, figsize=(13.2, 2.9), gridspec_kw={"wspace": 0.34})
    ks = list(km_table.index)

    ax = axes[0]
    ax.plot(ks, km_table["sse"], "-o", color=C.CAT[0], linewidth=2, markersize=6,
            markeredgecolor=C.SURFACE, markeredgewidth=1.4)
    ax.axvline(knee_k, color=C.CAT[1], linestyle="--", linewidth=1.4)
    ax.text(knee_k, km_table["sse"].max(), " knee k=%d" % knee_k, fontsize=7.6, color=C.CAT[1])
    ax.set_title("k-means SSE (elbow)", loc="left", color=C.INK, fontweight="bold", pad=8)
    ax.set_xlabel("k")

    ax = axes[1]
    ax.plot(ks, km_table["silhouette"], "-o", color=C.CAT[0], linewidth=2, markersize=6,
            markeredgecolor=C.SURFACE, markeredgewidth=1.4)
    ax.axhline(0.25, color=C.CAT[1], linestyle="--", linewidth=1.3)
    ax.text(ks[-1], 0.258, "weak-structure floor", ha="right", fontsize=7.2, color=C.CAT[1])
    for k, v in zip(ks, km_table["silhouette"]):
        ax.text(k, v + 0.006, "%.3f" % v, ha="center", fontsize=7, color=C.INK2)
    ax.set_ylim(0, max(0.30, float(km_table["silhouette"].max()) * 1.3))
    ax.set_title("Silhouette (higher better)", loc="left", color=C.INK, fontweight="bold", pad=8)
    ax.set_xlabel("k")

    ax = axes[2]
    ax.plot(ks, km_table["davies_bouldin"], "-o", color=C.CAT[2], linewidth=2, markersize=6,
            markeredgecolor=C.SURFACE, markeredgewidth=1.4)
    ax.set_title("Davies-Bouldin (lower better)", loc="left", color=C.INK, fontweight="bold", pad=8)
    ax.set_xlabel("k")

    ax = axes[3]
    ax.plot(ks, gmm_table["bic"], "-o", color=C.CAT[3], linewidth=2, markersize=6,
            markeredgecolor=C.SURFACE, markeredgewidth=1.4, label="BIC")
    ax.set_title("Gaussian-mixture BIC (lower better)", loc="left", color=C.INK,
                 fontweight="bold", pad=8)
    ax.set_xlabel("k")

    for ax in axes:
        ax.set_xticks(ks)
        style(ax)
    return save(fig, name)


def fig_gap(gap, name="fig11b_gap_statistic.png"):
    """Gap curve with Tibshirani error bars; k=1 included so 'no structure' is visible."""
    ks = gap["ks"]
    g = np.array(gap["gap"], dtype=float)
    sk = np.array(gap["s_k"], dtype=float)
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    ax.errorbar(ks, g, yerr=sk, fmt="-o", color=C.CAT[0], linewidth=2, markersize=6,
                markeredgecolor=C.SURFACE, markeredgewidth=1.4,
                ecolor=C.INK2, elinewidth=1, capsize=3)
    sel = gap["k_selected"]
    ax.scatter([sel], [g[ks.index(sel)]], s=170, marker="o", facecolor="none",
               edgecolor=C.CAT[1], linewidth=2, zorder=5)
    ax.annotate("selected k = %d" % sel, xy=(sel, g[ks.index(sel)]),
                xytext=(6, 10), textcoords="offset points", fontsize=8, color=C.CAT[1])
    ax.set_xticks(ks)
    ax.set_xlabel("k  (k = 1 means 'one homogeneous group')")
    ax.set_ylabel("Gap")
    ax.set_title("Gap statistic against a uniform null - %s"
                 % ("no structure found" if gap["supports_no_structure"]
                    else "structure found at k = %d" % sel),
                 loc="left", color=C.INK, fontweight="bold", pad=8)
    style(ax)
    return save(fig, name)


def fig_dendrogram(link, k, name="fig12_dendrogram.png"):
    fig, ax = plt.subplots(figsize=(9.0, 3.6))
    dendrogram(link, ax=ax, no_labels=True, color_threshold=link[-(k - 1), 2],
               above_threshold_color=C.BASE)
    ax.set_ylabel("Merge distance (Ward)")
    ax.set_title("Ward dendrogram - merge cost rises smoothly, so no k is obviously correct",
                 loc="left", color=C.INK, fontweight="bold", pad=8)
    style(ax)
    return save(fig, name)


def fig_silhouette_samples(sil_values, labels, names, name="fig13_silhouette_samples.png"):
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    y = 0
    ticks, ticklabels = [], []
    for i, c in enumerate(sorted(set(labels))):
        vals = np.sort(sil_values[labels == c])
        ax.fill_betweenx(np.arange(y, y + len(vals)), 0, vals,
                         color=C.CAT[i % len(C.CAT)], linewidth=0, alpha=0.85)
        ticks.append(y + len(vals) / 2)
        ticklabels.append(names[int(c)])
        y += len(vals) + 12
    ax.axvline(sil_values.mean(), color=C.INK, linestyle="--", linewidth=1.3)
    ax.text(sil_values.mean(), y, " mean %.3f" % sil_values.mean(), fontsize=7.6, color=C.INK)
    ax.axvline(0, color=C.MUTED, linewidth=1)
    ax.set_yticks(ticks)
    ax.set_yticklabels(ticklabels, fontsize=8)
    ax.set_xlabel("Silhouette coefficient")
    ax.set_title("Per-student silhouette: a large share of members sit near or below zero",
                 loc="left", color=C.INK, fontweight="bold", pad=8)
    style(ax, xgrid=True, ygrid=False)
    return save(fig, name)


def fig_stability(boot, seeds, name="fig14_stability.png"):
    fig, ax = plt.subplots(figsize=(6.6, 2.6))
    bars = [("Bootstrap ARI (mean)", boot["ari_mean"]),
            ("Bootstrap ARI (5th pct)", boot["ari_p05"]),
            ("Seed-to-seed ARI (mean)", seeds["pairwise_ari_mean"]),
            ("Seed-to-seed ARI (min)", seeds["pairwise_ari_min"])]
    y = np.arange(len(bars))[::-1]
    vals = [b[1] for b in bars]
    colors = [C.CAT[0] if v >= 0.75 else (C.CAT[3] if v >= 0.5 else C.CAT[7]) for v in vals]
    ax.barh(y, vals, height=0.6, color=colors, edgecolor=C.SURFACE, linewidth=1.4)
    for yy, v in zip(y, vals):
        ax.text(v + 0.015, yy, "%.3f" % v, va="center", fontsize=8, color=C.INK2)
    ax.axvline(0.75, color=C.MUTED, linestyle="--", linewidth=1.2)
    ax.text(0.755, y.max() + 0.45, "0.75 = stable", fontsize=7.4, color=C.MUTED)
    ax.set_yticks(y)
    ax.set_yticklabels([b[0] for b in bars], fontsize=8.2)
    ax.set_xlim(0, 1.05)
    ax.set_title("Cluster stability under resampling and reseeding",
                 loc="left", color=C.INK, fontweight="bold", pad=8)
    style(ax, xgrid=True, ygrid=False)
    return save(fig, name)


def fig_algorithm_agreement(M, name="fig15_algorithm_agreement.png"):
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    im = ax.imshow(M.values.astype(float), cmap=SEQ_CMAP, vmin=0, vmax=1)
    ax.set_xticks(range(len(M)))
    ax.set_xticklabels(M.columns, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(len(M)))
    ax.set_yticklabels(M.index, fontsize=8)
    for i in range(len(M)):
        for j in range(len(M)):
            v = float(M.values[i, j])
            ax.text(j, i, "%.2f" % v, ha="center", va="center", fontsize=7.6,
                    color="#ffffff" if v > 0.55 else C.INK)
    ax.set_title("Between-algorithm agreement (adjusted Rand)",
                 loc="left", color=C.INK, fontweight="bold", pad=8)
    _bare(ax)
    return save(fig, name)


# --------------------------------------------------------------------------
# Profiles
# --------------------------------------------------------------------------
def fig_cluster_heatmap(Zp, names, name="fig16_cluster_profiles.png"):
    order = sorted(Zp.index, key=lambda c: Zp.loc[c].mean())
    M = Zp.loc[order, C.ITEMS].T.values
    fig, ax = plt.subplots(figsize=(2.4 + 1.5 * len(order), 4.8))
    lim = float(np.abs(M).max())
    im = ax.imshow(M, cmap=DIV_CMAP, vmin=-lim, vmax=lim, aspect="auto")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([names[int(c)] for c in order], fontsize=8, rotation=16, ha="right")
    ax.set_yticks(range(len(C.ITEMS)))
    ax.set_yticklabels([C.ITEM_LABEL[i] for i in C.ITEMS], fontsize=8)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, "%+.2f" % M[i, j], ha="center", va="center", fontsize=7.4,
                    color="#ffffff" if abs(M[i, j]) > lim * 0.62 else C.INK)
    ax.set_title("Cluster item means as z-scores  (0 = sample average)",
                 loc="left", color=C.INK, fontweight="bold", pad=8)
    cb = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=7.5, length=0)
    _bare(ax)
    return save(fig, name)


def fig_composition(comp, names, name="fig17_cluster_composition.png"):
    keys = [("year", "Academic year mix", C.YEAR_ORD),
            ("living", "Living arrangement mix", C.LIVE_ORD),
            ("cgpa", "CGPA band mix", C.CGPA_ORD)]
    keys = [k for k in keys if k[0] in comp]
    fig, axes = plt.subplots(1, len(keys), figsize=(4.9 * len(keys), 3.2))
    if len(keys) == 1:
        axes = [axes]
    for ax, (var, title, order) in zip(axes, keys):
        pct = comp[var]["pct"]
        clusters = sorted(pct)
        levels = [c for c in order if c in pct[clusters[0]]]
        ys = np.arange(len(clusters))[::-1]
        for ci, c in enumerate(clusters):
            left = 0.0
            for li, lv in enumerate(levels):
                v = pct[c][lv]
                ax.barh(ys[ci], v, left=left, height=0.6, color=C.CAT[li % len(C.CAT)],
                        edgecolor=C.SURFACE, linewidth=1.6)
                if v >= 9:
                    ax.text(left + v / 2, ys[ci], "%.0f" % v, ha="center", va="center",
                            fontsize=7.2, color="#ffffff" if li in (0, 5) else C.INK)
                left += v
        ax.set_yticks(ys)
        ax.set_yticklabels([names[int(c)] for c in clusters], fontsize=7.8)
        ax.set_xlim(0, 100)
        ax.set_xticks([0, 50, 100])
        ax.set_xticklabels(["0%", "50%", "100%"])
        ax.set_title("%s  (chi-square p = %.4f)" % (title, comp[var]["p"]),
                     loc="left", color=C.INK, fontweight="bold", pad=22)
        ax.legend(handles=[Patch(facecolor=C.CAT[i % len(C.CAT)], label=lv)
                           for i, lv in enumerate(levels)],
                  loc="lower left", bbox_to_anchor=(0, 1.005), ncol=len(levels),
                  handlelength=1.0, fontsize=7.6)
        style(ax, xgrid=True, ygrid=False)
    return save(fig, name)


def fig_discriminating(disc, name="fig18_discriminating_items.png"):
    d = disc.sort_values("eta_squared")
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    y = np.arange(len(d))
    ax.barh(y, d["eta_squared"] * 100, height=0.62, color=C.CAT[0],
            edgecolor=C.SURFACE, linewidth=1.4)
    for yy, v in zip(y, d["eta_squared"] * 100):
        ax.text(v + 0.4, yy, "%.1f%%" % v, va="center", fontsize=7.6, color=C.INK2)
    ax.set_yticks(y)
    ax.set_yticklabels([C.ITEM_LABEL[i] for i in d.index], fontsize=8.2)
    ax.set_xlabel("Variance in the item explained by the cluster split (eta-squared, %)")
    ax.set_xlim(0, float((d["eta_squared"] * 100).max()) * 1.25)
    ax.set_title("Which items actually separate the profiles",
                 loc="left", color=C.INK, fontweight="bold", pad=8)
    style(ax, xgrid=True, ygrid=False)
    return save(fig, name)


# --------------------------------------------------------------------------
# Text
# --------------------------------------------------------------------------
def fig_theme_prevalence(prev1, prev2, n1, n2, name="fig19_theme_prevalence.png"):
    order = prev1.index.tolist()
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    yy = np.arange(len(order))[::-1]
    h = 0.36
    ax.barh(yy + h / 2, [prev1[t] for t in order], height=h, color=C.CAT[0],
            edgecolor=C.SURFACE, linewidth=1.2, label="Current year (n=%d)" % n1)
    ax.barh(yy - h / 2, [prev2[t] for t in order], height=h, color=C.CAT[1],
            edgecolor=C.SURFACE, linewidth=1.2, label="Previous years (n=%d)" % n2)
    for i, t in enumerate(order):
        ax.text(prev1[t] + 0.4, yy[i] + h / 2, "%.0f%%" % prev1[t], va="center",
                fontsize=7.2, color=C.INK2)
        ax.text(prev2[t] + 0.4, yy[i] - h / 2, "%.0f%%" % prev2[t], va="center",
                fontsize=7.2, color=C.INK2)
    ax.set_yticks(yy)
    ax.set_yticklabels(order, fontsize=8.2)
    ax.set_xlabel("% of answering students who mentioned the theme")
    ax.set_xlim(0, max(prev1.max(), prev2.max()) * 1.22)
    ax.set_title("Stressor themes volunteered in the free-text answers",
                 loc="left", color=C.INK, fontweight="bold", pad=22)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.005), ncol=2)
    style(ax, xgrid=True, ygrid=False)
    return save(fig, name)


def fig_theme_heatmap(by_group, order_themes, title, name):
    keys = list(by_group)
    M = np.array([[by_group[g][t] for g in keys] for t in order_themes])
    fig, ax = plt.subplots(figsize=(2.6 + 1.1 * len(keys), 4.6))
    im = ax.imshow(M, cmap=SEQ_CMAP, aspect="auto", vmin=0, vmax=M.max())
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(["%s\n(n=%d)" % (g, by_group[g]["n"]) for g in keys], fontsize=8)
    ax.set_yticks(range(len(order_themes)))
    ax.set_yticklabels(order_themes, fontsize=8)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, "%.0f" % M[i, j], ha="center", va="center", fontsize=7.4,
                    color="#ffffff" if M[i, j] > M.max() * 0.55 else C.INK)
    ax.set_title(title, loc="left", color=C.INK, fontweight="bold", pad=8)
    _bare(ax)
    return save(fig, name)


def fig_theme_by_cluster(tbc, names, themes, name="fig22_theme_by_cluster.png"):
    clusters = sorted(tbc)
    x = np.arange(len(themes))
    w = 0.8 / len(clusters)
    fig, ax = plt.subplots(figsize=(8.2, 3.8))
    for ci, c in enumerate(clusters):
        vals = [tbc[c][t] for t in themes]
        off = (ci - (len(clusters) - 1) / 2) * w
        ax.bar(x + off, vals, width=w * 0.88, color=C.CAT[ci % len(C.CAT)],
               edgecolor=C.SURFACE, linewidth=1.2, label=names[int(c)])
        for xi, v in zip(x + off, vals):
            ax.text(xi, v + 0.3, "%.0f" % v, ha="center", fontsize=6.6, color=C.INK2)
    ax.set_xticks(x)
    ax.set_xticklabels([t.split(" &")[0].split(",")[0] for t in themes],
                       rotation=18, ha="right", fontsize=8)
    ax.set_ylabel("% mentioning the theme")
    ax.set_title("Volunteered themes by cluster - independent corroboration of the profiles",
                 loc="left", color=C.INK, fontweight="bold", pad=22)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.005), ncol=len(clusters), fontsize=7.8)
    style(ax)
    return save(fig, name)


# --------------------------------------------------------------------------
# Supervised
# --------------------------------------------------------------------------
def fig_model_comparison(report, name="fig23_model_comparison.png"):
    rows = report["rows"]
    base = rows[0]["accuracy"] * 100
    body = rows[1:]
    names_ = [r["features"] for r in body]
    acc = [r["accuracy"] * 100 for r in body]
    sd = [r["accuracy_sd"] * 100 for r in body]
    f1 = [(r["macro_f1"] or 0) * 100 for r in body]
    x = np.arange(len(body))
    w = 0.36
    fig, ax = plt.subplots(figsize=(1.9 * len(body) + 2.4, 3.4))
    ax.bar(x - w / 2, acc, width=w * 0.9, color=C.CAT[0], edgecolor=C.SURFACE,
           linewidth=1.2, label="Accuracy", yerr=sd, capsize=3,
           error_kw={"elinewidth": 1, "ecolor": C.INK2})
    ax.bar(x + w / 2, f1, width=w * 0.9, color=C.CAT[1], edgecolor=C.SURFACE,
           linewidth=1.2, label="Macro-F1")
    for xi, v in zip(x - w / 2, acc):
        ax.text(xi, v + 1.2, "%.1f" % v, ha="center", fontsize=7.4, color=C.INK2)
    for xi, v in zip(x + w / 2, f1):
        ax.text(xi, v + 1.2, "%.1f" % v, ha="center", fontsize=7.4, color=C.INK2)
    ax.axhline(base, color=C.CAT[7], linestyle="--", linewidth=1.5)
    ax.text(len(body) - 0.5, base + 1.4, "majority baseline %.1f%%" % base,
            ha="right", fontsize=7.6, color=C.CAT[7])
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace(" + ", "\n+ ") for n in names_], fontsize=7.8)
    ax.set_ylabel("%")
    ax.set_ylim(0, max(max(acc), base) * 1.45)
    ax.set_title("Predicting %s - %s" % (report["target"], report["verdict"]),
                 loc="left", color=C.INK, fontweight="bold", pad=22)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.005), ncol=2)
    style(ax)
    return save(fig, name)
