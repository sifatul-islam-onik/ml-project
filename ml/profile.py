# -*- coding: utf-8 -*-
"""
Cluster profiling and plain-language naming (methodology 5.7).

Names are derived from the numbers, not chosen first. `name_clusters` ranks each
cluster's item z-scores against the grand mean and builds the label from the
items that actually separate it, so a reader can check the label against the
profile table. Ordering is by overall strain so P1 is always the lowest-strain
profile regardless of the arbitrary integer k-means assigns.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from . import config as C
from . import lexicon as LX


#: Short descriptors used when an item is a cluster's defining high/low feature.
_HIGH_TAG = {
    "MissMeal": "skipped meals",
    "PileUp": "coursework backlog",
    "SleepLoss": "sleep sacrifice",
    "LabStress": "lab strain",
    "ExamWorry": "exam anxiety",
    "ResultDemotiv": "result-driven demotivation",
    "CGPACompare": "peer CGPA comparison",
    "AskTeacher": "low help-seeking",
    "Feedback": "unsupportive feedback",
    "Financial": "financial pressure",
    "JobWorry": "career anxiety",
    "SocioPol": "socio-political worry",
}


def profile_table(items, labels):
    """Mean of every item within each cluster, plus the grand mean column."""
    df = items.copy()
    df["cluster"] = labels
    prof = df.groupby("cluster")[C.ITEMS].mean().round(3)
    prof.loc["ALL"] = items[C.ITEMS].mean().round(3)
    return prof


def zprofile(items, labels):
    """Cluster means expressed as z-scores of the item distribution.

    This is what the naming rule reads. Raw means are not comparable across
    items, because item means range from about 2.4 to 4.4 on the aligned scale.
    """
    mu = items[C.ITEMS].mean()
    sd = items[C.ITEMS].std()
    df = items.copy()
    df["cluster"] = labels
    prof = df.groupby("cluster")[C.ITEMS].mean()
    return ((prof - mu) / sd).round(3)


def order_by_strain(items, labels):
    """Map raw cluster ids to strain-ordered positions (0 = lowest strain)."""
    strain = pd.Series(items[C.ITEMS].mean(axis=1).to_numpy()).groupby(labels).mean()
    return {int(c): i for i, c in enumerate(strain.sort_values().index)}


def name_clusters(items, labels, z_threshold=0.30):
    """Derive a label per cluster from its most distinctive items.

    Rule: take the item z-scores of each cluster, keep those beyond
    +/- z_threshold, and build the name from the two strongest highs and the
    strongest low. A cluster with nothing beyond the threshold is named as the
    middle group, which is an honest outcome on weakly separated data.
    """
    Zp = zprofile(items, labels)
    order = order_by_strain(items, labels)
    k = len(order)
    names, detail = {}, {}

    for c in Zp.index:
        row = Zp.loc[c].sort_values(ascending=False)
        highs = [i for i in row.index if row[i] >= z_threshold]
        lows = [i for i in row.index if row[i] <= -z_threshold]
        pos = order[int(c)]
        rank = "P%d" % (pos + 1)

        # `row` is sorted descending, so lows[-1] is the most strongly suppressed item.
        if not highs and not lows:
            label = "%s - Middle / undifferentiated" % rank
        elif not highs:
            label = "%s - Lower strain, esp. low %s" % (rank, _HIGH_TAG[lows[-1]])
        else:
            bits = ["high " + _HIGH_TAG[h] for h in highs[:2]]
            if lows:
                bits.append("low " + _HIGH_TAG[lows[-1]])
            label = "%s - %s" % (rank, ", ".join(bits))

        names[int(c)] = label
        detail[int(c)] = {
            "strain_rank": pos + 1,
            "of": k,
            "elevated_items": [{"item": i, "z": float(row[i]), "label": C.ITEM_LABEL[i]}
                               for i in highs],
            "suppressed_items": [{"item": i, "z": float(row[i]), "label": C.ITEM_LABEL[i]}
                                 for i in lows],
        }
    return names, detail, Zp


def composition(labels, background, variables=("year", "cgpa", "gender", "living", "department"),
                orders=None):
    """Row-percentage composition of each cluster across background variables."""
    orders = orders or {"year": C.YEAR_ORD, "cgpa": C.CGPA_ORD,
                        "gender": C.GENDER_ORD, "living": C.LIVE_ORD}
    out = {}
    s = pd.Series(labels, index=background.index, name="cluster")
    for var in variables:
        if var not in background.columns:
            continue
        ct = pd.crosstab(s, background[var].astype(str), normalize="index") * 100
        order = orders.get(var)
        if order:
            keep = [c for c in order if c in ct.columns]
            ct = ct[keep + [c for c in ct.columns if c not in keep]]
        raw = pd.crosstab(s, background[var].astype(str))
        chi2, p, dof, _ = stats.chi2_contingency(raw)
        out[var] = {
            "pct": {int(c): {str(k): round(float(ct.loc[c, k]), 1) for k in ct.columns}
                    for c in ct.index},
            "chi2": round(float(chi2), 2), "dof": int(dof), "p": float(p),
        }
    return out


def theme_by_cluster(T, mask, labels):
    """Theme prevalence within each cluster - the triangulation step.

    Text themes were never clustering features, so agreement between what a
    cluster scores high on and what its members volunteer is independent
    corroboration rather than circular reasoning.
    """
    out = {}
    m = mask.to_numpy()
    for c in sorted(set(labels)):
        sel = m & (labels == c)
        out[int(c)] = {"n_answered": int(sel.sum()),
                       **{t: round(float(100 * T.loc[sel, t].mean()), 1) for t in LX.THEME_NAMES}}
    return out


def cluster_summary(items, labels, background, names):
    """One row per cluster: size, strain, modal background categories."""
    strain = items[C.ITEMS].mean(axis=1)
    rows = []
    for c in sorted(set(labels)):
        m = labels == c
        rows.append({
            "cluster": int(c),
            "profile": names[int(c)],
            "n": int(m.sum()),
            "pct_of_sample": round(float(100 * m.mean()), 1),
            "mean_strain": round(float(strain[m].mean()), 3),
            "sd_strain": round(float(strain[m].std()), 3),
            "modal_year": background.loc[m, "year"].mode().iat[0],
            "modal_cgpa": background.loc[m, "cgpa"].mode().iat[0],
            "modal_living": background.loc[m, "living"].mode().iat[0],
            "pct_female": round(float(100 * (background.loc[m, "gender"] == "Female").mean()), 1),
            "pct_backlog": round(float(100 * background.loc[m, "backlog"].mean()), 1),
        })
    return pd.DataFrame(rows).set_index("cluster")


def discriminating_items(items, labels, top=5):
    """Rank items by how much of their variance the partition explains (eta-squared).

    Answers the proposal's second question - which features most strongly
    differentiate the clusters - with an effect size rather than an F-statistic,
    so the answer does not depend on n.
    """
    rows = []
    for it in C.ITEMS:
        groups = [items.loc[labels == c, it].to_numpy() for c in sorted(set(labels))]
        f, p = stats.f_oneway(*groups)
        grand = items[it].mean()
        ss_between = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
        ss_total = ((items[it] - grand) ** 2).sum()
        rows.append({
            "item": it, "label": C.ITEM_LABEL[it],
            "f": round(float(f), 2), "p": float(p),
            "eta_squared": round(float(ss_between / ss_total), 4),
        })
    df = pd.DataFrame(rows).set_index("item").sort_values("eta_squared", ascending=False)
    return df, df.index[:top].tolist()


def recommendations(names, detail, theme_pct):
    """Map each profile to the support action its defining items point at.

    Kept as an explicit item-to-action table so the recommendation is traceable
    to a measurement rather than to the author's intuition.
    """
    action = {
        "PileUp": "coursework scheduling / deadline spreading across courses",
        "SleepLoss": "workload audit on lab-report turnaround times",
        "LabStress": "lab preparation support and clearer marking rubrics",
        "ExamWorry": "exam-anxiety workshops and low-stakes practice assessment",
        "ResultDemotiv": "post-result advising contact, not just grade release",
        "CGPACompare": "de-emphasise public ranking; individual progress feedback",
        "AskTeacher": "structured, low-barrier office hours and named advisers",
        "Feedback": "feedback-quality training and turnaround-time targets",
        "Financial": "signpost hardship funds, stipends and paid on-campus work",
        "JobWorry": "early careers advising, internships, alumni contact",
        "SocioPol": "acknowledge context; general counselling availability",
        "MissMeal": "timetable review around meal windows; canteen hours",
    }
    out = {}
    for c, d in detail.items():
        acts = [action[e["item"]] for e in d["elevated_items"][:3] if e["item"] in action]
        acts += [action[e["item"]] for e in d["suppressed_items"][:1]
                 if e["item"] in action and e["item"] in ("AskTeacher", "Feedback")]
        top_themes = sorted(
            [(t, v) for t, v in theme_pct.get(int(c), {}).items() if t in LX.THEME_NAMES],
            key=lambda kv: -kv[1])[:3]
        out[int(c)] = {
            "profile": names[int(c)],
            "actions": acts or ["no distinctive elevated item; general provision applies"],
            "top_volunteered_themes": [{"theme": t, "pct": v} for t, v in top_themes],
        }
    return out
