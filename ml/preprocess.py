# -*- coding: utf-8 -*-
"""
Preprocessing: direction alignment, encoding and the standardised feature matrix.

Implements section 5.2 of the methodology report:
  * the two positively worded items are recoded 6 - x, so HIGH is always
    "more strain / less support" for all twelve items;
  * ordered background fields become ordinal integers, unordered ones one-hot;
  * clustering features are z-scored, because the item SDs range 0.97-1.46 and
    un-standardised Euclidean distance would let the two most polarised items
    dominate every centroid.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from . import config as C


def build_items(df, names):
    """Return the 12 Likert items, direction-aligned, with short column names.

    Raw (un-recoded) values are returned alongside because response-composition
    figures and the data sheet must show what students actually ticked.
    """
    raw = df[names["likert"]].copy()
    raw.columns = C.ITEMS
    raw = raw.astype(int)

    items = raw.copy()
    for c in C.REVERSED:
        items[c] = 6 - items[c]
    return items, raw


def build_background(df, names):
    """Assemble the background block with explicit ordinal / nominal treatment."""
    bg = pd.DataFrame(index=df.index)
    bg["year"] = df[names["year"]].astype(str)
    bg["cgpa"] = df[names["cgpa"]].astype(str)
    bg["gender"] = df[names["gender"]].astype(str)
    bg["living"] = df[names["living"]].astype(str)
    bg["department"] = df[names["department"]].astype(str)
    # The backlog item ships as a bilingual "Yes / হ্যাঁ" string; match on the
    # English stem so a change to the Bangla half cannot flip the encoding.
    bg["backlog"] = df[names["backlog"]].astype(str).str.strip().str.lower().str.startswith("yes").astype(int)

    bg["year_ord"] = bg["year"].map(C.YEAR_ORDINAL).fillna(0).astype(int)
    bg["cgpa_ord"] = bg["cgpa"].map(C.CGPA_ORDINAL).fillna(0).astype(int)
    return bg


def encode_background(bg, include_department=True, exclude=()):
    """One-hot the nominal background fields, keep the ordinals as integers.

    Used only for the mixed-data cross-check and the supervised extension. The
    primary clustering runs on the twelve items alone (methodology 5.1), so the
    background variables stay external and can serve as held-out validation.

    `exclude` drops encoded columns by name. It exists because a supervised run
    that predicts one background variable must not be handed that same variable
    as a feature: leaving `cgpa_ord` in while predicting the CGPA band produces
    a 94%-accurate model that has learned nothing.
    """
    nominal = ["gender", "living"] + (["department"] if include_department else [])
    enc = pd.get_dummies(bg[nominal], prefix=nominal, drop_first=False).astype(int)
    enc["year_ord"] = bg["year_ord"]
    enc["cgpa_ord"] = bg["cgpa_ord"]
    enc["backlog"] = bg["backlog"]

    drop = [c for c in enc.columns
            if c in exclude or any(c.startswith(p + "_") for p in exclude)]
    return enc.drop(columns=drop)


def standardise(items):
    """Z-score the item block; returns (Z, fitted_scaler)."""
    scaler = StandardScaler()
    Z = scaler.fit_transform(items.astype(float).to_numpy())
    return Z, scaler


def composite_score(items):
    """Mean of the twelve aligned items.

    Reported as a descriptive summary only. Section 3.4 of the dataset report
    established alpha = 0.61, which is below the threshold for treating this as
    a single measured construct, so it is never used as a modelling target.
    """
    return items.mean(axis=1)


def distribution(series, order=None):
    """Counts and percentages for one categorical column, in a fixed level order."""
    n = len(series)
    vc = series.value_counts()
    if order:
        keep = [o for o in order if o in vc.index]
        extra = [i for i in vc.index if i not in order]
        vc = vc.reindex(keep + extra)
    return pd.DataFrame({"n": vc, "pct": (100 * vc / n).round(1)})


def imbalance_report(bg):
    """Quantify the four sample skews the methodology report flags as limits."""
    g = bg["gender"].value_counts()
    d = bg["department"].value_counts()
    y = bg["year"].value_counts()
    n = len(bg)
    return {
        "gender_ratio": round(float(g.max() / max(g.min(), 1)), 2),
        "dept_max_min_ratio": round(float(d.max() / max(d.min(), 1)), 1),
        "dept_top3_pct": round(float(100 * d.nlargest(3).sum() / n), 1),
        "dept_below_25": int((d < 25).sum()),
        "backlog_minority_pct": round(float(100 * bg["backlog"].mean()), 1),
        "lower_years_pct": round(float(100 * (y.get("1st Year", 0) + y.get("2nd Year", 0)) / n), 1),
    }


def item_descriptives(items, raw):
    """Per-item table: raw distribution plus direction-aligned mean/SD/skew."""
    from scipy import stats

    rows = []
    for it in C.ITEMS:
        r = raw[it]
        a = items[it]
        counts = r.value_counts().reindex([1, 2, 3, 4, 5], fill_value=0)
        rows.append({
            "item": it,
            "q": C.ITEM_QNUM[it],
            "label": C.ITEM_LABEL[it],
            "positively_worded": it in C.REVERSED,
            "raw_mean": round(float(r.mean()), 3),
            "raw_sd": round(float(r.std()), 3),
            "raw_median": float(r.median()),
            "raw_skew": round(float(stats.skew(r)), 3),
            "pct_agree_raw": round(float(100 * (r >= 4).mean()), 1),
            "pct_disagree_raw": round(float(100 * (r <= 2).mean()), 1),
            "aligned_mean": round(float(a.mean()), 3),
            "aligned_sd": round(float(a.std()), 3),
            **{"n_%d" % v: int(counts[v]) for v in range(1, 6)},
        })
    return pd.DataFrame(rows).set_index("item")
