# -*- coding: utf-8 -*-
"""
University Student Mental Stress Pattern Analysis Using Unsupervised Learning
KUET, CSE 4112 Machine Learning Laboratory.

This module is the command-line mirror of `notebooks/stress_personas.ipynb`.
Both carry the same function bodies, so `python src/stress_personas.py` and a
top-to-bottom Run All of the notebook produce identical numbers.

What it does, in one paragraph. Twelve Likert stress items do not form one
scale (alpha = 0.61), so clustering them raw measures *severity* and collapses
to a high/low split. An EFA recovers four clean facets; clustering the facet
scores plus a CGPA ordinal recovers stress *shapes* instead. A pre-registered
rule (CLUSTERING_PLAN.md S4.A) picks k from a panel of criteria, one of which -
the profile differentiation index - measures the very failure mode that
produced the original two-group result. The free text is never a clustering
input; it is held back so theme-cluster agreement is genuine corroboration.

Run:  python src/stress_personas.py             (writes ./outputs)
      STRESS_DATA=/path/to/file.xlsx python src/stress_personas.py
"""
from __future__ import annotations

# === SECTION 1: Setup =======================================================
import glob
import hashlib
import json
import os
import re
import sys
import time
import warnings

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering
from sklearn.decomposition import NMF, PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score,
                             davies_bouldin_score, silhouette_samples,
                             silhouette_score)
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*force_all_finite.*")

RANDOM_STATE = 42
K_RANGE = list(range(2, 9))       # k = 2..8 is computed and shown for everything
K_CAP = 6                         # ...but only k <= 6 may be crowned (plan S7.3 Q4)
N_INIT = 25                       # k-means restarts
PCA_VAR_TARGET = 0.95
BOOTSTRAP_B = 200
CONSENSUS_B = 100
SEED_TRIALS = 20
GAP_REFS = 50
PARALLEL_SIMS = 500
HOLDOUT_FRAC = 0.30
MIN_CLUSTER_FRAC = 0.05           # plan S4.A rule 1
MIN_BOOTSTRAP_ARI = 0.50          # plan S4.A rule 2
MIN_DIFFERENTIATION = 0.50        # plan S4.A rule 3
DEGENERATE_FRAC = 0.80            # a linkage leaving >80% in one cluster is excluded

# --- design tokens: one visual language for all figures ---------------------
SURFACE, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"
MUTED, GRID, BASE = "#898781", "#e1e0d9", "#c3c2b7"
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
       "#008300", "#4a3aa7", "#e34948"]
DIV5 = ["#b02f2f", "#e88a89", "#cfcec7", "#86b6ef", "#1c5cab"]

ON_KAGGLE = os.path.isdir("/kaggle/input")
try:
    _HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:                 # running inside a notebook cell
    _HERE = os.getcwd()
ROOT = os.path.dirname(_HERE) if os.path.basename(_HERE) in ("src", "notebooks") else _HERE

OUT_DIR = "/kaggle/working/outputs" if ON_KAGGLE else os.path.join(ROOT, "outputs")
FIG_DIR = os.path.join(OUT_DIR, "figures")
TAB_DIR = os.path.join(OUT_DIR, "tables")
MODEL_DIR = os.path.join(OUT_DIR, "models")

RESULTS = {}                      # everything reported also lands in results.json
_FIG_N = [0]


def setup_style():
    """Apply the shared plot style. Called once, from both entry points."""
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE, "savefig.dpi": 150, "figure.dpi": 100,
        "font.family": ["DejaVu Sans"], "font.size": 9,
        "axes.edgecolor": BASE, "axes.labelcolor": INK, "axes.titlecolor": INK,
        "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlepad": 10,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7,
        "axes.axisbelow": True, "xtick.color": INK2, "ytick.color": INK2,
        "legend.frameon": False,
    })


def ensure_dirs():
    for d in (OUT_DIR, FIG_DIR, TAB_DIR, MODEL_DIR):
        os.makedirs(d, exist_ok=True)


def head(title, rule="="):
    print("\n" + rule * 78)
    print(title)
    print(rule * 78)


def savefig(fig, name, title=None):
    """Write a numbered figure and return its path.

    Numbering is sequential in execution order, so figure N in outputs/figures
    is figure N in the report - no manual bookkeeping between runs.
    """
    _FIG_N[0] += 1
    # Titles carry a FIGNUM placeholder that is resolved here, so the number in
    # the caption always matches the number in the filename no matter how the
    # section order changes.
    if getattr(fig, "_suptitle", None) is not None:
        fig._suptitle.set_text(fig._suptitle.get_text().replace("FIGNUM", str(_FIG_N[0])))
    for _ax in fig.axes:
        _t = _ax.get_title()
        if "FIGNUM" in _t:
            _ax.set_title(_t.replace("FIGNUM", str(_FIG_N[0])))
    fname = "fig%02d_%s.png" % (_FIG_N[0], name)
    path = os.path.join(FIG_DIR, fname)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print("  [fig %02d] %s%s" % (_FIG_N[0], fname, ("  " + title) if title else ""))
    return path


def savetab(df, name, show=True, index=True, fmt="%.3f"):
    """Write a results table as Excel-friendly CSV and echo it to the log."""
    path = os.path.join(TAB_DIR, name + ".csv")
    df.to_csv(path, index=index, encoding="utf-8-sig")
    if show:
        with pd.option_context("display.width", 220, "display.max_columns", 50,
                               "display.max_colwidth", 60,
                               "display.float_format", lambda v: fmt % v):
            print(df.to_string())
    print("  [table] %s.csv" % name)
    return df


def jsonable(obj):
    """Recursively convert numpy/pandas objects so json.dump never chokes."""
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return jsonable(obj.tolist())
    if isinstance(obj, (pd.Series, pd.Index)):
        return jsonable(list(obj))
    if isinstance(obj, pd.DataFrame):
        return jsonable(obj.reset_index().to_dict("records"))
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        v = float(obj)
        return None if np.isnan(v) else v
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, float):
        return None if np.isnan(obj) else obj
    if isinstance(obj, pd.Timestamp):
        return str(obj)
    return obj


# === SECTION 2: Load and schema check =======================================
# The Google Forms export uses the full bilingual question text as the column
# header, so columns are addressed positionally. That is fragile the moment the
# form is edited, hence the keyword fingerprint below: it aborts the run rather
# than letting a shifted column produce a plausible but wrong analysis.
COL_IDX = {"timestamp": 0, "year": 1, "cgpa": 2, "gender": 3, "living": 4,
           "backlog": 11, "open_current": 18, "open_previous": 19, "department": 20}
LIKERT_IDX = [5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17]

ITEMS = ["MissMeal", "PileUp", "SleepLoss", "LabStress", "ExamWorry", "ResultDemotiv",
         "CGPACompare", "AskTeacher", "Feedback", "Financial", "JobWorry", "SocioPol"]

ITEM_LABEL = {
    "MissMeal": "Miss meals because of classes",
    "PileUp": "Assignments / lab reports pile up",
    "SleepLoss": "Sacrifice sleep for coursework",
    "LabStress": "Lab work more stressful than theory",
    "ExamWorry": "Worry about results despite preparing",
    "ResultDemotiv": "One poor result kills motivation",
    "CGPACompare": "Constantly compare CGPA with peers",
    "AskTeacher": "Uncomfortable asking teachers for help (R)",
    "Feedback": "Instructor feedback does not reduce stress (R)",
    "Financial": "Financial concerns hurt performance",
    "JobWorry": "Worry about a job after graduation",
    "SocioPol": "Socio-economic / political instability",
}
#: Positively worded items, recoded 6 - x so HIGH always means MORE strain.
REVERSED = ["AskTeacher", "Feedback"]

SCHEMA_FINGERPRINT = {
    1: "year", 2: "cgpa", 3: "gender", 4: "living", 5: "breakfast", 6: "pile",
    7: "sleep", 8: "lab", 9: "poor result", 10: "motivation", 11: "backlog",
    12: "compare", 13: "comfortable", 14: "feedback", 15: "financial",
    16: "job", 17: "instability", 18: "biggest source", 19: "previous academic",
    20: "department",
}

YEAR_ORD = ["1st Year", "2nd Year", "3rd Year", "4th Year", "Other"]
CGPA_ORD = ["Below 2.50", "2.50–2.99", "3.00–3.49", "3.50–3.79", "3.80–4.00"]
GENDER_ORD = ["Male", "Female"]
LIVE_ORD = ["Hall", "Mess", "Family", "Other"]
YEAR_ORDINAL = {"1st Year": 1, "2nd Year": 2, "3rd Year": 3, "4th Year": 4, "Other": 0}
CGPA_ORDINAL = {c: i + 1 for i, c in enumerate(CGPA_ORD)}

DATA_GLOB = "*Academic Stress among Bangladeshi Engineering Students*.xlsx"


def resolve_data_path(explicit=None):
    """Locate the response workbook: explicit path -> Kaggle inputs -> repo root."""
    explicit = explicit or os.environ.get("STRESS_DATA")
    if explicit:
        if os.path.isfile(explicit):
            return explicit
        raise FileNotFoundError("Explicit data path does not exist: %s" % explicit)

    roots = []
    if ON_KAGGLE:
        roots += sorted(glob.glob("/kaggle/input/*"))
    roots += [ROOT, os.getcwd()]
    for base in roots:
        hits = sorted(glob.glob(os.path.join(base, DATA_GLOB)))
        if hits:
            return hits[0]
    raise FileNotFoundError(
        "Could not find the response workbook (%r) in: %s\n"
        "On Kaggle: attach the dataset via '+ Add Input'. Locally: set STRESS_DATA."
        % (DATA_GLOB, roots))


def validate_schema(df):
    """Assert the positional column map still matches the header text."""
    cols = list(df.columns)
    if len(cols) != 21:
        raise ValueError("Expected 21 columns in the export, found %d." % len(cols))
    problems = [
        "column %d (%r) does not contain %r" % (i, str(cols[i])[:60], cue)
        for i, cue in SCHEMA_FINGERPRINT.items()
        if cue.lower() not in str(cols[i]).lower()
    ]
    if problems:
        raise ValueError("Column mapping no longer matches the export:\n  "
                         + "\n  ".join(problems))
    print("  schema fingerprint OK - all %d mapped columns match their keyword"
          % len(SCHEMA_FINGERPRINT))
    return {"n_columns": len(cols), "columns": [str(c) for c in cols]}


def column_names(df):
    """Map logical names onto the real (long, bilingual) column labels."""
    cols = list(df.columns)
    names = {k: cols[i] for k, i in COL_IDX.items()}
    names["likert"] = [cols[i] for i in LIKERT_IDX]
    return names


def export_clean_csv(df, names, path):
    """Write a short-named CSV so everything downstream is portable.

    The project ships only the .xlsx; the rest of the pipeline (and WEKA) is far
    easier against a CSV whose headers are not 200-character bilingual sentences.
    """
    out = pd.DataFrame({
        "timestamp": df[names["timestamp"]].astype(str),
        "year": df[names["year"]].astype(str),
        "cgpa": df[names["cgpa"]].astype(str),
        "gender": df[names["gender"]].astype(str),
        "living": df[names["living"]].astype(str),
        "department": df[names["department"]].astype(str),
        "backlog_raw": df[names["backlog"]].astype(str),
    })
    for it, col in zip(ITEMS, names["likert"]):
        out[it + "_raw"] = df[col].astype(int)
    out["open_current"] = df[names["open_current"]].fillna("").astype(str)
    out["open_previous"] = df[names["open_previous"]].fillna("").astype(str)
    out.to_csv(path, index=False, encoding="utf-8-sig")
    print("  wrote %s  (%d rows x %d cols)" % (os.path.basename(path), out.shape[0], out.shape[1]))
    return out


def build_items(df, names):
    """Return the 12 Likert items direction-aligned, plus the raw copy.

    After alignment HIGH always means MORE strain / LESS support, so a centroid
    can be read straight off without checking each item's polarity.

    The recoding is applied here, at load time, because the audit and the EDA
    both need aligned items; S5 is where it is argued and where the scaling
    decision is made. The RAW copy is returned alongside and is what the
    response-style checks and the distribution figure use.
    """
    raw = df[names["likert"]].copy()
    raw.columns = ITEMS
    raw = raw.astype(int)
    items = raw.copy()
    for c in REVERSED:
        items[c] = 6 - items[c]
    return items, raw


def build_background(df, names):
    """Assemble the background block with explicit ordinal / nominal treatment."""
    bg = pd.DataFrame(index=df.index)
    for key in ("year", "cgpa", "gender", "living", "department"):
        bg[key] = df[names[key]].astype(str).str.strip()
    # The backlog item ships as a bilingual "Yes / হ্যাঁ" string; match the
    # English stem so an edit to the Bangla half cannot silently flip the code.
    bg["backlog"] = (df[names["backlog"]].astype(str).str.strip().str.lower()
                     .str.startswith("yes").astype(int))
    bg["year_ord"] = bg["year"].map(YEAR_ORDINAL).fillna(0).astype(int)
    bg["cgpa_ord"] = bg["cgpa"].map(CGPA_ORDINAL).fillna(0).astype(int)
    unmapped = int((bg["cgpa_ord"] == 0).sum())
    if unmapped:
        raise ValueError("%d CGPA values did not match the expected bands %s"
                         % (unmapped, CGPA_ORD))
    return bg


# === SECTION 3: Data quality audit ==========================================
# Written as checks that print numbers rather than assumptions that print
# nothing. "0 missing" is a finding and has to appear in the output.
def audit_quality(df, names):
    lik = df[names["likert"]]
    closed = [names[k] for k in ("year", "cgpa", "gender", "living", "backlog", "department")]
    ts = pd.to_datetime(df[names["timestamp"]], errors="coerce")
    by_day = ts.dt.date.value_counts()
    return {
        "n_rows": int(len(df)),
        "n_columns": int(df.shape[1]),
        "closed_ended_missing": int(df[closed].isna().to_numpy().sum()
                                    + lik.isna().to_numpy().sum()),
        "likert_missing": int(lik.isna().to_numpy().sum()),
        "open_current_missing": int(df[names["open_current"]].isna().sum()),
        "open_previous_missing": int(df[names["open_previous"]].isna().sum()),
        "exact_duplicate_rows": int(df.duplicated().sum()),
        "duplicate_ignoring_timestamp": int(
            df.drop(columns=[names["timestamp"]]).duplicated().sum()),
        "likert_out_of_range": int(((lik < 1) | (lik > 5)).to_numpy().sum()),
        "likert_non_integer": int((lik != lik.round()).to_numpy().sum()),
        "collection_start": str(ts.min()),
        "collection_end": str(ts.max()),
        "collection_days": int((ts.max() - ts.min()).days) + 1,
        "peak_day": str(by_day.idxmax()),
        "peak_day_n": int(by_day.max()),
    }


def response_style(raw):
    """Straight-lining / acquiescence diagnostics on the RAW item block.

    Deliberately raw, not direction-aligned. A respondent who ticked 5 twelve
    times is straight-lining, but reverse coding turns two of those 5s into 1s
    and the pattern disappears - measuring this on aligned items would report
    that no one straight-lined when someone did.

    A respondent who ticked the same box twelve times contributes a centroid-
    pulling point carrying no information. Counting them is the difference
    between "the data is clean" as a claim and as a measurement.
    """
    items = raw
    within_sd = items.std(axis=1)

    def longest_run(row):
        best = run = 1
        for a, b in zip(row[:-1], row[1:]):
            run = run + 1 if a == b else 1
            best = max(best, run)
        return best

    runs = items.apply(lambda r: longest_run(list(r)), axis=1)
    n = len(items)
    same_all = int((items.nunique(axis=1) == 1).sum())
    return {
        "straight_lining_n": same_all,
        "straight_lining_pct": round(100 * same_all / n, 2),
        "low_within_row_sd_n": int((within_sd < 0.35).sum()),
        "mean_within_row_sd": round(float(within_sd.mean()), 3),
        "min_within_row_sd": round(float(within_sd.min()), 3),
        "long_run_ge8_n": int((runs >= 8).sum()),
        "long_run_ge8_pct": round(100 * float((runs >= 8).mean()), 2),
        "acquiescence_pct_top_box": round(100 * float((items == 5).to_numpy().mean()), 1),
        "pct_using_all_five_points": round(100 * float((items.nunique(axis=1) == 5).mean()), 1),
    }


def quality_table(audit, style):
    """Table 1 - one row per check, with the verdict computed from the number."""
    rows = [
        ("Responses in export", audit["n_rows"], "n"),
        ("Columns in export", audit["n_columns"], "expected 21"),
        ("Missing values, closed-ended items", audit["closed_ended_missing"], "must be 0"),
        ("Missing values, 12 Likert items", audit["likert_missing"], "must be 0"),
        ("Blank Q18 (current stressor)", audit["open_current_missing"], "optional field"),
        ("Blank Q19 (previous years)", audit["open_previous_missing"], "optional field"),
        ("Exact duplicate rows", audit["exact_duplicate_rows"], "must be 0"),
        ("Duplicates ignoring timestamp", audit["duplicate_ignoring_timestamp"],
         "identical answer patterns"),
        ("Likert values outside 1-5", audit["likert_out_of_range"], "must be 0"),
        ("Non-integer Likert values", audit["likert_non_integer"], "must be 0"),
        ("Straight-lined responses (all 12 identical)", style["straight_lining_n"],
         "%.2f%% of n" % style["straight_lining_pct"]),
        ("Rows with within-row SD < 0.35", style["low_within_row_sd_n"], "low-variance responders"),
        ("Rows with a run of >= 8 identical answers", style["long_run_ge8_n"],
         "%.2f%% of n" % style["long_run_ge8_pct"]),
        ("Mean within-row SD", style["mean_within_row_sd"], "spread across the 12 items"),
        ("Top-box (5) rate across all answers", style["acquiescence_pct_top_box"],
         "%, acquiescence check"),
        ("Respondents using all five scale points", style["pct_using_all_five_points"],
         "% of n"),
        ("Collection window", audit["collection_days"], "days"),
    ]
    return pd.DataFrame(rows, columns=["check", "value", "note"]).set_index("check")


# === SECTION 4: Exploratory data analysis ===================================
def distribution(series, order=None):
    """Counts and percentages for one categorical column, in a fixed level order."""
    n = len(series)
    vc = series.value_counts()
    if order:
        keep = [o for o in order if o in vc.index]
        extra = [i for i in vc.index if i not in order]
        vc = vc.reindex(keep + extra)
    return pd.DataFrame({"n": vc, "pct": (100 * vc / n).round(1)})


def item_descriptives(items, raw):
    """Per-item table: raw distribution plus direction-aligned mean/SD/skew."""
    rows = []
    for it in ITEMS:
        r, a = raw[it], items[it]
        counts = r.value_counts().reindex([1, 2, 3, 4, 5], fill_value=0)
        rows.append({
            "item": it,
            "label": ITEM_LABEL[it],
            "reversed": it in REVERSED,
            "raw_mean": round(float(r.mean()), 3),
            "raw_sd": round(float(r.std()), 3),
            "raw_skew": round(float(stats.skew(r)), 3),
            "pct_agree": round(float(100 * (r >= 4).mean()), 1),
            "pct_disagree": round(float(100 * (r <= 2).mean()), 1),
            "pct_top_box": round(float(100 * (r == 5).mean()), 1),
            "aligned_mean": round(float(a.mean()), 3),
            "aligned_sd": round(float(a.std()), 3),
            **{"n_%d" % v: int(counts[v]) for v in range(1, 6)},
        })
    out = pd.DataFrame(rows).set_index("item")
    # A near-ceiling item carries little between-student variance but still sits
    # in the distance metric at full weight - the mechanism behind S2.3.
    out["ceiling_flag"] = (out["raw_mean"] >= 4.0) & (out["pct_top_box"] >= 45)
    return out


def fig_composition(bg):
    """Figure FIGNUM - who answered: the five background variables side by side."""
    specs = [("year", YEAR_ORD, "Year of study"), ("cgpa", CGPA_ORD, "CGPA band"),
             ("gender", GENDER_ORD, "Gender"), ("living", LIVE_ORD, "Living arrangement")]
    fig, axes = plt.subplots(1, 4, figsize=(13.5, 3.4))
    for ax, (col, order, title) in zip(axes, specs):
        d = distribution(bg[col], order)
        ax.barh(range(len(d))[::-1], d["pct"], color=CAT[0], height=0.66)
        ax.set_yticks(range(len(d))[::-1])
        ax.set_yticklabels([str(i) for i in d.index], fontsize=8)
        for y, (v, nn) in zip(range(len(d))[::-1], zip(d["pct"], d["n"])):
            ax.text(v + 1, y, "%.1f%% (%d)" % (v, nn), va="center", fontsize=7.5, color=INK2)
        ax.set_xlim(0, max(d["pct"]) * 1.38)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("% of respondents")
        ax.grid(axis="y", visible=False)
    fig.suptitle("Figure FIGNUM  Sample composition (n = %d)" % len(bg), y=1.04,
                 fontsize=12, fontweight="bold")
    return fig


def fig_departments(bg):
    """Figure FIGNUM - department spread, the sharpest imbalance in the sample."""
    d = distribution(bg["department"])
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.bar(range(len(d)), d["n"], color=CAT[0], width=0.68)
    ax.set_xticks(range(len(d)))
    ax.set_xticklabels(list(d.index), rotation=45, ha="right", fontsize=8)
    for x, (nn, p) in enumerate(zip(d["n"], d["pct"])):
        ax.text(x, nn + 4, "%d" % nn, ha="center", fontsize=7.5, color=INK2)
    top3 = d["pct"].nlargest(3).sum()
    ax.set_ylabel("respondents")
    ax.set_title("Figure FIGNUM  Responses by department  (top 3 departments = %.0f%% of the sample)"
                 % top3)
    ax.grid(axis="x", visible=False)
    return fig


def fig_likert_stacked(raw):
    """Figure FIGNUM - the raw response distribution, as students actually ticked it.

    Drawn on the RAW scale, not the aligned one: this figure documents the
    instrument, and reversing two items here would misreport what was answered.
    """
    order = list(raw.mean().sort_values().index)
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    left = np.zeros(len(order))
    for v in range(1, 6):
        vals = np.array([100 * (raw[it] == v).mean() for it in order])
        ax.barh(range(len(order)), vals, left=left, color=DIV5[v - 1],
                height=0.7, edgecolor=SURFACE, linewidth=0.6)
        for y, (val, l) in enumerate(zip(vals, left)):
            if val >= 6:
                ax.text(l + val / 2, y, "%.0f" % val, ha="center", va="center",
                        fontsize=7, color="white" if v in (1, 5) else INK)
        left += vals
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(["%s  %s" % (it, "(R)" if it in REVERSED else "")
                        for it in order], fontsize=8.5)
    ax.set_xlim(0, 100)
    ax.set_xlabel("% of respondents")
    ax.grid(axis="y", visible=False)
    ax.legend(handles=[Patch(facecolor=DIV5[i], label=l) for i, l in enumerate(
        ["1 strongly disagree", "2", "3 neutral", "4", "5 strongly agree"])],
        ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.11), fontsize=8)
    ax.set_title("Figure FIGNUM  Response distribution per item, raw scale\n"
                 "(R) = positively worded, reverse-coded downstream", fontsize=11)
    return fig


def fig_item_means(desc):
    """Figure FIGNUM - item means with SD whiskers; ceiling-effect items flagged."""
    d = desc.sort_values("raw_mean")
    fig, ax = plt.subplots(figsize=(9, 4.6))
    colors = [CAT[1] if f else CAT[0] for f in d["ceiling_flag"]]
    ax.errorbar(d["raw_mean"], range(len(d)), xerr=d["raw_sd"], fmt="none",
                ecolor=BASE, elinewidth=1.6, capsize=3)
    ax.scatter(d["raw_mean"], range(len(d)), s=46, color=colors, zorder=3)
    ax.set_yticks(range(len(d)))
    ax.set_yticklabels(list(d.index), fontsize=8.5)
    ax.axvline(3, color=MUTED, ls="--", lw=1)
    ax.set_xlim(1, 5.6)
    ax.set_xlabel("mean response (1-5) with +/- 1 SD")
    ax.grid(axis="y", visible=False)
    nflag = int(d["ceiling_flag"].sum())
    ax.legend(handles=[Patch(facecolor=CAT[1], label="ceiling effect (mean >= 4.0 and >= 45%% top-box), n = %d" % nflag),
                       Patch(facecolor=CAT[0], label="other items")],
              loc="lower right", fontsize=8)
    ax.set_title("Figure FIGNUM  Item means and dispersion\n"
                 "Near-ceiling items carry little between-student variance "
                 "but full weight in Euclidean distance", fontsize=11)
    return fig


# === SECTION 5: Preprocessing ===============================================
def standardisation_evidence(items):
    """Table 2 - show that standardising is necessary rather than assert it.

    Un-standardised Euclidean distance weights each item by its SD, so the two
    most polarised items would dominate every centroid. The SD range is the
    argument; it belongs in the output, not in a comment.
    """
    sd = items.std()
    rows = pd.DataFrame({
        "aligned_mean": items.mean().round(3),
        "aligned_sd": sd.round(3),
        "variance": (sd ** 2).round(3),
        "weight_vs_smallest": (sd ** 2 / (sd ** 2).min()).round(2),
    })
    return rows.sort_values("aligned_sd", ascending=False)


def standardise(frame):
    """Z-score a feature block; returns (Z, fitted_scaler)."""
    scaler = StandardScaler()
    Z = scaler.fit_transform(np.asarray(frame, dtype=float))
    return Z, scaler


# === SECTION 6: Measurement structure =======================================
# This runs BEFORE any clustering on purpose. If the twelve items do not hang
# together, a single stress score is not a valid target, and the whole framing
# has to change from "levels of one trait" to "profiles across facets".
def cronbach_alpha(frame):
    """Standard raw-score Cronbach's alpha, written out rather than imported."""
    k = frame.shape[1]
    if k < 2:
        return float("nan")
    item_var = frame.var(axis=0, ddof=1).sum()
    total_var = frame.sum(axis=1).var(ddof=1)
    return float((k / (k - 1)) * (1 - item_var / total_var))


def corrected_item_total(frame):
    """Correlation of each item with the sum of the *other* items.

    Corrected, i.e. the item is removed from the total it is compared against;
    the uncorrected version is inflated by the item correlating with itself.
    """
    total = frame.sum(axis=1)
    return {c: float(np.corrcoef(frame[c], total - frame[c])[0, 1]) for c in frame.columns}


def alpha_if_deleted(frame):
    """Alpha recomputed with each item dropped in turn."""
    return {c: round(cronbach_alpha(frame.drop(columns=[c])), 3) for c in frame.columns}


def bartlett_sphericity(frame):
    """Bartlett's test that the correlation matrix is not an identity matrix.

    A non-significant result would mean the items are mutually uncorrelated and
    no factor or PCA solution is meaningful.
    """
    R = np.corrcoef(frame.to_numpy(), rowvar=False)
    n, p = frame.shape
    det = max(float(np.linalg.det(R)), 1e-12)
    chi2 = -(n - 1 - (2 * p + 5) / 6) * np.log(det)
    dof = p * (p - 1) / 2
    return {"chi2": round(float(chi2), 2), "dof": int(dof),
            "p": float(stats.chi2.sf(chi2, dof)), "determinant": round(det, 6)}


def kmo(frame):
    """Kaiser-Meyer-Olkin sampling adequacy, overall and per item.

    KMO compares raw correlations against partial correlations. Below ~0.60 the
    items share too little common variance for a factor solution to be trusted -
    a number worth reporting whichever way it lands.
    """
    R = np.corrcoef(frame.to_numpy(), rowvar=False)
    inv = np.linalg.pinv(R)
    d = np.sqrt(np.diag(inv))
    partial = -inv / np.outer(d, d)
    np.fill_diagonal(partial, 0.0)
    Rc = R.copy()
    np.fill_diagonal(Rc, 0.0)
    r2, p2 = (Rc ** 2).sum(), (partial ** 2).sum()
    per = {c: round(float((Rc[i] ** 2).sum() / ((Rc[i] ** 2).sum() + (partial[i] ** 2).sum())), 3)
           for i, c in enumerate(frame.columns)}
    return {"overall": round(float(r2 / (r2 + p2)), 3), "per_item": per}


def measurement_report(items):
    """Table 3 plus the scalar structure statistics."""
    frame = items[ITEMS].astype(float)
    R = frame.corr()
    off = R.to_numpy()[~np.eye(len(ITEMS), dtype=bool)]
    it_total = corrected_item_total(frame)
    aid = alpha_if_deleted(frame)

    table = pd.DataFrame({
        "label": pd.Series(ITEM_LABEL),
        "corrected_item_total_r": pd.Series(it_total).round(3),
        "alpha_if_deleted": pd.Series(aid),
    }).loc[ITEMS]
    table["weak_item"] = table["corrected_item_total_r"] < 0.20

    stats_d = {
        "cronbach_alpha_12": round(cronbach_alpha(frame), 3),
        "mean_interitem_r": round(float(off.mean()), 3),
        "mean_abs_interitem_r": round(float(np.abs(off).mean()), 3),
        "max_interitem_r": round(float(off.max()), 3),
        "min_interitem_r": round(float(off.min()), 3),
        "bartlett": bartlett_sphericity(frame),
        "kmo": kmo(frame),
        "n_weak_items": int(table["weak_item"].sum()),
    }
    return table, stats_d, R


def measurement_verdict(s):
    """A threshold rule, so the conclusion cannot drift from its numbers."""
    a, k, r = s["cronbach_alpha_12"], s["kmo"]["overall"], s["mean_abs_interitem_r"]
    if a >= 0.70 and k >= 0.70:
        return ("The twelve items behave as one reasonably reliable scale "
                "(alpha = %.2f, KMO = %.2f); a single composite stress score is "
                "defensible." % (a, k))
    return ("The twelve items do NOT form a single reliable scale (alpha = %.2f, "
            "KMO = %.2f, mean |inter-item r| = %.2f). They are a checklist of partly "
            "independent stressors. Consequence for this project: clustering the 12 "
            "raw items measures overall agreement level, i.e. SEVERITY, which is "
            "exactly how the earlier run collapsed to k = 2. The facets recovered in "
            "S7 are the fix, not a decoration." % (a, k, r))


def fig_interitem_heatmap(R):
    """Figure FIGNUM - the inter-item correlation matrix, direction-aligned."""
    fig, ax = plt.subplots(figsize=(7.4, 6.4))
    M = R.loc[ITEMS, ITEMS].to_numpy()
    im = ax.imshow(M, cmap="RdBu_r", vmin=-0.6, vmax=0.6)
    ax.set_xticks(range(len(ITEMS)))
    ax.set_xticklabels(ITEMS, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(ITEMS)))
    ax.set_yticklabels(ITEMS, fontsize=8)
    for i in range(len(ITEMS)):
        for j in range(len(ITEMS)):
            if i != j:
                ax.text(j, i, "%.2f" % M[i, j], ha="center", va="center",
                        fontsize=6.2, color=INK if abs(M[i, j]) < 0.4 else "white")
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.78, label="Pearson r")
    ax.set_title("Figure FIGNUM  Inter-item correlations (aligned items)\n"
                 "Blocks of related items are visible; no single dominant factor",
                 fontsize=11)
    return fig


# === SECTION 7: Dimensionality reduction and factor structure ===============
def pca_report(Z, var_target=PCA_VAR_TARGET):
    """Eigenvalues, explained variance and the 95%-retention count."""
    p = PCA().fit(Z)
    ev = p.explained_variance_
    evr = p.explained_variance_ratio_
    cum = np.cumsum(evr)
    n_95 = int(np.searchsorted(cum, var_target) + 1)
    table = pd.DataFrame({
        "eigenvalue": np.round(ev, 4),
        "pct_variance": np.round(100 * evr, 2),
        "cumulative_pct": np.round(100 * cum, 2),
    }, index=["PC%d" % (i + 1) for i in range(len(ev))])
    return p, table, {"n_components_95pct": n_95,
                      "n_eigenvalues_above_1": int((ev > 1).sum()),
                      "pc1_pct": round(float(100 * evr[0]), 2),
                      "pc1_pc2_pct": round(float(100 * evr[:2].sum()), 2)}


def parallel_analysis(Z, n_sims=PARALLEL_SIMS, random_state=RANDOM_STATE):
    """Horn's parallel analysis: keep components beating random-data eigenvalues.

    Kaiser's "eigenvalue > 1" is known to over-retain. Horn's test compares each
    observed eigenvalue against the 95th percentile of the eigenvalue obtained
    from random data of the same shape, which is a far stricter and better
    calibrated bar.
    """
    rng = np.random.default_rng(random_state)
    n, p = Z.shape
    obs = np.linalg.eigvalsh(np.corrcoef(Z, rowvar=False))[::-1]
    sims = np.empty((n_sims, p))
    for i in range(n_sims):
        R = np.corrcoef(rng.standard_normal((n, p)), rowvar=False)
        sims[i] = np.linalg.eigvalsh(R)[::-1]
    mean_r = sims.mean(axis=0)
    p95_r = np.percentile(sims, 95, axis=0)
    retain = int((obs > p95_r).sum())
    table = pd.DataFrame({
        "observed": np.round(obs, 4),
        "random_mean": np.round(mean_r, 4),
        "random_p95": np.round(p95_r, 4),
        "retain": obs > p95_r,
    }, index=["PC%d" % (i + 1) for i in range(p)])
    return table, {"n_retain_parallel": retain, "n_sims": int(n_sims)}


def principal_axis_factoring(R, m, iters=200, tol=1e-7):
    """Iterated principal-axis factoring with SMC communality starts.

    PAF rather than PCA because the question is about *common* variance: which
    items share a latent facet. PCA puts total variance (common + unique) on the
    diagonal and so inflates loadings on items that mostly carry noise.
    """
    h2 = 1.0 - 1.0 / np.diag(np.linalg.pinv(R))     # squared multiple correlations
    L = None
    for _ in range(iters):
        Rs = R.copy()
        np.fill_diagonal(Rs, h2)
        w, v = np.linalg.eigh(Rs)
        idx = np.argsort(w)[::-1][:m]
        L = v[:, idx] * np.sqrt(np.clip(w[idx], 0.0, None))
        h2_new = (L ** 2).sum(axis=1)
        if np.max(np.abs(h2_new - h2)) < tol:
            h2 = h2_new
            break
        h2 = h2_new
    return L, h2


def varimax(L, tol=1e-6, iters=1000):
    """Varimax rotation with Kaiser normalisation.

    Unrotated factors are ordered by variance, not by interpretability: the first
    one is a general factor everything loads on. Varimax spins the axes to make
    each item load high on one factor and near zero on the rest, which is what
    turns a loading matrix into a set of nameable facets.
    """
    L = np.asarray(L, dtype=float)
    p, k = L.shape
    if k < 2:
        return L.copy()
    hn = np.sqrt((L ** 2).sum(axis=1, keepdims=True))
    hn[hn == 0] = 1.0
    X = L / hn                                       # Kaiser normalisation
    Rot = np.eye(k)
    d_old = 0.0
    for _ in range(iters):
        Lam = X @ Rot
        B = Lam ** 3 - Lam @ np.diag((Lam ** 2).sum(axis=0)) / p
        u, s, vt = np.linalg.svd(X.T @ B)
        Rot = u @ vt
        d = s.sum()
        if d_old != 0 and d < d_old * (1 + tol):
            break
        d_old = d
    return (X @ Rot) * hn                            # undo the normalisation


def efa(items, m=4, random_state=RANDOM_STATE):
    """Table 4 - the m-factor varimax solution over the aligned items."""
    frame = items[ITEMS].astype(float)
    R = np.corrcoef(frame.to_numpy(), rowvar=False)
    L, h2 = principal_axis_factoring(R, m)
    Lr = varimax(L)
    # Deterministic orientation and ordering, so labels are stable run to run:
    # flip each factor to a positive majority, then sort by variance explained.
    for j in range(Lr.shape[1]):
        if Lr[:, j].sum() < 0:
            Lr[:, j] *= -1
    order = np.argsort(-(Lr ** 2).sum(axis=0))
    Lr = Lr[:, order]
    cols = ["F%d" % (j + 1) for j in range(m)]
    load = pd.DataFrame(np.round(Lr, 3), index=ITEMS, columns=cols)
    load["communality"] = np.round((Lr ** 2).sum(axis=1), 3)
    load["assigned"] = load[cols].abs().idxmax(axis=1)
    load["primary_loading"] = load[cols].abs().max(axis=1).round(3)
    second = np.sort(np.abs(Lr), axis=1)[:, -2]
    load["secondary_loading"] = np.round(second, 3)
    # A cross-loading item belongs to no facet cleanly; flagging them is how the
    # "every item loads on exactly one factor" claim gets tested, not assumed.
    load["cross_loading"] = load["secondary_loading"] >= 0.30
    ssl = pd.Series(np.round((Lr ** 2).sum(axis=0), 3), index=cols)
    return load, ssl, {"n_factors": int(m),
                       "n_cross_loading_items": int(load["cross_loading"].sum()),
                       "ss_loadings": {c: float(ssl[c]) for c in cols},
                       "total_variance_explained_pct": round(float(100 * ssl.sum() / len(ITEMS)), 2)}


def fig_scree(pca_table, par_table):
    """Figure FIGNUM - scree plot with the Kaiser line and Horn's random baseline."""
    ev = pca_table["eigenvalue"].to_numpy()
    x = np.arange(1, len(ev) + 1)
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    ax.plot(x, ev, "-o", color=CAT[0], lw=1.8, ms=5, label="observed eigenvalue")
    ax.plot(x, par_table["random_p95"], "--s", color=CAT[1], lw=1.4, ms=4,
            label="Horn parallel analysis, 95th pct of random data")
    ax.axhline(1.0, color=MUTED, ls=":", lw=1.3, label="Kaiser criterion (eigenvalue = 1)")
    n_ret = int(par_table["retain"].sum())
    ax.axvline(n_ret + 0.5, color=CAT[2], lw=1.2, alpha=0.6)
    ax.text(n_ret + 0.65, ev.max() * 0.82, "retain %d" % n_ret, color=CAT[2], fontsize=9)
    ax.set_xticks(x)
    ax.set_xlabel("component")
    ax.set_ylabel("eigenvalue")
    ax.legend(fontsize=8)
    ax.set_title("Figure FIGNUM  Scree plot: Kaiser over-retains, Horn's test is the stricter bar")
    return fig


def fig_cumvar(pca_table, n_95):
    """Figure FIGNUM - cumulative explained variance and the 95% retention point."""
    cum = pca_table["cumulative_pct"].to_numpy()
    x = np.arange(1, len(cum) + 1)
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    ax.bar(x, pca_table["pct_variance"], color=BASE, width=0.62, label="% variance")
    ax.plot(x, cum, "-o", color=CAT[0], lw=1.8, ms=5, label="cumulative %")
    ax.axhline(95, color=CAT[1], ls="--", lw=1.3)
    ax.axvline(n_95, color=CAT[1], ls="--", lw=1.3)
    ax.text(n_95 + 0.15, 40, "%d components\nreach 95%%" % n_95, color=CAT[1], fontsize=9)
    ax.set_xticks(x)
    ax.set_xlabel("component")
    ax.set_ylabel("% of total variance")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=8, loc="center right")
    ax.set_title("Figure FIGNUM  Explained variance\n"
                 "No dominant component: variance is spread, which is the "
                 "signature of a multi-facet instrument", fontsize=11)
    return fig


def fig_pca_projection(Z, pca, colour, colour_name, order=None):
    """Figure FIGNUM - the sample in PC1-PC2 space, tinted by a background variable.

    Deliberately shown before any clustering: it is the honest picture of how
    little visible separation exists, and it sets expectations for silhouette.
    """
    P = pca.transform(Z)[:, :2]
    levels = order or list(pd.unique(colour))
    fig, ax = plt.subplots(figsize=(6.6, 5.6))
    for i, lv in enumerate(levels):
        m = (colour == lv).to_numpy()
        if m.sum() == 0:
            continue
        ax.scatter(P[m, 0], P[m, 1], s=13, alpha=0.62, color=CAT[i % len(CAT)],
                   label="%s (n=%d)" % (lv, int(m.sum())), linewidths=0)
    ax.set_xlabel("PC1 (%.1f%% of variance)" % (100 * pca.explained_variance_ratio_[0]))
    ax.set_ylabel("PC2 (%.1f%% of variance)" % (100 * pca.explained_variance_ratio_[1]))
    ax.legend(fontsize=8, markerscale=1.6)
    ax.set_title("Figure FIGNUM  Students in PC1-PC2 space, coloured by %s\n"
                 "One continuous cloud - the structure is a gradient, not islands"
                 % colour_name, fontsize=11)
    return fig


def fig_loadings(load, cols):
    """Figure FIGNUM - the rotated loading matrix as a heatmap."""
    M = load[cols].to_numpy()
    fig, ax = plt.subplots(figsize=(6.0, 6.0))
    im = ax.imshow(M, cmap="RdBu_r", vmin=-0.85, vmax=0.85, aspect="auto")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, fontsize=9)
    ax.set_yticks(range(len(load)))
    ax.set_yticklabels(list(load.index), fontsize=8.5)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            strong = abs(M[i, j]) >= 0.30
            ax.text(j, i, "%.2f" % M[i, j], ha="center", va="center",
                    fontsize=7.6, fontweight="bold" if strong else "normal",
                    color="white" if abs(M[i, j]) > 0.55 else (INK if strong else MUTED))
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.8, label="rotated loading")
    ax.set_title("Figure FIGNUM  Four-factor varimax solution\n"
                 "Bold = |loading| >= 0.30. Each item marks one facet", fontsize=11)
    return fig


# === SECTION 8: Subscale construction =======================================
# The facet MEMBERSHIP is discovered (argmax loading from the EFA). The facet
# NAME comes from a pre-registered reading of each item, applied to whichever
# item marks the factor. Naming a factor is always interpretive; doing it from
# the marker item keeps the interpretation deterministic and inspectable.
FACET_OF_ITEM = {
    "MissMeal": "Workload", "PileUp": "Workload", "SleepLoss": "Workload",
    "LabStress": "Workload",
    "ExamWorry": "Evaluation", "ResultDemotiv": "Evaluation", "CGPACompare": "Evaluation",
    "AskTeacher": "SupportGap", "Feedback": "SupportGap",
    "JobWorry": "FutureMacro", "SocioPol": "FutureMacro", "Financial": "FutureMacro",
}
FACET_GLOSS = {
    "Workload": "Workload & self-care sacrifice",
    "Evaluation": "Evaluation & performance anxiety",
    "SupportGap": "Teacher-support gap",
    "FutureMacro": "Future & macro insecurity",
    "Financial": "Financial pressure",
}


def name_factors(load, cols):
    """Name each factor after its highest-loading (marker) item's facet."""
    names, used = {}, set()
    # Strongest marker first, so if two factors point at the same facet the
    # better-defined one keeps the plain name.
    strength = sorted(cols, key=lambda c: -load[c].abs().max())
    for c in strength:
        marker = load[c].abs().idxmax()
        base = FACET_OF_ITEM[marker]
        name = base
        n = 2
        while name in used:
            name = "%s%d" % (base, n)
            n += 1
        used.add(name)
        names[c] = {"name": name, "marker_item": marker,
                    "marker_loading": round(float(load.loc[marker, c]), 3)}
    return names


def build_subscales(items, load, cols, factor_names, split_financial):
    """Mean-score each facet from its EFA-assigned items.

    Mean rather than sum so facets with different item counts stay on the same
    1-5 metric and stay comparable in a profile plot. `split_financial` produces
    the 5-subscale variant of plan S3.1: Financial is the most polarised item in
    the instrument and is central to the persona question, so it is also tested
    as a facet of its own.
    """
    spec = {}
    for c in cols:
        members = [it for it in ITEMS if load.loc[it, "assigned"] == c]
        spec[factor_names[c]["name"]] = members
    if split_financial:
        host = next((n for n, m in spec.items() if "Financial" in m), None)
        if host is None:
            raise ValueError("Financial is not assigned to any factor")
        rest = [m for m in spec[host] if m != "Financial"]
        if not rest:
            # Financial already stands alone; the 5-subscale variant is identical.
            return spec, pd.DataFrame({k: items[v].mean(axis=1) for k, v in spec.items()})
        spec[host] = rest
        spec["Financial"] = ["Financial"]
    scores = pd.DataFrame({k: items[v].mean(axis=1) for k, v in spec.items()},
                          index=items.index)
    return spec, scores


def subscale_report(spec, scores, items):
    """Table 5 - per-facet composition, reliability and inter-facet correlation."""
    rows = []
    for name, members in spec.items():
        frame = items[members].astype(float)
        a = cronbach_alpha(frame) if len(members) >= 2 else float("nan")
        # Spearman-Brown is the right reliability for a 2-item facet; alpha
        # understates it there, and reporting alpha alone would look worse than
        # the facet actually is.
        if len(members) == 2:
            r = float(frame.corr().iloc[0, 1])
            sb = 2 * r / (1 + r)
        else:
            sb = float("nan")
        rows.append({
            "subscale": name,
            "gloss": FACET_GLOSS.get(name, name),
            "n_items": len(members),
            "items": ", ".join(members),
            "alpha": round(a, 3) if a == a else np.nan,
            "spearman_brown": round(sb, 3) if sb == sb else np.nan,
            "mean": round(float(scores[name].mean()), 3),
            "sd": round(float(scores[name].std()), 3),
        })
    table = pd.DataFrame(rows).set_index("subscale")
    return table, scores.corr().round(3)


# === SECTION 9: Feature spaces A / B / C ====================================
# A  12 z-scored items          - the baseline that reproduces the old k = 2
# B  stress subscales only      - the methodology-report-faithful model
# C  subscales + CGPA ordinal   - the primary persona model
#
# Held out of EVERY feature space, so external validation survives: year,
# gender, living arrangement, department, backlog. CGPA enters space C only,
# and the cost of that (it can no longer validate space C) is stated in S18.
HELD_OUT = ["year", "gender", "living", "department", "backlog"]


def build_feature_spaces(items, scores4, scores5, spec4, spec5, bg):
    """Assemble the comparison set. Every space is z-scored on the same footing."""
    spaces = {}

    A = items[ITEMS].astype(float)
    spaces["A_items12"] = {
        "frame": A, "subscale_spec": {it: [it] for it in ITEMS},
        "label": "A - 12 z-scored items (old baseline)",
        "role": "baseline", "includes_cgpa": False}

    for tag, sc, sp in (("B_subscales4", scores4, spec4), ("B5_subscales5", scores5, spec5)):
        spaces[tag] = {
            "frame": sc.copy(), "subscale_spec": sp,
            "label": "B - %d stress subscales (report-faithful)" % sc.shape[1],
            "role": "report-faithful", "includes_cgpa": False}

    for tag, sc, sp in (("C_subscales4_cgpa", scores4, spec4),
                        ("C5_subscales5_cgpa", scores5, spec5)):
        f = sc.copy()
        f["CGPA_ord"] = bg["cgpa_ord"].astype(float)
        spaces[tag] = {
            "frame": f, "subscale_spec": sp,
            "label": "C - %d subscales + CGPA ordinal (persona model)" % sc.shape[1],
            "role": "persona", "includes_cgpa": True}

    for tag, d in spaces.items():
        Z, scaler = standardise(d["frame"])
        d["Z"], d["scaler"], d["columns"] = Z, scaler, list(d["frame"].columns)
    return spaces


def choose_primary_space(ev_by_space, cap=K_CAP):
    """Pick the headline feature space on a STABILITY criterion (plan S3.1).

    The 4-subscale and 5-subscale groupings are both defensible readings of the
    same EFA. Financial loads on the future/macro factor, but it is the most
    polarised item in the instrument (see the SD table in S5) and central to the
    research question, so splitting it into a facet of its own is a live option.
    Rather than assert one, both are built and the rule decides.

    Criterion, fixed before the run: each space is judged AT THE k IT WOULD
    ACTUALLY SHIP - the k the S4.A rule crowns for that space - and the space
    whose shipped solution is more reproducible under resampling wins. Ties
    within 0.01 ARI go to the higher profile differentiation.

    Judging at the shipped k rather than averaging over k = 2..cap matters: an
    average includes k values the rule would discard for either space, and it
    systematically flatters the lower-dimensional space, since fewer dimensions
    make any partition easier to reproduce. That would decide the question by
    counting features rather than by measuring stability.
    """
    rows = {}
    for tag, ev in ev_by_space.items():
        t, k, _ = screen_and_vote(ev, cap)
        rows[tag] = {
            "crowned_k": int(k),
            "bootstrap_ari_at_k": round(float(ev.loc[k, "bootstrap_ari"]), 4),
            "differentiation_at_k": round(float(ev.loc[k, "differentiation"]), 4),
            "silhouette_at_k": round(float(ev.loc[k, "silhouette"]), 4),
            "min_size_frac_at_k": round(float(ev.loc[k, "min_size_frac"]), 4),
            "n_survivors": int(t["survives"].sum()),
        }
    table = pd.DataFrame(rows).T
    best = table["bootstrap_ari_at_k"].max()
    tied = table[table["bootstrap_ari_at_k"] >= best - 0.01]
    chosen = str(tied["differentiation_at_k"].idxmax()) if len(tied) > 1         else str(table["bootstrap_ari_at_k"].idxmax())
    return chosen, table


# === SECTION 10: Train / holdout split ======================================
def strain_tertile(items):
    """Coarse overall-strain tertile, used ONLY to stratify the split.

    Stratifying on the mean aligned score keeps the severity mix identical in
    both halves, so a holdout difference cannot be dismissed as one half simply
    being more stressed. It is never a feature and never a target.
    """
    comp = items[ITEMS].astype(float).mean(axis=1)
    return pd.qcut(comp, 3, labels=["low", "mid", "high"]).astype(str)


def train_holdout(index, strat, frac=HOLDOUT_FRAC, random_state=RANDOM_STATE):
    """70/30 stratified split. All model selection happens on train only."""
    tr, te = train_test_split(np.asarray(index), test_size=frac,
                              random_state=random_state, stratify=np.asarray(strat))
    return np.sort(tr), np.sort(te)


# === SECTION 11: The k sweep ================================================
ALGORITHMS = ["kmeans", "gmm_full", "gmm_diag", "ward", "complete", "average", "spectral"]


def fit_algorithm(algo, Z, k, random_state=RANDOM_STATE):
    """Fit one clusterer and return (labels, extras).

    Six families are run, not one, because agreement between algorithms that
    make different assumptions is evidence the partition is in the data rather
    than in k-means' preference for round, equal-sized blobs.
    """
    extras = {}
    if algo == "kmeans":
        m = KMeans(n_clusters=k, n_init=N_INIT, random_state=random_state).fit(Z)
        labels, extras["inertia"] = m.labels_, float(m.inertia_)
    elif algo in ("gmm_full", "gmm_diag"):
        cov = "full" if algo == "gmm_full" else "diag"
        m = GaussianMixture(n_components=k, covariance_type=cov, n_init=5,
                            random_state=random_state, reg_covar=1e-5).fit(Z)
        labels = m.predict(Z)
        resp = m.predict_proba(Z)
        extras["bic"] = float(m.bic(Z))
        extras["aic"] = float(m.aic(Z))
        extras["entropy"] = float(normalised_entropy(resp))
        extras["converged"] = bool(m.converged_)
    elif algo in ("ward", "complete", "average"):
        m = AgglomerativeClustering(n_clusters=k, linkage=algo).fit(Z)
        labels = m.labels_
    elif algo == "spectral":
        m = SpectralClustering(n_clusters=k, affinity="nearest_neighbors",
                               n_neighbors=15, assign_labels="kmeans",
                               random_state=random_state, n_init=10)
        labels = m.fit_predict(Z)
    else:
        raise ValueError("unknown algorithm %r" % algo)
    return np.asarray(labels), extras


def normalised_entropy(resp):
    """GMM classification entropy, normalised to [0, 1]; 1 = crisp assignment.

    A mixture can fit well and still assign everyone 50/50 between components.
    Entropy catches that; BIC does not.
    """
    r = np.clip(resp, 1e-12, 1.0)
    n, k = r.shape
    if k < 2:
        return 1.0
    ent = -(r * np.log(r)).sum()
    return float(1.0 - ent / (n * np.log(k)))


def internal_indices(Z, labels):
    """Silhouette, Davies-Bouldin, Calinski-Harabasz - all undefined at k = 1."""
    uniq = len(np.unique(labels))
    if uniq < 2 or uniq >= len(labels):
        return {"silhouette": np.nan, "davies_bouldin": np.nan, "calinski_harabasz": np.nan}
    return {
        "silhouette": float(silhouette_score(Z, labels)),
        "davies_bouldin": float(davies_bouldin_score(Z, labels)),
        "calinski_harabasz": float(calinski_harabasz_score(Z, labels)),
    }


def cluster_sizes(labels):
    v = pd.Series(labels).value_counts()
    n = len(labels)
    return {"n_clusters_found": int(len(v)),
            "min_size": int(v.min()), "max_size": int(v.max()),
            "min_size_frac": float(v.min() / n), "max_size_frac": float(v.max() / n)}


def profile_differentiation(Z, labels):
    """Share of between-cluster variation that is SHAPE rather than LEVEL.

    This is the criterion that measures the exact failure mode of the earlier
    run. Silhouette asks only whether clusters are separated; it cannot tell a
    set of personas from a set of severity levels. Decompose each centroid's
    deviation from the grand centroid into

        level  - how high it sits on average across all dimensions
                 ("this group is stressed MORE")
        shape  - each dimension's departure from that centroid's own level
                 ("this group is stressed DIFFERENTLY")

    The two parts are orthogonal (shape sums to zero across dimensions, level is
    constant across them), so the between-cluster sum of squares splits cleanly
    and the ratio is a genuine variance share, not a heuristic.
    """
    labels = np.asarray(labels)
    n, d = Z.shape
    grand = Z.mean(axis=0)
    ss_level = ss_shape = 0.0
    for c in np.unique(labels):
        idx = labels == c
        nc = int(idx.sum())
        delta = Z[idx].mean(axis=0) - grand
        lev = float(delta.mean())
        shape = delta - lev
        ss_level += nc * d * lev ** 2
        ss_shape += nc * float((shape ** 2).sum())
    total = ss_level + ss_shape
    if total <= 0:
        return {"differentiation": np.nan, "ss_shape": 0.0, "ss_level": 0.0}
    return {"differentiation": float(ss_shape / total),
            "ss_shape": float(ss_shape), "ss_level": float(ss_level),
            "between_ss": float(total)}


def sweep(spaces, ks=K_RANGE, algos=ALGORITHMS, subset=None):
    """Table 6 - every feature space x algorithm x k, with a degeneracy screen."""
    rows, excluded = [], []
    for tag, d in spaces.items():
        Z = d["Z"] if subset is None else d["Z"][subset]
        for algo in algos:
            for k in ks:
                try:
                    labels, extras = fit_algorithm(algo, Z, k)
                except Exception as exc:            # keep the sweep going
                    excluded.append({"space": tag, "algorithm": algo, "k": k,
                                     "reason": "failed: %s" % type(exc).__name__})
                    continue
                sz = cluster_sizes(labels)
                degenerate = sz["max_size_frac"] > DEGENERATE_FRAC
                row = {"space": tag, "algorithm": algo, "k": k,
                       **internal_indices(Z, labels), **sz,
                       **profile_differentiation(Z, labels),
                       "bic": extras.get("bic", np.nan),
                       "entropy": extras.get("entropy", np.nan),
                       "inertia": extras.get("inertia", np.nan),
                       "degenerate": degenerate}
                rows.append(row)
                if degenerate:
                    excluded.append({
                        "space": tag, "algorithm": algo, "k": k,
                        "reason": "degenerate: largest cluster holds %.1f%% of n (> %.0f%%)"
                                  % (100 * sz["max_size_frac"], 100 * DEGENERATE_FRAC)})
    table = pd.DataFrame(rows)
    return table, pd.DataFrame(excluded)


def fig_elbow(table, space_tag):
    """Figure FIGNUM - k-means SSE with the kneedle-style elbow marker."""
    d = table[(table.space == space_tag) & (table.algorithm == "kmeans")].sort_values("k")
    k, sse = d["k"].to_numpy(), d["inertia"].to_numpy()
    knee = kneedle(k, sse)
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.plot(k, sse, "-o", color=CAT[0], lw=1.8, ms=5)
    ax.axvline(knee, color=CAT[1], ls="--", lw=1.4)
    ax.text(knee + 0.08, sse.max() * 0.9, "elbow at k = %d" % knee, color=CAT[1], fontsize=9)
    ax.set_xticks(k)
    ax.set_xlabel("k")
    ax.set_ylabel("within-cluster SSE")
    ax.set_title("Figure FIGNUM  Elbow curve, k-means on space %s" % space_tag)
    return fig, int(knee)


def kneedle(x, y):
    """Maximum-distance knee: the point furthest from the chord of the curve.

    Reading an elbow by eye is the classic place where a k-selection quietly
    becomes a preference. This makes it arithmetic.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    xn = (x - x.min()) / max(np.ptp(x), 1e-12)
    yn = (y - y.min()) / max(np.ptp(y), 1e-12)
    p0, p1 = np.array([xn[0], yn[0]]), np.array([xn[-1], yn[-1]])
    v = p1 - p0
    v = v / max(np.linalg.norm(v), 1e-12)
    dist = [np.linalg.norm((np.array([a, b]) - p0) - np.dot(np.array([a, b]) - p0, v) * v)
            for a, b in zip(xn, yn)]
    return int(x[int(np.argmax(dist))])


def fig_index_panels(table, spaces, algo="kmeans"):
    """Figure FIGNUM - silhouette / DB / CH / differentiation across all spaces.

    Silhouette peaking at k = 2 in EVERY space is the single most important
    thing this panel shows: an unweighted vote would have re-crowned k = 2.
    """
    metrics = [("silhouette", "Silhouette (higher better)", False),
               ("davies_bouldin", "Davies-Bouldin (lower better)", True),
               ("calinski_harabasz", "Calinski-Harabasz (higher better)", False),
               ("differentiation", "Profile differentiation, share shape (higher better)", False)]
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 7.4))
    tags = list(spaces)
    for ax, (col, title, lower_better) in zip(axes.ravel(), metrics):
        for i, tag in enumerate(tags):
            d = table[(table.space == tag) & (table.algorithm == algo)].sort_values("k")
            if d.empty:
                continue
            y = d[col].to_numpy()
            ax.plot(d["k"], y, "-o", ms=4, lw=1.6, color=CAT[i % len(CAT)], label=tag)
            best = d["k"].to_numpy()[int(np.nanargmin(y) if lower_better else np.nanargmax(y))]
            ax.scatter([best], [y[list(d["k"]).index(best)]], s=90, facecolors="none",
                       edgecolors=CAT[i % len(CAT)], linewidths=1.6, zorder=4)
        if col == "differentiation":
            ax.axhline(MIN_DIFFERENTIATION, color=CAT[7], ls="--", lw=1.2)
            ax.text(K_RANGE[0], MIN_DIFFERENTIATION + 0.015, "rule 3 floor = 50%",
                    color=CAT[7], fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("k")
        ax.set_xticks(K_RANGE)
    axes.ravel()[0].legend(fontsize=7.6, ncol=1)
    fig.suptitle("Figure FIGNUM  Validity indices by k, %s, all feature spaces" % algo,
                 y=0.99, fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def fig_sizes(table, space_tag, algo="kmeans"):
    """Figure FIGNUM - smallest-cluster share against the 5% usability floor."""
    d = table[(table.space == space_tag) & (table.algorithm == algo)].sort_values("k")
    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    cols = [CAT[2] if f >= MIN_CLUSTER_FRAC else CAT[7] for f in d["min_size_frac"]]
    ax.bar(d["k"], 100 * d["min_size_frac"], color=cols, width=0.6)
    ax.axhline(100 * MIN_CLUSTER_FRAC, color=CAT[7], ls="--", lw=1.3)
    ax.text(K_RANGE[0] - 0.35, 100 * MIN_CLUSTER_FRAC + 0.6,
            "rule 1 floor = 5% of n", color=CAT[7], fontsize=8)
    for x, v, nn in zip(d["k"], 100 * d["min_size_frac"], d["min_size"]):
        ax.text(x, v + 0.5, "%.1f%%\n(n=%d)" % (v, nn), ha="center", fontsize=7.4, color=INK2)
    ax.set_xticks(K_RANGE)
    ax.set_xlabel("k")
    ax.set_ylabel("smallest cluster, % of n")
    ax.set_title("Figure FIGNUM  Smallest cluster size by k, %s on %s" % (algo, space_tag))
    return fig


# --- Per-k evidence, and the screens that read it -------------------------
# These live with the sweep rather than with the decision in S14: they are
# what PRODUCES the evidence, and the primary feature space is chosen from
# that evidence at the end of this section. S13 keeps the diagnostics that
# are reported but do not feed the vote; S14 keeps the rule itself.

def bootstrap_stability(Z, k, reference_labels, b=BOOTSTRAP_B,
                        random_state=RANDOM_STATE, sample_frac=0.80):
    """Re-cluster subsamples and compare to the reference on shared rows (ARI).

    Subsampling without replacement rather than a true bootstrap: duplicated
    points would inflate agreement, since a duplicate is trivially assigned with
    its twin. A partition that dissolves under resampling is a property of one
    sample draw, not of the population.
    """
    rng = np.random.default_rng(random_state)
    n = Z.shape[0]
    size = int(round(sample_frac * n))
    aris = []
    for _ in range(b):
        idx = rng.choice(n, size=size, replace=False)
        lab = KMeans(n_clusters=k, n_init=10,
                     random_state=int(rng.integers(1 << 31))).fit_predict(Z[idx])
        aris.append(adjusted_rand_score(np.asarray(reference_labels)[idx], lab))
    a = np.asarray(aris)
    return {"n_resamples": int(b), "sample_fraction": sample_frac,
            "ari_mean": float(a.mean()), "ari_sd": float(a.std()),
            "ari_p05": float(np.percentile(a, 5)), "ari_median": float(np.median(a)),
            "ari_p95": float(np.percentile(a, 95)),
            "pct_above_0.5": float(100 * (a > 0.5).mean())}


def cross_algorithm_ari(Z, k, algos=ALGORITHMS):
    """Do algorithms with different assumptions find the same partition?"""
    labs, ok = {}, []
    for a in algos:
        try:
            lb, _ = fit_algorithm(a, Z, k)
        except Exception:
            continue
        if pd.Series(lb).value_counts().max() / len(lb) > DEGENERATE_FRAC:
            continue                       # a degenerate linkage would flatter nobody
        labs[a] = lb
        ok.append(a)
    M = pd.DataFrame(np.eye(len(ok)), index=ok, columns=ok)
    vals = []
    for i, a in enumerate(ok):
        for bnm in ok[i + 1:]:
            v = adjusted_rand_score(labs[a], labs[bnm])
            M.loc[a, bnm] = M.loc[bnm, a] = v
            vals.append(v)
    return M.round(3), (float(np.mean(vals)) if vals else np.nan), labs


def kselect_evidence(space, ks=K_RANGE, subset=None):
    """Assemble every criterion the S4.A rule consults, per k."""
    Z = space["Z"] if subset is None else space["Z"][subset]
    rows, stab_rows = [], []
    for k in ks:
        km_labels, km_extra = fit_algorithm("kmeans", Z, k)
        gm_labels, gm_extra = fit_algorithm("gmm_full", Z, k)
        idx = internal_indices(Z, km_labels)
        sz = cluster_sizes(km_labels)
        dif = profile_differentiation(Z, km_labels)
        boot = bootstrap_stability(Z, k, km_labels)
        _, xari, _ = cross_algorithm_ari(Z, k)
        stab_rows.append({"k": k, **boot})
        rows.append({
            "k": k,
            "silhouette": idx["silhouette"],
            "davies_bouldin": idx["davies_bouldin"],
            "calinski_harabasz": idx["calinski_harabasz"],
            "bic_gmm": gm_extra.get("bic", np.nan),
            "entropy_gmm": gm_extra.get("entropy", np.nan),
            "bootstrap_ari": boot["ari_mean"],
            "cross_algo_ari": xari,
            "differentiation": dif["differentiation"],
            "min_size_frac": sz["min_size_frac"],
            "min_size": sz["min_size"],
            "inertia": km_extra.get("inertia", np.nan),
        })
    return pd.DataFrame(rows).set_index("k"), stab_rows


#: Criteria in the quality vote (S4.A rule 6), with their direction.
VOTE_CRITERIA = [("silhouette", "high"), ("davies_bouldin", "low"),
                 ("calinski_harabasz", "high"), ("bic_gmm", "low"),
                 ("bootstrap_ari", "high"), ("cross_algo_ari", "high"),
                 ("differentiation", "high")]


def screen_and_vote(ev, cap=K_CAP):
    """Rules 1-4 (usability screens), then rule 6-7 (quality vote among survivors).

    Split out of choose_k so the same rule can judge a candidate FEATURE SPACE at
    the k it would actually ship, rather than at an average over k values the
    rule would have discarded anyway.
    """
    t = ev.copy()
    reasons = {}
    for k in t.index:
        why = []
        if t.loc[k, "min_size_frac"] < MIN_CLUSTER_FRAC:
            why.append("rule 1: smallest cluster %.1f%% < %.0f%%"
                       % (100 * t.loc[k, "min_size_frac"], 100 * MIN_CLUSTER_FRAC))
        if t.loc[k, "bootstrap_ari"] < MIN_BOOTSTRAP_ARI:
            why.append("rule 2: bootstrap ARI %.3f < %.2f"
                       % (t.loc[k, "bootstrap_ari"], MIN_BOOTSTRAP_ARI))
        if t.loc[k, "differentiation"] < MIN_DIFFERENTIATION:
            why.append("rule 3: differentiation %.1f%% < %.0f%% (severity split, not personas)"
                       % (100 * t.loc[k, "differentiation"], 100 * MIN_DIFFERENTIATION))
        if k > cap:
            why.append("rule 4: k > %d actionability cap" % cap)
        reasons[k] = "; ".join(why)
    t["survives"] = [reasons[k] == "" for k in t.index]
    t["discarded_because"] = [reasons[k] or "-" for k in t.index]

    surv = t[t["survives"]]
    ranks = pd.DataFrame(index=t.index)
    for col, direction in VOTE_CRITERIA:
        ranks[col + "_rank"] = surv[col].rank(ascending=(direction == "low"), method="average")
    t["mean_rank"] = ranks.mean(axis=1)

    if surv.empty:
        return t, int(t.index.min()), "no k survived the screens; falling back to the smallest k"
    best = t.loc[t["survives"], "mean_rank"].min()
    tied = [int(k) for k in t.index if t.loc[k, "survives"] and t.loc[k, "mean_rank"] == best]
    chosen = min(tied)                                      # rule 7: parsimony
    note = ("chosen by mean rank across %d criteria among survivors" % len(VOTE_CRITERIA)
            + (" (tie with k = %s, smaller k preferred)" % tied if len(tied) > 1 else ""))
    return t, chosen, note


# === SECTION 12: The gap statistic ==========================================
def gap_statistic(Z, ks=None, n_refs=GAP_REFS, random_state=RANDOM_STATE):
    """Tibshirani's gap statistic, evaluated from k = 1 upward.

    The only criterion here that can answer "is there any structure at all",
    because it alone admits k = 1 as a candidate. Silhouette, Davies-Bouldin and
    Calinski-Harabasz are undefined at k = 1 and therefore cannot report the
    absence of clusters - they only ever rank the partitions they are asked for.

    Reference data is drawn uniformly over the bounding box of the principal
    components, which is the standard "one homogeneous blob" null. Selection is
    Tibshirani's 1-SE rule: the smallest k whose gap is within one standard
    error of the next k's.
    """
    ks = list(ks or ([1] + list(K_RANGE)))
    rng = np.random.default_rng(random_state)

    def dispersion(X, k):
        if k == 1:
            return float(((X - X.mean(axis=0, keepdims=True)) ** 2).sum())
        return float(KMeans(n_clusters=k, n_init=10, random_state=random_state).fit(X).inertia_)

    p = PCA().fit(Z)
    Zr = Z @ p.components_.T
    lo, hi = Zr.min(axis=0), Zr.max(axis=0)

    obs_log, ref_mean, ref_sd = [], [], []
    for k in ks:
        obs_log.append(np.log(dispersion(Z, k)))
        refs = [np.log(dispersion(rng.uniform(lo, hi, size=Z.shape) @ p.components_, k))
                for _ in range(n_refs)]
        ref_mean.append(float(np.mean(refs)))
        ref_sd.append(float(np.std(refs)))

    gap = np.array(ref_mean) - np.array(obs_log)
    sk = np.array(ref_sd) * np.sqrt(1 + 1.0 / n_refs)

    chosen = ks[-1]
    for i in range(len(ks) - 1):
        if gap[i] >= gap[i + 1] - sk[i + 1]:
            chosen = ks[i]
            break
    return {
        "ks": [int(x) for x in ks],
        "gap": [float(g) for g in gap],
        "s_k": [float(s) for s in sk],
        "k_selected": int(chosen),
        "n_references": int(n_refs),
        "supports_no_structure": bool(chosen == 1),
        "interpretation": (
            "the gap statistic selects k = 1: the data is better described as one "
            "homogeneous group than as any partition tested"
            if chosen == 1 else
            "the gap statistic selects k = %d over the single-group null" % chosen),
    }


def fig_gap(gap, space_tag):
    """Figure FIGNUM - the gap curve with 1-SE bars."""
    ks, g, s = np.array(gap["ks"]), np.array(gap["gap"]), np.array(gap["s_k"])
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.errorbar(ks, g, yerr=s, fmt="-o", color=CAT[0], ecolor=BASE,
                elinewidth=1.5, capsize=3, lw=1.8, ms=5)
    ax.axvline(gap["k_selected"], color=CAT[1], ls="--", lw=1.4)
    ax.text(gap["k_selected"] + 0.1, g.min() + 0.02 * max(np.ptp(g), 1e-9),
            "1-SE rule selects k = %d" % gap["k_selected"], color=CAT[1], fontsize=9)
    ax.set_xticks(ks)
    ax.set_xlabel("k  (k = 1 is the 'no clusters' null)")
    ax.set_ylabel("gap")
    ax.set_title("Figure FIGNUM  Gap statistic on %s, %d uniform reference draws"
                 % (space_tag, gap["n_references"]))
    return fig


# === SECTION 13: Stability ==================================================
def seed_stability(Z, k, trials=SEED_TRIALS):
    """Pairwise ARI across k-means runs from different seeds.

    A narrower question than the bootstrap: given this exact sample, is the
    solution an artefact of initialisation?
    """
    labs = [KMeans(n_clusters=k, n_init=10, random_state=s).fit_predict(Z)
            for s in range(trials)]
    pair = np.asarray([adjusted_rand_score(labs[i], labs[j])
                       for i in range(trials) for j in range(i + 1, trials)])
    return {"n_seeds": int(trials), "pairwise_ari_mean": float(pair.mean()),
            "pairwise_ari_min": float(pair.min()),
            "pct_identical": float(100 * (pair > 0.99).mean())}


def consensus_matrix(Z, k, b=CONSENSUS_B, random_state=RANDOM_STATE, sample_frac=0.80):
    """Pairwise co-assignment rate across resamples."""
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
    """Mean within-cluster co-assignment: which profiles are trustworthy."""
    out = {}
    for c in sorted(set(labels)):
        idx = np.flatnonzero(np.asarray(labels) == c)
        if len(idx) < 2:
            out[int(c)] = None
            continue
        sub = M[np.ix_(idx, idx)]
        iu = np.triu_indices(len(idx), 1)
        out[int(c)] = round(float(np.nanmean(sub[iu])), 3)
    return out


def fig_stability_by_k(stab_rows):
    """Figure FIGNUM - bootstrap ARI by k with the reproducibility floor."""
    d = pd.DataFrame(stab_rows)
    fig, ax = plt.subplots(figsize=(7.4, 4.1))
    ax.errorbar(d["k"], d["ari_mean"],
                yerr=[d["ari_mean"] - d["ari_p05"], d["ari_p95"] - d["ari_mean"]],
                fmt="-o", color=CAT[0], ecolor=BASE, elinewidth=1.6, capsize=3, lw=1.8, ms=5)
    ax.axhline(MIN_BOOTSTRAP_ARI, color=CAT[7], ls="--", lw=1.3)
    ax.text(d["k"].min() - 0.35, MIN_BOOTSTRAP_ARI + 0.012,
            "rule 2 floor = 0.50", color=CAT[7], fontsize=8)
    ax.set_xticks(d["k"])
    ax.set_xlabel("k")
    ax.set_ylabel("bootstrap ARI (mean, 5th-95th pct)")
    ax.set_title("Figure FIGNUM  Reproducibility of the partition under resampling\n"
                 "%d subsamples at %d%% of n per k"
                 % (int(d['n_resamples'].iloc[0]), int(100 * d['sample_fraction'].iloc[0])),
                 fontsize=11)
    return fig


def fig_cross_algo(M, k, space_tag):
    """Figure FIGNUM - cross-algorithm agreement matrix at the chosen k."""
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    im = ax.imshow(M.to_numpy(), cmap="YlGnBu", vmin=0, vmax=1)
    ax.set_xticks(range(len(M)))
    ax.set_xticklabels(list(M.columns), rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(M)))
    ax.set_yticklabels(list(M.index), fontsize=8)
    for i in range(len(M)):
        for j in range(len(M)):
            v = M.to_numpy()[i, j]
            ax.text(j, i, "%.2f" % v, ha="center", va="center", fontsize=7.6,
                    color="white" if v > 0.6 else INK)
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.8, label="adjusted Rand index")
    ax.set_title("Figure FIGNUM  Cross-algorithm agreement at k = %d on %s" % (k, space_tag),
                 fontsize=11)
    return fig


def fig_consensus(M, labels, k):
    """Figure FIGNUM - the consensus matrix, rows ordered by cluster.

    A crisp block-diagonal means students land together regardless of the draw;
    smeared blocks mean the boundary is arbitrary. Reading this honestly is what
    separates "four groups" from "four regions of a continuum".
    """
    order = np.argsort(np.asarray(labels), kind="stable")
    Mo = M[np.ix_(order, order)]
    fig, ax = plt.subplots(figsize=(6.4, 5.8))
    im = ax.imshow(Mo, cmap="YlGnBu", vmin=0, vmax=1, interpolation="nearest")
    bounds = np.cumsum(pd.Series(np.asarray(labels)[order]).value_counts(sort=False)
                       .reindex(pd.unique(np.asarray(labels)[order])).to_numpy())
    for b in bounds[:-1]:
        ax.axhline(b - 0.5, color=INK, lw=0.9)
        ax.axvline(b - 0.5, color=INK, lw=0.9)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.8, label="co-assignment rate")
    ax.set_title("Figure FIGNUM  Consensus matrix at k = %d (students ordered by cluster)" % k,
                 fontsize=11)
    return fig


# === SECTION 14: Choosing k =================================================
def choose_k(ev, gap, elbow_k, cap=K_CAP):
    """Table 7 - the pre-registered selection rule of CLUSTERING_PLAN.md S4.A.

    Rules 1-4 are screens on USABILITY and are applied before any quality
    criterion is consulted; rule 6 is the quality vote among what survives. The
    order is the whole point: k = 2 is not rejected for scoring badly - it scores
    well on silhouette - but for failing a stated, measurable requirement of the
    research question. A 36%-shape solution is a severity split, and the proposal
    asks for interpretable stress profiles.
    """
    t, chosen, note = screen_and_vote(ev, cap)
    weak = bool(gap["supports_no_structure"])
    decision = {
        "k_chosen": int(chosen),
        "note": note,
        "elbow_k": int(elbow_k),
        "gap_k": int(gap["k_selected"]),
        "silhouette_k": int(ev["silhouette"].idxmax()),
        "davies_bouldin_k": int(ev["davies_bouldin"].idxmin()),
        "calinski_harabasz_k": int(ev["calinski_harabasz"].idxmax()),
        "bic_k": int(ev["bic_gmm"].idxmin()),
        "differentiation_k": int(ev["differentiation"].idxmax()),
        "bootstrap_ari_k": int(ev["bootstrap_ari"].idxmax()),
        "weak_structure": weak,
        "framing": ("a defensible segmentation of a continuum" if weak
                    else "discovered groups"),
        "survivors": [int(k) for k in t.index if t.loc[k, "survives"]],
    }
    return t, decision


def print_k_decision(t, decision, ev):
    """The full disagreement record: which criterion voted for what."""
    print("\n  Criterion votes (the panel disagrees, and that is reported, not hidden):")
    for label, key in [("elbow (kneedle)", "elbow_k"), ("silhouette", "silhouette_k"),
                       ("Davies-Bouldin", "davies_bouldin_k"),
                       ("Calinski-Harabasz", "calinski_harabasz_k"),
                       ("GMM BIC", "bic_k"), ("gap statistic", "gap_k"),
                       ("bootstrap ARI", "bootstrap_ari_k"),
                       ("differentiation", "differentiation_k")]:
        print("    %-22s -> k = %d" % (label, decision[key]))
    print("\n  Screens (applied BEFORE any quality criterion):")
    for k in t.index:
        mark = "keep" if t.loc[k, "survives"] else "DROP"
        print("    k = %d  %s  %s" % (k, mark, t.loc[k, "discarded_because"]))
    print("\n  CHOSEN k = %d  (%s)" % (decision["k_chosen"], decision["note"]))
    if decision["weak_structure"]:
        print("  NOTE  the gap statistic selected k = 1, so structure is weak. Output is "
              "labelled\n        '%s' rather than 'discovered groups'." % decision["framing"])


# === SECTION 15: Holdout test ===============================================
def canonicalise_labels(Z, labels):
    """Relabel clusters in a deterministic order so names are stable across runs.

    k-means numbers its clusters by initialisation order, which changes with the
    seed. Sorting them by overall strain level (tie-broken lexicographically on
    the centroid) makes cluster 0 mean the same thing every run - a requirement
    if the persona names are to be generated rather than typed in.
    """
    labels = np.asarray(labels)
    cents = {c: Z[labels == c].mean(axis=0) for c in np.unique(labels)}
    order = sorted(cents, key=lambda c: (-float(cents[c].mean()), tuple(np.round(cents[c], 6))))
    remap = {old: new for new, old in enumerate(order)}
    return np.array([remap[l] for l in labels]), remap


def fit_frozen_model(Z_train, k, random_state=RANDOM_STATE):
    """Fit the headline model on TRAIN only, then freeze it."""
    km = KMeans(n_clusters=k, n_init=N_INIT, random_state=random_state).fit(Z_train)
    lab, remap = canonicalise_labels(Z_train, km.labels_)
    # Reorder the stored centroids to match the canonical labels, so
    # model.predict() and the profile tables agree without a translation step.
    centres = np.zeros_like(km.cluster_centers_)
    for old, new in remap.items():
        centres[new] = km.cluster_centers_[old]
    km.cluster_centers_ = centres
    km._canonical = True
    return km, lab


def holdout_test(Z_tr, lab_tr, Z_te, model, k):
    """Table 8 - four tests that the personas are not an artefact of the sample.

    (a) do the centroids reproduce, (b) do the proportions reproduce, (c) does
    silhouette hold up out of sample, (d) does a model fitted fresh on the
    holdout recover the same partition as the frozen model's predictions.
    Test (d) is the strict one: it can fail even when (a)-(c) pass.
    """
    lab_te = model.predict(Z_te)
    fresh = KMeans(n_clusters=k, n_init=N_INIT, random_state=RANDOM_STATE).fit(Z_te)
    fresh_lab, _ = canonicalise_labels(Z_te, fresh.labels_)

    cent_tr = np.vstack([Z_tr[lab_tr == c].mean(axis=0) for c in range(k)])
    cent_te = np.vstack([Z_te[lab_te == c].mean(axis=0) if (lab_te == c).sum() else np.full(Z_te.shape[1], np.nan)
                         for c in range(k)])
    drift = np.abs(cent_tr - cent_te)

    p_tr = np.array([(lab_tr == c).mean() for c in range(k)])
    p_te = np.array([(lab_te == c).mean() for c in range(k)])
    obs = np.array([(lab_te == c).sum() for c in range(k)], dtype=float)
    exp = p_tr * len(lab_te)
    chi2 = float(((obs - exp) ** 2 / np.maximum(exp, 1e-9)).sum())

    sil_tr = float(silhouette_score(Z_tr, lab_tr))
    sil_te = float(silhouette_score(Z_te, lab_te)) if len(np.unique(lab_te)) > 1 else np.nan
    ari_fresh = float(adjusted_rand_score(lab_te, fresh_lab))

    rows = [
        ("(a) centroid reproduction: max |train - holdout| z-drift",
         round(float(np.nanmax(drift)), 3), "tolerance 0.50 z"),
        ("(a) centroid reproduction: mean |z-drift|",
         round(float(np.nanmean(drift)), 3), "lower is better"),
        ("(b) proportion match: max |train%% - holdout%%|",
         round(float(100 * np.abs(p_tr - p_te).max()), 2), "percentage points"),
        ("(b) proportion match: chi-square (df = %d)" % (k - 1),
         round(chi2, 2), "p = %.3f" % float(stats.chi2.sf(chi2, k - 1))),
        ("(c) silhouette, train", round(sil_tr, 4), "reference"),
        ("(c) silhouette, holdout (frozen model)", round(sil_te, 4),
         "drop = %.4f" % (sil_tr - sil_te)),
        ("(d) ARI: frozen-predict vs fresh-fit on holdout", round(ari_fresh, 4),
         "the strict test"),
    ]
    table = pd.DataFrame(rows, columns=["test", "value", "note"]).set_index("test")
    summary = {
        "max_centroid_drift": float(np.nanmax(drift)),
        "mean_centroid_drift": float(np.nanmean(drift)),
        "proportion_chi2": chi2,
        "proportion_p": float(stats.chi2.sf(chi2, k - 1)),
        "silhouette_train": sil_tr, "silhouette_holdout": sil_te,
        "ari_frozen_vs_fresh": ari_fresh,
        "passes_centroid_tolerance": bool(np.nanmax(drift) <= 0.50),
        "passes_proportion": bool(stats.chi2.sf(chi2, k - 1) > 0.05),
    }
    return table, lab_te, summary


def fig_holdout(cent_tr, cent_te, cols, k):
    """Figure FIGNUM - train vs holdout centroids, dimension by dimension."""
    fig, axes = plt.subplots(1, k, figsize=(3.05 * k, 3.5), sharey=True)
    axes = np.atleast_1d(axes)
    x = np.arange(len(cols))
    for c, ax in enumerate(axes):
        ax.axhline(0, color=MUTED, lw=1)
        ax.plot(x, cent_tr[c], "-o", color=CAT[c % len(CAT)], lw=1.8, ms=4, label="train")
        ax.plot(x, cent_te[c], "--s", color=INK2, lw=1.4, ms=4, label="holdout")
        ax.set_xticks(x)
        ax.set_xticklabels(cols, rotation=60, ha="right", fontsize=7.4)
        ax.set_title("Cluster %d" % c, fontsize=10)
    axes[0].set_ylabel("centroid (z)")
    axes[0].legend(fontsize=8)
    fig.suptitle("Figure FIGNUM  Centroid reproduction on the 30% holdout", y=1.03,
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    return fig


# === SECTION 16: Profiling ==================================================
def cluster_profiles(Z, labels, cols):
    """Centroid z-profile per cluster, plus size."""
    k = len(np.unique(labels))
    prof = pd.DataFrame(
        np.vstack([Z[labels == c].mean(axis=0) for c in range(k)]),
        index=["C%d" % c for c in range(k)], columns=cols)
    prof.insert(0, "n", [int((labels == c).sum()) for c in range(k)])
    prof.insert(1, "pct", [round(float(100 * (labels == c).mean()), 1) for c in range(k)])
    return prof


def eta_squared(frame, labels):
    """Share of each feature's variance explained by cluster membership.

    Ranks what actually separates the clusters. A dimension with a high eta^2 is
    doing the work; a low one is along for the ride and should not appear in a
    persona name.
    """
    labels = np.asarray(labels)
    rows = {}
    for c in frame.columns:
        x = frame[c].astype(float).to_numpy()
        grand = x.mean()
        ss_between = sum(int((labels == g).sum()) * (x[labels == g].mean() - grand) ** 2
                         for g in np.unique(labels))
        ss_total = float(((x - grand) ** 2).sum())
        f_stat, p = stats.f_oneway(*[x[labels == g] for g in np.unique(labels)])
        rows[c] = {"eta_squared": round(float(ss_between / ss_total), 4),
                   "F": round(float(f_stat), 2), "p": float(p)}
    out = pd.DataFrame(rows).T.sort_values("eta_squared", ascending=False)
    # Cohen's conventions for eta^2: .01 small, .06 medium, .14 large.
    out["effect"] = pd.cut(out["eta_squared"], [-0.01, 0.01, 0.06, 0.14, 1.0],
                           labels=["negligible", "small", "medium", "large"])
    return out


def fig_profile_heatmap(prof, cols, names=None):
    """Figure FIGNUM - the persona z-profile heatmap. The core result figure."""
    M = prof[cols].to_numpy()
    lim = float(np.abs(M).max()) * 1.02
    fig, ax = plt.subplots(figsize=(1.35 * len(cols) + 3.2, 0.86 * len(prof) + 2.4))
    im = ax.imshow(M, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=30, ha="right", fontsize=9)
    ylab = ["%s\nn=%d (%.1f%%)" % (names[i] if names else prof.index[i],
                                   prof["n"].iloc[i], prof["pct"].iloc[i])
            for i in range(len(prof))]
    ax.set_yticks(range(len(prof)))
    ax.set_yticklabels(ylab, fontsize=8.4)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, "%+.2f" % M[i, j], ha="center", va="center", fontsize=8,
                    color="white" if abs(M[i, j]) > 0.55 * lim else INK)
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.82, label="centroid, z-units from the sample mean")
    ax.set_title("Figure FIGNUM  Persona profiles\n"
                 "Read across a row for a SHAPE, down a column for who is highest",
                 fontsize=11)
    return fig


def fig_radar(prof, cols, names=None):
    """Figure FIGNUM - the same profiles as radar charts.

    Redundant with the heatmap by design: the heatmap is precise, the radar makes
    the shape difference legible at a glance in a slide.
    """
    k = len(prof)
    ang = np.linspace(0, 2 * np.pi, len(cols), endpoint=False).tolist()
    ang += ang[:1]
    ncol = min(k, 4)
    nrow = int(np.ceil(k / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 3.6 * nrow),
                             subplot_kw={"projection": "polar"})
    axes = np.atleast_1d(axes).ravel()
    lim = float(np.abs(prof[cols].to_numpy()).max()) * 1.15
    for i in range(len(axes)):
        ax = axes[i]
        if i >= k:
            ax.axis("off")
            continue
        vals = prof[cols].iloc[i].tolist()
        vals += vals[:1]
        ax.plot(ang, vals, color=CAT[i % len(CAT)], lw=2)
        ax.fill(ang, vals, color=CAT[i % len(CAT)], alpha=0.22)
        ax.plot(ang, [0] * len(ang), color=MUTED, lw=1, ls="--")
        ax.set_xticks(ang[:-1])
        ax.set_xticklabels(cols, fontsize=7.4)
        ax.set_ylim(-lim, lim)
        ax.set_yticklabels([])
        ax.set_title("%s\nn=%d (%.1f%%)" % (names[i] if names else prof.index[i],
                                            prof["n"].iloc[i], prof["pct"].iloc[i]),
                     fontsize=9, pad=16)
    fig.suptitle("Figure FIGNUM  Persona shapes (dashed ring = sample mean)", y=1.0,
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    return fig


def fig_item_means_by_cluster(items, labels, names):
    """Figure FIGNUM - item-level means per cluster, back on the original 1-5 scale."""
    k = len(names)
    d = items[ITEMS].astype(float).copy()
    d["_c"] = labels
    M = d.groupby("_c")[ITEMS].mean()
    fig, ax = plt.subplots(figsize=(10.6, 4.6))
    w = 0.8 / k
    x = np.arange(len(ITEMS))
    for c in range(k):
        ax.bar(x + c * w - 0.4 + w / 2, M.loc[c], width=w * 0.92,
               color=CAT[c % len(CAT)], label=names[c])
    ax.axhline(float(items[ITEMS].to_numpy().mean()), color=INK, ls="--", lw=1,
               label="sample mean")
    ax.set_xticks(x)
    ax.set_xticklabels(ITEMS, rotation=45, ha="right", fontsize=8.4)
    ax.set_ylabel("mean, aligned 1-5 scale")
    ax.set_ylim(1, 5.2)
    ax.legend(fontsize=8, ncol=min(k + 1, 5))
    ax.set_title("Figure FIGNUM  Item-level means by persona (high = more strain everywhere)")
    return fig


def fig_eta(eta):
    """Figure FIGNUM - what actually separates the personas."""
    d = eta.sort_values("eta_squared")
    fig, ax = plt.subplots(figsize=(7.6, 0.42 * len(d) + 2.0))
    ax.barh(range(len(d)), d["eta_squared"], color=CAT[0], height=0.62)
    for i, (v, e) in enumerate(zip(d["eta_squared"], d["effect"])):
        ax.text(v + 0.006, i, "%.3f  (%s)" % (v, e), va="center", fontsize=8, color=INK2)
    ax.set_yticks(range(len(d)))
    ax.set_yticklabels(list(d.index), fontsize=8.6)
    ax.set_xlim(0, float(d["eta_squared"].max()) * 1.45)
    ax.set_xlabel("eta-squared: share of variance explained by cluster membership")
    ax.grid(axis="y", visible=False)
    ax.set_title("Figure FIGNUM  What separates the personas\n"
                 "Cohen: .01 small, .06 medium, .14 large", fontsize=11)
    return fig


def background_composition(bg, labels, names):
    """Table 9 - demographic tilt of each persona, versus the sample base rate."""
    d = bg.copy()
    d["_c"] = labels
    out = []
    for c in range(len(names)):
        sub = d[d["_c"] == c]
        row = {"persona": names[c], "n": len(sub),
               "pct_of_sample": round(100 * len(sub) / len(d), 1)}
        row["mean_cgpa_band"] = round(float(sub["cgpa_ord"].mean()), 2)
        row["mean_year"] = round(float(sub["year_ord"].mean()), 2)
        row["pct_backlog"] = round(float(100 * sub["backlog"].mean()), 1)
        row["pct_female"] = round(float(100 * (sub["gender"] == "Female").mean()), 1)
        for lv in LIVE_ORD:
            row["pct_" + lv.lower()] = round(float(100 * (sub["living"] == lv).mean()), 1)
        row["top_department"] = sub["department"].value_counts().idxmax()
        out.append(row)
    table = pd.DataFrame(out).set_index("persona")
    base = {"n": len(d), "pct_of_sample": 100.0,
            "mean_cgpa_band": round(float(d["cgpa_ord"].mean()), 2),
            "mean_year": round(float(d["year_ord"].mean()), 2),
            "pct_backlog": round(float(100 * d["backlog"].mean()), 1),
            "pct_female": round(float(100 * (d["gender"] == "Female").mean()), 1),
            **{"pct_" + lv.lower(): round(float(100 * (d["living"] == lv).mean()), 1)
               for lv in LIVE_ORD},
            "top_department": d["department"].value_counts().idxmax()}
    table.loc["SAMPLE (base rate)"] = base
    return table


def fig_cgpa_by_cluster(bg, labels, names):
    """Figure FIGNUM - CGPA band composition per persona."""
    d = pd.crosstab(pd.Series(labels, name="c"), bg["cgpa"].to_numpy())
    d = d.reindex(columns=[c for c in CGPA_ORD if c in d.columns])
    pct = 100 * d.div(d.sum(axis=1), axis=0)
    fig, ax = plt.subplots(figsize=(8.6, 0.62 * len(pct) + 2.4))
    left = np.zeros(len(pct))
    for i, col in enumerate(pct.columns):
        ax.barh(range(len(pct)), pct[col], left=left, color=DIV5[i % len(DIV5)],
                height=0.66, edgecolor=SURFACE, linewidth=0.7, label=col)
        for y, (v, l) in enumerate(zip(pct[col], left)):
            if v >= 7:
                ax.text(l + v / 2, y, "%.0f" % v, ha="center", va="center", fontsize=7.4,
                        color="white" if i in (0, len(pct.columns) - 1) else INK)
        left += pct[col].to_numpy()
    ax.set_yticks(range(len(pct)))
    ax.set_yticklabels(names, fontsize=8.6)
    ax.set_xlim(0, 100)
    ax.set_xlabel("% of persona")
    ax.grid(axis="y", visible=False)
    ax.legend(fontsize=8, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.16))
    ax.set_title("Figure FIGNUM  CGPA composition by persona", fontsize=11)
    return fig


# === SECTION 17: Persona naming =============================================
# Names are COMPOSED from the dimensions that actually separate a cluster, never
# typed in. The vocabulary below is the only editorial input: one phrase per
# (dimension, direction). Which phrases fire, and for which cluster, is decided
# by the centroid.
PHRASE = {
    ("Workload", "high"): "Overloaded", ("Workload", "low"): "Light workload",
    ("Evaluation", "high"): "Exam-anxious", ("Evaluation", "low"): "Exam-calm",
    ("SupportGap", "high"): "Unsupported", ("SupportGap", "low"): "Well-supported",
    ("Financial", "high"): "Financially strained", ("Financial", "low"): "Financially secure",
    ("FutureMacro", "high"): "Future-anxious", ("FutureMacro", "low"): "Future-confident",
    ("CGPA_ord", "high"): "High-achieving", ("CGPA_ord", "low"): "Low-CGPA",
}
NAME_THRESHOLD = 0.35        # |z| below this is not a defining feature
NAME_MAX_TERMS = 3


def name_cluster(profile_row, cols, threshold=NAME_THRESHOLD, max_terms=NAME_MAX_TERMS,
                 eligible=None):
    """Compose one persona label from its own strongest z-deviations.

    `eligible` restricts naming to dimensions that actually separate the
    clusters (eta^2 above Cohen's "small" floor, from S16). Without that screen a
    persona can be named after a dimension on which every cluster sits in much
    the same place, which reads as a distinction but is not one.
    """
    use = [c for c in cols if eligible is None or c in eligible] or list(cols)
    z = profile_row[use].astype(float)
    ranked = z.reindex(z.abs().sort_values(ascending=False).index)
    terms, used = [], []
    for dim, val in ranked.items():
        if abs(val) < threshold or len(terms) >= max_terms:
            break
        key = (dim, "high" if val > 0 else "low")
        phrase = PHRASE.get(key, "%s %s" % (dim, key[1]))
        terms.append(phrase)
        used.append({"dimension": dim, "z": round(float(val), 3), "direction": key[1]})
    if not terms:
        return "Mid-range / undifferentiated", used
    return " & ".join(terms), used


def name_all_clusters(prof, cols, eta=None, eta_floor=0.01):
    """Table 10 - the generated names with the evidence behind each one."""
    eligible = None
    if eta is not None:
        eligible = [c for c in cols if float(eta.loc[c, "eta_squared"]) >= eta_floor]
    names, rows = [], []
    for i in range(len(prof)):
        nm, used = name_cluster(prof.iloc[i], cols, eligible=eligible)
        # Disambiguate should two clusters compose the same phrase.
        base, n = nm, 2
        while nm in names:
            nm = "%s (%d)" % (base, n)
            n += 1
        names.append(nm)
        rows.append({
            "cluster": prof.index[i], "persona": nm,
            "n": int(prof["n"].iloc[i]), "pct": float(prof["pct"].iloc[i]),
            "defining_dimensions": "; ".join(
                "%s %+.2f" % (u["dimension"], u["z"]) for u in used) or "none above |z| = %.2f" % NAME_THRESHOLD,
        })
    return names, pd.DataFrame(rows).set_index("cluster")


def runner_up_solution(Z, ev, vote, cols, chosen_k, eta_floor=0.01):
    """The second-ranked surviving k, profiled and named alongside the headline.

    The S4.A rule crowns exactly one k, and that k is the headline. But the rule
    can separate two solutions by a hair, and when it does, the runner-up is a
    real finding rather than a discard: a finer segmentation that also passed
    every usability screen. Reporting it costs one table and one figure and stops
    the crowned k from looking more inevitable than the vote actually made it.
    """
    surv = [int(k) for k in vote.index if vote.loc[k, "survives"] and int(k) != int(chosen_k)]
    if not surv:
        return None
    runner = min(surv, key=lambda k: (float(vote.loc[k, "mean_rank"]), k))
    lab, _ = fit_algorithm("kmeans", Z, runner)
    lab, _ = canonicalise_labels(Z, lab)
    prof = cluster_profiles(Z, lab, cols)
    eta = eta_squared(pd.DataFrame(Z, columns=cols), lab)
    names, ntab = name_all_clusters(prof, cols, eta=eta, eta_floor=eta_floor)
    return {"k": int(runner), "labels": lab, "profile": prof, "names": names,
            "name_table": ntab,
            "mean_rank": float(vote.loc[runner, "mean_rank"]),
            "headline_mean_rank": float(vote.loc[chosen_k, "mean_rank"]),
            "margin": float(vote.loc[runner, "mean_rank"] - vote.loc[chosen_k, "mean_rank"]),
            "bootstrap_ari": float(ev.loc[runner, "bootstrap_ari"]),
            "differentiation": float(ev.loc[runner, "differentiation"]),
            "silhouette": float(ev.loc[runner, "silhouette"])}


# === SECTION 18: External validation ========================================
def cramers_v(conf):
    """Bias-corrected Cramer's V.

    Mandatory alongside chi-square: at n = 987 a chi-square test calls a trivial
    association significant, so a p-value alone would overstate every result in
    this section.
    """
    chi2 = stats.chi2_contingency(conf)[0]
    n = conf.to_numpy().sum()
    phi2 = chi2 / n
    r, c = conf.shape
    phi2c = max(0.0, phi2 - (c - 1) * (r - 1) / max(n - 1, 1))
    rc = r - (r - 1) ** 2 / max(n - 1, 1)
    cc = c - (c - 1) ** 2 / max(n - 1, 1)
    denom = min(rc - 1, cc - 1)
    return float(np.sqrt(phi2c / denom)) if denom > 0 else np.nan


def classes_to_clusters(bg, labels, held_out=HELD_OUT, cgpa_in_model=True):
    """Table 11 - agreement between the partition and each held-out variable.

    None of these entered the feature block, so above-chance agreement is genuine
    external validation and at-chance agreement is a real, reportable negative.
    CGPA is listed separately when it is in the model: it is then a manipulation
    check, not evidence.
    """
    rows = []
    checks = list(held_out) + (["cgpa"] if not cgpa_in_model else [])
    for var in checks:
        conf = pd.crosstab(pd.Series(np.asarray(labels), name="cluster"),
                           bg[var].to_numpy())
        chi2, p, dof, _ = stats.chi2_contingency(conf)
        v = cramers_v(conf)
        rows.append({
            "variable": var, "levels": conf.shape[1],
            "chi2": round(float(chi2), 2), "dof": int(dof), "p": float(p),
            "cramers_v": round(v, 3),
            "effect": ("negligible" if v < 0.10 else "small" if v < 0.20
                       else "moderate" if v < 0.30 else "large"),
            "adjusted_rand": round(float(adjusted_rand_score(
                bg[var].astype(str).to_numpy(), np.asarray(labels))), 4),
            "significant_bonferroni": bool(p < 0.05 / max(len(checks), 1)),
        })
    table = pd.DataFrame(rows).set_index("variable")
    if cgpa_in_model:
        conf = pd.crosstab(pd.Series(np.asarray(labels), name="cluster"), bg["cgpa"].to_numpy())
        chi2, p, dof, _ = stats.chi2_contingency(conf)
        table.loc["cgpa (IN MODEL - not validation)"] = {
            "levels": conf.shape[1], "chi2": round(float(chi2), 2), "dof": int(dof),
            "p": float(p), "cramers_v": round(cramers_v(conf), 3),
            "effect": "manipulation check", "adjusted_rand": round(float(
                adjusted_rand_score(bg["cgpa"].astype(str).to_numpy(), np.asarray(labels))), 4),
            "significant_bonferroni": bool(p < 0.05 / max(len(checks), 1))}
    return table


def fig_external(table):
    """Figure FIGNUM - effect sizes for the held-out variables."""
    d = table[table["effect"] != "manipulation check"].sort_values("cramers_v")
    fig, ax = plt.subplots(figsize=(7.6, 0.5 * len(d) + 2.2))
    cols = [CAT[2] if s else BASE for s in d["significant_bonferroni"]]
    ax.barh(range(len(d)), d["cramers_v"], color=cols, height=0.6)
    for i, (v, e, s) in enumerate(zip(d["cramers_v"], d["effect"], d["significant_bonferroni"])):
        ax.text(v + 0.004, i, "V = %.3f  (%s%s)" % (v, e, ", sig." if s else ", n.s."),
                va="center", fontsize=8, color=INK2)
    for x, lab in [(0.10, "small"), (0.20, "moderate"), (0.30, "large")]:
        ax.axvline(x, color=MUTED, ls=":", lw=1)
        ax.text(x, len(d) - 0.4, lab, fontsize=7.4, color=MUTED, ha="center")
    ax.set_yticks(range(len(d)))
    ax.set_yticklabels(list(d.index), fontsize=8.8)
    ax.set_xlim(0, max(0.34, float(d["cramers_v"].max()) * 1.7))
    ax.set_xlabel("Cramer's V (effect size)")
    ax.grid(axis="y", visible=False)
    ax.set_title("Figure FIGNUM  External validation against held-out variables\n"
                 "Effect size, not p-value: at n = 987 chi-square is significant for "
                 "trivial associations", fontsize=11)
    return fig


# === SECTION 19: Free-text corroboration ====================================
# The clusterer NEVER sees this text. That is the whole point: if the
# financially-strained persona also over-mentions money in its own words, that
# is independent corroboration rather than a result true by construction.
try:
    from lexicon import (BANGLA_RE, LATIN_RE, LEXICON_VERSION, ROMANISED_CUES,
                         THEME_NAMES, THEMES, answered_mask, coverage,
                         fingerprint, language_of, prevalence, tag_frame,
                         tag_text, text_profile, untagged_answers)
except ImportError:                       # running from the repo root
    sys.path.insert(0, os.path.join(ROOT, "src"))
    from lexicon import (BANGLA_RE, LATIN_RE, LEXICON_VERSION, ROMANISED_CUES,
                         THEME_NAMES, THEMES, answered_mask, coverage,
                         fingerprint, language_of, prevalence, tag_frame,
                         tag_text, text_profile, untagged_answers)

#: Stressors students volunteer that NO Likert item asks about. Frozen probes,
#: fixed before the run. This is the one thing only the free text can deliver,
#: and the most directly actionable output for the department.
UNASKED_PROBES = {
    "Preparatory leave too short": [r"\bshort pl\b", r"\bpl\b(?!.*plan)", r"preparatory leave",
                                    r"prep leave", r"\bpl er\b"],
    "Session jam / semester length": [r"session ?jam", r"\bsession\b", r"semester (length|duration|too)",
                                      r"long semester", r"সেশন"],
    "Viva / oral examination": [r"\bviva\b", r"\boral\b", r"ভাইভা"],
    "Hall / mess food quality": [r"\bmess\b", r"hall food", r"canteen", r"খাবারের মান", r"মেসের"],
    "Homesickness / family separation": [r"homesick", r"away from (my )?family", r"miss my family",
                                         r"far from home", r"না দেখে", r"পরিবারকে"],
    "Adjusting to a new city": [r"new city", r"new place", r"adjust", r"adapt", r"নতুন জায়গা"],
    "English-language weakness": [r"english", r"ইংরেজি"],
    "Cannot follow lectures": [r"can'?t understand", r"cannot understand", r"not understand",
                               r"hard to follow", r"বুঝতে পারি না"],
    "Non-departmental courses": [r"non.?dept", r"non.?departmental", r"\bhum\b", r"outside course"],
    "Cost of models / drawings": [r"model ?mak", r"drawing (cost|sheet)", r"sheet cost", r"মডেল"],
    "Transport / commuting": [r"transport", r"commut", r"\bbus\b", r"যাতায়াত"],
    "Campus politics / ragging": [r"politic", r"ragging", r"\bbully", r"রাজনীতি"],
}
_UNASKED_COMPILED = {k: [re.compile(p, re.IGNORECASE) for p in v]
                     for k, v in UNASKED_PROBES.items()}

#: Procedural non-answers. Coded as an explicit level rather than dropped: for
#: Q19 in particular, "I am in 1st year" is information, not missingness.
NONANSWER_RE = re.compile(
    r"^\s*(n/?a|na|nil|none|no|nothing|nai|nei|same|same as (above|before)|"
    r"already (said|mentioned)|\.|-|--|x|xx|1st year|first year|i am in 1st year)\s*[.!]*\s*$",
    re.IGNORECASE)


def nonanswer_mask(series):
    """Procedural non-answers among the rows that did contain something."""
    s = series.fillna("").astype(str).str.strip()
    return (s.str.len() > 0) & s.map(lambda t: bool(NONANSWER_RE.match(t)))


def tag_unasked(series):
    """0/1 matrix over the unasked-stressor probes."""
    txt = series.fillna("").astype(str)
    return pd.DataFrame(
        {k: [int(any(p.search(t) for p in pats)) for t in txt]
         for k, pats in _UNASKED_COMPILED.items()}, index=series.index)


def theme_by_cluster(T, mask, labels, names, alpha=0.05):
    """Table 12 - theme prevalence per persona, chi-square, Bonferroni-corrected.

    Bonferroni over the 13 themes (alpha = .05/13 = .0038), because testing every
    theme against every persona and reporting the ones that happened to clear
    .05 is exactly how a corroboration analysis turns into noise-mining.
    """
    m = np.asarray(mask)
    lab = np.asarray(labels)[m]
    sub = T[m]
    k = len(names)
    corrected = alpha / len(THEME_NAMES)
    rows = []
    for th in THEME_NAMES:
        y = sub[th].to_numpy()
        conf = pd.crosstab(lab, y)
        if conf.shape[1] < 2:
            continue
        chi2, p, dof, _ = stats.chi2_contingency(conf)
        r = {"theme": th, "overall_pct": round(float(100 * y.mean()), 1)}
        for c in range(k):
            r[names[c]] = round(float(100 * y[lab == c].mean()), 1) if (lab == c).sum() else np.nan
        r["chi2"] = round(float(chi2), 2)
        r["p"] = float(p)
        r["cramers_v"] = round(cramers_v(conf), 3)
        r["sig_bonferroni"] = bool(p < corrected)
        peak = max(range(k), key=lambda c: r.get(names[c], -1) if r.get(names[c]) == r.get(names[c]) else -1)
        r["highest_persona"] = names[peak]
        rows.append(r)
    table = pd.DataFrame(rows).set_index("theme").sort_values("overall_pct", ascending=False)
    return table, corrected


def prompted_vs_unprompted(desc, prev):
    """Table - what students agree with when asked, against what they volunteer.

    The two need not coincide, and a mismatch is itself a result about the
    instrument: an item everyone endorses but nobody volunteers is measuring
    assent, not salience.
    """
    #: Frozen mapping from a Likert item to the lexicon theme covering it. Items
    #: with no counterpart theme are listed as unmapped rather than forced.
    ITEM_TO_THEME = {
        "PileUp": "Lab & coursework load", "LabStress": "Lab & coursework load",
        "ExamWorry": "Exams & results", "ResultDemotiv": "Exams & results",
        "CGPACompare": "Exams & results",
        "JobWorry": "Career & job uncertainty",
        "Financial": "Financial stress",
        "AskTeacher": "Teachers & teaching quality", "Feedback": "Teachers & teaching quality",
        "SleepLoss": "Sleep & health", "MissMeal": "Living conditions & food",
    }
    rank_prompt = desc["raw_mean"].rank(ascending=False)
    rows = []
    for it, th in ITEM_TO_THEME.items():
        rows.append({
            "item": it, "theme": th,
            "prompted_mean": round(float(desc.loc[it, "raw_mean"]), 2),
            "prompted_rank": int(rank_prompt[it]),
            "volunteered_pct": float(prev.get(th, np.nan)),
            "volunteered_rank": int(prev.rank(ascending=False).get(th, 0)),
        })
    t = pd.DataFrame(rows).set_index("item")
    t["rank_gap"] = t["volunteered_rank"] - t["prompted_rank"]
    return t.sort_values("rank_gap")


def q18_q19_shift(T18, T19, m18, m19):
    """Within-person retrospective comparison: previous years vs now.

    Q19 asks about earlier years, so pairing the two for the same student gives a
    before/after read the cross-sectional design otherwise cannot provide.
    """
    both = np.asarray(m18) & np.asarray(m19)
    rows = []
    for th in THEME_NAMES:
        a = T19[th].to_numpy()[both]          # then
        b = T18[th].to_numpy()[both]          # now
        # McNemar on the discordant pairs: who gained vs who dropped the theme.
        n01 = int(((a == 0) & (b == 1)).sum())
        n10 = int(((a == 1) & (b == 0)).sum())
        if n01 + n10 > 0:
            p = float(stats.binomtest(n01, n01 + n10, 0.5).pvalue)
        else:
            p = np.nan
        rows.append({"theme": th,
                     "previous_years_pct": round(float(100 * a.mean()), 1),
                     "current_pct": round(float(100 * b.mean()), 1),
                     "change_pp": round(float(100 * (b.mean() - a.mean())), 1),
                     "gained_n": n01, "dropped_n": n10, "mcnemar_p": p})
    t = pd.DataFrame(rows).set_index("theme").sort_values("change_pp")
    t["sig_bonferroni"] = t["mcnemar_p"] < (0.05 / len(THEME_NAMES))
    return t, int(both.sum())


def tfidf_nmf_crosscheck(series, mask, n_topics=6, random_state=RANDOM_STATE):
    """A data-driven cross-check that is EXPECTED to partially fail.

    TF-IDF over Latin script cannot represent the ~12% of answers written in
    Bangla script - they vectorise to all-zero rows. Running it anyway and
    measuring that failure is the evidence for why a hand-built code-mixed
    lexicon was necessary; it turns a design choice into a reported finding.
    """
    s = series.fillna("").astype(str)[np.asarray(mask)]
    lang = s.map(language_of)
    vec = TfidfVectorizer(min_df=4, max_df=0.6, ngram_range=(1, 2),
                          token_pattern=r"(?u)\b[A-Za-z][A-Za-z]+\b", stop_words="english")
    X = vec.fit_transform(s)
    empty = np.asarray((X.sum(axis=1) == 0)).ravel()
    stats_d = {
        "n_documents": int(X.shape[0]),
        "vocabulary_size": int(X.shape[1]),
        "empty_vector_n": int(empty.sum()),
        "empty_vector_pct": round(float(100 * empty.mean()), 1),
        "empty_among_bangla_pct": round(float(100 * empty[(lang == "bangla").to_numpy()].mean()), 1)
        if (lang == "bangla").sum() else np.nan,
        "empty_among_latin_pct": round(float(100 * empty[(lang == "latin").to_numpy()].mean()), 1)
        if (lang == "latin").sum() else np.nan,
    }
    nmf = NMF(n_components=n_topics, random_state=random_state, init="nndsvda", max_iter=600)
    W = nmf.fit_transform(X)
    terms = np.array(vec.get_feature_names_out())
    topics = pd.DataFrame(
        {"topic": ["T%d" % (i + 1) for i in range(n_topics)],
         "top_terms": [", ".join(terms[np.argsort(-nmf.components_[i])[:8]]) for i in range(n_topics)],
         "share_of_docs_pct": [round(float(100 * (W.argmax(axis=1) == i).mean()), 1)
                               for i in range(n_topics)]}).set_index("topic")
    return topics, stats_d


def text_null_checks(series, mask, strain):
    """The two pre-specified null tests, reported whatever they show."""
    m = np.asarray(mask)
    a = np.asarray(strain)[m]
    b = np.asarray(strain)[~m]
    words = series.fillna("").astype(str)[m].str.split().str.len().to_numpy()
    if len(b) >= 2:
        t, p = stats.ttest_ind(a, b, equal_var=False)
        d = float((a.mean() - b.mean()) / np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2))
    else:
        t = p = d = np.nan
    r, pr = stats.pearsonr(words, a)
    return {
        "A_answered_n": int(m.sum()), "A_skipped_n": int((~m).sum()),
        "A_strain_answered": round(float(a.mean()), 3),
        "A_strain_skipped": round(float(b.mean()), 3) if len(b) else None,
        "A_welch_t": round(float(t), 3), "A_p": float(p), "A_cohens_d": round(float(d), 3),
        "A_verdict": ("skippers do NOT differ in measured strain (p = %.3f)" % p
                      if p > 0.05 else
                      "skippers differ in measured strain (p = %.4f, d = %.2f)" % (p, d)),
        "B_length_strain_r": round(float(r), 3), "B_p": float(pr),
        "B_verdict": ("answer length is unrelated to strain (r = %.3f, p = %.3f)" % (r, pr)
                      if pr > 0.05 else
                      "longer answers come from more strained students (r = %.3f, p = %.4f)" % (r, pr)),
    }


def fig_text_profile(p18, p19, cov18, cov19):
    """Figure FIGNUM - what the free text actually is, and why the method follows."""
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 3.8))
    ax = axes[0]
    ax.bar([0, 1], [p18["response_rate_pct"], p19["response_rate_pct"]],
           color=[CAT[0], CAT[1]], width=0.55)
    for x, v, n in zip([0, 1], [p18["response_rate_pct"], p19["response_rate_pct"]],
                       [p18["n_answered"], p19["n_answered"]]):
        ax.text(x, v + 1.5, "%.1f%%\n(n=%d)" % (v, n), ha="center", fontsize=8, color=INK2)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Q18 current", "Q19 previous"], fontsize=9)
    ax.set_ylim(0, 108)
    ax.set_ylabel("% answered")
    ax.set_title("Response rate", fontsize=10)

    ax = axes[1]
    x = np.arange(2)
    w = 0.38
    ax.bar(x - w / 2, [p18["pct_le2_words"], p19["pct_le2_words"]], w, color=CAT[3],
           label="<= 2 words")
    ax.bar(x + w / 2, [p18["pct_ge10_words"], p19["pct_ge10_words"]], w, color=CAT[6],
           label=">= 10 words")
    ax.set_xticks(x)
    ax.set_xticklabels(["Q18", "Q19"], fontsize=9)
    ax.set_ylabel("% of answers")
    ax.legend(fontsize=8)
    ax.set_title("Answer length\n(median %g / %g words)"
                 % (p18["words_median"], p19["words_median"]), fontsize=10)

    ax = axes[2]
    keys = ["lang_latin_pct", "lang_bangla_pct", "lang_mixed_pct"]
    labs = ["Latin script", "Bangla script", "mixed"]
    left18 = left19 = 0.0
    for i, (kk, lb) in enumerate(zip(keys, labs)):
        ax.barh(1, p18[kk], left=left18, color=CAT[i], height=0.5,
                edgecolor=SURFACE, label=lb)
        ax.barh(0, p19[kk], left=left19, color=CAT[i], height=0.5, edgecolor=SURFACE)
        left18 += p18[kk]
        left19 += p19[kk]
    ax.set_yticks([1, 0])
    ax.set_yticklabels(["Q18", "Q19"], fontsize=9)
    ax.set_xlim(0, 100)
    ax.set_xlabel("% of answers")
    ax.grid(axis="y", visible=False)
    ax.legend(fontsize=8, loc="lower right")
    ax.set_title("Writing system\n(%.0f%% / %.0f%% carry Bangla script)"
                 % (p18["lang_bangla_pct"] + p18["lang_mixed_pct"],
                    p19["lang_bangla_pct"] + p19["lang_mixed_pct"]), fontsize=10)

    fig.suptitle("Figure FIGNUM  The free text is short, code-mixed and uniformly negative - "
                 "which is what rules out embeddings, stemming and sentiment",
                 y=1.06, fontsize=11.5, fontweight="bold")
    fig.tight_layout()
    return fig


def fig_theme_by_cluster(tt, names):
    """Figure FIGNUM - theme prevalence per persona. The independent corroboration."""
    d = tt.head(10)
    fig, ax = plt.subplots(figsize=(11.0, 0.56 * len(d) + 2.6))
    x = np.arange(len(d))
    k = len(names)
    w = 0.8 / k
    for c, nm in enumerate(names):
        ax.bar(x + c * w - 0.4 + w / 2, d[nm], width=w * 0.9, color=CAT[c % len(CAT)], label=nm)
    ax.plot(x, d["overall_pct"], "k_", ms=26, mew=1.4, label="sample overall")
    for i, s in enumerate(d["sig_bonferroni"]):
        if s:
            ax.text(i, max(d.iloc[i][list(names)].max(), d["overall_pct"].iloc[i]) + 1.4,
                    "*", ha="center", fontsize=13, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels([t[:26] for t in d.index], rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("% of answering students mentioning the theme")
    ax.legend(fontsize=8, ncol=min(k + 1, 4))
    ax.set_title("Figure FIGNUM  Volunteered stressor themes by persona\n"
                 "* = chi-square significant after Bonferroni correction over %d themes. "
                 "The clusterer never saw this text." % len(THEME_NAMES), fontsize=11)
    return fig


def fig_unasked(unasked_pct, tf_stats):
    """Figure FIGNUM - stressors the questionnaire never asked about."""
    d = unasked_pct.sort_values()
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 0.42 * len(d) + 2.6),
                             gridspec_kw={"width_ratios": [2.0, 1.0]})
    ax = axes[0]
    ax.barh(range(len(d)), d.to_numpy(), color=CAT[1], height=0.62)
    for i, v in enumerate(d):
        ax.text(v + 0.12, i, "%.1f%%" % v, va="center", fontsize=8, color=INK2)
    ax.set_yticks(range(len(d)))
    ax.set_yticklabels(list(d.index), fontsize=8.4)
    ax.set_xlim(0, float(d.max()) * 1.35)
    ax.set_xlabel("% of answering students who volunteered it")
    ax.grid(axis="y", visible=False)
    ax.set_title("Volunteered but never asked\n(no Likert item covers these)", fontsize=10.5)

    ax = axes[1]
    vals = [tf_stats["empty_among_latin_pct"], tf_stats["empty_among_bangla_pct"]]
    ax.bar([0, 1], vals, color=[CAT[0], CAT[7]], width=0.55)
    for x, v in zip([0, 1], vals):
        if v == v:
            ax.text(x, v + 2, "%.0f%%" % v, ha="center", fontsize=9, color=INK2)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Latin-script\nanswers", "Bangla-script\nanswers"], fontsize=8.6)
    ax.set_ylim(0, 108)
    ax.set_ylabel("% vectorising to an all-zero row")
    ax.set_title("Why the lexicon was necessary:\nTF-IDF discards Bangla script entirely",
                 fontsize=10.5)

    fig.suptitle("Figure FIGNUM  What only the free text can deliver", y=1.03,
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    return fig


# === SECTION 20: Supervised interpretability layer ==========================
# This is NOT prediction. The labels come from the clusterer, so accuracy here
# measures how compactly the partition can be re-described as rules - the J48
# equivalent the methodology report asks for. High accuracy would be alarming
# only if the tree were fed something outside the feature block.
def tree_rules(X, y, names, max_depth=4, folds=10, random_state=RANDOM_STATE):
    """Table 13 - a depth-limited tree over the cluster labels, with CV and rules."""
    clf = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=20,
                                 random_state=random_state)
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
    scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")
    clf.fit(X, y)
    text = export_text(clf, feature_names=list(X.columns), decimals=2)

    t = clf.tree_
    paths, stack = [], [(0, [])]
    while stack:
        node, cond = stack.pop()
        if t.children_left[node] == -1:
            # sklearn >= 1.4 stores class PROPORTIONS in tree_.value, not counts,
            # so the leaf size has to come from n_node_samples.
            v = t.value[node][0]
            cls = int(np.argmax(v))
            paths.append({
                "persona": names[cls],
                "n_at_leaf": int(t.n_node_samples[node]),
                "purity_pct": round(float(100 * v.max() / v.sum()), 1),
                "rule": " AND ".join(cond) if cond else "(root)",
            })
            continue
        f = X.columns[t.feature[node]]
        th = t.threshold[node]
        stack.append((t.children_left[node], cond + ["%s <= %.2f" % (f, th)]))
        stack.append((t.children_right[node], cond + ["%s > %.2f" % (f, th)]))
    rules = (pd.DataFrame(paths).sort_values(["persona", "n_at_leaf"], ascending=[True, False])
             .set_index("persona"))
    return clf, rules, text, {"cv_folds": int(folds),
                              "cv_accuracy_mean": float(scores.mean()),
                              "cv_accuracy_sd": float(scores.std()),
                              "train_accuracy": float(clf.score(X, y)),
                              "max_depth": int(max_depth),
                              "n_leaves": int(clf.get_n_leaves())}


def rf_importance(X, y, random_state=RANDOM_STATE):
    """Permutation importance from a random forest, as a second opinion.

    Gini importance is biased toward high-cardinality features; permutation
    importance measures the accuracy actually lost when a column is shuffled.
    """
    rf = RandomForestClassifier(n_estimators=400, min_samples_leaf=5,
                                random_state=random_state, n_jobs=-1).fit(X, y)
    from sklearn.inspection import permutation_importance
    pi = permutation_importance(rf, X, y, n_repeats=20, random_state=random_state, n_jobs=-1)
    return rf, pd.DataFrame({
        "permutation_importance": np.round(pi.importances_mean, 4),
        "sd": np.round(pi.importances_std, 4),
        "gini_importance": np.round(rf.feature_importances_, 4),
    }, index=X.columns).sort_values("permutation_importance", ascending=False)


def fig_tree(clf, X, names):
    """Figure FIGNUM - the decision tree, drawn."""
    from sklearn.tree import plot_tree
    fig, ax = plt.subplots(figsize=(15.5, 8.2))
    plot_tree(clf, feature_names=list(X.columns), class_names=list(names), filled=True,
              rounded=True, fontsize=7, impurity=False, proportion=True, ax=ax)
    ax.set_title("Figure FIGNUM  Persona membership as a rule set (depth <= %d)\n"
                 "Interpretability, not prediction: the labels came from the clusterer"
                 % clf.get_depth(), fontsize=11)
    return fig


def fig_importance(imp):
    """Figure FIGNUM - permutation importance."""
    d = imp.sort_values("permutation_importance")
    fig, ax = plt.subplots(figsize=(7.6, 0.45 * len(d) + 2.0))
    ax.barh(range(len(d)), d["permutation_importance"], xerr=d["sd"],
            color=CAT[0], height=0.6, error_kw={"ecolor": BASE, "elinewidth": 1.3})
    ax.set_yticks(range(len(d)))
    ax.set_yticklabels(list(d.index), fontsize=8.8)
    ax.set_xlabel("accuracy lost when the feature is shuffled")
    ax.grid(axis="y", visible=False)
    ax.set_title("Figure FIGNUM  Random-forest permutation importance\n"
                 "Second opinion on what defines the personas", fontsize=11)
    return fig


# === SECTION 21: Persona cards ==============================================
def persona_cards(prof, cols, names, bgtab, themes_tab, decision, recs):
    """The figure the report is built around: one card per persona.

    Everything on a card is generated - the name from the centroid, the tilt from
    the held-out background variables, the themes from the students' own words,
    the recommendation from the persona's own top dimension. Nothing is typed in
    per cluster, so the cards cannot drift from the model behind them.
    """
    import textwrap
    k = len(names)
    ncol = min(k, 2)
    nrow = int(np.ceil(k / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(7.4 * ncol, 3.75 * nrow))
    axes = np.atleast_1d(axes).ravel()

    for i in range(len(axes)):
        ax = axes[i]
        ax.axis("off")
        if i >= k:
            continue
        z = prof[cols].iloc[i]
        strong = z.reindex(z.abs().sort_values(ascending=False).index).head(3)
        row = bgtab.iloc[i]
        if themes_tab is not None and not themes_tab.empty and names[i] in themes_tab:
            th = themes_tab[names[i]].sort_values(ascending=False).head(3)
            theme_txt = "\n".join("   %-32s %5.1f%%" % (t[:32], v) for t, v in th.items())
        else:
            theme_txt = "   (no text themes available)"

        colour = CAT[i % len(CAT)]
        ax.add_patch(plt.Rectangle((0.01, 0.02), 0.98, 0.96, transform=ax.transAxes,
                                   facecolor="white", edgecolor=colour, linewidth=2.2,
                                   zorder=0, clip_on=False))
        ax.add_patch(plt.Rectangle((0.01, 0.84), 0.98, 0.14, transform=ax.transAxes,
                                   facecolor=colour, edgecolor=colour, zorder=1,
                                   clip_on=False))
        # Long generated names need a smaller face rather than a clipped one.
        nm = names[i]
        fs = 12.5 if len(nm) <= 34 else (10.8 if len(nm) <= 46 else 9.4)
        ax.text(0.03, 0.945, nm, transform=ax.transAxes, fontsize=fs,
                fontweight="bold", color="white", va="center", zorder=2)
        ax.text(0.03, 0.876, "n = %d   (%.1f%% of the sample)"
                % (prof["n"].iloc[i], prof["pct"].iloc[i]), transform=ax.transAxes,
                fontsize=8.6, color="white", va="center", zorder=2)

        body = ["DEFINING DIMENSIONS  (z from sample mean)"]
        body += ["   %-14s %+5.2f  %s" % (d, v, "high" if v > 0 else "low")
                 for d, v in strong.items()]
        body += ["", "WHO THEY ARE",
                 "   mean CGPA band  %.2f / 5      backlog  %.1f%%"
                 % (row["mean_cgpa_band"], row["pct_backlog"]),
                 "   mean year       %.2f          female   %.1f%%"
                 % (row["mean_year"], row["pct_female"]),
                 "   largest dept    %s" % row["top_department"],
                 "", "TOP VOLUNTEERED STRESSORS  (their own words)", theme_txt]
        ax.text(0.03, 0.79, "\n".join(body), transform=ax.transAxes, fontsize=8.2,
                va="top", family="DejaVu Sans Mono", color=INK, zorder=2)

        rec = textwrap.fill(recs.get(names[i], ""), 58)
        ax.text(0.03, 0.20, "RECOMMENDATION", transform=ax.transAxes, fontsize=8.2,
                va="top", family="DejaVu Sans Mono", color=INK, zorder=2)
        ax.text(0.03, 0.155, rec, transform=ax.transAxes, fontsize=8.6, va="top",
                color=colour, style="italic", zorder=2)

    fig.suptitle("Figure FIGNUM  Persona cards  -  %s (k = %d)"
                 % (decision["framing"], decision["k_chosen"]),
                 y=1.005, fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


def recommendation(name, z, cols):
    """A rule-composed recommendation, from the persona's own top dimension.

    Composed the same way the names are: the persona's strongest deviation picks
    the advice. A cluster that sits BELOW average on every stressor gets an
    explicit "no targeted intervention" reading rather than being forced into an
    intervention it does not need - a low-strain group is a finding, not a gap in
    the rule table.
    """
    ADVICE_HIGH = {
        "Financial": "prioritise for need-based aid, paid TA slots and fee-instalment options",
        "SupportGap": "structured office hours and a named academic mentor per student",
        "Workload": "coordinate assessment calendars across courses to break the pile-up",
        "Evaluation": "exam-anxiety workshops and low-stakes formative assessment",
        "FutureMacro": "early career counselling, internship pipeline, alumni contact",
    }
    zz = z[cols].astype(float)
    ranked = zz.reindex(zz.abs().sort_values(ascending=False).index)
    top = ranked.index[0]
    if abs(ranked.iloc[0]) < NAME_THRESHOLD:
        return ("no dimension reaches |z| = %.2f: an undifferentiated middle group, "
                "monitor only" % NAME_THRESHOLD)
    for dim, val in ranked.items():
        if abs(val) < NAME_THRESHOLD:
            break
        if val > 0 and dim in ADVICE_HIGH:
            return ADVICE_HIGH[dim]
        if dim == "CGPA_ord" and val < 0:
            return "early-warning academic support and backlog recovery planning"
    # Every defining deviation is BELOW average - a low-strain / coping profile.
    return ("below the sample mean on every defining dimension: no targeted "
            "intervention indicated, use as the comparison group when evaluating "
            "any of the above")


# === SECTION 22: Persistence ================================================
_ARFF_SAFE = re.compile(r"[^0-9A-Za-z_]+")


def _arff_name(s):
    return _ARFF_SAFE.sub("_", str(s)).strip("_")


def _arff_value(v):
    if pd.isna(v):
        return "?"
    s = str(v).replace(chr(92), chr(92) * 2).replace("'", chr(92) + "'")
    return "'" + s.replace("\n", " ").replace("\r", " ") + "'"


def write_arff(df, path, relation, numeric, nominal, string_cols=()):
    """Write an ARFF so the same prepared table opens in WEKA.

    The methodology report specifies a WEKA workflow. Exporting from the
    identical preprocessing the Python pipeline used beats preparing the data
    twice and hoping the two agree. See docs/WEKA_APPENDIX.md for the filter
    chain and clusterer options that reproduce this run.
    """
    lines = ["@RELATION " + _arff_name(relation), ""]
    order = []
    for c in df.columns:
        nm = _arff_name(c)
        if c in numeric:
            lines.append("@ATTRIBUTE %s NUMERIC" % nm)
        elif c in nominal:
            lines.append("@ATTRIBUTE %s {%s}" % (nm, ",".join(_arff_value(v) for v in nominal[c])))
        elif c in string_cols:
            lines.append("@ATTRIBUTE %s STRING" % nm)
        else:
            continue
        order.append(c)
    lines += ["", "@DATA"]
    for _, row in df[order].iterrows():
        vals = ["?" if (c in numeric and pd.isna(row[c])) else
                ("%g" % float(row[c]) if c in numeric else _arff_value(row[c]))
                for c in order]
        lines.append(",".join(vals))
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("  wrote %s" % os.path.basename(path))
    return path


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(jsonable(obj), fh, indent=2, ensure_ascii=False)
    print("  wrote %s" % os.path.basename(path))
    return path


def persist(bundle, assignments, results):
    """Write the model bundle, per-student assignments and results.json."""
    import joblib
    mp = os.path.join(MODEL_DIR, "persona_model.joblib")
    joblib.dump(bundle, mp)
    print("  wrote %s" % os.path.basename(mp))
    ap = os.path.join(OUT_DIR, "student_assignments.csv")
    assignments.to_csv(ap, index=False, encoding="utf-8-sig")
    print("  wrote %s  (%d rows)" % (os.path.basename(ap), len(assignments)))
    save_json(results, os.path.join(OUT_DIR, "results.json"))
    return mp, ap


# === SECTION 23: Reuse - scoring a new respondent ===========================
def assign_persona(new_raw, bundle):
    """Score fresh questionnaire responses against the frozen personas.

    Takes RAW 1-5 answers exactly as the form collects them, plus the CGPA band
    string, and applies reverse coding, subscale construction, ordinal encoding
    and scaling internally. That is what makes this a reusable instrument for
    next semester rather than a one-off script: the caller cannot get the
    preprocessing wrong because the caller never sees it.
    """
    X = pd.DataFrame(new_raw).copy()
    missing = [c for c in ITEMS if c not in X.columns]
    if missing:
        raise ValueError("missing raw item columns: %s" % missing)
    bad = [(c, v) for c in ITEMS for v in X[c] if not (1 <= float(v) <= 5)]
    if bad:
        raise ValueError("Likert answers must be in 1-5; got %s" % bad[:3])

    aligned = X[ITEMS].astype(float).copy()
    for c in REVERSED:
        aligned[c] = 6 - aligned[c]

    feats = pd.DataFrame(index=X.index)
    for name, members in bundle["subscale_spec"].items():
        feats[name] = aligned[members].mean(axis=1)
    if bundle["includes_cgpa"]:
        if "cgpa" not in X.columns:
            raise ValueError("this model needs a 'cgpa' band column, one of %s" % CGPA_ORD)
        ord_ = X["cgpa"].map(CGPA_ORDINAL)
        if ord_.isna().any():
            raise ValueError("unrecognised CGPA band(s): %s"
                             % list(X["cgpa"][ord_.isna()].unique()))
        feats["CGPA_ord"] = ord_.astype(float)

    feats = feats[bundle["columns"]]
    Z = bundle["scaler"].transform(feats.to_numpy())
    lab = bundle["model"].predict(Z)
    d = bundle["model"].transform(Z)
    return pd.DataFrame({
        "cluster": lab,
        "persona": [bundle["names"][c] for c in lab],
        "distance_to_own_centroid": np.round(d[np.arange(len(lab)), lab], 3),
        "margin_to_next": np.round(np.sort(d, axis=1)[:, 1] - np.sort(d, axis=1)[:, 0], 3),
    }, index=X.index)


# === MAIN ===================================================================
def main(argv=None):
    t0 = time.time()
    setup_style()
    ensure_dirs()
    R = RESULTS

    head("SECTION 1-2  LOAD AND SCHEMA CHECK")
    path = resolve_data_path(None if argv is None else (argv[0] if argv else None))
    print("  data: %s" % path)
    df = pd.read_excel(path)
    R["schema"] = validate_schema(df)
    names = column_names(df)
    R["data_path"] = os.path.basename(path)
    export_clean_csv(df, names, os.path.join(OUT_DIR, "stress_clean.csv"))
    items, raw = build_items(df, names)
    bg = build_background(df, names)

    head("SECTION 3  DATA QUALITY AUDIT")
    audit = audit_quality(df, names)
    style = response_style(raw)
    R["audit"], R["response_style"] = audit, style
    savetab(quality_table(audit, style), "table01_data_quality", fmt="%.2f")

    head("SECTION 4  EXPLORATORY DATA ANALYSIS")
    desc = item_descriptives(items, raw)
    savetab(desc[["raw_mean", "raw_sd", "raw_skew", "pct_agree", "pct_disagree",
                  "pct_top_box", "ceiling_flag"]], "table01b_item_descriptives")
    R["ceiling_items"] = [i for i in desc.index if bool(desc.loc[i, "ceiling_flag"])]
    print("\n  ceiling-effect items: %s" % (R["ceiling_items"] or "none"))
    savefig(fig_composition(bg), "sample_composition", "sample composition")
    savefig(fig_departments(bg), "departments", "department spread")
    savefig(fig_likert_stacked(raw), "likert_distribution", "raw response distribution")
    savefig(fig_item_means(desc), "item_means", "item means with ceiling flags")

    head("SECTION 5  PREPROCESSING")
    print("  reverse-coded (6 - x): %s" % ", ".join(REVERSED))
    sdt = savetab(standardisation_evidence(items), "table02_standardisation")
    lo, hi = float(sdt["aligned_sd"].min()), float(sdt["aligned_sd"].max())
    print("\n  item SD range %.2f - %.2f  ->  the most polarised item would carry "
          "%.1fx the\n  weight of the least in un-standardised Euclidean distance. "
          "Standardising is\n  therefore necessary, not conventional."
          % (lo, hi, float(sdt["weight_vs_smallest"].max())))
    R["sd_range"] = [round(lo, 3), round(hi, 3)]

    head("SECTION 6  MEASUREMENT STRUCTURE")
    mtab, mstats, Rmat = measurement_report(items)
    savetab(mtab, "table03_measurement")
    print("\n  Cronbach alpha (12 items) = %.3f" % mstats["cronbach_alpha_12"])
    print("  mean |inter-item r|       = %.3f" % mstats["mean_abs_interitem_r"])
    print("  KMO overall               = %.3f" % mstats["kmo"]["overall"])
    print("  Bartlett chi2(%d)         = %.1f, p = %.3g"
          % (mstats["bartlett"]["dof"], mstats["bartlett"]["chi2"], mstats["bartlett"]["p"]))
    verdict = measurement_verdict(mstats)
    print("\n  VERDICT: %s" % verdict)
    R["measurement"] = {**mstats, "verdict": verdict}
    savefig(fig_interitem_heatmap(Rmat), "interitem_correlations", "inter-item correlations")

    head("SECTION 7  DIMENSIONALITY AND FACTOR STRUCTURE")
    Zitems, _ = standardise(items[ITEMS])
    pca, pcatab, pcastats = pca_report(Zitems)
    savetab(pcatab, "table04a_pca_variance")
    partab, parstats = parallel_analysis(Zitems)
    savetab(partab, "table04b_parallel_analysis")
    print("\n  Kaiser (eigenvalue > 1) retains %d; Horn's parallel analysis retains %d; "
          "%d components\n  are needed for %.0f%% of variance."
          % (pcastats["n_eigenvalues_above_1"], parstats["n_retain_parallel"],
             pcastats["n_components_95pct"], 100 * PCA_VAR_TARGET))
    R["pca"], R["parallel_analysis"] = pcastats, parstats
    savefig(fig_scree(pcatab, partab), "scree", "scree + parallel analysis")
    savefig(fig_cumvar(pcatab, pcastats["n_components_95pct"]), "cumulative_variance",
            "explained variance")
    savefig(fig_pca_projection(Zitems, pca, bg["cgpa"], "CGPA band", CGPA_ORD),
            "pca_projection", "students in PC space")

    n_factors = max(2, int(parstats["n_retain_parallel"]))
    load, ssl, efastats = efa(items, m=n_factors)
    fcols = ["F%d" % (j + 1) for j in range(n_factors)]
    savetab(load, "table04c_efa_loadings")
    R["efa"] = efastats
    print("\n  %d-factor varimax solution: %d item(s) cross-load at |loading| >= 0.30, "
          "\n  total common variance explained = %.1f%%"
          % (n_factors, efastats["n_cross_loading_items"],
             efastats["total_variance_explained_pct"]))
    savefig(fig_loadings(load, fcols), "efa_loadings", "rotated loadings")

    head("SECTION 8  SUBSCALE CONSTRUCTION")
    fnames = name_factors(load, fcols)
    for c in fcols:
        print("  %s -> %-12s (marker item %s, loading %.2f)"
              % (c, fnames[c]["name"], fnames[c]["marker_item"], fnames[c]["marker_loading"]))
    spec4, scores4 = build_subscales(items, load, fcols, fnames, split_financial=False)
    spec5, scores5 = build_subscales(items, load, fcols, fnames, split_financial=True)
    stab4, scorr4 = subscale_report(spec4, scores4, items)
    stab5, scorr5 = subscale_report(spec5, scores5, items)
    savetab(stab4, "table05a_subscales_4")
    savetab(stab5, "table05b_subscales_5")
    savetab(scorr5, "table05c_subscale_correlations")
    R["subscale_spec_4"], R["subscale_spec_5"] = spec4, spec5

    head("SECTION 9-10  FEATURE SPACES AND TRAIN / HOLDOUT SPLIT")
    spaces = build_feature_spaces(items, scores4, scores5, spec4, spec5, bg)
    for tag, d in spaces.items():
        print("  %-20s %2d features  %s" % (tag, len(d["columns"]), d["label"]))
    print("\n  held out of EVERY feature space: %s" % ", ".join(HELD_OUT))
    print("  CGPA enters spaces C/C5 only; it therefore cannot validate them (S18).")

    strat = strain_tertile(items)
    tr_idx, te_idx = train_holdout(np.arange(len(items)), strat)
    print("\n  train n = %d, holdout n = %d, stratified on overall-strain tertile"
          % (len(tr_idx), len(te_idx)))
    R["split"] = {"n_train": int(len(tr_idx)), "n_holdout": int(len(te_idx)),
                  "holdout_frac": HOLDOUT_FRAC}

    head("SECTION 11  THE k SWEEP  (all spaces x %d algorithms x k = %d..%d)"
         % (len(ALGORITHMS), K_RANGE[0], K_RANGE[-1]))
    sweep_tab, excluded = sweep(spaces, subset=tr_idx)
    savetab(sweep_tab.round(4), "table06_full_sweep", show=False, index=False)
    print("  %d model fits recorded." % len(sweep_tab))
    if not excluded.empty:
        print("\n  Excluded from the vote (reason printed, not silently dropped).")
        print("  A linkage that leaves nearly everyone in one cluster has not found "
              "structure; it has found the chaining artefact, so it must not vote.")
        for (sp, al), g in excluded.groupby(["space", "algorithm"], sort=False):
            ks = ", ".join(str(int(v)) for v in sorted(g["k"]))
            print("    %-20s %-9s k = %-18s %s" % (sp, al, ks, g["reason"].iloc[-1]))
        savetab(excluded, "table06b_excluded", show=False, index=False)
    R["n_model_fits"] = int(len(sweep_tab))
    R["excluded_runs"] = excluded.to_dict("records") if not excluded.empty else []

    BASELINE = "A_items12"
    print("\n  Baseline check - does space A reproduce the old two-group result?")
    base = sweep_tab[(sweep_tab.space == BASELINE) & (sweep_tab.algorithm == "kmeans")]
    print("    space A silhouette by k: %s"
          % ", ".join("k=%d %.3f" % (r.k, r.silhouette) for r in base.itertuples()))
    print("    space A best k by silhouette = %d  (the old result, reproduced)"
          % int(base.loc[base.silhouette.idxmax(), "k"]))
    R["baseline_A_best_k_silhouette"] = int(base.loc[base.silhouette.idxmax(), "k"])

    print("\n  Choosing between the 4- and 5-subscale groupings")
    print("  (plan S3.1: the notebook decides this on stability, it is not asserted):")
    CANDIDATES = ["C_subscales4_cgpa", "C5_subscales5_cgpa"]
    ev_by_space, stab_by_space = {}, {}
    for tag in CANDIDATES:
        ev_by_space[tag], stab_by_space[tag] = kselect_evidence(spaces[tag], subset=tr_idx)
    PRIMARY, spacetab = choose_primary_space(ev_by_space)
    savetab(spacetab, "table05d_primary_space_choice", fmt="%.4f")
    print("\n  PRIMARY feature space -> %s" % PRIMARY)
    print("  %s" % spaces[PRIMARY]["label"])
    print("  subscales: %s" % ", ".join(spaces[PRIMARY]["subscale_spec"]))
    R["primary_space"] = PRIMARY
    R["primary_space_choice"] = spacetab.to_dict("index")

    fig_e, elbow_k = fig_elbow(sweep_tab, PRIMARY)
    savefig(fig_e, "elbow", "elbow, primary space")
    savefig(fig_index_panels(sweep_tab, spaces), "validity_indices", "validity indices by k")
    savefig(fig_sizes(sweep_tab, PRIMARY), "cluster_sizes", "smallest cluster by k")

    head("SECTION 12  GAP STATISTIC")
    Ztr = spaces[PRIMARY]["Z"][tr_idx]
    gap = gap_statistic(Ztr)
    print("  %s" % gap["interpretation"])
    R["gap"] = gap
    savefig(fig_gap(gap, PRIMARY), "gap_statistic", "gap statistic")

    head("SECTION 13-14  STABILITY AND THE CHOICE OF k")
    ev, stab_rows = ev_by_space[PRIMARY], stab_by_space[PRIMARY]
    savetab(ev.round(4), "table07a_k_evidence")
    savefig(fig_stability_by_k(stab_rows), "bootstrap_stability", "bootstrap ARI by k")

    vote, decision = choose_k(ev, gap, elbow_k)
    savetab(vote[["min_size_frac", "bootstrap_ari", "differentiation", "mean_rank",
                  "survives", "discarded_because"]], "table07b_k_decision")
    print_k_decision(vote, decision, ev)
    R["k_decision"] = decision
    k = decision["k_chosen"]

    seed = seed_stability(Ztr, k)
    xari_M, xari, _ = cross_algorithm_ari(Ztr, k)
    print("\n  seed stability over %d seeds: mean pairwise ARI = %.3f (min %.3f)"
          % (seed["n_seeds"], seed["pairwise_ari_mean"], seed["pairwise_ari_min"]))
    print("  cross-algorithm mean ARI at k = %d: %.3f" % (k, xari))
    R["seed_stability"], R["cross_algorithm_ari_mean"] = seed, xari
    savetab(xari_M, "table07c_cross_algorithm_ari")
    savefig(fig_cross_algo(xari_M, k, PRIMARY), "cross_algorithm", "cross-algorithm ARI")

    head("SECTION 15  HOLDOUT TEST")
    model, lab_tr = fit_frozen_model(Ztr, k)
    Zte = spaces[PRIMARY]["Z"][te_idx]
    cons = consensus_matrix(Ztr, k)
    savefig(fig_consensus(cons, lab_tr, k), "consensus", "consensus matrix")
    R["consensus_by_cluster"] = consensus_by_cluster(cons, lab_tr)
    print("  within-cluster co-assignment rate: %s" % R["consensus_by_cluster"])

    htab, lab_te, hsum = holdout_test(Ztr, lab_tr, Zte, model, k)
    savetab(htab, "table08_holdout")
    R["holdout"] = hsum
    cols = spaces[PRIMARY]["columns"]
    cent_tr = np.vstack([Ztr[lab_tr == c].mean(axis=0) for c in range(k)])
    cent_te = np.vstack([Zte[lab_te == c].mean(axis=0) if (lab_te == c).sum()
                         else np.full(Zte.shape[1], np.nan) for c in range(k)])
    savefig(fig_holdout(cent_tr, cent_te, cols, k), "holdout_centroids", "centroid reproduction")

    # The reported personas use every student: the holdout has done its job of
    # testing reproducibility, and withholding 30% of a 987-row sample from the
    # descriptive tables would only make the profiles noisier.
    Zall = spaces[PRIMARY]["Z"]
    labels = model.predict(Zall)
    labels, _ = canonicalise_labels(Zall, labels)

    head("SECTION 16  PROFILING")
    prof = cluster_profiles(Zall, labels, cols)
    savetab(prof, "table09a_profiles")
    eta = eta_squared(spaces[PRIMARY]["frame"], labels)
    savetab(eta, "table09b_eta_squared")
    R["profiles"] = prof.to_dict("index")
    R["eta_squared"] = eta["eta_squared"].to_dict()

    head("SECTION 17  PERSONA NAMING")
    pnames, ntab = name_all_clusters(prof, cols, eta=eta)
    savetab(ntab, "table10_persona_names")
    print("\n  Names are composed from each cluster's own z-deviations "
          "(|z| >= %.2f, at most %d terms)." % (NAME_THRESHOLD, NAME_MAX_TERMS))
    R["persona_names"] = pnames

    runner = runner_up_solution(Zall, ev, vote, cols, k)
    if runner is not None:
        print("\n  RUNNER-UP  k = %d survived every screen and lost the vote by %.3f of a"
              % (runner["k"], runner["margin"]))
        print("  rank point (mean rank %.3f vs %.3f). It is reported, not discarded:"
              % (runner["mean_rank"], runner["headline_mean_rank"]))
        savetab(runner["name_table"], "table10b_runner_up_k%d" % runner["k"])
        R["runner_up"] = {kk: vv for kk, vv in runner.items()
                          if kk not in ("labels", "profile", "name_table")}

    savefig(fig_profile_heatmap(prof, cols, pnames), "persona_profiles", "PERSONA PROFILES")
    savefig(fig_radar(prof, cols, pnames), "persona_radar", "persona shapes")
    savefig(fig_item_means_by_cluster(items, labels, pnames), "item_means_by_persona",
            "item means by persona")
    savefig(fig_eta(eta), "eta_squared", "what separates the personas")
    if runner is not None:
        f = fig_profile_heatmap(runner["profile"], cols, runner["names"])
        f.axes[0].set_title("Figure FIGNUM  RUNNER-UP solution, k = %d  (not the headline)\n"
                            "Survived every usability screen; lost the quality vote by "
                            "%.3f of a rank point" % (runner["k"], runner["margin"]),
                            fontsize=11)
        savefig(f, "runner_up_profiles", "runner-up k = %d" % runner["k"])
    bgtab = background_composition(bg, labels, pnames)
    savetab(bgtab, "table09c_background_by_persona")
    savefig(fig_cgpa_by_cluster(bg, labels, pnames), "cgpa_by_persona", "CGPA by persona")

    head("SECTION 18  EXTERNAL VALIDATION")
    ext = classes_to_clusters(bg, labels, cgpa_in_model=spaces[PRIMARY]["includes_cgpa"])
    savetab(ext, "table11_external_validation")
    R["external_validation"] = ext.to_dict("index")
    savefig(fig_external(ext), "external_validation", "external validation")

    head("SECTION 19  FREE-TEXT CORROBORATION  (never a clustering input)")
    print("  lexicon v%s, %d themes, fingerprint %s" % (LEXICON_VERSION, len(THEME_NAMES),
                                                        fingerprint()))
    R["lexicon"] = {"version": LEXICON_VERSION, "n_themes": len(THEME_NAMES),
                    "fingerprint": fingerprint()}
    q18 = df[names["open_current"]]
    q19 = df[names["open_previous"]]
    T18, m18, _ = tag_frame(q18)
    T19, m19, _ = tag_frame(q19)
    p18, p19 = text_profile(q18, m18), text_profile(q19, m19)
    c18, c19 = coverage(T18, m18), coverage(T19, m19)
    R["text_profile"] = {"Q18": p18, "Q19": p19}
    R["text_coverage"] = {"Q18": c18, "Q19": c19}
    print("  Q18 answered %d (%.1f%%), median %g words, %.1f%% carry Bangla script"
          % (p18["n_answered"], p18["response_rate_pct"], p18["words_median"],
             p18["lang_bangla_pct"] + p18["lang_mixed_pct"]))
    print("  Q19 answered %d (%.1f%%), median %g words, %.1f%% carry Bangla script"
          % (p19["n_answered"], p19["response_rate_pct"], p19["words_median"],
             p19["lang_bangla_pct"] + p19["lang_mixed_pct"]))
    print("  lexicon coverage: Q18 %.1f%% of answers tagged (%d untagged), Q19 %.1f%%"
          % (c18["coverage_pct"], c18["n_uncovered"], c19["coverage_pct"]))
    na18 = int(nonanswer_mask(q18).sum())
    na19 = int(nonanswer_mask(q19).sum())
    print("  procedural non-answers coded explicitly: Q18 %d, Q19 %d" % (na18, na19))
    R["nonanswers"] = {"Q18": na18, "Q19": na19}
    savefig(fig_text_profile(p18, p19, c18, c19), "text_profile", "what the free text is")

    prev18 = prevalence(T18, m18)
    tt, corrected_alpha = theme_by_cluster(T18, m18, labels, pnames)
    savetab(tt, "table12a_themes_by_persona")
    nsig = int(tt["sig_bonferroni"].sum())
    print("\n  %d of %d themes differ across personas after Bonferroni correction "
          "(alpha = %.4f)." % (nsig, len(tt), corrected_alpha))
    if nsig:
        print("  The clusterer never saw this text, so each of those is independent "
              "corroboration\n  of a persona built purely from Likert answers.")
    else:
        print("  This is a NULL result and is reported as one: at this k the personas do\n"
              "  not differ in what students volunteer, once the correction for %d tests\n"
              "  is applied. The uncorrected pattern (see the table) leans the right way -\n"
              "  financial and living-conditions themes both peak in the financially\n"
              "  strained persona - but it does not clear the bar, so it is not claimed\n"
              "  as support." % len(THEME_NAMES))
    R["themes_significant"] = nsig
    savefig(fig_theme_by_cluster(tt, pnames), "themes_by_persona", "THEME CORROBORATION")

    U = tag_unasked(q18)
    unasked_pct = (100 * U[np.asarray(m18)].mean()).round(1).sort_values(ascending=False)
    savetab(unasked_pct.to_frame("pct_of_answering_students"), "table12b_unasked_stressors")
    R["unasked_stressors"] = unasked_pct.to_dict()
    topics, tf_stats = tfidf_nmf_crosscheck(q18, m18)
    savetab(topics, "table12c_tfidf_nmf_topics")
    print("\n  TF-IDF cross-check: %.1f%% of Bangla-script answers vectorise to an "
          "all-zero row\n  (Latin-script: %.1f%%). That failure is the evidence for the "
          "code-mixed lexicon."
          % (tf_stats["empty_among_bangla_pct"], tf_stats["empty_among_latin_pct"]))
    R["tfidf_nmf"] = tf_stats
    savefig(fig_unasked(unasked_pct, tf_stats), "unasked_stressors", "unasked stressors")

    pvu = prompted_vs_unprompted(desc, prev18)
    savetab(pvu, "table12d_prompted_vs_unprompted")
    shift, n_both = q18_q19_shift(T18, T19, m18, m19)
    savetab(shift, "table12e_q19_to_q18_shift")
    print("\n  Q19->Q18 within-person comparison over %d students who answered both." % n_both)
    R["q19_q18_pairs"] = n_both

    strain = items[ITEMS].astype(float).mean(axis=1)
    nulls = text_null_checks(q18, m18, strain)
    R["null_checks"] = nulls
    print("\n  Null check A: %s" % nulls["A_verdict"])
    print("  Null check B: %s" % nulls["B_verdict"])

    head("SECTION 20  SUPERVISED INTERPRETABILITY  (rules, not prediction)")
    Xsup = spaces[PRIMARY]["frame"].copy()
    clf, rules, rules_text, tstats = tree_rules(Xsup, labels, pnames)
    savetab(rules, "table13a_tree_rules", fmt="%.1f")
    print("\n  %d-fold CV accuracy = %.3f (SD %.3f), train = %.3f, %d leaves"
          % (tstats["cv_folds"], tstats["cv_accuracy_mean"], tstats["cv_accuracy_sd"],
             tstats["train_accuracy"], tstats["n_leaves"]))
    print("\n" + rules_text)
    with open(os.path.join(TAB_DIR, "table13b_tree_text.txt"), "w", encoding="utf-8") as fh:
        fh.write(rules_text)
    R["tree"] = tstats
    savefig(fig_tree(clf, Xsup, pnames), "decision_tree", "persona rule set")
    _, imp = rf_importance(Xsup, labels)
    savetab(imp, "table13c_permutation_importance", fmt="%.4f")
    R["permutation_importance"] = imp["permutation_importance"].to_dict()
    savefig(fig_importance(imp), "permutation_importance", "permutation importance")

    head("SECTION 21  PERSONA CARDS")
    recs = {pnames[i]: recommendation(pnames[i], prof.iloc[i], cols) for i in range(k)}
    savefig(persona_cards(prof, cols, pnames, bgtab, tt, decision, recs), "persona_cards",
            "PERSONA CARDS")
    for nm, rc in recs.items():
        print("  %-46s -> %s" % (nm[:46], rc))
    R["recommendations"] = recs

    head("SECTION 22  PERSIST")
    bundle = {
        "model": model, "scaler": spaces[PRIMARY]["scaler"],
        "columns": cols, "subscale_spec": spaces[PRIMARY]["subscale_spec"],
        "includes_cgpa": spaces[PRIMARY]["includes_cgpa"],
        "names": pnames, "k": k, "reversed_items": REVERSED,
        "lexicon_version": LEXICON_VERSION, "lexicon_fingerprint": fingerprint(),
        "random_state": RANDOM_STATE, "space": PRIMARY,
    }
    assignments = pd.DataFrame({
        "row": np.arange(len(labels)),
        "cluster": labels,
        "persona": [pnames[c] for c in labels],
        "split": ["train" if i in set(tr_idx.tolist()) else "holdout" for i in range(len(labels))],
        "year": bg["year"].to_numpy(), "cgpa": bg["cgpa"].to_numpy(),
        "gender": bg["gender"].to_numpy(), "living": bg["living"].to_numpy(),
        "department": bg["department"].to_numpy(), "backlog": bg["backlog"].to_numpy(),
        **{c: np.round(spaces[PRIMARY]["frame"][c].to_numpy(), 3) for c in cols},
        "silhouette_sample": np.round(silhouette_samples(Zall, labels), 3),
    })

    arff = spaces[PRIMARY]["frame"].copy()
    arff["cluster"] = ["C%d" % c for c in labels]
    write_arff(arff, os.path.join(OUT_DIR, "stress_prepared.arff"),
               "kuet_stress_personas", numeric=set(cols),
               nominal={"cluster": ["C%d" % c for c in range(k)]})

    head("SECTION 23  REUSE DEMO")
    demo_raw = pd.DataFrame([
        {**{it: 5 for it in ITEMS}, "cgpa": "2.50–2.99"},
        {**{it: 2 for it in ITEMS}, "cgpa": "3.80–4.00"},
        {**{it: 3 for it in ITEMS}, "AskTeacher": 1, "Feedback": 1,
         "Financial": 5, "JobWorry": 5, "cgpa": "3.50–3.79"},
    ])
    print("  scoring three synthetic respondents through the frozen instrument:")
    print(assign_persona(demo_raw, bundle).to_string())

    R["runtime_seconds"] = round(time.time() - t0, 1)
    R["n_figures"] = _FIG_N[0]
    R["limitations"] = [
        "Structure is genuinely weak: silhouette ~%.2f. These are a defensible, stable, "
        "interpretable segmentation of a continuum, not %d naturally separated populations."
        % (hsum["silhouette_train"], k),
        "Subscale reliabilities are modest (short facets on heterogeneous topics); reported, not hidden.",
        "The subscales come from an EFA on the same data used to cluster; the 70/30 holdout "
        "partially mitigates this, but no independent confirmatory sample exists.",
        "CGPA is self-reported and banded, and including it in the primary model costs it as "
        "a validation variable.",
        "Sample imbalance: %.0f%% male, %.0f%% in years 1-2, %d departments; personas describe "
        "the dominant strata best."
        % (100 * (bg["gender"] == "Male").mean(),
           100 * bg["year"].isin(["1st Year", "2nd Year"]).mean(),
           bg["department"].nunique()),
        "The lexicon is single-coded: no second coder, so no inter-rater reliability figure "
        "and theme prevalences are lower bounds.",
        "The free text is never a clustering input, which is what makes theme-cluster "
        "agreement genuine corroboration - but it means this project cannot say whether the "
        "text alone would reproduce the same personas.",
        "Self-report, cross-sectional, single-site: no causal claim is available.",
    ]
    persist(bundle, assignments, R)

    head("SUMMARY")
    print("  k = %d  (%s)" % (k, decision["framing"]))
    for i, nm in enumerate(pnames):
        print("    C%d  %-46s n = %3d  (%.1f%%)"
              % (i, nm[:46], prof["n"].iloc[i], prof["pct"].iloc[i]))
    print("\n  silhouette (train) %.3f | differentiation %.1f%% shape | bootstrap ARI %.3f"
          % (hsum["silhouette_train"], 100 * ev.loc[k, "differentiation"],
             ev.loc[k, "bootstrap_ari"]))
    print("  %d figures, %d tables -> %s" % (_FIG_N[0],
                                             len(glob.glob(os.path.join(TAB_DIR, "*.csv"))),
                                             OUT_DIR))
    print("  runtime %.1f s" % R["runtime_seconds"])
    print("\n  LIMITATIONS (carried into the notebook output, not buried):")
    for L in R["limitations"]:
        print("    - %s" % L)
    return R


if __name__ == "__main__":
    main(sys.argv[1:])
