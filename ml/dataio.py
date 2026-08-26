# -*- coding: utf-8 -*-
"""
Loading, schema validation, integrity auditing and export.

Nothing here silently repairs the data. Every check writes a number into the
audit dictionary so the report can state what was found (including "nothing"),
which is what the dataset-documentation sheet required by the course asks for.
"""
from __future__ import annotations

import json
import os
import re

import numpy as np
import pandas as pd

from . import config as C


# --------------------------------------------------------------------------
# Load
# --------------------------------------------------------------------------
def load_raw(path=None):
    """Read the workbook and return (dataframe, resolved_path)."""
    resolved = C.resolve_data_path(path)
    df = pd.read_excel(resolved)
    return df, resolved


def validate_schema(df):
    """Assert the positional column mapping still matches the header text.

    The export is addressed by position because the headers are long bilingual
    sentences. That is fragile if the form is ever edited, so each mapped index
    is fingerprinted against a keyword; a mismatch aborts the run instead of
    producing a plausible-looking but wrong analysis.
    """
    cols = list(df.columns)
    if len(cols) != 21:
        raise ValueError("Expected 21 columns in the export, found %d." % len(cols))

    problems = []
    for idx, cue in C.SCHEMA_FINGERPRINT.items():
        if cue.lower() not in str(cols[idx]).lower():
            problems.append("column %d (%r) does not contain %r" % (idx, str(cols[idx])[:60], cue))
    if problems:
        raise ValueError("Column mapping no longer matches the export:\n  " + "\n  ".join(problems))

    return {"n_columns": len(cols), "columns": [str(c) for c in cols]}


def column_names(df):
    """Map the logical names in config.COL_IDX onto real column labels."""
    cols = list(df.columns)
    names = {k: cols[i] for k, i in C.COL_IDX.items()}
    names["likert"] = [cols[i] for i in C.LIKERT_IDX]
    return names


# --------------------------------------------------------------------------
# Integrity audit
# --------------------------------------------------------------------------
def audit(df, names):
    """Completeness, range and duplication checks over the raw export."""
    lik = df[names["likert"]]
    closed = [names[k] for k in ("year", "cgpa", "gender", "living", "backlog", "department")]

    out_of_range = int(((lik < 1) | (lik > 5)).to_numpy().sum())
    non_integer = int((lik != lik.round()).to_numpy().sum())

    ts = pd.to_datetime(df[names["timestamp"]], errors="coerce")
    by_day = ts.dt.date.value_counts()

    return {
        "n_rows": int(len(df)),
        "closed_ended_missing": int(df[closed].isna().to_numpy().sum() + lik.isna().to_numpy().sum()),
        "open_current_missing": int(df[names["open_current"]].isna().sum()),
        "open_previous_missing": int(df[names["open_previous"]].isna().sum()),
        "exact_duplicate_rows": int(df.duplicated().sum()),
        "duplicate_ignoring_timestamp": int(df.drop(columns=[names["timestamp"]]).duplicated().sum()),
        "likert_out_of_range": out_of_range,
        "likert_non_integer": non_integer,
        "collection_start": str(ts.min()),
        "collection_end": str(ts.max()),
        "collection_days": int((ts.max() - ts.min()).days) + 1,
        "peak_day": str(by_day.idxmax()),
        "peak_day_n": int(by_day.max()),
    }


def response_style(items):
    """Straight-lining / acquiescence diagnostics on the direction-aligned block."""
    within_sd = items.std(axis=1)
    same_all = int((items.nunique(axis=1) == 1).sum())

    def longest_run(row):
        best = run = 1
        for a, b in zip(row[:-1], row[1:]):
            run = run + 1 if a == b else 1
            best = max(best, run)
        return best

    runs = items.apply(lambda r: longest_run(list(r)), axis=1)
    n = len(items)
    return {
        "straight_lining_n": same_all,
        "straight_lining_pct": round(100 * same_all / n, 2),
        "low_variance_n": int((within_sd < 0.35).sum()),
        "low_variance_pct": round(100 * float((within_sd < 0.35).mean()), 2),
        "mean_within_row_sd": round(float(within_sd.mean()), 3),
        "long_run_ge8_n": int((runs >= 8).sum()),
        "long_run_ge8_pct": round(100 * float((runs >= 8).mean()), 2),
        "acquiescence_pct_top_box": round(100 * float((items == 5).to_numpy().mean()), 1),
        "pct_using_all_five_points": round(100 * float((items.nunique(axis=1) == 5).mean()), 1),
    }


# --------------------------------------------------------------------------
# Export
# --------------------------------------------------------------------------
_ARFF_SAFE = re.compile(r"[^0-9A-Za-z_]+")


def _arff_name(s):
    return _ARFF_SAFE.sub("_", str(s)).strip("_")


def _arff_value(v):
    if pd.isna(v):
        return "?"
    s = str(v)
    s = s.replace(chr(92), chr(92) * 2)
    s = s.replace("'", chr(92) + "'")
    s = s.replace("\n", " ").replace("\r", " ")
    return "'" + s + "'"


def write_arff(df, path, relation, numeric, nominal, string_cols):
    """Write an ARFF file so the same prepared table can be opened in WEKA.

    The methodology report specifies a WEKA workflow; this keeps that route open
    from the identical preprocessing the Python pipeline uses, instead of
    preparing the data twice and hoping the two agree.
    """
    lines = ["@RELATION " + _arff_name(relation), ""]
    order = []
    for c in df.columns:
        nm = _arff_name(c)
        if c in numeric:
            lines.append("@ATTRIBUTE %s NUMERIC" % nm)
        elif c in nominal:
            levels = ",".join(_arff_value(v) for v in nominal[c])
            lines.append("@ATTRIBUTE %s {%s}" % (nm, levels))
        elif c in string_cols:
            lines.append("@ATTRIBUTE %s STRING" % nm)
        else:
            continue
        order.append(c)
    lines += ["", "@DATA"]
    for _, row in df[order].iterrows():
        vals = []
        for c in order:
            v = row[c]
            if c in numeric:
                vals.append("?" if pd.isna(v) else ("%g" % float(v)))
            else:
                vals.append(_arff_value(v))
        lines.append(",".join(vals))

    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    return path


def jsonable(obj):
    """Recursively convert numpy scalars/arrays so json.dump never chokes."""
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return jsonable(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, float) and np.isnan(obj):
        return None
    if isinstance(obj, pd.Timestamp):
        return str(obj)
    return obj


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(jsonable(obj), fh, indent=2, ensure_ascii=False)
    return path


def save_table(df, name, index=True):
    """Write a results table to outputs/tables as UTF-8-BOM CSV (Excel-friendly)."""
    path = os.path.join(C.TAB_DIR, name + ".csv")
    df.to_csv(path, index=index, encoding="utf-8-sig")
    return path
