# -*- coding: utf-8 -*-
"""
Single source of truth for paths, column mapping, item metadata and plot tokens.

Every other module imports from here, so the pipeline behaves identically on a
local Windows machine and inside a Kaggle kernel: the only thing that changes is
where `resolve_data_path()` finds the workbook and where OUT_DIR points.
"""
from __future__ import annotations

import glob
import os

# --------------------------------------------------------------------------
# Environment detection
# --------------------------------------------------------------------------
ON_KAGGLE = os.path.isdir("/kaggle/input")

PKG_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(PKG_DIR)

#: Filenames this project has used for the same export, newest first. The
#: pipeline accepts any of them so a re-download from Google Forms (which always
#: lands as "... (Responses).xlsx") does not break the run.
DATA_FILENAMES = [
    "Academic Stress among Bangladeshi Engineering Students (Responses).xlsx",
    "Academic Stress among Bangladeshi Engineering Students (Finalv2).xlsx",
    "Academic Stress among Bangladeshi Engineering Students (Final).xlsx",
]
DATA_GLOB = "*Academic Stress among Bangladeshi Engineering Students*.xlsx"


def resolve_data_path(explicit: str | None = None) -> str:
    """Locate the response workbook.

    Search order: an explicit path / the STRESS_DATA env var -> the Kaggle input
    mounts -> the repository root. Raises with an actionable message rather than
    letting pandas fail on a path the user cannot see.
    """
    explicit = explicit or os.environ.get("STRESS_DATA")
    if explicit:
        if os.path.isfile(explicit):
            return explicit
        raise FileNotFoundError(f"Explicit data path does not exist: {explicit}")

    roots = []
    if ON_KAGGLE:
        roots += sorted(glob.glob("/kaggle/input/*"))
        roots.append("/kaggle/working")
    roots.append(ROOT)

    for base in roots:
        for name in DATA_FILENAMES:
            cand = os.path.join(base, name)
            if os.path.isfile(cand):
                return cand
        hits = sorted(glob.glob(os.path.join(base, DATA_GLOB)))
        if hits:
            return hits[0]

    raise FileNotFoundError(
        "Could not find the response workbook.\n"
        f"Looked for {DATA_FILENAMES[0]!r} (or {DATA_GLOB!r}) in: {roots}\n"
        "On Kaggle: attach the dataset via '+ Add Input'. Locally: set STRESS_DATA."
    )


OUT_DIR = "/kaggle/working" if ON_KAGGLE else os.path.join(ROOT, "outputs")
FIG_DIR = os.path.join(OUT_DIR, "figures")
TAB_DIR = os.path.join(OUT_DIR, "tables")
MODEL_DIR = os.path.join(OUT_DIR, "models")


def ensure_dirs() -> None:
    for d in (OUT_DIR, FIG_DIR, TAB_DIR, MODEL_DIR):
        os.makedirs(d, exist_ok=True)


# --------------------------------------------------------------------------
# Column mapping
# --------------------------------------------------------------------------
# The Google Forms export carries the full bilingual question text as the column
# header, so columns are addressed positionally and then verified against a
# keyword fingerprint (see dataio.validate_schema).
COL_IDX = {
    "timestamp": 0,
    "year": 1,
    "cgpa": 2,
    "gender": 3,
    "living": 4,
    "backlog": 11,
    "open_current": 18,
    "open_previous": 19,
    "department": 20,
}
LIKERT_IDX = [5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17]

#: Short keys for the twelve Likert items, in questionnaire order.
ITEMS = [
    "MissMeal", "PileUp", "SleepLoss", "LabStress", "ExamWorry", "ResultDemotiv",
    "CGPACompare", "AskTeacher", "Feedback", "Financial", "JobWorry", "SocioPol",
]

#: Q-numbers used in the dataset-characteristics report, for cross-referencing.
ITEM_QNUM = {it: f"Q{i + 1}" for i, it in enumerate(ITEMS)}

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

#: Positively worded items. Recoded as 6 - x so that HIGH always means MORE
#: strain / LESS support across the whole instrument (methodology report S5.2).
REVERSED = ["AskTeacher", "Feedback"]

#: One-word fingerprints used to assert the positional mapping still holds.
SCHEMA_FINGERPRINT = {
    1: "year", 2: "cgpa", 3: "gender", 4: "living",
    5: "breakfast", 6: "pile", 7: "sleep", 8: "lab", 9: "poor result",
    10: "motivation", 11: "backlog", 12: "compare", 13: "comfortable",
    14: "feedback", 15: "financial", 16: "job", 17: "instability",
    18: "biggest source", 19: "previous academic", 20: "department",
}

# --------------------------------------------------------------------------
# Category level orders (fixed so every table/figure sorts identically)
# --------------------------------------------------------------------------
YEAR_ORD = ["1st Year", "2nd Year", "3rd Year", "4th Year", "Other"]
CGPA_ORD = ["Below 2.50", "2.50\u20132.99", "3.00\u20133.49", "3.50\u20133.79", "3.80\u20134.00"]
GENDER_ORD = ["Male", "Female"]
LIVE_ORD = ["Hall", "Mess", "Family", "Other"]

#: Ordinal encodings for the two ordered background variables.
YEAR_ORDINAL = {"1st Year": 1, "2nd Year": 2, "3rd Year": 3, "4th Year": 4, "Other": 0}
CGPA_ORDINAL = {c: i + 1 for i, c in enumerate(CGPA_ORD)}

# --------------------------------------------------------------------------
# Modelling constants
# --------------------------------------------------------------------------
RANDOM_STATE = 42
K_RANGE = list(range(2, 9))          # k = 2..8, as specified in the methodology
N_INIT = 50                          # k-means restarts
PCA_VAR_TARGET = 0.95                # retain components covering 95% of variance
BOOTSTRAP_B = 200                    # resamples for the ARI stability check
SEED_TRIALS = 20                     # distinct seeds for the seed-stability check
CV_FOLDS = 5

# --------------------------------------------------------------------------
# Design tokens (validated light-surface palette, print-safe)
# --------------------------------------------------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASE = "#c3c2b7"
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
DIV5 = ["#b02f2f", "#e88a89", "#cfcec7", "#86b6ef", "#1c5cab"]
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
