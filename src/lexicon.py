# -*- coding: utf-8 -*-
"""
The frozen code-mixed stressor lexicon (methodology 5.3).

Thirteen themes, each a regular-expression pattern list covering English terms,
Bangla script and romanised Bangla. Multi-label by design: one answer can carry
several themes, because "lab reports and money problems" is two stressors, not
a tie to be broken.

Why a lexicon rather than an off-the-shelf NLP pipeline, restated here so the
choice travels with the code:
  * the answers are short (median 5 words, ~28% are two words or fewer), so
    contextual embeddings have almost no context to work with;
  * roughly one answer in eight carries Bangla script, and a further ~2% is
    romanised Bangla, which English stemmers and stopword lists silently damage;
  * the question asks for a *source of stress*, so sentiment is negative by
    construction and carries no between-student variance.

This module is frozen: the patterns are the published instrument. It is the
single source of truth; the notebook carries a verbatim inline copy and both
print `fingerprint()` into results.json, so any drift between the two is
detectable from the outputs rather than having to be trusted.

Frozen note: the patterns are the published instrument. Changing them
after seeing results would invalidate the prevalence figures, so any edit should
be a new version with the whole pipeline re-run.
"""
from __future__ import annotations

import hashlib
import re

import numpy as np
import pandas as pd

LEXICON_VERSION = "1.0"

THEMES = {
    "Lab & coursework load": [
        r"\blabs?\b", r"lab ?report", r"lab ?task", r"lab ?test", r"labwork", r"lap ?report",
        r"\bassignment", r"\bprojects?\b", r"presentation", r"sessional", r"\bquiz", r"\bviva",
        r"\bcts?\b", r"workload", r"\bdrawing\b", r"\bworkshop\b", r"\bstudio\b",
        r"ল্যাব", r"রিপোর্ট", r"অ্যাসাইনমেন্ট"],
    "Exams & results": [
        r"\bexam", r"\bresults?\b", r"\bcgpa\b", r"\bcg\b", r"\bgpa\b", r"\bmid ?term",
        r"term final", r"\bmarks?\b", r"\bgrade", r"\bbacklog\b", r"\bfail",
        r"পরীক্ষা", r"রেজাল্ট", r"ফলাফল", r"সিজি"],
    "Time & schedule pressure": [
        r"time manage", r"\bschedule", r"\broutine\b", r"8 ?am", r"short time", r"little time",
        r"session ?jam", r"\bsyllabus\b", r"deadline", r"\bpile", r"\battendance\b",
        r"\bshort pl\b", r"\bbreak\b", r"সময়", r"রুটিন", r"ক্লাস", r"সিলেবাস"],
    "Career & job uncertainty": [
        r"\bjobs?\b", r"\bcareer\b", r"\bfuture\b", r"graduat", r"\bskills?\b", r"placement",
        r"higher stud", r"\bscholarship\b", r"\babroad\b", r"job market",
        r"চাকরি", r"ক্যারিয়ার", r"ভবিষ্যৎ", r"স্কিল"],
    "Financial stress": [
        r"financ", r"\bmoney\b", r"\btaka\b", r"\bcosts?\b", r"\bexpens", r"tuition",
        r"middle class", r"\bincome\b", r"আর্থিক", r"টাকা", r"খরচ"],
    "Family & homesickness": [
        r"\bfamily\b", r"\bhome\b", r"homesick", r"away from home", r"far from home",
        r"\bparents?\b", r"\balone\b", r"lonel", r"\bmiss(ing)?\b",
        r"পরিবার", r"বাসা", r"একা", r"একাকিত্ব", r"দূরে"],
    "Teachers & teaching quality": [
        r"\bteachers?\b", r"\bfaculty\b", r"\bsir\b", r"instructor", r"lectur",
        r"can'?t understand", r"not understand", r"\bguideline", r"শিক্ষক", r"স্যার"],
    "Peers, comparison & campus climate": [
        r"compar", r"keep up with", r"\bpeers?\b", r"classmate", r"selfish", r"\btoxic\b",
        r"ragging", r"\bfriends?\b", r"politic", r"\bbully", r"তুলনা", r"বন্ধু", r"রাজনীতি"],
    "Living conditions & food": [
        r"\bfoods?\b", r"\bmess\b", r"\bhall\b", r"canteen", r"\bmeals?\b", r"breakfast",
        r"hostel", r"\bwater\b", r"transport", r"commut", r"\blifestyle\b",
        r"খাবার", r"হল", r"মেস", r"নাস্তা"],
    "Sleep & health": [
        r"\bsleep", r"insomnia", r"\bhealth", r"\bsick\b", r"\btired\b", r"exhaust",
        r"depress", r"anxiet", r"\bmental\b", r"ঘুম", r"স্বাস্থ্য"],
    "Self-regulation & procrastination": [
        r"procrastinat", r"irregular", r"\blazy\b", r"laziness", r"motivat", r"\bfocus\b",
        r"distract", r"\bdiscipline\b", r"অলস", r"মনোযোগ"],
    "Curriculum & non-departmental courses": [
        r"non.?dept", r"non.?departmental", r"memoriz", r"rote", r"curriculum",
        r"course ?load", r"credit hour", r"\bhum\b"],
    "Generic academic pressure": [
        r"academic", r"study pressure", r"\bstudies\b", r"porasuna", r"porashona",
        r"পড়াশোনা", r"পড়ালেখা", r"পড়া", r"চাপ"],
}

THEME_NAMES = list(THEMES)

#: Compiled once; the tagger runs over ~1,700 answers twice per pipeline run.
_COMPILED = {t: [re.compile(p, re.IGNORECASE) for p in pats] for t, pats in THEMES.items()}

BANGLA_RE = re.compile(r"[ঀ-৿]")
LATIN_RE = re.compile(r"[A-Za-z]")

#: Romanised-Bangla cues, used only to size that slice of the corpus.
ROMANISED_CUES = [
    "porasuna", "porashona", "porar", "chap", "tension", "ovab", "obhab", "somoy",
    "onek", "kichu", "amar", "ami", "kore", "kora", "hoy", "nai", "valo", "bhalo",
    "khub", "beshi", "besi", "problem er", "tar por",
]


def tag_text(text):
    """Return the list of themes matched in one answer."""
    s = str(text)
    return [t for t, pats in _COMPILED.items() if any(p.search(s) for p in pats)]


def answered_mask(series):
    """True where the free-text field has any non-whitespace content."""
    return series.fillna("").astype(str).str.strip().str.len() > 0


def tag_frame(series):
    """Multi-label 0/1 theme matrix for a free-text column (unanswered rows are all-zero)."""
    mask = answered_mask(series)
    txt = series.fillna("").astype(str)
    tags = [tag_text(t) if a else [] for t, a in zip(txt, mask)]
    T = pd.DataFrame({th: [int(th in ts) for ts in tags] for th in THEME_NAMES},
                     index=series.index)
    return T, mask, tags


def language_of(text):
    bn = bool(BANGLA_RE.search(text))
    lat = bool(LATIN_RE.search(text))
    if bn and lat:
        return "mixed"
    if bn:
        return "bangla"
    return "latin"


def text_profile(series, mask):
    """Length and script composition of one free-text field."""
    s = series.fillna("").astype(str)[mask]
    wl = s.str.split().str.len()
    lc = s.map(language_of).value_counts()
    # Romanised Bangla is counted only among Latin-script answers: an answer that
    # already carries Bangla script is captured by the script counts above.
    low = s.str.lower()
    latin_only = s.map(language_of) == "latin"
    romanised = int(low[latin_only].apply(
        lambda t: any(c in t for c in ROMANISED_CUES)).sum())
    n = len(s)
    return {
        "n_answered": int(mask.sum()),
        "response_rate_pct": round(float(100 * mask.mean()), 1),
        "words_mean": round(float(wl.mean()), 1),
        "words_median": float(wl.median()),
        "words_max": int(wl.max()),
        "chars_mean": round(float(s.str.len().mean()), 1),
        "pct_le2_words": round(float(100 * (wl <= 2).mean()), 1),
        "pct_ge10_words": round(float(100 * (wl >= 10).mean()), 1),
        "lang_latin_pct": round(float(100 * lc.get("latin", 0) / n), 1),
        "lang_bangla_pct": round(float(100 * lc.get("bangla", 0) / n), 1),
        "lang_mixed_pct": round(float(100 * lc.get("mixed", 0) / n), 1),
        "romanised_n": romanised,
        "romanised_pct": round(float(100 * romanised / n), 1),
        "distinct_answers": int(s.nunique()),
    }


def coverage(T, mask):
    """Share of answered rows receiving at least one theme, and themes per answer."""
    counts = T.to_numpy().sum(axis=1)[mask.to_numpy()]
    return {
        "coverage_pct": round(float(100 * (counts >= 1).mean()), 1),
        "mean_themes_per_answer": round(float(counts.mean()), 2),
        "pct_multi_theme": round(float(100 * (counts >= 2).mean()), 1),
        "n_uncovered": int((counts == 0).sum()),
    }


def prevalence(T, mask):
    """Percentage of answered students mentioning each theme, descending."""
    sub = T[mask.to_numpy()]
    return (100 * sub.mean()).round(1).sort_values(ascending=False)


def export_patterns():
    """The lexicon as a flat table, so the frozen instrument ships with the report."""
    rows = []
    for t, pats in THEMES.items():
        rows.append({"theme": t, "n_patterns": len(pats), "patterns": " | ".join(pats)})
    return pd.DataFrame(rows).set_index("theme")


def fingerprint():
    """SHA-256 over the frozen pattern set.

    The notebook inlines these patterns so it can run on Kaggle without `src/` on
    the path. That duplication is only safe if drift is detectable, so both
    copies hash the same canonical string and write the digest into results.json.
    """
    canon = "|".join(
        t + "::" + ",".join(THEMES[t]) for t in sorted(THEMES))
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()[:16]


def untagged_answers(series, T, mask, limit=None):
    """Answered rows that matched no theme - the lexicon's known blind spot.

    Reported rather than hidden: methodology S8 records theme prevalences as
    lower bounds precisely because this set is not a random subset of answers.
    """
    counts = T.to_numpy().sum(axis=1)
    sel = mask.to_numpy() & (counts == 0)
    out = series.fillna("").astype(str)[sel]
    return out if limit is None else out.head(limit)
