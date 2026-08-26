# -*- coding: utf-8 -*-
"""
Unsupervised cross-checks over the free text, plus the two pre-specified null tests.

The lexicon in `lexicon.py` is the primary text method. This module supplies the
corroboration the methodology promises - a TF-IDF/NMF topic model and an LDA run
- and the two checks that were specified in advance so a null result cannot be
quietly dropped:

  1. non-response signal: do students who skip the free-text field differ in
     measured strain from those who answer?
  2. answer-length signal: does how much a student writes correlate with strain?

Both are reported whatever they show.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from . import config as C
from . import lexicon as LX


def tfidf_matrix(texts, max_features=1500, min_df=3):
    """TF-IDF over the answered rows.

    Deliberately Latin-script only, with a token pattern requiring three or more
    letters. That is a documented limitation, not an oversight: this branch is
    the cross-check, and the lexicon is what covers the Bangla-script answers.
    """
    vec = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2), min_df=min_df,
                          stop_words="english", token_pattern=r"[a-zA-Z][a-zA-Z']{2,}")
    X = vec.fit_transform(texts)
    return X, vec


def nmf_topics(X, vec, n_components=6, random_state=C.RANDOM_STATE):
    """Non-negative matrix factorisation over TF-IDF; returns topic terms and shares."""
    nmf = NMF(n_components=n_components, random_state=random_state,
              init="nndsvda", max_iter=800)
    W = nmf.fit_transform(X)
    terms = np.array(vec.get_feature_names_out())
    dominant = W.argmax(axis=1)
    topics = []
    for i, comp in enumerate(nmf.components_):
        topics.append({
            "topic": i,
            "top_terms": terms[np.argsort(comp)[::-1][:10]].tolist(),
            "share_pct": round(float(100 * (dominant == i).mean()), 1),
        })
    return topics, W, nmf


def lda_topics(texts, n_components=6, random_state=C.RANDOM_STATE, min_df=3):
    """LDA over raw counts, as a second opinion on the NMF solution."""
    cv = CountVectorizer(max_features=1500, min_df=min_df, stop_words="english",
                         token_pattern=r"[a-zA-Z][a-zA-Z']{2,}")
    Xc = cv.fit_transform(texts)
    lda = LatentDirichletAllocation(n_components=n_components, random_state=random_state,
                                    learning_method="batch", max_iter=50)
    Wl = lda.fit_transform(Xc)
    terms = np.array(cv.get_feature_names_out())
    dom = Wl.argmax(axis=1)
    topics = []
    for i, comp in enumerate(lda.components_):
        topics.append({
            "topic": i,
            "top_terms": terms[np.argsort(comp)[::-1][:10]].tolist(),
            "share_pct": round(float(100 * (dom == i).mean()), 1),
        })
    return {"topics": topics, "perplexity": round(float(lda.perplexity(Xc)), 1),
            "vocabulary": int(Xc.shape[1])}


def topic_model_report(texts):
    """Full topic-model cross-check, including how much text it cannot represent."""
    X, vec = tfidf_matrix(texts)
    zero_rows = int(np.asarray((X.sum(axis=1) == 0)).ravel().sum())
    topics, W, nmf = nmf_topics(X, vec)
    return {
        "tfidf_shape": list(X.shape),
        "vocabulary": int(X.shape[1]),
        "rows_with_no_features": zero_rows,
        "rows_with_no_features_pct": round(100 * zero_rows / X.shape[0], 1),
        "nmf_topics": topics,
        "lda": lda_topics(texts),
        "note": ("Rows with no features are answers written entirely in Bangla script or in "
                 "tokens below the min_df threshold. They are covered by the lexicon but "
                 "invisible to this Latin-script cross-check."),
    }, (X, vec, nmf, W)


def top_terms(texts, n=25, min_df=2):
    """Raw document-frequency ranking, for the descriptive table."""
    cv = CountVectorizer(min_df=min_df, stop_words="english",
                         token_pattern=r"[a-zA-Z][a-zA-Z']{2,}", binary=True)
    Xc = cv.fit_transform(texts)
    freq = np.asarray(Xc.sum(axis=0)).ravel()
    terms = np.array(cv.get_feature_names_out())
    order = np.argsort(freq)[::-1][:n]
    return [{"term": str(terms[i]), "n_answers": int(freq[i])} for i in order]


# --------------------------------------------------------------------------
# Pre-specified null checks
# --------------------------------------------------------------------------
def null_checks(strain, texts, mask):
    """The two checks specified in advance in methodology 5.3.

    `strain` is any per-student strain summary (the composite mean is used, as a
    descriptive index only). Both tests are reported with effect sizes, because
    at n ~ 1000 a p-value alone will call a trivial difference significant.
    """
    answered = strain[mask.to_numpy()]
    skipped = strain[~mask.to_numpy()]

    t, p = stats.ttest_ind(answered, skipped, equal_var=False)
    pooled = np.sqrt((answered.var(ddof=1) + skipped.var(ddof=1)) / 2)
    d = float((answered.mean() - skipped.mean()) / pooled) if pooled > 0 else float("nan")

    wl = texts.fillna("").astype(str).str.split().str.len()[mask.to_numpy()]
    rho, p_rho = stats.spearmanr(wl, answered)

    return {
        "non_response": {
            "n_answered": int(mask.sum()),
            "n_skipped": int((~mask).sum()),
            "mean_strain_answered": round(float(answered.mean()), 3),
            "mean_strain_skipped": round(float(skipped.mean()), 3),
            "difference": round(float(answered.mean() - skipped.mean()), 3),
            "welch_t": round(float(t), 3),
            "p": float(p),
            "cohens_d": round(d, 3),
            "verdict": ("no usable signal: skipping the free-text field does not indicate "
                        "different measured strain"
                        if p >= 0.05 or abs(d) < 0.2 else
                        "students who skip the field differ in measured strain"),
        },
        "answer_length": {
            "spearman_rho": round(float(rho), 3),
            "p": float(p_rho),
            "verdict": ("no usable signal: answer length does not track measured strain"
                        if p_rho >= 0.05 or abs(rho) < 0.1 else
                        "answer length tracks measured strain"),
        },
    }


def theme_strain_association(T, mask, strain, min_n=15):
    """Per-theme difference in strain between mentioners and non-mentioners.

    Bonferroni-corrected across the themes actually tested, because thirteen
    uncorrected t-tests will manufacture a significant result on their own.
    """
    rows = []
    m_all = mask.to_numpy()
    for t in LX.THEME_NAMES:
        flag = T[t].to_numpy().astype(bool)
        with_t = strain[flag & m_all]
        without = strain[(~flag) & m_all]
        if len(with_t) < min_n:
            continue
        stat, p = stats.ttest_ind(with_t, without, equal_var=False)
        pooled = np.sqrt((with_t.var(ddof=1) + without.var(ddof=1)) / 2)
        rows.append({
            "theme": t,
            "n_mentioning": int(len(with_t)),
            "mean_with": round(float(with_t.mean()), 3),
            "mean_without": round(float(without.mean()), 3),
            "difference": round(float(with_t.mean() - without.mean()), 3),
            "cohens_d": round(float((with_t.mean() - without.mean()) / pooled), 3) if pooled else None,
            "p": float(p),
        })
    if not rows:
        return {"bonferroni_alpha": None, "rows": []}
    alpha = 0.05 / len(rows)
    for r in rows:
        r["significant_bonferroni"] = bool(r["p"] < alpha)
    rows.sort(key=lambda r: -abs(r["difference"]))
    return {"n_tests": len(rows), "bonferroni_alpha": round(alpha, 5), "rows": rows}


def theme_by_group(T, mask, groups, min_n=20):
    """Theme prevalence within each level of a background variable."""
    out = {}
    m = mask.to_numpy()
    for g in pd.unique(groups):
        sel = m & (groups == g).to_numpy()
        if sel.sum() < min_n:
            continue
        out[str(g)] = {"n": int(sel.sum()),
                       **{t: round(float(100 * T.loc[sel, t].mean()), 1) for t in LX.THEME_NAMES}}
    return out
