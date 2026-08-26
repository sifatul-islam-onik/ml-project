# -*- coding: utf-8 -*-
"""
Optional GPU branch: multilingual sentence embeddings over the free text.

The methodology report *asserts* that transformer embeddings are unreliable on
answers this short and this code-mixed. That is a reasonable prior, but on
Kaggle a GPU is free, so the claim can be tested instead of assumed - which
turns a stated assumption into a measured result and strengthens the write-up
either way.

What this does:
  * embeds both free-text fields with a multilingual model (LaBSE by default,
    which covers Bangla script, unlike English-only sentence encoders);
  * clusters the embeddings with k-means and HDBSCAN;
  * scores those clusters against the frozen lexicon themes with adjusted Rand
    and normalised mutual information, and against answer length - because if
    embedding clusters mostly track *how long* an answer is rather than what it
    says, that is exactly the failure the report predicted.

Entirely optional. If sentence-transformers or a network connection is missing,
`run()` returns a skipped-status dict and the pipeline continues. Nothing
downstream depends on it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C

DEFAULT_MODEL = "sentence-transformers/LaBSE"
FALLBACK_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def available():
    """True when sentence-transformers can be imported."""
    try:
        import sentence_transformers  # noqa: F401
        return True
    except Exception:
        return False


def _device():
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def embed(texts, model_name=DEFAULT_MODEL, batch_size=64):
    """Encode a list of strings; returns (matrix, info)."""
    from sentence_transformers import SentenceTransformer

    dev = _device()
    try:
        model = SentenceTransformer(model_name, device=dev)
        used = model_name
    except Exception:
        model = SentenceTransformer(FALLBACK_MODEL, device=dev)
        used = FALLBACK_MODEL

    X = model.encode(list(texts), batch_size=batch_size, show_progress_bar=False,
                     convert_to_numpy=True, normalize_embeddings=True)
    return X, {"model": used, "device": dev, "dim": int(X.shape[1])}


def run(texts, mask, theme_frame, k=None, model_name=DEFAULT_MODEL):
    """Embed, cluster, and compare against the lexicon. Never raises."""
    if not available():
        return {"status": "skipped",
                "reason": ("sentence-transformers is not installed. On Kaggle: turn Internet ON "
                           "in the notebook settings and run "
                           "`pip install -q sentence-transformers`.")}
    try:
        from sklearn.cluster import HDBSCAN, KMeans
        from sklearn.metrics import (adjusted_rand_score, normalized_mutual_info_score,
                                     silhouette_score)

        sel = mask.to_numpy()
        answers = texts.fillna("").astype(str)[sel].tolist()
        X, info = embed(answers, model_name=model_name)

        # Reference labelling from the frozen lexicon: the single dominant theme
        # per answer, so a hard partition can be compared against a hard partition.
        T = theme_frame[sel]
        dominant = T.to_numpy().argmax(axis=1)
        has_theme = T.to_numpy().sum(axis=1) > 0

        k = k or int(T.to_numpy().sum(axis=0).astype(bool).sum())
        k = max(2, min(k, 13))

        km = KMeans(n_clusters=k, n_init=20, random_state=C.RANDOM_STATE).fit(X)
        hdb = HDBSCAN(min_cluster_size=15).fit(X)

        lengths = np.array([len(a.split()) for a in answers], dtype=float)
        length_bins = pd.qcut(lengths, q=min(4, len(np.unique(lengths))),
                              labels=False, duplicates="drop")

        out = {
            "status": "ok",
            **info,
            "n_texts": int(len(answers)),
            "kmeans_k": int(k),
            "kmeans_silhouette": round(float(silhouette_score(X, km.labels_)), 4),
            "hdbscan_n_clusters": int(len(set(hdb.labels_)) - (1 if -1 in hdb.labels_ else 0)),
            "hdbscan_pct_noise": round(float(100 * (hdb.labels_ == -1).mean()), 1),
            "agreement_with_lexicon": {
                "kmeans_ari": round(float(adjusted_rand_score(dominant[has_theme],
                                                             km.labels_[has_theme])), 4),
                "kmeans_nmi": round(float(normalized_mutual_info_score(dominant[has_theme],
                                                                      km.labels_[has_theme])), 4),
                "hdbscan_ari": round(float(adjusted_rand_score(dominant[has_theme],
                                                              hdb.labels_[has_theme])), 4),
            },
            "confound_with_answer_length": {
                "kmeans_nmi_with_length_quartile": round(
                    float(normalized_mutual_info_score(length_bins, km.labels_)), 4),
                "note": ("If this is comparable to or larger than the agreement with the "
                         "lexicon, the embedding clusters are sorting answers by length "
                         "rather than by stressor - the failure mode the methodology "
                         "report predicted for text this short."),
            },
        }
        agree = out["agreement_with_lexicon"]["kmeans_nmi"]
        conf = out["confound_with_answer_length"]["kmeans_nmi_with_length_quartile"]
        out["verdict"] = (
            "embedding clusters recover the lexicon themes only weakly (NMI=%.2f) and track "
            "answer length about as strongly (NMI=%.2f); the lexicon remains the primary "
            "text method" % (agree, conf)
            if agree < 0.35 or conf >= agree else
            "embedding clusters recover the lexicon themes substantially (NMI=%.2f), so the "
            "transformer route is viable on this corpus after all" % agree)
        return out
    except Exception as exc:  # pragma: no cover - defensive by design
        return {"status": "failed", "reason": "%s: %s" % (type(exc).__name__, exc)}
