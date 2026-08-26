# -*- coding: utf-8 -*-
"""
Optional supervised extension (methodology 5.7) and the incremental-value test.

Two questions, both answered against an explicit baseline so a number like "44%
accuracy" cannot be mistaken for a result when the majority class is 33%:

  1. Are the discovered profiles predictable from information the clustering
     never saw (demographics, backlog, free-text themes)? If yes, the profiles
     correspond to something outside the twelve items. If no - which is the
     likely outcome given the weak structure - that is reported as the finding.

  2. Do free-text theme features add predictive value over the closed-ended
     items and demographics? This is the nested feature-set comparison the
     project's research question asks for.

Every score is a stratified 5-fold cross-validated mean with its standard
deviation, and a permutation test is run on the headline model so the gap over
baseline is checked against chance rather than assumed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (StratifiedKFold, cross_val_score,
                                     cross_validate, permutation_test_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import config as C


def _cv(random_state=0):
    return StratifiedKFold(C.CV_FOLDS, shuffle=True, random_state=random_state)


def _models():
    return {
        "RandomForest": RandomForestClassifier(
            n_estimators=400, min_samples_leaf=3, random_state=0, n_jobs=-1),
        "LogisticRegression": make_pipeline(
            StandardScaler(with_mean=False),
            LogisticRegression(max_iter=2000)),
    }


def evaluate(X, y, model_name="RandomForest"):
    """Cross-validated accuracy and macro-F1 for one feature set."""
    clf = _models()[model_name]
    res = cross_validate(clf, X, y, cv=_cv(), scoring=("accuracy", "f1_macro"), n_jobs=1)
    return {
        "accuracy": round(float(res["test_accuracy"].mean()), 4),
        "accuracy_sd": round(float(res["test_accuracy"].std()), 4),
        "macro_f1": round(float(res["test_f1_macro"].mean()), 4),
        "macro_f1_sd": round(float(res["test_f1_macro"].std()), 4),
    }


def baseline(y):
    """Majority-class baseline, computed the same way as the models."""
    X = np.zeros((len(y), 1))
    dummy = DummyClassifier(strategy="most_frequent")
    acc = cross_val_score(dummy, X, y, cv=_cv(), scoring="accuracy").mean()
    return {"accuracy": round(float(acc), 4), "accuracy_sd": 0.0,
            "macro_f1": None, "macro_f1_sd": None}


def feature_set_comparison(feature_sets, y, label, model_name="RandomForest"):
    """Compare nested feature sets against the majority-class baseline."""
    rows = [{"features": "Majority-class baseline", "n_features": 0, **baseline(y)}]
    for name, X in feature_sets:
        rows.append({"features": name, "n_features": int(X.shape[1]),
                     **evaluate(X, y, model_name)})
    best = max(rows[1:], key=lambda r: r["accuracy"])
    lift = best["accuracy"] - rows[0]["accuracy"]
    return {
        "target": label,
        "model": model_name,
        "rows": rows,
        "best_feature_set": best["features"],
        "best_accuracy": best["accuracy"],
        "lift_over_baseline": round(float(lift), 4),
        "verdict": ("adds no usable predictive value over the majority-class baseline"
                    if lift < 0.03 else
                    "beats the majority-class baseline by %.1f points" % (100 * lift)),
    }


def permutation_check(X, y, model_name="RandomForest", n_permutations=200):
    """Permutation test: is the cross-validated score better than label chance?"""
    clf = _models()[model_name]
    score, perm_scores, p = permutation_test_score(
        clf, X, y, cv=_cv(), scoring="accuracy",
        n_permutations=n_permutations, random_state=C.RANDOM_STATE, n_jobs=1)
    return {
        "observed_accuracy": round(float(score), 4),
        "permuted_mean": round(float(np.mean(perm_scores)), 4),
        "permuted_p95": round(float(np.percentile(perm_scores, 95)), 4),
        "p_value": float(p),
        "n_permutations": int(n_permutations),
    }


def importances(X, y, model_name="RandomForest", top=15):
    """Permutation importance on a held-out split.

    Impurity importance is biased toward high-cardinality features, which matters
    here because department has fourteen levels; permutation importance on unseen
    data does not have that failure mode.
    """
    from sklearn.model_selection import train_test_split

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.25, random_state=C.RANDOM_STATE, stratify=y)
    clf = _models()[model_name].fit(Xtr, ytr)
    r = permutation_importance(clf, Xte, yte, n_repeats=15,
                               random_state=C.RANDOM_STATE, n_jobs=1)
    df = pd.DataFrame({
        "feature": list(X.columns),
        "importance": r.importances_mean.round(5),
        "sd": r.importances_std.round(5),
    }).sort_values("importance", ascending=False).set_index("feature")
    return df.head(top), round(float(clf.score(Xte, yte)), 4)


def profile_recovery(cluster_labels, demo, themes):
    """Can held-out information recover the cluster a student landed in?

    The clustering used only the twelve items. Demographics and text themes are
    genuinely external, so this is a real external-validity test rather than a
    re-description of the input.
    """
    y = pd.Series(cluster_labels).astype(str)
    sets = [
        ("Demographics only", demo),
        ("Text themes only", themes),
        ("Demographics + text themes", pd.concat([demo, themes], axis=1)),
    ]
    return feature_set_comparison(sets, y, "cluster profile")


def incremental_text_value(items, demo, themes, y, label):
    """Nested comparison: do text features add anything over items + demographics?"""
    sets = [
        ("Survey items only (12)", items),
        ("Text themes only (13)", themes),
        ("Items + demographics", pd.concat([items, demo], axis=1)),
        ("Items + demographics + text", pd.concat([items, demo, themes], axis=1)),
    ]
    report = feature_set_comparison(sets, y, label)
    rows = {r["features"]: r for r in report["rows"]}
    with_text = rows["Items + demographics + text"]["accuracy"]
    without = rows["Items + demographics"]["accuracy"]
    report["text_increment"] = round(float(with_text - without), 4)
    report["text_verdict"] = (
        "free-text theme features do not improve prediction over items + demographics"
        if with_text - without < 0.01 else
        "free-text theme features add %.1f accuracy points" % (100 * (with_text - without)))
    return report
