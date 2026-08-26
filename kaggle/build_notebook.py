# -*- coding: utf-8 -*-
"""
Generate the Kaggle notebook from the `ml/` package.

The notebook is *generated* rather than hand-written so it can never drift from
the code that actually runs. Every analysis cell embeds the current source of
one module, so `stress_clustering_kaggle.ipynb` is a faithful, self-contained
snapshot: it needs no pip install, no GitHub clone, and no utility-script
attachment - only the dataset.

Regenerate after any change to ml/:   python kaggle/build_notebook.py
"""
from __future__ import annotations

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ML = os.path.join(ROOT, "ml")
OUT = os.path.join(HERE, "stress_clustering_kaggle.ipynb")

#: Import order matters - each module is written to disk before the next needs it.
MODULES = [
    "config", "dataio", "preprocess", "structure", "reduce",
    "cluster", "validate", "lexicon", "textmodel", "profile",
    "supervised", "figures", "embeddings", "run_pipeline",
]


def _lines(text):
    """nbformat stores source as a list of lines that still carry their newline.

    Splitting without re-attaching "\n" makes Jupyter join every line into one,
    which silently produces a notebook that looks fine in JSON and is unusable
    when opened.
    """
    parts = text.split("\n")
    return [p + "\n" for p in parts[:-1]] + ([parts[-1]] if parts[-1] else [])


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": _lines(text.strip())}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": _lines(text.strip("\n"))}


def _src(name):
    with open(os.path.join(ML, name + ".py"), encoding="utf-8") as fh:
        return fh.read()


def build():
    cells = []

    cells.append(md("""
# University Student Mental Stress Pattern Analysis - Unsupervised Learning

**KUET, CSE 4112 Machine Learning Laboratory**

Clustering 987 anonymous survey responses from Bangladeshi engineering students to
find interpretable *stress profiles*, with the free-text answers analysed through a
purpose-built code-mixed lexicon.

This notebook is generated from the project's `ml/` package, so what runs here is
exactly what runs locally. It is fully self-contained: **no pip install and no
internet are required** for the main pipeline.

## What it does

| Step | Method |
|---|---|
| Integrity audit | completeness, duplicates, range checks, straight-lining |
| Direction alignment | the two positively worded items recoded `6 - x` |
| Measurement structure | Cronbach's alpha, item-total, KMO, Bartlett |
| Dimensionality reduction | PCA + Horn's parallel analysis + varimax rotation |
| Clustering | k-means, Gaussian mixture (EM), Ward/average/complete hierarchical, spectral, Gower k-medoids |
| Choosing k | SSE elbow (Kneedle), silhouette, Davies-Bouldin, Calinski-Harabasz, BIC, CV log-likelihood, **gap statistic** |
| Validation | bootstrap ARI, seed stability, consensus matrix, classes-to-clusters against held-out demographics |
| Free text | frozen 13-theme code-mixed lexicon, NMF + LDA cross-check, two pre-specified null tests |
| Profiling | z-score profiles, data-derived names, eta-squared item ranking, recommendations |
| Supervised | profile recovery and the incremental value of text features, vs an explicit baseline |

## Setup (one manual step)

Attach the response workbook as a Kaggle dataset via **+ Add Input**. The loader
finds any file matching `*Academic Stress among Bangladeshi Engineering Students*.xlsx`
anywhere under `/kaggle/input`.

GPU is **not** needed. The optional embedding cross-check in the last section is
the only part that wants a GPU and internet.
"""))

    cells.append(md("## 0. Environment"))
    cells.append(code("""
import os, sys, platform, warnings
warnings.filterwarnings("ignore")

print("python  ", platform.python_version())
for m in ("numpy", "pandas", "scipy", "sklearn", "matplotlib", "joblib"):
    try:
        print("%-11s %s" % (m, __import__(m).__version__))
    except Exception as e:
        print("%-11s MISSING (%s)" % (m, e))

if os.path.isdir("/kaggle/input"):
    print("\\nAttached inputs:")
    for root, _, files in os.walk("/kaggle/input"):
        for f in files:
            print("   ", os.path.join(root, f))
else:
    print("\\nNot on Kaggle - running against the local project folder.")
"""))

    cells.append(md("""
## 1. The pipeline package

The next cells write the `ml/` package to disk and import it. Each cell is one
module; the docstring at the top of each explains the choices it makes and why.
"""))

    cells.append(code("""
import os, sys

# The module cells below use `%writefile ml/<name>.py`, which resolves against the
# working directory. On Kaggle that is /kaggle/working, so nothing outside the
# session is touched. If you run this notebook locally, run it from a scratch
# directory - otherwise it rewrites the project's own ml/ package with this snapshot.
os.makedirs("ml", exist_ok=True)
with open(os.path.join("ml", "__init__.py"), "w", encoding="utf-8") as fh:
    fh.write('__version__ = "1.0.0"\\n')
if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())
print("writing package into:", os.path.abspath("ml"))
"""))

    for name in MODULES:
        src = _src(name)
        first = src.split('"""')[1].strip().split("\n")[0] if '"""' in src else name
        cells.append(md("### `ml/%s.py` - %s" % (name, first)))
        # The %%writefile magic writes the cell body verbatim, so the module source
        # needs no escaping and stays readable in the notebook exactly as on disk.
        cells.append(code("%%writefile ml/" + name + ".py\n" + src))

    cells.append(md("""
## 2. Run the full pipeline

Roughly 3-6 minutes on a Kaggle CPU instance. Add `--fast` for a ~2 minute smoke run
with fewer resamples.

Everything lands in `/kaggle/working`: `results.json`, `RESULTS_SUMMARY.md`, the
figures, the tables, the fitted models, the per-student assignments and the ARFF
export for WEKA.
"""))
    cells.append(code("""
import importlib
import ml.run_pipeline as RP
importlib.reload(RP)

R = RP.main([])          # add "--fast" for a quick run, "--k", "3" to force k
"""))

    cells.append(md("## 3. Run summary"))
    cells.append(code("""
from IPython.display import Markdown, display
import ml.config as C

with open(os.path.join(C.OUT_DIR, "RESULTS_SUMMARY.md"), encoding="utf-8") as fh:
    display(Markdown(fh.read()))
"""))

    cells.append(md("## 4. Figures"))
    cells.append(code("""
from IPython.display import Image, display
import ml.config as C

for f in sorted(os.listdir(C.FIG_DIR)):
    if f.endswith(".png"):
        print("\\n" + "=" * 78 + "\\n" + f)
        display(Image(filename=os.path.join(C.FIG_DIR, f)))
"""))

    cells.append(md("## 5. Key tables"))
    cells.append(code("""
import pandas as pd
import ml.config as C

pd.set_option("display.width", 160, "display.max_columns", 40)
for t in ["t06_kmeans_sweep", "t14_cluster_summary", "t13_cluster_item_zscores",
          "t15_discriminating_items", "t10_theme_prevalence", "t09_algorithm_agreement"]:
    p = os.path.join(C.TAB_DIR, t + ".csv")
    if os.path.exists(p):
        print("\\n" + "=" * 78 + "\\n" + t)
        display(pd.read_csv(p, index_col=0))
"""))

    cells.append(md("""
## 6. Using the trained model

The pipeline persists a self-contained bundle. This is what makes the analysis a
*reusable instrument* rather than a one-off: next semester's responses can be
scored against this semester's profiles without re-fitting anything.
"""))
    cells.append(code("""
import joblib, numpy as np, pandas as pd
import ml.config as C

bundle = joblib.load(os.path.join(C.MODEL_DIR, "stress_profile_model.joblib"))
print("k = %d | silhouette = %.4f | trained on n = %d"
      % (bundle["k"], bundle["silhouette"], bundle["trained_on"]["n"]))
for i, nm in bundle["cluster_names"].items():
    print("  cluster %d -> %s" % (i, nm))


def assign_profile(responses):
    \"\"\"Score new responses.

    `responses` is a DataFrame with the twelve RAW item columns (1-5) named by the
    short keys in `bundle["items"]`. Reverse coding is applied here, so callers
    pass raw questionnaire answers exactly as exported.
    \"\"\"
    X = responses[bundle["items"]].astype(float).copy()
    for c in bundle["reversed_items"]:
        X[c] = 6 - X[c]
    Z = bundle["scaler"].transform(X.to_numpy())
    labels = bundle["kmeans"].predict(Z)
    return pd.DataFrame({
        "cluster": labels,
        "profile": [bundle["cluster_names"][int(l)] for l in labels],
    }, index=responses.index)


demo = pd.DataFrame([
    dict(zip(bundle["items"], [5, 5, 5, 5, 5, 5, 5, 1, 1, 5, 5, 5])),   # high strain, no support
    dict(zip(bundle["items"], [2, 2, 2, 2, 2, 2, 2, 5, 5, 2, 2, 2])),   # low strain, well supported
])
display(assign_profile(demo))
"""))

    cells.append(md("""
## 7. Optional - multilingual embedding cross-check (GPU + internet)

The methodology report *asserts* that transformer embeddings are unreliable on
answers this short and this code-mixed. On Kaggle that claim can be **tested**
rather than assumed.

To run it: **Settings -> Accelerator: GPU**, **Settings -> Internet: On**, then run
the two cells below. If either is unavailable the pipeline records a skip and
nothing downstream changes.
"""))
    cells.append(code("""
# !pip install -q sentence-transformers
"""))
    cells.append(code("""
import ml.embeddings as EMB
import ml.lexicon as LX
import ml.dataio as IO
import ml.config as C

if EMB.available():
    df, _ = IO.load_raw()
    names = IO.column_names(df)
    txt = df[names["open_current"]]
    T, mask, _ = LX.tag_frame(txt)
    out = EMB.run(txt, mask, T)
    for k, v in out.items():
        print("%-28s %s" % (k, v))
else:
    print("sentence-transformers not installed - uncomment the pip cell above "
          "(needs Internet: On).")
"""))

    cells.append(md("""
## 8. Output manifest
"""))
    cells.append(code("""
import ml.config as C

total = 0
for root, _, files in os.walk(C.OUT_DIR):
    if ".ipynb_checkpoints" in root or root.endswith("ml"):
        continue
    for f in sorted(files):
        p = os.path.join(root, f)
        sz = os.path.getsize(p)
        total += sz
        print("%9.1f KB  %s" % (sz / 1024, os.path.relpath(p, C.OUT_DIR)))
print("\\n%.1f MB total" % (total / 1024 / 1024))
"""))

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(nb, fh, indent=1, ensure_ascii=False)
    return OUT, len(cells)


if __name__ == "__main__":
    path, n = build()
    print("wrote %s  (%d cells, %.0f KB)" % (path, n, os.path.getsize(path) / 1024))
