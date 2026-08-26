# Running this project on Kaggle

Everything needed to train and evaluate the model on Kaggle. Three manual steps,
then one notebook run.

---

## Step 1 — Upload the data as a Kaggle Dataset

The workbook is not in this folder on purpose: it holds student responses and
should be uploaded deliberately, not pushed automatically.

**Web route (easiest)**

1. Go to <https://www.kaggle.com/datasets> → **New Dataset**.
2. Upload `Academic Stress among Bangladeshi Engineering Students (Responses).xlsx`
   from the project root.
3. Title it **KUET Academic Stress Survey**.
4. Set visibility to **Private**.

**CLI route**

```bash
pip install kaggle                 # once
# put kaggle.json in %USERPROFILE%\.kaggle\  (Account → Create New API Token)

cd kaggle/dataset
# edit dataset-metadata.json: replace REPLACE_WITH_YOUR_KAGGLE_USERNAME
cp "../../Academic Stress among Bangladeshi Engineering Students (Responses).xlsx" .
kaggle datasets create -p . --dir-mode zip
```

> **Privacy.** These are anonymous responses — no names or roll numbers — but the
> free-text answers are students' own words about their mental health. Keep the
> dataset **private**. The project proposal commits to reporting aggregate results
> only.

---

## Step 2 — Create the notebook

**Web route**

1. <https://www.kaggle.com/code> → **New Notebook** → **File → Import Notebook**.
2. Upload `stress_clustering_kaggle.ipynb`.
3. **+ Add Input** → **Datasets** → attach the dataset from Step 1.
4. Settings → **Accelerator: None**, **Internet: Off**. The core pipeline needs
   neither. Turn both on only for the optional section 7.

**CLI route**

```bash
cd kaggle
# edit kernel-metadata.json: replace both REPLACE_WITH_YOUR_KAGGLE_USERNAME
kaggle kernels push -p .
```

---

## Step 3 — Run

**Run All.** About 3–6 minutes on a Kaggle CPU instance.

Everything lands in `/kaggle/working`:

| Output | What it is |
|---|---|
| `results.json` | every number the report cites, one file |
| `RESULTS_SUMMARY.md` | narrative summary generated from those numbers |
| `figures/` | 26 figures |
| `tables/` | 22 CSVs, ready to paste into the report |
| `models/stress_profile_model.joblib` | fitted scaler + PCA + k-means + profile names |
| `student_level_assignments.csv` | per-student profile, strain index, PC scores, theme flags |
| `stress_prepared.arff` | the identical prepared table, for WEKA |

Then **Save Version → Save & Run All (Commit)** so the outputs persist and become
downloadable from the notebook's Output tab.

---

## Does this need a GPU?

**No.** The pipeline is CPU-bound and finishes in minutes. Leave the accelerator
off — a GPU makes it no faster and burns quota.

The one exception is **section 7**, the optional multilingual-embedding
cross-check. It tests, rather than assumes, the methodology report's claim that
transformer embeddings are unreliable on answers this short and code-mixed. For
that section only:

1. Settings → **Internet: On** (needs a phone-verified Kaggle account).
2. Settings → **Accelerator: GPU T4 x2** (optional; CPU works, just slower).
3. Uncomment and run the `pip install -q sentence-transformers` cell.

If either is unavailable the pipeline records a `skipped` status and everything
else is unaffected.

---

## Useful run options

The pipeline is a normal CLI. Edit the `RP.main([])` call in section 2:

| Call | Effect |
|---|---|
| `RP.main([])` | default full run |
| `RP.main(["--fast"])` | ~2 min smoke run, fewer resamples |
| `RP.main(["--k", "3"])` | force k = 3 instead of the vote |
| `RP.main(["--secondary-k", "0"])` | skip the alternative-k comparison |
| `RP.main(["--with-embeddings"])` | include the embedding branch in the main run |
| `RP.main(["--data", "/kaggle/input/…/file.xlsx"])` | point at a specific file |

---

## Regenerating the notebook

The notebook is **generated** from `ml/`, so it can never drift from the code that
actually runs. After changing anything under `ml/`:

```bash
python kaggle/build_notebook.py
```

Each module becomes one `%%writefile` cell, so the notebook is self-contained: no
pip install, no clone, no utility-script attachment.

---

## Troubleshooting

**`FileNotFoundError: Could not find the response workbook`**
The dataset is not attached. Use **+ Add Input** in the notebook sidebar. The
loader searches every `/kaggle/input/*` mount for
`*Academic Stress among Bangladeshi Engineering Students*.xlsx`.

**`ValueError: Column mapping no longer matches the export`**
The Google Form was edited after this code was written. The loader addresses
columns positionally and fingerprints each against a keyword, so it stops rather
than silently analysing the wrong columns. Fix `SCHEMA_FINGERPRINT` and
`COL_IDX` in `ml/config.py` to match the new export.

**`ModuleNotFoundError: No module named 'ml'`**
Section 1's cells were not run, or were run out of order. Run All from the top.
