# Implementation Report — What Was Built and Why

**Project:** University Student Mental Stress Pattern Analysis Using Unsupervised Learning
**Course:** KUET, CSE 4112 Machine Learning Laboratory
**Data:** 987 anonymous survey responses, 21 columns, collected 30 Jul – 10 Aug 2026
**Date of this work:** 23 August 2026

---

## 1. What I found before writing any code

I read the proposal, both written reports, the existing `analysis/` scripts, and the
raw workbook. Four things needed attention.

### 1.1 Both existing analysis scripts were broken

`analysis/analysis_pipeline.py` and `analysis/characterise_v2.py` pointed at
workbook filenames that no longer exist:

| Script | Expected file | Actually present |
|---|---|---|
| `analysis_pipeline.py` | `… (Final).xlsx` | — |
| `characterise_v2.py` | `… (Finalv2).xlsx` | — |
| — | — | `… (Responses).xlsx` |

Neither script could run. `analysis/results.json` was stale, computed on an
n = 985 export while the current file has n = 987.

**Fixed:** both now resolve the path through one shared resolver that accepts all
three historical filenames plus a glob, so a fresh download from Google Forms
(which always lands as `… (Responses).xlsx`) does not break anything.

### 1.2 The methodology report specifies far more than was implemented

The methodology report (sections 5.2–5.7) commits to a substantial protocol. The
old pipeline delivered maybe a third of it:

| Committed in the report | Existed before | Now |
|---|---|---|
| PCA with explained-variance table | partial | ✅ full table + 95% retention |
| 2-D component projection | ✗ | ✅ |
| SSE elbow across k = 2…8 | ✗ | ✅ (numeric knee, not eyeballed) |
| EM with automatic k | ✗ | ✅ Gaussian mixture, BIC + CV log-likelihood |
| Ward dendrogram | ✗ | ✅ + average and complete linkage |
| Silhouette | ✅ | ✅ |
| Davies–Bouldin | ✗ | ✅ (+ Calinski–Harabasz) |
| Seed-stability check | ✗ | ✅ |
| Bootstrap stability | ✗ | ✅ 200 resamples, ARI |
| Classes-to-clusters vs held-out variables | ✗ | ✅ 5 variables |
| Gower / K-Prototypes mixed-data check | ✗ (noted as "if time permits") | ✅ |
| Frozen lexicon pattern file | ✗ | ✅ exported as a table |
| Two pre-specified null checks | ✗ | ✅ |
| Cleaned ARFF export | ✗ | ✅ |

### 1.3 The old supervised experiment had target leakage

I reproduced the old CGPA-prediction experiment inside the new pipeline and it
reported **93.6% accuracy against a 33.4% baseline** — a 60-point lift, which is not
plausible for predicting CGPA from stress items.

The cause: the encoded demographics block included `cgpa_ord`, an ordinal encoding
of the CGPA band, while the target *was* the CGPA band. Permutation importance
confirmed it: `cgpa_ord` scored 0.614, everything else below 0.031.

**Fixed.** `encode_background()` gained an `exclude` parameter, and the CGPA
experiment now drops `cgpa_ord`. The honest result is **40.4% vs 33.4% baseline —
a 7-point lift**. Reporting the 93.6% figure would have been a serious error in a
submitted report.

### 1.4 The dataset itself is clean

Worth stating because it changes what preprocessing is needed:

- 0 missing values in every closed-ended field (all were marked required)
- 0 exact duplicate rows, 0 duplicates ignoring the timestamp
- 0 out-of-range or non-integer Likert values
- Straight-lining affects 0.1% of respondents — the risk the proposal flagged did
  not materialise

So no imputation and no row deletion are warranted. The pipeline *checks* all of
this and records the numbers rather than assuming it.

---

## 2. What I built

A new `ml/` package — 14 modules, ~3,600 lines — replacing ad-hoc scripts with a
single reproducible pipeline, plus a generated Kaggle notebook.

```
ml/
  config.py       paths, column mapping, item metadata, palette — one source of truth
  dataio.py       load, schema validation, integrity audit, ARFF/CSV/JSON export
  preprocess.py   direction alignment, encoding, standardisation, descriptives
  structure.py    Cronbach's alpha, item-total, KMO, Bartlett
  reduce.py       PCA, Horn's parallel analysis, varimax rotation, t-SNE
  cluster.py      k-means, GMM, hierarchical, spectral, Gower k-medoids, gap statistic
  validate.py     bootstrap ARI, seed stability, consensus, classes-to-clusters
  lexicon.py      the frozen 13-theme code-mixed stressor lexicon
  textmodel.py    NMF/LDA cross-check, null tests, theme–strain association
  profile.py      z-score profiles, data-derived naming, eta-squared, recommendations
  supervised.py   profile recovery, incremental text value, permutation tests
  figures.py      all 26 figures, one consistent visual language
  embeddings.py   optional multilingual-embedding cross-check (GPU branch)
  run_pipeline.py orchestrator + generated narrative summary

kaggle/
  build_notebook.py              generates the notebook from ml/
  stress_clustering_kaggle.ipynb the notebook to upload (self-contained)
  kernel-metadata.json           for `kaggle kernels push`
  dataset/dataset-metadata.json  for `kaggle datasets create`
  requirements.txt
  README.md                      step-by-step Kaggle instructions
```

### 2.1 Design decisions, and why

**Positional column access with a keyword fingerprint.** The Google Forms export
uses the full bilingual question text as column headers, so columns must be
addressed by position. That silently breaks if the form is ever edited. Every
mapped index is now fingerprinted against a keyword, and a mismatch **aborts the
run**. A pipeline that analyses the wrong columns and produces plausible numbers
is worse than one that stops.

**Direction alignment before anything else.** Two items are positively worded
("comfortable asking teachers for help", "feedback reduces my stress"). They are
recoded `6 − x` so high always means more strain / less support. Without this,
those two items correlate *negatively* with the rest and both alpha and the
clustering are meaningless.

**Standardisation, even though all items share a 1–5 scale.** It looks
unnecessary — but the item SDs range from 0.97 to 1.46. Left unstandardised, the
two most polarised items (peer CGPA comparison, financial concern) would dominate
every Euclidean distance and effectively define the clusters on their own.

**Measurement structure computed before clustering, not after.** If the twelve
items do not hang together, a single "stress score" is not a valid target and the
clusters must be read as response-pattern profiles rather than stress *levels*.
Establishing that first is what keeps the interpretation honest. The verdict is
generated by an explicit threshold rule, so the prose cannot drift from the number.

**Parallel analysis alongside Kaiser.** Kaiser's eigenvalue > 1 rule over-retains
on short instruments. Horn's parallel analysis compares each observed eigenvalue
against the 95th percentile from random data of the same shape. Both are reported.

**The gap statistic — the most important addition.** Silhouette, Davies–Bouldin
and Calinski–Harabasz are all *undefined at k = 1*. They can only rank the
partitions you ask for; they can never tell you there are no clusters. The gap
statistic includes k = 1 as a candidate, so it can. This is the difference between
"k = 2 scored best" and "there is genuinely no group structure here" — and on this
data it matters (see §3).

**A model zoo, not one algorithm.** k-means, Gaussian mixture, three hierarchical
linkages, spectral, and Gower k-medoids over mixed data. If different algorithms
agree, the partition is a property of the *data*. If they disagree, it is a
property of the *algorithm* — which is itself the finding. Between-algorithm ARI
is reported as a matrix.

**Gower distance implemented from scratch.** The `gower` and `kmodes` packages are
not in the Kaggle base image. With n < 1000 the full matrix is cheap, so it is
computed directly in float32 with reused buffers — zero extra dependencies, and it
scales to a few thousand students without blowing up.

**k chosen by vote, with the disagreement published.** Seven independent signals
each cast one vote; ties break on silhouette (the criterion the proposal names as
primary), then on parsimony. The full vote record goes into `results.json` because
on weakly structured data the disagreement between criteria *is* a result.

**Degenerate linkages excluded from the vote.** Average linkage scored the highest
cophenetic correlation (0.694) — but it leaves >90% of students in one cluster at
every k, because it chains: it hangs a few outliers off one enormous cluster,
which reproduces the distance matrix faithfully and segments nothing. The pipeline
detects this and reports the reason, then uses complete linkage for the cut.
Without the check, "average linkage is best" would have been a misleading headline.

**The lexicon is frozen and versioned.** Editing the patterns after seeing the
results would invalidate every prevalence figure. It is exported as a table so the
instrument ships with the report, and it carries a version number.

**Text themes are never clustering features.** Clustering runs on the twelve items
alone. Theme prevalence is compared to the clusters *afterwards*, which makes the
agreement independent corroboration instead of circular reasoning.

**Two null checks specified in advance.** Whether non-response and answer length
carry any signal. Both are reported whatever they show, so a null result cannot be
quietly dropped. Both came back null.

**Every supervised score against an explicit baseline.** With effect sizes, cross-
validated standard deviations, and a permutation test on the headline model. At
n ≈ 1000, a p-value alone will call a trivial difference significant.

**Cluster names derived from the numbers.** `name_clusters()` ranks each cluster's
item z-scores and builds the label from the items that actually separate it, so a
reader can check the label against the profile table. Names are ordered by strain,
so P1 is always the lowest-strain profile regardless of the arbitrary integer
k-means assigns.

**The narrative summary is templated from `results.json`.** Every sentence in
`RESULTS_SUMMARY.md` is interpolated from a number the run produced. The write-up
cannot drift from the results.

**The notebook is generated, not hand-written.** `build_notebook.py` embeds each
module as a `%%writefile` cell. The notebook can never fall out of sync with the
code, and it is fully self-contained — no pip install, no clone, no utility-script
attachment.

---

## 3. What the pipeline found

Full run: n = 987, `random_state = 42`, 483 seconds.

### 3.1 The instrument is not one scale

- Cronbach's α = **0.609** (0.620 on the 9-item core) — below the 0.70 threshold
- KMO = 0.669; mean |inter-item r| = 0.123
- Weakest items: socio-political instability, peer CGPA comparison, feedback

**Consequence:** no composite stress score is used as a target or headline. The
twelve items are a checklist of partly independent stressors.

### 3.2 There is no natural group structure

This is the central finding, and it is supported four ways:

| Evidence | Value | Reading |
|---|---|---|
| Silhouette at k = 2 | **0.150** | below 0.25 → "no substantial structure" |
| Students with negative silhouette | 18.9% | closer to another cluster than their own |
| **Gap statistic** | selects **k = 1** | one homogeneous group beats every partition tested |
| PC1 + PC2 | 33.1% of variance | 11 of 12 components needed for 95% |
| Criterion agreement | 3 of 7 signals | elbow says 5, DB says 8, CV log-likelihood says 7 |
| Between-algorithm ARI | 0.27–0.39 | the partition depends on the algorithm |

The clusters *are* highly reproducible — bootstrap ARI 0.937 over 200 resamples,
seed-to-seed ARI 0.999. That is not a contradiction: k-means reliably draws the
**same cut through a continuum**. Stability says the boundary is reproducible; the
silhouette and gap statistic say the boundary is not a real gap.

**This is reported as the finding, not hidden.** The methodology report anticipated
it ("Weak cluster structure is likely… that will be reported as the finding rather
than presented as a discovery"). It is a defensible negative result, and the
pipeline states it in exactly those terms.

### 3.3 The k = 2 segmentation

| Profile | n | % | Mean strain | Modal year | % female |
|---|---|---|---|---|---|
| P2 — high sleep sacrifice, high result-driven demotivation | 603 | 61.1 | 4.09 | 2nd Year | 23.4 |
| P1 — lower strain, esp. low sleep sacrifice | 384 | 38.9 | 3.26 | 1st Year | 17.7 |

Most discriminating items (η²): sleep sacrifice 0.230, result-driven demotivation
0.226, coursework pile-up 0.211, exam worry 0.210, missed meals 0.180.

A k = 3 solution is also profiled, because the proposal anticipated roughly three
personas. It is reported as an *alternative*, with the cost stated plainly:
silhouette 0.080 versus 0.150 at k = 2. Use `--k 3` to make it the primary run.

### 3.4 The clusters do not correspond to anything demographic

All five held-out background variables reach nominal significance (p = 0.008–0.042)
but with Cramér's V of 0.057–0.101 and adjusted Rand of ≈ 0. At n = 987, χ² detects
trivial associations. **Effect sizes say the profiles are not year, gender, living
arrangement, CGPA or department in disguise** — which is exactly why reporting
effect sizes alongside p-values matters here.

Consistent with that: held-out features recover the cluster at 62.5% versus a
61.1% majority baseline — no usable predictive value.

### 3.5 The free text is where the real information is

- 86.7% answered; median 5 words; 28% two words or fewer
- 88.1% Latin script, 9.3% Bangla script, 2.6% mixed, plus 63 romanised-Bangla answers
- The frozen lexicon tags **85.7%** of answers, 1.40 themes each
- Top themes: lab & coursework load 28%, exams & results 24%, generic academic pressure 14%

The TF-IDF cross-check confirms the value of the lexicon by failing where the
lexicon succeeds: answers written entirely in Bangla script produce **zero
features** in the Latin-script vectoriser. An English-only pipeline would silently
drop roughly one answer in eight, all of which contain real content.

Both pre-specified null checks came back null:
- non-response vs strain: p = 0.329, d = 0.10
- answer length vs strain: ρ = 0.036, p = 0.288

### 3.6 Free-text features add no predictive value

Predicting CGPA band: items + demographics 40.1%, adding text 40.4%, baseline
33.4%. The text tells you *what* the strain is about; it does not predict grades.
Reported as a clean negative.

---

## 4. Things you need to do manually

Everything below needs your account or your judgement — I cannot do any of it.

### 4.1 Required — upload the data to Kaggle

The workbook is deliberately **not** in `kaggle/`. It holds student responses and
should be uploaded by you, on purpose.

1. <https://www.kaggle.com/datasets> → **New Dataset**
2. Upload `Academic Stress among Bangladeshi Engineering Students (Responses).xlsx`
3. Title: **KUET Academic Stress Survey**
4. **Set visibility to Private**

> The responses are anonymous — no names or roll numbers — but the free-text
> answers are students' own words about their mental health, and your proposal
> commits to reporting aggregate results only. Keep the dataset private.

### 4.2 Required — create and run the notebook

1. <https://www.kaggle.com/code> → **New Notebook** → **File → Import Notebook**
2. Upload `kaggle/stress_clustering_kaggle.ipynb`
3. **+ Add Input** → attach the dataset from step 4.1
4. Settings → **Accelerator: None**, **Internet: Off**
5. **Run All** (~3–6 minutes)
6. **Save Version → Save & Run All (Commit)** so outputs persist and are downloadable

If you prefer the CLI, replace `REPLACE_WITH_YOUR_KAGGLE_USERNAME` in both
`kaggle/kernel-metadata.json` and `kaggle/dataset/dataset-metadata.json` first.

### 4.3 A GPU is not needed

The pipeline is CPU-bound. Leave the accelerator off — a GPU makes it no faster
and burns quota.

The one exception is notebook **section 7**, the optional multilingual-embedding
cross-check. It exists because the methodology report *asserts* that transformer
embeddings are unreliable on text this short and code-mixed; on Kaggle that claim
can be tested rather than assumed, which turns an assumption into a measured
result. For that section only, set **Internet: On** (needs a phone-verified Kaggle
account) and optionally **Accelerator: GPU T4 x2**, then run the
`pip install sentence-transformers` cell. If unavailable, the pipeline records a
skip and nothing else changes.

### 4.4 Decisions only you can make

- **k = 2 or k = 3.** The data supports k = 2; the proposal anticipated three
  personas. Both are computed. If the report needs three, run `RP.main(["--k","3"])`
  — but keep the silhouette comparison visible, because it is the honest cost.
- **Author names.** The reports still carry `[Student name(s) · Roll · Department]`
  and `[Name 1 – Roll]` placeholders.
- **The two written reports predate this run.** `Academic_Stress_Report.docx`
  quotes n = 985 and a k = 3 solution from the old, partly-broken pipeline. Its
  numbers need updating from the new `outputs/results.json` — in particular the
  CGPA-prediction figures, which were affected by the leakage bug in §1.3.

### 4.5 Environment note (local only)

If you ever run the pipeline on a memory-constrained machine, cap the thread pools:

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m ml.run_pipeline
```

Kaggle has 16–30 GB of RAM, so this is not needed there.

---

## 5. What the pipeline produces

Written to `outputs/` locally, `/kaggle/working` on Kaggle.

| Output | Contents |
|---|---|
| `results.json` | 80 KB — every number the report cites |
| `RESULTS_SUMMARY.md` | narrative summary generated from those numbers |
| `figures/` | 26 figures |
| `tables/` | 22 CSVs, ready to paste into the report |
| `models/stress_profile_model.joblib` | scaler + PCA + k-means + profile names |
| `models/gaussian_mixture_k2.joblib` | the EM alternative |
| `student_level_assignments.csv` | per-student profile, strain index, PC scores, theme flags |
| `stress_prepared.arff` | the identical prepared table, for WEKA |

### 5.1 Figures

| # | Figure | Shows |
|---|---|---|
| 01–02 | respondent profile, department | sample composition |
| 03–04 | Likert composition, item means | raw responses and aligned strain |
| 05 | inter-item correlation | why α is low |
| 06–08 | scree + parallel analysis, cumulative variance, varimax loadings | dimensionality |
| 09–10 | PCA scatter, t-SNE | cluster overlap |
| 11, 11b | four k-selection criteria, **gap statistic** | how many clusters |
| 12 | Ward dendrogram | merge structure |
| 13–15 | per-student silhouette, stability, algorithm agreement | are the clusters real |
| 16–18 | z-score profiles, composition, η² ranking | what separates them |
| 19–22 | theme prevalence, by year, by living, by cluster | free-text findings |
| 23–24 | CGPA model comparison, profile recovery | supervised checks |
| 25 | alternative k = 3 profiles | the secondary solution |

### 5.2 Reusing the trained model

The persisted bundle is what makes this a reusable instrument rather than a
one-off — the department can score next semester's responses against this
semester's profiles without re-fitting. Section 6 of the notebook demonstrates it:

```python
bundle = joblib.load("models/stress_profile_model.joblib")
assign_profile(new_responses)   # takes RAW 1-5 answers; reverse coding is applied inside
```

---

## 6. Changes to existing files

| File | Change |
|---|---|
| `analysis/analysis_pipeline.py` | data path now uses the shared resolver (was pointing at a missing file) |
| `analysis/characterise_v2.py` | same |

Nothing else was modified. The `.docx`/`.pdf` reports, `form_autofill/` and
`google_form_csv_submitter/` are untouched. The old `analysis/` outputs are left in
place for comparison; the new pipeline writes to `outputs/` and does not overwrite them.

---

## 7. Honest limitations

Carried forward from the methodology report, plus what this run adds:

- **The headline result is a negative one.** There is no natural grouping. The
  k = 2 partition is a useful segmentation of a continuum, and is labelled as such
  everywhere. Presenting it as discovered structure would misrepresent the data.
- **Ceiling effects.** Several items sit near the top of the scale (job worry
  mean 4.35, sleep loss 4.28), which compresses usable variance.
- **Sample imbalance.** 79% male, 64% in the first two years, three departments
  supply 61% of responses, only 6.4% report a backlog.
- **The lexicon is single-coded.** Authored by one group with no second
  independent coder, so there is no inter-rater reliability figure and theme
  prevalences should be read as lower bounds. The ~14% of answers it leaves
  untagged are a known gap.
- **Self-report, cross-sectional, single-site.** Measures perceived stress at one
  point in time. No causal claim is available from this design.
- **The TF-IDF cross-check is Latin-script only.** A documented limitation, not an
  oversight — the lexicon is what covers the Bangla-script answers.
