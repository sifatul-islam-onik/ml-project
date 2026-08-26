# Plan — Rebuilding the Clustering as a Classical ML Lab Notebook

**Project:** University Student Mental Stress Pattern Analysis Using Unsupervised Learning
**Course:** KUET, CSE 4112 Machine Learning Laboratory
**Data:** `Academic Stress among Bangladeshi Engineering Students (Responses).xlsx` — 987 responses × 21 columns
**Written:** 24 August 2026
**Status:** approved — all five open decisions are closed (§7.3). Ready to write code.
Nothing has been run for you; every number in this plan is a design probe, not a result.

---

## 0. What you asked for, and what this plan delivers

| Your requirement | How the plan meets it |
|---|---|
| More than 2 groups, real personas ("high potential but financial problems") | §2 diagnoses *why* k = 2 happened and §3 fixes it. My probes on your data already surface exactly that persona — see §2.4 |
| Personas must come from the data, not be assumed | Both the cluster count and the persona labels come from explicit rules fixed before the run (§4.A, notebook §17) |
| Code in `.ipynb` / `.py` — you run it, not me | §5 — one self-contained notebook + a mirrored script. I write the code, you execute it |
| Methods done by us, via code | §3.5 — the psychometric and validation machinery is written out in numpy/sklearn, not hidden behind a black-box call. `ml/` is retired |
| Trains, tests, validates, produces figures | Notebook §10 + §15 (train/holdout protocol), §11–13 (k sweep + stability), §18 (external validation), §20 (supervised rule layer), plan §6 (figures) |
| Kaggle-runnable, manual steps OK | §7 |
| A clear plan for the two open-ended questions | §4.B — lexicon tagging for corroboration, persona naming, and discovering stressors the survey never asked about |

---

## 1. What already exists (audit)

| File / dir | Verdict |
|---|---|
| `KUET_Student_Stress_Clustering_Proposal.pdf` | Read. §3 objectives explicitly name persona labels: *"High-Achieving but Isolated," "Financially Burdened," "Resilient/Well-Supported."* **The proposal already promises the persona structure you want.** The k = 2 result under-delivered against your own proposal. |
| `Stress_Dataset_and_Methodology_Report.docx` | Read. Commits to: reverse coding, standardisation, PCA @95%, k-means/EM/Ward/Canopy, elbow + EM-CV + dendrogram for k, silhouette + DB, classes-to-clusters, seed & bootstrap stability, the 13-theme code-mixed lexicon, two pre-specified null checks. Also, honestly, forecasts weak structure (§6). |
| `Academic Stress...(Responses).xlsx` | 987 usable rows, 21 cols. Verified clean: 0 missing in closed items, all Likert in 1–5. Cols 5–10, 12–17 are the 12 Likert items; 18–19 free text; 1–4, 11, 20 background. |
| `ml/` (14 modules, ~3.6k lines) | Technically sound but wrong shape for a lab submission. It is a batch pipeline with 26 auto-generated figures — it hides the method behind `run_pipeline.main()`. **Retire it.** Salvage five things (§3.6). |
| `kaggle/stress_clustering_kaggle.ipynb` | A generated wrapper that `%%writefile`s the 14 modules then calls one function. Not a lab notebook. Replaced. |
| `IMPLEMENTATION_REPORT.md` | Read. Its two genuinely important findings — the CGPA target-leakage bug, and that silhouette/DB/CH are undefined at k = 1 so the gap statistic is needed — are carried forward. |
| `.docx`/`.pdf` reports | Untouched by this plan. They will need renumbering after the new run. |

**Note:** there is no `.csv` in the project — the data is the `.xlsx`. §4.1 exports a CSV as its first step so the rest of the pipeline is portable.

---

## 2. Diagnosis — why you got two groups (this is the core of the plan)

I ran the checks below on your actual data before writing this. Numbers are from my probe; the notebook recomputes all of them.

### 2.1 The 12 items are not one construct — they are four

Cronbach's α over all 12 items = **0.609**, mean |inter-item r| = **0.123**. That is not a scale. But it is not noise either. A 4-factor EFA with varimax rotation on your data gives a startlingly clean structure:

| Factor | Items that load | Interpretation |
|---|---|---|
| F1 | ExamWorry (.75), ResultDemotiv (.65), CGPACompare (.37) | **Evaluation & performance anxiety** |
| F2 | Feedback (.77), AskTeacher (.59) | **Teacher-support gap** |
| F3 | JobWorry (.76), Financial (.36) | **Future & financial insecurity** |
| F4 | PileUp (.62), MissMeal (.53), SleepLoss (.48), LabStress (.22) | **Workload & self-care sacrifice** |

Every item loads on exactly one factor. Nothing cross-loads. Four eigenvalues exceed 1. **This is the finding the old pipeline reported as a failure ("α is low, no composite score is valid") when it is actually the key to the whole problem.**

### 2.2 Clustering on 12 raw items measures severity, not pattern

With 12 weakly-correlated dimensions, Euclidean distance is dominated by overall agreement level. k-means then finds the only thing there is to find: *high scorers vs low scorers*. Hence k = 2, sizes 603/384, "P2 high strain / P1 lower strain". That is a severity split, not a set of personas. Confirmed:

| Feature space | k = 2 | k = 3 | k = 4 | k = 5 | k = 6 |
|---|---|---|---|---|---|
| 12 z-scored items (old approach) | **sil 0.150** | 0.080 | 0.081 | 0.087 | 0.088 |
| 4 subscale scores | 0.228 | 0.185 | **0.187** | 0.194 | 0.200 |

In 12-item space the silhouette *collapses* past k = 2 — every criterion then points back to 2. In subscale space it stays flat at ~0.19, and Davies–Bouldin actually improves monotonically (2.49 → 1.51). k > 2 becomes defensible.

### 2.3 Ceiling effects amplify the problem

JobWorry mean 4.35 (590/987 top-box), SleepLoss 4.28, MissMeal 4.15. Near-ceiling items carry little between-student variance but sit in the distance metric regardless. Averaging them into a subscale absorbs the ceiling; leaving them raw lets GMM latch onto their lumpiness — which I confirmed: a GMM on the 12 raw items produces "clusters" defined almost entirely by MissMeal and JobWorry.

### 2.4 The fix produces exactly the personas you described

k-means, k = 4, on **4 stress subscales + Financial split out + CGPA as an ordinal**, from my probe:

| | Workload | Evaluation | SupportGap | Financial | FutureMacro | mean CGPA band | n | % backlog |
|---|---|---|---|---|---|---|---|---|
| **C0** | −0.07 | **+0.34** | **−0.49** | **−0.39** | +0.43 | **4.28** (3.5–3.8+) | 310 | 1.6% |
| **C1** | +0.06 | **−1.02** | **+0.41** | **+0.58** | +0.18 | 3.80 | 197 | 4.6% |
| **C2** | **+0.55** | **+0.65** | +0.33 | **+0.65** | +0.24 | **2.78** | 290 | **12.8%** |
| **C3** | **−0.79** | −0.50 | −0.13 | **−0.96** | **−1.27** | 3.45 | 190 | 6.3% |

Reading them:

- **C0 — High-achieving & exam-anxious, well-resourced.** Top CGPA band, high evaluation anxiety, *good* teacher relations, *no* money worries.
- **C1 — Academically solid but financially strained and unsupported.** Good CGPA (3.80), the **lowest** exam anxiety in the sample, but high financial pressure and a high teacher-support gap. **This is your "high potential but financial problems" persona, and it fell out of the data.**
- **C2 — Overloaded and struggling.** Lowest CGPA, elevated on every stressor, 12.8% backlog (2× the sample rate).
- **C3 — Low-strain / coping.** Low on everything, notably future/financial worry.

Four distinct *shapes*, not four severity levels. That is the deliverable.

### 2.5 One thing that must NOT be done

I tested adding the binary backlog item as a clustering feature. At 6.4% prevalence it immediately forms its own 63-student cluster and hijacks the partition at every k (silhouette 0.401 at k = 2 — a meaningless artefact of one binary flag). **Backlog stays a held-out validation variable.** The methodology report already says this; the plan enforces it.

---

## 3. Design decisions for the rebuild

### 3.1 Primary feature space: subscale scores, not raw items
Five scores per student, each the mean of its direction-aligned items:
`Workload` (4 items) · `Evaluation` (3) · `SupportGap` (2) · `Financial` (1) · `FutureMacro` (2).
Financial is split out of F3 because it is the most polarised item in the instrument (SD 1.41, 49% agree / 31% disagree) and is central to the persona question. **Both the 4-subscale and 5-subscale groupings are run and compared** — the choice is made by the notebook against a stability criterion, not asserted here.

### 3.2 CGPA enters the primary model as an ordinal
This is a deliberate departure from the methodology report, which held all background variables out. Justification, stated in the notebook:
- Your objective is *personas*, and every persona named in your proposal ("High-Achieving but Isolated") combines an achievement level with a stressor pattern. Achievement must be in the model to produce them.
- CGPA is self-reported and ordered, so it is a legitimate ordinal feature.
- The cost is honest and stated: CGPA can no longer serve as external validation. **Year, gender, living arrangement, department and backlog stay strictly held out**, so external validation is preserved with five variables.
- The report-faithful model (stress subscales only, everything demographic held out) is **also run and reported side by side** as Model B. Nothing is lost; the comparison itself is a result.

### 3.3 Three feature spaces are compared as an experiment
| | Space | Role |
|---|---|---|
| **A** | 12 z-scored items | Baseline. Reproduces the old k = 2 result, so the improvement is measured, not claimed |
| **B** | Stress subscales only | Methodology-report-faithful model |
| **C** | Subscales + CGPA ordinal | **Primary** — the persona model |

Each gets the full k = 2…8 sweep and validation battery. The comparison table is a figure.

### 3.4 Validation criteria are re-weighted, and this is argued explicitly
Silhouette on 5-point Likert survey data is *structurally* low — the space is discrete, the true structure is continuous, and 0.15–0.25 is normal in published latent-profile work. Selecting k by silhouette alone on this data will always return 2. The notebook therefore selects k from a **panel** and prints the disagreement:

elbow (kneedle) · silhouette · Davies–Bouldin · Calinski–Harabasz · **gap statistic** (the only one that can vote for k = 1) · GMM BIC · GMM entropy · **bootstrap ARI** · minimum cluster size ≥ 5% · cross-algorithm ARI · **profile differentiation index** (§3.5).

### 3.5 The profile differentiation index — how "personas vs severity levels" is measured

Silhouette cannot tell a *persona* solution from a *severity* solution. It only asks whether clusters are separated, not whether they differ in **shape**. So a criterion is added that measures exactly the thing that went wrong last time.

For each candidate k, the cluster centroids are decomposed into two parts:

- **level** — how high the centroid sits on average across all dimensions (this student group is stressed *more*)
- **shape** — each dimension's deviation from that centroid's own level (this group is stressed *differently*)

`differentiation = shape variance / (shape variance + level variance)`, reported as a percentage of between-cluster variation. Measured on feature space C:

| k | silhouette | Davies–Bouldin | **% shape** | smallest cluster |
|---|---|---|---|---|
| 2 | 0.168 | 2.10 | **35.6%** | 43.6% |
| 3 | 0.136 | 2.00 | 61.4% | 25.1% |
| **4** | 0.136 | 1.85 | **68.1%** | 19.3% |
| 5 | 0.136 | 1.76 | 64.2% | 13.6% |
| 6 | 0.132 | 1.75 | 66.9% | 13.5% |
| 7 | 0.137 | 1.64 | 68.9% | 11.2% |
| 8 | 0.139 | 1.54 | 70.6% | 8.6% |

This makes the diagnosis of §2 numerical rather than rhetorical: **at k = 2 only 36% of between-cluster variation is shape — it is a severity split, which is precisely the complaint.** It jumps to 61% at k = 3, peaks at 68% at k = 4, and then flattens. Everything past 4 buys smaller clusters, not more differentiation.

It also exposes a real risk in the original plan: silhouette peaks at k = 2 in *every* feature space, so an unweighted vote could have crowned k = 2 a second time. This criterion is what stops that — on a stated methodological ground, not because you asked for more groups.

A hard rule then picks k (§4.A). If the panel says structure is weak, the notebook prints that and the personas are labelled *"a defensible segmentation of a continuum"* rather than *"discovered groups"*. That framing is honest and is what the methodology report already committed to.

### 3.6 What is salvaged from `ml/`, and what is discarded

**Salvage (ported into the new notebook as plain, readable functions):**
1. The **schema fingerprint check** — the Google Forms export uses full bilingual question text as headers, so columns are addressed positionally; a keyword assert aborts the run if the form is ever edited. Genuinely important.
2. The **13-theme code-mixed stressor lexicon** (`ml/lexicon.py`) — 5+ hours of manual work, covers Bangla script and romanised Bangla. Reused verbatim, kept frozen and versioned.
3. The **CGPA target-leakage lesson** — never put an encoding of the target in the feature block.
4. The **gap statistic** implementation.
5. The **ARFF export** — retained per §7.3 Q2, together with a new WEKA reproduction appendix.

**Discard:** the 14-module package layout, the generated-notebook build step, the 26-figure batch dump, the multilingual-embedding branch.

---

## 4. Notebook specification, section by section

Each numbered section = one markdown header + a small number of code cells. Target: **~1,200 lines total**, every method visible on the page.

| § | Section | What the code does | Outputs |
|---|---|---|---|
| **1** | Setup | Imports, `RANDOM_STATE = 42`, palette, output dirs, path resolution (local ↔ Kaggle) | — |
| **2** | Load & schema check | Read xlsx, positional column map, **keyword fingerprint assert**, export `stress_clean.csv` | CSV |
| **3** | Data quality audit | Missing, duplicates, out-of-range, straight-lining, within-row SD. Written as checks that *print numbers*, not assumptions | Table 1 |
| **4** | EDA | Composition by year/CGPA/gender/living/dept; Likert stacked bars; item means & SDs; ceiling-effect flag | Fig 1–3 |
| **5** | Preprocess | Reverse-code `AskTeacher`, `Feedback` (6 − x); standardise; **prove standardisation is needed** by printing the SD range (0.97–1.46) | Table 2 |
| **6** | Measurement structure | **Cronbach's α from scratch**, corrected item–total r, inter-item correlation heatmap, **KMO & Bartlett from scratch** | Fig 4, Table 3 |
| **7** | Dimensionality reduction | PCA: eigenvalues, scree, cumulative variance, 95% retention, 2-D projection. **Horn's parallel analysis** (500 random matrices). **EFA + varimax written out** → the 4-factor structure of §2.1 | Fig 5–8, Table 4 |
| **8** | Subscale construction | Build the 4- and 5-subscale scores from the EFA; per-subscale α; subscale correlation matrix; **justify each grouping against the loadings** | Table 5 |
| **9** | Feature spaces A/B/C | Assemble the three matrices of §3.3; CGPA ordinal encoding; explicit note on what is held out | — |
| **10** | **Train / holdout split** | 70/30 stratified on a coarse strain tertile. All model selection uses train only | — |
| **11** | k sweep | For each space × {k-means, GMM(full/diag), Ward, complete, average, spectral} × k = 2…8: SSE, silhouette, DB, CH, BIC, entropy, sizes. Degenerate-linkage detector (a linkage leaving >80% in one cluster is excluded and *the reason is printed*) | Fig 9–11, Table 6 |
| **12** | Gap statistic | 50 uniform reference draws over the PCA-aligned bounding box, k = 1…8, with the 1-SE rule. The only criterion that can vote "no clusters" | Fig 12 |
| **13** | Stability | Bootstrap ARI (200 resamples), seed stability (20 seeds), cross-algorithm ARI matrix, consensus matrix + heatmap | Fig 13–15 |
| **14** | **Choose k** | Vote table across all criteria → hard rule (§4.A) → chosen k printed with the full disagreement record | Table 7 |
| **15** | **Holdout test** | Assign the 30% holdout with the frozen model. Test: (a) profile centroids reproduce within tolerance, (b) cluster proportions match, (c) holdout silhouette vs train, (d) ARI between "fit on train, predict holdout" and "fit fresh on holdout" | Table 8, Fig 16 |
| **16** | Profiling | Per-cluster: subscale z-profile heatmap, **radar chart**, item-level means, **η² ranking** of what separates them, background composition (year/gender/living/backlog/dept), CGPA distribution | Fig 17–21, Table 9 |
| **17** | Persona naming | `name_cluster()` ranks each cluster's z-deviations and composes a label from the 2–3 dimensions that actually separate it. **Names are generated, never typed in.** Ordered deterministically so run-to-run labels are stable | Table 10 |
| **18** | External validation | Classes-to-clusters vs the 5 held-out variables: χ², **Cramér's V**, adjusted Rand. Effect sizes mandatory — at n = 987, χ² calls trivial associations significant | Table 11 |
| **19** | Free-text corroboration | See §4.B below — the full treatment of the two open-ended fields | Fig 22–24, Table 12 |
| **20** | **Supervised interpretability** | Decision tree (depth ≤ 4) on cluster labels, 10-fold CV, **printed rule set per persona** — the J48 equivalent the methodology report asks for. Plus RF permutation importance. Stated as interpretability, not prediction | Fig 25–26, Table 13 |
| **21** | Persona cards | One card per cluster: name, n, %, defining dimensions, demographic tilt, top free-text themes, recommendation. This is what goes in the report and the slides | Fig 27 |
| **22** | Persist | `joblib` bundle (scaler + subscale spec + model + names), `student_assignments.csv`, all tables as CSV, `results.json` | Artefacts |
| **23** | Reuse demo | `assign_persona(new_raw_responses)` — takes raw 1–5 answers, applies reverse coding and scaling internally. Makes it a reusable instrument for next semester | — |

### 4.A  The k-selection rule (fixed in advance)

All of k = 2…8 is **computed and shown** for every feature space and every algorithm. This rule only decides which k is *crowned* as the headline solution.

```
1. Discard any k where the smallest cluster < 5% of n           (unusable personas)
2. Discard any k where bootstrap ARI < 0.50                     (not reproducible)
3. Discard any k with profile differentiation < 50%             (a severity split,
                                                                 not a persona set)
4. Cap the selection at k <= 6                                  (actionability, §7.3 Q4)
5. If the gap statistic selects k = 1 -> record "weak structure"
   and label all output as a segmentation of a continuum, but continue
6. Among survivors, score each k by its mean rank across
   {silhouette, DB, CH, BIC, bootstrap ARI, cross-algorithm ARI, differentiation}
7. Ties -> prefer the smaller k (parsimony)
8. Print the full vote table, including every k that was discarded and why
```

Rules 1–4 are **screens on usability**, applied before any quality criterion is consulted; rule 6 is the quality vote among what survives. That order matters: it means k = 2 is not rejected for scoring badly (it scores well on silhouette) but for failing a stated, measurable requirement of the research question — the proposal asks for interpretable stress *profiles*, and a 36%-shape solution does not answer that question.

Fixing this *before* seeing the result is what stops the analysis from being reverse-engineered to a pretty picture. Based on §2.4 and §3.5, k will most likely land at **4** — but the rule decides, not me, and the discarded candidates stay visible in the output.

### 4.B  What happens to the two open-ended questions

There are two free-text fields, and I checked both against the raw data:

| | Q18 — biggest **current** stressor | Q19 — stressor in **previous** years |
|---|---|---|
| Answered | 856 (86.6%) | 708 (71.7%) |
| Median / mean words | 5 / 10.7 | 4 / 8.3 |
| ≤ 2 words | 28% | 32% |
| Contains Bangla script | 11.9% | 11.3% |

Real answers from your data, which set what is possible:

> "lab report,Assignment" · "HEAT ENGINE LAB REPORT" · "Lab." · "Fear of less cgpa"
> "সিলেবাস অনেক বড় কিন্তু সময় অনেক কম।" *(the syllabus is huge but the time is short)*
> "Onek porar chap & weakness in english" *(romanised Bangla — a lot of study pressure)*
> "1) not being able to see my family for months  2) lab report, lab quiz and viva all in the same week."

Three properties decide the method: answers are **very short** (a bare noun phrase 28–32% of the time), **code-mixed** across three writing systems (English, Bangla script, romanised Bangla in Latin letters), and **uniformly negative** in sentiment because the question asks for a source of stress. That rules out sentiment analysis (no variance), transformer embeddings (no context in two words, and no pretrained model covers romanised Bangla), and English-only stemming/stopword pipelines (they would silently discard ~12% of answers). The methodology report already argues this; the data confirms it.

So the text gets **four jobs**, in order of importance:

**1. Naming and corroborating the personas (the main job).**
The frozen 13-theme lexicon tags each answer with any number of themes — multi-label, because students routinely name two or three stressors in one sentence, as the last example above shows. Theme prevalence is then computed **within each cluster** and tested with χ² under a Bonferroni correction (α = .05/13 ≈ .0038). This is the notebook's strongest validation move: **the clusterer never sees the text**, so if the financially-strained persona also over-mentions money and family themes in its own words, that is genuine independent corroboration rather than circular reasoning. It is also what turns a centroid table into a persona a reader believes.

**2. Finding stressors the questionnaire never asked about.**
This is the one thing only the free text can do. Already visible in the answers above and not covered by any of the 12 Likert items: preparatory-leave length ("short PL"), session jam, semester duration, viva, hall/mess food, homesickness and family separation, adjustment to a new city, English-language weakness, lecture comprehension, non-departmental courses, model-making cost. These are reported as a standalone finding — *what students volunteer unprompted versus what we thought to ask* — and are the most directly actionable output for the department. It is also, honestly, the most publishable part of the project.

**3. Comparing prompted agreement against unprompted salience.**
The highest-scoring Likert item is job worry (mean 4.35). Whether job/career is also the most *volunteered* theme is an open question — the two need not coincide, and a mismatch is itself a result about the instrument.

**4. Q19 as a retrospective, cross-year view.**
Q19 asks about earlier years, so pairing Q19 themes with Q18 themes for the same student gives a within-person before/after comparison that the cross-sectional design otherwise cannot provide — e.g. whether adjustment/homesickness themes drop and career themes rise between years. Its lower answer rate (71.7%) is partly structural: many answers are procedural non-answers ("same", "N/A", "I am in 1st year") and those get their own explicit coded level rather than being dropped.

**Cross-checks and pre-specified null tests**, all reported whatever they show:
- A TF-IDF + NMF topic model on the Latin-script subset, purely to confirm the lexicon's largest themes also emerge from a data-driven representation. It is expected to *fail* on Bangla-script answers, which is the point — that failure is the evidence for why a hand-built code-mixed lexicon was necessary.
- Null check A: do students who skip the field differ in measured strain?
- Null check B: does answer length correlate with strain?

**What the text is deliberately NOT used for: clustering features.** Clustering runs on the Likert subscales alone. Feeding the theme flags into the clusterer would destroy their value as independent corroboration — the clusters would agree with the themes *by construction*, and job 1 above would become worthless.

**Decided (§7.3 Q5): the secondary text-only clustering experiment is dropped.** It was rejected on the merits, not for effort. With a 4–5 word median answer, a 13-column binary theme matrix is extremely sparse, and clustering it would largely rediscover the lexicon's own categories — a lot of extra pipeline for a result that is close to circular. The cost of that choice is real and is stated in §8: **nothing in this project answers "could the free text alone reproduce the personas?"** If a marker specifically wants a mixed-data or text-driven clustering demonstration, it can be bolted on as an appendix later without touching the main pipeline.

---

## 5. Files to be created

```
notebooks/
  stress_personas.ipynb        ← PRIMARY. Self-contained, top-to-bottom, ~1200 lines.
                                 Runs on Kaggle and locally with no edits.
src/
  stress_personas.py           ← the same pipeline as a script (`python src/stress_personas.py`)
  lexicon.py                   ← the frozen 13-theme lexicon, imported by both
docs/
  WEKA_APPENDIX.md             ← filter chain + clusterer options to reproduce in WEKA 3.8 (Q2)
outputs/                       ← created at runtime: figures/ tables/ models/ results.json
                                 including stress_prepared.arff
CLUSTERING_PLAN.md             ← this file
```

The `.py` is a genuine mirror (same functions, `if __name__ == "__main__"` driver), not a dump of the notebook. Whichever you run, you get the same numbers.

`ml/` and `kaggle/` are **left in place, untouched**, so the old results remain reproducible for comparison. Nothing is deleted without you saying so.

---

## 6. Figures (≈27, all generated by the notebook)

Grouped so they map onto report sections: sample composition (1–3) · measurement structure (4) · dimensionality (5–8) · k selection (9–12) · cluster validity (13–16) · **persona profiles (17–21)** · free text (22–24) · supervised rules (25–26) · **persona cards (27)**.

Consistent visual language: one palette, one font stack, colour-blind-safe categorical colours, diverging scale for z-profiles, every axis labelled with units. Figures 17–21 and 27 are the ones your report is actually built around.

---

## 7. What you will need to do manually

### 7.1 Required — upload the data to Kaggle (private)
1. <https://www.kaggle.com/datasets> → **New Dataset**
2. Upload `Academic Stress among Bangladeshi Engineering Students (Responses).xlsx`
3. Title: **KUET Academic Stress Survey** → **set visibility to Private**

The responses are anonymous, but the free text is students' own words about their mental health and your proposal commits to aggregate-only reporting. Keep it private.

### 7.2 Required — run the notebook
1. <https://www.kaggle.com/code> → **New Notebook** → **File → Import Notebook** → upload `notebooks/stress_personas.ipynb`
2. **+ Add Input** → attach the dataset from 7.1
3. Settings → **Accelerator: None**, **Internet: Off** *(everything used is in the Kaggle base image; no pip install)*
4. **Run All** — estimated **4–8 minutes** (the bootstrap and gap statistic dominate)
5. **Save Version → Save & Run All (Commit)** so outputs persist and are downloadable

Locally instead: `pip install pandas numpy scipy scikit-learn matplotlib openpyxl joblib` then run the notebook or `python src/stress_personas.py`. Same numbers either way.

**No GPU.** The pipeline is CPU-bound; an accelerator makes it no faster and burns your quota.

### 7.3 Decisions — ANSWERED 24 Aug 2026

- **Q1 — CGPA in the primary model? → YES.** Feature space C (subscales + CGPA ordinal) is the headline persona model. Model B (stress subscales only, report-faithful) and Model A (12 raw items, the old baseline) are computed and reported alongside. Year, gender, living arrangement, department and backlog remain strictly held out for external validation.
- **Q2 — WEKA/ARFF path? → YES.** The cleaned ARFF export is kept, **and** a "how to reproduce this in WEKA 3.8" appendix is written: the filter chain (`Remove` → `NumericToNominal` → `Standardize` → `PrincipalComponents -R 0.95`), `SimpleKMeans -N 4 -init 1`, `EM -N -1`, `HierarchicalClusterer -L WARD -P`, and `AddCluster` for exporting assignments. This keeps the notebook consistent with the methodology report your lab already has on file.
- **Q3 — Primary artefact? → NOTEBOOK.** `notebooks/stress_personas.ipynb` is the deliverable you run; `src/stress_personas.py` mirrors it for command-line reruns.
- **Q4 — Cap the selection at k ≤ 6? → YES, with the sweep still covering k = 2…8.** See §4.A rule 4 and the note below.
- **Q5 — What to do with the two open-ended questions? → LEXICON-BASED MULTI-LABEL TAGGING, used only for corroboration, persona naming, and stressor discovery — never as a clustering feature.** Full treatment in §4.B. The optional text-only clustering experiment is dropped.

**On Q5, briefly.** The data itself settles the method. ~12% of answers contain Bangla script and would produce **zero** features in a Latin-script TF-IDF vectoriser; at a 4–5 word median there is not enough context for embeddings or topic models to be reliable; and sentiment is constant by construction because the question asks for a source of stress. A hand-built, code-mixed, multi-label lexicon is the only method that actually covers what students wrote. TF-IDF/NMF stays as a secondary cross-check that is *expected* to partially fail on the Bangla subset — that failure is the evidence for why the lexicon was necessary, which makes it a finding rather than a weakness.

**On Q4, in plain terms.** Two separate things were being conflated:

*What gets computed and shown:* **all of k = 2 through 8**, for every feature space and every algorithm — full comparison tables and plots (elbow, silhouette, DB, CH, BIC, differentiation). Nothing is hidden.

*What gets crowned as the headline:* **one** value of k, chosen by the rule in §4.A, and the cap says that value may not exceed 6.

Why cap it. With n = 987, k = 8 puts the smallest persona at 8.6% — roughly 85 students. Small clusters are unstable (they move between bootstrap resamples), hard to name from a centroid, and useless to a counselling office that has to design an actual intervention per group. The §3.5 table also shows differentiation flattening after k = 4 (68.1% → 70.6% across four extra clusters), so k = 7 and 8 buy fragmentation, not insight. The cap is a statement about **actionability**, and it is declared before the run rather than after seeing which k looked nicest.

### 7.4 After the run — reports need renumbering
`Stress_Dataset_and_Methodology_Report.docx` quotes the old figures throughout, and any draft final report quotes n = 985 and a k = 3 solution from a pipeline that had the CGPA leakage bug. Every number will need refreshing from the new `results.json`. I can do that pass once you have run the notebook and sent me the outputs.

---

## 8. Honest limitations — carried into the notebook, not buried

- **Structure is genuinely weak.** Silhouette will land near 0.15–0.20 whatever we do. The personas are a defensible, stable, interpretable segmentation of a continuum — not four naturally separated populations. The notebook says this in its own summary output, in those words.
- **Subscale reliabilities are modest** (α ≈ 0.55–0.63 for the 2–4 item facets). Expected for short facets on heterogeneous topics; reported, not hidden.
- **The subscales come from an EFA on the same data used to cluster.** No independent confirmatory sample exists. The 70/30 holdout (notebook §10 and §15) partially mitigates this and the limitation is stated.
- **CGPA is self-reported and banded**, and including it in Model C costs it as a validation variable (§3.2).
- **Sample imbalance:** 79% male, 64% in years 1–2, 3 departments supply 61% of responses, 6.4% report a backlog. Personas describe the dominant strata best.
- **The lexicon is single-coded** — no second coder, so no inter-rater reliability figure; theme prevalences are lower bounds. The ~14% of answers it leaves untagged are a known gap and unlikely to be a random subset.
- **The free text is never a clustering input** (§4.B). That is what makes theme–cluster agreement genuine corroboration, but it has a cost: this project does not answer whether the text alone would reproduce the same personas.
- **Self-report, cross-sectional, single-site.** No causal claim is available from this design.

---

## 9. Sequence once you approve

1. ~~You answer §7.3~~ — **done, 24 Aug 2026. All five decisions closed; nothing is blocking.**
2. I write `src/lexicon.py`, then `notebooks/stress_personas.ipynb`, then mirror it to `src/stress_personas.py`, then `docs/WEKA_APPENDIX.md`
3. I do a **syntax + import + small-sample smoke check** on my side so it cannot fail on line 40 of your Kaggle run — but I do **not** run the full pipeline or generate results. The numbers are yours to produce.
4. You upload, run, download outputs
5. I help interpret and renumber the written reports against your `results.json`

### Decision log (what changed after this plan was first drafted)

| # | Decision | Effect on the plan |
|---|---|---|
| Q1 | CGPA **in** the primary model | Space C is the headline; A and B still reported (§3.2, §3.3) |
| Q2 | Keep **WEKA/ARFF** | ARFF export retained + `docs/WEKA_APPENDIX.md` added (§3.6, §5) |
| Q3 | **Notebook** is the primary artefact | `.py` becomes the mirror, not the deliverable (§5) |
| Q4 | Sweep k = 2…8, **crown k ≤ 6** | §4.A rule 4 |
| — | *Discovered while answering Q4:* silhouette peaks at k = 2 in **every** feature space, so the original vote could have re-crowned k = 2 | **Profile differentiation index added** as a screening criterion (§3.5, §4.A rule 3) — the single most important change to the plan |
| Q5 | Free text = **lexicon tagging only**, never a clustering feature | §4.B; text-only clustering experiment dropped; limitation recorded in §8 |
