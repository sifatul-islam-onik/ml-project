# Reproducing the clustering in WEKA 3.8

**Companion to** `notebooks/stress_personas.ipynb` · **Project:** KUET CSE 4112 Machine Learning Laboratory

The methodology report on file specifies a WEKA workflow, so this appendix keeps that route
open. The notebook exports `outputs/stress_prepared.arff` from **exactly the same
preprocessing** the Python pipeline uses — the alternative would be preparing the data twice
in two tools and hoping the two agree, which is how quiet discrepancies get into a report.

Everything below is stated both as a **GUI click-path** and as an equivalent **command line**,
because the Explorer is easier to demonstrate and the CLI is easier to paste into a report
appendix.

---

## 0. What is in the ARFF

`outputs/stress_prepared.arff` is written by notebook §22 and contains one row per student:

| Attribute | Type | Meaning |
|---|---|---|
| `Evaluation` | NUMERIC | Evaluation & performance anxiety, mean of its items, 1–5 |
| `Workload` | NUMERIC | Workload & self-care sacrifice, 1–5 |
| `FutureMacro` | NUMERIC | Future & macro insecurity, 1–5 |
| `SupportGap` | NUMERIC | Teacher-support gap, 1–5 |
| `Financial` | NUMERIC | Financial pressure, 1–5 |
| `CGPA_ord` | NUMERIC | CGPA band as an ordinal, 1 = *Below 2.50* … 5 = *3.80–4.00* |
| `cluster` | NOMINAL | The persona assigned by the notebook — **the label, not an input** |

A **second file, `outputs/stress_validation.arff`**, carries the same rows plus the five
held-out background variables (`year`, `gender`, `living`, `department`, `backlog`). It exists
only to make §4.1 possible; never cluster on those columns.

Three things to note before you start.

1. **The subscale columns depend on which feature space the notebook crowned.** §11 chooses
   between the 4- and 5-subscale groupings on a stability criterion, so the attribute list
   above reflects the current run. Check `primary_space` in `outputs/results.json` — the
   attributes in your ARFF are always whatever that run actually used.
2. **The values are unstandardised**, on the original 1–5 metric. That is intentional: it
   keeps the file readable in the Explorer's *Preprocess* tab, and standardisation is applied
   as an explicit filter step below so it is visible in the workflow rather than baked in.
3. **`cluster` must never be an input to clustering.** It is the notebook's answer. Feeding it
   back in would produce a perfect, meaningless result. Step 2 removes it.

---

## 1. Load and inspect

**GUI** — `Explorer` → *Preprocess* → **Open file…** → `outputs/stress_prepared.arff`.

You should see 987 instances and 7 attributes. Click each attribute to confirm the ranges are
1–5 and that `cluster` has as many levels as `k_decision.k_chosen` in `results.json`.

---

## 2. The filter chain

Applied in this order, in the *Preprocess* tab (**Choose** → filter → set options → **Apply**).

### 2.1 `Remove` — drop the label

```
weka.filters.unsupervised.attribute.Remove -R last
```

**GUI:** `filters/unsupervised/attribute/Remove`, set `attributeIndices` = `last`.

Removes `cluster`. Do this first so no later filter is fitted using information from the
label. (In the *Cluster* tab you can instead use **Ignore attributes** and select `cluster`,
which is equivalent and lets you keep the label for the classes-to-clusters evaluation in
step 4 — see the note there.)

### 2.2 `NumericToNominal` — only if you want CGPA treated as categorical

```
weka.filters.unsupervised.attribute.NumericToNominal -R 6
```

**Optional, and off by default.** The notebook treats `CGPA_ord` as an *ordinal* and leaves it
numeric, because the bands are ordered and the distance between adjacent bands is meaningful.
Converting it to nominal discards that ordering and, after `Standardize`, WEKA would binarise
it into five indicator columns that jointly outweigh any single stress facet. Use this only if
you are deliberately demonstrating the difference — and if you do, say so in the write-up,
because it changes the model.

### 2.3 `Standardize` — put every dimension on the same footing

```
weka.filters.unsupervised.attribute.Standardize
```

**GUI:** `filters/unsupervised/attribute/Standardize`.

Not optional. Un-standardised Euclidean distance weights each attribute by its variance, so
the most dispersed facet would dominate every centroid. Notebook §5 prints the SD range and
the resulting weight ratio as the evidence for this step.

> Use `Standardize` (zero mean, unit variance), **not** `Normalize` (min–max to [0,1]).
> `Normalize` is the WEKA filter people reach for by name; it is the wrong one here, because
> it makes the weight of an attribute depend on its observed range and so lets a single
> extreme respondent set the scale.

### 2.4 `PrincipalComponents` — optional, for the report-faithful variant

```
weka.filters.unsupervised.attribute.PrincipalComponents -R 0.95 -A 5
```

**GUI:** `filters/unsupervised/attribute/PrincipalComponents`, set `varianceCovered` = `0.95`.

The methodology report specifies PCA at 95% retention. On the subscale space this retains
almost every component, because the facets are close to uncorrelated by construction — which
is itself worth reporting. **The notebook's headline model does not apply PCA** (it clusters
the facet scores directly, so the centroids stay interpretable in named facets rather than in
rotated components). Run this only for the report-faithful comparison, and expect the personas
to become harder to name, not easier.

---

## 3. Clusterers

*Cluster* tab → **Choose** → clusterer → set options → **Cluster mode: Use training set** →
**Start**.

Replace `-N 3` with whatever `k_decision.k_chosen` says in your run.

### 3.1 k-means — the headline model

```
weka.clusterers.SimpleKMeans -N 3 -init 1 -A "weka.core.EuclideanDistance -R first-last" -I 500 -S 42
```

| Option | Why |
|---|---|
| `-N 3` | the crowned k, from `results.json` |
| `-init 1` | **k-means++ initialisation** — matches scikit-learn's default. WEKA's default (`-init 0`) is random initialisation and is markedly less stable; leaving it at the default is the most common reason a WEKA rerun disagrees with the notebook |
| `-I 500` | max iterations |
| `-S 42` | the same random seed the notebook uses |

### 3.2 EM — the mixture model, with k chosen by cross-validation

```
weka.clusterers.EM -I 100 -N -1 -X 10 -M 1.0E-6 -S 42
```

`-N -1` tells EM to **select the number of clusters itself** by 10-fold cross-validated
log-likelihood. This is the WEKA analogue of the notebook's BIC criterion, and it is worth
running precisely because it may disagree — the notebook prints its own criterion
disagreements for the same reason. To force a specific k instead, use `-N 3`.

### 3.3 Ward hierarchical

```
weka.clusterers.HierarchicalClusterer -N 3 -L WARD -P -A "weka.core.EuclideanDistance -R first-last"
```

`-P` prints the dendrogram in Newick form; in the GUI, right-click the result in the
*Result list* → **Visualize tree** to see it drawn.

> **Do not bother with `-L SINGLE` or `-L AVERAGE` on this data.** The notebook's degeneracy
> screen (§11) rejects average linkage at every k because it leaves 87–99% of students in one
> cluster — the classic chaining artefact. WEKA will produce the same thing without comment,
> so you would be reporting a partition that has found nothing.

---

## 4. Classes-to-clusters evaluation

To reproduce the notebook's agreement figures, keep the `cluster` attribute in the file (skip
step 2.1) and instead:

*Cluster* tab → **Ignore attributes** → select `cluster` → **Cluster mode: Classes to clusters
evaluation** → pick `cluster` as the class → **Start**.

WEKA prints the confusion matrix and the percentage of incorrectly clustered instances. That
number answers "does WEKA's k-means recover the notebook's partition?" and is the right thing
to quote when comparing the two tools.

### 4.1 Genuine external validation — use the second ARFF

`stress_prepared.arff` holds only the six clustering dimensions, so the *held-out* variables
are not in it. **`outputs/stress_validation.arff` is the same 987 students with `year`,
`gender`, `living`, `department` and `backlog` added back as nominal attributes**, purely so
this step is demonstrable in WEKA. Those five never entered any feature space, which is what
makes them valid external checks.

> ⚠️ **Do not use this file for the validation run.** Everything not ignored and not the
> class is a clustering input — and WEKA **auto-ignores the class attribute and resets your
> manual ignore selection whenever you change the class**. That silently puts `cluster` back in
> as a feature, and the run then reproduces the notebook's partition perfectly and meaninglessly.

Use the **per-variable files** instead. Each holds the six clustering dimensions plus exactly
one held-out nominal:

| File | Class attribute | Ignore |
|---|---|---|
| `outputs/validate_backlog.arff` (strongest, V = 0.188) | `backlog` | — nothing |
| `outputs/validate_gender.arff` | `gender` | — nothing |
| `outputs/validate_year.arff` | `year` | — nothing |
| `outputs/validate_living.arff` | `living` | — nothing |
| `outputs/validate_department.arff` | `department` | — nothing |

The single nominal *is* the class, WEKA excludes it automatically, and there is nothing left to
ignore — so the reset cannot introduce a leak. Apply `Standardize`, then
`SimpleKMeans -N 3 -init 1 -S 42`, **Classes to clusters evaluation**, and Start.

**Read the output the right way.** WEKA reports "incorrectly clustered instances" as a raw
percentage, which on this data will look alarmingly high and mean very little — the personas
genuinely are *not* backlog or gender in disguise, and that is a finding, not a failure.
Notebook §18 quantifies it properly with Cramér's V (all ≤ 0.19) and adjusted Rand (all ≈ 0).
Quote Table 11 for the numbers and use WEKA's confusion matrix as the visual.

---

## 5. Exporting the assignments

```
weka.filters.unsupervised.attribute.AddCluster -W "weka.clusterers.SimpleKMeans -N 3 -init 1 -S 42" -I last
```

**GUI:** *Preprocess* → `filters/unsupervised/attribute/AddCluster`, set the wrapped clusterer,
then **Apply** and **Save…** as CSV or ARFF.

`-I last` ignores the trailing `cluster` attribute while fitting. The filter appends a new
`cluster` column holding WEKA's own assignment, which you can then compare row by row against
the notebook's `outputs/student_assignments.csv`.

---

## 6. Full command-line pipeline

Assuming `weka.jar` is on the classpath:

```bash
# 2.1 + 2.3  drop the label, then standardise
java -cp weka.jar weka.filters.unsupervised.attribute.Remove \
     -R last -i outputs/stress_prepared.arff -o /tmp/nolabel.arff

java -cp weka.jar weka.filters.unsupervised.attribute.Standardize \
     -i /tmp/nolabel.arff -o /tmp/std.arff

# 3.1  k-means
java -cp weka.jar weka.clusterers.SimpleKMeans \
     -N 3 -init 1 -I 500 -S 42 -t /tmp/std.arff

# 3.2  EM, choosing k by cross-validation
java -cp weka.jar weka.clusterers.EM -I 100 -N -1 -X 10 -S 42 -t /tmp/std.arff

# 3.3  Ward
java -cp weka.jar weka.clusterers.HierarchicalClusterer \
     -N 3 -L WARD -P -t /tmp/std.arff

# 5  export assignments
java -cp weka.jar weka.filters.unsupervised.attribute.AddCluster \
     -W "weka.clusterers.SimpleKMeans -N 3 -init 1 -S 42" -I last \
     -i outputs/stress_prepared.arff -o outputs/weka_assignments.arff
```

---

## 7. Why WEKA will not match the notebook exactly, and what to do about it

Expect **close but not identical** results. The honest framing for the report is that WEKA
corroborates the partition; it does not re-derive it. Sources of difference, roughly in order
of how much they matter:

| Source | Effect | Mitigation |
|---|---|---|
| **Initialisation** | WEKA defaults to random init, scikit-learn to k-means++ with 25 restarts | `-init 1`, and accept that restart counts still differ |
| **Restarts** | `SimpleKMeans` runs once per seed; the notebook keeps the best of 25 | run several seeds and keep the lowest within-cluster SSE |
| **Empty-cluster handling** | the two implementations resolve them differently | rare here; check the reported cluster sizes |
| **Standardisation denominator** | population vs sample SD | a fractional shift in z-values, no practical effect |
| **Cluster numbering** | arbitrary in both tools | compare with ARI or the classes-to-clusters matrix, **never** by cluster number |

That last row is the one that trips people up: WEKA's "cluster 0" and the notebook's "C0" have
no reason to be the same group. Always compare partitions with an agreement measure.

Finally — the pieces of this project that WEKA **cannot** reproduce, and which therefore have
to be cited from the notebook rather than recreated: the profile differentiation index (§14),
the bootstrap-ARI stability screen (§13), Horn's parallel analysis (§7), the code-mixed
free-text lexicon (§19), and the generated persona names (§17). Those are the parts that turn
a partition into personas, and they only exist on the Python side.

---

## 8. J48 — turning the personas into readable rules

The methodology report on file names **J48** explicitly, and notebook §20 is described as "the
J48 equivalent". This is the one step where WEKA is arguably the better demo: it draws the tree
for you.

Load `stress_prepared.arff` (the label stays in), then:

*Classify* tab → **Choose** → `trees/J48` → set **(Nom) cluster** as the class →
**Cross-validation, Folds = 10** → **Start**.

```
weka.classifiers.trees.J48 -C 0.25 -M 20
```

`-M 20` sets the minimum instances per leaf. Raise it until the tree fits on a slide — at the
default `-M 2` you get a sprawling tree that classifies well and explains nothing, which
defeats the purpose. Notebook §20 caps depth at 4 and gets **76.8% ± 4.1%** under 10-fold CV
with 16 leaves; expect J48 to land in the same neighbourhood.

Right-click the result in the *Result list* → **Visualize tree**. That drawn tree, with one
readable path per persona, is the single most convincing artefact you can put on a slide.

> **Say what this is.** The tree is trained on labels the clustering produced, so its accuracy
> measures *how cleanly the personas can be described*, not how well anything predicts stress.
> It is interpretability, not prediction. Claiming 77% "accuracy at predicting student stress"
> would be wrong, and it is the mistake a marker is most likely to probe.

---

## 9. A 12-minute live demonstration runbook

> **These runs are done.** The completed set, with every number and the exact settings, is in
> [`weka_runs/README.md`](../weka_runs/README.md). Six clean runs: k-means (ARI 0.459), EM
> independently selecting **k = 3**, Ward (ARI 0.215), J48 (80.5% CV, κ 0.703), backlog
> validation (**V = 0.163** vs the notebook's 0.188) and year validation (ARI 0.0099, null).
> Use this section to rehearse; quote the README for the figures.

Ordered so each step motivates the next. Times are for a lab demo with narration.

**Before you start:** WEKA is not installed on this machine — get 3.8.6 or newer from
<https://waikato.github.io/weka-wiki/downloading_weka/>. Newer releases are the ones that
behave on current JDKs; if the GUI misbehaves on the installed Java 25, use the bundled JRE
that WEKA's Windows installer offers.

| # | Time | Do this | Say this |
|---|---|---|---|
| **1** | 1 min | *Preprocess* → open `stress_prepared.arff`. Click each attribute. | "987 students, six dimensions. Five are stress facets recovered by factor analysis, one is the CGPA band as an ordinal. `cluster` is the notebook's answer — it is a label, not an input." |
| **2** | 1 min | Apply `Standardize`. Show the before/after SD in the attribute panel. | "Item SDs ran 0.98 to 1.46. Un-standardised, the most dispersed facet would decide every centroid on its own." |
| **3** | 2 min | *Cluster* → `SimpleKMeans -N 3 -init 1 -I 500 -S 42` → **Use training set** → Start. | "Same k, same seed, k-means++ init to match scikit-learn. Compare the centroid table to Table 09a — same shapes." |
| **4** | 2 min | Re-run with **Classes to clusters evaluation**, class = `cluster`, ignoring `cluster` as an input. | "This asks whether WEKA independently recovers the notebook's partition. Compare by the confusion matrix, never by cluster number — numbering is arbitrary in both tools." |
| **5** | 2 min | `EM -N -1 -X 10 -S 42`. | "EM picks k for itself by cross-validated log-likelihood. If it disagrees, that is the point — the notebook's own panel of eight criteria voted for 1, 2, 4 and 8. The disagreement is the result." |
| **6** | 1 min | `HierarchicalClusterer -N 3 -L WARD -P` → right-click → **Visualize tree**. | "Ward's dendrogram. Note there is no clean gap to cut at — which is exactly why we report a segmentation of a continuum rather than discovered groups." |
| **7** | 2 min | Open `stress_validation.arff`. Classes-to-clusters against `backlog` (§4.1 ignore list). | "Backlog was never a model input. The personas track it — 12.2% vs 1.3% — but Cramér's V is only 0.19, so they are not backlog in disguise." |
| **8** | 1 min | *Classify* → `J48 -M 20`, class = `cluster`, 10-fold → **Visualize tree**. | "And here is each persona as a readable rule. This describes the clusters; it does not predict stress." |

**Have these open in another window** for the moments WEKA cannot cover: `fig13_gap_statistic.png`
(the gap statistic votes k = 1), `fig18_persona_profiles.png` (the z-profile heatmap) and
`fig27_unasked_stressors.png` (the twelve stressors the survey never asked about).

### 9.1 The questions you will be asked

| Question | Answer |
|---|---|
| "Why is the silhouette only 0.139?" | Structurally low on 5-point Likert data; published latent-profile work sits at 0.15–0.25. We report it rather than hiding it, and we label the output a segmentation of a continuum. |
| "Why not k = 2? It scores better." | It does — on silhouette. It fails a pre-registered screen: only 40.7% of its between-cluster variation is *shape*, so it is a severity split, not a persona set. The rule was fixed before the run. |
| "Isn't CGPA doing all the work?" | Partly, by design — η² = 0.355, the highest of any dimension. The proposal's personas combine achievement with a stressor pattern, so achievement had to be in the model. The cost is that CGPA can no longer serve as external validation, which is why five other variables were held out. |
| "Why doesn't WEKA give identical clusters?" | Initialisation, restart count and empty-cluster handling differ (§7). WEKA corroborates the partition; it does not re-derive it. |
| "Where is the text analysis?" | Not in WEKA — §12% of answers are Bangla script and would vectorise to nothing. Notebook §19, and `fig27`. |

### 9.2 Check every clustering run before you quote it

Three runs in the first pass silently fed the `cluster` label back in as a clustering feature and
produced a perfect, meaningless partition — Ward returned sizes identical to the notebook's. A
fourth ran a good configuration on a poor local optimum and destroyed the backlog association.

Four checks, ten seconds:

1. `Ignored:` lists `cluster` — or the file has no `cluster` column at all
2. `cluster` does **not** appear under `Attributes:` or as a centroid-table row
3. `Test mode:` says **Classes to clusters evaluation**, not "evaluate on training data"
4. For k-means on this data, `Within cluster sum of squared errors` is ~**256.7**, not ~263.2

Check 4 is the one people skip. `-S 42` converges to WCSS 263.2, where `CGPA_ord` spans 0.22
instead of 1.46 and the backlog signal flattens from 6.4× to 1.2×. `SimpleKMeans` runs one start
per seed; sweep seeds and keep the lowest WCSS.

### 9.3 What to say if something breaks

Every WEKA step above is a *corroboration* of a result that already exists in `outputs/`. If the
GUI fails on the day, the demo still stands on `results.json` and the 30 figures — open
`PROJECT_ANALYSIS.md` §5 and walk the numbers instead.
