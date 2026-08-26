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

For the *genuine* external validation — against year, gender, living arrangement, department
and backlog — those variables are deliberately **not in the ARFF**, because they were held out
of every feature space. Notebook §18 does that analysis with Cramér's V effect sizes, which
WEKA's classes-to-clusters output does not report. Use the notebook's Table 11 for it; at
n = 987 a raw agreement percentage is easy to over-read.

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
