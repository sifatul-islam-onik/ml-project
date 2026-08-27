# WEKA run record

**Project:** KUET CSE 4112 — Academic Stress Personas · **Run date:** 27 August 2026
**WEKA:** 3.8 Explorer · **Companion:** [`docs/WEKA_APPENDIX.md`](../docs/WEKA_APPENDIX.md)

Six clean runs reproducing the Python pipeline in an independent tool, plus one
deliberately-broken run kept as evidence. Every superseded run has been deleted; what remains
is what the report should cite.

**Headline:** WEKA corroborates the *profile shapes*, the *cluster count* and the *external
validation*. It does **not** reproduce the exact partition — and that disagreement independently
confirms the project's own §13 finding that the structure is weak and algorithm-dependent.

---

## The runs

| # | File | What it shows | Result |
|---|---|---|---|
| 01 | `01_kmeans_S9.txt` | Does WEKA's k-means recover the personas? | 76.2% agreement, **ARI 0.459** |
| 02 | `02_em_autok.txt` | How many clusters does EM pick on its own? | **k = 3**, by 10-fold CV log-likelihood |
| 03 | `03_ward_standardised.txt` | Does Ward's hierarchical agree? | 62.3% agreement, **ARI 0.215** |
| 04 | `04_j48_rules.txt` + `.png` | Can the personas be stated as rules? | **80.5%** CV, κ = 0.703, 17 leaves |
| 05 | `05_validate_backlog_S9.txt` | Do the personas track a held-out variable? | **V = 0.163**, p = 1.9e-06 |
| 06 | `06_validate_year_S9.txt` | Are they academic year in disguise? | **ARI 0.0099** — null, as intended |
| 90 | `90_backlog_hijack_demo.txt` | *Deliberately broken.* See §4. | 63-student artefact cluster |

---

## 1. Settings used

**Input files** (all written by the Python pipeline into `outputs/`):

| File | Contents | Used by |
|---|---|---|
| `stress_prepared.arff` | 6 clustering dimensions + `cluster` label | 01, 02, 03, 04 |
| `validate_backlog.arff` | 6 dimensions + `backlog` only | 05 |
| `validate_year.arff` | 6 dimensions + `year` only | 06 |

The `validate_*.arff` files exist because WEKA **auto-ignores the class attribute and resets any
manual ignore selection whenever the class changes** — which silently readmits `cluster` as a
feature. Giving each validation file exactly one nominal makes that impossible. See §4.

**Preprocessing:** `weka.filters.unsupervised.attribute.Standardize` on every run except 04
(J48 is scale-invariant, and unstandardised thresholds like `CGPA_ord <= 3` are readable).

**Clusterers:**

```
01/05/06  weka.clusterers.SimpleKMeans -N 3 -init 1 -I 500 -S 9
02        weka.clusterers.EM -I 100 -N -1 -X 10 -S 42
03        weka.clusterers.HierarchicalClusterer -N 3 -L WARD -P
04        weka.classifiers.trees.J48 -C 0.25 -M 20      (10-fold CV, class = cluster)
```

`-init 1` is k-means++, matching scikit-learn's default. WEKA's `-init 0` default is random and
is a common cause of a WEKA rerun disagreeing with the notebook.

---

## 2. Results in detail

### 01 — k-means vs the notebook

Sizes 330 / 318 / 339. **WCSS 256.6951**, 76.2% agreement, **ARI 0.459**.

The centroids line up dimension by dimension:

| | WEKA cl1 | notebook C0 | WEKA cl0 | notebook C1 | WEKA cl2 | notebook C2 |
|---|---|---|---|---|---|---|
| SupportGap | +0.52 | +0.47 | −0.40 | −0.38 | −0.10 | −0.10 |
| Financial | +0.78 | +0.67 | +0.43 | −0.12 | −1.15 | −0.78 |
| CGPA_ord | −0.75 | −0.70 | **+0.70** | **+0.68** | +0.02 | −0.02 |

Financial on cl0 (+0.43 vs −0.12) is the one real mismatch; everything else corroborates.

> **The seed matters more than anything else here.** `-S 42` converges to WCSS **263.209**, a
> worse local optimum in which `CGPA_ord` spans only 0.22 across clusters — the project's
> strongest discriminator does nothing, agreement drops to 64.2% (ARI 0.234), and the backlog
> association in run 05 disappears entirely (6.7 / 6.7 / 5.6%, ratio 1.2×).
>
> `SimpleKMeans` performs **one** start per seed; the notebook keeps the best of 25 (`n_init=25`).
> Sweeping seeds and keeping the lowest WCSS is not optional. This is a live demonstration of
> why the Python pipeline restarts 25 times.

### 02 — EM chooses k for itself

**`-N -1` lets EM select the cluster count by 10-fold cross-validated log-likelihood, and it
returned k = 3** — a separate tool, a separate criterion, landing on the crowned k. This is the
strongest single line of corroboration in the set.

The partition itself is weak (56.6%, ARI 0.109). The reason is visible in the output: Financial
has means 1.21 / 5.00 / 3.33 with SDs 0.40 / **0.053** / 1.07, so cluster 1 is essentially
"everyone who answered Financial = 5". EM latched onto one item's ceiling spike. This run is
unstandardised, which matters less for EM (it fits per-attribute variances) than for k-means,
but it should be stated rather than left unexplained.

### 03 — Ward hierarchical

Sizes 397 / 362 / 228. 62.3% agreement, **ARI 0.215**. Ward is deterministic — no seed, no
restarts — so this file is final.

The unstandardised version scored marginally higher (ARI 0.255), but standardisation is still
the correct preprocessing: without it, Euclidean distance weights each dimension by its variance,
so Financial (SD 1.41) and CGPA_ord dominate by construction. Report the standardised number.

`-P` prints the dendrogram in Newick form (the long block in the file). For a drawable version,
right-click the result in the *Result list* → **Visualize tree**.

### 04 — J48 rules

**80.5% under 10-fold CV, κ = 0.703, 17 leaves** — against the notebook's §20 tree at 76.8% ± 4.1%
with 16 leaves. Independent implementation, same neighbourhood.

The tree splits on **CGPA_ord → Financial → FutureMacro**, matching the notebook's permutation
importances exactly (0.275 / 0.190 / 0.170). `04_j48_tree.png` is the drawn tree — the single most
presentable artefact in this folder.

> **State what this measures.** The tree is trained on labels the clustering produced, so its
> accuracy measures *how cleanly the personas can be described*, not how well anything predicts
> stress. It is interpretability, not prediction. "77% accuracy at predicting student stress"
> would be wrong, and it is the claim a marker is most likely to probe.

### 05 — External validation against backlog

`backlog` never entered any feature space, so this is a genuine external check.

| Cluster | backlog rate | CGPA_ord (z) |
|---|---|---|
| cl0 | **1.8%** (6/330) | +0.702 |
| cl1 | **11.6%** (37/318) | −0.755 |
| cl2 | **5.9%** (20/339) | +0.025 |

χ² = 26.32, dof = 2, p = 1.9e-06, **Cramér's V = 0.163**, ARI 0.0043.

Against the notebook's Table 11 (12.2 / 5.6 / 1.3, V = 0.188, ARI 0.0055) — same association, same
effect size, from a file that never contained the labels. The rate ordering follows CGPA exactly:
the high-backlog cluster is the low-CGPA one, the low-backlog cluster is the high-CGPA one.

**V = 0.163 alongside ARI ≈ 0 is the point**: a real but small association. The personas *relate to*
backlog; they are not backlog.

### 06 — External validation against year

| | cl0 | cl1 | cl2 | sample |
|---|---|---|---|---|
| 1st Year | 40.6% | 22.6% | 34.2% | 32.6% |
| 2nd Year | 30.0% | 33.0% | 30.1% | 31.0% |
| 3rd Year | 10.3% | 18.9% | 15.6% | 14.9% |
| 4th Year | 19.1% | 25.2% | 19.8% | 21.3% |

**ARI = 0.0099**, against the notebook's 0.0117 — an independent tool reproducing the null to the
third decimal. This is what supports the claim that the personas are not academic year in disguise.

`gender`, `living` and `department` were not re-run at `-S 9`. All three are null in the notebook
(V = 0.08–0.11, ARI ≈ 0) and would only add more nulls; `validate_gender.arff`,
`validate_living.arff` and `validate_department.arff` are available if the set needs completing.

---

## 3. How to read WEKA's output here

| WEKA prints | Why it is misleading | Use instead |
|---|---|---|
| "Incorrectly clustered instances: 63.4%" | Matching 3 clusters to a binary class where one level is 6.4% of the sample cannot produce anything else | The per-cluster rate and Cramér's V |
| "Cluster 2 <-- No class" | 3 clusters, 2 class levels — one cluster necessarily gets no label | Nothing; it is an artefact of the display |
| Cluster numbering | Arbitrary in both tools; WEKA's "cluster 0" and the notebook's "C0" have no reason to match | ARI or the confusion matrix |
| A high agreement % | Can come from a leak (see §4) rather than from agreement | Check the `Ignored:` block first |

---

## 4. `90_backlog_hijack_demo.txt` — kept on purpose

This run accidentally left `backlog` in as a **clustering feature**. k-means immediately built a
cluster of exactly **63 students — precisely the 63 with `backlog = Yes`** — splitting 44 / 5 / 14
across the notebook's personas, matching `table09c` exactly.

`CLUSTERING_PLAN.md §2.5` predicted this before any code was written:

> *"I tested adding the binary backlog item as a clustering feature. At 6.4% prevalence it
> immediately forms its own 63-student cluster and hijacks the partition at every k."*

Reproducing that in a second tool, down to the exact cluster size, is better evidence for the
design decision than the plan's own prose. Keep it; label it clearly as the counter-example.

### The leak that produced it

Three earlier runs (now deleted) fed the `cluster` label back in as a clustering feature. The
tell is unmistakable: **Ward returned sizes 250 / 376 / 361, identical to the notebook's**, and EM
"discovered" 5 clusters that were just the 3 labels subdivided.

**Check before reporting any clustering run:**

1. The `Ignored:` block lists `cluster` (or the file has no `cluster` column at all)
2. `cluster` does **not** appear under `Attributes:`
3. `cluster` does **not** appear as a row in the centroid table
4. For k-means, WCSS is ~256.7, not ~263.2

---

## 5. Reproducing this set

```
01  stress_prepared.arff    → Standardize → SimpleKMeans -N 3 -init 1 -I 500 -S 9
                              Classes to clusters, class = cluster
02  stress_prepared.arff    → EM -I 100 -N -1 -X 10 -S 42
                              Classes to clusters, class = cluster
03  stress_prepared.arff    → Standardize → HierarchicalClusterer -N 3 -L WARD -P
                              Classes to clusters, class = cluster
04  stress_prepared.arff    → Classify → J48 -C 0.25 -M 20, class = cluster, 10-fold CV
05  validate_backlog.arff   → Standardize → SimpleKMeans -N 3 -init 1 -I 500 -S 9
                              Classes to clusters, class = backlog, ignore nothing
06  validate_year.arff      → Standardize → SimpleKMeans -N 3 -init 1 -I 500 -S 9
                              Classes to clusters, class = year, ignore nothing
```

Runs 01–03 and 05–06 take under three seconds each; 04 takes about a tenth of a second.

---

## 6. What WEKA cannot do

These have no WEKA equivalent and must be cited from the notebook, not recreated:

- the **profile differentiation index** (§14) — the criterion that rejected k = 2
- the **bootstrap-ARI stability screen** (§13)
- **Horn's parallel analysis** (§7)
- the **code-mixed free-text lexicon** (§19) — ~12% of answers are Bangla script
- the **generated persona names** (§17)

Those are the parts that turn a partition into personas.
