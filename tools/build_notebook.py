# -*- coding: utf-8 -*-
"""Regenerate notebooks/stress_personas.ipynb from src/stress_personas.py.

    python tools/build_notebook.py .

Run this after editing src/stress_personas.py. The notebook is a DELIVERABLE, not
a build artefact - it is committed, readable, and self-contained - but it carries
the module's function bodies verbatim, and this is what keeps the two from
drifting. Editing the notebook by hand is fine for prose; edit the module for
code, then re-run this.

The notebook must be self-contained (Kaggle has no src/ on the path), so the
function bodies are copied verbatim out of the module rather than retyped. That
is what makes "whichever you run, you get the same numbers" a structural fact
instead of a promise.
"""
import io, json, os, re, sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__)))
PROJ = sys.argv[1]
MOD = os.path.join(PROJ, "src", "stress_personas.py")
LEX = os.path.join(PROJ, "src", "lexicon.py")
OUT = os.path.join(PROJ, "notebooks", "stress_personas.ipynb")

src = io.open(MOD, encoding="utf-8").read()
lex = io.open(LEX, encoding="utf-8").read()

# ---------------------------------------------------------------- module split
# Everything before "# === SECTION 1" is the module docstring + __future__.
marks = [(m.start(), int(m.group(1)), m.group(2).strip())
         for m in re.finditer(r"^# === SECTION (\d+): ([^=]+?)\s*=+\s*$", src, re.M)]
main_at = src.index("# === MAIN ===")
sections = {}
for i, (pos, num, title) in enumerate(marks):
    end = marks[i + 1][0] if i + 1 < len(marks) else main_at
    body = src[pos:end].rstrip() + "\n"
    sections[num] = {"title": title, "code": body}

# ------------------------------------------------------------------ main split
main_src = src[main_at:]
body_start = main_src.index("def main(argv=None):")
body = main_src[body_start:]
body = body[body.index("\n") + 1:]
body = body[:body.index("\nif __name__ ==")]
lines = body.split("\n")
# strip the docstring-free 4-space indent
ded = [(l[4:] if l.startswith("    ") else l) for l in lines]
ded = [l for l in ded if l.strip() != "return R"]
body = "\n".join(ded)

chunks, keys = {}, []
for m in re.finditer(r'^head\("((?:SECTION|SUMMARY)[^"]*)"', body, re.M):
    keys.append((m.start(), m.group(1)))
pre = body[:keys[0][0]].rstrip()
for i, (pos, key) in enumerate(keys):
    end = keys[i + 1][0] if i + 1 < len(keys) else len(body)
    chunks[key.split("  ")[0].strip()] = body[pos:end].rstrip()


def drv(k):
    return chunks[k]


# ------------------------------------------------------------------- notebook
cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").split("\n")})


def code(*parts):
    text = "\n\n".join(p.rstrip() for p in parts if p and p.strip())
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {},
                  "outputs": [], "source": text.split("\n")})


md(r"""
# University Student Mental Stress Pattern Analysis Using Unsupervised Learning

**KUET — CSE 4112 Machine Learning Laboratory**
Data: *Academic Stress among Bangladeshi Engineering Students* — 987 responses × 21 columns

---

## What this notebook does, and why it is built this way

An earlier run of this project clustered the twelve Likert stress items directly and got
**two groups**: high scorers and low scorers. That is a *severity split*, not a set of
personas, and it under-delivered against the project proposal, which promises profiles
such as *"High-Achieving but Isolated"* and *"Financially Burdened."*

This notebook diagnoses why that happened and fixes it:

| | |
|---|---|
| **The diagnosis** | The twelve items are not one construct (Cronbach's α ≈ 0.61). With twelve weakly-correlated dimensions, Euclidean distance is dominated by overall agreement level, so k-means finds the only thing there is to find — high vs low. |
| **The fix** | An exploratory factor analysis recovers four clean facets. Clustering the *facet scores* plus a CGPA ordinal recovers stress **shapes** instead of stress **levels**. |
| **The new criterion** | Silhouette cannot tell a persona solution from a severity solution — it only asks whether clusters are separated, not whether they differ in shape. A **profile differentiation index** is added that measures exactly that, and it is a screening criterion, not a tiebreaker. |
| **The guard against wishful thinking** | The k-selection rule is written down in §14 **before** the result is seen, every k from 2 to 8 is computed and shown, and every discarded candidate is printed with the reason it was discarded. |

**The free text is never a clustering input.** It is held back precisely so that
agreement between a persona's centroid and what those students volunteer in their own
words is *independent corroboration* rather than a result that is true by construction.

---

## How to run this

**On Kaggle** — `+ Add Input` → attach the survey dataset → Accelerator **None**, Internet
**Off** → **Run All**. Everything used is in the Kaggle base image; there is no `pip install`.
Expect roughly 4–8 minutes; the bootstrap and the gap statistic dominate.

**Locally** — `pip install pandas numpy scipy scikit-learn matplotlib openpyxl joblib`,
then Run All, or `python src/stress_personas.py` for the identical pipeline as a script.

Outputs land in `outputs/`: `figures/`, `tables/`, `models/`, `results.json`,
`student_assignments.csv`, and `stress_prepared.arff` for the WEKA route
(see `docs/WEKA_APPENDIX.md`).
""")

md("""
## 1. Setup

Imports, the fixed random state, the shared plot style, and path resolution that works
unchanged on Kaggle and locally. `RANDOM_STATE = 42` is threaded through every stochastic
step in the notebook — k-means initialisation, the bootstrap, the gap statistic's reference
draws, the train/holdout split — so a rerun reproduces the run exactly.

The three constants that encode the selection rule (`MIN_CLUSTER_FRAC`,
`MIN_BOOTSTRAP_ARI`, `MIN_DIFFERENTIATION`) are set **here**, at the top, rather than next
to the code that applies them in §14. That is deliberate: they are pre-registered
thresholds, and putting them beside the decision they govern would make them look
adjustable after the fact.
""")
# main() takes argv; a notebook has none. Declaring it here keeps every other
# line of the driver byte-identical to the module.
code(sections[1]["code"],
     "# main(argv) takes a command-line data path; a notebook has none.\nargv = None\n\n" + pre)

md("""
## 2. Load and schema check

The Google Forms export uses the full bilingual question text as each column header, so
columns are addressed **positionally**. That is fragile the moment the form is edited, so
every mapped position is fingerprinted against a keyword and the run **aborts** on a
mismatch rather than producing a plausible-looking but wrong analysis.

This is the single most important defensive check in the notebook: a silently shifted
column would not raise an error anywhere downstream — it would just quietly cluster the
wrong variables.
""")
code(sections[2]["code"], drv("SECTION 1-2"))

md("""
## 3. Data quality audit

Written as checks that **print numbers**, not assumptions that print nothing. "Zero missing
values" is a finding and has to appear in the output where a reader can see it.

Two of these go beyond completeness. **Straight-lining** counts respondents who ticked the
same box twelve times — they contribute a centroid-pulling point carrying no information.
**Within-row SD** measures how much a respondent actually discriminated between items. Both
matter more than usual here, because the clustering runs on between-student differences and
a respondent with no within-row variance has none to offer.
""")
code(sections[3]["code"], drv("SECTION 3"))

md("""
## 4. Exploratory data analysis

Sample composition, the raw response distribution, and per-item means with a **ceiling-effect
flag**.

The ceiling flag is not decoration. Job worry (mean ≈ 4.35), sleep loss and missed meals sit
near the top of the scale, which means they carry very little *between-student* variance
while still occupying a full dimension in the distance metric. That is one of the mechanisms
behind the original two-group collapse, and §8 fixes it by averaging such items into a facet
rather than leaving them raw.

Figure 3 is drawn on the **raw** scale, before reverse coding, because it documents the
instrument as students actually answered it.
""")
code(sections[4]["code"], drv("SECTION 4"))

md("""
## 5. Preprocessing

Two positively worded items (`AskTeacher`, `Feedback`) are recoded `6 − x`, so that after
alignment **high always means more strain / less support** across all twelve items. Without
this a centroid could not be read without checking each item's polarity one at a time.

Standardisation is then *demonstrated to be necessary* rather than asserted: the table prints
the item SD range and the resulting weight ratio. Un-standardised Euclidean distance weights
each item by its variance, so the most polarised item would count for several times the least
polarised one in every distance computation.
""")
code(sections[5]["code"], drv("SECTION 5"))

md("""
## 6. Measurement structure

**This runs before any clustering, on purpose.** If the twelve items do not hang together,
then a single stress score is not a valid target, and the whole framing has to change from
"levels of one latent trait" to "profiles across several partly independent facets."

Cronbach's α, the corrected item–total correlations, Bartlett's test of sphericity and the
KMO statistic are all written out in numpy here rather than imported from a package — the
lab brief asks for the method to be visible, and each is a few lines.

The verdict at the end is produced by an explicit threshold rule, so the conclusion cannot
drift away from the numbers it rests on.
""")
code(sections[6]["code"], drv("SECTION 6"))

md("""
## 7. Dimensionality reduction and factor structure

Three things happen here, in increasing order of importance to the final result.

**PCA** gives the eigenvalues, the scree curve and the number of components needed for 95%
of variance. **Horn's parallel analysis** (500 simulated random matrices) then supplies a
much stricter retention bar than Kaiser's "eigenvalue > 1", which is well known to
over-retain.

**The EFA is the finding.** Principal-axis factoring with varimax rotation — both written
out below — recovers a strikingly clean four-facet structure in which essentially every item
marks exactly one factor. The earlier pipeline reported the low α as a *failure* ("no
composite score is valid"). It is not a failure; it is the key to the whole problem. Those
four facets are the feature space that makes personas possible.

PAF is used rather than PCA for the factor step because the question is about *common*
variance — which items share a latent facet. PCA puts total variance on the diagonal and so
inflates loadings on items that mostly carry noise.
""")
code(sections[7]["code"], drv("SECTION 7"))

md("""
## 8. Subscale construction

Each facet becomes a mean score over the items the EFA assigned to it.

Note carefully what is discovered and what is pre-registered. The **membership** of each
facet is discovered — it is the argmax loading straight out of §7. The **name** comes from a
fixed, declared reading of each item, applied to whichever item marks the factor. Naming a
factor is always an interpretive act; doing it from the marker item keeps that interpretation
deterministic and inspectable instead of hidden.

Two groupings are built, not one. `Financial` loads on the future/macro factor, but it is the
most polarised item in the instrument and is central to the research question, so splitting
it into a facet of its own is a live option. Both the 4-subscale and 5-subscale versions are
carried forward, and §11 decides between them **on a stability criterion** rather than by
assertion.
""")
code(sections[8]["code"], drv("SECTION 8"))

md("""
## 9–10. Feature spaces and the train / holdout split

Three feature spaces are compared as an experiment:

| | Space | Role |
|---|---|---|
| **A** | 12 z-scored items | Baseline. Reproduces the old k = 2 result, so the improvement is **measured**, not claimed |
| **B** | Stress subscales only | Methodology-report-faithful model |
| **C** | Subscales + CGPA ordinal | **Primary** — the persona model |

Every persona named in the project proposal combines an achievement level with a stressor
pattern ("High-Achieving but Isolated"), so achievement has to be *in* the model to produce
them. CGPA is self-reported and ordered, which makes it a legitimate ordinal feature. The
cost is stated plainly and paid in §18: **CGPA can no longer serve as external validation.**

Year, gender, living arrangement, department and backlog stay **strictly held out of every
space**, so five genuine validation variables survive.

The split is 70/30, stratified on a coarse overall-strain tertile so that both halves carry
the same severity mix — otherwise a holdout difference could be waved away as one half simply
being more stressed. All model selection happens on train only.
""")
code(sections[9]["code"], sections[10]["code"], drv("SECTION 9-10"))

md("""
## 11. The k sweep

Every feature space × seven algorithms × k = 2…8. Six algorithm families are run rather than
one, because agreement between methods that make *different* assumptions is evidence the
partition is in the data rather than in k-means' preference for round, equally-sized blobs.

A **degeneracy screen** excludes any run that leaves more than 80% of students in a single
cluster, and prints why. Average linkage does this almost everywhere — it is the classic
chaining artefact — and a run that has not found structure must not be allowed to vote.

Two things to watch in the output:

1. **Space A reproduces the old result.** Silhouette peaks at k = 2 and collapses after. The
   improvement claimed by this notebook is therefore measured against a baseline that is
   actually recomputed here, not against a remembered number.
2. **Silhouette peaks at k = 2 in *every* feature space** (Figure 11, top-left). This is the
   single most important observation in the notebook, and the reason §14 cannot select k by
   silhouette: an unweighted vote would simply crown k = 2 a second time.

The primary feature space is chosen at the end of this section, on a stability criterion,
with each candidate judged at the k it would actually ship.
""")
code(sections[11]["code"], drv("SECTION 11"))

md("""
## 12. The gap statistic

The only criterion in this notebook that can answer **"is there any structure at all?"**,
because it alone admits k = 1 as a candidate. Silhouette, Davies–Bouldin and
Calinski–Harabasz are all mathematically undefined at k = 1, so they can never report the
*absence* of clusters — they only ever rank the partitions they are handed.

Reference data is drawn uniformly over the bounding box of the principal components, which is
the standard "one homogeneous blob" null, and selection follows Tibshirani's 1-SE rule.

If this returns k = 1 that is not a bug and it does not stop the analysis. It is recorded,
and every downstream persona is then labelled *"a defensible segmentation of a continuum"*
rather than *"discovered groups"* — which is the honest description and what the methodology
report already committed to.
""")
code(sections[12]["code"], drv("SECTION 12"))

md("""
## 13–14. Stability, and the choice of k

**Stability first.** Bootstrap ARI re-clusters 200 subsamples and asks whether the same
partition comes back; seed stability asks whether it is an artefact of initialisation;
cross-algorithm ARI asks whether other methods agree; the consensus matrix asks, pair by pair,
how often two students land together at all.

**Then the rule.** It is stated in full in `choose_k` and was fixed before the result was
seen:

```
1. Discard any k where the smallest cluster < 5% of n        (unusable personas)
2. Discard any k where bootstrap ARI < 0.50                  (not reproducible)
3. Discard any k with profile differentiation < 50%          (a severity split,
                                                              not a persona set)
4. Cap the selection at k <= 6                               (actionability)
5. If the gap statistic selects k = 1 -> record "weak structure" and relabel,
   but continue
6. Among survivors, score each k by its mean rank across
   {silhouette, DB, CH, BIC, bootstrap ARI, cross-algorithm ARI, differentiation}
7. Ties -> prefer the smaller k (parsimony)
8. Print the full vote table, including every k that was discarded and why
```

Rules 1–4 are **screens on usability**, applied *before* any quality criterion is consulted;
rule 6 is the quality vote among what survives. That ordering is the whole point. It means
k = 2 is not rejected for scoring badly — it scores *well* on silhouette — but for failing a
stated, measurable requirement of the research question. The proposal asks for interpretable
stress *profiles*, and a solution whose between-cluster variation is mostly level rather than
shape does not answer that question.

The panel disagrees with itself, and the disagreement is printed rather than smoothed over.
""")
code(sections[13]["code"], sections[14]["code"], drv("SECTION 13-14"))

md("""
## 15. Holdout test

The model is fitted on train only, frozen, and then applied to the 30% that model selection
never saw. Four tests:

- **(a)** do the centroids reproduce within tolerance?
- **(b)** do the cluster proportions match?
- **(c)** does silhouette hold up out of sample?
- **(d)** does a model fitted *fresh* on the holdout recover the same partition as the frozen
  model's predictions?

Test (d) is the strict one — it can fail even when (a)–(c) all pass, because it asks whether
the structure is genuinely there in new data rather than whether the frozen boundaries can be
applied to it.

Cluster labels are also **canonicalised** here. k-means numbers its clusters by initialisation
order, which changes with the seed; sorting them deterministically by overall strain level is
what allows the persona names in §17 to be *generated* and still be stable run to run.
""")
code(sections[15]["code"], drv("SECTION 15"))

md("""
## 16. Profiling

Centroid z-profiles, radar charts, item-level means back on the original 1–5 scale, and an
**η² ranking** of what actually separates the clusters.

η² earns its place: a dimension with a high η² is doing the work of separating the personas,
while a low-η² dimension is along for the ride. §17 uses this to decide which dimensions are
even *eligible* to appear in a persona name — otherwise a persona can end up named after a
dimension on which every cluster sits in much the same place, which reads as a distinction
but is not one.
""")
code(sections[16]["code"], drv("SECTION 16"))

md("""
## 17. Persona naming

**Names are composed, never typed in.** Each label is built from the cluster's own strongest
z-deviations, restricted to dimensions that actually separate the clusters (η² above Cohen's
"small" floor), capped at three terms. The only editorial input is a fixed vocabulary of one
phrase per (dimension, direction) pair — *which* phrases fire, and for which cluster, is
decided entirely by the centroid.

This section also reports the **runner-up k**. The rule crowns exactly one value, but it can
separate two candidates by a hair, and when it does, the runner-up is a real finding rather
than a discard: a finer segmentation that also passed every usability screen. Reporting it
costs one table and one figure, and it stops the crowned k from looking more inevitable than
the vote actually made it.
""")
code(sections[17]["code"], drv("SECTION 17"))

md("""
## 18. External validation

Classes-to-clusters against the five variables that were held out of every feature space:
year, gender, living arrangement, department and backlog.

**Effect sizes are mandatory here, not optional.** At n = 987 a chi-square test returns a
significant p-value for associations far too small to mean anything, so Cramér's V is reported
alongside every test and is the number to read. At-chance agreement is a real, reportable
negative and is printed as such.

CGPA appears in this table too, but flagged as a **manipulation check rather than evidence** —
it is inside the primary feature space, so agreement with it is guaranteed by construction.
That is the cost of the §9 decision, paid openly.
""")
code(sections[18]["code"], drv("SECTION 18"))

md("""
## 19. Free-text corroboration

Two open-ended questions, and three properties of the answers decide the method: they are
**very short** (median 4–5 words, roughly 30% are two words or fewer), **code-mixed** across
three writing systems (English, Bangla script, romanised Bangla in Latin letters), and
**uniformly negative** in sentiment because the question asks for a source of stress.

Those three facts rule out the obvious approaches. Sentiment analysis has no variance to find.
Transformer embeddings have almost no context to work with at four words, and no pretrained
model covers romanised Bangla. English stemming and stopword pipelines would silently discard
the ~12% of answers written in Bangla script. A hand-built, code-mixed, multi-label lexicon is
the method that actually covers what students wrote.

The text does **four jobs**, in order of importance:

1. **Corroborating the personas.** The clusterer never saw this text, so if a persona built
   purely from Likert answers also over-mentions the matching theme in students' own words,
   that is genuine independent evidence. Tested with χ² under Bonferroni correction across the
   13 themes — and reported honestly whether it clears the bar or not.
2. **Finding stressors the questionnaire never asked about.** The one thing only free text can
   do, and the most directly actionable output for the department.
3. **Comparing prompted agreement against unprompted salience.** An item everyone endorses but
   nobody volunteers is measuring assent, not salience — a result about the instrument itself.
4. **Q19 as a retrospective view**, giving a within-person before/after comparison the
   cross-sectional design otherwise cannot provide.

A TF-IDF + NMF topic model runs as a cross-check that is **expected to partially fail** on the
Bangla-script answers. That failure is not a limitation of the analysis — it is the evidence
for why the code-mixed lexicon was necessary, which turns a design choice into a reported
finding.

The lexicon below is frozen and versioned. `fingerprint()` hashes the pattern set and the
digest is written into `results.json`, so this inline copy and `src/lexicon.py` can be checked
for drift from the outputs alone.
""")
code(lex.replace("from __future__ import annotations\n", ""))

sec19 = sections[19]["code"]
imp_start = sec19.index("try:\n    from lexicon import")
imp_end = sec19.index("#: Stressors students volunteer")
sec19 = (sec19[:imp_start]
         + "# The frozen lexicon is defined in the cell above, so this notebook stays\n"
           "# self-contained on Kaggle. src/lexicon.py holds the identical patterns and\n"
           "# both report fingerprint() into results.json, which makes drift detectable.\n\n"
         + sec19[imp_end:])
code(sec19, drv("SECTION 19"))

md("""
## 20. Supervised interpretability

A depth-limited decision tree over the cluster labels, with 10-fold cross-validation and a
printed rule set per persona — the J48 equivalent the methodology report asks for — plus
random-forest permutation importance as a second opinion.

**This is interpretability, not prediction.** The labels came from the clusterer, so accuracy
here measures how compactly the partition can be re-described as if-then rules, not how well
anything generalises to new students. The rules are the deliverable: they are what lets a
counselling office assign a student to a persona from a handful of answers without running
k-means.

Permutation importance is preferred over Gini importance because Gini is biased toward
high-cardinality features, whereas permutation importance measures the accuracy actually lost
when a column is shuffled.
""")
code(sections[20]["code"], drv("SECTION 20"))

md("""
## 21. Persona cards

One card per persona: name, size, defining dimensions, demographic tilt, the top stressors
those students volunteered in their own words, and a recommendation.

Every element on a card is generated — the name from the centroid, the tilt from the held-out
background variables, the themes from the free text, the recommendation from the persona's own
top dimension. Nothing is typed in per cluster, so the cards cannot drift away from the model
behind them.

This is the figure the written report and the slides are built around.
""")
code(sections[21]["code"], drv("SECTION 21"))

md("""
## 22. Persist

A `joblib` bundle (scaler + subscale spec + model + generated names), per-student assignments
with silhouette values, every table as CSV, `results.json`, and `stress_prepared.arff` for the
WEKA route documented in `docs/WEKA_APPENDIX.md`.

The bundle deliberately stores the **subscale spec** alongside the fitted scaler and model.
Without it, scoring a new respondent would require re-deriving the facets by hand, which is
exactly the kind of step that silently drifts between semesters.
""")
code(sections[22]["code"], drv("SECTION 22"))

md("""
## 23. Reuse — scoring a new respondent

`assign_persona()` takes **raw 1–5 answers** exactly as the form collects them, plus the CGPA
band, and applies reverse coding, subscale construction, ordinal encoding and scaling
internally.

That interface is the point. It makes this a reusable instrument for next semester rather than
a one-off script, because the caller cannot get the preprocessing wrong — the caller never sees
it. It also returns the distance to the assigned centroid and the margin to the next one, so a
borderline student is visible as borderline instead of being silently rounded into a persona.
""")
code(sections[23]["code"], drv("SECTION 23"))

md("""
## Summary and limitations

The limitations below are printed by the notebook itself, into its own output, in its own
words. They are part of the result rather than an appendix to it — in particular the first
one, which states plainly that the structure here is weak and that these personas are a
defensible segmentation of a **continuum**, not four naturally separated populations.
""")
code(drv("SUMMARY"))

for _i, _c in enumerate(cells):
    _c["id"] = "%s%02d" % ("md" if _c["cell_type"] == "markdown" else "code", _i)

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                  "name": "python3"},
                   "language_info": {"name": "python", "version": "3.11"}},
      "nbformat": 4, "nbformat_minor": 5}

# nbformat wants each source line to keep its newline except the last.
for c in nb["cells"]:
    s = c["source"]
    c["source"] = [l + "\n" for l in s[:-1]] + [s[-1]] if s else []

os.makedirs(os.path.dirname(OUT), exist_ok=True)
io.open(OUT, "w", encoding="utf-8").write(json.dumps(nb, indent=1, ensure_ascii=False))
n_code = sum(1 for c in cells if c["cell_type"] == "code")
n_lines = sum(len(c["source"]) for c in cells if c["cell_type"] == "code")
print("wrote %s" % OUT)
print("  %d cells (%d markdown, %d code), %d lines of code"
      % (len(cells), len(cells) - n_code, n_code, n_lines))
