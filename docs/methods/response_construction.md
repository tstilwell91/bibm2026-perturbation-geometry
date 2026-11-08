# Response construction

*Internal specification identifier: V31. Code, configurations and provenance records refer to this method by that identifier.*

> Artifact paths in this specification identify the original frozen research inputs or provenance. They are not distributed in this repository. Included files are listed in [MANIFEST.csv](../../MANIFEST.csv); external-input requirements are described in [inputs and execution](../inputs_and_execution.md).

## Locked cross-representation methodology for Kraj Perturb-seq

**Status:** CANONICAL / PRIMARY ANALYSIS  
**Applies to:** PCA, Geneformer, scVI, STATE, and any future cell representation used in the perturbation-geometry study  
**Supersedes for primary cross-model analysis:** the V24/V26 1:1 NTC cohort rule and the genome-wide Geneformer equal-guide target estimator  
**Sensitivity analyses remain archived and should not be deleted.**

---

## 1. Purpose

The central experimental design is to compare perturbation-response geometry across representations while holding the biological observations and response-construction procedure fixed.

The **representation is the experimental variable**.

Everything downstream of the cell representation must be held constant across models:

- eligible cell identities;
- donor labels;
- context labels;
- lane labels;
- perturbation / target assignments;
- NTC matching;
- NTC reference population;
- cell correction;
- target aggregation;
- donor handling;
- context handling;
- support/QC rules.

No model may use a different response estimator in the primary cross-representation comparison.

---

## 2. Canonical population

Use the full frozen eligible population when the representation is available for those cells.

Current full-population reference:

- **21,710,586 eligible cells**
- **939,535 NTC cells**
- **20,771,051 target-assigned cells**
- **12,604 perturbation targets**
- **4 donors**
- **3 contexts:** Rest, Stim8hr, Stim48hr
- NTC matching strata: **donor × context × lane**

The historical 15-target 1:1 target:NTC cohort is no longer the primary cohort definition for the genome-wide cross-model comparison.

It remains useful as a sensitivity / historical reproduction analysis.

---

## 3. Canonical response estimator

For each representation `m`, let `z_i^(m)` be the representation vector for cell `i`.

### 3.1 NTC centroid

For every occupied donor × context × lane stratum `(d, c, l)`, compute the centroid using **all eligible NTC cells** in that stratum:

`mu_dcl^(m) = mean(z_j^(m) for all eligible NTC cells j in donor d, context c, lane l)`

**Primary rule:** use all eligible matched NTCs.  
Do not subsample NTCs for the primary analysis.

### 3.2 Correct each target cell

For target cell `i` in donor `d`, context `c`, lane `l`:

`corrected_i^(m) = z_i^(m) - mu_dcl^(m)`

Each target cell is corrected using the NTC centroid from its own donor × context × lane stratum.

Do not pool NTCs across donors, contexts, or lanes.

### 3.3 Construct the target response vector

For each perturbation target `t`, donor `d`, and context `c`, compute the **direct arithmetic mean of all corrected target cells**:

`response_tdc^(m) = mean(corrected_i^(m) for all target cells i in target t, donor d, context c)`

This is the canonical response vector used for cross-representation geometry.

---

## 4. No guide weighting in the primary estimator

Guide identity may be retained for:

- provenance;
- QC;
- guide support counts;
- sensitivity analyses;
- diagnosing guide imbalance or outlier guides.

Guide identity must **not** change the weighting of cells in the primary response vector.

Specifically, do **not** use this as the primary estimator:

`corrected cells -> guide means -> equal-weight mean across guides`

That estimator gives each guide equal target-level weight regardless of its cell count and therefore estimates a different quantity.

The primary estimator is:

`corrected cells -> direct target-cell mean`

Every corrected target cell contributes equally within its target × donor × context response unit.

---

## 5. Historical methods and sensitivity analyses

### 5.1 Historical 15-target V24/V26 method

The final focused 15-target analysis used:

- donor × context × lane NTC matching;
- a frozen **1:1 target:NTC sampled cohort** within each occupied stratum;
- cell-level NTC subtraction;
- **direct corrected-cell averaging** within target × donor × context;
- no guide weighting in the primary response.

This historical analysis should be reproducible, but its 1:1 NTC sampling rule is no longer the primary genome-wide rule.

### 5.2 Genome-wide Geneformer package

The first genome-wide Geneformer response package used:

- donor × context × lane NTC correction;
- guide-level corrected responses;
- equal guide weighting when forming target responses.

That package is **not canonical for the primary cross-model comparison**.

Keep it intact as a sensitivity / historical artifact.

### 5.3 PCA V29 genome-wide responses

The PCA V29 genome-wide response workflow used:

- all eligible matched NTCs;
- donor × context × lane NTC centroids;
- direct mean of corrected target cells.

This matches the newly locked canonical estimator and should be treated as the reference implementation of the response-construction rule.

---

## 6. Canonical pipeline for every model

Every representation must follow this exact sequence:

1. **Generate cell representations**
   - PCA: 50-D PCA coordinates
   - Geneformer: frozen encoder cell embedding
   - scVI: latent cell representation
   - STATE: frozen encoder cell representation
   - future models: model-specific cell representation

2. **Align identical eligible cell IDs and metadata**

3. **Compute all-NTC centroids by donor × context × lane**

4. **Correct each target cell**
   - target representation minus matched NTC centroid

5. **Average corrected target cells directly**
   - group by target × donor × context
   - arithmetic mean
   - no guide weighting

6. **Keep donors separate**
   - donor-specific target responses are the primary response units

7. **Keep contexts separate**

8. **Only after this identical construction compare geometry across representations**

---

## 7. What must be identical across representations

For a valid apples-to-apples comparison, all models must use the same:

- eligible cell manifest;
- target/NTC classification;
- donor labels;
- context labels;
- lane labels;
- target assignments;
- NTC population;
- NTC matching rule;
- target-cell inclusion rule;
- response aggregation rule;
- support filtering;
- response-unit ordering where practical.

The model-specific embedding dimension may differ.

Examples:

- PCA: 50 dimensions
- Geneformer: 1,152 dimensions
- STATE: 2,058 dimensions
- scVI: configured latent dimension

Unequal representation dimension is acceptable because geometry is computed within each representation and compared through representation-invariant summary metrics.

---

## 8. Required audits for each representation

Before geometry analysis, each representation package must report:

- represented cell count;
- exact cell-ID digest / manifest;
- number of NTC cells;
- number of target cells;
- number of donor × context × lane NTC groups;
- proof every target cell has a matched NTC group;
- number of distinct targets;
- number of target × donor × context response units;
- min / median / mean / max cells per response unit;
- finite-value checks;
- duplicate/missing-cell checks;
- checksum(s) for response artifacts;
- method configuration recording this V31 contract.

A representation package that does not satisfy this contract must not enter the primary cross-model comparison.

---

## 9. Primary cross-representation geometry

After canonical response construction, downstream analyses may include:

- response magnitude;
- pairwise cosine similarity between perturbation responses;
- representation agreement of flattened cosine maps;
- per-target relational/profile stability;
- k-nearest-neighbor overlap;
- same-nearest-neighbor retention;
- donor reproducibility;
- context-specific geometry;
- graph/community/spectral analyses;
- cell-level directional coherence where appropriate.

These analyses must consume the canonical response vectors, not representation-specific alternative estimators.

---

## 10. Sensitivity analyses

The following are useful sensitivity analyses but are **not** substitutes for the primary canonical estimator:

### NTC sensitivity
- all matched NTCs vs 1:1 sampled NTCs

### Aggregation sensitivity
- direct corrected-cell mean vs equal-guide target mean

### Other sensitivity analyses
- minimum response-unit cell-count thresholds;
- guide support thresholds;
- alternative target aggregation schemes;
- sketch-based vs direct PCA fitting;
- representation-specific preprocessing ablations.

Sensitivity outputs must be clearly labeled and stored separately from canonical primary outputs.

---

## 11. Locked methodological statement

> For every representation, target-cell embeddings are corrected by subtracting the centroid of all eligible non-targeting control cells from the same donor, stimulation context, and lane. Perturbation responses are then defined as the arithmetic mean of corrected target-cell embeddings within each target, donor, and context. Guide identities are retained for support and quality-control analyses but do not affect weighting in the primary response estimator. This response-construction procedure is held fixed across representations so that differences in downstream perturbation geometry can be attributed to the representation rather than to differences in control selection or aggregation.

---
