# PCA fitting and cell representation

*Internal specification identifier: V32. Code, configurations and provenance records refer to this method by that identifier.*

> Artifact paths in this specification identify the original frozen research inputs or provenance. They are not distributed in this repository. Included files are listed in [MANIFEST.csv](../../MANIFEST.csv); external-input requirements are described in [inputs and execution](../inputs_and_execution.md).

## Locked PCA methodology for the Kraj Perturb-seq cross-representation study

**Status:** CANONICAL / PRIMARY PCA REPRESENTATION  
**Applies to:** all primary PCA-based analyses in the Kraj Perturb-seq project  
**Relationship to V31:** V32 defines how the PCA cell representation is fit. `response_construction.md` defines what happens afterward to construct perturbation-response vectors.

---

## 1. Purpose

This document freezes the production PCA definition for the primary cross-representation study.

The project evaluated several PCA fitting and implementation strategies during development, including sketch-based fitting, lazy normalization, and materialized normalization. Those experiments were useful for validation and performance engineering, but the primary PCA representation is now fixed.

The scientific PCA definition and the preferred computational implementation are separated below:

- **Scientific definition:** the mathematical PCA space that must be reproduced.
- **Preferred implementation:** the fastest validated way to compute that same space.

Do not change the scientific definition for convenience or performance.

---

## 2. Canonical PCA population

Fit PCA directly on the full frozen eligible population.

Current canonical population:

- **21,710,586 eligible cells**
- all eligible target and NTC cells included
- one shared PCA coordinate system
- no donor-specific PCA
- no context-specific PCA
- no target-specific PCA
- no 15-target-only PCA refit

PCA is fit globally once.

Focused analyses, including the 15-target biological panel, must use cells extracted from this same global PCA representation rather than fitting a new PCA space on the subset.

This ensures that the focused 15-target analysis is a true subset of the genome-wide analysis.

---

## 3. Canonical gene universe

The source files contain 18,130 genes, but the validated endogenous analysis basis contains:

- **18,129 canonical endogenous genes**
- `CUSTOM001_PuroR` excluded

The full 18,129-gene count vector is used to compute each cell's library-size normalization denominator.

Do not compute library totals from only the 2,000 PCA features.

---

## 4. Canonical PCA feature set

Use the frozen set of:

- **2,000 VST highly variable genes**

These features are selected according to the validated project feature-selection procedure and then frozen.

For the production PCA:

- use the exact ordered frozen 2,000-HVG list;
- do not re-select HVGs per donor;
- do not re-select HVGs per context;
- do not re-select HVGs for the 15-target subset;
- do not allow a rerun to silently choose a different feature set.

The ordered feature list and its digest should be recorded with every production PCA run.

---

## 5. Canonical normalization

For each cell, compute the library-size denominator from all 18,129 canonical endogenous genes.

For each of the frozen 2,000 PCA genes, use:

`log1p(raw_count / full_library_total * 10000)`

Canonical settings:

- raw input: UMI counts
- denominator genes: all 18,129 canonical endogenous genes
- scale factor: **10,000**
- transform: `log1p`
- PCA features after normalization: frozen 2,000 VST HVGs

This normalization definition is part of the PCA scientific specification.

---

## 6. Centering and scaling

Compute gene-level centering and scaling statistics across the full eligible population.

Production settings:

- center each of the 2,000 normalized genes using the full-population mean;
- scale using the validated full-population sample standard deviation;
- retain the validated clipping behavior used by the production PCA implementation;
- current production clipping threshold: **10**

The same center and scale statistics must be used for all cells in the shared PCA coordinate system.

Do not compute separate statistics for donors, contexts, targets, or focused subsets.

---

## 7. PCA decomposition

Compute a single truncated SVD/PCA over:

**21,710,586 cells × 2,000 normalized, centered/scaled genes**

Canonical output:

- **50 principal components**
- one shared 50-dimensional representation for every eligible cell

Validated production solver settings:

- `k = 50`
- `nu = 50`
- `nv = 50`
- `threads = 40`
- `tol = 1e-5`
- `maxitr = 1000`

The exact `ncv` value was not recorded in the historical production comparison and should be recorded in future reruns if explicitly set.

---

## 8. Preferred computational implementation

The preferred production implementation is:

**raw counts  
→ full-library totals  
→ select frozen 2,000 HVGs  
→ LogNormalize those 2,000 genes using the full 18,129-gene denominator  
→ materialize the sparse normalized 2,000-gene matrix  
→ apply centering/scaling lazily  
→ multithreaded truncated SVD  
→ persist 21,710,586 × 50 cell scores**

The key engineering decision is to materialize the sparse normalized 2,000-gene matrix before iterative SVD.

Do **not** materialize the centered dense matrix.

Normalization preserves sparsity; centering does not. Therefore the preferred implementation materializes the normalized sparse matrix and keeps center/scale as a lazy transformation during SVD.

---

## 9. Validated implementation comparison

Two successful full-population PCA runs established that the materialized-normalization approach is substantially faster while producing the same PCA representation.

### Job 6848695 — lazy normalized matrix

Pipeline:

`lazy normalization → center/scale → SVD`

Results:

- Slurm runtime: **2:53:06**
- PCA SVD time: **2:21:10**
- library totals: **6:08**
- center/scale: **5:33**
- cells: **21,710,586**
- features: **2,000**
- PCs: **50**
- validation: **PASS**

Primary score artifact:

`source/derived/pca_direct_full_v28/runs/job-6848695/scores/pca_scores_21m_x_50.h5`

This representation was consumed by the completed V29 PCA response pipeline.

### Job 6848910 — materialized normalized 2k matrix

Pipeline:

`materialize normalized 2k sparse matrix → center/scale → SVD`

Results:

- Slurm runtime: **1:32:33**
- PCA SVD time: **48:54**
- normalized 2k materialization: **13:59**
- materialized normalized matrix size: approximately **84.9 GiB**
- cells: **21,710,586**
- features: **2,000**
- PCs: **50**
- validation: **PASS**

The materialized run completed **1:20:33 faster overall**, approximately a 47% reduction in observed Slurm runtime.

Its SVD completed **1:32:16 faster**, approximately a 65% reduction in SVD time.

## 10. Scientific equivalence of lazy and materialized implementations

The two successful implementations were compared directly.

They are effectively the same PCA representation.

Validated comparison:

- same **21,710,586 cells**
- same cell-order digest
- same ordered **2,000 HVGs**
- identical gene means
- identical gene scales
- identical full-library denominator digest
- matching normalization settings
- matching solver settings

### Singular values

- maximum absolute difference: **0**
- maximum relative difference: **0**
- explained-fraction differences: **0**

### Loadings

- minimum per-PC correlation after alignment: **1.0**
- maximum loading RMS difference: **0**
- sign flips required: **none**
- PC reordering required: **none**

### Subspace

- maximum principal angle: **2.09 × 10^-6 degrees**
- projector Frobenius difference: **5.16 × 10^-8**

These differences are at numerical precision.

### Cell scores

All **21,710,586** cell-score rows were compared.

- minimum per-PC correlation: **1.0**
- flattened correlation: **1.0**
- relative Frobenius error: **0**
- maximum absolute difference: **0**

Therefore:

> Materializing the normalized 2,000-gene sparse matrix changes computational efficiency, not the scientific PCA representation.

Reference comparison report:

the original implementation-equivalence report (not distributed)

---

## 11. Canonical production choice

The **scientific PCA definition** is the full-population direct PCA described in this document.

The **preferred implementation for future reruns** is the materialized normalized 2k workflow demonstrated by job 6848910.

Job 6848695 remains a valid production representation because its saved PCA scores were demonstrated to be equivalent to job 6848910.

Therefore:

- existing validated downstream analyses using job 6848695 do **not** need to be regenerated merely to switch implementation;
- future PCA reruns should default to the materialized normalized 2k workflow;
- implementation differences must not be interpreted as different PCA models when all canonical inputs and transformations are identical.

---

## 12. Relationship to response construction

PCA fitting/representation generation occurs **before** perturbation-response construction.

The canonical sequence is:

1. fit the global PCA space on all eligible cells;
2. obtain the 50-D PCA coordinate for every eligible cell;
3. apply `response_construction.md`.

Under V31, the response stage then performs:

`PCA cell coordinate  
→ matched all-NTC donor×context×lane centroid subtraction  
→ direct corrected-cell mean by target×donor×context`

PCA fitting must not depend on which perturbation targets will later be analyzed.

A 15-target analysis should therefore select the corresponding canonical response units after global fitting and response construction rather than refitting PCA on those 15 targets.

---

## 13. Completed V29 downstream validation

The completed response job **6848992** consumed the validated job-6848695 PCA scores and did not refit PCA.

Results:

- runtime: **5:10**
- exit code: **0**
- eligible/PCA cells: **21,710,586**
- NTC cells: **939,535**
- donor × context × lane NTC groups: **284**
- target cells: **20,771,051**
- perturbation targets: **12,604**
- donor × context × target response vectors: **144,368**
- target × context aggregated vectors: **37,578**
- response-unit cell count:
  - minimum: **1**
  - median: **106**
  - mean: **143.88**
  - maximum: **10,728**

Primary downstream artifacts:

`source/derived/pca_full_response_v29/production/`

V31 compliance audit:

the original V31 compliance report (not distributed)

The V31 audit found the PCA response implementation compliant with the locked response methodology.

---

## 14. Historical PCA experiments

Earlier project work evaluated:

- leverage-score sketches;
- proportional-uniform sketches;
- 25k / 50k / other sketch sizes;
- lazy full-population transformations;
- direct full-population PCA;
- materialized normalized-matrix PCA.

These experiments remain valuable as development history and sensitivity evidence.

They are **not** the primary production definition.

In particular:

- do not revert production PCA to sketch fitting merely because historical sketch outputs exist;
- do not fit a separate PCA space for the 15-target subset;
- do not treat the lazy and materialized implementations as separate biological representations;
- do not use historical leverage jobs as the canonical PCA implementation.

---

## 15. Locked methodological statement

> PCA is fit once on the complete frozen eligible cell population using 2,000 frozen VST highly variable genes. Per-cell normalization denominators are computed from the full 18,129-gene endogenous expression basis, after exclusion of CUSTOM001_PuroR, and the selected genes are LogNormalized with a scale factor of 10,000 before full-population centering and scaling. A single 50-component truncated-SVD PCA coordinate system is then used for all cells. For computational efficiency, future reruns should materialize the sparse normalized 2,000-gene matrix before applying lazy centering/scaling and multithreaded SVD; this implementation has been demonstrated to produce the same saved PCA representation as the fully lazy workflow while substantially reducing runtime.

---
