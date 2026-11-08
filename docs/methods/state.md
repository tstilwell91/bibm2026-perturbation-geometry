# STATE inference

*Internal specification identifier: V35. Code, configurations and provenance records refer to this method by that identifier.*

> Artifact paths in this specification identify the original frozen research inputs or provenance. They are not distributed in this repository. Included files are listed in [MANIFEST.csv](../../MANIFEST.csv); external-input requirements are described in [inputs and execution](../inputs_and_execution.md).


**Status: CANONICAL / PRIMARY ANALYSIS**

## Purpose and scope

This document defines the authoritative method for producing STATE cell-level
representations for the genome-wide Kraj Perturb-seq cross-representation
analysis. It governs scientific inference methodology independently of the supported GPU execution platform.

## Frozen population and manifest

Use the frozen eligible population of 21,710,586 cells: 20,771,051
target-assigned cells and 939,535 eligible NTC cells. It comprises donors D1–D4
and contexts Rest, Stim8hr, and Stim48hr, represented by twelve authoritative
donor × context source H5ADs.

The frozen manifest is
`source/derived/v35_state_all_targets/STATE_ALL_TARGETS_CELL_MANIFEST.tsv.gz`.
Each production execution unit is one source H5AD / donor × context shard, for
twelve independent inference shards. The shards may execute independently;
one context-wide or whole-population inference process is not required.

## Authoritative inputs and preprocessing

Production input files are
`source/cell/D{1-4}_{Rest|Stim8hr|Stim48hr}.assigned_guide.h5ad`. Use raw UMI
counts from each source H5AD's `X` matrix. Input counts must be finite,
nonnegative, and integer-valued. Supply raw counts to STATE's native
preprocessing and inference path.

Do not externally apply log1p, CP10K/library-size normalization, scaling, or
HVG filtering. Do not substitute V29 prelogged matrices.

The following V29 items are historical/diagnostic artifacts, not canonical
production inputs: V29 log1p(raw UMI) derived H5ADs, preprocessing A/B/C
experiments, `preprocessing_contract/report.json`, and
`preprocessing_contract/repeatability_report.json`. They may document prior
investigation but must not replace the raw source H5ADs for V35 production.

## Gene handling

Use the canonical endogenous source universe of 18,129 genes in its canonical
ordering. Exclude `CUSTOM001_PuroR`. The established STATE mapping contains
17,958 mapped genes and 171 unmapped canonical genes. Do not invent mappings
for the 171 unmapped genes.

## Frozen model and implementation

Use pretrained STATE SE-600M epoch16 without retraining or fine-tuning.

- Checkpoint: `external_models/state/SE-600M/se600m_epoch16.ckpt`
- Protein embeddings: `external_models/state/SE-600M/protein_embeddings.pt`
- arc-state version: `0.11.3`
- STATE source revision: `9bbfe78a434a55205e4de834e1ea99f85f7a3add`
- STATE source: `external_models/state/state`

Use the established STATE inference configuration and fixed seed behavior. Do
not create a special per-file seed profile. Record relevant inference
parameters and seed in provenance. The repeatability diagnostic found exact
reproduction for repeated identical inputs under identical conditions.

## Canonical representation and outputs

Each eligible cell receives the 2,048-D normalized STATE CLS embedding and the
official 10-D STATE dataset correction, concatenated into a 2,058-dimensional
float32 representation. Every output shard must have an exact ordered cell-ID
mapping and must be validated before downstream use.

The current V35 worker is
`source/derived/v35_state_all_targets/jobs/infer_state_shard.py`. Its bounded
I/O streaming, incremental output writing, and I/O chunk size are implementation
mechanisms; they must not redefine the scientific input or output. Do not
silently change STATE preprocessing or model semantics for memory convenience.

## Execution and hardware provenance

Execution infrastructure is not part of the biological representation
definition. Shards may run on supported GPU infrastructure, including A100 or H100 hardware, provided the locked model, input contract, gene
mapping, STATE implementation, and output definition are preserved. These
hardware choices do not define separate scientific methodologies. Do not
require bitwise equality across GPU architectures; record and validate any
cross-hardware numerical differences as numerically reasonable.

Record for each shard, where practical: cluster, node, GPU model, STATE version
and source revision, checkpoint, source H5AD, selected cell count, ordered
cell-ID digest, seed and configuration, and final dimensions and dtype.

## Relationship to V31 response construction

STATE inference produces cell representations only. The required order is:

`V35 STATE inference`
→ cell-level STATE representations
→ `response_construction.md`
→ perturbation-response geometry

V35 inference must not perform NTC centroid construction, NTC subtraction,
guide weighting, target aggregation, perturbation-response construction,
cosine geometry, or biological donor/context pooling.

## Locked methodological statement

For the frozen 21,710,586-cell V35 eligible population, produce one STATE
representation shard for each of the twelve authoritative donor × context raw
source H5ADs by passing finite, nonnegative, integer-valued raw UMI counts in
canonical 18,129-gene order (excluding `CUSTOM001_PuroR`) through native STATE
0.11.3 preprocessing and frozen SE-600M epoch16 inference, using the established
17,958-gene mapping and fixed seed behavior, and persist an ordered-cell-ID,
2,058-D float32 representation consisting of the normalized CLS embedding and
official 10-D dataset correction; response construction follows V31.
