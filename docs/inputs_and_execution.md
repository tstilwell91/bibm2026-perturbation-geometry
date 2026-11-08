# Inputs and execution

This repository contains code, specifications and compact results. It does not contain the large frozen inputs needed to rerun the analyses, and public raw counts alone are not a drop-in replacement for them.

## How the scripts find their data

Code and data live in separate places:

- **Code** is in this repository. Scripts that depend on other scripts find them relative to their own location, so the repository can sit anywhere.
- **Data** are read from and written to a fixed *data workspace* layout. All data paths are relative to the **current working directory**.

To run a step, change into the root of a prepared data workspace and call the script by its repository path:

```bash
cd /path/to/data-workspace
python /path/to/multiscale-perturbation-geometry/analysis/responses/build_responses.py
```

The workspace directory names below (for example `source/derived/downstream/response_construction_v1/`) are those of the original analysis. They are kept unchanged because the scripts, checksums and provenance records refer to them. The result tables in this repository's `results/` directory are copies; scripts never read them and never write into this repository.

## Order of steps and required inputs

Every path in this section is inside the data workspace. None of these inputs is distributed.

### 1. Cell alignment and response construction: `analysis/responses/`

- `source/derived/v35_state_all_targets/STATE_ALL_TARGETS_CELL_MANIFEST.tsv.gz`: the frozen 21,710,586 eligible cells, with source-row pointers.
- Cell-level representations, each with a cell-ID sidecar and validation audit:
  - PCA scores: `source/derived/pca_direct_full_materialized_v28/runs/job-6848910/`
  - scVI posterior means: `global_scvi/final/primary_2k_allcell_v1/seed42/latent_v2/`
  - Geneformer chunked embeddings: `geneformer_global_cell_embeddings/`
  - STATE shards: `source/derived/v35_state_all_targets/STATE/`
- `audit_common_cells.py` also cross-checks historical manifest files under `source/derived/`.
- `build_responses.py` writes `source/derived/downstream/common_cell_audit_v1/` outputs (via the audit) and `source/derived/downstream/response_construction_v1/{pca,scvi,geneformer,state}/`. Each representation directory holds `response_vectors.npy` (144,368 rows; 50, 50, 1,152 and 2,058 dimensions), `response_units.tsv.gz` and the NTC-centroid files.
- For provenance only, `build_responses.py` hashes the response-construction specification (`V31_CANONICAL_RESPONSE_CONSTRUCTION.md` at the workspace root) and two historical PCA-response files under `source/derived/pca_full_response_v29/`. These must be present in the workspace, but they are not inputs to the calculation.

### 2. Agreement metrics: `analysis/`

Run in this order. Each step reads the outputs of the earlier ones from `source/derived/downstream/`.

1. `global_geometry/run_response_geometry.py`: all-pair cosine parquet files and whole-map agreement.
2. `local_geometry/run_local_geometry.py`, then `finalize_local_geometry.py`: exact kNN (deterministic ordering, cosine descending with alphabetical ties).
3. `global_geometry/run_distance_spectral.py`: target-profile agreement and response spectra. Needs the kNN output from step 2.
4. `graph_geometry/run_graph_geometry.py`: union-kNN graphs and Louvain communities (k = 15, seed 42, resolution 1.0). Refuses to overwrite existing graph outputs.
5. `multiscale/run_multiscale.py`: O_k across k, RBO and strongest-edge overlap. Writes to `results/bibm_extensions_20261005/agent_c_multiscale/`.
6. `edge_variance/run_edge_variance.py` and `run_edge_variance_tails.py`: factorial variance decomposition over targets present in all 48 blocks.
7. `cell_coherence/run_cell_coherence.py`: needs the cell-level representations, the common-cell index and the NTC centroids. Writes `sampled_cell_coherence.parquet` (1,166,604 sampled leave-one-out cells), which the robustness steps use.

### 3. Known biology: `biology/`

`run_known_biology.py` needs the exact resource snapshots listed in [configs/biology_resources.tsv](../configs/biology_resources.tsv) under `source/derived/downstream/response_prebiology_v1/known_biology/resources/`, including the STRING protein-name mapping. The filters are HuMAP3 probability ≥ 0.5, STRING v12 combined score ≥ 700, the frozen Reactome human interaction export, and SIGNOR direct human protein–protein records. Do not substitute current downloads if you are claiming exact reproduction. `analyze_biology.py` additionally needs the graph edge parquet from step 2.4.

### 4. Robustness: `robustness/`

**Split-half** (`split_half/`):

1. `split_half.py` loads the response builder from `analysis/responses/`. It needs the persisted split assignment `split_cells.npy` (20 splits × 21,710,586 cells) and the cell-level representations.
2. `run_parallel.py` uses `kernels.py` (Numba) for accelerated split construction.
3. `summarize_contiguous.py` (which uses `slab_access.py`) computes within-representation reliability.
4. `run_matched.py` performs the matched opposite-half calibration. It imports `summarize_contiguous.py` and takes only the pure `rbo` function from `analysis/multiscale/run_multiscale.py`. It needs the persisted half-response and support arrays, the fixed pair samples under `pair_samples/`, and the split provenance and integration audit JSON files. It also appends to an existing report in its workspace directory, so run it only in an expendable working copy.
5. `summarize_self_cross.py` summarizes the full `matched_half_calibration.tsv` written by `run_matched.py`.

All split-half data are read and written under `results/bibm_extensions_20261005/agent_a_reliability/` and `results/bibm_extensions_20261005/integrated/matched_half_calibration/`.

**≥10-cell sensitivity** (`min_cell_count/`):

- `run_local_sensitivity.py` imports the local-geometry and graph functions from `analysis/`. It rebuilds exact top-15 neighborhoods and communities on the retained units, falling back to full-vector queries when the stored top-30 cache is too short. It is not a subset of the original graph.
- `validate_filtered_neighbors.py` spot-checks those neighborhoods against exact queries.
- `run_global_sensitivity.py` reranks the all-pair cosine parquet files with and without the mask and recomputes coherence ranks. It needs a lot of memory, so run donor blocks one at a time unless you have checked.
- `summarize.py` combines the outputs above and the bootstrap results into the final sensitivity table.

**Coherence bootstrap** (`coherence_bootstrap/`): `bootstrap_sampled_cell.py` imports helpers from `bootstrap_coherence.py` and needs `sampled_cell_coherence.parquet` from step 2.7.

Outputs of these steps go under `results/camera_ready_robustness_20261108/` in the workspace.

## Limits

- Cohort sizes, file names and representation sets are fixed and checked. There is no generic `--input`/`--output` interface.
- Several scripts check the checksums of upstream outputs and stop if an input does not match the frozen record.
- Only syntax, local-dependency and link checks were run when this repository was assembled. No numerical analysis, model fitting or embedding extraction was rerun.

## Changes from the original analysis code

The scripts were moved from the original analysis tree into this layout. Changes were limited to file locations:

- Absolute researcher paths were replaced by working-directory-relative paths, before this reorganization.
- Where one script loads another (`split_half.py` → response builder; `run_parallel.py` → `split_half.py`; `run_matched.py` → split-half summary module and multiscale RBO function; `run_local_sensitivity.py` → local-geometry and graph modules), the path now points at the script's location in this repository.
- `run_matched.py` and `summarize_self_cross.py` previously wrote next to the script file. They now use the same workspace directory explicitly, so they never write into this repository.
- The script path recorded in two provenance summaries (`build_responses.py`, `run_response_geometry.py`) is now computed relative to the working directory, so the code no longer has to sit inside the workspace.
- `robustness/min_cell_count/run_local_sensitivity.py` was named `run_sensitivity.py`.

No numerical expression, estimator, cohort, seed, threshold or parameter was changed. [MANIFEST.csv](../MANIFEST.csv) maps each file to its original release path and states whether it is byte-identical.
