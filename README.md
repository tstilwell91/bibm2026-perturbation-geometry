# Multiscale Stability of Perturbation-Response Geometry Across Four Single-Cell Representations

Research code for the accepted **IEEE BIBM 2026 RLPD-Bio workshop** paper by **Terry Stilwell, Piotr Kraj, and Desh Ranjan**. See [CITATION.md](CITATION.md).

We encode the same 21,710,586 cells from a genome-scale CRISPRi Perturb-seq screen of primary human CD4+ T cells (four donors; rest, 8 h and 48 h stimulation) with PCA, scVI, Geneformer V2-316M and STATE SE-600M. In every representation we construct perturbation responses identically and ask how well perturbation relationships agree across representations at different scales, from the whole similarity map down to single cells.

This repository contains the analysis code, method specifications, model provenance and compact result tables behind the paper. **It does not contain cell-level data, embeddings or model weights, so it is not a turnkey reproduction.** See [what you need to run the code](docs/inputs_and_execution.md).

## Repository layout

| Directory | Contents |
|---|---|
| [`analysis/`](analysis) | Response construction and the core cross-representation agreement metrics |
| [`robustness/`](robustness) | Split-half reproducibility, the ≥10-cell sensitivity analysis and the coherence bootstrap |
| [`biology/`](biology) | Known-pair enrichment (HuMAP3, STRING, Reactome, SIGNOR) |
| [`configs/`](configs) | Cohort definition, the ordered 2,000-gene PCA/scVI feature list, scVI configuration and exact model provenance |
| [`results/`](results) | Compact numerical tables supporting the paper; see [results/README.md](results/README.md) |
| [`docs/`](docs) | Method specifications, input requirements and recorded software versions |

## Analyses and code

| Analysis | Code |
|---|---|
| Matched-control response construction | [`analysis/responses/`](analysis/responses): `audit_common_cells.py` aligns cell identities across representations; `build_responses.py` builds the responses |
| Whole-map agreement and target profiles | [`analysis/global_geometry/`](analysis/global_geometry) |
| Exact k-nearest neighbors | [`analysis/local_geometry/`](analysis/local_geometry) |
| Union-kNN graphs and communities | [`analysis/graph_geometry/`](analysis/graph_geometry) |
| Neighborhood scales, RBO and strongest-edge overlap | [`analysis/multiscale/`](analysis/multiscale) |
| Edge-level variance decomposition | [`analysis/edge_variance/`](analysis/edge_variance) |
| Cell-level coherence | [`analysis/cell_coherence/`](analysis/cell_coherence) |
| Known-biology enrichment | [`biology/`](biology) |
| Split-half reproducibility | [`robustness/split_half/`](robustness/split_half) |
| ≥10-cell sensitivity | [`robustness/min_cell_count/`](robustness/min_cell_count) |
| Coherence-rank confidence intervals | [`robustness/coherence_bootstrap/`](robustness/coherence_bootstrap) |

The curated-set and prespecified-program analyses described in the paper are not included.

## Methods

Responses are built the same way in every representation. Each target cell is corrected by subtracting the centroid of all eligible non-targeting control (NTC) cells from the same donor, context and lane. Corrected cells are then averaged directly within each target × donor × context, with no guide weighting and no pooling of donors or contexts. This gives 144,368 response units per representation.

Full specifications:

- [Response construction](docs/methods/response_construction.md)
- [PCA fitting](docs/methods/pca.md): one global fit on all eligible cells, frozen ordered 2,000 HVGs, 50 PCs
- [scVI fitting](docs/methods/scvi.md): one global fit on raw counts of the same 2,000 genes, 50 latent dimensions, no batch covariates
- [STATE inference](docs/methods/state.md): frozen SE-600M epoch-16 checkpoint, native preprocessing, 2,058-dimensional output

Exact checkpoint identities, content hashes and implementation revisions for Geneformer and STATE are in [configs/models.json](configs/models.json).

## Data

Data are from Zhu et al., Cell (2026), [doi:10.1016/j.cell.2026.08.002](https://doi.org/10.1016/j.cell.2026.08.002): [GEO GSE314342](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE314342), SRA SRP643211, and [CZI cell-level count matrices](https://virtualcellmodels.cziscience.com/dataset/genome-scale-tcell-perturb-seq). No raw data, cell identifiers, embeddings, response arrays, model weights or biological databases are redistributed here.

## Running the code

Use Python 3.12 with [requirements.txt](requirements.txt); see [software versions](docs/software_versions.md). Clean installation and full numerical reruns have not been tested.

The scripts are full-cohort research programs, not a general toolkit. They read and write a fixed data workspace layout, using paths relative to the **current working directory**. Run them from the root of a prepared data workspace, not from this repository, for example `python /path/to/repo/analysis/responses/build_responses.py`. Some scripts start running as soon as they are imported, and some refuse to overwrite existing outputs. [docs/inputs_and_execution.md](docs/inputs_and_execution.md) lists the required inputs and the order of the steps.

## Notes on interpretation

- Edges within a similarity map are dependent. Nominal correlation p-values are not biological-replicate inference.
- Unannotated biological pairs are unknown, not negatives.
- The coherence bootstrap intervals are conditional on the observed donors. They are not intervals for the whole-map, neighbor-overlap or graph metrics.
- Whether any Zhu et al. cells were in the Geneformer or STATE pretraining data has been neither established nor excluded.

## License

No license has been selected yet.

## Acknowledgments

See [ACKNOWLEDGMENTS.md](ACKNOWLEDGMENTS.md).
