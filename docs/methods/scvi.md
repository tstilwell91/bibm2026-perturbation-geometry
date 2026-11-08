# scVI fitting and cell representation

*Internal specification identifier: V33. Code, configurations and provenance records refer to this method by that identifier.*

This public documentation summarizes the completed primary scVI analysis. Scientific settings are unchanged; obsolete pre-production instructions and infrastructure details are omitted. The finalized [method lock](../../configs/scvi_method_lock.json) and [seed-42 configuration](../../configs/scvi_config.json) provide the run-specific training definition. [V31](response_construction.md) governs downstream responses; [V32](pca.md) governs the matched PCA feature set.

## Population and input

One global model was fitted to all **21,710,586** frozen eligible target and NTC cells, with no production validation holdout. All 10,000 calibration holdout cells returned to production training. Donor, context, lane, guide and target labels were retained for evaluation, not supplied as batch variables or covariates.

Input was raw integer-valued UMI counts for the same exact ordered **2,000 VST HVGs** used by primary PCA. `CUSTOM001_PuroR` was excluded. Counts were not externally normalized, logged, scaled or PCA-transformed for scVI.

The included [ordered feature list](../../configs/pca_features_2000.txt) has SHA-256 `4c8af16cbb81ba473e5a773014206298f1cbd88940996580168fb57f0b8eccbb`. The distinct source RDS hash is `ca0ddd506af7a2caeb3990b1668904c3694b76ab31cf7e3abbc5c7ad0e0a6d3a`.

Production inputs came from validated frozen-manifest AnnBatch/Zarr shards. Sparse CSR data used 32,768-cell AnnData groups and 512-row Zarr chunks; the combined collection retained deterministic cell ordering (`shuffle=False`), `layer=None`, `batch_key=None` and no categorical or continuous covariates. Shard validation checked exact IDs/order, raw-count integrity and the ordered gene axis before fitting. These large input shards and row manifests are external to this release; see [input contracts](../inputs_and_execution.md).

## Architecture and training

| Setting | Primary value |
|---|---:|
| `n_hidden` | 128 |
| `n_latent` | 50 |
| `n_layers` | 1 |
| `dropout_rate` | 0.1 |
| `dispersion` | `gene` |
| `gene_likelihood` | `zinb` |
| `latent_distribution` | `normal` |
| `batch_key` | `None` |
| Covariates | none |

The primary seed is **42**, with fresh initialization, Adam learning rate **0.005**, weight decay **1e-6**, batch size **4096**, and KL warmup over **4000 optimizer steps**. Learning-rate reduction on plateau was disabled.

Training completed exactly **one full epoch / 5301 optimizer updates**: 5300 full batches and a final partial batch of 1786 cells. Every eligible cell was visited once; the final batch was retained. Seeds 4242 and 2026 identify robustness replicates, not substitutes for the predeclared primary fit. The stopping point was a conservative complete-population choice informed by calibration, not a claim of the exact validation-optimal checkpoint. Downstream geometry and biological results were not used to select the configuration.

## Recorded software and outputs

The documented stack uses Python 3.12.8, scvi-tools 1.5.0.post1, AnnBatch 0.2.2, zarrs 0.2.3, Zarr 3.3.0, PyTorch 2.8.0+cu128 and CUDA 12.8. The [software notes](../software_versions.md) distinguish production-state evidence from shared calibration environment records; no complete portable production environment lock is claimed.

[Training completion](../../configs/scvi_training_record.json) records one epoch, 5301 steps, finite model parameters, corrected loss capture and successful strict save/reload checks. Persistence uses a production-owned state artifact rather than assuming the standard SCVI save path works for a registry-only AnnBatch model.

The validated [latent audit](../../configs/scvi_latent_audit.json) identifies **latent_v2**: one finite float32 50-dimensional posterior-mean vector per eligible cell, exact canonical manifest order, no missing/duplicate cells. The output arrays and model state are not included here.

Representation fitting precedes response construction: global raw-count scVI fit → cell-level posterior means → unchanged V31 matched-control correction and target × donor × context arithmetic means. No donor-, context- or target-panel-specific scVI refitting is part of the primary analysis.
