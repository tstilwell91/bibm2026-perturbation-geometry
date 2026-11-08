# Recorded software, not an installation guarantee

The finalized robustness records specify Python 3.12.8, NumPy 2.2.6, pandas 3.0.5, SciPy 1.18.1, PyArrow 25.0.1 and NetworkX 3.5. [requirements.txt](../requirements.txt) records these pins and leaves packages without established run-specific pins explicit and unpinned. No clean installation or dependency resolution has been performed.

Encoder/training software is provenance only: it is not required merely to inspect the code or read result tables. Geneformer production records PyTorch 2.8.0+cu128 / CUDA 12.8, Python 3.12.8 and Transformers 4.46.0. STATE records arc-state 0.11.3 and a frozen source revision; a complete contemporaneous environment lock is missing. scVI production records scvi-tools 1.5.0.post1 and PyTorch 2.8.0+cu128; the documented shared calibration stack has AnnBatch 0.2.2, Lightning 2.6.5, Zarr 3.3.0, zarrs 0.2.3. Those calibration records are not a complete production-run environment lock. PCA records BPCells 0.3.1, Seurat 5.3.0, SeuratObject 5.2.0 and an R/4.5.1 launcher module.

No end-to-end execution, full model fitting or encoder extraction is claimed for this research-code release. Only syntax, import resolution and selected command-line help checks have been performed; no full numerical rerun is claimed.
