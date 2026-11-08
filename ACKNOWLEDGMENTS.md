# Acknowledgments and attribution

This work uses the original Zhu et al. Perturb-seq dataset, [Geneformer](https://huggingface.co/ctheodoris/Geneformer), [STATE](https://github.com/ArcInstitute/state), scVI/scvi-tools, BPCells, Seurat/SeuratObject, and the Python scientific-computing libraries identified in [requirements.txt](requirements.txt) and the [recorded software versions](docs/software_versions.md).

Biological evaluation uses HuMAP3, STRING, Reactome and SIGNOR. Exact source snapshots, definitions, retrieval URLs and hashes are recorded in the included [resource manifest](configs/biology_resources.tsv). No raw databases or pretrained weights are redistributed.

STATE attribution: Adduri et al., *Predicting cellular responses to perturbation across diverse contexts with State*; see the [original model publication](https://doi.org/10.1016/j.cell.2026.07.052). Exact checkpoint identities and implementation revisions are provided in [model provenance](configs/models.json).

ChatGPT and Claude Code assisted with release preparation and independent code-release reviews. These reviews do not constitute independent numerical reproduction of the scientific analyses. Responsibility for the scientific methods and interpretation remains with the authors.
