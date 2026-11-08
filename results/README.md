# Result tables

Compact numerical results behind the paper. Every table is a byte-identical copy of the finalized analysis output; only its file name and location changed (the original names are recorded in [MANIFEST.csv](../MANIFEST.csv)).

Common columns: `donor` (D1–D4), `context` (`Rest`, `Stim8hr`, `Stim48hr`), and `representation_a`/`representation_b` for a representation pair (PCA, scVI, Geneformer V2-316M, STATE SE-600M). "By context" tables average over the four donors.

## agreement/

| File | Contents | Produced by |
|---|---|---|
| `whole_map_by_donor_context.tsv` | Spearman and Pearson correlation between two representations' full target × target cosine maps, per donor and context | `analysis/global_geometry/run_response_geometry.py` |
| `whole_map_by_context.tsv.gz` | Context means and ranges of the above | same |
| `table_II_concordance.csv` | Check that the whole-map ρ and O15 values printed in Table II match these tables (`absolute_error`, `passed`) | final audit (script not included) |
| `target_profiles_by_donor_context.tsv.gz` | Agreement of each target's full cosine profile between representations (distribution over targets) | `analysis/global_geometry/run_distance_spectral.py` |
| `neighbors_by_donor_context.tsv.gz` | Exact k-nearest-neighbor overlap (O_k) per donor and context, with random expectation | `analysis/local_geometry/run_local_geometry.py` |
| `neighbors_by_context.tsv.gz` | Context means of the above | same |
| `multiscale_overlap_by_context.tsv` | O_k across neighborhood sizes k | `analysis/multiscale/run_multiscale.py` |
| `multiscale_rbo_by_context.tsv` | Finite rank-biased overlap (p = 0.9) at depths 15 and 20 | same |
| `strongest_edges_by_context.tsv` | Jaccard overlap of the strongest edges, from the top 50% down to the top 0.1% | same |
| `graph_edge_agreement.tsv.gz` | Union-kNN graph edge Jaccard per donor, context and k | `analysis/graph_geometry/run_graph_geometry.py` |
| `graph_community_agreement.tsv.gz` | Louvain community agreement (ARI, NMI) per donor, context and k | same |
| `graph_k15_by_context.tsv.gz` | Context summaries of graph agreement at k = 15 | same |
| `edge_variance_tails.tsv.gz` | Fraction of edge-level cosine variance due to each factorial term (`frac_R` representation, `frac_D` donor, …) in the strongest-edge tails | `analysis/edge_variance/run_edge_variance_tails.py` |
| `cell_coherence_agreement.tsv.gz` | Cross-representation correlation of leave-one-out cell coherence on the sampled cells | `analysis/cell_coherence/run_cell_coherence.py` |

## biology/

| File | Contents | Produced by |
|---|---|---|
| `resource_mapping_summary.tsv` | How many relationships in each biological resource mapped to screened targets | `biology/run_known_biology.py` |
| `recovery_by_context.tsv.gz` | Recall and enrichment of known pairs in the top 0.1%, 1% and 5% most similar pairs, by representation, context and resource | same |
| `enrichment_by_threshold.tsv.gz` | Known-pair enrichment at five similarity thresholds for every representation, donor, context and resource | `biology/analyze_biology.py` |
| `strength_matched_uncertainty.tsv` | Paired block-bootstrap intervals for the strength-matched annotation comparisons | same |

The exact resource snapshots and filters are listed in [configs/biology_resources.tsv](../configs/biology_resources.tsv).

## robustness/

| File | Contents | Produced by |
|---|---|---|
| `split_half_by_pair.tsv` | Matched opposite-half reproducibility: within-representation versus cross-representation agreement and their gap, per representation pair | `robustness/split_half/run_matched.py` |
| `split_half_by_context.tsv` | The same, by context | same |
| `split_half_self_vs_cross.tsv` | Classification of self versus cross agreement for each pair and metric | `robustness/split_half/summarize_self_cross.py` |
| `min10_cells_retention.csv` | Response units retained when units with fewer than 10 target cells are excluded (132,840 of 144,368; 92.0%) | `robustness/min_cell_count/run_local_sensitivity.py` |
| `min10_cells_sensitivity.csv` | Original versus ≥10-cell value for every metric, block and representation pair, with neighborhoods and communities rebuilt on the retained units | `robustness/min_cell_count/summarize.py` |
| `coherence_bootstrap_ci.csv` | 95% intervals for coherence-rank comparisons from 1,000 paired target-cluster bootstrap replicates, conditional on the observed donors | `robustness/coherence_bootstrap/bootstrap_sampled_cell.py`, `robustness/min_cell_count/summarize.py` |
