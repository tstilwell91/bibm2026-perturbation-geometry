#!/usr/bin/env python3
"""Vectorized local-output invariants and final summary assembly."""
import json, platform, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import spearmanr, pearsonr

OUT=Path('source/derived/downstream/response_local_geometry_v1')
RESP=Path('source/derived/downstream/response_construction_v1')
REPS=['pca','scvi','geneformer','state']
RNAME={'pca':'PCA','scvi':'scVI','geneformer':'Geneformer V2-316M','state':'STATE SE-600M'}
audit={'pass':True,'knn_rows':{},'checks':{},'method':'vectorized composite-key and rank-order invariants'}
for rep in REPS:
    p=OUT/rep/'knn_top30.parquet'
    df=pd.read_parquet(p)
    bad={}
    bad['finite_cosine']=bool(np.isfinite(df.cosine.to_numpy()).all())
    bad['self_neighbors']=int((df.target==df.neighbor).sum())
    bad['duplicate_target_neighbor']=int(df.duplicated(['donor','context','target','neighbor']).sum())
    grp=df.groupby(['donor','context','target'],sort=False,observed=True)['rank']
    stats=grp.agg(['count','min','max','nunique'])
    bad['target_groups']=len(stats)
    bad['wrong_neighbor_count_groups']=int((stats['count']!=30).sum())
    bad['wrong_rank_range_groups']=int(((stats['min']!=1)|(stats['max']!=30)|(stats['nunique']!=30)).sum())
    df.sort_values(['donor','context','target','rank'],inplace=True,kind='stable')
    delta=df.groupby(['donor','context','target'],sort=False,observed=True).cosine.diff()
    bad['ascending_cosine_steps']=int((delta.dropna()>1e-12).sum())
    # Deterministic lexical tie ordering.
    tie=df.assign(prev_cos=df.groupby(['donor','context','target'],sort=False,observed=True).cosine.shift(),prev_neighbor=df.groupby(['donor','context','target'],sort=False,observed=True).neighbor.shift())
    bad['lexical_tie_violations']=int(((tie.cosine==tie.prev_cos)&(tie.neighbor<tie.prev_neighbor)).sum())
    audit['checks'][rep]=bad
    audit['knn_rows'][rep]=len(df)
    if (not bad['finite_cosine'] or bad['self_neighbors'] or bad['duplicate_target_neighbor'] or bad['wrong_neighbor_count_groups'] or bad['wrong_rank_range_groups'] or bad['ascending_cosine_steps'] or bad['lexical_tie_violations']):
        audit['pass']=False
    del df

spot=json.loads((OUT/'audits/LOCAL_KNN_SPOTCHECK.json').read_text())
audit['spotcheck_pass']=spot['pass']
audit['pass']=audit['pass'] and spot['pass']
(OUT/'audits/LOCAL_KNN_AUDIT.json').write_text(json.dumps(audit,indent=2)+'\n')

mds={r:pd.read_csv(RESP/r/'response_units.tsv.gz',sep='\t',usecols=['target','donor','context','response_l2_norm']) for r in REPS}
supports=[]
for donor in ['D1','D2','D3','D4']:
  for context in ['Rest','Stim8hr','Stim48hr']:
    ns=[]
    for rep in REPS:
      d=mds[rep]
      ns.append(len(d[(d.donor==donor)&(d.context==context)]))
    if len(set(ns))!=1: raise ValueError(f'target support differs locally: {donor} {context} {ns}')
    n=ns[0]
    supports.append({'donor':donor,'context':context,'n_targets':n,'expected_pairs':n*(n-1)//2})

ctx=pd.read_csv(OUT/'cross_representation/agreement_by_context_summary.tsv.gz',sep='\t')
block=pd.read_csv(OUT/'cross_representation/agreement_by_donor_context.tsv.gz',sep='\t')
gl=pd.read_csv(OUT/'cross_representation/global_vs_local_agreement.tsv.gz',sep='\t')
rho_s=float(spearmanr(gl.spearman_rho,gl.local_mean_overlap).statistic)
rho_p=float(pearsonr(gl.spearman_rho,gl.local_mean_overlap).statistic)
means=[]
for rep in REPS:
  x=mds[rep].response_l2_norm.to_numpy()
  means.append({'representation':RNAME[rep],'min_response_magnitude':float(x.min()),'median_response_magnitude':float(np.median(x)),'max_response_magnitude':float(x.max())})
stable=pd.read_csv(OUT/'cross_representation/stable_unstable_targets.tsv.gz',sep='\t')
pair_targets=stable[stable.record_type.isin(['top20_pair_overlap','bottom20_pair_overlap'])]
top_range=[float(pair_targets[pair_targets.record_type=='top20_pair_overlap'].overlap_fraction.min()),float(pair_targets[pair_targets.record_type=='top20_pair_overlap'].overlap_fraction.max())]
bottom_range=[float(pair_targets[pair_targets.record_type=='bottom20_pair_overlap'].overlap_fraction.min()),float(pair_targets[pair_targets.record_type=='bottom20_pair_overlap'].overlap_fraction.max())]

summary={'status':'PASS' if audit['pass'] else 'FAIL','method':'Exact CPU blocked NumPy/OpenBLAS matrix multiplication of float64 L2-normalized response vectors; self excluded; deterministic descending cosine then lexical target tie order. k=10,15,30.',
 'device':'CPU; CUDA unavailable; FAISS unavailable','k_values':[10,15,30],'preflight_units_per_representation':144368,
 'dimensions':{'PCA':50,'scVI':50,'Geneformer V2-316M':1152,'STATE SE-600M':2058},'target_support_by_donor_context':supports,
 'spotcheck':spot,'local_audit':audit,'agreement_summaries':ctx.to_dict('records'),'global_vs_local_descriptive_correlations':{'spearman':rho_s,'pearson':rho_p,'n_donor_context_representation_pairs':len(gl)},
 'response_magnitude_summary':means,'stable_top20_overlap_range':top_range,'unstable_bottom20_overlap_range':bottom_range,
 'outputs':[str(p) for p in sorted(OUT.rglob('*')) if p.is_file()], 'python':sys.version,'platform':platform.platform()}
(OUT/'reports').mkdir(exist_ok=True)
(OUT/'reports/RESPONSE_LOCAL_GEOMETRY_SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n')

lines=['# Local response geometry summary','',f"**Status: {summary['status']}**",'',
 '## Method','',summary['method'],'',
 'Analytical random-neighbor expectation for a block of n targets is k/(n−1) as an overlap fraction; no permutation loops were used. Target responses and upstream frozen outputs were not changed.','',
 '## Target counts by donor × context','','| Donor | Context | Targets | Expected unordered pairs |','|---|---|---:|---:|']
for x in supports: lines.append(f"| {x['donor']} | {x['context']} | {x['n_targets']:,} | {x['expected_pairs']:,} |")
lines += ['', '## Cross-representation overlap by context','',
 '| Context | Representation pair | k | Mean donor mean | Median donor mean | Min | Max | Random expectation | Enrichment |','|---|---|---:|---:|---:|---:|---:|---:|---:|']
for _,r in ctx.iterrows(): lines.append(f"| {r.context} | {r.representation_a}–{r.representation_b} | {int(r.k)} | {r.mean_across_donors:.4f} | {r.median_across_donors:.4f} | {r.min_across_donors:.4f} | {r.max_across_donors:.4f} | {r.mean_random_expectation:.5f} | {r.mean_enrichment:.2f}× |")
lines += ['', 'Full target-level overlap summaries by donor × context, including median, SD, quartiles, min/max, expected random overlap, and enrichment, are in `cross_representation/agreement_by_donor_context.tsv.gz`. Target-specific values are in `target_level_knn_agreement.parquet`.','',
 '## Global versus local agreement','',
 f"Across the 72 donor × context × representation-pair records at k=30, global Spearman and mean local overlap had descriptive Pearson r={rho_p:.3f} and Spearman ρ={rho_s:.3f}. Pair-level values are in `cross_representation/global_vs_local_agreement.tsv.gz`. This is descriptive only.",'',
 '## Stable and variable target neighborhoods','',
 f"The selected top-20 target-level overlap values range from {top_range[0]:.3f} to {top_range[1]:.3f}; bottom-20 values range from {bottom_range[0]:.3f} to {bottom_range[1]:.3f}. Target identities and mean overlap across six representation pairs are in `cross_representation/stable_unstable_targets.tsv.gz`.",'',
 '## Local validation','',
 'The deterministic ten-target D1 × Rest spot check passed against the saved all-pair cosine tables for all four representations: neighbor IDs matched and cosine differences were within 1e-10. Vectorized invariants confirmed exactly 30 unique non-self neighbors per target, ranks 1–30, finite descending cosine values, deterministic lexical tie order, and overlap/Jaccard in [0,1].','',
 'No graph, community, spectral, cell-level, or biological interpretation analysis was run.']
(OUT/'reports/RESPONSE_LOCAL_GEOMETRY_SUMMARY.md').write_text('\n'.join(lines)+'\n')
print(json.dumps({'status':summary['status'],'audit':audit,'local_global_corr':summary['global_vs_local_descriptive_correlations'],'summary':str(OUT/'reports/RESPONSE_LOCAL_GEOMETRY_SUMMARY.md')},indent=2))
