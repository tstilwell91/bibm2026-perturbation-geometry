#!/usr/bin/env python3
"""Additive biological consensus analysis on frozen V31 results.

Full-map positive recovery is imported from the frozen known-biology analysis.
The novel consensus analysis is explicitly restricted to the union of frozen
k=15 graph edges, since it does not recompute the all-pairs cosine universe.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pyarrow.compute as pc

ROOT=Path('.').resolve()
KB=ROOT/'source/derived/downstream/response_prebiology_v1/known_biology'
GRAPH=ROOT/'source/derived/downstream/response_graph_geometry_v1/graphs/edges.parquet'
OUT=ROOT/'results/bibm_extensions_20261005/agent_b_biological_utility'
REPS=['PCA','scVI','Geneformer','STATE']
RESOURCES=['HuMAP3','STRING','Reactome','SIGNOR']
SEED=20261005

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    stab=pq.read_table(KB/'known_pair_representation_stability.parquet').to_pandas()
    # Annotation definition comes from resource-specific frozen positive pairs.
    ann={r:set(map(tuple,stab.loc[stab.resource.eq(r),['target_a','target_b']].drop_duplicates().to_numpy())) for r in RESOURCES}
    # Read the already materialized graph; retain the primary k=15 edge definition.
    edge=pq.read_table(GRAPH,columns=['representation','donor','context','k','source','target','raw_cosine'])
    edge=edge.filter(pc.equal(edge['k'],15)).to_pandas()
    edge['target_a']=np.where(edge.source<edge.target,edge.source,edge.target)
    edge['target_b']=np.where(edge.source<edge.target,edge.target,edge.source)
    # One undirected score per representation/block/pair. Directed ranks can
    # differ, but the underlying cosine is symmetric; average removes repeats.
    edge=edge.groupby(['representation','donor','context','target_a','target_b'],as_index=False).raw_cosine.mean()
    wide=edge.pivot_table(index=['donor','context','target_a','target_b'],columns='representation',values='raw_cosine').reset_index()
    wide.columns.name=None
    for r in REPS:
        if r not in wide: wide[r]=np.nan
    scorecols=wide[REPS]
    wide['observed_representations']=scorecols.notna().sum(axis=1)
    wide['mean_cosine_observed']=scorecols.mean(axis=1,skipna=True)
    wide['cosine_sd_observed']=scorecols.std(axis=1,skipna=True,ddof=0)
    wide['cosine_range_observed']=scorecols.max(axis=1)-scorecols.min(axis=1)
    wide['representation_recurrence']=wide.observed_representations/len(REPS)
    # Prespecified composite: product of average cosine percentile (within
    # block) and recurrence. This rewards both similarity and replication.
    wide['mean_cosine_pct']=wide.groupby(['donor','context']).mean_cosine_observed.rank(pct=True)
    wide['similarity_stability_score']=wide.mean_cosine_pct*wide.representation_recurrence
    for r in RESOURCES:
        pairs=ann[r]
        wide[f'{r}_positive']=[(a,b) in pairs for a,b in zip(wide.target_a,wide.target_b)]
    wide['any_resource_positive']=wide[[f'{r}_positive' for r in RESOURCES]].any(axis=1)
    # pair-level output preserves exact block/key and all frozen resource labels
    wide.to_parquet(OUT/'k15_graph_candidate_biology.parquet',index=False)

    rows=[]
    thresholds=[.001,.005,.01,.02,.05]
    score_methods={'similarity':'mean_cosine_observed','consensus':'representation_recurrence','similarity_plus_stability':'similarity_stability_score'}
    for (donor,context),g in wide.groupby(['donor','context'],sort=True):
        for method,sc in score_methods.items():
            g=g.sort_values(sc,ascending=False,kind='mergesort')
            for frac in thresholds:
                n=max(1,int(np.ceil(frac*len(g))))
                selected=g.iloc[:n]
                for res in RESOURCES+['any_resource']:
                    col=f'{res}_positive'
                    positives=int(selected[col].sum()); total=int(g[col].sum())
                    prevalence=total/len(g) if len(g) else np.nan
                    rows.append(dict(donor=donor,context=context,method=method,threshold=frac,
                        candidate_pairs=len(g),selected_pairs=n,resource=res,annotated_selected=positives,
                        annotated_candidate_total=total,selected_annotation_fraction=positives/n,
                        candidate_annotation_fraction=prevalence,
                        enrichment=(positives/n/prevalence if prevalence else np.nan),
                        recall=(positives/total if total else np.nan)))
    tail=pd.DataFrame(rows)
    tail.to_csv(OUT/'k15_enrichment_by_block.tsv.gz',sep='\t',index=False,compression='gzip')
    # Descriptive donor/context-block summaries. Inference unit is the 12
    # donor-context blocks; graph pairs are never treated as replicates.
    summary=tail.groupby(['method','threshold','resource']).agg(
        blocks=('enrichment','count'),mean_enrichment=('enrichment','mean'),
        median_enrichment=('enrichment','median'),mean_recall=('recall','mean'),
        mean_selected_fraction=('selected_annotation_fraction','mean')).reset_index()
    summary.to_csv(OUT/'k15_enrichment_summary.tsv',sep='\t',index=False)
    # Strength-matched comparison: within donor/context, sample one unlabeled
    # graph candidate per annotated positive from the same mean-cosine decile.
    # Unknown labels remain unknown; this estimates annotation association only.
    rng=np.random.default_rng(SEED); match_rows=[]
    for (donor,context),g0 in wide.groupby(['donor','context'],sort=True):
      g=g0.copy(); g['strength_decile']=pd.qcut(g.mean_cosine_pct,10,labels=False,duplicates='drop')
      for res in RESOURCES:
        pos=g[g[f'{res}_positive']]
        neg=g[~g[f'{res}_positive']]
        if not len(pos) or not len(neg): continue
        # Individual positive can be positive in another resource; for this
        # resource only its own annotation is tested.
        pools={d:x for d,x in neg.groupby('strength_decile')}
        controls=[]
        for row in pos.itertuples():
          pool=pools.get(row.strength_decile)
          if pool is None or pool.empty:
            pool=neg
          controls.append(pool.iloc[int(rng.integers(len(pool)))])
        ctl=pd.DataFrame(controls)
        match_rows.append(dict(donor=donor,context=context,resource=res,n_positive=len(pos),
           n_matched_controls=len(ctl),positive_mean_cosine=pos.mean_cosine_observed.mean(),
           matched_control_mean_cosine=ctl.mean_cosine_observed.mean(),
           positive_recurrence=pos.representation_recurrence.mean(),
           matched_control_recurrence=ctl.representation_recurrence.mean(),
           positive_cosine_sd=pos.cosine_sd_observed.mean(),
           matched_control_cosine_sd=ctl.cosine_sd_observed.mean(),
           positive_similarity_stability=pos.similarity_stability_score.mean(),
           matched_control_similarity_stability=ctl.similarity_stability_score.mean()))
    matched=pd.DataFrame(match_rows)
    matched.to_csv(OUT/'strength_matched_annotation_comparison.tsv',sep='\t',index=False)
    boot_rng=np.random.default_rng(SEED+1); uncertainty=[]
    for res,g in matched.groupby('resource'):
      for metric,pp,cc in [
        ('representation_recurrence','positive_recurrence','matched_control_recurrence'),
        ('cosine_sd_observed','positive_cosine_sd','matched_control_cosine_sd'),
        ('similarity_stability_score','positive_similarity_stability','matched_control_similarity_stability')]:
        d=(g[pp]-g[cc]).dropna().to_numpy(); n=len(d)
        if n:
          bs=np.array([boot_rng.choice(d,size=n,replace=True).mean() for _ in range(10000)])
          uncertainty.append(dict(resource=res,metric=metric,n_blocks=n,mean_paired_difference=d.mean(),
             median_paired_difference=float(np.median(d)),ci95_low=float(np.quantile(bs,.025)),
             ci95_high=float(np.quantile(bs,.975)),blocks_positive=int((d>0).sum()),
             blocks_negative=int((d<0).sum()),blocks_tied=int((d==0).sum())))
    pd.DataFrame(uncertainty).to_csv(OUT/'strength_matched_paired_block_uncertainty.tsv',sep='\t',index=False)
    # Frozen full-map recovery summaries already evaluated whole-map top tails.
    # Preserve them as imported results and explicitly flag unavailable added
    # thresholds rather than treating graph-candidate thresholds as equivalent.
    recovery=pd.read_csv(KB/'recovery_by_donor_context.tsv.gz',sep='\t')
    recovery.to_csv(OUT/'frozen_full_map_recovery.tsv.gz',sep='\t',index=False,compression='gzip')
    ctx=pd.read_csv(KB/'recovery_context_summary.tsv.gz',sep='\t')
    ctx.to_csv(OUT/'frozen_full_map_recovery_summary.tsv.gz',sep='\t',index=False,compression='gzip')
    # Extend whole-population top-tail results to all requested fractions using
    # the exact stored positive-pair percentiles and whole-map denominators.
    # Midrank ties can straddle a cutoff; these estimates use pct >= 1-f.
    thresholds=[.001,.005,.01,.02,.05]
    full=[]
    for rep in REPS:
      for (donor,context),g in stab.groupby(['donor','context'],sort=True):
        base=recovery[(recovery.representation==rep)&(recovery.donor==donor)&(recovery.context==context)]
        if base.empty: continue
        n_targets=int(base.n_targets.iloc[0]); n_all=n_targets*(n_targets-1)//2
        for resource,h in g.groupby('resource'):
          n_pos=int(base.loc[base.resource.eq(resource),'n_eligible_known_positive_pairs'].iloc[0])
          for frac in thresholds:
            recovered=int((h[rep] >= 1-frac).sum())
            full.append(dict(representation=rep,donor=donor,context=context,resource=resource,
              threshold=frac,n_targets=n_targets,all_pairs=n_all,known_positive_pairs=n_pos,
              recovered_known_positives=recovered,recall=recovered/n_pos if n_pos else np.nan,
              enrichment=(recovered/(frac*n_all))/(n_pos/n_all) if n_pos else np.nan,
              annotation_precision=recovered/(frac*n_all)))
    pd.DataFrame(full).to_csv(OUT/'full_map_enrichment_5_thresholds.tsv.gz',sep='\t',index=False,compression='gzip')
    # Geneformer contrast: known-positive recovery versus its all-pairs rank
    # agreement is taken from current frozen geometry summary reports.
    ranks=pd.read_csv(ROOT/'source/derived/downstream/response_geometry_v1/cross_representation/agreement_by_donor_context.tsv',sep='\t')
    ranks.to_csv(OUT/'frozen_cross_representation_agreement.tsv',sep='\t',index=False)
    print('candidate rows',len(wide),'positive prevalence',wide.any_resource_positive.mean())
    print(summary.to_string(index=False))
    print('matched rows',len(matched))

if __name__=='__main__': main()
