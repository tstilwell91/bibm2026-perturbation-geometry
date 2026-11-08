from pathlib import Path
import pandas as pd,numpy as np,json
O=Path('results/camera_ready_robustness_20261108')
local=pd.read_csv(O/'local_graph/metrics_by_block.csv').rename(columns={'n_retained':'n_ge10'})
assert len(local)==288
parts=[local]
for p in sorted((O/'global').glob('rho_D*.csv')):parts.append(pd.read_csv(p).rename(columns={'filtered_ge10':'filtered','change':'difference'}))
assert len(list((O/'global').glob('rho_D*.csv')))==12
parts.append(pd.read_csv(O/'global/sampled_cell_coherence_rank_sensitivity.csv').rename(columns={'filtered_ge10':'filtered','change':'difference','n_original':'n_sampled_cells_original','n_ge10':'n_sampled_cells_ge10'}))
d=pd.concat(parts,ignore_index=True); assert len(d)==432
assert not d.duplicated(['metric','donor','context','representation_a','representation_b']).any()
assert np.isfinite(d[['original','filtered']].to_numpy()).all()
unit=pd.read_csv(O/'local_graph/unit_mask.csv.gz');counts=unit.groupby(['donor','context']).agg(n_original=('retained','size'),n_ge10=('retained','sum')).reset_index()
d=d.drop(columns=['n_original','n_ge10'],errors='ignore').merge(counts,on=['donor','context'],validate='many_to_one');d['scope']='donor_context';d['aggregation']='original block statistic'
keys=['metric','representation_a','representation_b'];summ=[]
for (m,a,b),g in d.groupby(keys):
 fun=np.median if m=='sampled_cell_LOO_coherence_rank_spearman' else np.mean
 for c,h in [('ALL',g)]+[(c,h) for c,h in g.groupby('context')]:
  summ.append(dict(metric=m,representation_a=a,representation_b=b,donor='ALL',context=c,scope='overall' if c=='ALL' else 'context',aggregation='median of donor-context statistics' if fun==np.median else 'arithmetic mean of donor-context statistics',original=float(fun(h.original)),filtered=float(fun(h.filtered))))
s=pd.DataFrame(summ);s['difference']=s.filtered-s.original;s['absolute_change']=s.difference.abs();allrows=pd.concat([d,s],ignore_index=True)
rankkeys=['metric','scope','donor','context'];allrows['pair_rank_original']=allrows.groupby(rankkeys).original.rank(method='min',ascending=False);allrows['pair_rank_filtered']=allrows.groupby(rankkeys).filtered.rank(method='min',ascending=False)
allrows['qualitative_conclusion']=allrows.apply(lambda r:('positive agreement retained' if r.original>0 and r.filtered>0 else 'agreement sign changed')+('; representation-pair rank retained' if r.pair_rank_original==r.pair_rank_filtered else f'; representation-pair rank changed {int(r.pair_rank_original)} to {int(r.pair_rank_filtered)}'),axis=1)
allrows.to_csv(O/'sensitivity_ge10.csv',index=False)
# counts unique biological response units, never sixfold representation-pair records
r=[]
for scope,cols in [('overall',[]),('donor',['donor']),('context',['context']),('donor_context',['donor','context']),('target',['target'])]:
 groups=[('ALL',unit)] if not cols else unit.groupby(cols)
 for label,h in groups:
  if not isinstance(label,tuple):label=(label,)
  row=dict(scope=scope,n_original=len(h),n_retained=int(h.retained.sum()),n_excluded=int((~h.retained).sum()),retained_pct=100*h.retained.mean(),target_cells_original=int(h.n_target_cells.sum()),target_cells_retained=int(h.loc[h.retained,'n_target_cells'].sum()))
  row.update(dict(zip(cols,label)));r.append(row)
pd.DataFrame(r).to_csv(O/'retention_by_group.csv',index=False)
# Only primary paper coherence statistic belongs in requested top-level CI file.
bp=O/'bootstrap/sampled_cell/bootstrap_ci.csv'
ci=pd.read_csv(bp)
ci.to_csv(O/'bootstrap_ci.csv',index=False)
print(s[['metric','context','representation_a','representation_b','original','filtered','absolute_change']].to_string(index=False))
qc={}
qc['response_unit_count']=len(unit);qc['retained_units']=int(unit.retained.sum());qc['unique_response_units']=not unit.duplicated(['target','donor','context']).any();qc['primary_block_metric_rows']=len(d);qc['all_metric_estimates_finite']=bool(np.isfinite(d[['original','filtered']].to_numpy()).all())
v=pd.read_csv(O/'local_graph/filtered_neighbor_validation.csv');qc['filtered_exact_queries']=len(v);qc['filtered_exact_query_pass']=bool(len(v)==144 and v.passed.all());qc['max_filtered_cosine_error']=float(v.max_cosine_error.max())
q=pd.read_csv(O/'local_graph/original_qc.csv');qc['original_O15_comparisons']=len(q);qc['max_original_O15_error']=float(q.abs_error.max())
qc['max_original_rho_error']=float(d[d.metric=='whole_map_spearman_rho'].original_reproduction_error.max())
qc['max_original_cell_coherence_error']=float(d[d.metric=='sampled_cell_LOO_coherence_rank_spearman'].original_reproduction_error.max())
qc['bootstrap']=json.loads((O/'bootstrap/sampled_cell/bootstrap_audit.json').read_text())
qc['pass']=bool(qc['unique_response_units'] and qc['all_metric_estimates_finite'] and qc['filtered_exact_query_pass'] and qc['max_original_rho_error']<1e-12 and qc['max_original_cell_coherence_error']<1e-12 and qc['max_original_O15_error']<1e-12 and len(q)==72 and len(ci)==24 and (ci.n_valid_replicates==1000).all() and qc['bootstrap']['status']=='PASS')
(O/'FINAL_QC.json').write_text(json.dumps(qc,indent=2));assert qc['pass']
