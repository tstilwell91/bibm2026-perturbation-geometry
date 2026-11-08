import sys, time, json, importlib.util, itertools, platform
from pathlib import Path
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
ROOT=Path('source/derived/downstream'); OUT=Path('results/camera_ready_robustness_20261108/local_graph'); REPS=['pca','scvi','geneformer','state']
def loadmod(name,path):
 s=importlib.util.spec_from_file_location(name,path); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
CODE=Path(__file__).resolve().parents[2]/'analysis'
lg=loadmod('local',CODE/'local_geometry/run_local_geometry.py'); gg=loadmod('graph',CODE/'graph_geometry/run_graph_geometry.py')
md={r:pd.read_csv(ROOT/'response_construction_v1'/r/'response_units.tsv.gz',sep='\t') for r in REPS}; key=['target','donor','context']; base=md['pca']
assert len(base)==144368 and not base.duplicated(key).any()
for r in REPS:
 assert base[key+['n_target_cells','contributing_cell_ids_sha256']].equals(md[r][key+['n_target_cells','contributing_cell_ids_sha256']])
mask=base.n_target_cells>=10
base[key+['n_target_cells']].assign(retained=mask).to_csv(OUT/'unit_mask.csv.gz',index=False)
counts=base.assign(retained=mask).groupby(['donor','context']).retained.agg(['size','sum']).rename(columns={'size':'original','sum':'retained'}); counts['excluded']=counts.original-counts.retained; counts['retained_pct']=100*counts.retained/counts.original; counts.to_csv(OUT/'retention.csv')
orig_o=pd.read_csv(ROOT/'response_local_geometry_v1/cross_representation/agreement_by_donor_context.tsv.gz',sep='\t').query('k==15'); orig_e=pd.read_csv(ROOT/'response_graph_geometry_v1/cross_representation/edge_agreement.tsv.gz',sep='\t').query('k==15'); orig_c=pd.read_csv(ROOT/'response_graph_geometry_v1/cross_representation/community_agreement.tsv.gz',sep='\t').query('k==15')
rows=[]; qc=[]; audit=[]
for donor in lg.DONORS:
 for context in lg.CONTEXTS:
  tag=f'{donor}_{context}'; nodes=sorted(base.loc[mask & base.donor.eq(donor)&base.context.eq(context),'target']); allowed=set(nodes); neigh={}; edges={}; labels={}
  for rep in REPS:
   cache=OUT/f'{rep}_{tag}_top15.parquet'
   f=pd.read_parquet(ROOT/'response_local_geometry_v1'/rep/'knn_top30.parquet',filters=[('donor','==',donor),('context','==',context)]).sort_values(['target','rank'])
   assert set(f.target.unique())==set(base.loc[base.donor.eq(donor)&base.context.eq(context),'target'])
   # Verify original O15 using cached full-cohort directed neighborhoods.
   fullneigh={t:set(g.neighbor.iloc[:15]) for t,g in f.groupby('target',sort=False)}
   if rep==REPS[0]: full={}
   full[rep]=fullneigh
   if cache.exists(): filt=pd.read_parquet(cache); fallback=-1
   else:
    cand=f[f.target.isin(allowed)&f.neighbor.isin(allowed)].copy(); cand['rank']=cand.groupby('target',sort=False).cumcount()+1
    good=set(cand.groupby('target').size().loc[lambda x:x>=15].index); fallback=len(nodes)-len(good)
    filt=cand[cand.target.isin(good)&cand['rank'].le(15)].copy()
    if fallback:
     ix=np.flatnonzero(mask.to_numpy()&base.donor.eq(donor).to_numpy()&base.context.eq(context).to_numpy()); order=np.argsort(base.iloc[ix].target.to_numpy()); ix=ix[order]
     mat=np.asarray(np.load(ROOT/'response_construction_v1'/rep/'response_vectors.npy',mmap_mode='r')[ix],dtype=np.float64); mat/=np.linalg.norm(mat,axis=1)[:,None]
     extra=[]
     for i,t in enumerate(nodes):
      if t in good: continue
      s=mat[i]@mat.T; s[i]=-np.inf; choice=np.lexsort((np.asarray(nodes),-s))[:15]
      extra.extend({'representation':lg.REP_NAMES[rep],'donor':donor,'context':context,'target':t,'neighbor':nodes[j],'rank':rank,'cosine':np.clip(s[j],-1,1)} for rank,j in enumerate(choice,1))
     filt=pd.concat([filt,pd.DataFrame(extra)],ignore_index=True)
    filt=filt.sort_values(['target','rank']); filt.to_parquet(cache,index=False)
   assert len(filt)==15*len(nodes) and not filt.duplicated(['target','neighbor']).any() and set(filt.target)==allowed and set(filt.neighbor)<=allowed
   neigh[rep]={t:set(g.neighbor) for t,g in filt.groupby('target',sort=False)}
   e=gg.canonical_union(filt,15); e['community_weight']=e.raw_cosine; assert (e.raw_cosine>=0).all()
   edges[rep]=set(zip(e.source,e.target)); G=nx.Graph(); G.add_nodes_from(nodes); G.add_edges_from((r.source,r.target,{'community_weight':float(r.community_weight)}) for r in e.itertuples())
   labpath=OUT/f'{rep}_{tag}_communities.csv'
   if labpath.exists(): lab=pd.read_csv(labpath); labels[rep]=dict(zip(lab.target,lab.community_id))
   else:
    labels[rep],sizes,mod=gg.community_partition(G); pd.DataFrame({'target':nodes,'community_id':[labels[rep][x] for x in nodes]}).to_csv(labpath,index=False)
   audit.append({'rep':rep,'donor':donor,'context':context,'nodes':len(nodes),'edges':len(e),'fallback_queries':fallback,'min_cosine':float(e.raw_cosine.min())})
   print(time.strftime('%H:%M:%S'),rep,tag,len(nodes),'fallback',fallback,flush=True)
  for a,b in itertools.combinations(REPS,2):
   sel=dict(donor=donor,context=context,representation_a=lg.REP_NAMES[a],representation_b=lg.REP_NAMES[b])
   def lookup(df,col):
    z=df
    for k,v in sel.items(): z=z[z[k]==v]
    assert len(z)==1
    return float(z.iloc[0][col])
   oldo=lookup(orig_o,'mean_overlap_fraction'); reproduced=np.mean([len(full[a][t]&full[b][t])/15 for t in full[a]]); assert abs(oldo-reproduced)<1e-12
   qc.append({**sel,'metric':'O15','original':oldo,'reproduced':reproduced,'abs_error':abs(oldo-reproduced)})
   vals={'O15':(oldo,np.mean([len(neigh[a][t]&neigh[b][t])/15 for t in nodes])),'edge_jaccard':(lookup(orig_e,'edge_jaccard'),len(edges[a]&edges[b])/len(edges[a]|edges[b])),'community_ARI':(lookup(orig_c,'adjusted_rand_index'),adjusted_rand_score([labels[a][t] for t in nodes],[labels[b][t] for t in nodes])),'community_NMI':(lookup(orig_c,'normalized_mutual_information'),normalized_mutual_info_score([labels[a][t] for t in nodes],[labels[b][t] for t in nodes]))}
   for metric,(old,new) in vals.items(): rows.append({**sel,'metric':metric,'original':old,'filtered':new,'difference':new-old,'absolute_change':abs(new-old),'n_retained':len(nodes)})
  pd.DataFrame(rows).to_csv(OUT/'metrics_by_block.csv',index=False); pd.DataFrame(qc).to_csv(OUT/'original_qc.csv',index=False); pd.DataFrame(audit).to_csv(OUT/'graph_audit.csv',index=False)
(OUT/'runtime.json').write_text(json.dumps({'python':sys.version,'numpy':np.__version__,'pandas':pd.__version__,'networkx':nx.__version__,'seed':42,'resolution':1,'k':15,'weight':'raw_cosine','retained':int(mask.sum()),'total':len(base),'status':'complete'},indent=2))
