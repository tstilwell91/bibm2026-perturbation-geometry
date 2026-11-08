from pathlib import Path
import numpy as np,pandas as pd
R=Path('source/derived/downstream/response_construction_v1'); O=Path('results/camera_ready_robustness_20261108/local_graph'); md=pd.read_csv(R/'pca/response_units.tsv.gz',sep='\t'); rows=[]; rng=np.random.default_rng(20261108)
for d in ['D1','D2','D3','D4']:
 for c in ['Rest','Stim8hr','Stim48hr']:
  ix=np.flatnonzero((md.n_target_cells>=10)&md.donor.eq(d)&md.context.eq(c)); ix=ix[np.argsort(md.iloc[ix].target.to_numpy())]; names=md.iloc[ix].target.to_numpy(); sampled=rng.choice(len(ix),3,replace=False)
  for rep in ['pca','scvi','geneformer','state']:
   path=O/f'{rep}_{d}_{c}_top15.parquet'
   if not path.exists(): continue
   f=pd.read_parquet(path); x=np.asarray(np.load(R/rep/'response_vectors.npy',mmap_mode='r')[ix]); x/=np.linalg.norm(x,axis=1)[:,None]
   for i in sampled:
    s=x[i]@x.T;s[i]=-np.inf; chosen=np.lexsort((names,-s))[:15]; actual=f[f.target.eq(names[i])].sort_values('rank'); same=actual.neighbor.tolist()==names[chosen].tolist(); error=float(np.max(np.abs(actual.cosine.to_numpy()-np.clip(s[chosen],-1,1)))); rows.append(dict(donor=d,context=c,representation=rep,target=names[i],same_neighbor_ids=same,max_cosine_error=error,passed=same and error<1e-10))
   del x
pd.DataFrame(rows).to_csv(O/'filtered_neighbor_validation.csv',index=False); assert all(r['passed'] for r in rows); print('PASS',len(rows),'exact full-universe filtered-query checks')
