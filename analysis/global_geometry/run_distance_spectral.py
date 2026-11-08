#!/usr/bin/env python3
"""Exact target distance-rank preservation and response spectral summaries."""
from pathlib import Path
import json, time
import numpy as np
import pandas as pd
from scipy.stats import rankdata, pearsonr, spearmanr

ROOT=Path('source/derived/downstream')
RESP=ROOT/'response_construction_v1'
OUT=ROOT/'response_prebiology_v1'
REPS=['pca','scvi','geneformer','state']
NAMES={'pca':'PCA','scvi':'scVI','geneformer':'Geneformer V2-316M','state':'STATE SE-600M'}
PAIRS=[('pca','scvi'),('pca','geneformer'),('pca','state'),('scvi','geneformer'),('scvi','state'),('geneformer','state')]
DONORS=['D1','D2','D3','D4']; CONTEXTS=['Rest','Stim8hr','Stim48hr']

def load_rep(rep):
    meta=pd.read_csv(RESP/rep/'response_units.tsv.gz',sep='\t')
    arr=np.load(RESP/rep/'response_vectors.npy',mmap_mode='r')
    assert len(meta)==arr.shape[0]==144368
    assert {'target','donor','context'} <= set(meta.columns)
    return meta,arr

def rank_corr_rows(a,b):
    # a,b are target-by-all-other-target cosine matrices with self already removed.
    ra=rankdata(a,axis=1,method='average'); rb=rankdata(b,axis=1,method='average')
    ra-=ra.mean(axis=1,keepdims=True); rb-=rb.mean(axis=1,keepdims=True)
    den=np.sqrt((ra*ra).sum(axis=1)*(rb*rb).sum(axis=1))
    return (ra*rb).sum(axis=1)/den

def main():
    t0=time.time(); metas={}; arrays={}
    for r in REPS: metas[r],arrays[r]=load_rep(r)
    # Cheap metadata alignment check only.
    base=metas['pca'][['target','donor','context']]
    for r in REPS[1:]:
        assert base.equals(metas[r][['target','donor','context']])
    dist_rows=[]; spec_rows=[]; spectrum_rows=[]; spec_pairs=[]
    knn_path=ROOT/'response_local_geometry_v1/cross_representation/target_level_knn_agreement.parquet'
    knn=pd.read_parquet(knn_path)
    print('kNN fields',knn.columns.tolist(),flush=True)
    for donor in DONORS:
      for context in CONTEXTS:
        indices={r:np.flatnonzero((metas[r].donor.to_numpy()==donor)&(metas[r].context.to_numpy()==context)) for r in REPS}
        ids=metas['pca'].iloc[indices['pca']].target.to_numpy()
        order=np.argsort(ids,kind='stable'); ids=ids[order]
        X={}
        for r in REPS:
            ii=indices[r][order]
            assert np.array_equal(metas[r].iloc[ii].target.to_numpy(),ids)
            X[r]=np.asarray(arrays[r][ii],dtype=np.float64)
        n=len(ids); assert n>1
        norms={r:np.linalg.norm(X[r],axis=1) for r in REPS}
        U={r:X[r]/norms[r][:,None] for r in REPS}
        # target-wise exact rank correlations; chunk queries to bound memory.
        rank_values={pair:[] for pair in PAIRS}
        chunksize=128
        for start in range(0,n,chunksize):
            stop=min(n,start+chunksize)
            sims={r:U[r][start:stop]@U[r].T for r in REPS}
            rows=np.arange(stop-start); focal=np.arange(start,stop)
            rankmat={}
            for r in REPS:
                sims[r][rows,focal]=-np.inf
                rankmat[r]=rankdata(sims[r],axis=1,method='average')
            m=n-1
            for a,b in PAIRS:
                ra,rb=rankmat[a],rankmat[b]
                da,db=ra[rows,focal],rb[rows,focal]
                sa=ra.sum(axis=1)-da; sb=rb.sum(axis=1)-db
                saa=(ra*ra).sum(axis=1)-da*da; sbb=(rb*rb).sum(axis=1)-db*db
                sab=(ra*rb).sum(axis=1)-da*db
                cov=sab-sa*sb/m
                va=saa-sa*sa/m; vb=sbb-sb*sb/m
                rank_values[(a,b)].extend((cov/np.sqrt(va*vb)).tolist())
            print(f'DIST {donor} {context} {stop}/{n}',flush=True) if stop==n else None
        for (a,b),vals in rank_values.items():
            for target,val in zip(ids,vals):
                dist_rows.append((donor,context,target,NAMES[a],NAMES[b],n-1,float(val)))
        # Spectral analysis from centered feature covariance. Eigenvalues of Xc'Xc.
        eigs={}
        for r in REPS:
            xc=X[r]-X[r].mean(axis=0,keepdims=True)
            cov=xc.T@xc
            lam=np.linalg.eigvalsh(cov)[::-1]
            lam=np.maximum(lam,0.0)
            total=lam.sum(); assert np.isfinite(lam).all() and total>0
            p=lam/total; cs=np.cumsum(p)
            pc={v:int(np.searchsorted(cs,v)+1) for v in (.8,.9,.95)}
            pr=(total*total)/np.sum(lam*lam)
            nz=p[p>0]; er=float(np.exp(-(nz*np.log(nz)).sum()))
            d=len(lam)
            rec={'representation':NAMES[r],'donor':donor,'context':context,'dimension':d,
              'n_targets':n,'pc1_variance_fraction':float(p[0]),
              'cumvar_pc5':float(cs[min(4,d-1)]),'cumvar_pc10':float(cs[min(9,d-1)]),
              'cumvar_pc20':float(cs[min(19,d-1)]),'cumvar_pc50':float(cs[min(49,d-1)]),
              'pc80':pc[.8],'pc90':pc[.9],'pc95':pc[.95],
              'participation_ratio':float(pr),'participation_ratio_normalized':float(pr/d),
              'entropy_effective_rank':er,'entropy_effective_rank_normalized':er/d,
              'pc95_normalized':pc[.95]/d}
            spec_rows.append(rec); eigs[r]=p
            for idx,v in enumerate(p,1): spectrum_rows.append((NAMES[r],donor,context,d,idx,idx/d,float(v),float(cs[idx-1])))
        # L1 area between cumulative curves on common normalized component-rank grid.
        grid=np.linspace(0,1,1001)
        cdfs={r:np.interp(grid,np.arange(1,len(eigs[r])+1)/len(eigs[r]),np.cumsum(eigs[r]),left=0,right=1) for r in REPS}
        for ai,a in enumerate(REPS):
            for b in REPS[ai+1:]:
                area=float(np.trapezoid(np.abs(cdfs[a]-cdfs[b]),grid))
                spec_pairs.append((donor,context,NAMES[a],NAMES[b],area))
        print(f'SPECTRAL {donor} {context} n={n}',flush=True)
    ddf=pd.DataFrame(dist_rows,columns=['donor','context','target','representation_a','representation_b','n_other_targets','spearman_rho'])
    # Join canonical k=30 overlap target level, selecting same representation pair.
    print('KNN schema/sample',knn.head(2).to_dict('records'),flush=True)
    ddf.to_parquet(OUT/'distance_rank/target_level_distance_rank.parquet',index=False)
    dc=ddf.groupby(['donor','context','representation_a','representation_b']).spearman_rho.agg(['mean','median','std',lambda x:x.quantile(.25),lambda x:x.quantile(.75),'min','max']).reset_index()
    dc.columns=['donor','context','representation_a','representation_b','mean','median','std','q25','q75','min','max']
    dc.to_csv(OUT/'distance_rank/agreement_by_donor_context.tsv.gz',sep='\t',index=False,compression='gzip')
    dctx=dc.groupby(['context','representation_a','representation_b']).agg({c:['mean','median','min','max'] for c in ['mean','median']}).reset_index()
    dctx.to_csv(OUT/'distance_rank/agreement_by_context.tsv.gz',sep='\t',index=False,compression='gzip')
    # Knn fields may encode pair columns differently; normalize known schema.
    aliases={'representation_a':['representation_a','rep_a'],'representation_b':['representation_b','rep_b']}
    kn=knn.copy()
    for dst,cands in aliases.items():
        if dst not in kn:
            src=next((x for x in cands if x in kn),None)
            if src: kn[dst]=kn[src]
    if 'k' in kn: kn=kn[kn.k==30]
    targcol=next((x for x in ['target','focal_target'] if x in kn),None)
    ovcol=next((x for x in ['overlap_fraction','mean_overlap_fraction'] if x in kn),None)
    if targcol and ovcol and {'representation_a','representation_b','donor','context'}<=set(kn):
        kk=kn[['donor','context',targcol,'representation_a','representation_b',ovcol]].rename(columns={targcol:'target',ovcol:'knn_overlap_fraction'})
        # match representation name casing
        def canon(v):
            s=str(v).lower(); return {'pca':'PCA','scvi':'scVI','geneformer':'Geneformer V2-316M','state':'STATE SE-600M'}.get(s,str(v))
        for c in ['representation_a','representation_b']: kk[c]=kk[c].map(canon)
        joined=ddf.merge(kk,on=['donor','context','target','representation_a','representation_b'],how='inner',validate='one_to_one')
        corr=[]
        for (a,b),g in joined.groupby(['representation_a','representation_b']):
            corr.append({'representation_a':a,'representation_b':b,'n_targets':len(g),
              'pearson_r':pearsonr(g.spearman_rho,g.knn_overlap_fraction).statistic,
              'spearman_rho':spearmanr(g.spearman_rho,g.knn_overlap_fraction).statistic})
        joined.to_csv(OUT/'distance_rank/distance_rank_vs_knn.tsv.gz',sep='\t',index=False,compression='gzip')
        pd.DataFrame(corr).to_csv(OUT/'distance_rank/distance_rank_vs_knn_correlations.tsv',sep='\t',index=False)
    sdf=pd.DataFrame(spec_rows)
    sdf.to_csv(OUT/'spectral/spectral_metrics.tsv.gz',sep='\t',index=False,compression='gzip')
    pd.DataFrame(spectrum_rows,columns=['representation','donor','context','dimension','component','normalized_component_rank','variance_fraction','cumulative_variance']).to_parquet(OUT/'spectral/normalized_spectra.parquet',index=False)
    pd.DataFrame(spec_pairs,columns=['donor','context','representation_a','representation_b','cumulative_spectrum_l1_area']).to_csv(OUT/'spectral/spectral_pairwise_distance.tsv.gz',sep='\t',index=False,compression='gzip')
    print('DONE seconds',time.time()-t0,'distance rows',len(ddf),'spec rows',len(sdf),flush=True)

if __name__=='__main__': main()
