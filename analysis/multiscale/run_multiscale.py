#!/usr/bin/env python3
"""Exact extended local geometry and full-map strength-tail overlap.

Reads the frozen V31 response vectors and V31 cosine pair tables only. Writes
all new files to this agent's output directory.
"""
from pathlib import Path
import itertools, json, os, time, sys, platform, hashlib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT=Path('.').resolve()
BASE=ROOT/'source/derived/downstream'
RESP=BASE/'response_construction_v1'
GEOM=BASE/'response_geometry_v1'
LOCAL=BASE/'response_local_geometry_v1'
OUT=ROOT/'results/bibm_extensions_20261005/agent_c_multiscale'
REPS=['pca','scvi','geneformer','state']
RN={'pca':'PCA','scvi':'scVI','geneformer':'Geneformer V2-316M','state':'STATE SE-600M'}
DIMS={'pca':50,'scvi':50,'geneformer':1152,'state':2058}
DONORS=['D1','D2','D3','D4']; CONTEXTS=['Rest','Stim8hr','Stim48hr']
KS=[5,10,15,20,30,50,100,200]
TAILS=[.5,.25,.10,.05,.02,.01,.005,.001]
BLOCK_ROWS=int(os.environ.get('MULTISCALE_BLOCK_ROWS','128'))

def log(s): print(time.strftime('%F %T'),s,flush=True)

def exact_knn(x, names, kmax):
    x=np.asarray(x,dtype=np.float64)
    norms=np.linalg.norm(x,axis=1)
    if not np.isfinite(x).all() or (norms<=0).any(): raise ValueError('invalid response vector')
    x=np.ascontiguousarray(x/norms[:,None]); n=len(names)
    all_n=[]; all_c=[]
    namearr=np.asarray(names,dtype=object)
    for lo in range(0,n,BLOCK_ROWS):
        hi=min(n,lo+BLOCK_ROWS); scores=x[lo:hi]@x.T
        for j,row in enumerate(range(lo,hi)):
            s=scores[j]; s[row]=-np.inf
            cutoff=np.partition(s,n-kmax)[n-kmax]
            above=np.flatnonzero(s>cutoff); tied=np.flatnonzero(s==cutoff)
            need=kmax-len(above)
            if need<len(tied):
                tied=sorted(tied.tolist(),key=lambda q:names[q])[:need]
                chosen=np.concatenate((above,np.asarray(tied,dtype=np.int64)))
            else: chosen=np.concatenate((above,tied))
            order=np.lexsort((namearr[chosen],-s[chosen])); chosen=chosen[order]
            all_n.append(chosen); all_c.append(np.clip(s[chosen],-1,1))
        if lo==0 or hi==n or (lo//BLOCK_ROWS)%20==0: log(f'knn rows {hi}/{n}')
    return np.asarray(all_n,dtype=np.int32),np.asarray(all_c,dtype=np.float64)

def rbo(a,b,p=.9):
    # Finite, non-extrapolated RBO for equal-length ordered lists.
    seen_a=set(); seen_b=set(); total=0.0
    for d,(x,y) in enumerate(zip(a,b),1):
        seen_a.add(int(x)); seen_b.add(int(y))
        overlap=len(seen_a&seen_b)/d
        total += (1-p)*overlap*(p**(d-1))
    return float(total)

def local_curves():
    rows=[]; pairrows=[]; rep_pairs=list(itertools.combinations(REPS,2))
    meta={r:pd.read_csv(RESP/r/'response_units.tsv.gz',sep='\t',usecols=['target','donor','context']) for r in REPS}
    mats={r:np.load(RESP/r/'response_vectors.npy',mmap_mode='r') for r in REPS}
    for d in DONORS:
      for c in CONTEXTS:
        ix={}; target_order={}
        for r in REPS:
            md=meta[r]; ii=np.flatnonzero((md.donor.to_numpy()==d)&(md.context.to_numpy()==c))
            names=md.iloc[ii].target.astype(str).tolist(); order=np.argsort(np.asarray(names,dtype=str),kind='stable')
            ix[r]=ii[order]; target_order[r]=[names[z] for z in order]
        base=target_order['pca']
        if any(target_order[r]!=base for r in REPS[1:]): raise ValueError(f'key/order mismatch {d} {c}')
        n=len(base); kmax=min(max(KS),n-1); maps={}
        for r in REPS:
            neighbors,cos=exact_knn(mats[r][ix[r],:],base,kmax)
            maps[r]=(neighbors,cos)
            # Per representation rank detail retained compactly as wide NPZ per block.
            np.savez_compressed(OUT/f'knn_{r}_{d}_{c}.npz',targets=np.asarray(base,dtype='U'),indices=neighbors,cosines=cos)
        for ra,rb in rep_pairs:
            a=maps[ra][0]; b=maps[rb][0]
            for k in KS:
                if k>kmax: continue
                vals=np.empty(n,float); jacs=np.empty(n,float)
                for q in range(n):
                    sa=set(map(int,a[q,:k])); sb=set(map(int,b[q,:k])); inter=len(sa&sb)
                    vals[q]=inter/k; jacs[q]=inter/(2*k-inter)
                rows.append({'donor':d,'context':c,'representation_a':RN[ra],'representation_b':RN[rb], 'k':k,'n_targets':n,'mean_Ok':vals.mean(),'median_Ok':np.median(vals),'sd_Ok':vals.std(ddof=1),'mean_jaccard':jacs.mean(),'random_expected_Ok':k/(n-1)})
            for k in (15,20):
                if k<=kmax:
                    v=np.array([rbo(a[q,:k],b[q,:k],.9) for q in range(n)])
                    pairrows.append({'donor':d,'context':c,'representation_a':RN[ra],'representation_b':RN[rb],'rbo_p':.9,'depth':k,'n_targets':n,'mean_RBO':v.mean(),'median_RBO':np.median(v),'sd_RBO':v.std(ddof=1)})
        log(f'local complete {d} {c}: {n} targets')
    df=pd.DataFrame(rows); rdf=pd.DataFrame(pairrows).drop_duplicates(['donor','context','representation_a','representation_b','depth'])
    df.to_csv(OUT/'ok_by_donor_context.tsv.gz',sep='\t',index=False,compression='gzip')
    rdf.to_csv(OUT/'rbo_by_donor_context.tsv.gz',sep='\t',index=False,compression='gzip')
    df.groupby(['context','representation_a','representation_b','k'],as_index=False).agg(mean_Ok=('mean_Ok','mean'),sd_across_donors=('mean_Ok','std'),min_donor=('mean_Ok','min'),max_donor=('mean_Ok','max')).to_csv(OUT/'ok_context_summary.tsv',sep='\t',index=False)
    rdf.groupby(['context','representation_a','representation_b','depth'],as_index=False).agg(mean_RBO=('mean_RBO','mean'),sd_across_donors=('mean_RBO','std'),min_donor=('mean_RBO','min'),max_donor=('mean_RBO','max')).to_csv(OUT/'rbo_context_summary.tsv',sep='\t',index=False)

def edge_tails():
    # Same canonical unordered edge rows in every representation; verify ordering
    # by checking endpoint arrays blockwise while streaming each parquet.
    results=[]
    for d in DONORS:
      for c in CONTEXTS:
        vals={}; nrows=None; baseline_hashes=None
        for r in REPS:
            p=GEOM/r/'cosine_pairs'/f'{d}_{c}.parquet'; pf=pq.ParquetFile(p)
            n=pf.metadata.num_rows
            if nrows is None: nrows=n
            elif n!=nrows: raise ValueError(f'edge row count mismatch {d} {c} {r}')
            arr=np.empty(n,dtype=np.float64); pos=0; endpoint_hashes=[]
            for batch in pf.iter_batches(batch_size=1_000_000,columns=['target_a','target_b','cosine_similarity']):
                a=batch.column(0).to_numpy(zero_copy_only=False); b=batch.column(1).to_numpy(zero_copy_only=False)
                endpoint_hashes.append(hashlib.sha256(pd.util.hash_pandas_object(pd.DataFrame({'a':a,'b':b}),index=False).to_numpy().tobytes()).hexdigest())
                x=batch.column(2).to_numpy(zero_copy_only=False); arr[pos:pos+len(x)]=x; pos+=len(x)
            if pos!=n: raise ValueError('incomplete parquet scan')
            if baseline_hashes is None: baseline_hashes=endpoint_hashes
            elif endpoint_hashes!=baseline_hashes: raise ValueError(f'edge key order mismatch in {r} {d} {c}')
            vals[r]=arr
            log(f'tail input read {r} {d} {c}: {n:,} edges')
        # Full pair files are row ordered by canonical target_a,target_b. Verify
        # the first 100 rows across reps were identical during each read below
        # (the upstream geometry key audit establishes full alignment).
        idx_by_rep={}
        for r,x in vals.items():
            idx_by_rep[r]={}
            for frac in TAILS:
                # Deterministic top-n selection. Index order resolves exact ties.
                m=max(1,int(np.ceil(frac*nrows)))
                threshold=np.partition(x,nrows-m)[nrows-m]
                gt=np.flatnonzero(x>threshold); eq=np.flatnonzero(x==threshold)
                take=np.concatenate((gt,eq[:max(0,m-len(gt))])).astype(np.int32)
                idx_by_rep[r][frac]=np.sort(take)
        for ra,rb in itertools.combinations(REPS,2):
            for frac in TAILS:
                aa=idx_by_rep[ra][frac]; bb=idx_by_rep[rb][frac]
                inter=np.intersect1d(aa,bb,assume_unique=True).size
                union=len(aa)+len(bb)-inter
                results.append({'donor':d,'context':c,'representation_a':RN[ra],'representation_b':RN[rb],'tail_fraction':frac,'n_edges_universe':int(nrows),'n_edges_each':int(len(aa)),'intersection':int(inter),'jaccard':float(inter/union),'overlap_fraction':float(inter/len(aa))})
        del vals,idx_by_rep
    out=pd.DataFrame(results); out.to_csv(OUT/'strength_tail_edge_agreement.tsv.gz',sep='\t',index=False,compression='gzip')
    out.groupby(['context','representation_a','representation_b','tail_fraction'],as_index=False).agg(mean_jaccard=('jaccard','mean'),sd_across_donors=('jaccard','std'),mean_overlap=('overlap_fraction','mean'),min_donor_jaccard=('jaccard','min'),max_donor_jaccard=('jaccard','max')).to_csv(OUT/'strength_tail_context_summary.tsv',sep='\t',index=False)

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    started=time.time()
    local_curves(); edge_tails()
    report={'status':'PASS','authority':'root 00_audit/AUDIT.md and SHARED_SPEC.md; V31 response construction','sources':[str(RESP/r/'response_vectors.npy') for r in REPS]+[str(RESP/r/'response_units.tsv.gz') for r in REPS]+[str(GEOM/r/'cosine_pairs') for r in REPS], 'k':KS,'tails':TAILS,'rbo_p':.9,'block_rows':BLOCK_ROWS,'python':sys.version,'platform':platform.platform(),'runtime_seconds':time.time()-started}
    (OUT/'run_metadata.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__': main()
