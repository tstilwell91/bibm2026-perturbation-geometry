#!/usr/bin/env python3
"""Chunked complete-case 4x4x3 fixed-effects ANOVA on response cosine edges."""
from pathlib import Path
from datetime import datetime, timezone
import json, time, tempfile, shutil, os
import numpy as np, pandas as pd

ROOT=Path('./source/derived/downstream').resolve()
RESP=ROOT/'response_construction_v1'; OUT=ROOT/'response_prebiology_v1/edge_variance'
REPS=['pca','scvi','geneformer','state']; RLAB=['PCA','scVI','Geneformer V2-316M','STATE SE-600M']
DONORS=['D1','D2','D3','D4']; CONTEXTS=['Rest','Stim8hr','Stim48hr']
TERMS=['frac_R','frac_D','frac_C','frac_RD','frac_RC','frac_DC','frac_RDC','representation_involving_fraction','biology_only_fraction']
STATCOLS=TERMS+['mean_cosine','sd_cosine_across_48','cosine_min','cosine_max','cosine_range','SS_total']
CHUNK=50000; CONSTANT_TOL=1e-12; SAMPLE_MOD=61; HASH_SEED=np.uint64(0x9E3779B97F4A7C15)

def edge_chunks(n, chunk):
    bi=np.empty(chunk,dtype=np.int32); bj=np.empty(chunk,dtype=np.int32)
    ptr=0; k0=0
    for i in range(n-1):
        j0=i+1
        while j0<n:
            take=min(chunk-ptr,n-j0)
            bi[ptr:ptr+take]=i; bj[ptr:ptr+take]=np.arange(j0,j0+take,dtype=np.int32)
            ptr+=take; j0+=take
            if ptr==chunk:
                yield k0,bi.copy(),bj.copy(); k0+=ptr; ptr=0
    if ptr: yield k0,bi[:ptr].copy(),bj[:ptr].copy()

def splitmix64(x):
    z=(x.astype(np.uint64)+HASH_SEED)&np.uint64(0xffffffffffffffff)
    z=(z^(z>>np.uint64(30)))*np.uint64(0xbf58476d1ce4e5b9)
    z=(z^(z>>np.uint64(27)))*np.uint64(0x94d049bb133111eb)
    return z^(z>>np.uint64(31))

def summarize(v):
    q=np.quantile(v,[.05,.25,.5,.75,.95])
    return {'mean':float(np.mean(v)),'median':float(q[2]),'std':float(np.std(v)),
            'q05':float(q[0]),'q25':float(q[1]),'q75':float(q[3]),'q95':float(q[4])}

def md_table(df):
    cols=[str(c) for c in df.columns]
    rows=[]
    for row in df.itertuples(index=False,name=None):
        vals=[]
        for v in row:
            if pd.isna(v): vals.append('NA')
            elif isinstance(v,(float,np.floating)): vals.append(f'{v:.4g}')
            else: vals.append(str(v))
        rows.append('| '+' | '.join(vals)+' |')
    return '| '+' | '.join(cols)+' |\n| '+' | '.join(['---']*len(cols))+' |\n'+'\n'.join(rows)

def main():
  t0=time.time(); metas={}; arrs={}; lookups={}; targets_by_block={}
  for r in REPS:
      meta=pd.read_csv(RESP/r/'response_units.tsv.gz',sep='\t',usecols=['target','donor','context'])
      arr=np.load(RESP/r/'response_vectors.npy',mmap_mode='r',allow_pickle=False)
      if arr.shape[0]!=144368 or len(meta)!=144368: raise RuntimeError(f'Unexpected response-unit shape {r}: {arr.shape}')
      metas[r]=meta; arrs[r]=arr
      lookups[r]={(str(t),str(d),str(c)):i for i,(t,d,c) in enumerate(meta[['target','donor','context']].itertuples(index=False,name=None))}
  base=metas['pca'][['target','donor','context']]
  for r in REPS[1:]:
      if not base.equals(metas[r][['target','donor','context']]): raise RuntimeError(f'Response-unit key order differs: {r}')
  if base.duplicated(['target','donor','context']).any(): raise RuntimeError('Duplicate target/donor/context response key')
  support=base.groupby('target').agg(n=('donor','size'),donors=('donor','nunique'),blocks=('context','nunique'))
  complete=sorted(support.index[(support.n==12)&(support.donors==4)&(support.blocks==3)].tolist())
  if not complete: raise RuntimeError('Complete target set is empty')
  n=len(complete); nedge=n*(n-1)//2
  (OUT/'complete_target_set.txt').write_text('\n'.join(complete)+'\n')
  # Direct native response vectors, normalized solely for cosine scoring.
  vec={}
  for r in REPS:
    for d in DONORS:
      for c in CONTEXTS:
        ii=np.fromiter((lookups[r][(t,d,c)] for t in complete),dtype=np.int64,count=n)
        x=np.asarray(arrs[r][ii],dtype=np.float64)
        norm=np.linalg.norm(x,axis=1)
        if not np.isfinite(x).all() or not np.isfinite(norm).all() or (norm<=0).any(): raise RuntimeError(f'Invalid complete-case responses: {r}/{d}/{c}')
        vec[(r,d,c)]=x/norm[:,None]
  tmp=Path(tempfile.mkdtemp(prefix='edgevar_',dir='.'))
  try:
    stats=np.lib.format.open_memmap(tmp/'stats.npy',mode='w+',dtype='float64',shape=(nedge,len(STATCOLS)))
    const=np.lib.format.open_memmap(tmp/'constant.npy',mode='w+',dtype='uint8',shape=(nedge,))
    sample_ids=[]; top_score=np.empty(0); top_index=np.empty(0,dtype=np.int64)
    abs_seen=0; constant_count=0; edge_done=0
    chunks_total=(nedge+CHUNK-1)//CHUNK
    last_heartbeat=t0
    for chunk_number,(k0,ia,ib) in enumerate(edge_chunks(n,CHUNK),start=1):
      m=len(ia); Y=np.empty((m,4,4,3),dtype=np.float64)
      for ri,r in enumerate(REPS):
        for di,d in enumerate(DONORS):
          for ci,c in enumerate(CONTEXTS):
            x=vec[(r,d,c)]
            Y[:,ri,di,ci]=np.einsum('ij,ij->i',x[ia],x[ib],optimize=False)
      if not np.isfinite(Y).all() or np.any(np.abs(Y)>1+1e-10): raise RuntimeError(f'Invalid cosine score in edge chunk {k0}')
      grand=Y.mean(axis=(1,2,3))
      er=Y.mean(axis=(2,3))-grand[:,None]
      ed=Y.mean(axis=(1,3))-grand[:,None]
      ec=Y.mean(axis=(1,2))-grand[:,None]
      erd=Y.mean(axis=3)-grand[:,None,None]-er[:,:,None]-ed[:,None,:]
      erc=Y.mean(axis=2)-grand[:,None,None]-er[:,:,None]-ec[:,None,:]
      edc=Y.mean(axis=1)-grand[:,None,None]-ed[:,:,None]-ec[:,None,:]
      fitted=(grand[:,None,None,None]+er[:,:,None,None]+ed[:,None,:,None]+ec[:,None,None,:]
              +erd[:,:,:,None]+erc[:,:,None,:]+edc[:,None,:,:])
      erdc=Y-fitted
      ssr=12*np.sum(er*er,axis=1); ssd=12*np.sum(ed*ed,axis=1); ssc=16*np.sum(ec*ec,axis=1)
      ssrd=3*np.sum(erd*erd,axis=(1,2)); ssrc=4*np.sum(erc*erc,axis=(1,2)); ssdc=4*np.sum(edc*edc,axis=(1,2))
      ssrdc=np.sum(erdc*erdc,axis=(1,2,3)); sst=np.sum((Y-grand[:,None,None,None])**2,axis=(1,2,3))
      ssum=ssr+ssd+ssc+ssrd+ssrc+ssdc+ssrdc
      tol=1e-10*np.maximum(sst,1e-20)+1e-25
      if np.any(np.abs(ssum-sst)>tol): raise RuntimeError(f'ANOVA SS additivity failure at edge chunk {k0}')
      crange=Y.max(axis=(1,2,3))-Y.min(axis=(1,2,3)); isconst=crange<=CONSTANT_TOL
      constant_count+=int(isconst.sum()); const[k0:k0+m]=isconst
      fr=np.full((m,9),np.nan,dtype=np.float64); valid=~isconst
      if valid.any():
        f7=np.column_stack([ssr[valid],ssd[valid],ssc[valid],ssrd[valid],ssrc[valid],ssdc[valid],ssrdc[valid]])/sst[valid,None]
        fri=f7[:,0]+f7[:,3]+f7[:,4]+f7[:,6]; fbi=f7[:,1]+f7[:,2]+f7[:,5]
        fr[valid,:7]=f7; fr[valid,7]=fri; fr[valid,8]=fbi
        if not np.allclose(f7.sum(axis=1),1,atol=2e-8,rtol=2e-8):
            raise RuntimeError(f'Variance fractions do not sum to one at edge chunk {k0}')
      stats[k0:k0+m,0:9]=fr
      stats[k0:k0+m,9]=grand; stats[k0:k0+m,10]=Y.std(axis=(1,2,3)); stats[k0:k0+m,11]=Y.min(axis=(1,2,3)); stats[k0:k0+m,12]=Y.max(axis=(1,2,3)); stats[k0:k0+m,13]=crange; stats[k0:k0+m,14]=sst
      # Keep a bounded top-SS_R index list during the one edge pass.
      sr=ssr
      cand_score=np.concatenate([top_score,sr]); cand_idx=np.concatenate([top_index,np.arange(k0,k0+m,dtype=np.int64)])
      if len(cand_score)>10000:
        keep=np.argpartition(cand_score,-10000)[-10000:]; top_score=cand_score[keep]; top_index=cand_idx[keep]
      else: top_score=cand_score; top_index=cand_idx
      # Fixed 64-bit hash of deterministic edge ordinal; mod 61 keeps <=1M for this cohort.
      eid=np.arange(k0,k0+m,dtype=np.uint64); take=(splitmix64(eid)%np.uint64(SAMPLE_MOD))==0
      if take.any(): sample_ids.extend(eid[take].astype(np.int64).tolist())
      edge_done+=m
      now=time.time()
      if now-last_heartbeat>=300 or edge_done==nedge:
        elapsed=now-t0; rate=edge_done/elapsed if elapsed>0 else 0.0
        eta=(nedge-edge_done)/rate if rate>0 else None
        print(json.dumps({'event':'edge_heartbeat','timestamp_utc':datetime.now(timezone.utc).isoformat(),
          'chunk_number':chunk_number,'chunks_total':chunks_total,'edges_processed':edge_done,
          'total_edges':nedge,'percent_complete':100.0*edge_done/nedge,'elapsed_seconds':elapsed,
          'edges_per_second':rate,'estimated_seconds_remaining':eta,'current_output':str(tmp/'stats.npy'),
          'constant_edges_so_far':constant_count}),flush=True)
        last_heartbeat=now
    stats.flush(); const.flush()
    if edge_done!=nedge: raise RuntimeError(f'Edge count mismatch {edge_done}/{nedge}')
    nonconst=const[:]==0
    # Overall exact summaries of nonconstant edge fraction distributions.
    summary=[]
    for j,name in enumerate(TERMS):
      vals=np.asarray(stats[nonconst,j]); s=summarize(vals)
      summary.append({'term':name,**s})
    v=np.asarray(stats[nonconst,:9]);
    summary += [
      {'term':'fraction_frac_R_gt_frac_D','mean':float(np.mean(v[:,0]>v[:,1]))},
      {'term':'fraction_frac_R_gt_frac_C','mean':float(np.mean(v[:,0]>v[:,2]))},
      {'term':'fraction_frac_R_largest_main_effect','mean':float(np.mean((v[:,0]>v[:,1])&(v[:,0]>v[:,2])))},
      {'term':'fraction_representation_involving_gt_biology_only','mean':float(np.mean(v[:,7]>v[:,8]))}]
    pd.DataFrame(summary).to_csv(OUT/'factorial_summary.tsv.gz',sep='\t',index=False,compression='gzip')
    # Fixed absolute mean-cosine strength bins.
    meanabs=np.abs(stats[:,9]); bins=[(0,.1,'[0,.1)'),(.1,.2,'[.1,.2)'),(.2,.4,'[.2,.4)'),(.4,.6,'[.4,.6)'),(.6,1.0000000001,'[.6,1]')]
    bybin=[]
    for lo,hi,label in bins:
      mask=nonconst&(meanabs>=lo)&(meanabs<hi)
      row={'cosine_strength_bin':label,'n_edges':int(mask.sum())}
      for j,name in enumerate(TERMS):
        row.update({f'{name}_{key}':val for key,val in summarize(np.asarray(stats[mask,j])).items()}) if mask.any() else None
      bybin.append(row)
    pd.DataFrame(bybin).to_csv(OUT/'factorial_by_cosine_bin.tsv.gz',sep='\t',index=False,compression='gzip')
    median_sd=float(np.median(stats[nonconst,10]))
    # Exact top/bottom representation-involving and biology-only edges among nontrivial SD > median.
    nontrivial=nonconst&(stats[:,10]>median_sd)
    def top_indices(col,k=10000,largest=True,restrict=None):
      values=np.asarray(stats[:,col]).copy()
      if restrict is not None: values[~restrict]=(-np.inf if largest else np.inf)
      nvalid=int(np.isfinite(values).sum())
      kk=min(k,nvalid)
      if kk==0: return np.empty(0,dtype=np.int64)
      ix=np.argpartition(values,-kk)[-kk:] if largest else np.argpartition(values,kk-1)[:kk]
      if largest: ix=ix[np.argsort(values[ix])[::-1]]
      else: ix=ix[np.argsort(values[ix])]
      return ix
    selections=[('top_abs_SS_R',top_index),('top_representation_involving_fraction',top_indices(7,restrict=nontrivial)),
      ('top_biology_only_fraction',top_indices(8,restrict=nontrivial)),('lowest_representation_involving_fraction',top_indices(7,largest=False,restrict=nontrivial))]
    def decode_edge(ids):
      k=np.asarray(ids,dtype=np.float64); nn=float(n); disc=(2*nn-1)**2-8*k
      ii=np.floor(((2*nn-1)-np.sqrt(np.maximum(disc,0)))/2).astype(np.int64)
      pref=ii*(2*n-ii-1)//2
      adjust=ii*(2*n-ii-1)//2
      # Floating quadratic rounding can only move a boundary by one row; correct exactly.
      over=pref>ids; ii[over]-=1
      pref=ii*(2*n-ii-1)//2
      nxt=(ii+1)*(2*n-ii-2)//2
      under=ids>=nxt
      ii[under]+=1; pref=ii*(2*n-ii-1)//2
      jj=ii+1+(ids-pref)
      if np.any(jj>=n): raise RuntimeError('Edge ordinal decode outside target set')
      return ii,jj
    def records(edge_ids,selection):
      ii,jj=decode_edge(edge_ids); s=np.asarray(stats[edge_ids]); ssrvals=s[:,0]*s[:,14]
      out=pd.DataFrame({'selection':selection,'edge_index':edge_ids,'target_a':[complete[x] for x in ii],'target_b':[complete[x] for x in jj]})
      for q,col in enumerate(STATCOLS): out[col]=s[:,q]
      out['SS_R']=ssrvals; out['constant_edge']=np.asarray(const[edge_ids],dtype=bool)
      return out
    extreme=pd.concat([records(ids,label) for label,ids in selections],ignore_index=True)
    extreme.to_parquet(OUT/'edge_variance_extremes.parquet',index=False)
    sample_ids=np.asarray(sorted(set(sample_ids)),dtype=np.int64)
    sample=records(sample_ids,'hash_sample')
    sample['hash_seed']='splitmix64(edge_ordinal + fixed_seed) % 61 == 0'
    sample.to_parquet(OUT/'edge_variance_sample.parquet',index=False)
    # Scientific/analysis-specific integrity audit, no all-pair artifacts are retained.
    audit={'status':'PASS','complete_target_count':n,'complete_edge_count':nedge,'edges_processed':edge_done,
      'constant_edges':constant_count,'nonconstant_edges':int(nonconst.sum()),'chunk_size':CHUNK,
      'constant_tolerance':'cosine_range <= 1e-12','sampled_edges':len(sample_ids),'sample_rule':'SplitMix64 hash of deterministic lexicographic edge ordinal, seed 0x9E3779B97F4A7C15, retain hash modulo 61 == 0; maximum below 1,000,000 for this cohort.',
      'anova_formula':'Balanced 4x4x3 marginal-mean orthogonal SS: SS_R=12*sum(r_effect^2), SS_D=12*sum(d_effect^2), SS_C=16*sum(c_effect^2), SS_RD=3*sum(rd_effect^2), SS_RC=4*sum(rc_effect^2), SS_DC=4*sum(dc_effect^2), SS_RDC=sum(full residual^2). No error term or inferential test.',
      'runtime_seconds':time.time()-t0,'representations':RLAB,'optional_resource_annotation':'Not run; resource-labeled edge summaries were optional and would require another large pair-list join.'}
    (OUT/'EDGE_VARIANCE_SUMMARY.json').write_text(json.dumps(audit,indent=2)+'\n')
    # concise markdown, including requested full and binned descriptions
    sd=pd.DataFrame(summary); bd=pd.DataFrame(bybin)
    text=['# Edge-level factorial variance decomposition','',f"Complete targets: **{n:,}**; unordered edges: **{nedge:,}**; constant edges: **{constant_count:,}**.",'',
      f'Chunk size: {CHUNK:,}. Runtime: {audit["runtime_seconds"]:.1f} seconds. Cosine range <= {CONSTANT_TOL:g} marks a numerically constant edge; these remain counted and are excluded from fraction summaries. Fixed balanced marginal-mean ANOVA sums of squares saturate the 4 × 4 × 3 design; no residual/error term, p-values, or F-tests are used.','',
      '## Edge fraction distributions','',md_table(sd[sd.term.isin(TERMS)]),'',
      '## Descriptive edge comparisons','',md_table(sd[~sd.term.isin(TERMS)]),'',
      '## Absolute mean-cosine bins','',md_table(bd),'',
      'Retained edge records comprise a deterministic hash sample and four requested edge-extreme selections. Pair IDs are canonical (`target_a < target_b`). Optional known-resource labels were not added.','']
    (OUT/'EDGE_VARIANCE_SUMMARY.md').write_text('\n'.join(text))
  finally:
    del stats, const
    shutil.rmtree(tmp,ignore_errors=True)
  print(json.dumps({'event':'edge_variance_complete','seconds':time.time()-t0,'targets':n,'edges':nedge,'constant':constant_count}),flush=True)

if __name__=='__main__': main()
