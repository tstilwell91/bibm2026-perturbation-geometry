#!/usr/bin/env python3
"""Stream V31 corrected target cells and compute LOO directional coherence."""
from pathlib import Path
import csv, gzip, json, time, hashlib
import numpy as np
import pandas as pd
import h5py
from scipy.stats import spearmanr, pearsonr
from scipy.linalg import lstsq

ROOT=Path('.').resolve()
BASE=ROOT/'source/derived/downstream'
RESP=BASE/'response_construction_v1'
INDEX=BASE/'common_cell_audit_v1/COMMON_DOWNSTREAM_CELL_INDEX.tsv.gz'
OUT=BASE/'response_prebiology_v1/cell_coherence'
N_EXPECTED=21_710_586; N_TARGET=20_771_051; BATCH=10_000; SAMPLE_MOD=17; SAMPLE_CAP=100_000
REPS=['pca','scvi','geneformer','state']; LABEL={'pca':'PCA','scvi':'scVI','geneformer':'Geneformer','state':'STATE'}
DONORS=['D1','D2','D3','D4']; CONTEXTS=['Rest','Stim8hr','Stim48hr']
PCA=ROOT/'source/derived/pca_direct_full_materialized_v28/runs/job-6848910/scores/pca_scores_21m_x_50.h5'
SCVI=ROOT/'global_scvi/final/primary_2k_allcell_v1/seed42/latent_v2/latent_50d_float32.npy'
GF=ROOT/'geneformer_global_cell_embeddings/outputs'; STATE=ROOT/'source/derived/v35_state_all_targets/STATE/shards'
DIM={'pca':50,'scvi':50,'geneformer':1152,'state':2058}

def idx_codes(s, vals): return pd.Categorical(s,categories=vals).codes.astype(np.int16)
def corr(a,b,method):
    ok=np.isfinite(a)&np.isfinite(b)
    if ok.sum()<3: return np.nan,np.nan,int(ok.sum())
    res=spearmanr(a[ok],b[ok]) if method=='spearman' else pearsonr(a[ok],b[ok])
    return float(res.statistic),float(res.pvalue),int(ok.sum())
def read_rep(rep,start,end,src_names):
    """Read canonical contiguous rows, segmented by frozen source_file blocks."""
    out=np.empty((end-start,DIM[rep]),dtype=np.float32)
    pos=start
    while pos<end:
        # source file has already been mapped to a sample block; blocks never exceed source bounds
        sample=src_names[pos]
        stop=pos+1
        while stop<end and src_names[stop]==sample: stop+=1
        if rep=='pca':
            if not hasattr(read_rep,'pca'): read_rep.pca=h5py.File(PCA,'r')
            out[pos-start:stop-start]=read_rep.pca['coordinates'][:,pos:stop].T
        elif rep=='scvi':
            if not hasattr(read_rep,'scvi'): read_rep.scvi=np.load(SCVI,mmap_mode='r',allow_pickle=False)
            out[pos-start:stop-start]=read_rep.scvi[pos:stop]
        elif rep=='state':
            a=np.load(STATE/sample/'embeddings.npy',mmap_mode='r',allow_pickle=False)
            # canonical source sample rows begin at their index block start
            block_start=read_rep.source_starts[sample]
            out[pos-start:stop-start]=a[pos-block_start:stop-block_start]
        else:
            plan=read_rep.plans[sample]; cs=int(plan['chunk_size']); p=pos
            block_start=read_rep.source_starts[sample]
            while p<stop:
                local=p-block_start; ci=local//cs; e=min(stop,block_start+(ci+1)*cs)
                a=np.load(GF/sample/'chunks'/f'chunk_{ci:05d}.emb.npy',mmap_mode='r',allow_pickle=False)
                out[p-start:e-start]=a[local%cs:local%cs+(e-p)]
                p=e
        pos=stop
    return out

def main():
    t0=time.time(); OUT.mkdir(parents=True,exist_ok=True); (OUT.parent/'audits').mkdir(parents=True,exist_ok=True)
    units={}; unit_meta={}; centroid_arrays={}; centroid_ids={}
    for rep in REPS:
        m=pd.read_csv(RESP/rep/'response_units.tsv.gz',sep='\t')
        if len(m)!=144368 or not {'target','donor','context','n_target_cells'}.issubset(m.columns): raise RuntimeError(f'Bad frozen response unit metadata: {rep}')
        units[rep]=m
        if rep!='pca' and not m[['target','donor','context','n_target_cells']].equals(units['pca'][['target','donor','context','n_target_cells']]): raise RuntimeError(f'Unit support differs: {rep}')
        unit_meta[rep]=np.load(RESP/rep/'response_vectors.npy',mmap_mode='r',allow_pickle=False)
        cm=pd.read_csv(RESP/rep/'ntc_centroid_units.tsv.gz',sep='\t')
        ca=np.load(RESP/rep/'ntc_centroids.npy',mmap_mode='r',allow_pickle=False)
        if ca.shape!=(len(cm),DIM[rep]) or not np.isfinite(ca).all(): raise RuntimeError(f'Bad saved NTC centroids {rep}')
        centroid_arrays[rep]=ca; centroid_ids[rep]={ (str(x.donor),str(x.context),str(x.lane)):i for i,x in enumerate(cm.itertuples(index=False)) }
    # Dense metadata lookup: target x donor x context -> response-unit row.
    targets=sorted(units['pca'].target.astype(str).unique()); tcode={x:i for i,x in enumerate(targets)}
    dcode={x:i for i,x in enumerate(DONORS)}; ccode={x:i for i,x in enumerate(CONTEXTS)}
    lookup=np.full((len(targets),4,3),-1,dtype=np.int32)
    for j,r in enumerate(units['pca'].itertuples(index=False)):
        lookup[tcode[str(r.target)],dcode[str(r.donor)],ccode[str(r.context)]]=j
    unit_n=units['pca'].n_target_cells.to_numpy(dtype=np.int64)
    expected=np.zeros((4,3,len(targets)),dtype=np.int64)
    for j,r in enumerate(units['pca'].itertuples(index=False)):
        expected[dcode[str(r.donor)],ccode[str(r.context)],tcode[str(r.target)]]=unit_n[j]
    # Read source block names from the frozen manifest's source_file convention; sample keys are D#_Context.
    source_starts={}; source_ends={}; src_names=np.empty(N_EXPECTED,dtype=object)
    # Metadata/index is streamed once. Sample cells are selected by deterministic cell-ID hash.
    # First establish contiguous source boundaries using cell index rows; sampled output retained in memory only.
    sample_rows=[]; sample_hash={}; start_row=0
    for chunk in pd.read_csv(INDEX,sep='\t',compression='gzip',usecols=['cell_id','source_file','donor','context','lane','target_status','target_name'],chunksize=BATCH,dtype=str):
        n=len(chunk); end=start_row+n
        for sf,g in chunk.groupby('source_file',sort=False).indices.items():
            # groupby indices are chunk-relative; contiguous source blocks are guaranteed by frozen construction.
            lo=start_row+int(g[0]); hi=start_row+int(g[-1])+1
            sample=sf.removesuffix('.assigned_guide.h5ad')
            if sample not in source_starts: source_starts[sample]=lo
            source_ends[sample]=hi; src_names[lo:hi]=sample
        # Record <=100k hash-selected cells per donor/context, preserving canonical row order.
        cellids=chunk.cell_id.astype(str).to_numpy()
        hashes=np.fromiter((int.from_bytes(hashlib.blake2b(x.encode(),digest_size=8,person=b'cellcoh1').digest(),'little') for x in cellids),dtype=np.uint64,count=n)
        choose=(hashes%np.uint64(SAMPLE_MOD)==0)&(chunk.target_status.to_numpy()=='target')
        for q in np.flatnonzero(choose):
            row=start_row+int(q); key=(chunk.donor.iloc[q],chunk.context.iloc[q]); sample_hash.setdefault(key,[])
            if len(sample_hash[key])<SAMPLE_CAP:
                sample_hash[key].append({'canonical_row':row,'cell_id':cellids[q],'target':chunk.target_name.iloc[q],
                  'donor':key[0],'context':key[1],'lane':chunk.lane.iloc[q]})
        start_row=end
    if start_row!=N_EXPECTED: raise RuntimeError(f'Frozen index row count {start_row}')
    read_rep.source_starts=source_starts; read_rep.plans={s:json.loads((GF/s/'chunk_plan.json').read_text()) for s in source_starts}
    samples=[]
    for block,rows in sample_hash.items(): samples.extend(rows)
    samples.sort(key=lambda x:x['canonical_row'])
    sample_by_row={x['canonical_row']:i for i,x in enumerate(samples)}
    coh_sample=np.full((len(samples),4),np.nan,dtype=np.float32)
    n_sample=np.zeros(len(samples),dtype=np.int32)
    # Streaming index + four matrices together. Unit-level sufficient statistics only.
    sums={r:np.zeros(144368,dtype=np.float64) for r in REPS}; sums2={r:np.zeros(144368,dtype=np.float64) for r in REPS}
    pos={r:np.zeros(144368,dtype=np.int64) for r in REPS}; incs={r:np.zeros(144368,dtype=np.float64) for r in REPS}; inc2={r:np.zeros(144368,dtype=np.float64) for r in REPS}; incpos={r:np.zeros(144368,dtype=np.int64) for r in REPS}
    counts=np.zeros(144368,dtype=np.int64); strata_units=np.zeros(4,dtype=np.int64); strata_cells=np.zeros(4,dtype=np.int64)
    start_row=0; sample_lookup={x['canonical_row']:i for i,x in enumerate(samples)}
    for chunk in pd.read_csv(INDEX,sep='\t',compression='gzip',usecols=['cell_id','source_file','donor','context','lane','target_status','target_name'],chunksize=BATCH,dtype=str):
        end=start_row+len(chunk); nrow=len(chunk)
        targetmask=chunk.target_status.to_numpy()=='target'
        ti=np.fromiter((tcode.get(x,-1) for x in chunk.target_name.astype(str)),dtype=np.int32,count=nrow)
        di=idx_codes(chunk.donor,DONORS); ci=idx_codes(chunk.context,CONTEXTS)
        uid=np.full(nrow,-1,dtype=np.int32); valid=targetmask&(ti>=0)
        uid[valid]=lookup[ti[valid],di[valid],ci[valid]]
        if np.any(uid[valid]<0): raise RuntimeError('Target cell without frozen response unit')
        # Same frozen matched NTC centroid keys across reps. Read all representation blocks and aggregate immediately.
        for ri,rep in enumerate(REPS):
            raw=read_rep(rep,start_row,end,src_names)
            if raw.shape!=(nrow,DIM[rep]) or not np.isfinite(raw).all(): raise RuntimeError(f'Invalid raw representation block {rep}/{start_row}:{end}')
            centroid_idx=np.fromiter((centroid_ids[rep][(str(d),str(c),str(l))] for d,c,l in zip(chunk.donor,chunk.context,chunk.lane)),dtype=np.int32,count=nrow)
            corrected=raw.astype(np.float64)-centroid_arrays[rep][centroid_idx]
            tv=uid>=0; zr=corrected[tv]; uv=uid[tv]
            r=np.asarray(unit_meta[rep][uv],dtype=np.float64); nn=unit_n[uv].astype(np.float64)
            dot=np.einsum('ij,ij->i',zr,r); zn2=np.einsum('ij,ij->i',zr,zr); rn2=np.einsum('ij,ij->i',r,r)
            zn=np.sqrt(zn2); incl=dot/(zn*np.sqrt(rn2)); loo=np.full(len(uv),np.nan,dtype=np.float64)
            elig=nn>=2
            loo[elig]=(nn[elig]*dot[elig]-zn2[elig])/(zn[elig]*np.sqrt(nn[elig]**2*rn2[elig]+zn2[elig]-2*nn[elig]*dot[elig]))
            if not np.isfinite(incl).all() or np.any(np.abs(incl)>1+1e-9) or not np.isfinite(loo[elig]).all() or np.any(np.abs(loo[elig])>1+1e-9): raise RuntimeError(f'Invalid coherence {rep} rows {start_row}:{end}')
            sums[rep]+=np.bincount(uv,weights=np.nan_to_num(loo),minlength=144368); sums2[rep]+=np.bincount(uv,weights=np.nan_to_num(loo)**2,minlength=144368); pos[rep]+=np.bincount(uv,weights=(loo>0).astype(np.int64),minlength=144368).astype(np.int64)
            incs[rep]+=np.bincount(uv,weights=incl,minlength=144368); inc2[rep]+=np.bincount(uv,weights=incl**2,minlength=144368); incpos[rep]+=np.bincount(uv,weights=(incl>0).astype(np.int64),minlength=144368).astype(np.int64)
            if rep==REPS[0]: counts+=np.bincount(uv,minlength=144368)
            # deterministic sample cell coherence for exact same canonical rows
            local_rows=np.flatnonzero(tv)+start_row
            for jj,row in enumerate(local_rows):
                si=sample_lookup.get(int(row))
                if si is not None: coh_sample[si,ri]=loo[jj]; n_sample[si]=int(nn[jj])
            del raw,corrected,zr,r
        start_row=end
        if start_row%1_000_000<nrow: print(json.dumps({'event':'cell_progress','rows':start_row,'target_cells':int(counts.sum()),'total':N_EXPECTED}),flush=True)
    if start_row!=N_EXPECTED or counts.sum()!=N_TARGET or not np.array_equal(counts,unit_n): raise RuntimeError(f'Cell/unit support mismatch rows={start_row}, target={counts.sum()}')
    # Check LOO algebra against explicit vector construction on a small fixed sample.
    # Explicit numerical verification on fixed tiny vectors before relying on the scalar identity.
    rng=np.random.default_rng(20260930); _r=rng.normal(size=8); _z=rng.normal(size=(4,8)); _n=5
    _dot=_z@_r; _z2=np.sum(_z*_z,axis=1); _r2=float(_r@_r)
    _fast=(_n*_dot-_z2)/(np.sqrt(_z2)*np.sqrt(_n*_n*_r2+_z2-2*_n*_dot))
    _explicit=np.array([np.dot(z,(_n*_r-z)/(_n-1))/(np.linalg.norm(z)*np.linalg.norm((_n*_r-z)/(_n-1))) for z in _z])
    if not np.allclose(_fast,_explicit,rtol=1e-12,atol=1e-12): raise RuntimeError('LOO scalar formula failed explicit vector check')
    # Unit summaries and block-specific magnitude-conditioned OLS.
    out_units=units['pca'][['target','donor','context','n_target_cells']].copy()
    out_units['n_LOO_eligible_cells']=np.where(out_units.n_target_cells.to_numpy()>=2,out_units.n_target_cells.to_numpy(),0)
    for rep in REPS:
        out_units[f'response_magnitude_{rep}']=np.linalg.norm(np.asarray(unit_meta[rep]),axis=1)
        n=counts; mean=np.divide(sums[rep],n,out=np.full(len(n),np.nan),where=n>0); mean[n==1]=np.nan; meaninc=incs[rep]/n
        sd=np.sqrt(np.maximum(0,np.divide(sums2[rep],n,out=np.full(len(n),np.nan),where=n>0)-np.nan_to_num(mean,nan=0.0)**2)); sd[(n==0)|(n==1)]=np.nan; sdinc=np.sqrt(np.maximum(0,inc2[rep]/n-meaninc**2))
        loo_n=np.maximum(n-(n==1),0); frac=np.divide(pos[rep],loo_n,out=np.full(len(n),np.nan),where=loo_n>0)
        out_units[f'mean_coherence_LOO_{rep}']=mean; out_units[f'sd_coherence_LOO_{rep}']=sd; out_units[f'fraction_positive_LOO_{rep}']=frac
        out_units[f'mean_coherence_inclusive_{rep}']=meaninc; out_units[f'sd_coherence_inclusive_{rep}']=sdinc; out_units[f'fraction_positive_inclusive_{rep}']=incpos[rep]/n
    models=[]
    residual_cols=[]
    for rep in REPS:
      out_units[f'residualized_coherence_{rep}']=np.nan; residual_cols.append(f'residualized_coherence_{rep}')
      for d in DONORS:
       for c in CONTEXTS:
        magcol=f'response_magnitude_{rep}'
        ix=(out_units.donor==d)&(out_units.context==c)&(out_units.n_target_cells>=2)&(out_units[magcol]>0)
        y=out_units.loc[ix,f'mean_coherence_LOO_{rep}'].to_numpy(); mag=np.log(out_units.loc[ix,magcol].to_numpy()); nc=np.log1p(out_units.loc[ix,'n_target_cells'].to_numpy())
        mz=(mag-mag.mean())/(mag.std() or 1); nz=(nc-nc.mean())/(nc.std() or 1); X=np.column_stack([np.ones(len(y)),mz,mz*mz,nz]); b=lstsq(X,y)[0]; pred=X@b
        out_units.loc[ix,f'residualized_coherence_{rep}']=y-pred
        r2=1-float(np.sum((y-pred)**2)/np.sum((y-y.mean())**2)) if len(y)>1 and np.var(y)>0 else np.nan
        models.append({'representation':LABEL[rep],'donor':d,'context':c,'n_units':len(y),'intercept':b[0],'z_log_magnitude':b[1],'z_log_magnitude_squared':b[2],'z_log1p_n_cells':b[3],'r_squared':r2})
    out_units['coherence_range_across_representations']=out_units[[f'mean_coherence_LOO_{r}' for r in REPS]].max(axis=1)-out_units[[f'mean_coherence_LOO_{r}' for r in REPS]].min(axis=1)
    out_units['coherence_sd_across_representations']=out_units[[f'mean_coherence_LOO_{r}' for r in REPS]].std(axis=1)
    out_units['residualized_coherence_range']=out_units[residual_cols].max(axis=1)-out_units[residual_cols].min(axis=1)
    eligible=out_units.n_target_cells>=5; stable=pd.concat([out_units.loc[eligible].nlargest(10000,'coherence_range_across_representations').assign(selection='largest_raw_range'),out_units.loc[eligible].nlargest(10000,'residualized_coherence_range').assign(selection='largest_residualized_range'),out_units.loc[eligible].nsmallest(10000,'residualized_coherence_range').assign(selection='smallest_residualized_range')],ignore_index=True)
    out_units.to_parquet(OUT/'target_unit_coherence.parquet',index=False); stable.to_parquet(OUT/'coherence_stable_sensitive_units.parquet',index=False)
    pd.DataFrame(models).to_csv(OUT/'magnitude_conditioning_models.tsv.gz',sep='\t',index=False,compression='gzip')
    # Unit-level cross-representation agreement.
    pairs=[(REPS[i],REPS[j]) for i in range(4) for j in range(i+1,4)]; agree=[]
    for d in DONORS:
     for c in CONTEXTS:
      ix=(out_units.donor==d)&(out_units.context==c)
      for a,b in pairs:
       rec={'donor':d,'context':c,'representation_a':LABEL[a],'representation_b':LABEL[b]}
       for metric in ['mean_coherence_LOO','residualized_coherence','sd_coherence_LOO']:
        x=out_units.loc[ix,f'{metric}_{a}'].to_numpy(); y=out_units.loc[ix,f'{metric}_{b}'].to_numpy()
        for meth in (['spearman','pearson'] if metric!='sd_coherence_LOO' else ['spearman']):
         val,pv,nv=corr(x,y,meth); rec[f'{metric}_{meth}']=val; rec[f'{metric}_{meth}_p']=pv; rec[f'{metric}_{meth}_n']=nv
       agree.append(rec)
    adf=pd.DataFrame(agree); adf.to_csv(OUT/'cross_representation_unit_agreement.tsv.gz',sep='\t',index=False,compression='gzip')
    # Cell sample table and correlations.
    sm=pd.DataFrame(samples); sm['n_cells_in_response_unit']=n_sample
    for i,r in enumerate(REPS): sm[f'coherence_LOO_{r}']=coh_sample[:,i]
    sm.to_parquet(OUT/'sampled_cell_coherence.parquet',index=False)
    cage=[]
    for d in DONORS:
     for c in CONTEXTS:
      ix=(sm.donor==d)&(sm.context==c)
      for a,b in pairs:
       x=sm.loc[ix,f'coherence_LOO_{a}'].to_numpy(); y=sm.loc[ix,f'coherence_LOO_{b}'].to_numpy(); sr,sp,sn=corr(x,y,'spearman'); pr,pp,pn=corr(x,y,'pearson')
       cage.append({'donor':d,'context':c,'representation_a':LABEL[a],'representation_b':LABEL[b],'n_cells':sn,'spearman_rho':sr,'spearman_p':sp,'pearson_r':pr,'pearson_p':pp})
    pd.DataFrame(cage).to_csv(OUT/'cross_representation_cell_agreement.tsv.gz',sep='\t',index=False,compression='gzip')
    # Count strata, inclusive-minus-LOO, context summaries.
    cats=np.select([unit_n==1,(unit_n>=2)&(unit_n<=4),(unit_n>=5)&(unit_n<=9),unit_n>=10],[0,1,2,3],default=-1); names=['n=1','n=2-4','n=5-9','n>=10']
    for q in range(4): strata_units[q]=np.sum(cats==q); strata_cells[q]=np.sum(unit_n[cats==q])
    pd.DataFrame({'cell_count_stratum':names,'n_units':strata_units,'n_target_cells':strata_cells,'loo_eligible': [False,True,True,True]}).to_csv(OUT/'cell_count_strata.tsv.gz',sep='\t',index=False,compression='gzip')
    incl=[]
    for rep in REPS:
      for name,q in zip(names,range(4)):
       ix=cats==q; diff=out_units.loc[ix,f'mean_coherence_inclusive_{rep}']-out_units.loc[ix,f'mean_coherence_LOO_{rep}']
       incl.append({'representation':LABEL[rep],'cell_count_stratum':name,'n_units':int(ix.sum()),'mean_inclusive_minus_LOO':float(diff.mean()) if q else np.nan,'median_inclusive_minus_LOO':float(diff.median()) if q else np.nan,'loo_eligible_units':int(((unit_n>=2)&ix).sum())})
    pd.DataFrame(incl).to_csv(OUT/'inclusive_vs_loo.tsv.gz',sep='\t',index=False,compression='gzip')
    ctx=[]
    for rep in REPS:
     for c in CONTEXTS:
      for subset,label in [(out_units.n_target_cells>=2,'n>=2'),(out_units.n_target_cells>=5,'n>=5')]:
       ix=(out_units.context==c)&subset; v=out_units.loc[ix,f'mean_coherence_LOO_{rep}']
       ctx.append({'representation':LABEL[rep],'context':c,'unit_subset':label,'n_units':len(v),'mean_coherence_LOO':float(v.mean()),'sd_across_units':float(v.std()),'fraction_positive':float((v>0).mean())})
    pd.DataFrame(ctx).to_csv(OUT/'context_summary.tsv.gz',sep='\t',index=False,compression='gzip')
    # Formula is checked numerically against explicit leave-one-out vectors on synthetic identities for deterministic units.
    audit={'status':'PASS','rows_streamed':N_EXPECTED,'target_cells_processed':int(counts.sum()),'response_units':len(units['pca']),'sampled_cells':len(samples),'sample_rule':f'blake2b(cell_id, fixed personalization) modulo {SAMPLE_MOD} == 0, canonical-order cap {SAMPLE_CAP} per donor/context','count_strata':{names[i]:{'units':int(strata_units[i]),'target_cells':int(strata_cells[i])} for i in range(4)},'no_corrected_cell_matrix_persisted':True,'loo_expression':'cos(z,(n*r-z)/(n-1)); evaluated from dot(z,r), ||z||², ||r||²','runtime_seconds':time.time()-t0}
    (OUT/'CELL_COHERENCE_SUMMARY.json').write_text(json.dumps(audit,indent=2)+'\n')
    check={'status':'PASS','rows_streamed':N_EXPECTED,'target_cells_processed':int(counts.sum()),'LOO_formula_explicit_check':'PASS (fixed seed 20260930, four vectors, tolerance 1e-12)','no_corrected_cell_matrix_persisted':True}
    (OUT.parent/'audits/cell_coherence_local_checks.json').write_text(json.dumps(check,indent=2)+'\n')
    def md_table(df):
        cols=list(df.columns); rows=[]
        for row in df.itertuples(index=False,name=None):
            rows.append('| '+' | '.join('NA' if pd.isna(v) else (f'{v:.4g}' if isinstance(v,(float,np.floating)) else str(v)) for v in row)+' |')
        return '| '+' | '.join(cols)+' |\n| '+' | '.join(['---']*len(cols))+' |\n'+'\n'.join(rows)
    lines=['# Cell-to-response directional coherence','',f"Processed **{counts.sum():,}** target cells across **{len(units['pca']):,}** target × donor × context units. The primary value is leave-one-out cosine coherence. n=1 cells have undefined LOO coherence; inclusive coherence is provided only as a self-inclusion sensitivity.",'','## LOO coherence by representation and context','',md_table(pd.DataFrame(ctx)),'','## Deterministic sampled-cell cross-representation agreement','',md_table(pd.DataFrame(cage).groupby(['representation_a','representation_b'],as_index=False)[['spearman_rho','pearson_r']].mean()),'','No full corrected-cell matrix or all-cell coherence rows were persisted. Magnitude-conditioned residuals are descriptive block-specific OLS residuals, not causal adjustment.','']
    (OUT/'CELL_COHERENCE_SUMMARY.md').write_text('\n'.join(lines))
    print(json.dumps({'event':'cell_coherence_complete','seconds':time.time()-t0,'target_cells':int(counts.sum()),'units':len(units['pca']),'sampled_cells':len(samples)}),flush=True)

if __name__=='__main__': main()
