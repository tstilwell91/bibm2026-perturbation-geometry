"""Read-only exact whole-map and cached LOO coherence support sensitivity."""
from pathlib import Path
import hashlib, itertools, json, time, platform, sys, argparse
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import rankdata, spearmanr
import scipy, pyarrow

BASE=Path('source/derived/downstream')
OUT=Path('results/camera_ready_robustness_20261108/global')
REPS=['pca','scvi','geneformer','state']
NAMES=['PCA','scVI','Geneformer V2-316M','STATE SE-600M']
PAIRS=list(itertools.combinations(range(4),2))
def log(s): print(time.strftime('%H:%M:%S'),s,flush=True)
def metadata():
    cols=['response_unit','target','target_gene_id','donor','context','n_target_cells','contributing_cell_ids_sha256']
    ref=None
    for rep in REPS:
        df=pd.read_csv(BASE/'response_construction_v1'/rep/'response_units.tsv.gz',sep='\t',usecols=cols)
        assert len(df)==144368 and not df.duplicated(['target','donor','context']).any()
        if ref is None: ref=df
        else: pd.testing.assert_frame_equal(ref,df)
    ref['keep_ge10']=ref.n_target_cells>=10
    ref.to_csv(OUT/'unit_mask.csv.gz',index=False)
    return ref
def centered_rank(v,path):
    # Keep completed ranks disk-backed so another representation's sort does
    # not retain all prior rank arrays in anonymous RAM.
    r=rankdata(v,method='average'); r-=r.mean(); r/=np.linalg.norm(r)
    temporary=path.with_suffix('.tmp.npy'); np.save(temporary,r); temporary.replace(path)
    del r
    return np.load(path,mmap_mode='r')
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--block',default='all'); args=ap.parse_args()
    OUT.mkdir(exist_ok=True,parents=True); meta=metadata()
    original=pd.read_csv(BASE/'response_geometry_v1/cross_representation/agreement_by_donor_context.tsv',sep='\t')
    blocks=list(meta[['donor','context']].drop_duplicates().itertuples(index=False,name=None))
    for donor,context in blocks:
        if args.block!='all' and args.block not in (donor,f'{donor}_{context}'): continue
        dest=OUT/f'rho_{donor}_{context}.csv'
        if dest.exists():
            checkpoint=pd.read_csv(dest)
            assert len(checkpoint)==6 and checkpoint.donor.eq(donor).all() and checkpoint.context.eq(context).all()
            assert checkpoint.original_reproduction_error.max()<1e-12
            assert not checkpoint.duplicated(['representation_a','representation_b']).any()
            continue
        started=time.time(); m=meta[(meta.donor==donor)&(meta.context==context)].sort_values('target')
        targets=m.target.tolist(); th=hashlib.sha256('\n'.join(targets).encode()).hexdigest()
        orig=original[(original.donor==donor)&(original.context==context)]
        assert orig.target_order_sha256.eq(th).all()
        n=len(m); keep=m.keep_ge10.to_numpy(); mask=np.concatenate([keep[i]&keep[i+1:] for i in range(n-1)])
        n_ge10=int(keep.sum()); assert mask.sum()==n_ge10*(n_ge10-1)//2
        full=[]; filtered=[]
        for rep in REPS:
            p=BASE/'response_geometry_v1'/rep/'cosine_pairs'/f'{donor}_{context}.parquet'
            pf=pq.ParquetFile(p); vals=pf.read(columns=['cosine_similarity']).column(0).to_numpy()
            assert len(vals)==n*(n-1)//2 and np.isfinite(vals).all()
            # The frozen file order is the verified lexicographic upper triangle; verify
            # row-group endpoints against that ordering, independently of cosine values.
            first=pf.read_row_group(0,columns=['target_a','target_b']).slice(0,1).to_pydict()
            last_table=pf.read_row_group(pf.num_row_groups-1,columns=['target_a','target_b']); last=last_table.slice(last_table.num_rows-1,1).to_pydict()
            assert first['target_a']==[targets[0]] and first['target_b']==[targets[1]]
            assert last['target_a']==[targets[-2]] and last['target_b']==[targets[-1]]
            full.append(centered_rank(vals,OUT/f'rank_{donor}_{context}_{rep}_full.npy'))
            filtered.append(centered_rank(vals[mask],OUT/f'rank_{donor}_{context}_{rep}_ge10.npy'))
            del vals; log(f'{donor} {context} {rep} ranked ({time.time()-started:.1f}s)')
        rows=[]
        for a,b in PAIRS:
            old=float(full[a]@full[b]); new=float(filtered[a]@filtered[b])
            saved=float(orig[(orig.representation_a==NAMES[a])&(orig.representation_b==NAMES[b])].spearman_rho.iloc[0])
            assert abs(old-saved)<1e-12,(old,saved)
            rows.append(dict(metric='whole_map_spearman_rho',donor=donor,context=context,representation_a=NAMES[a],representation_b=NAMES[b],original=old,filtered_ge10=new,change=new-old,absolute_change=abs(new-old),n_original=n,n_ge10=n_ge10,original_reproduction_error=abs(old-saved)))
        pd.DataFrame(rows).to_csv(dest,index=False); del full,filtered,mask
        for p in OUT.glob(f'rank_{donor}_{context}_*.npy'): p.unlink()
        log(f'completed {donor} {context}: {time.time()-started:.1f}s')
    coherence(meta)
    (OUT/'versions.json').write_text(json.dumps(dict(python=sys.version,numpy=np.__version__,pandas=pd.__version__,scipy=scipy.__version__,pyarrow=pyarrow.__version__,platform=platform.platform()),indent=2))
def coherence(meta):
    src=BASE/'response_prebiology_v1/cell_coherence'; co=pd.read_parquet(src/'target_unit_coherence.parquet')
    pd.testing.assert_frame_equal(co[['target','donor','context','n_target_cells']].reset_index(drop=True),meta[['target','donor','context','n_target_cells']].reset_index(drop=True))
    orig=pd.read_csv(src/'cross_representation_unit_agreement.tsv.gz',sep='\t'); rows=[]
    # Existing coherence ranking agreement is Spearman of per-unit mean LOO coherence.
    # n=1 is excluded by the original finite-pair convention.
    for (d,c),g in co.groupby(['donor','context']):
        for a,b in PAIRS:
            x=g[f'mean_coherence_LOO_{REPS[a]}'].to_numpy(); y=g[f'mean_coherence_LOO_{REPS[b]}'].to_numpy()
            ok=np.isfinite(x)&np.isfinite(y); fil=ok&(g.n_target_cells.to_numpy()>=10)
            old=float(spearmanr(x[ok],y[ok]).statistic); new=float(spearmanr(x[fil],y[fil]).statistic)
            labels={'pca':'PCA','scvi':'scVI','geneformer':'Geneformer','state':'STATE'}
            saved=orig[(orig.donor==d)&(orig.context==c)&(orig.representation_a==labels[REPS[a]])&(orig.representation_b==labels[REPS[b]])].mean_coherence_LOO_spearman.iloc[0]
            assert abs(old-saved)<1e-12
            rows.append(dict(metric='mean_LOO_coherence_rank_spearman',donor=d,context=c,representation_a=NAMES[a],representation_b=NAMES[b],original=old,filtered_ge10=new,change=new-old,absolute_change=abs(new-old),n_original=int(ok.sum()),n_ge10=int(fil.sum()),original_reproduction_error=abs(old-saved)))
    pd.DataFrame(rows).to_csv(OUT/'coherence_rank_sensitivity.csv',index=False)
    summary=[]; ctx=pd.read_csv(src/'context_summary.tsv.gz',sep='\t')
    for context,g in co.groupby('context'):
        for i,rep in enumerate(REPS):
            v=g[f'mean_coherence_LOO_{rep}']; old=float(v[g.n_target_cells>=2].mean()); new=float(v[g.n_target_cells>=10].mean())
            label={'pca':'PCA','scvi':'scVI','geneformer':'Geneformer','state':'STATE'}[rep]
            saved=ctx[(ctx.context==context)&(ctx.representation==label)&(ctx.unit_subset=='n>=2')].mean_coherence_LOO.iloc[0]
            assert abs(old-saved)<1e-12
            summary.append(dict(context=context,representation=NAMES[i],original_mean_LOO=old,ge10_mean_LOO=new,change=new-old))
    s=pd.DataFrame(summary); s['original_rank']=s.groupby('context').original_mean_LOO.rank(ascending=False); s['ge10_rank']=s.groupby('context').ge10_mean_LOO.rank(ascending=False)
    s.to_csv(OUT/'coherence_representation_rank.csv',index=False)
    sm=pd.read_parquet(src/'sampled_cell_coherence.parquet')
    # This deterministic frozen sample, rather than unit-mean ranks, is the
    # accepted paper's primary cell-to-response coherence rank comparator.
    cell_orig=pd.read_csv(src/'cross_representation_cell_agreement.tsv.gz',sep='\t')
    cell_rows=[]
    counts=meta.set_index(['target','donor','context']).n_target_cells
    if 'target' in sm:
        mapped=pd.MultiIndex.from_frame(sm[['target','donor','context']]).map(counts)
        assert np.array_equal(np.asarray(mapped),sm.n_cells_in_response_unit.to_numpy())
    for (d,c),g in sm.groupby(['donor','context']):
        for a,b in PAIRS:
            x=g[f'coherence_LOO_{REPS[a]}'].to_numpy(); y=g[f'coherence_LOO_{REPS[b]}'].to_numpy()
            ok=np.isfinite(x)&np.isfinite(y); fil=ok&(g.n_cells_in_response_unit.to_numpy()>=10)
            old=float(spearmanr(x[ok],y[ok]).statistic); new=float(spearmanr(x[fil],y[fil]).statistic)
            labels={'pca':'PCA','scvi':'scVI','geneformer':'Geneformer','state':'STATE'}
            saved=cell_orig[(cell_orig.donor==d)&(cell_orig.context==c)&(cell_orig.representation_a==labels[REPS[a]])&(cell_orig.representation_b==labels[REPS[b]])].spearman_rho.iloc[0]
            assert abs(old-saved)<1e-12
            cell_rows.append(dict(metric='sampled_cell_LOO_coherence_rank_spearman',donor=d,context=c,representation_a=NAMES[a],representation_b=NAMES[b],original=old,filtered_ge10=new,change=new-old,absolute_change=abs(new-old),n_original=int(ok.sum()),n_ge10=int(fil.sum()),original_reproduction_error=abs(old-saved)))
    pd.DataFrame(cell_rows).to_csv(OUT/'sampled_cell_coherence_rank_sensitivity.csv',index=False)
if __name__=='__main__': main()
