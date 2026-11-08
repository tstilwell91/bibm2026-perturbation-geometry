#!/usr/bin/env python3
"""Paired perturbation-target cluster bootstrap of original LOO-coherence ranks.

Frozen response-unit coherence values; targets are resampled across the union of
all targets, carrying all donor/context observations and all representations.
Repeated target draws are integer frequency weights, NOT deduplicated targets.
Exact weighted midranks equal scipy.stats.spearmanr on expanded repeated rows.
"""
import argparse, hashlib, itertools, json, platform, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import scipy

BASE=Path('source/derived/downstream/response_prebiology_v1/cell_coherence')
OUT=Path('results/camera_ready_robustness_20261108/bootstrap')
REPS=['pca','scvi','geneformer','state']
LABEL=['PCA','scVI','Geneformer','STATE']
DONORS=['D1','D2','D3','D4']
CONTEXTS=['Rest','Stim8hr','Stim48hr']
PAIRS=list(itertools.combinations(range(4),2))

def rank_spec(x):
    order=np.argsort(x,kind='stable'); s=x[order]
    starts=np.r_[0,1+np.flatnonzero(s[1:]!=s[:-1])]
    sizes=np.diff(np.r_[starts,len(s)])
    return order,starts,sizes

def weighted_ranks(spec,w):
    order,starts,sizes=spec
    group_w=np.add.reduceat(w[order],starts)
    # Frequency weighted average ranks, including correct tied observations.
    group_r=np.cumsum(group_w)-(group_w-1)/2
    result=np.empty(len(w)); result[order]=np.repeat(group_r,sizes)
    return result

def matrix_corr(ranks,w):
    total=w.sum(); centered=ranks-(ranks@w/total)[:,None]
    cov=(centered*w)@centered.T
    sd=np.sqrt(np.diag(cov))
    corr=cov/np.outer(sd,sd)
    return np.array([corr[a,b] for a,b in PAIRS])

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--replicates',type=int,default=1000); parser.add_argument('--seed',type=int,default=20261108); args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True); start=time.monotonic()
    path=BASE/'target_unit_coherence.parquet'; df=pd.read_parquet(path)
    assert len(df)==144368 and not df.duplicated(['target','donor','context']).any()
    ref=pd.read_csv('source/derived/downstream/response_construction_v1/pca/response_units.tsv.gz',sep='\t')
    merged=df.merge(ref[['target','donor','context','n_target_cells']],on=['target','donor','context'],suffixes=('','_ref'),validate='one_to_one')
    assert len(merged)==144368 and (merged.n_target_cells==merged.n_target_cells_ref).all()
    targets=sorted(df.target.unique()); target_id={t:i for i,t in enumerate(targets)}
    assert len(targets)==12604
    original=pd.read_csv(BASE/'cross_representation_unit_agreement.tsv.gz',sep='\t')
    blocks=[]; point=[]; errors=[]
    for donor in DONORS:
      for context in CONTEXTS:
        b=df[(df.donor==donor)&(df.context==context)].copy()
        vals=b[[f'mean_coherence_LOO_{r}' for r in REPS]].to_numpy()
        ok=np.isfinite(vals).all(axis=1)
        # Original function corr excludes n=1 NaNs. All four reps share eligibility.
        assert np.array_equal(ok,b.n_target_cells.to_numpy()>=2)
        b=b.loc[ok]; vals=vals[ok]
        idx=np.array([target_id[t] for t in b.target],dtype=np.int64)
        specs=[rank_spec(vals[:,r]) for r in range(4)]
        w=np.ones(len(b),dtype=np.int64)
        rr=np.array([weighted_ranks(s,w) for s in specs]); corr=matrix_corr(rr,w)
        for k,(a,c) in enumerate(PAIRS):
            old=original[(original.donor==donor)&(original.context==context)&(original.representation_a==LABEL[a])&(original.representation_b==LABEL[c])].iloc[0]
            errors.append(abs(corr[k]-old.mean_coherence_LOO_spearman))
        blocks.append((donor,context,idx,specs,vals)); point.append(corr)
    assert max(errors)<1e-12
    # Validate multiplicity + tie implementation against direct expansion.
    rng=np.random.default_rng(args.seed)
    test=rng.integers(0,4,size=len(blocks[0][2])); test[:4]=[0,1,2,3]
    ranks=np.array([weighted_ranks(s,test) for s in blocks[0][3]])
    test_result=matrix_corr(ranks,test)
    expanded=np.repeat(np.arange(len(test)),test)
    expected=np.array([spearmanr(blocks[0][4][expanded,a],blocks[0][4][expanded,b]).statistic for a,b in PAIRS])
    assert np.max(abs(test_result-expected))<1e-12
    # Independent RNG reset: validation does not consume bootstrap draws.
    rng=np.random.default_rng(args.seed); values=np.empty((args.replicates,12,6))
    pilot_time=None
    for replicate in range(args.replicates):
        draw=rng.integers(0,len(targets),size=len(targets))
        counts=np.bincount(draw,minlength=len(targets))
        assert counts.sum()==len(targets)
        for k,(_,_,idx,specs,_) in enumerate(blocks):
            weights=counts[idx]
            ranks=np.array([weighted_ranks(s,weights) for s in specs])
            values[replicate,k]=matrix_corr(ranks,weights)
        if replicate==199:
            pilot_time=time.monotonic()-start
            print(f'200 replicates completed in {pilot_time:.3f}s',flush=True)
    assert np.isfinite(values).all()
    point=np.array(point); rows=[]
    for context in ['overall']+CONTEXTS:
        select=np.arange(12) if context=='overall' else np.array([k for k,b in enumerate(blocks) if b[1]==context])
        est=point[select].mean(axis=0); boot=values[:,select].mean(axis=1)
        for k,(a,b) in enumerate(PAIRS):
            lower,upper=np.quantile(boot[:,k],[.025,.975])
            rows.append(dict(metric='coherence_rank_spearman',comparison=f'{LABEL[a]} vs {LABEL[b]}',context=context,point_estimate=est[k],bootstrap_lower_95=lower,bootstrap_upper_95=upper,n_valid_replicates=args.replicates,resampling_method='paired target-cluster bootstrap; all donor/context observations carried; integer multiplicities; exact weighted midranks; equal block mean',seed=args.seed,estimand='frozen unit-mean LOO-coherence rank agreement conditional on observed donors and response estimates'))
    pd.DataFrame(rows).to_csv(OUT/'bootstrap_ci.csv',index=False)
    np.savez_compressed(OUT/'coherence_bootstrap_replicates.npz',values=values,point=point,contexts=np.array(CONTEXTS),donors=np.array(DONORS),pairs=np.array([[LABEL[a],LABEL[b]] for a,b in PAIRS]))
    audit=dict(status='PASS',source=str(path),source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),reference=str(BASE/'cross_representation_unit_agreement.tsv.gz'),response_units=len(df),targets=len(targets),loo_eligible_units=int((df.n_target_cells>=2).sum()),max_original_reproduction_error=max(errors),max_expanded_multiplicity_validation_error=float(np.max(abs(test_result-expected))),seed=args.seed,replicates=args.replicates,pilot_200_seconds=pilot_time,runtime_seconds=time.monotonic()-start,python=sys.version,executable=sys.executable,numpy=np.__version__,pandas=pd.__version__,scipy=scipy.__version__,platform=platform.platform(),resampling='targets with replacement from full 12604-target union; missing donor/context rows remain missing; n>=2 original LOO eligibility; same multiplicities across all representations and blocks; duplicated selections retained as frequency weights',limitations=['Conditional on four observed donors; does not estimate donor-population uncertainty.','Conditional on frozen cell means and matched NTC corrections; does not estimate cell/NTC sampling or model-training uncertainty.','Perturbation targets assumed exchangeable for target-superpopulation inference; target-cluster bootstrap does not account for higher-level gene families.','Only raw LOO coherence rank agreement receives intervals. Whole-map rho/O15/edge/community and residualized coherence intervals not computed.'])
    (OUT/'bootstrap_audit.json').write_text(json.dumps(audit,indent=2)+'\n'); print(json.dumps(audit,indent=2),flush=True)
if __name__=='__main__': main()
