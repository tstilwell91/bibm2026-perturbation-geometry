#!/usr/bin/env python3
"""Accepted-paper cell-coherence Spearman: paired target-cluster bootstrap."""
import argparse, hashlib, json, platform, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import scipy
from bootstrap_coherence import rank_spec,weighted_ranks,matrix_corr,BASE,REPS,LABEL,DONORS,CONTEXTS,PAIRS
OUT=Path('results/camera_ready_robustness_20261108/bootstrap/sampled_cell')

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--replicates',type=int,default=1000); parser.add_argument('--seed',type=int,default=20261108); args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True); start=time.monotonic()
    source=BASE/'sampled_cell_coherence.parquet'; df=pd.read_parquet(source)
    assert len(df)==1166740 and not df.canonical_row.duplicated().any()
    assert not df.cell_id.duplicated().any()
    ref=pd.read_csv('source/derived/downstream/response_construction_v1/pca/response_units.tsv.gz',sep='\t')
    keys=['target','donor','context']; assert len(ref)==144368 and not ref.duplicated(keys).any()
    aligned=df.merge(ref[keys+['n_target_cells']],on=keys,validate='many_to_one')
    assert len(aligned)==len(df) and (aligned.n_cells_in_response_unit==aligned.n_target_cells).all()
    targets=sorted(ref.target.unique()); assert len(targets)==12604
    target_id={t:i for i,t in enumerate(targets)}
    original=pd.read_csv(BASE/'cross_representation_cell_agreement.tsv.gz',sep='\t')
    blocks=[]; point=[]; errors=[]; unitcounts=[]
    for donor in DONORS:
      for context in CONTEXTS:
        b=df[(df.donor==donor)&(df.context==context)]
        vals=b[[f'coherence_LOO_{r}' for r in REPS]].to_numpy(dtype=np.float64)
        ok=np.isfinite(vals).all(axis=1)
        assert np.array_equal(ok,b.n_cells_in_response_unit.to_numpy()>=2)
        b=b.loc[ok]; vals=vals[ok]
        idx=np.array([target_id[t] for t in b.target],dtype=np.int64)
        specs=[rank_spec(vals[:,r]) for r in range(4)]
        w=np.ones(len(b),dtype=np.int64)
        ranks=np.array([weighted_ranks(s,w) for s in specs]); corr=matrix_corr(ranks,w)
        for k,(a,c) in enumerate(PAIRS):
            old=original[(original.donor==donor)&(original.context==context)&(original.representation_a==LABEL[a])&(original.representation_b==LABEL[c])].iloc[0]
            assert len(b)==old.n_cells
            errors.append(abs(corr[k]-old.spearman_rho))
        blocks.append((donor,context,idx,specs,vals)); point.append(corr); unitcounts.append(len(b))
    assert max(errors)<1e-12
    point=np.array(point)
    assert abs(np.median(point[:,3])-.386934)<1e-6 and abs(np.median(point[:,2])-.835352)<1e-6
    rng=np.random.default_rng(args.seed)
    test=rng.integers(0,4,size=len(targets)); w=test[blocks[0][2]]
    ranks=np.array([weighted_ranks(s,w) for s in blocks[0][3]])
    actual=matrix_corr(ranks,w)
    expanded=np.repeat(np.arange(len(w)),w)
    expected=np.array([spearmanr(blocks[0][4][expanded,a],blocks[0][4][expanded,b]).statistic for a,b in PAIRS])
    validation_error=float(np.max(abs(actual-expected))); assert validation_error<1e-12
    rng=np.random.default_rng(args.seed); values=np.empty((args.replicates,12,6)); pilot_time=None
    for replicate in range(args.replicates):
        counts=np.bincount(rng.integers(0,len(targets),size=len(targets)),minlength=len(targets))
        assert counts.sum()==len(targets)
        for k,(_,_,idx,specs,_) in enumerate(blocks):
            weights=counts[idx]
            ranks=np.array([weighted_ranks(s,weights) for s in specs])
            values[replicate,k]=matrix_corr(ranks,weights)
        if replicate==199:
            pilot_time=time.monotonic()-start; print(f'200 replicates completed in {pilot_time:.3f}s',flush=True)
        if (replicate+1)%100==0:
            print(f'{replicate+1} complete, elapsed {time.monotonic()-start:.3f}s',flush=True)
    assert np.isfinite(values).all(); rows=[]
    for context in ['overall']+CONTEXTS:
        select=np.arange(12) if context=='overall' else np.array([k for k,b in enumerate(blocks) if b[1]==context])
        # Accepted manuscript headline uses median over donor/context correlations.
        est=np.median(point[select],axis=0); boot=np.median(values[:,select],axis=1)
        for k,(a,b) in enumerate(PAIRS):
            lower,upper=np.quantile(boot[:,k],[.025,.975])
            rows.append(dict(metric='sampled_cell_coherence_rank_spearman_median',comparison=f'{LABEL[a]} vs {LABEL[b]}',context=context,point_estimate=est[k],bootstrap_lower_95=lower,bootstrap_upper_95=upper,n_valid_replicates=args.replicates,resampling_method='paired target-cluster bootstrap; all sampled cells and donor/context blocks carried; integer target multiplicities; exact weighted midranks; median block statistic',seed=args.seed,estimand='accepted-paper median sampled-cell LOO-coherence rank agreement conditional on observed donors, fixed sampled cells, and frozen response estimates'))
    pd.DataFrame(rows).to_csv(OUT/'bootstrap_ci.csv',index=False)
    np.savez_compressed(OUT/'bootstrap_replicates.npz',values=values,point=point,contexts=np.array(CONTEXTS),donors=np.array(DONORS),pairs=np.array([[LABEL[a],LABEL[b]] for a,b in PAIRS]))
    audit=dict(status='PASS',source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),reference=str(BASE/'cross_representation_cell_agreement.tsv.gz'),sampled_cells=len(df),loo_eligible_sampled_cells=sum(unitcounts),canonical_response_units=len(ref),canonical_targets=len(targets),max_original_reproduction_error=max(errors),max_expanded_multiplicity_validation_error=validation_error,seed=args.seed,replicates=args.replicates,pilot_200_seconds=pilot_time,runtime_seconds=time.monotonic()-start,python=sys.version,executable=sys.executable,numpy=np.__version__,pandas=pd.__version__,scipy=scipy.__version__,platform=platform.platform(),resampling='global canonical perturbation-target identities with replacement; carry all target sampled cells and all observed donor/context blocks together; paired across representations; duplicated selections represented by exact frequency weights; original singleton LOO exclusion',limitations=['Conditional on four observed donors, fixed reproducible cell sample, frozen per-cell LOO coherences and NTC corrections.','Not an interval for donor population, underlying cell/NTC sampling, or model-training uncertainty.','Assumes perturbation targets exchangeable; does not account for gene-family/pathway dependence.','Whole-map rho/O15/edge/community intervals not computed.'])
    (OUT/'bootstrap_audit.json').write_text(json.dumps(audit,indent=2)+'\n'); print(json.dumps(audit,indent=2),flush=True)
if __name__=='__main__': main()
