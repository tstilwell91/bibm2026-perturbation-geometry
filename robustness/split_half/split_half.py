#!/usr/bin/env python3
"""V31 cell-level split-half reliability, accumulating 20 partitions in two passes.

The same canonical cell split is used for all four representations. Each target
unit and each NTC matching stratum is split independently; cells are never split
after response aggregation.
"""
from __future__ import annotations
import argparse, csv, gzip, importlib.util, json, os, time
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

ROOT=Path('.').resolve()
OUT=ROOT/'results/bibm_extensions_20261005/agent_a_reliability'
BUILDER=Path(__file__).resolve().parents[2]/'analysis/responses/build_responses.py'
spec=importlib.util.spec_from_file_location('v31builder', BUILDER); b=importlib.util.module_from_spec(spec); spec.loader.exec_module(b)
REPS=b.REPRESENTATIONS

def write_json(p,x):
    tmp=p.with_suffix(p.suffix+'.tmp'); tmp.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n'); os.replace(tmp,p)

def make_splits(meta, nsplits, seed):
    """One Bernoulli partition per cell, stratified within unit / NTC stratum."""
    rng=np.random.default_rng(seed)
    n=b.N_EXPECTED
    split=np.full((nsplits,n),-1,dtype=np.int8)
    # Randomized balanced split independently within each target unit and NTC
    # matching stratum. Singleton groups occupy one half and leave the other
    # unsupported, as required by the cell-level estimator.
    for ids, keep_rows, n_groups in ((meta['unit_ids'],meta['is_target'],len(meta['unit_rows'])),(meta['stratum_ids'],~meta['is_target'],len(meta['stratum_rows']))):
        rows=np.flatnonzero(keep_rows); group_ids=ids[rows]
        order=np.argsort(group_ids,kind='stable'); sorted_ids=group_ids[order]
        starts=np.r_[0,np.flatnonzero(np.diff(sorted_ids))+1]; ends=np.r_[starts[1:],len(order)]
        if len(starts)!=n_groups: raise RuntimeError('Unexpected split group count')
        for a,z in zip(starts,ends):
            ix=rows[order[a:z]]; count=len(ix)
            ranks=np.argsort(rng.random((nsplits,count)),axis=1,kind='stable')
            for s in range(nsplits):
                labels=np.arange(count,dtype=np.int8)%2
                if count==1: labels[0]=rng.integers(0,2)
                split[s,ix[ranks[s]]]=labels
    if np.any(split<0): raise RuntimeError('unassigned canonical cell')
    return split

def run_rep(rep, meta, splits, nsplits, seed, batch, outdir, max_rows=0):
    dim=rep['dim']; nu=len(meta['unit_rows']); nstr=len(meta['stratum_rows'])
    source_audit=b.external_rep_checks(rep)
    # Split × half × stratum/unit × dimension. Float64 mirrors the canonical
    # arithmetic accumulator. Only one representation is resident at a time.
    csum=np.zeros((nsplits,2,nstr,dim),np.float64); ccnt=np.zeros((nsplits,2,nstr),np.int64)
    usum=np.zeros((nsplits,2,nu,dim),np.float64); ucnt=np.zeros((nsplits,2,nu),np.int64); invalid=np.zeros((nsplits,2,nu),bool)
    b.BATCH=batch
    t0=time.time(); seen=0
    for start,end,x in b.iter_blocks(rep,meta['row_maps'][rep['name']]):
        if max_rows and start>=max_rows: break
        if max_rows: end=min(end,max_rows); x=x[:end-start]
        if x.shape!=(end-start,dim) or x.dtype!=np.float32 or not np.isfinite(x).all(): raise RuntimeError(f'Invalid frozen embedding block {rep["name"]} {start}:{end}')
        ids=meta['stratum_ids'][start:end]; target=meta['is_target'][start:end]
        for s in range(nsplits):
            z=splits[s,start:end]; nt=~target
            for h in (0,1):
                m=nt & (z==h)
                if m.any():
                    vals=x[m].astype(np.float64); si=ids[m]
                    ix=np.argsort(si,kind='stable'); ss=si[ix]; starts=np.r_[0,np.flatnonzero(np.diff(ss))+1]; keys=ss[starts]
                    csum[s,h,keys]+=np.add.reduceat(vals[ix],starts,axis=0); ccnt[s,h,keys]+=np.diff(np.r_[starts,len(ss)])
        seen=end
    cmeans=np.divide(csum,ccnt[:,:,:,None],out=np.full_like(csum,np.nan),where=ccnt[:,:,:,None]>0)
    # Second pass: corrected target cells using the split-specific matched NTC centroid.
    for start,end,x in b.iter_blocks(rep,meta['row_maps'][rep['name']]):
        if max_rows and start>=max_rows: break
        if max_rows: end=min(end,max_rows); x=x[:end-start]
        if x.shape!=(end-start,dim) or x.dtype!=np.float32 or not np.isfinite(x).all(): raise RuntimeError(f'Invalid frozen embedding block {rep["name"]} {start}:{end}')
        target=meta['is_target'][start:end]
        if not target.any(): continue
        uids=meta['unit_ids'][start:end]; sids=meta['stratum_ids'][start:end]
        for s in range(nsplits):
            z=splits[s,start:end]
            for h in (0,1):
                m=target & (z==h)
                if not m.any(): continue
                ui=uids[m]; si=sids[m]
                bad=ccnt[s,h,si]==0
                if bad.any(): invalid[s,h,np.unique(ui[bad])]=True
                keep=~bad
                if not keep.any(): continue
                ui=ui[keep]; si=si[keep]; vals=x[m][keep].astype(np.float64)-cmeans[s,h,si]
                ix=np.argsort(ui,kind='stable'); uu=ui[ix]; starts=np.r_[0,np.flatnonzero(np.diff(uu))+1]; keys=uu[starts]
                usum[s,h,keys]+=np.add.reduceat(vals[ix],starts,axis=0); ucnt[s,h,keys]+=np.diff(np.r_[starts,len(uu)])
    np.divide(usum,ucnt[:,:,:,None],out=usum,where=ucnt[:,:,:,None]>0)
    usum[(ucnt==0)|invalid]=np.nan
    # Preserve the canonical builder's float64 arithmetic-mean precision.
    responses=usum
    zero_norm=np.zeros((nsplits,2),np.int64)
    for s in range(nsplits):
        for h in (0,1):
            for st in range(0,nu,8192):
                v=responses[s,h,st:min(nu,st+8192)]
                finite=np.isfinite(v).all(axis=1)
                if finite.any(): zero_norm[s,h]+=np.count_nonzero(np.linalg.norm(v[finite],axis=1)==0)
    audit={"representation":rep['name'],"splits":nsplits,"seed":seed,"pilot_rows":int(max_rows),"invalid_unit_halves_due_missing_ntc_centroid":int(invalid.sum()),"zero_target_support_unit_halves":int((ucnt==0).sum()),"zero_norm_response_half_by_split":zero_norm.tolist(),"paired_valid_unit_count_by_split":[],"target_cells_per_split_half":ucnt.sum(axis=2).tolist(),"ntc_cells_per_split_half":ccnt.sum(axis=2).tolist(),"unit_half_target_support_minmax":[int(ucnt[ucnt>0].min()) if np.any(ucnt>0) else 0,int(ucnt.max())],"ntc_half_stratum_support_minmax":[int(ccnt[ccnt>0].min()) if np.any(ccnt>0) else 0,int(ccnt.max())],"source_audit_status":source_audit.get('audit_status','unknown')}
    for s in range(nsplits): audit['paired_valid_unit_count_by_split'].append(int(np.sum((ucnt[s,0]>0)&(ucnt[s,1]>0)&(~invalid[s,0])&(~invalid[s,1]))))
    if not max_rows:
        expected_target=np.empty((nsplits,2),np.int64); expected_ntc=np.empty((nsplits,2),np.int64)
        for s in range(nsplits):
            for h in (0,1):
                mask=(splits[s]==h)
                expected_target[s,h]=np.count_nonzero(mask&meta['is_target'])
                expected_ntc[s,h]=np.count_nonzero(mask&(~meta['is_target']))
        if not np.array_equal(ucnt.sum(axis=2),expected_target): raise RuntimeError(f'{rep["name"]} target half support does not reconcile to assigned cells')
        if not np.array_equal(ccnt.sum(axis=2),expected_ntc): raise RuntimeError(f'{rep["name"]} NTC half support does not reconcile to assigned cells')
        if invalid.any(): raise RuntimeError(f'{rep["name"]} has target cells with unavailable matched NTC half-centroids')
    outdir.mkdir(parents=True,exist_ok=True)
    np.save(outdir/f'{rep["out"]}_split_responses.npy',responses,allow_pickle=False)
    np.save(outdir/f'{rep["out"]}_split_target_support.npy',ucnt,allow_pickle=False)
    np.save(outdir/f'{rep["out"]}_split_ntc_support.npy',ccnt,allow_pickle=False)
    np.save(outdir/f'{rep["out"]}_split_valid_unit.npy',(~invalid)&(ucnt>0),allow_pickle=False)
    write_json(outdir/f'{rep["out"]}_split_audit.json',audit)
    del csum,ccnt,usum,ucnt
    write_json(outdir/f'{rep["out"]}_split_source_audit.json',source_audit)
    return responses

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--splits',type=int,default=20); ap.add_argument('--seed',type=int,default=28091005); ap.add_argument('--batch',type=int,default=25000); ap.add_argument('--pilot-rows',type=int,default=0); ap.add_argument('--representations',default='PCA,scVI,Geneformer V2-316M,STATE SE-600M'); ap.add_argument('--split-audit-only',action='store_true'); ap.add_argument('--skip-support-audit',action='store_true'); ap.add_argument('--output-subdir',default='split_vectors'); a=ap.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    meta=b.load_and_validate_index(); b.meta_source_ranges=meta['source_ranges']; split_path=OUT/'split_cells.npy'
    if split_path.exists():
        split=np.load(split_path,mmap_mode='r',allow_pickle=False)
        if split.shape!=(a.splits,b.N_EXPECTED) or split.dtype!=np.int8: raise RuntimeError('Shared split assignment shape/dtype mismatch')
    else:
        split=make_splits(meta,a.splits,a.seed)
        with split_path.open('wb') as f: np.save(f,split,allow_pickle=False)
    if not a.skip_support_audit:
        def support(ids, rows, ngroups):
            group=ids[rows]; arr=np.zeros((a.splits,2,ngroups),np.int32)
            for s in range(a.splits):
                for h in (0,1): arr[s,h]=np.bincount(group[split[s,rows]==h],minlength=ngroups)
            return arr
        target=support(meta['unit_ids'],meta['is_target'],len(meta['unit_rows']))
        ntc=support(meta['stratum_ids'],~meta['is_target'],len(meta['stratum_rows']))
        np.save(OUT/'target_half_support.npy',target,allow_pickle=False)
        np.save(OUT/'ntc_half_support.npy',ntc,allow_pickle=False)
        need_ntc=np.zeros_like(target,dtype=bool)
        # A response unit requires every lane-specific NTC centroid touched by its
        # target cells. Mark required stratum/half combinations from split membership.
        for uid, unit in enumerate(meta['unit_rows']):
            strata=[meta['stratum_by_key'][(unit['donor'],unit['context'],lane)] for lane,count in meta['unit_lane_counts'][uid]]
            for s in range(a.splits):
                for h in (0,1):
                    if target[s,h,uid]>0: need_ntc[s,h,uid]=bool(np.all(ntc[s,h,strata]>0))
        support_values=np.asarray([x['n_target_cells'] for x in meta['unit_rows']],dtype=np.int64)
        unit_halves=target[target>0]; ntc_halves=ntc[ntc>0]
        summary={'splits':a.splits,'seed':a.seed,'index_sha256':meta['index_sha256'],'target_cell_support_range':[int(support_values.min()),int(support_values.max())],'target_cell_support_quantiles':{str(q):float(np.percentile(support_values,q)) for q in [1,5,25,50,75,95,99]},'target_half_nonzero_support_range':[int(unit_halves.min()),int(unit_halves.max())],'target_half_nonzero_support_quantiles':{str(q):float(np.percentile(unit_halves,q)) for q in [1,5,25,50,75,95,99]},'ntc_stratum_total_support_range':[int(min(x['n_ntc'] for x in meta['stratum_rows'])),int(max(x['n_ntc'] for x in meta['stratum_rows']))],'ntc_half_nonzero_support_range':[int(ntc_halves.min()),int(ntc_halves.max())], 'target_valid_half_records':int(need_ntc.sum()),'target_nonempty_half_records':int((target>0).sum()),'target_units_valid_both_halves_per_split': [int((need_ntc[s,0]&need_ntc[s,1]).sum()) for s in range(a.splits)],'target_singletons':int(np.sum(support_values==1)),'ntc_singleton_strata':[i for i,x in enumerate(meta['stratum_rows']) if x['n_ntc']==1], 'ntc_empty_half_count':int((ntc==0).sum()),'singleton_target_half_empty_count':int((target==0).sum()),'input_counts':{'cells':b.N_EXPECTED,'target_cells':b.N_TARGET,'ntc_cells':b.N_NTC,'units':len(meta['unit_rows']),'strata':len(meta['stratum_rows'])}}
        write_json(OUT/'split_support_audit.json',summary)
        if a.split_audit_only:
            print(json.dumps(summary,indent=2)); return
    ids=[]
    vector_out=OUT/a.output_subdir
    for r in REPS:
        if r['name'] in a.representations.split(','):
            arr=run_rep(r,meta,split,a.splits,a.seed,a.batch,vector_out,a.pilot_rows); ids.append(r['name']); del arr
    write_json(OUT/f'RUN_METADATA_{a.output_subdir.replace("/","_")}.json',{'seed':a.seed,'splits':a.splits,'batch':a.batch,'pilot_rows':a.pilot_rows,'representations':ids,'source_index':str(b.INDEX.relative_to(ROOT)),'index_sha256':meta['index_sha256'],'method':'For each split, target units and matched NTC strata partitioned independently at cell level; two-pass V31 centroid and corrected target-cell arithmetic mean.','completed_unix':time.time(),'pilot_only':bool(a.pilot_rows)})

if __name__=='__main__': main()
