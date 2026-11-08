#!/usr/bin/env python3
"""Summarize V31 cell split-half reproducibility and sampled map/local geometry."""
import csv, gzip, json, itertools, time
from pathlib import Path
import multiprocessing, os
from threadpoolctl import threadpool_limits
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from slab_access import SlabAccess
import numpy as np
from scipy.stats import spearmanr, pearsonr

ROOT=Path('.').resolve()
OUT=ROOT/'results/bibm_extensions_20261005/agent_a_reliability'
RESP=ROOT/'source/derived/downstream/response_construction_v1'
REPS=[('PCA','pca',50),('scVI','scvi',50),('Geneformer V2-316M','geneformer',1152),('STATE SE-600M','state',2058)]
VEC_DIRS={'PCA':OUT/'split_vectors','scVI':OUT/'split_vectors_scvi','Geneformer V2-316M':OUT/'split_vectors_geneformer','STATE SE-600M':OUT/'split_vectors_state'}
KS=[5,10,15,20,30,50,100]
SEED=731005
PAIR_N=100_000
FOCAL_N=500

def load_units():
    p=RESP/'pca/response_units.tsv.gz'; records=[]
    with gzip.open(p,'rt',newline='') as f:
        for x in csv.DictReader(f,delimiter='\t'):
            lane_counts=[y.split(':') for y in x['target_cells_by_lane'].split(';') if y]
            records.append({'target':x['target'],'donor':x['donor'],'context':x['context'],'support':int(x['n_target_cells']),'unit':int(x['response_unit']),'lane_counts':lane_counts})
    return records

def sampled_pairs(n,rng,count):
    m=n*(n-1)//2; count=min(count,m)
    vals=np.empty(count,dtype=np.int64); have=0
    while have<count:
        draw=rng.integers(0,m,size=max(1024,(count-have)*2),dtype=np.int64)
        draw=np.unique(draw)
        if have:
            draw=draw[~np.isin(draw,vals[:have],assume_unique=False)]
        take=min(len(draw),count-have); vals[have:have+take]=draw[:take]; have+=take
    vals.sort()
    starts=np.arange(n,dtype=np.int64)*(2*n-np.arange(n,dtype=np.int64)-1)//2
    i=np.searchsorted(starts,vals,side='right')-1
    j=i+1+(vals-starts[i])
    return i.astype(np.int32),j.astype(np.int32)

def edge_cosines(x,i,j,chunk=10000):
    out=np.empty(len(i),np.float32)
    for s in range(0,len(i),chunk):
        e=min(len(i),s+chunk); out[s:e]=np.einsum('ij,ij->i',x[i[s:e]],x[j[s:e]],optimize=True)
    return out

def top_ids(scores, targets, focal_local, kmax):
    out=np.empty((len(focal_local),kmax),np.int32)
    for q,row in enumerate(focal_local):
        v=scores[q].copy(); v[row]=-np.inf
        threshold=np.partition(v,-kmax)[-kmax]
        cand=np.flatnonzero(v>=threshold)
        order=cand[np.lexsort((targets[cand],-v[cand]))][:kmax]
        out[q]=order
    return out

def focal_sample(targets,support,rng,n=FOCAL_N):
    n=min(n,len(targets)); order=np.argsort(support,kind='stable'); bins=np.array_split(order,4)
    chosen=[]; each=n//4
    for b in bins:
        take=min(each,len(b)); chosen.extend(rng.choice(b,size=take,replace=False).tolist())
    if len(chosen)<n:
        remain=np.setdiff1d(np.arange(len(targets)),np.asarray(chosen,dtype=np.int64),assume_unique=False)
        chosen.extend(rng.choice(remain,size=n-len(chosen),replace=False).tolist())
    return np.asarray(sorted(chosen),dtype=np.int32)


_CTX = None
def metric_task(task):
    name,sub,dim,split = task
    vectors = _CTX['vectors']
    supports = _CTX['supports']
    common_units = _CTX['common_units']
    ntc_min_support = _CTX['ntc_min_support']
    loc = _CTX['loc']
    common_targets = _CTX['common_targets']
    common_support = _CTX['common_support']
    focal_local = _CTX['focal_local']
    pi = _CTX['pi']
    pj = _CTX['pj']
    donor = _CTX['donor']
    context = _CTX['context']
    rows=[]; support_rows=[]
    A=np.asarray(vectors[name][split,0,common_units,:],dtype=np.float64)
    B=np.asarray(vectors[name][split,1,common_units,:],dtype=np.float64)
    na=np.linalg.norm(A,axis=1); nb=np.linalg.norm(B,axis=1)
    if np.any(na<=0) or np.any(nb<=0): raise RuntimeError(f'zero-norm valid split response in {name}/{donor}/{context}/split{split+1}; revise common support before geometry')
    cosv=np.einsum('ij,ij->i',A,B)/(na*nb)
    target_half=np.asarray(supports[name][split][:,common_units],dtype=np.int32)
    ntc_common=np.min(ntc_min_support[split][:,loc],axis=0)
    An=(A/(na[:,None])).astype(np.float32); Bn=(B/(nb[:,None])).astype(np.float32)
    ecA=edge_cosines(An,pi,pj); ecB=edge_cosines(Bn,pi,pj)
    maprho=float(spearmanr(ecA,ecB).statistic); mappear=float(pearsonr(ecA,ecB).statistic)
    # Per-target rank agreement among sampled incident edges; report only
    # focal targets having >=10 sampled partners.
    incident=[[] for _ in range(len(loc))]
    for e,(i,j) in enumerate(zip(pi,pj)):
        incident[i].append(e); incident[j].append(e)
    trank=[]
    for ix in focal_local:
        edge_ix=incident[ix]
        if len(edge_ix)>=10: trank.append(spearmanr(ecA[edge_ix],ecB[edge_ix]).statistic)
    # Exact self-excluded kNN for a deterministic support-stratified focal
    # subset, against the complete shared target candidate universe.
    sa=An[focal_local]@An.T; sb=Bn[focal_local]@Bn.T
    topa=top_ids(sa,common_targets,focal_local,100); topb=top_ids(sb,common_targets,focal_local,100)
    ok={}
    for k in KS:
        vals=[len(set(topa[q,:k]).intersection(topb[q,:k]))/k for q in range(len(focal_local))]
        ok[k]=float(np.mean(vals))
    mA=na; mB=nb
    row={'representation':name,'donor':donor,'context':context,'split':split+1,'n_common_targets':len(loc),'n_units':len(loc),'n_sampled_pairs':len(pi),'n_focal_targets':len(focal_local),'n_targets_per_target_rank_ge10':len(trank),'vector_cosine_mean':float(np.mean(cosv)),'vector_cosine_median':float(np.median(cosv)),'magnitude_spearman':float(spearmanr(mA,mB).statistic),'magnitude_pearson':float(pearsonr(mA,mB).statistic),'magnitude_abs_log_ratio_median':float(np.median(np.abs(np.log((mA+1e-12)/(mB+1e-12))))),'whole_map_spearman_sampled':maprho,'whole_map_pearson_sampled':mappear,'per_target_sampled_partner_spearman_median':float(np.nanmedian(trank)) if trank else float('nan'),'per_target_sampled_partner_spearman_mean':float(np.nanmean(trank)) if trank else float('nan'),'target_support_median':float(np.median(common_support)),'target_half_min_support_median':float(np.median(np.min(target_half,axis=0))),'ntc_support_min_lane_half_median':float(np.median(ntc_common)),'target_support_reliability_spearman':float(spearmanr(common_support,cosv).statistic),'target_half_support_reliability_spearman':float(spearmanr(np.min(target_half,axis=0),cosv).statistic),'ntc_support_reliability_spearman':float(spearmanr(ntc_common,cosv).statistic)}
    row.update({f'O{k}':v for k,v in ok.items()}); rows.append(row)
    for label,vals in [('target_cell_support',common_support),('matched_NTC_min_lane_half_support',ntc_common)]:
        bins=np.array_split(np.argsort(vals,kind='stable'),5)
        for q,ix in enumerate(bins,1):
            support_rows.append({'representation':name,'donor':donor,'context':context,'split':split+1,'support_metric':label,'support_quintile':q,'n_units':len(ix),'support_median':float(np.median(vals[ix])),'vector_cosine_median':float(np.median(cosv[ix])),'vector_cosine_mean':float(np.mean(cosv[ix])),'vector_cosine_q10':float(np.quantile(cosv[ix],.1)),'vector_cosine_q90':float(np.quantile(cosv[ix],.9))})
    return rows[0], support_rows

def main():
    threadpool_limits(limits=1)
    started=time.time(); records=load_units(); nunit=len(records)
    byblock={}
    for r in records: byblock.setdefault((r['donor'],r['context']),[]).append(r)
    valid={}; vectors={}; supports={}
    for name,sub,dim in REPS:
        vecdir=VEC_DIRS[name]
        valid[name]=np.load(vecdir/f'{sub}_split_valid_unit.npy',mmap_mode='r',allow_pickle=False)
        vectors[name]=SlabAccess(np.load(vecdir/f'{sub}_split_responses.npy',mmap_mode='r',allow_pickle=False))
        supports[name]=np.load(vecdir/f'{sub}_split_target_support.npy',mmap_mode='r',allow_pickle=False)
        if vectors[name].shape!=(20,2,nunit,dim): raise RuntimeError(f'bad response split array shape {name}: {vectors[name].shape}')
    stratum_map={}
    with gzip.open(RESP/'pca/ntc_centroid_units.tsv.gz','rt',newline='') as f:
        for z in csv.DictReader(f,delimiter='\t'): stratum_map[(z['donor'],z['context'],z['lane'])]=int(z['centroid_row'])
    ntc_support=np.load(VEC_DIRS['PCA']/'pca_split_ntc_support.npy',mmap_mode='r',allow_pickle=False)
    rng=np.random.default_rng(SEED); rows=[]; support_rows=[]; sample_manifest=[]; sample_dir=OUT/'pair_samples'; sample_dir.mkdir(exist_ok=True)
    for (donor,context),recs in sorted(byblock.items()):
        recs=sorted(recs,key=lambda z:z['target']); targets=np.asarray([x['target'] for x in recs]); units=np.asarray([x['unit'] for x in recs],np.int32); sup=np.asarray([x['support'] for x in recs])
        shared=np.ones(len(recs),bool)
        for name,_,_ in REPS:
            shared &= np.asarray(valid[name][:,:,units]).all(axis=(0,1))
        loc=np.flatnonzero(shared); common_targets=targets[loc]; common_units=units[loc]; common_support=sup[loc]
        if len(loc)<102: raise RuntimeError(f'insufficient common support for {donor}/{context}: {len(loc)}')
        pi,pj=sampled_pairs(len(loc),rng,PAIR_N)
        focal_local=focal_sample(common_targets,common_support,rng)
        sample_path=sample_dir/f'{donor}_{context}.npz'
        sample_values=dict(targets=common_targets,units=common_units,support=common_support,pair_i=pi,pair_j=pj,focal_local=focal_local)
        if sample_path.exists():
            with np.load(sample_path,allow_pickle=False) as existing:
                if any(not np.array_equal(existing[k],v) for k,v in sample_values.items()):
                    raise RuntimeError(f'Persisted edge/focal sample mismatch: {sample_path}')
        else:
            np.savez_compressed(sample_path,**sample_values)
        ntc_min_support=np.empty((20,2,len(recs)),np.int32)
        for split0 in range(20):
            for q,record in enumerate(recs):
                strata=[stratum_map[(donor,context,lane)] for lane,count in record['lane_counts']]
                ntc_min_support[split0,:,q]=ntc_support[split0][:,strata].min(axis=1)
        sample_manifest.append({'donor':donor,'context':context,'common_targets':len(loc),'sampled_pairs':len(pi),'focal_targets':len(focal_local),'neighbor_universe':'all common targets with both split halves valid in every representation and split'})
        global _CTX
        _CTX = {
            'vectors': vectors,
            'supports': supports,
            'common_units': common_units,
            'ntc_min_support': ntc_min_support,
            'loc': loc,
            'common_targets': common_targets,
            'common_support': common_support,
            'focal_local': focal_local,
            'pi': pi,
            'pj': pj,
            'donor': donor,
            'context': context,
        }
        tasks=[(name,sub,dim,split) for name,sub,dim in REPS for split in range(20)]
        checkpoint_dir=OUT/'metric_checkpoints';checkpoint_dir.mkdir(exist_ok=True)
        checkpoint=checkpoint_dir/f'{donor}_{context}.json'
        if checkpoint.exists():
            saved=json.loads(checkpoint.read_text())
            if saved.get('implementation')!='contiguous_read_v1' or len(saved['rows'])!=80:
                raise RuntimeError(f'Unexpected checkpoint {checkpoint}')
            rows.extend(saved['rows']);support_rows.extend(saved['support_rows'])
        else:
            block_rows=[];block_support=[];block_start=time.time()
            with multiprocessing.get_context('fork').Pool(int(os.environ.get('BIBM_METRIC_WORKERS','16'))) as pool:
                for row, support in pool.imap(metric_task,tasks,chunksize=1):
                    block_rows.append(row);block_support.extend(support)
                    if len(block_rows)%20==0:
                        print(json.dumps({'event':'metric_progress','donor':donor,'context':context,'tasks_complete':len(block_rows),'tasks_total':80,'elapsed_seconds':time.time()-block_start}),flush=True)
            saved={'implementation':'contiguous_read_v1','rows':block_rows,'support_rows':block_support,'elapsed_seconds':time.time()-block_start}
            temp=checkpoint.with_suffix('.json.tmp');temp.write_text(json.dumps(saved)+'\n');os.replace(temp,checkpoint)
            rows.extend(block_rows);support_rows.extend(block_support)
        print(f'completed {donor}/{context}; common={len(loc)} pairs={len(pi)}',flush=True)
    fields=list(rows[0]); out=OUT/'split_reliability_metrics.tsv'
    with out.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,delimiter='\t',lineterminator='\n'); w.writeheader(); w.writerows(rows)
    spath=OUT/'reliability_by_support_quintile.tsv'
    with spath.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(support_rows[0]),delimiter='\t',lineterminator='\n'); w.writeheader(); w.writerows(support_rows)
    (OUT/'split_pair_sample_manifest.json').write_text(json.dumps({'seed':SEED,'pairs_per_block':PAIR_N,'focal_target_n':FOCAL_N,'focal_sampling':'support-sorted quartile strata, uniformly sampled within quartile','pair_sampling':'unique unordered pair IDs uniformly sampled without replacement by rejection from upper-triangle index space','pair_blocks':sample_manifest,'elapsed_seconds':round(time.time()-started,2)},indent=2)+'\n')
    print(json.dumps({'output':str(out),'rows':len(rows),'elapsed_seconds':round(time.time()-started,2)},indent=2))

if __name__=='__main__': main()
