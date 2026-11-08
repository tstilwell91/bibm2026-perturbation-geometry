"""Two-pass V31 split responses using existing labels and parallel accumulation."""
import argparse, importlib.util, json, time, hashlib, os
from pathlib import Path
import numpy as np
from numba import set_num_threads
from kernels import accumulate

ROOT=Path('.').resolve()
OUT=ROOT/'results/bibm_extensions_20261005/agent_a_reliability'
spec=importlib.util.spec_from_file_location('original_split',Path(__file__).resolve().parent/'split_half.py')
original=importlib.util.module_from_spec(spec); spec.loader.exec_module(original)
b=original.b

def main():
    p=argparse.ArgumentParser(); p.add_argument('--representation',choices=['geneformer','state'],required=True)
    p.add_argument('--threads',type=int,default=20); a=p.parse_args()
    start=time.time(); set_num_threads(a.threads)
    split_path=OUT/'split_cells.npy'
    if not split_path.exists(): raise RuntimeError('Existing frozen assignments required; never generate splits')
    labels=np.load(split_path,mmap_mode='r',allow_pickle=False)
    if labels.shape!=(20,b.N_EXPECTED) or labels.dtype!=np.int8: raise RuntimeError('Split assignment shape/dtype mismatch')
    h=hashlib.sha256()
    with split_path.open('rb') as f:
        for data in iter(lambda:f.read(8<<20),b''): h.update(data)
    provenance=json.loads((OUT/'SPLIT_PROVENANCE.json').read_text())
    if h.hexdigest()!=provenance['split_assignment_sha256']: raise RuntimeError('Frozen split checksum mismatch')
    print(json.dumps({'event':'loading_validated_index','representation':a.representation}),flush=True)
    meta=b.load_and_validate_index(); b.meta_source_ranges=meta['source_ranges']
    rep=next(r for r in b.REPRESENTATIONS if r['out']==a.representation)
    source_audit=b.external_rep_checks(rep)
    nu=len(meta['unit_rows']); ns=len(meta['stratum_rows']); dim=rep['dim']
    sums=np.zeros((20,2,nu,dim),np.float64); counts=np.zeros((20,2,nu),np.int64)
    cs=np.zeros((20,2,ns,dim),np.float64); cc=np.zeros((20,2,ns),np.int64); cm=np.zeros_like(cs)
    for phase in (0,1):
        if phase==1:
            if np.any(cc==0): raise RuntimeError('Unavailable half-specific NTC centroid')
            np.divide(cs,cc[:,:,:,None],out=cm)
        t=time.time(); last=t; seen=0
        for st,en,x in b.iter_blocks(rep,meta['row_maps'][rep['name']]):
            if st!=seen or x.shape!=(en-st,dim) or x.dtype!=np.float32 or not np.isfinite(x).all():
                raise RuntimeError(f'Frozen block integrity failure {st}:{en}')
            accumulate(x,labels[:,st:en],meta['unit_ids'][st:en],meta['stratum_ids'][st:en],meta['is_target'][st:en],sums,counts,cs,cc,cm,phase)
            seen=en
            if time.time()-last>60:
                rate=seen/(time.time()-t)
                print(json.dumps({'event':'aggregation_progress','phase':phase,'cells':seen,'cells_total':b.N_EXPECTED,'cells_per_second':rate,'phase_remaining_seconds':(b.N_EXPECTED-seen)/rate}),flush=True); last=time.time()
        if seen!=b.N_EXPECTED: raise RuntimeError('Incomplete frozen population traversal')
    # Compare counts with authoritative completed PCA split support, without
    # recalculating labels or changing any half/unit support rule.
    expected_target=np.load(OUT/'split_vectors/pca_split_target_support.npy',mmap_mode='r')
    expected_ntc=np.load(OUT/'split_vectors/pca_split_ntc_support.npy',mmap_mode='r')
    if not np.array_equal(counts,expected_target) or not np.array_equal(cc,expected_ntc):
        raise RuntimeError('Split counts disagree with authoritative completed PCA support')
    np.divide(sums,counts[:,:,:,None],out=sums,where=counts[:,:,:,None]>0)
    sums[counts==0]=np.nan
    zero=np.zeros((20,2),np.int64)
    for s in range(20):
        for h0 in range(2):
            for st in range(0,nu,8192):
                v=sums[s,h0,st:st+8192]; finite=np.isfinite(v).all(axis=1)
                if not np.array_equal(finite,counts[s,h0,st:st+8192]>0): raise RuntimeError('Response finiteness/support mismatch')
                zero[s,h0]+=np.count_nonzero(np.linalg.norm(v[finite],axis=1)==0)
    if zero.any(): raise RuntimeError('Zero-norm supported response')
    dest=OUT/f'split_vectors_{a.representation}'
    if dest.exists() and any(dest.iterdir()): raise RuntimeError('Refusing to overwrite existing representation results')
    dest.mkdir(exist_ok=True)
    for suffix,value in [('responses',sums),('target_support',counts),('ntc_support',cc),('valid_unit',counts>0)]:
        tmp=dest/f'{a.representation}_split_{suffix}.npy.tmp'
        with tmp.open('wb') as f: np.save(f,value,allow_pickle=False)
        os.replace(tmp,dest/f'{a.representation}_split_{suffix}.npy')
    audit={'representation':rep['name'],'splits':20,'seed':28091005,'pilot_rows':0,'invalid_unit_halves_due_missing_ntc_centroid':0,'zero_target_support_unit_halves':int((counts==0).sum()),'zero_norm_response_half_by_split':zero.tolist(),'paired_valid_unit_count_by_split':((counts[:,0]>0)&(counts[:,1]>0)).sum(axis=1).tolist(),'target_cells_per_split_half':counts.sum(axis=2).tolist(),'ntc_cells_per_split_half':cc.sum(axis=2).tolist(),'unit_half_target_support_minmax':[int(counts[counts>0].min()),int(counts.max())],'ntc_half_stratum_support_minmax':[int(cc.min()),int(cc.max())],'source_audit_status':source_audit.get('audit_status'),'implementation':'numba split-owned parallel sequential cell accumulation; fastmath=False','threads':a.threads,'split_sha256':h.hexdigest(),'elapsed_seconds':time.time()-start}
    original.write_json(dest/f'{a.representation}_split_audit.json',audit)
    original.write_json(dest/f'{a.representation}_split_source_audit.json',source_audit)
    original.write_json(OUT/f'RUN_METADATA_split_vectors_{a.representation}.json',audit)
    print(json.dumps({'event':'representation_complete',**audit}),flush=True)

if __name__=='__main__': main()
