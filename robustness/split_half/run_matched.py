"""Equal-support, opposite-half geometry from authoritative response arrays only."""
import argparse, ast, hashlib, itertools, json, multiprocessing, os, sys, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from threadpoolctl import threadpool_limits

ROOT=Path('.').resolve()
A=ROOT/'results/bibm_extensions_20261005/agent_a_reliability'
HERE=ROOT/'results/bibm_extensions_20261005/integrated/matched_half_calibration'
sys.path.insert(0,str(Path(__file__).resolve().parent))
import summarize_contiguous as original
REPS=original.REPS
KS=[5,10,15,20,30,50,100,200]
METRICS=['sampled_map_spearman']+[f'O{k}' for k in KS]+['finite_RBO15','finite_RBO20']
BOOTSTRAPS=10000
BOOT_SEED=20261006
CTX=None

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()

def write_json(path,value):
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');os.replace(temp,path)

def existing_rbo():
    # Execute only the inspected pure metric function; never execute C's analysis.
    path=Path(__file__).resolve().parents[2]/'analysis/multiscale/run_multiscale.py'
    tree=ast.parse(path.read_text());node=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='rbo')
    scope={};exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),scope)
    return scope['rbo'],{'definition_file':str(path),'definition_file_sha256':sha(path),'p':.9,'depths':[15,20],'finite':True,'extrapolated':False}

def input_checks():
    started=time.time()
    integration=json.loads((HERE.parent/'INTEGRATION_AUDIT.json').read_text())
    assert integration['status']=='PASS'
    provenance=json.loads((A/'SPLIT_PROVENANCE.json').read_text())
    split_path=A/'split_cells.npy'
    assert sha(split_path)==provenance['split_assignment_sha256']
    labels=np.load(split_path,mmap_mode='r',allow_pickle=False)
    assert labels.shape==(20,21710586) and labels.dtype==np.int8
    # One binary label per canonical cell makes opposite halves disjoint for
    # BOTH target and NTC cells. No assignments or cell embeddings are generated.
    assignment_counts=[]
    for s in range(20):
        zeros=ones=0
        for lo in range(0,labels.shape[1],1000000):
            v=labels[s,lo:lo+1000000]
            assert np.all((v==0)|(v==1))
            zeros+=int((v==0).sum());ones+=int((v==1).sum())
        assert zeros+ones==21710586
        assignment_counts.append([zeros,ones])
    del labels
    vectors={};supports={};valids={};ntcs={};artifacts={}
    expected_target=np.load(A/'split_vectors/pca_split_target_support.npy',mmap_mode='r')
    expected_ntc=np.load(A/'split_vectors/pca_split_ntc_support.npy',mmap_mode='r')
    expected_valid=np.load(A/'split_vectors/pca_split_valid_unit.npy',mmap_mode='r')
    assert np.all(expected_ntc>0)
    assert np.all(np.abs(expected_target[:,0]-expected_target[:,1])<=1)
    assert np.all(np.abs(expected_ntc[:,0]-expected_ntc[:,1])<=1)
    assert np.all(expected_target.sum(axis=1)==expected_target[0].sum(axis=0))
    assert np.all(expected_ntc.sum(axis=1)==expected_ntc[0].sum(axis=0))
    for name,sub,dim in REPS:
        dest=original.VEC_DIRS[name]
        p=dest/f'{sub}_split_responses.npy'
        vectors[name]=np.load(p,mmap_mode='r',allow_pickle=False)
        assert vectors[name].shape==(20,2,144368,dim) and vectors[name].dtype==np.float64
        supports[name]=np.load(dest/f'{sub}_split_target_support.npy',mmap_mode='r')
        ntcs[name]=np.load(dest/f'{sub}_split_ntc_support.npy',mmap_mode='r')
        valids[name]=np.load(dest/f'{sub}_split_valid_unit.npy',mmap_mode='r')
        assert np.array_equal(supports[name],expected_target)
        assert np.array_equal(ntcs[name],expected_ntc)
        assert np.array_equal(valids[name],expected_valid)
        assert np.array_equal(valids[name],supports[name]>0)
        audit=json.loads((dest/f'{sub}_split_audit.json').read_text())
        assert audit['splits']==20 and audit['seed']==28091005 and audit['pilot_rows']==0
        assert audit['paired_valid_unit_count_by_split']==[141830]*20
        assert audit['invalid_unit_halves_due_missing_ntc_centroid']==0
        assert not np.any(audit['zero_norm_response_half_by_split'])
        if sub in ('geneformer','state'):assert audit['split_sha256']==provenance['split_assignment_sha256']
        else:
            metadata_path=A/('RUN_METADATA_split_vectors.json' if sub=='pca' else 'RUN_METADATA_split_vectors_scvi.json')
            metadata=json.loads(metadata_path.read_text())
            assert metadata['seed']==28091005 and metadata['splits']==20 and not metadata['pilot_only']
            assert metadata['index_sha256']==provenance['common_cell_index_sha256']
        # Prior completed integration explicitly checked all Geneformer shard
        # audits despite its missing top-level source_audit_status field.
        assert integration['representations'][name]['source_audit']=='PASS'
        for s in range(20):
            assert supports[name][s].sum()==20771051 and ntcs[name][s].sum()==939535
            assert supports[name][s,0].sum()+ntcs[name][s,0].sum()==assignment_counts[s][0]
            assert supports[name][s,1].sum()+ntcs[name][s,1].sum()==assignment_counts[s][1]
        artifacts[name]={'path':str(p),'shape':list(vectors[name].shape),'dtype':str(vectors[name].dtype),'bytes':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns,'split_audit_sha256':sha(dest/f'{sub}_split_audit.json'),'source_audit_sha256':sha(dest/f'{sub}_split_source_audit.json'),'prior_source_audit':'PASS','shared_target_counts_exact':True,'shared_ntc_counts_exact':True,'shared_validity_exact':True}
    samples={};sample_info={}
    manifest=json.loads((A/'split_pair_sample_manifest.json').read_text())
    previous=pd.read_csv(A/'split_reliability_metrics.tsv',sep='\t')
    assert len(previous)==960
    previous={ (r.representation,r.donor,r.context,int(r.split)):r._asdict() for r in previous.itertuples(index=False)}
    for block in manifest['pair_blocks']:
        donor=block['donor'];context=block['context'];path=A/'pair_samples'/f'{donor}_{context}.npz'
        with np.load(path,allow_pickle=False) as z:sample={k:z[k].copy() for k in z.files}
        units=sample['units'];n=len(units);targets=sample['targets'];pi=sample['pair_i'];pj=sample['pair_j'];focal=sample['focal_local']
        assert n==block['common_targets'] and len(pi)==100000 and len(focal)==500
        assert len(set(targets.tolist()))==n and np.array_equal(targets,np.sort(targets))
        assert len(np.unique(units))==n and np.all((units>=0)&(units<144368))
        assert np.all((pi>=0)&(pi<pj)&(pj<n))
        assert len(np.unique(pi.astype(np.int64)*n+pj))==len(pi)
        assert len(np.unique(focal))==500 and np.all((focal>=0)&(focal<n)) and n>200
        for name,sub,dim in REPS:
            assert valids[name][:,:,units].all()
            for s in range(20):
                assert np.array_equal(supports[name][s,0,units]+supports[name][s,1,units],sample['support'])
                prior=previous[(name,donor,context,s+1)]
                assert prior['n_common_targets']==n and prior['n_sampled_pairs']==len(pi) and prior['n_focal_targets']==len(focal)
                assert prior['ntc_support_min_lane_half_median']==previous[('PCA',donor,context,s+1)]['ntc_support_min_lane_half_median']
        samples[(donor,context)]=sample
        sample_info[f'{donor}_{context}']={'path':str(path),'sha256':sha(path),'targets':n,'sampled_edges':len(pi),'focal_targets':len(focal),'unit_row_span':int(units.max()-units.min()+1)}
    assert len(samples)==12
    rbo,rbo_info=existing_rbo()
    audit={'status':'INPUT_PASS','checked_utc':datetime.now(timezone.utc).isoformat(),'split_assignment_path':str(split_path),'split_assignment_sha256':provenance['split_assignment_sha256'],'assignment_counts':assignment_counts,'labels_binary':True,'opposite_halves_disjoint':True,'target_and_ntc_half_counts_identical_across_representations':True,'balanced_group_support_max_half_difference':1,'half_support_matching':'Every within and cross comparison uses one A response and one B response; R_A/S_A and R_B/S_B supports exactly equal. Odd groups can differ by one cell between A and B.','representation_specific_filtering':False,'ntc_support_indexing':'No new NTC stratum indexing: all 284 half-count vectors exactly match PCA; matched min-lane support is reused from corrected Agent A summary, never indexed with global units.','cell_embeddings_read':False,'full_data_response_vectors_read':False,'response_inputs':artifacts,'samples':sample_info,'prior_integration_audit_sha256':sha(HERE.parent/'INTEGRATION_AUDIT.json'),'original_within_metrics_sha256':sha(A/'split_reliability_metrics.tsv'),'rbo':rbo_info,'input_validation_seconds':time.time()-started}
    signature=hashlib.sha256(json.dumps({'code':sha(__file__),'samples':sample_info,'inputs':artifacts,'assignment':audit['split_assignment_sha256']},sort_keys=True).encode()).hexdigest()
    return {'vectors':vectors,'supports':supports,'ntcs':ntcs,'samples':samples,'previous':previous,'rbo':rbo,'signature':signature},audit

def agreement_local(a,b):
    result={}
    for k in KS:
        result[f'O{k}']=float(np.mean([len(set(a[q,:k]).intersection(b[q,:k]))/k for q in range(len(a))]))
    for k in (15,20):
        result[f'finite_RBO{k}']=float(np.mean([CTX['rbo'](a[q,:k],b[q,:k],.9) for q in range(len(a))]))
    return result

def calculate(task):
    donor,context,s=task
    cache=HERE/'checkpoints'/f'{donor}_{context}_split{s+1:02d}.json'
    if cache.exists():
        saved=json.loads(cache.read_text())
        assert saved['signature']==CTX['signature'],'Checkpoint input/code signature changed'
        return saved
    started=time.time();sample=CTX['samples'][(donor,context)];units=sample['units'];targets=sample['targets'];pi=sample['pair_i'];pj=sample['pair_j'];focal=sample['focal_local']
    lo=int(units.min());hi=int(units.max())+1;edges=[];tops=[]
    for name,sub,dim in REPS:
        for h in (0,1):
            slab=np.array(CTX['vectors'][name][s,h,lo:hi],copy=True)
            v=np.asarray(slab[units-lo],dtype=np.float64);del slab
            assert np.isfinite(v).all()
            norms=np.linalg.norm(v,axis=1);assert np.all(norms>0)
            v=(v/norms[:,None]).astype(np.float32)
            edges.append(original.edge_cosines(v,pi,pj))
            tops.append(original.top_ids(v[focal]@v.T,targets,focal,200))
            del v,norms
    # Same average-tie Spearman definition; cache each rank vector once rather
    # than repeatedly sorting it for each representation-pair/direction.
    ranks=np.stack([rankdata(e,method='average') for e in edges])
    corr=np.corrcoef(ranks);assert np.isfinite(corr).all()
    within={};max_diff=0.
    for r,(name,sub,dim) in enumerate(REPS):
        within[r]={'sampled_map_spearman':float(corr[2*r,2*r+1]),**agreement_local(tops[2*r],tops[2*r+1])}
        prior=CTX['previous'][(name,donor,context,s+1)]
        for metric,previous_metric in [('sampled_map_spearman','whole_map_spearman_sampled')]+[(f'O{k}',f'O{k}') for k in original.KS]:
            diff=abs(within[r][metric]-prior[previous_metric]);max_diff=max(max_diff,diff)
            assert diff<=1e-12,f'Existing within result changed: {name}/{donor}/{context}/{s+1}/{metric}: {diff}'
    rows=[]
    for r,t in itertools.combinations(range(4),2):
        # Both cross directions compare one A set and one B set, never A/A or B/B.
        ra,rb=2*r,2*r+1;sa,sb=2*t,2*t+1
        assert ra%2!=sb%2 and rb%2!=sa%2
        ab={'sampled_map_spearman':float(corr[ra,sb]),**agreement_local(tops[ra],tops[sb])}
        ba={'sampled_map_spearman':float(corr[rb,sa]),**agreement_local(tops[rb],tops[sa])}
        for metric in METRICS:
            wr=within[r][metric];ws=within[t][metric];cab=ab[metric];cba=ba[metric]
            wm=(wr+ws)/2;cm=(cab+cba)/2
            rows.append({'representation_R':REPS[r][0],'representation_S':REPS[t][0],'donor':donor,'context':context,'split':s+1,'metric':metric,'n_common_targets':len(units),'n_sampled_edges':len(pi),'n_focal_targets':len(focal),'within_R':wr,'within_S':ws,'within_mean':wm,'cross_AB':cab,'cross_BA':cba,'cross_mean':cm,'representation_gap':wm-cm,'cross_AB_halves':'R_A,S_B','cross_BA_halves':'R_B,S_A','target_A_cells_in_matched_universe':int(CTX['supports'][REPS[r][0]][s,0,units].sum()),'target_B_cells_in_matched_universe':int(CTX['supports'][REPS[r][0]][s,1,units].sum()),'matched_NTC_min_lane_half_median':float(CTX['previous'][(REPS[r][0],donor,context,s+1)]['ntc_support_min_lane_half_median'])})
    result={'signature':CTX['signature'],'donor':donor,'context':context,'split':s+1,'seconds':time.time()-started,'max_existing_within_difference':max_diff,'rows':rows}
    write_json(cache,result)
    return result

def summaries(frame,groups):
    rng=np.random.default_rng(BOOT_SEED);out=[]
    for key,g in frame.groupby(groups,sort=False):
        if not isinstance(key,tuple):key=(key,)
        row=dict(zip(groups,key));values=g.representation_gap.to_numpy()
        row.update({'n_comparisons':len(g),'n_donors':g.donor.nunique(),'n_contexts':g.context.nunique(),'n_splits':g.split.nunique(),'mean_gap':float(values.mean()),'median_gap':float(np.median(values)),'SD_gap':float(np.std(values,ddof=1)),'n_gap_positive':int((values>0).sum()),'fraction_gap_positive':float((values>0).mean())})
        for col in ['within_R','within_S','within_mean','cross_AB','cross_BA','cross_mean']:
            row[col+'_mean']=float(g[col].mean())
        if g.donor.nunique()>1:
            # Resample whole donors, retaining all fixed contexts and all 20
            # overlapping partitions for each selected donor together.
            cluster=g.groupby('donor').representation_gap.mean().to_numpy()
            boot=cluster[rng.integers(0,len(cluster),size=(BOOTSTRAPS,len(cluster)))].mean(axis=1)
            method='whole_donor_cluster; contexts and 20 partitions retained together'
        else:
            # One-donor summaries cannot estimate between-donor uncertainty.
            # Display only conditional random-partition bootstrap variability,
            # resampling each split across all that donor's contexts together.
            cluster=g.groupby('split').representation_gap.mean().to_numpy()
            boot=cluster[rng.integers(0,len(cluster),size=(BOOTSTRAPS,len(cluster)))].mean(axis=1)
            method='whole_split_index; conditional partition variability for one donor only'
        row.update({'bootstrap_CI95_low':float(np.quantile(boot,.025)),'bootstrap_CI95_high':float(np.quantile(boot,.975)),'bootstrap_method':method,'bootstrap_replicates':BOOTSTRAPS})
        out.append(row)
    return pd.DataFrame(out)

def markdown(frame):
    def value(x):return f'{x:.4f}' if isinstance(x,(float,np.floating)) else str(x)
    lines=['| '+' | '.join(frame.columns)+' |','| '+' | '.join(['---']*len(frame.columns))+' |']
    lines += ['| '+' | '.join(value(v) for v in row)+' |' for row in frame.itertuples(index=False,name=None)]
    return '\n'.join(lines)

def figure(frame):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    colors=dict(zip(['D1','D2','D3','D4'],['#1f77b4','#ff7f0e','#2ca02c','#9467bd']))
    markers={'Rest':'o','Stim8hr':'^','Stim48hr':'s'}
    fig,axes=plt.subplots(6,2,figsize=(11,17),sharex=True)
    abbreviate={'PCA':'PCA','scVI':'scVI','Geneformer V2-316M':'Geneformer','STATE SE-600M':'STATE'}
    primary=frame[frame.metric.isin(['sampled_map_spearman','O15'])]
    blocks=primary.groupby(['representation_R','representation_S','donor','context','metric'])[['within_mean','cross_mean']].mean().reset_index()
    for i,(r,s) in enumerate(itertools.combinations([x[0] for x in REPS],2)):
        for j,metric in enumerate(['sampled_map_spearman','O15']):
            ax=axes[i,j];g=blocks[(blocks.representation_R==r)&(blocks.representation_S==s)&(blocks.metric==metric)]
            for row in g.itertuples():
                ax.plot([0,1],[row.within_mean,row.cross_mean],color=colors[row.donor],marker=markers[row.context],alpha=.65,linewidth=1,markersize=4)
            ax.plot([0,1],[g.within_mean.mean(),g.cross_mean.mean()],color='black',marker='D',linewidth=2.4,markersize=5)
            ax.set_title(f'{abbreviate[r]} – {abbreviate[s]}',fontsize=11)
            ax.set_xticks([0,1],['Within, independent halves','Cross, opposite halves'])
            ax.tick_params(axis='x',labelbottom=True,labelsize=8)
            ax.set_ylabel('Sampled map Spearman' if j==0 else 'O15')
            ax.grid(axis='y',alpha=.2)
    handles=[Line2D([0],[0],color=c,label=d) for d,c in colors.items()]+[Line2D([0],[0],color='gray',marker=m,linestyle='',label=c) for c,m in markers.items()]+[Line2D([0],[0],color='black',marker='D',label='Mean over 12 blocks')]
    fig.legend(handles=handles,loc='lower center',ncol=4,frameon=False)
    fig.suptitle('Matched half-support, independent-cell calibration\nEach colored line is one donor/context block averaged over 20 partitions',fontsize=14)
    fig.tight_layout(rect=(0,.055,1,.95));fig.savefig(HERE/'matched_half_calibration.png',dpi=180);fig.savefig(HERE/'matched_half_calibration.pdf');plt.close(fig)

def integrate(pair_summary,report_text,audit):
    path=HERE.parent/'BIBM_EXTENSION_RESULTS.md'
    old=path.read_text();stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    backup=HERE/f'BIBM_EXTENSION_RESULTS.before_matched_half_{stamp}.md'
    backup.write_text(old)
    primary=pair_summary[pair_summary.metric.isin(['sampled_map_spearman','O15'])]
    table=primary[['representation_R','representation_S','metric','within_mean_mean','cross_mean_mean','mean_gap','bootstrap_CI95_low','bootstrap_CI95_high','fraction_gap_positive']]
    positive=int((primary.mean_gap>0).sum());n=len(primary)
    assessment=('All six representation pairs have positive average gaps for both primary metrics.' if positive==12 else 'The primary result is mixed across representation pairs and metrics.' if positive else 'No pair has a positive average gap for either primary metric.')
    subsection='''### A — Matched opposite-half calibration (additional analysis)

The original calibration above compares **full/full, same-cell cross-representation**
agreement with **half/half, independent-cell within-representation** agreement.
It is retained as provenance, but its unequal support and shared-cell structure
prevent it from deciding the matched finite-cell question.

This additive comparison uses only the authoritative split-response arrays and
the exact existing target universes, sampled edges and focal targets. For each
pair R/S, block and split, it computes within R_A/R_B and S_A/S_B, and cross
R_A/S_B and R_B/S_A. All comparisons therefore use opposite persisted halves
for both target cells and NTCs. The gap is the mean within agreement minus the
mean of the two directional cross agreements; this uses the requested mean
baseline, distinct from the earlier lower-median baseline.

'''+markdown(table)+'\n\n'+assessment+'''

Positive matched gaps indicate that representation changes reduce agreement
relative to independent half-sampling for that pair/metric in this frozen data.
They do not establish causal effects or remove all estimation noise. The analysis
uses no attenuation/noise correction. Individual edges, targets, splits and
donor/context observations are not independent biological replicates.

Overall and context-specific 95% bootstrap intervals resample whole donors,
retaining their fixed contexts and all 20 overlapping partitions together. Only
four donors are available, so intervals are descriptive. Donor-specific intervals
resample complete split indices across that donor's contexts and quantify only
conditional partition variability, not between-donor uncertainty.

All source/support/disjoint-half checks passed, and recomputed legacy within
metrics agree within 1e-12. RBO is the existing finite, non-extrapolated p=0.9
definition at depths 15/20; O200 extends the existing O_k definition on the same
targets. Details, donor/context tables, both cross directions, bootstrap methods
and the block-paired figure are in `matched_half_calibration/`.

'''
    marker='## Updated synthesis and evidence ranking';assert marker in old
    updated=old.replace(marker,subsection+marker,1)
    if positive:
        conclusion='''The earlier negative calibration result applies to the unmatched full/full
versus half/half comparison and is not the final matched-design conclusion.
The opposite-half comparison resolves the support/shared-cell mismatch directly:
'''+assessment+'''
Use the pair/metric gaps and donor/context tables above to scope any claim that
representation choice adds disagreement beyond matched finite-cell sampling.
This is descriptive paired evidence on the frozen cohort, not a global
noise decomposition or a test based on 240 independent biological replicates.

1. **Matched reliability-calibrated geometry:** pair-specific positive matched
   gaps are the direct new evidence for additional representation disagreement;
   preserve any negative/near-zero cases, donor/context variation and limited
   donor-cluster uncertainty.
2. **Resource-specific biological utility:** retain B's HuMAP3/STRING findings
   without universal representation-superiority claims.
3. **Multiscale geometry:** interpret C's neighborhood/edge-tail dependence
   alongside matched A calibration; C's broader cohorts remain distinct.
4. **Program B focused case:** retain E's random/support-matched comparisons and
   the three-pair limitation.
5. **Exact variance/support sensitivity:** retain D's signed-tail, complete-target
   results as explanatory evidence, not a replacement for matched A calibration.

## Updated candidate paper stories

- **Conservative:** representation geometry and external biological recovery vary
  across models, with explicit finite-cell calibration and resource dependence.
- **Matched-calibration story:** representation changes add disagreement relative
  to independent half-sampling for the supported pair/metric combinations. Show
  the matched within/cross figure and donor/context variation; do not substitute
  the older full/full baseline or claim that half responses are noise-free.
- **Expanded story:** connect matched geometry to C's multiscale patterns and
  B's biological utility, with E/D as focused/supplementary analyses and their
  distinct cohorts/estimands clearly stated.

'''
    else:
        conclusion='''The matched opposite-half design also has no positive average gap for either
primary metric in any representation pair. This strengthens the descriptive
conclusion that these data do not support representation disagreement beyond
matched finite-cell estimation variability. Retain interval/context/donor
variation; this does not establish that all differences are solely noise.

1. Resource-specific biological recovery remains strong supporting evidence.
2. Multiscale geometry remains descriptive and must retain the matched reliability limit.
3. Program B remains a focused case with three-pair limitations.
4. Matched reliability calibration is an important negative finding.
5. Exact variance/support analyses remain explanatory/supplementary.

## Updated candidate paper stories

Use descriptive representation dependence and resource-specific biology with
the matched negative calibration prominently reported. A story claiming added
representation disagreement beyond finite-cell sampling is unsupported by
this analysis.

'''
    start=updated.index(marker)+len(marker);end=updated.index('## Finalization and provenance',start)
    updated=updated[:start]+'\n\n'+conclusion+updated[end:]
    updated=updated.replace('**Status: completed A/D integration; audits PASS.**','**Status: completed A/D and additional matched-half calibration; audits PASS.**',1)
    updated += '\nAdditional matched-half calibration is additive; B/C/D/E and the manuscript remain unchanged. The pre-edit integrated report is backed up in `matched_half_calibration/`.\n'
    temp=path.with_suffix('.md.tmp');temp.write_text(updated);os.replace(temp,path)
    audit['integration']={'report':str(path),'backup':str(backup),'original_report_sha256':hashlib.sha256(old.encode()).hexdigest(),'updated_report_sha256':sha(path),'original_calibration_retained':True,'matched_subsection_immediately_after_A':True}

def main():
    global CTX
    parser=argparse.ArgumentParser();parser.add_argument('--benchmark',action='store_true');parser.add_argument('--workers',type=int,default=16);args=parser.parse_args()
    threadpool_limits(limits=1);started=time.time();(HERE/'checkpoints').mkdir(exist_ok=True)
    print(json.dumps({'event':'validating_existing_response_inputs','embeddings_read':False}),flush=True)
    CTX,audit=input_checks();write_json(HERE/('benchmark_input_audit.json' if args.benchmark else 'matched_half_input_audit.json'),audit)
    if args.benchmark:
        result=calculate(('D1','Rest',0))
        print(json.dumps({'event':'benchmark_complete','task_seconds':result['seconds'],'total_seconds':time.time()-started,'max_existing_within_difference':result['max_existing_within_difference'],'rows':len(result['rows'])}),flush=True)
        write_json(HERE/'benchmark_result.json',{k:v for k,v in result.items() if k!='rows'})
        return
    tasks=[(d,c,s) for d,c in sorted(CTX['samples']) for s in range(20)];rows=[];differences=[];task_seconds=[]
    # 16 allocated CPUs; one numerical thread per process. Completed split tasks
    # are checkpointed, and their file/code signatures are checked on reuse.
    with multiprocessing.get_context('fork').Pool(args.workers) as pool:
        for completed,result in enumerate(pool.imap(calculate,tasks,chunksize=1),1):
            rows.extend(result['rows']);differences.append(result['max_existing_within_difference']);task_seconds.append(result['seconds'])
            if completed%20==0:
                print(json.dumps({'event':'block_complete','donor':result['donor'],'context':result['context'],'split_tasks_complete':completed,'split_tasks_total':240,'elapsed_seconds':time.time()-started}),flush=True)
    frame=pd.DataFrame(rows);assert len(frame)==6*12*20*len(METRICS)
    assert not frame.duplicated(['representation_R','representation_S','donor','context','split','metric']).any()
    numeric=['within_R','within_S','within_mean','cross_AB','cross_BA','cross_mean','representation_gap'];assert np.isfinite(frame[numeric]).all().all()
    assert np.allclose(frame.within_mean,(frame.within_R+frame.within_S)/2,rtol=0,atol=1e-15)
    assert np.allclose(frame.cross_mean,(frame.cross_AB+frame.cross_BA)/2,rtol=0,atol=1e-15)
    assert np.allclose(frame.representation_gap,frame.within_mean-frame.cross_mean,rtol=0,atol=1e-15)
    frame.to_csv(HERE/'matched_half_calibration.tsv',sep='\t',index=False)
    pair=summaries(frame,['representation_R','representation_S','metric'])
    context=summaries(frame,['representation_R','representation_S','context','metric'])
    donor=summaries(frame,['representation_R','representation_S','donor','metric'])
    for name,value in [('pair',pair),('context',context),('donor',donor)]:value.to_csv(HERE/f'matched_half_{name}_summary.tsv',sep='\t',index=False)
    assert len(pair)==66 and len(context)==198 and len(donor)==264
    figure(frame)
    primary=pair[pair.metric.isin(['sampled_map_spearman','O15'])]
    text='''# Matched half-support calibration

All comparisons use the existing half-response arrays. Within: R_A/R_B and
S_A/S_B. Cross: R_A/S_B and R_B/S_A. Both endpoints therefore use opposite,
disjoint target and NTC halves with identical support vectors across models.
Group support may differ by one cell between A and B for odd group sizes;
the two cross directions are retained and averaged, not silently forced equal.

The exact existing common target keys, 100,000 sampled edges and 500 focal
targets per block are reused. No embedding matrices or full-data response
vectors are read. No split assignments are regenerated. Normalization to unit
length, float32 geometry, cosine edges, average-tie Spearman, alphabetic neighbor
tie handling and O_k match Agent A. Finite p=0.9 RBO at 15/20 uses the existing
Agent C pure function, restricted to the same Agent A focal targets. O200 extends
the established O_k definition without changing its universe. Within metrics
through O100 are checked against all 960 original within results at tolerance 1e-12.

## Primary pair summaries

'''+markdown(primary[['representation_R','representation_S','metric','within_R_mean','within_S_mean','within_mean_mean','cross_AB_mean','cross_BA_mean','cross_mean_mean','mean_gap','median_gap','SD_gap','bootstrap_CI95_low','bootstrap_CI95_high','n_gap_positive','fraction_gap_positive']])+'''

## Uncertainty and interpretation

Each pair has 240 comparisons (20 overlapping random partitions × 12 recurring
donor/context blocks). These are not 240 independent biological replicates.
SD and positive fractions are descriptive over the paired observations.
Overall/context intervals resample whole donors, retaining all fixed contexts
and all 20 partitions together (10,000 percentile bootstraps, seed 20261006).
With four donors, these intervals are descriptive and unstable as population
uncertainty estimates. Contexts are fixed conditions, not exchangeable replicates.
Donor-specific intervals resample whole split indices jointly across that donor's
contexts, quantifying conditional random-partition variation only. They are not
cell/population or between-donor uncertainty intervals. No p-values are reported.

Positive gaps mean that changing representation reduces opposite-half agreement
relative to the mean within-model opposite-half agreement for that pair/metric.
This is a paired descriptive contrast, not an attenuation correction, causal
effect or complete noise decomposition. Negative and context/donor-specific
patterns must remain visible. The earlier full/full versus half/half comparison
also used a lower-median within baseline; it remains historical provenance and
is not numerically interchangeable with the requested within-mean gap.

## Outputs

Long-format matched_half_calibration.tsv retains both directional cross values.
Pair/context/donor tables contain mean/median/SD gaps, bootstrap methods and
intervals, positive counts/fractions and all endpoint means. The PNG/PDF figure
shows 12 paired donor/context lines per representation pair, each averaged over
20 partitions; donor colors and context markers preserve the repeated structure.
Audit JSON records exact inputs, source audits, support/disjointness checks,
sample hashes, within-metric equivalence and report backup. Checkpoints are
diagnostic/resumable intermediate task results, not new response arrays.
'''
    (HERE/'MATCHED_HALF_CALIBRATION.md').write_text(text)
    # Confirm input files were not changed while this analysis ran.
    for artifact in audit['response_inputs'].values():assert Path(artifact['path']).stat().st_mtime_ns==artifact['mtime_ns']
    for sample in audit['samples'].values():assert sha(sample['path'])==sample['sha256']
    audit.update({'status':'PASS','completed_utc':datetime.now(timezone.utc).isoformat(),'response_array_finiteness_and_norms_checked_on_all_matched_units':True,'metrics':METRICS,'comparison_rows':len(frame),'pair_summary_rows':len(pair),'context_summary_rows':len(context),'donor_summary_rows':len(donor),'split_tasks':240,'workers':args.workers,'max_existing_within_difference':max(differences),'existing_within_equivalence_tolerance':1e-12,'bootstrap_replicates':BOOTSTRAPS,'bootstrap_seed':BOOT_SEED,'bootstrap_overall_context':'whole donors; contexts and partitions fixed together','bootstrap_donor':'whole split index jointly across contexts; conditional partition variability only','new_metric_definitions':False,'full_embedding_arrays_read':False,'attenuation_correction':False,'script_sha256':sha(__file__),'task_seconds_mean':float(np.mean(task_seconds)),'elapsed_seconds':time.time()-started})
    write_json(HERE/'matched_half_calibration_audit.json',audit)
    integrate(pair,text,audit)
    write_json(HERE/'matched_half_calibration_audit.json',audit)
    print(json.dumps({'event':'matched_half_complete','status':'PASS','elapsed_seconds':time.time()-started,'rows':len(frame),'primary_positive_mean_gaps':int((primary.mean_gap>0).sum()),'primary_pair_metrics':len(primary)}),flush=True)

if __name__=='__main__':main()
