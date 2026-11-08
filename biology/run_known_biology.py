#!/usr/bin/env python3
"""Genome-wide known-pair recovery using official human resources and frozen responses."""
from pathlib import Path
import gzip, json, re, time, hashlib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT=Path('source/derived/downstream')
RESP=ROOT/'response_construction_v1'; GEOM=ROOT/'response_geometry_v1'
OUT=ROOT/'response_prebiology_v1/known_biology'; RES=OUT/'resources'
REPS=['pca','scvi','geneformer','state']; LABELS={'pca':'PCA','scvi':'scVI','geneformer':'Geneformer','state':'STATE'}
DONORS=['D1','D2','D3','D4']; CONTEXTS=['Rest','Stim8hr','Stim48hr']

def pairs_from_symbols(a,b):
    a=str(a).strip().upper(); b=str(b).strip().upper()
    if not a or not b or a in {'NA','N/A','-'} or b in {'NA','N/A','-'} or a==b: return None
    return (a,b) if a<b else (b,a)

def main():
  t0=time.time()
  # Canonical target identifier mapping comes from frozen response metadata.
  meta=pd.read_csv(RESP/'pca/response_units.tsv.gz',sep='\t',usecols=['target','target_gene_id'])
  tg=meta[['target','target_gene_id']].drop_duplicates()
  assert not tg.target.duplicated().any() and not tg.target_gene_id.duplicated().any()
  symbol_to_id={str(r.target).upper():str(r.target_gene_id).split('.')[0] for r in tg.itertuples()}
  ensg_to_symbol={v:k for k,v in symbol_to_id.items()}
  targetset=set(symbol_to_id)
  sources={}; manifests=[]; mapping=[]
  def register(name, pairs, raw_count, version, url, definition, filename):
      p=set(pairs); valid=set(x for x in p if x[0] in targetset and x[1] in targetset)
      sources[name]=valid
      mapping.append({'resource':name,'raw_relationships':int(raw_count),'parsed_unique_pairs':len(p),
        'mapped_unique_pairs':len(valid),'unique_mapped_genes':len(set(z for edge in valid for z in edge)),
        'mapping_failures_pairs':len(p)-len(valid),'mapping_failures_note':'pairs with one or both symbols outside canonical target universe'})
      raw=RES/filename
      manifests.append({'resource':name,'version_release':version,'retrieval_date':'2026-09-30','source_url':url,
        'raw_file':str(raw),'sha256':hashlib.sha256(raw.read_bytes()).hexdigest(),'definition':definition,'status':'AVAILABLE'})
      print('MAPPED',name,'raw',raw_count,'unique',len(p),'mapped',len(valid),flush=True)
  # HuMAP3 high-confidence published pair probabilities >= 0.5 (fixed ex ante).
  hp=RES/'humap3_complex_pairs_genenames.pairsWprob'; pairs=[]; raw=0
  for chunk in pd.read_csv(hp,sep='\t',header=None,names=['a','b','score'],chunksize=500000):
      raw+=len(chunk)
      for a,b in chunk.loc[chunk.score>=0.5,['a','b']].itertuples(index=False,name=None):
          p=pairs_from_symbols(a,b)
          if p: pairs.append(p)
  register('HuMAP3',''.join([]) if False else pairs,raw,'HuMAP 3.0; pairs probability >=0.5','https://humap3.proteincomplexes.org/download',
    'Official HuMAP3 protein-pair predictions within complexes; retain probability >=0.5; unordered pairs.','humap3_complex_pairs_genenames.pairsWprob')
  # Reactome current human relation file, use its explicit protein association and reaction relation rows.
  rp=RES/'reactome_human_interactions.tsv'; pairs=[]; raw=0
  for chunk in pd.read_csv(rp,sep='\t',comment='#',header=None,usecols=[1,4,6],names=['ens1','ens2','type'],chunksize=200000,dtype=str):
      raw+=len(chunk)
      for x,y in chunk[['ens1','ens2']].itertuples(index=False,name=None):
          gx=set(re.findall(r'ENSG\d+',str(x))); gy=set(re.findall(r'ENSG\d+',str(y)))
          for ex in gx:
              for ey in gy:
                  # Keep unmapped Ensembl IDs as explicit identifiers so the
                  # mapping summary can count them rather than silently omit them.
                  a=ensg_to_symbol.get(ex,ex); b=ensg_to_symbol.get(ey,ey)
                  p=pairs_from_symbols(a,b)
                  if p: pairs.append(p)
  register('Reactome',pairs,raw,'Current human interactions snapshot (last modified 2026-06-20)','https://reactome.org/download/current/interactors/reactome.homo_sapiens.interactions.tab-delimited.txt',
    'All non-self gene pairs from the official Reactome human protein-interaction export; Ensembl gene IDs mapped to canonical target_gene_id; export interaction contexts define pair relations; unordered pairs.','reactome_human_interactions.tsv')
  # SIGNOR 4.0, human protein-protein directed records curated direct; direction/sign ignored.
  sp=RES/'SIGNOR_Jul2026_release.txt'; pairs=[]; raw=0
  for chunk in pd.read_csv(sp,sep='\t',chunksize=100000,dtype=str,low_memory=False):
      mask=(chunk.TAX_ID=='9606')&(chunk.DIRECT=='t')&(chunk.TYPEA=='protein')&(chunk.TYPEB=='protein')
      z=chunk[mask]; raw+=len(z)
      for a,b in z[['ENTITYA','ENTITYB']].itertuples(index=False,name=None):
          p=pairs_from_symbols(a,b)
          if p: pairs.append(p)
  register('SIGNOR',pairs,raw,'SIGNOR 4.0 July 2026; human direct protein-protein records','https://www.signor.uniroma2.it/downloads.php',
    'Official SIGNOR human records with TAX_ID=9606, DIRECT=t, both entity types protein; ignore direction and effect for pair recovery.','SIGNOR_Jul2026_release.txt')
  # STRING v12.0 human functional association edges with conventional high-confidence combined_score >=700.
  ip=RES/'9606.protein.info.v12.0.txt.gz'; info=pd.read_csv(ip,sep='\t',compression='gzip')
  prot=dict(zip(info['#string_protein_id'],info.preferred_name))
  stp=RES/'9606.protein.links.v12.0.txt.gz'; pairs=[]; raw=0
  for chunk in pd.read_csv(stp,sep=' ',compression='gzip',chunksize=1000000):
      z=chunk[chunk.combined_score>=700]; raw+=len(z)
      for a,b in z[['protein1','protein2']].itertuples(index=False,name=None):
          p=pairs_from_symbols(prot.get(a,''),prot.get(b,''))
          if p: pairs.append(p)
  register('STRING',pairs,raw,'STRING v12.0; human combined_score >=700','https://string-db.org/cgi/download',
    'Official STRING human functional association network; combined_score >=700 fixed high-confidence cutoff; protein IDs mapped by official preferred-name file; unordered pairs.','9606.protein.links.v12.0.txt.gz')
  # CORUM unavailable from official endpoint; no mirror substitution.
  manifests.append({'resource':'CORUM','version_release':'Current release not downloaded','retrieval_date':'2026-09-30','source_url':'https://mips.helmholtz-muenchen.de/corum/download','raw_file':'','sha256':'','definition':'Same-curated-human-complex co-membership; not run because official download endpoint returned 502 and no official direct file could be verified.','status':'UNAVAILABLE'})
  pd.DataFrame(manifests).to_csv(OUT/'resource_manifest.tsv',sep='\t',index=False)
  pd.DataFrame(mapping).to_csv(OUT/'resource_mapping_summary.tsv',sep='\t',index=False)
  # Evaluate canonical target-pair ranks blockwise. Score distributions come only from saved canonical geometry.
  metas={r:pd.read_csv(RESP/r/'response_units.tsv.gz',sep='\t',usecols=['target','donor','context']) for r in REPS}
  arrs={r:np.load(RESP/r/'response_vectors.npy',mmap_mode='r') for r in REPS}
  pair_rep=[]; recover=[]; total_targets={}
  # positive pair membership is fixed per resource, independent of representation.
  for donor in DONORS:
    for context in CONTEXTS:
      idx_by={}; target_by={}
      for r in REPS:
          ii=np.flatnonzero((metas[r].donor.to_numpy()==donor)&(metas[r].context.to_numpy()==context))
          ix=metas[r].iloc[ii].target.to_numpy(); order=np.argsort(ix); ix=ix[order]; ii=ii[order]
          idx_by[r]=ii; target_by[r]=ix
      targets=target_by['pca']; assert all(np.array_equal(targets,target_by[r]) for r in REPS)
      n=len(targets); total_pairs=n*(n-1)//2; total_targets[(donor,context)]=n
      tindex={s:i for i,s in enumerate(targets)}
      all_known={res:sorted([(a,b) for a,b in pp if a in tindex and b in tindex]) for res,pp in sources.items()}
      union=sorted(set(p for xs in all_known.values() for p in xs))
      pos_i=np.array([tindex[a] for a,b in union],dtype=np.int32); pos_j=np.array([tindex[b] for a,b in union],dtype=np.int32)
      # Stream only cosine column from the saved table and sort exact background scores.
      for r in REPS:
          path=GEOM/r/'cosine_pairs'/f'{donor}_{context}.parquet'
          tab=pq.read_table(path,columns=['cosine_similarity'])
          # Arrow may expose a read-only zero-copy buffer; sorting needs a writable copy.
          scores=np.array(tab.column(0).combine_chunks().to_numpy(),dtype=np.float64,copy=True)
          assert scores.size==total_pairs and np.isfinite(scores).all()
          scores.sort()
          X=np.asarray(arrs[r][idx_by[r]],dtype=np.float64); X/=np.linalg.norm(X,axis=1)[:,None]
          pos_scores=np.einsum('ij,ij->i',X[pos_i],X[pos_j]) if len(union) else np.empty(0)
          # Percentile: fraction of background scores below the positive score, with half tie mass.
          lo=np.searchsorted(scores,pos_scores,'left'); hi=np.searchsorted(scores,pos_scores,'right')
          pct=(lo+hi)/(2*total_pairs)
          score_map={p:(float(s),float(q)) for p,s,q in zip(union,pos_scores,pct)}
          # Tail thresholds are exactly the top ceil(f*N) edge scores.
          fracs=[.001,.01,.05]; counts={}
          for f in fracs:
              m=max(1,int(np.ceil(f*total_pairs))); threshold=scores[-m]
              counts[f]=(m,threshold)
          for res,pp in all_known.items():
              vals=[score_map[p][1] for p in pp]
              observed={}; enrich={}
              for f in fracs:
                  m,threshold=counts[f]; k=sum(score_map[p][0]>=threshold for p in pp)
                  selected_den=int(np.sum(scores>=threshold))
                  obsfrac=k/selected_den if selected_den else float('nan')
                  bgfrac=len(pp)/total_pairs if total_pairs else float('nan')
                  observed[f]=k; enrich[f]=obsfrac/bgfrac if bgfrac else float('nan')
              recover.append({'representation':LABELS[r],'donor':donor,'context':context,'resource':res,
                'n_targets':n,'n_eligible_known_positive_pairs':len(pp),'background_positive_fraction':len(pp)/total_pairs,
                'median_percentile_rank':float(np.median(vals)) if vals else np.nan,
                'top_0.1pct_recovered':observed[.001],'top_0.1pct_recall':observed[.001]/len(pp) if pp else np.nan,'top_0.1pct_enrichment':enrich[.001],
                'top_1pct_recovered':observed[.01],'top_1pct_recall':observed[.01]/len(pp) if pp else np.nan,'top_1pct_enrichment':enrich[.01],
                'top_5pct_recovered':observed[.05],'top_5pct_recall':observed[.05]/len(pp) if pp else np.nan,'top_5pct_enrichment':enrich[.05]})
              for p in pp:
                  pair_rep.append({'resource':res,'donor':donor,'context':context,'target_a':p[0],'target_b':p[1],
                    'representation':LABELS[r],'cosine_percentile':score_map[p][1]})
          del scores
      print('KNOWN',donor,context,'n',n,'pairs',total_pairs,flush=True)
  rdf=pd.DataFrame(recover)
  rdf.to_csv(OUT/'recovery_by_donor_context.tsv.gz',sep='\t',index=False,compression='gzip')
  cols=[c for c in rdf if c not in ['donor','context','representation']]
  ctx=rdf.groupby(['representation','context','resource'])[cols].mean(numeric_only=True).reset_index()
  ctx.to_csv(OUT/'recovery_context_summary.tsv.gz',sep='\t',index=False,compression='gzip')
  pfd=pd.DataFrame(pair_rep)
  if len(pfd):
      wide=pfd.pivot_table(index=['resource','donor','context','target_a','target_b'],columns='representation',values='cosine_percentile').reset_index()
      reps=['PCA','scVI','Geneformer','STATE']; wide['percentile_range']=wide[reps].max(axis=1)-wide[reps].min(axis=1); wide['percentile_sd']=wide[reps].std(axis=1,ddof=0)
      wide.to_parquet(OUT/'known_pair_representation_stability.parquet',index=False)
  print('DONE known biology seconds',time.time()-t0,'rows',len(rdf),'known-pair rows',len(pfd),flush=True)

if __name__=='__main__': main()
