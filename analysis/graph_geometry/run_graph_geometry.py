#!/usr/bin/env python3
"""Construct union kNN graphs and compare communities using frozen exact neighborhoods."""
from __future__ import annotations
import itertools, json, math, os, platform, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import networkx as nx
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

IN=Path('source/derived/downstream/response_local_geometry_v1')
OUT=Path('source/derived/downstream/response_graph_geometry_v1')
REPS=['pca','scvi','geneformer','state']
RNAME={'pca':'PCA','scvi':'scVI','geneformer':'Geneformer V2-316M','state':'STATE SE-600M'}
DONORS=['D1','D2','D3','D4']
CONTEXTS=['Rest','Stim8hr','Stim48hr']
KS=[10,15,30]
SEED=42
RESOLUTION=1.0
ALGORITHM='NetworkX Louvain'

def log(s): print(f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] {s}',flush=True)

def setup():
    existing_artifacts=[OUT/'graphs/nodes.parquet',OUT/'graphs/edges.parquet',OUT/'communities/community_assignments.parquet',OUT/'communities/graph_descriptors.tsv.gz']
    if any(p.exists() for p in existing_artifacts) or (OUT/'graphml/k15').exists() and any((OUT/'graphml/k15').glob('*.graphml')):
        raise FileExistsError(f'Graph outputs already exist; preserving them: {OUT}')
    for d in ['graphs','graphml/k15','communities','cross_representation','reports','audits']:
        (OUT/d).mkdir(parents=True,exist_ok=True)
    (OUT/'graphml/README.md').write_text('''# Opening the primary graphs\n\nThe `k15/` directory contains 48 GraphML files: one per donor, context, and representation. Open a `.graphml` file directly in Gephi or Cytoscape.\n\nIn Gephi, size nodes by `degree` or `weighted_degree` and color nodes by `community_id`. Edge width or color can use `community_weight`; `raw_cosine`, `mutual_knn`, `rank_ab`, and `rank_ba` are also preserved. A force-directed layout such as ForceAtlas2 can help explore the graph. Weak-edge filters may be applied for display if desired.\n\nVisualization filters and layout changes do not modify the canonical graph analysis. The canonical analytical edge and node tables are in `../graphs/edges.parquet` and `../graphs/nodes.parquet`; GraphML is an interchange and visualization format.\n''')

def preflight():
    needed={'representation','donor','context','target','neighbor','rank','cosine'}
    stats={}; global_min=math.inf; global_max=-math.inf
    for rep in REPS:
        p=IN/rep/'knn_top30.parquet'
        if not p.is_file(): raise FileNotFoundError(p)
        pf=pq.ParquetFile(p)
        cols=set(pf.schema_arrow.names)
        if not needed.issubset(cols): raise ValueError(f'{rep}: missing columns {needed-cols}')
        ranks=set(); donors=set(); contexts=set(); rows=0; mincos=math.inf; maxcos=-math.inf
        for b in pf.iter_batches(batch_size=500_000,columns=['donor','context','rank','cosine']):
            d=b.column(0).to_pylist(); c=b.column(1).to_pylist(); r=b.column(2).to_numpy(); x=b.column(3).to_numpy()
            if not np.isfinite(x).all(): raise ValueError(f'nonfinite cosine in {rep}')
            if np.any((r<1)|(r>30)): raise ValueError(f'invalid rank in {rep}')
            rows+=len(r); ranks.update(np.unique(r).tolist()); donors.update(d); contexts.update(c)
            mincos=min(mincos,float(x.min())); maxcos=max(maxcos,float(x.max()))
        if ranks!=set(range(1,31)): raise ValueError(f'{rep}: ranks are {sorted(ranks)}')
        if donors!=set(DONORS) or contexts!=set(CONTEXTS): raise ValueError(f'{rep}: donor/context metadata incomplete')
        if rows!=4_331_040: raise ValueError(f'{rep}: unexpected row count {rows}')
        stats[rep]={'path':str(p),'rows':rows,'ranks':sorted(map(int,ranks)),'donors':sorted(donors),'contexts':sorted(contexts),'min_cosine':mincos,'max_cosine':maxcos}
        global_min=min(global_min,mincos); global_max=max(global_max,maxcos)
        log(f'preflight {rep}: {rows:,} rows, cosine [{mincos:.6f}, {maxcos:.6f}]')
    rule='(raw_cosine + 1) / 2 for all graphs because at least one selected edge is negative' if global_min<0 else 'raw_cosine for all graphs because every selected edge is nonnegative'
    return stats,global_min,global_max,rule

def canonical_union(directed,k):
    d=directed[directed['rank']<=k][['target','neighbor','rank','cosine']].copy()
    a=d.target.astype(str).to_numpy(); b=d.neighbor.astype(str).to_numpy(); ranks=d['rank'].to_numpy(dtype=np.int16)
    u=np.where(a<b,a,b); v=np.where(a<b,b,a)
    d=pd.DataFrame({'source':u,'target':v,'rank_ab':np.where(a==u,ranks,-1),'rank_ba':np.where(a==v,ranks,-1),'raw_cosine':d.cosine.to_numpy(dtype=np.float64)})
    e=d.groupby(['source','target'],sort=False,as_index=False).agg(rank_ab=('rank_ab','max'),rank_ba=('rank_ba','max'),raw_cosine=('raw_cosine','mean'))
    e['mutual_knn']=(e.rank_ab>=1)&(e.rank_ba>=1)
    e['source_to_target_rank']=e.rank_ab.astype(np.int16)
    e['target_to_source_rank']=e.rank_ba.astype(np.int16)
    return e

NODE_SCHEMA=pa.schema([('representation',pa.string()),('donor',pa.string()),('context',pa.string()),('k',pa.int8()),('target',pa.string()),('community_id',pa.int32()),('degree',pa.int32()),('weighted_degree',pa.float64()),('response_magnitude',pa.float64())])
EDGE_SCHEMA=pa.schema([('representation',pa.string()),('donor',pa.string()),('context',pa.string()),('k',pa.int8()),('source',pa.string()),('target',pa.string()),('raw_cosine',pa.float64()),('community_weight',pa.float64()),('rank_ab',pa.int16()),('rank_ba',pa.int16()),('source_to_target_rank',pa.int16()),('target_to_source_rank',pa.int16()),('mutual_knn',pa.bool_())])

def community_partition(G):
    comms=list(nx.community.louvain_communities(G,weight='community_weight',resolution=RESOLUTION,seed=SEED))
    comms.sort(key=lambda c:min(str(x) for x in c))
    labels={node:i for i,c in enumerate(comms) for node in c}
    modularity=float(nx.community.modularity(G,comms,weight='community_weight',resolution=RESOLUTION))
    return labels,[len(c) for c in comms],modularity

def main():
    started=time.time(); setup()
    preflight_stats,mincos,maxcos,weight_rule=preflight()
    use_shift=mincos<0
    # The local geometry magnitude sidecars are inexpensive, frozen per-response QC outputs.
    magnitudes={}
    for rep in REPS:
        p=IN/rep/'response_magnitudes.tsv.gz'
        if p.is_file():
            m=pd.read_csv(p,sep='\t',usecols=['target','donor','context','response_magnitude'])
            magnitudes[rep]={(r.target,r.donor,r.context):float(r.response_magnitude) for r in m.itertuples(index=False)}
        else: magnitudes[rep]={}
    node_writer=pq.ParquetWriter(OUT/'graphs/nodes.parquet',NODE_SCHEMA,compression='zstd')
    edge_writer=pq.ParquetWriter(OUT/'graphs/edges.parquet',EDGE_SCHEMA,compression='zstd')
    comm_writer=pq.ParquetWriter(OUT/'communities/community_assignments.parquet',NODE_SCHEMA,compression='zstd')
    descriptors=[]; edge_agreements=[]; comm_agreements=[]; graph_checks=[]
    for donor in DONORS:
      for context in CONTEXTS:
       directed={}; target_sets={}
       for rep in REPS:
        d=pd.read_parquet(IN/rep/'knn_top30.parquet',filters=[('donor','==',donor),('context','==',context)])
        directed[rep]=d
        target_sets[rep]=set(d.target.astype(str).unique())
       base=target_sets[REPS[0]]
       if any(target_sets[r]!=base for r in REPS[1:]): raise ValueError(f'target set mismatch {donor} {context}')
       nodes=sorted(base)
       for k in KS:
        graph_by_rep={}; edge_sets={}; mutual_sets={}; label_by_rep={}; ncomm_by_rep={}
        for rep in REPS:
         repname=RNAME[rep]
         edges=canonical_union(directed[rep],k)
         if edges.duplicated(['source','target']).any() or (edges.source==edges.target).any(): raise AssertionError(f'duplicate/self edge {rep} {donor} {context} k{k}')
         if not np.isfinite(edges.raw_cosine.to_numpy()).all(): raise AssertionError('nonfinite edge cosine')
         edges['community_weight']=(edges.raw_cosine+1)/2 if use_shift else edges.raw_cosine
         if not np.isfinite(edges.community_weight.to_numpy()).all() or (edges.community_weight<0).any(): raise AssertionError('invalid community weights')
         G=nx.Graph()
         G.add_nodes_from(nodes)
         G.add_edges_from((r.source,r.target,{'raw_cosine':float(r.raw_cosine),'community_weight':float(r.community_weight),'rank_ab':int(r.rank_ab),'rank_ba':int(r.rank_ba),'mutual_knn':bool(r.mutual_knn)}) for r in edges.itertuples(index=False))
         labels,sizes,modularity=community_partition(G)
         label_by_rep[rep]=labels; ncomm_by_rep[rep]=len(sizes); graph_by_rep[rep]=G
         edge_sets[rep]=set(zip(edges.source,edges.target))
         mutual_sets[rep]=set(zip(edges.loc[edges.mutual_knn,'source'],edges.loc[edges.mutual_knn,'target']))
         deg=dict(G.degree()); wdeg=dict(G.degree(weight='community_weight'))
         components=list(nx.connected_components(G)); giant=max((len(c) for c in components),default=0)
         degree_values=np.fromiter(deg.values(),dtype=np.int64,count=len(deg))
         weighted_values=np.fromiter(wdeg.values(),dtype=np.float64,count=len(wdeg))
         desc={'representation':repname,'donor':donor,'context':context,'k':k,'n_nodes':G.number_of_nodes(),'n_edges':G.number_of_edges(),'graph_density':float(nx.density(G)),
               'number_connected_components':len(components),'giant_component_nodes':giant,'giant_component_fraction':giant/G.number_of_nodes() if G.number_of_nodes() else 0,
               'mean_degree':float(degree_values.mean()),'median_degree':float(np.median(degree_values)),'max_degree':int(degree_values.max()),'mean_weighted_degree':float(weighted_values.mean()),
               'number_of_communities':len(sizes),'modularity':modularity,'community_size_min':int(min(sizes)),'community_size_median':float(np.median(sizes)),'community_size_max':int(max(sizes)),
               'mutual_edge_fraction':float(edges.mutual_knn.mean()) if len(edges) else 0}
         descriptors.append(desc)
         meta={ (str(r.target),str(r.donor),str(r.context)):float(r.response_magnitude) for r in [] }
         node_rows=[]
         for node in nodes:
          node_rows.append((repname,donor,context,k,node,int(labels[node]),int(deg[node]),float(wdeg[node]),float(magnitudes[rep].get((node,donor,context),np.nan))))
         node_table=pa.Table.from_pylist([dict(zip(NODE_SCHEMA.names,x)) for x in node_rows],schema=NODE_SCHEMA)
         edge_table_frame=edges.copy()
         edge_table_frame.insert(0,'representation',repname); edge_table_frame.insert(1,'donor',donor); edge_table_frame.insert(2,'context',context); edge_table_frame.insert(3,'k',k)
         edge_table=pa.Table.from_pandas(edge_table_frame[EDGE_SCHEMA.names],schema=EDGE_SCHEMA,preserve_index=False)
         node_writer.write_table(node_table); comm_writer.write_table(node_table); edge_writer.write_table(edge_table)
         if k==15:
          H=nx.Graph()
          H.graph.update({'representation':repname,'donor':donor,'context':context,'k':15,'community_algorithm':ALGORITHM,'random_seed':SEED,'resolution':RESOLUTION,'weighting_rule':weight_rule})
          for node in nodes:
           H.add_node(node,target=node,donor=donor,context=context,representation=repname,k=15,community_id=int(labels[node]),degree=int(deg[node]),weighted_degree=float(wdeg[node]))
           mag=magnitudes[rep].get((node,donor,context))
           if mag is not None: H.nodes[node]['response_magnitude']=float(mag)
          for r in edges.itertuples(index=False):
           H.add_edge(r.source,r.target,raw_cosine=float(r.raw_cosine),community_weight=float(r.community_weight),mutual_knn=bool(r.mutual_knn),rank_ab=int(r.rank_ab),rank_ba=int(r.rank_ba))
          gp=OUT/'graphml/k15'/f'{donor}_{context}_{repname.replace(" ","").replace("-","")}.graphml'
          nx.write_graphml(H,gp)
         graph_checks.append({'representation':repname,'donor':donor,'context':context,'k':k,'n_nodes':len(nodes),'n_edges':len(edges),'duplicate_edges':0,'self_edges':0,'all_edge_attrs_finite':bool(np.isfinite(edges[['raw_cosine','community_weight']].to_numpy()).all()),'community_assignment_count':len(labels),'community_count':len(sizes)})
         log(f'graph {repname} {donor} {context} k={k}: nodes={len(nodes)} edges={len(edges)} comm={len(sizes)}')
        # Same node set by construction. Compare edge and mutual-edge sets, then partitions.
        for ra,rb in itertools.combinations(REPS,2):
         A=edge_sets[ra]; B=edge_sets[rb]; common=len(A&B); union=len(A|B)
         MA=mutual_sets[ra]; MB=mutual_sets[rb]; mcommon=len(MA&MB); munion=len(MA|MB)
         edge_agreements.append({'donor':donor,'context':context,'k':k,'representation_a':RNAME[ra],'representation_b':RNAME[rb],'common_edges':common,'union_edges':union,'edge_jaccard':common/union if union else 1.0,'mutual_common_edges':mcommon,'mutual_union_edges':munion,'mutual_edge_jaccard':mcommon/munion if munion else 1.0})
         aa=[label_by_rep[ra][t] for t in nodes]; bb=[label_by_rep[rb][t] for t in nodes]
         ari=float(adjusted_rand_score(aa,bb)); nmi=float(normalized_mutual_info_score(aa,bb))
         comm_agreements.append({'donor':donor,'context':context,'k':k,'representation_a':RNAME[ra],'representation_b':RNAME[rb],'n_nodes':len(nodes),'n_communities_a':ncomm_by_rep[ra],'n_communities_b':ncomm_by_rep[rb],'adjusted_rand_index':ari,'normalized_mutual_information':nmi})
    node_writer.close(); edge_writer.close(); comm_writer.close()
    # Tabular summaries.
    desc=pd.DataFrame(descriptors); desc.to_csv(OUT/'communities/graph_descriptors.tsv.gz',sep='\t',index=False,compression='gzip')
    ea=pd.DataFrame(edge_agreements); ca=pd.DataFrame(comm_agreements)
    ea.to_csv(OUT/'cross_representation/edge_agreement.tsv.gz',sep='\t',index=False,compression='gzip')
    ca.to_csv(OUT/'cross_representation/community_agreement.tsv.gz',sep='\t',index=False,compression='gzip')
    primary=ea[ea.k==15].merge(ca[ca.k==15],on=['donor','context','k','representation_a','representation_b'],validate='one_to_one')
    primary_summary=primary.groupby(['context','representation_a','representation_b'],as_index=False).agg(**{f'{col}_{stat}':(col,stat) for col in ['edge_jaccard','adjusted_rand_index','normalized_mutual_information'] for stat in ['mean','median','min','max']})
    primary_summary.to_csv(OUT/'cross_representation/k15_context_summary.tsv.gz',sep='\t',index=False,compression='gzip')
    sens=ea.merge(ca,on=['donor','context','k','representation_a','representation_b'],validate='one_to_one')
    metric_cols=['edge_jaccard','adjusted_rand_index','normalized_mutual_information']
    sens=sens.melt(id_vars=['donor','context','k','representation_a','representation_b'],value_vars=metric_cols,var_name='metric',value_name='value')
    sens=sens.pivot(index=['donor','context','representation_a','representation_b','metric'],columns='k',values='value').reset_index().rename(columns={10:'k10',15:'k15',30:'k30'})
    sens.to_csv(OUT/'cross_representation/k_sensitivity.tsv.gz',sep='\t',index=False,compression='gzip')
    # A representative GraphML load-back test only.
    one=next((OUT/'graphml/k15').glob('D1_Rest_PCA.graphml'))
    check=nx.read_graphml(one)
    graphml_pass=(check.number_of_nodes()==int(desc.query("donor=='D1' and context=='Rest' and representation=='PCA' and k==15").n_nodes.iloc[0]) and check.number_of_edges()==int(desc.query("donor=='D1' and context=='Rest' and representation=='PCA' and k==15").n_edges.iloc[0]))
    # Local analysis-specific sanity checks.
    valid=bool((ea.edge_jaccard.between(0,1).all()) and (ca.adjusted_rand_index.between(-1,1).all()) and (ca.normalized_mutual_information.between(0,1).all()) and graphml_pass)
    audit={'pass':valid,'preflight':preflight_stats,'global_selected_edge_cosine_min':mincos,'global_selected_edge_cosine_max':maxcos,'weighting_rule':weight_rule,'graph_count':len(desc),'expected_graph_count':144,'graphml_count':len(list((OUT/'graphml/k15').glob('*.graphml'))),'graphml_roundtrip_test':{'file':str(one),'pass':graphml_pass,'nodes':check.number_of_nodes(),'edges':check.number_of_edges()},'graph_local_checks':graph_checks,'edge_jaccard_range':[float(ea.edge_jaccard.min()),float(ea.edge_jaccard.max())],'ari_range':[float(ca.adjusted_rand_index.min()),float(ca.adjusted_rand_index.max())],'nmi_range':[float(ca.normalized_mutual_information.min()),float(ca.normalized_mutual_information.max())]}
    (OUT/'audits/GRAPH_ANALYSIS_LOCAL_CHECK.json').write_text(json.dumps(audit,indent=2)+'\n')
    summary={'status':'PASS' if valid else 'FAIL','graph_method':'undirected union kNN','k_values':KS,'primary_k':15,'community_algorithm':ALGORITHM,'random_seed':SEED,'resolution':RESOLUTION,'weighting_rule':weight_rule,'selected_edge_cosine_range':[mincos,maxcos],
             'graph_count':len(desc),'graphml_count':len(list((OUT/'graphml/k15').glob('*.graphml'))),'graphml_roundtrip_pass':graphml_pass,'edge_agreement':ea.to_dict('records'),'community_agreement':ca.to_dict('records'),'k15_context_summary':primary_summary.to_dict('records'),'k_sensitivity':sens.to_dict('records'),'runtime_seconds':time.time()-started,
             'outputs':[str(p) for p in sorted(OUT.rglob('*')) if p.is_file()]}
    (OUT/'reports/RESPONSE_GRAPH_GEOMETRY_SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n')
    lines=['# Response graph geometry summary','',f"**Status: {summary['status']}**",'',
           '## Method','',f"Undirected **union kNN** graphs from frozen exact neighborhoods; k=15 primary, k=10 and 30 sensitivity. Community detection used {ALGORITHM} with fixed seed 42, resolution 1.0, weighted by `community_weight`. Weight rule: {weight_rule}. No all-pair cosine tables or response vectors were recomputed.",'',
           '## Primary k=15 cross-representation agreement','', '| Donor | Context | Pair | Edge Jaccard | ARI | NMI |','|---|---|---|---:|---:|---:|']
    for r in primary.itertuples(index=False): lines.append(f'| {r.donor} | {r.context} | {r.representation_a}–{r.representation_b} | {r.edge_jaccard:.4f} | {r.adjusted_rand_index:.4f} | {r.normalized_mutual_information:.4f} |')
    lines += ['', '## Context summary','', 'Mean, median, minimum, and maximum across the four donor-specific graph comparisons are in `cross_representation/k15_context_summary.tsv.gz`.','',
              '## Graph descriptors and k sensitivity','', 'Per-graph node/edge/component/community descriptors are in `communities/graph_descriptors.tsv.gz`. k10/k15/k30 edge Jaccard, ARI, and NMI comparisons are in `cross_representation/k_sensitivity.tsv.gz`.','',
              '## GraphML','',f"Exported {summary['graphml_count']} primary k=15 graphs under `graphml/k15/`. One representative D1 × Rest PCA file round-tripped successfully. See `graphml/README.md` for Gephi/Cytoscape use. Static previews were skipped because full ~12k-node renderings are not useful at this stage.",'',
              'No spectral, cell-level, differential expression, pathway, or biological interpretation analysis was run.']
    (OUT/'reports/RESPONSE_GRAPH_GEOMETRY_SUMMARY.md').write_text('\n'.join(lines)+'\n')
    log(f'finished graph geometry in {time.time()-started:.1f}s; status={summary["status"]}')

if __name__=='__main__': main()
