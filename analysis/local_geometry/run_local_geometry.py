#!/usr/bin/env python3
"""Exact cross-representation local neighborhoods from frozen response vectors."""
from __future__ import annotations

import itertools
import json
import math
import os
import platform
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

ROOT = Path("source/derived/downstream")
RESP = ROOT / "response_construction_v1"
GEOM = ROOT / "response_geometry_v1"
OUT = ROOT / "response_local_geometry_v1"
REPS = ["pca", "scvi", "geneformer", "state"]
REP_NAMES = {"pca":"PCA", "scvi":"scVI", "geneformer":"Geneformer V2-316M", "state":"STATE SE-600M"}
EXPECTED_DIMS = {"pca":50, "scvi":50, "geneformer":1152, "state":2058}
DONORS = ["D1", "D2", "D3", "D4"]
CONTEXTS = ["Rest", "Stim8hr", "Stim48hr"]
KS = [10, 15, 30]
BLOCK_ROWS = int(os.environ.get("LOCAL_KNN_BLOCK_ROWS", "128"))


def say(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def write_tsv(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False, compression="gzip" if str(path).endswith(".gz") else None)


def cosine_knn(mat, targets, donor, context, rep):
    norms = np.linalg.norm(mat, axis=1)
    if not np.isfinite(mat).all() or not np.isfinite(norms).all() or (norms <= 0).any():
        raise ValueError(f"invalid/nonfinite or zero response norm in {rep} {donor} {context}")
    x = np.ascontiguousarray(mat / norms[:, None], dtype=np.float64)
    n = len(targets)
    out_target, out_neighbor, out_rank, out_cos = [], [], [], []
    cut_index = n - 30
    for lo in range(0, n, BLOCK_ROWS):
        hi = min(n, lo + BLOCK_ROWS)
        scores = x[lo:hi] @ x.T
        for local, row in enumerate(range(lo, hi)):
            s = scores[local]
            s[row] = -np.inf
            cutoff = np.partition(s, cut_index)[cut_index]
            above = np.flatnonzero(s > cutoff)
            tied = np.flatnonzero(s == cutoff)
            need = 30 - len(above)
            if need < len(tied):
                tied = sorted(tied.tolist(), key=lambda j: targets[j])[:need]
                chosen = np.concatenate((above, np.asarray(tied, dtype=np.int64)))
            else:
                chosen = np.concatenate((above, tied))
            order = np.lexsort((np.asarray(targets, dtype=object)[chosen], -s[chosen]))
            chosen = chosen[order]
            vals = np.clip(s[chosen], -1.0, 1.0)
            out_target.extend([targets[row]] * 30)
            out_neighbor.extend(targets[j] for j in chosen)
            out_rank.extend(range(1, 31))
            out_cos.extend(vals.tolist())
        if (lo // BLOCK_ROWS) % 20 == 0:
            say(f"{rep} {donor} {context}: encoded query rows {hi}/{n}")
    return pd.DataFrame({
        "representation": REP_NAMES[rep], "donor": donor, "context": context,
        "target": out_target, "neighbor": out_neighbor, "rank": out_rank,
        "cosine": np.asarray(out_cos, dtype=np.float64),
    })


def main():
    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    # Minimal preflight only: frozen row counts, dimensions, required schema.
    metadata, matrices = {}, {}
    for rep in REPS:
        vpath = RESP / rep / "response_vectors.npy"
        mpath = RESP / rep / "response_units.tsv.gz"
        if not vpath.is_file() or not mpath.is_file():
            raise FileNotFoundError(f"missing frozen input: {vpath if not vpath.is_file() else mpath}")
        mat = np.load(vpath, mmap_mode="r")
        md = pd.read_csv(mpath, sep="\t")
        if mat.ndim != 2 or mat.shape != (144368, EXPECTED_DIMS[rep]) or len(md) != 144368:
            raise ValueError(f"input shape mismatch for {rep}: {mat.shape}, metadata={len(md)}")
        if not {"target", "donor", "context"}.issubset(md.columns):
            raise ValueError(f"required metadata schema absent: {rep}")
        metadata[rep], matrices[rep] = md, mat
        say(f"preflight {rep}: {mat.shape}, dtype={mat.dtype}, metadata rows={len(md)}")

    # Cheap target support equality, as required, block by block.
    blocks = {}
    for donor in DONORS:
        for context in CONTEXTS:
            per_rep = {}
            for rep in REPS:
                md = metadata[rep]
                ix = np.flatnonzero((md.donor.to_numpy() == donor) & (md.context.to_numpy() == context))
                targets = md.iloc[ix].target.astype(str).tolist()
                if len(targets) != len(set(targets)):
                    raise ValueError(f"duplicate target in {rep} {donor} {context}")
                per_rep[rep] = (ix, targets)
            base = set(per_rep[REPS[0]][1])
            for rep in REPS[1:]:
                if set(per_rep[rep][1]) != base:
                    raise ValueError(f"target support differs in {donor} {context}: {REPS[0]} vs {rep}")
            blocks[(donor, context)] = per_rep
            say(f"target support {donor} {context}: {len(base)} targets")

    # Exact local neighborhoods.
    knns = {}
    for rep in REPS:
        rep_dir = OUT / rep
        rep_dir.mkdir(parents=True, exist_ok=True)
        for donor in DONORS:
            for context in CONTEXTS:
                ix, targets = blocks[(donor, context)][rep]
                # Consistent deterministic order independent of input row order.
                order = np.argsort(np.asarray(targets, dtype=str), kind="stable")
                targets = [targets[i] for i in order]
                vec = np.asarray(matrices[rep][ix[order], :], dtype=np.float64)
                frame = cosine_knn(vec, targets, donor, context, rep)
                if len(frame) != len(targets) * 30:
                    raise AssertionError("wrong number of top-30 rows")
                knns[(rep, donor, context)] = frame
                say(f"finished top30 {rep} {donor} {context}: {len(targets)} targets")
        all_knn = pd.concat([knns[(rep,d,c)] for d in DONORS for c in CONTEXTS], ignore_index=True)
        pq.write_table(pa.Table.from_pandas(all_knn, preserve_index=False), rep_dir / "knn_top30.parquet", compression="zstd")

    # Required local validation: deterministic ten focal targets in D1/Rest,
    # compared against only the four corresponding stored all-pair files.
    spot_rng = random.Random(20260930)
    spot_targets = sorted(blocks[("D1", "Rest")]["pca"][1])
    spot_targets = sorted(spot_rng.sample(spot_targets, 10))
    spot_rows = []
    spot_checks = []
    for rep in REPS:
        local = knns[(rep,"D1","Rest")]
        wanted = local[local.target.isin(spot_targets)]
        pair_path = GEOM / rep / "cosine_pairs" / "D1_Rest.parquet"
        pfile = pq.ParquetFile(pair_path)
        hits = []
        focal = pa.array(spot_targets)
        for batch in pfile.iter_batches(batch_size=1_000_000, columns=["target_a", "target_b", "cosine_similarity"]):
            a, b = batch.column(0), batch.column(1)
            mask = pc.or_(pc.is_in(a, value_set=focal), pc.is_in(b, value_set=focal))
            selected = batch.filter(mask).to_pydict()
            if selected["target_a"]:
                hits.append(pd.DataFrame(selected))
        edges = pd.concat(hits, ignore_index=True)
        for target in spot_targets:
            edge = edges[(edges.target_a == target) | (edges.target_b == target)].copy()
            edge["neighbor"] = np.where(edge.target_a == target, edge.target_b, edge.target_a)
            edge = edge.sort_values(["cosine_similarity", "neighbor"], ascending=[False, True], kind="stable").head(30)
            calc = wanted[wanted.target == target].sort_values("rank")
            same_ids = edge.neighbor.astype(str).tolist() == calc.neighbor.astype(str).tolist()
            maxdiff = float(np.max(np.abs(edge.cosine_similarity.to_numpy() - calc.cosine.to_numpy())))
            passed = same_ids and maxdiff <= 1e-10
            spot_checks.append({"representation":REP_NAMES[rep],"target":target,"same_neighbor_ids":same_ids,"max_abs_cosine_diff":maxdiff,"pass":bool(passed)})
            spot_rows.extend({"representation":REP_NAMES[rep],"target":target,"neighbor":n,"knn_cosine":float(cv),"stored_cosine":float(sv)} for n,cv,sv in zip(calc.neighbor,calc.cosine,edge.cosine_similarity))
    spot_pass = all(r["pass"] for r in spot_checks)
    (OUT / "audits").mkdir(exist_ok=True)
    (OUT / "audits" / "LOCAL_KNN_SPOTCHECK.json").write_text(json.dumps({"seed":20260930,"donor":"D1","context":"Rest","focal_targets":spot_targets,"checks":spot_checks,"pass":spot_pass},indent=2)+"\n")
    if not spot_pass:
        raise RuntimeError("D1/Rest top-neighbor spot check failed; stopping before agreement analysis")
    say("D1/Rest ten-target all-pair spot check PASS")

    # Set-valued overlap for every focal target, pair, and k.
    rep_pairs = list(itertools.combinations(REPS, 2))
    target_rows, context_summaries, by_block = [], [], []
    for donor in DONORS:
        for context in CONTEXTS:
            targets = sorted(blocks[(donor,context)]["pca"][1])
            n = len(targets)
            neigh = {}
            for rep in REPS:
                f = knns[(rep,donor,context)]
                neigh[rep] = {t:g.sort_values("rank").neighbor.astype(str).tolist() for t,g in f.groupby("target", sort=False)}
            for ra, rb in rep_pairs:
                vals_by_k = {}
                for k in KS:
                    for target in targets:
                        a, b = set(neigh[ra][target][:k]), set(neigh[rb][target][:k])
                        inter = len(a & b)
                        union = len(a | b)
                        frac = inter / k
                        expected = k / (n - 1)
                        row = {"donor":donor,"context":context,"representation_a":REP_NAMES[ra],"representation_b":REP_NAMES[rb],"target":target,"k":k,
                               "overlap_count":inter,"overlap_fraction":frac,"jaccard":inter/union,
                               "expected_random_overlap_count":k*k/(n-1),"expected_random_overlap_fraction":expected,
                               "neighborhood_enrichment":frac/expected}
                        target_rows.append(row)
                        vals_by_k.setdefault(k, []).append(frac)
                    arr = np.asarray(vals_by_k[k], dtype=float)
                    by_block.append({"donor":donor,"context":context,"representation_a":REP_NAMES[ra],"representation_b":REP_NAMES[rb],"k":k,"n_targets":n,
                                     "mean_overlap_fraction":float(arr.mean()),"median_overlap_fraction":float(np.median(arr)),"sd_overlap_fraction":float(arr.std(ddof=1)),
                                     "q25_overlap_fraction":float(np.quantile(arr,.25)),"q75_overlap_fraction":float(np.quantile(arr,.75)),"min_overlap_fraction":float(arr.min()),"max_overlap_fraction":float(arr.max()),
                                     "expected_random_overlap_fraction":k/(n-1),"expected_random_overlap_count":k*k/(n-1),"enrichment_over_random":float(arr.mean()/(k/(n-1)))})
    targ = pd.DataFrame(target_rows)
    (OUT/"cross_representation").mkdir(parents=True, exist_ok=True)
    targ.to_parquet(OUT/"cross_representation"/"target_level_knn_agreement.parquet",index=False,compression="zstd")
    block_df = pd.DataFrame(by_block)
    write_tsv(block_df, OUT/"cross_representation"/"agreement_by_donor_context.tsv.gz")
    ctx = block_df.groupby(["context","representation_a","representation_b","k"],as_index=False).agg(
        mean_across_donors=("mean_overlap_fraction","mean"),median_across_donors=("mean_overlap_fraction","median"),min_across_donors=("mean_overlap_fraction","min"),max_across_donors=("mean_overlap_fraction","max"),
        mean_random_expectation=("expected_random_overlap_fraction","mean"),mean_enrichment=("enrichment_over_random","mean"))
    write_tsv(ctx, OUT/"cross_representation"/"agreement_by_context_summary.tsv.gz")

    # k=30 join to frozen global Spearman table.
    global_path = GEOM/"cross_representation"/"agreement_by_donor_context.tsv"
    global_df = pd.read_csv(global_path,sep="\t")
    global_df["representation_a"] = global_df.representation_a.astype(str).str.replace("scVI", "scVI", regex=False)
    global30 = targ[targ.k == 30].groupby(["donor","context","representation_a","representation_b"],as_index=False).agg(local_mean_overlap=("overlap_fraction","mean"),local_median_overlap=("overlap_fraction","median"))
    # Existing geometry naming is PCA/scVI/Geneformer/STATE.
    global_vs_local = global30.merge(global_df[["donor","context","representation_a","representation_b","spearman_rho"]],on=["donor","context","representation_a","representation_b"],how="left",validate="one_to_one")
    if global_vs_local.spearman_rho.isna().any():
        # Pair orientation may differ; normalize both orientations explicitly.
        swapped = global_df.rename(columns={"representation_a":"representation_b","representation_b":"representation_a"})
        missing = global_vs_local.spearman_rho.isna()
        lookup = global30.loc[missing].merge(swapped[["donor","context","representation_a","representation_b","spearman_rho"]],on=["donor","context","representation_a","representation_b"],how="left",validate="one_to_one")
        global_vs_local.loc[missing,"spearman_rho"] = lookup.spearman_rho.to_numpy()
    if global_vs_local.spearman_rho.isna().any():
        raise ValueError("could not align k30 local table with frozen global agreement table")
    write_tsv(global_vs_local,OUT/"cross_representation"/"global_vs_local_agreement.tsv.gz")

    # Target stability: mean overlap across six rep-pairs per block, plus per pair top/bottom 20.
    target_mean = targ[targ.k==30].groupby(["donor","context","target"],as_index=False).agg(mean_overlap_across_representation_pairs=("overlap_fraction","mean"))
    target_mean["record_type"] = "target_mean_across_six_pairs"
    topbottom=[]
    for (d,c,ra,rb),g in targ[targ.k==30].groupby(["donor","context","representation_a","representation_b"]):
        high=g.sort_values(["overlap_fraction","target"],ascending=[False,True]).head(20).assign(record_type="top20_pair_overlap")
        low=g.sort_values(["overlap_fraction","target"],ascending=[True,True]).head(20).assign(record_type="bottom20_pair_overlap")
        topbottom.extend(high.to_dict("records")); topbottom.extend(low.to_dict("records"))
    tb=pd.DataFrame(topbottom)
    tb["mean_overlap_across_representation_pairs"] = np.nan
    stable=pd.concat([target_mean, tb],ignore_index=True,sort=False)
    write_tsv(stable,OUT/"cross_representation"/"stable_unstable_targets.tsv.gz")

    # Local-only invariants.
    sanity = {"pass":True,"blocks":{},"knn_rows":{}}
    for rep in REPS:
        f=pd.concat([knns[(rep,d,c)] for d in DONORS for c in CONTEXTS],ignore_index=True)
        assert np.isfinite(f.cosine).all()
        for (d,c,t),g in f.groupby(["donor","context","target"],sort=False):
            g=g.sort_values("rank")
            if g.neighbor.eq(t).any() or g.neighbor.duplicated().any() or g["rank"].tolist()!=list(range(1,31)) or not g.cosine.is_monotonic_decreasing:
                raise AssertionError(f"kNN invariant failure {rep} {d} {c} {t}")
        sanity["knn_rows"][rep]=len(f)
    if ((targ.overlap_fraction<0)|(targ.overlap_fraction>1)|(targ.jaccard<0)|(targ.jaccard>1)).any():
        raise AssertionError("overlap values out of [0,1]")
    sanity["pass"] = True
    (OUT/"audits"/"LOCAL_KNN_AUDIT.json").write_text(json.dumps(sanity,indent=2)+"\n")

    # Summary and provenance.
    support=[]
    for (d,c),per in blocks.items(): support.append({"donor":d,"context":c,"n_targets":len(per["pca"][1]),"expected_pairs":len(per["pca"][1])*(len(per["pca"][1])-1)//2})
    magnitudes=[]
    for rep in REPS:
        md=metadata[rep]
        magnitudes.append({"representation":REP_NAMES[rep],"min_response_l2_norm":float(md.response_l2_norm.min()),"median_response_l2_norm":float(md.response_l2_norm.median()),"max_response_l2_norm":float(md.response_l2_norm.max())})
    report={"status":"PASS","method":"exact CPU blocked NumPy/BLAS inner product on float64 L2-normalized canonical response vectors; deterministic score-descending then lexical-neighbor tie sort","k_values":KS,"preflight":{"units_per_representation":144368,"dimensions":EXPECTED_DIMS},"target_counts":support,
            "spotcheck":json.loads((OUT/"audits"/"LOCAL_KNN_SPOTCHECK.json").read_text()),"magnitude_norms":magnitudes,"runtime_seconds":time.time()-started,"python":sys.version,"platform":platform.platform(),"block_rows":BLOCK_ROWS,
            "outputs":[str(p.relative_to(Path.cwd())) for p in sorted(OUT.rglob("*")) if p.is_file()]}
    (OUT/"reports").mkdir(exist_ok=True)
    (OUT/"reports"/"RESPONSE_LOCAL_GEOMETRY_SUMMARY.json").write_text(json.dumps(report,indent=2)+"\n")

    lines=["# Local response geometry summary","",f"**Status: {report['status']}**","", "## Method", "",report["method"],"", "Neighborhood sizes: k=10, 15, 30. The analytical independent-random baseline is k/(n−1); no permutations were run.","", "## Target counts by donor × context", "", "| Donor | Context | Targets | Expected unordered edges |", "|---|---|---:|---:|"]
    for x in support: lines.append(f"| {x['donor']} | {x['context']} | {x['n_targets']:,} | {x['expected_pairs']:,} |")
    lines += ["", "## Cross-representation neighborhood overlap", "", "The following table gives the mean of the four donor-level mean overlap fractions for each context and pair, plus mean donor-level enrichment over the analytical random expectation. Full donor-specific distributions are in `cross_representation/agreement_by_donor_context.tsv.gz`.","", "| Context | Pair | k | Mean overlap | Median across donor means | Random expectation | Enrichment |", "|---|---|---:|---:|---:|---:|---:|"]
    for _,r in ctx.iterrows(): lines.append(f"| {r.context} | {r.representation_a}–{r.representation_b} | {r.k} | {r.mean_across_donors:.4f} | {r.median_across_donors:.4f} | {r.mean_random_expectation:.5f} | {r.mean_enrichment:.2f}× |")
    lines += ["", "## Global versus local agreement", "", "The k=30 comparison with frozen global Spearman values is in `cross_representation/global_vs_local_agreement.tsv.gz`. This is descriptive; no predictive model or representation ranking was fit.","", "## Stable and variable target neighborhoods", "", "`cross_representation/stable_unstable_targets.tsv.gz` contains per-target mean overlap across all six representation pairs and the top/bottom 20 target identities per donor, context, and representation pair.","", "## Validation", "", "The deterministic 10-target D1 × Rest check against stored pairwise cosine values passed for all four representations. Local neighborhood invariants passed: 30 non-self unique neighbors per target, ranks 1–30, descending finite cosine scores, and overlap/Jaccard in [0,1].", "", f"Exact CPU calculation runtime: {report['runtime_seconds']/3600:.2f} hours. Outputs are under `{OUT}/`. "]
    (OUT/"reports"/"RESPONSE_LOCAL_GEOMETRY_SUMMARY.md").write_text("\n".join(lines)+"\n")
    say(f"completed local geometry in {time.time()-started:.1f}s; outputs {OUT}")


if __name__ == "__main__":
    main()
