"""Summarize saved opposite-half components; never load cells or response arrays."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

HERE = Path('results/bibm_extensions_20261005/integrated/matched_half_calibration').resolve()
SOURCE = HERE / 'matched_half_calibration.tsv'
OUT = HERE / 'self_cross_summary'
OUT.mkdir(exist_ok=True)
SEED = 20261006
B = 10000
f = pd.read_csv(SOURCE, sep='\t')
f = f[f.metric.isin(['sampled_map_spearman', 'O15'])].copy()
assert np.allclose(f.cross_mean, (f.cross_AB + f.cross_BA) / 2, atol=1e-15, rtol=0)
keys = ['representation', 'donor', 'context', 'split', 'metric']
parts = []
for side in ['R', 'S']:
    parts.append(f[[f'representation_{side}', 'donor', 'context', 'split', 'metric', f'within_{side}']].rename(columns={f'representation_{side}': 'representation', f'within_{side}': 'agreement'}))
self_long = pd.concat(parts, ignore_index=True)
assert self_long.groupby(keys).agreement.nunique().eq(1).all()
self_long = self_long.drop_duplicates(keys)
assert self_long.groupby(['representation', 'metric']).size().eq(240).all()
assert f.groupby(['representation_R', 'representation_S', 'metric']).size().eq(240).all()
assert set(f.split) == set(range(1, 21))
# Shared donor resamples keep all component estimates paired. Contexts and
# overlapping partitions remain together within each sampled donor.
donors = sorted(f.donor.unique())
draws = np.random.default_rng(SEED).integers(0, len(donors), size=(B, len(donors)))

def summarize(data, groups, value):
    rows = []
    for key, g in data.groupby(groups, sort=True):
        if not isinstance(key, tuple):
            key = (key,)
        cluster = g.groupby('donor')[value].mean().reindex(donors).to_numpy()
        assert np.isfinite(cluster).all()
        lo, hi = np.quantile(cluster[draws].mean(axis=1), [.025, .975])
        row = dict(zip(groups, key))
        row.update(mean=g[value].mean(), CI95_low=lo, CI95_high=hi,
                   SD_block_split=g[value].std(ddof=1), donor_mean_min=cluster.min(),
                   donor_mean_max=cluster.max(), n_values=len(g))
        rows.append(row)
    return pd.DataFrame(rows)

ss = summarize(self_long, ['metric', 'representation'], 'agreement')
cs = summarize(f, ['metric', 'representation_R', 'representation_S'], 'cross_mean')
direction = f.groupby(['metric', 'representation_R', 'representation_S'])[['cross_AB', 'cross_BA']].mean().reset_index()
cs = cs.merge(direction, on=['metric', 'representation_R', 'representation_S'])
comparison = []
for r in cs.itertuples():
    a = ss[(ss.metric == r.metric) & (ss.representation == r.representation_R)].iloc[0]
    b = ss[(ss.metric == r.metric) & (ss.representation == r.representation_S)].iloc[0]
    flag = 'below_both' if r.mean < min(a['mean'], b['mean']) else 'above_both' if r.mean > max(a['mean'], b['mean']) else 'between'
    comparison.append(dict(metric=r.metric, representation_R=r.representation_R,
                           representation_S=r.representation_S, self_R=a['mean'],
                           self_S=b['mean'], cross=r.mean, classification=flag))
comparison = pd.DataFrame(comparison)
self_long.to_csv(OUT / 'self_agreement_components.tsv', sep='\t', index=False)
f.drop(columns=['within_mean', 'representation_gap']).to_csv(OUT / 'cross_agreement_components.tsv', sep='\t', index=False)
for name, table in [('self_agreement_summary', ss), ('cross_agreement_summary', cs), ('pair_comparison', comparison)]:
    table.to_csv(OUT / (name + '.tsv'), sep='\t', index=False)
for metric in ['sampled_map_spearman', 'O15']:
    for name, table in [('self', ss), ('cross', cs), ('comparison', comparison)]:
        table[table.metric == metric].to_csv(OUT / f'{metric}_{name}.tsv', sep='\t', index=False)
(OUT / 'SUMMARY_AUDIT.json').write_text(json.dumps(dict(
    source=str(SOURCE), source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
    rerun_split_analysis=False, metrics=['sampled_map_spearman', 'O15'],
    splits=20, donors=donors, contexts=sorted(f.context.unique()),
    values_per_estimate=240, bootstrap_replicates=B, bootstrap_seed=SEED,
    bootstrap_method='whole donor percentile bootstrap; all contexts and splits retained together; shared resamples across estimates',
    self_duplicates_verified_identical=True, cross_formula_verified=True,
    old_gap_used=False), indent=2) + '\n')
print(ss.to_string(index=False))
print(cs.to_string(index=False))
print(comparison.to_string(index=False))
