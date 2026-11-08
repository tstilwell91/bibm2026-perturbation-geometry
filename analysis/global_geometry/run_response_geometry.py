#!/usr/bin/env python3
"""V1 response-level cosine geometry, response magnitudes, and agreement QC."""
from __future__ import annotations

import csv
import gzip
import hashlib
import itertools
import json
import os
import platform
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy import stats

ROOT = Path('.').resolve()
RESP = ROOT / "source/derived/downstream/response_construction_v1"
OUT = ROOT / "source/derived/downstream/response_geometry_v1"
SUMMARY = RESP / "reports/RESPONSE_CONSTRUCTION_SUMMARY.json"
INDEX_AUDIT = ROOT / "source/derived/downstream/common_cell_audit_v1/COMMON_DOWNSTREAM_CELL_AUDIT.json"
EXPECTED_UNITS = 144_368
EXPECTED_TARGET_CELLS = 20_771_051
EXPECTED_NTC_CELLS = 939_535
EXPECTED_STRATA = 284
DONORS = ["D1", "D2", "D3", "D4"]
CONTEXTS = ["Rest", "Stim8hr", "Stim48hr"]
REPRESENTATIONS = [
    {"name": "PCA", "dir": "pca", "dim": 50},
    {"name": "scVI", "dir": "scvi", "dim": 50},
    {"name": "Geneformer V2-316M", "dir": "geneformer", "dim": 1152},
    {"name": "STATE SE-600M", "dir": "state", "dim": 2058},
]
REP_PAIRS = list(itertools.combinations([r["name"] for r in REPRESENTATIONS], 2))
DONOR_PAIRS = list(itertools.combinations(DONORS, 2))
CONTEXT_PAIRS = list(itertools.combinations(CONTEXTS, 2))
PARQUET_ROW_CHUNK = 1_000_000
TRIANGLE_ROW_WINDOW = 128


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(tmp, path)


def csv_gz(path: Path, fields, rows) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(tmp, "wt", newline="", compresslevel=1) as f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)


def expected_sha_from_construction(rep_audit, filename):
    return rep_audit["outputs"][filename + "_sha256"]


def read_and_validate_inputs():
    summary = json.loads(SUMMARY.read_text())
    if summary.get("status") != "PASS":
        raise RuntimeError("Canonical response-construction summary is not PASS")
    if summary["counts"].get("response_units") != EXPECTED_UNITS or summary["counts"].get("target_cells") != EXPECTED_TARGET_CELLS:
        raise RuntimeError("Canonical response summary counts do not match the frozen counts")
    common_audit = json.loads(INDEX_AUDIT.read_text())
    if common_audit.get("status") != "PASS" or not common_audit.get("cross_representation_set_equality", {}).get("pass"):
        raise RuntimeError("Common-cell identity audit is not PASS")

    ref_records = None
    ref_block_targets = None
    ref_target_ids = None
    rep_data = {}
    target_to_gene_id = {}
    unit_metadata_by_key = {}
    block_units = {}
    all_targets = set()
    target_counts_by_donor = {d: 0 for d in DONORS}
    target_counts_by_context = {c: 0 for c in CONTEXTS}

    for rep in REPRESENTATIONS:
        name, sub, dim = rep["name"], rep["dir"], rep["dim"]
        d = RESP / sub
        audit = json.loads((d / "RESPONSE_CONSTRUCTION_AUDIT.json").read_text())
        if audit.get("status") != "PASS" or audit.get("response_units") != EXPECTED_UNITS:
            raise RuntimeError(f"Response-construction audit failed for {name}")
        for filename in ("response_vectors.npy", "response_units.tsv.gz", "ntc_centroids.npy", "ntc_centroid_units.tsv.gz"):
            p = d / filename
            if sha256(p) != expected_sha_from_construction(audit, filename):
                raise RuntimeError(f"Canonical response artifact checksum mismatch: {p}")
        vpath = d / "response_vectors.npy"
        vectors = np.load(vpath, mmap_mode="r", allow_pickle=False)
        if vectors.shape != (EXPECTED_UNITS, dim) or vectors.dtype != np.float64:
            raise RuntimeError(f"Unexpected canonical response shape/dtype for {name}: {vectors.shape}/{vectors.dtype}")
        centroid = np.load(d / "ntc_centroids.npy", mmap_mode="r", allow_pickle=False)
        if centroid.shape != (EXPECTED_STRATA, dim) or centroid.dtype != np.float64 or not np.isfinite(centroid).all():
            raise RuntimeError(f"Invalid centroid artifact for {name}")
        del centroid

        records = []
        seen_keys = set()
        vectors_finite = True
        stored_norms = np.empty(EXPECTED_UNITS, dtype=np.float64)
        support_counts = np.empty(EXPECTED_UNITS, dtype=np.int64)
        block_targets = {(d0, c0): [] for d0 in DONORS for c0 in CONTEXTS}
        block_rows = {(d0, c0): [] for d0 in DONORS for c0 in CONTEXTS}
        rep_target_counts = {d0: 0 for d0 in DONORS}
        rep_context_counts = {c0: 0 for c0 in CONTEXTS}

        with gzip.open(d / "response_units.tsv.gz", "rt", newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row_number, row in enumerate(reader):
                if row_number >= EXPECTED_UNITS or int(row["response_unit"]) != row_number:
                    raise RuntimeError(f"Invalid unit row order/count in {name}")
                target, gene_id = row["target"], row["target_gene_id"]
                donor, context = row["donor"], row["context"]
                n_cells, row_dim = int(row["n_target_cells"]), int(row["dimensions"])
                if donor not in DONORS or context not in CONTEXTS or row_dim != dim or n_cells <= 0:
                    raise RuntimeError(f"Invalid response-unit metadata in {name}: {row}")
                key = (target, donor, context)
                if key in seen_keys:
                    raise RuntimeError(f"Duplicate response unit in {name}: {key}")
                seen_keys.add(key)
                old_gene_id = target_to_gene_id.setdefault(target, gene_id)
                if old_gene_id != gene_id:
                    raise RuntimeError(f"Target symbol maps to conflicting IDs: {target}")
                target, gene_id = str(target), str(gene_id)
                lane_count = int(row["n_contributing_lanes"])
                lanes = row["contributing_lanes"].split(";") if row["contributing_lanes"] else []
                lane_cells = row["target_cells_by_lane"].split(";") if row["target_cells_by_lane"] else []
                if lane_count != len(lanes) or len(lane_cells) != lane_count:
                    raise RuntimeError(f"Lane metadata mismatch in {name}, unit {row_number}")
                if sum(int(x.rsplit(":", 1)[1]) for x in lane_cells) != n_cells:
                    raise RuntimeError(f"Lane target counts do not sum to response support in {name}, unit {row_number}")
                digest = row["contributing_cell_ids_sha256"]
                if len(digest) != 64 or row["finite"] != "true":
                    raise RuntimeError(f"Invalid finite/identity metadata in {name}, unit {row_number}")
                record = {"response_unit": row_number, "target": target, "target_gene_id": gene_id,
                          "donor": donor, "context": context, "n_target_cells": n_cells,
                          "n_contributing_lanes": lane_count, "contributing_lanes": row["contributing_lanes"],
                          "target_cells_by_lane": row["target_cells_by_lane"], "cell_ids_sha256": digest,
                          "stored_norm": float(row["response_l2_norm"])}
                records.append(record)
                stored_norms[row_number] = record["stored_norm"]
                support_counts[row_number] = n_cells
                block_targets[(donor, context)].append(target)
                block_rows[(donor, context)].append(row_number)
                all_targets.add(target)
                rep_target_counts[donor] += n_cells
                rep_context_counts[context] += n_cells
        if len(records) != EXPECTED_UNITS or len(seen_keys) != EXPECTED_UNITS:
            raise RuntimeError(f"Response unit count mismatch in {name}: {len(records)}")
        if np.any(support_counts <= 0) or int(support_counts.sum()) != EXPECTED_TARGET_CELLS:
            raise RuntimeError(f"Response support count mismatch in {name}")
        # Independently read every response vector, verify finiteness and reproduce construction QC norms.
        actual_norms = np.empty(EXPECTED_UNITS, dtype=np.float64)
        for start in range(0, EXPECTED_UNITS, 8192):
            end = min(EXPECTED_UNITS, start + 8192)
            block = np.asarray(vectors[start:end])
            finite_rows = np.isfinite(block).all(axis=1)
            vectors_finite &= bool(finite_rows.all())
            actual_norms[start:end] = np.linalg.norm(block, axis=1)
        if not vectors_finite or not np.isfinite(actual_norms).all() or np.any(actual_norms <= 0):
            raise RuntimeError(f"Nonfinite or zero-norm response vector in {name}")
        max_norm_delta = float(np.max(np.abs(actual_norms - stored_norms)))
        if max_norm_delta > 1e-10:
            raise RuntimeError(f"Response magnitude QC norms differ for {name}: max delta {max_norm_delta}")

        for block_key in block_targets:
            pairs = sorted(zip(block_targets[block_key], block_rows[block_key]))
            block_targets[block_key] = [x[0] for x in pairs]
            block_rows[block_key] = [x[1] for x in pairs]

        signature = [(x["target"], x["target_gene_id"], x["donor"], x["context"], x["n_target_cells"],
                      x["n_contributing_lanes"], x["contributing_lanes"], x["target_cells_by_lane"], x["cell_ids_sha256"])
                     for x in records]
        if ref_records is None:
            ref_records, ref_block_targets, ref_target_ids = records, block_targets, dict(target_to_gene_id)
            for x in records:
                unit_metadata_by_key[(x["target"], x["donor"], x["context"])] = x
            for k in block_targets:
                block_targets[k] = sorted(block_targets[k])
            ref_block_targets = block_targets
        else:
            ref_signature = [(x["target"], x["target_gene_id"], x["donor"], x["context"], x["n_target_cells"],
                              x["n_contributing_lanes"], x["contributing_lanes"], x["target_cells_by_lane"], x["cell_ids_sha256"])
                             for x in ref_records]
            if signature != ref_signature:
                raise RuntimeError(f"Canonical response-unit/support/identity mismatch for {name}")
            for k in block_targets:
                block_targets[k] = sorted(block_targets[k])
                if block_targets[k] != ref_block_targets[k]:
                    raise RuntimeError(f"Target membership differs in {name}, donor/context={k}")
        if rep_target_counts != {d0: summary["counts"]["target_cells_by_donor"][d0] for d0 in DONORS}:
            raise RuntimeError(f"Target cell counts by donor differ in {name}")
        if rep_context_counts != {c0: summary["counts"]["target_cells_by_context"][c0] for c0 in CONTEXTS}:
            raise RuntimeError(f"Target cell counts by context differ in {name}")
        target_counts_by_donor = rep_target_counts
        target_counts_by_context = rep_context_counts
        rep_data[name] = {"vectors": vectors, "records": records, "norms": actual_norms,
                          "block_targets": block_targets, "block_rows": {k: np.asarray(v, dtype=np.int64) for k, v in block_rows.items()},
                          "artifact": str(vpath.relative_to(ROOT)), "metadata": str((d / "response_units.tsv.gz").relative_to(ROOT)),
                          "artifact_sha256": sha256(vpath), "metadata_sha256": sha256(d / "response_units.tsv.gz"),
                          "audit": str((d / "RESPONSE_CONSTRUCTION_AUDIT.json").relative_to(ROOT)),
                          "audit_status": "PASS", "dtype": str(vectors.dtype), "shape": list(vectors.shape),
                          "max_norm_difference_from_response_audit": max_norm_delta}

    if len(all_targets) != 12_604:
        raise RuntimeError(f"Expected 12,604 unique perturbation targets; found {len(all_targets)}")
    for rep in REPRESENTATIONS:
        for d0 in DONORS:
            for c0 in CONTEXTS:
                t = rep_data[rep["name"]]["block_targets"][(d0, c0)]
                if len(t) != len(set(t)):
                    raise RuntimeError(f"Duplicate target in geometry block {rep['name']} {d0}/{c0}")
    for rep in REPRESENTATIONS[1:]:
        for d0 in DONORS:
            for c0 in CONTEXTS:
                if rep_data[rep["name"]]["block_targets"][(d0, c0)] != rep_data["PCA"]["block_targets"][(d0, c0)]:
                    raise RuntimeError(f"Representation-specific target set in {rep['name']} {d0}/{c0}")

    return {"summary": summary, "rep_data": rep_data, "targets": sorted(all_targets),
            "target_to_gene_id": target_to_gene_id,
            "target_cells_by_donor": target_counts_by_donor,
            "target_cells_by_context": target_counts_by_context,
            "all_target_count": len(all_targets)}


def do_input_audit(inputs):
    target_counts = inputs["rep_data"]["PCA"]["block_targets"]
    blocks = []
    edge_total = 0
    all_donors_targets = {}
    all_context_targets = {}
    for d in DONORS:
        all_donors_targets[d] = sorted(set(t for c in CONTEXTS for t in target_counts[(d, c)]))
    for c in CONTEXTS:
        all_context_targets[c] = sorted(set(t for d in DONORS for t in target_counts[(d, c)]))
    coverage_rows = []
    missing_rows = []
    all_targets = inputs["targets"]
    for d in DONORS:
        for c in CONTEXTS:
            targets = target_counts[(d, c)]
            edges = len(targets) * (len(targets) - 1) // 2
            edge_total += edges
            missing = sorted(set(all_targets) - set(targets))
            miss_sha = hashlib.sha256("\n".join(missing).encode()).hexdigest()
            blocks.append({"donor": d, "context": c, "n_targets": len(targets), "n_response_units": len(targets),
                           "n_unique_pairs": edges, "expected_global_targets": len(all_targets),
                           "missing_global_targets": len(missing), "missing_targets_sha256": miss_sha})
            coverage_rows.append({"donor": d, "context": c, "n_targets": len(targets), "n_response_units": len(targets),
                                  "n_unique_pairs": edges, "global_target_count": len(all_targets),
                                  "missing_target_count": len(missing), "missing_targets_sha256": miss_sha})
            for target in missing:
                missing_rows.append({"donor": d, "context": c, "missing_target": target})
    if sum(x["n_response_units"] for x in blocks) != EXPECTED_UNITS:
        raise RuntimeError("Donor/context block counts do not sum to 144,368")
    geometry_bytes_float64 = edge_total * len(REPRESENTATIONS) * 8
    audit = {
        "status": "PASS", "canonical_response_summary": str(SUMMARY.relative_to(ROOT)),
        "response_construction_summary_sha256": sha256(SUMMARY),
        "common_cell_identity_audit": str(INDEX_AUDIT.relative_to(ROOT)),
        "common_cell_audit_status": "PASS",
        "representations": {name: {k: v for k, v in data.items() if k not in {"vectors", "records", "norms", "block_targets", "block_rows"}}
                            for name, data in inputs["rep_data"].items()},
        "response_units_expected": EXPECTED_UNITS, "response_units_found_per_representation": {k: len(v["records"]) for k, v in inputs["rep_data"].items()},
        "response_unit_metadata_equality": "PASS", "target_support_identity_hash_equality": "PASS",
        "unique_targets": inputs["all_target_count"],
        "target_cells_by_donor": inputs["target_cells_by_donor"],
        "target_cells_by_context": inputs["target_cells_by_context"],
        "donor_context_blocks": blocks, "unique_upper_triangle_pairs_per_representation": edge_total,
        "all_representations_unique_pair_rows_total": edge_total * len(REPRESENTATIONS),
        "cosine_value_bytes_float64_before_parquet_compression": geometry_bytes_float64,
        "coverage_varies_by_donor_context": any(x["missing_global_targets"] > 0 for x in blocks),
        "do_not_interpret_missing_support_as_representation_failure": True,
    }
    write_json(OUT / "audits/RESPONSE_GEOMETRY_INPUT_AUDIT.json", audit)
    lines = ["# Response geometry input audit", "", "**Status: PASS**", "",
             f"Canonical source: `{SUMMARY.relative_to(ROOT)}`. Response-construction summary SHA-256: `{audit['response_construction_summary_sha256']}`.",
             f"Common-cell audit: `{INDEX_AUDIT.relative_to(ROOT)}` (**PASS**). No older response tables were used.", "",
             "| Representation | Canonical vector path | Unit metadata path | Shape | Dtype | Units | Unit/support/identity match |",
             "|---|---|---|---:|---|---:|---|"]
    for r in REPRESENTATIONS:
        x = inputs["rep_data"][r["name"]]
        lines.append(f"| {r['name']} | `{x['artifact']}` | `{x['metadata']}` | {x['shape'][0]} × {x['shape'][1]} | {x['dtype']} | {len(x['records']):,} | PASS |")
    lines += ["", f"Unique perturbation targets: {inputs['all_target_count']:,}. Response units: {EXPECTED_UNITS:,} in each representation; target/donor/context, cell support, lane support, and cell-ID hashes match exactly.", "",
              "## Donor × context blocks", "", "| Donor | Context | Targets / units | Upper-triangle pairs | Globally unsupported targets |",
              "|---|---|---:|---:|---:|"]
    for b in blocks:
        lines.append(f"| {b['donor']} | {b['context']} | {b['n_targets']:,} | {b['n_unique_pairs']:,} | {b['missing_global_targets']:,} |")
    lines += ["", f"Pairs per representation: {edge_total:,}; all four: {edge_total*4:,}. Float64 similarity values alone would occupy {geometry_bytes_float64/1e9:.1f} GB before Parquet compression.",
              "Targets without support in a donor × context block are reported in `target_coverage_missing.tsv.gz`; support differences across blocks are biological coverage, not representation-specific drops.", ""]
    (OUT / "RESPONSE_GEOMETRY_INPUT_AUDIT.md").write_text("\n".join(lines))
    csv_gz(OUT / "reports/target_coverage_by_block.tsv.gz",
           ["donor", "context", "n_targets", "n_response_units", "n_unique_pairs", "global_target_count", "missing_target_count", "missing_targets_sha256"], coverage_rows)
    csv_gz(OUT / "reports/target_coverage_missing.tsv.gz", ["donor", "context", "missing_target"], missing_rows)
    with (OUT / "reports/target_order.tsv").open("w") as f:
        f.write("target_order\ttarget\ttarget_gene_id\n")
        for i, t in enumerate(inputs["targets"]):
            f.write(f"{i}\t{t}\t{inputs['target_to_gene_id'][t]}\n")
    return audit, blocks


def cosine_edge_chunks(n, window=TRIANGLE_ROW_WINDOW):
    """Yield lexicographically ordered upper-triangle local row indices."""
    for i0 in range(0, n - 1, window):
        i1 = min(n - 1, i0 + window)
        ii = np.repeat(np.arange(i0, i1, dtype=np.int32), n - np.arange(i0, i1) - 1)
        jj = np.concatenate([np.arange(i + 1, n, dtype=np.int32) for i in range(i0, i1)])
        if len(ii) != len(jj):
            raise RuntimeError("Upper-triangle edge-index generation error")
        yield ii, jj


def parquet_schema():
    dstr = pa.dictionary(pa.int8(), pa.string())
    dtarget = pa.dictionary(pa.int32(), pa.string())
    return pa.schema([("donor", dstr), ("context", dstr), ("target_a", dtarget),
                      ("target_b", dtarget), ("cosine_similarity", pa.float64())])


def write_cosine_block(path, donor, context, target_codes, global_targets, cos, block_key_hash):
    n = len(target_codes)
    n_edges = n * (n - 1) // 2
    values = np.empty(n_edges, dtype=np.float64)
    cursor = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    schema = parquet_schema()
    dictionary = pa.array(global_targets, type=pa.string())
    didx = DONORS.index(donor)
    cidx = CONTEXTS.index(context)
    writer = pq.ParquetWriter(tmp, schema, compression="zstd", compression_level=3,
                              use_dictionary=True, write_statistics=True)
    try:
        for ii, jj in cosine_edge_chunks(n):
            m = len(ii)
            edge_vals = np.asarray(cos[ii, jj], dtype=np.float64)
            if not np.isfinite(edge_vals).all() or np.any(edge_vals < -1.0 - 1e-10) or np.any(edge_vals > 1.0 + 1e-10):
                raise RuntimeError(f"Invalid cosine value in {donor}/{context}")
            edge_vals = np.clip(edge_vals, -1.0, 1.0)
            values[cursor:cursor + m] = edge_vals
            cursor += m
            di = pa.DictionaryArray.from_arrays(pa.array(np.full(m, didx, dtype=np.int8)), pa.array(DONORS))
            ci = pa.DictionaryArray.from_arrays(pa.array(np.full(m, cidx, dtype=np.int8)), pa.array(CONTEXTS))
            ta = pa.DictionaryArray.from_arrays(pa.array(target_codes[ii], type=pa.int32()), dictionary)
            tb = pa.DictionaryArray.from_arrays(pa.array(target_codes[jj], type=pa.int32()), dictionary)
            table = pa.Table.from_arrays([di, ci, ta, tb, pa.array(edge_vals)], schema=schema)
            writer.write_table(table, row_group_size=m)
        if cursor != n_edges:
            raise RuntimeError(f"Wrong upper-triangle row count in {path}: {cursor} vs {n_edges}")
    finally:
        writer.close()
    os.replace(tmp, path)
    return values, n_edges, block_key_hash


def key_order_hash(targets, target_to_idx):
    h = hashlib.sha256()
    codes = np.asarray([target_to_idx[t] for t in targets], dtype=np.int32)
    for ii, jj in cosine_edge_chunks(len(targets)):
        keys = (codes[ii].astype(np.uint64) << 32) | codes[jj].astype(np.uint64)
        h.update(keys.tobytes())
    return h.hexdigest()


def correlation_values(a, b):
    if len(a) != len(b) or len(a) < 3:
        raise RuntimeError("Correlation inputs have incompatible or insufficient edge counts")
    with warnings.catch_warnings():
        warnings.simplefilter("error", stats.ConstantInputWarning)
        sr = stats.spearmanr(a, b)
        pr = stats.pearsonr(a, b)
    return {"spearman_rho": float(sr.statistic), "spearman_p": float(sr.pvalue),
            "pearson_r": float(pr.statistic), "pearson_p": float(pr.pvalue)}


def magnitude_outputs(inputs):
    summaries = {}
    for rep in REPRESENTATIONS:
        name, sub = rep["name"], rep["dir"]
        records, norms = inputs["rep_data"][name]["records"], inputs["rep_data"][name]["norms"]
        rows = []
        for i, rec in enumerate(records):
            rows.append({"representation": name, "target": rec["target"], "target_gene_id": rec["target_gene_id"],
                         "donor": rec["donor"], "context": rec["context"],
                         "n_target_cells": rec["n_target_cells"], "response_magnitude": format(float(norms[i]), ".17g")})
        path = OUT / sub / "response_magnitudes.tsv.gz"
        csv_gz(path, ["representation", "target", "target_gene_id", "donor", "context", "n_target_cells", "response_magnitude"], rows)
        summaries[name] = {"finite": bool(np.isfinite(norms).all()), "zero_values": int(np.sum(norms == 0)),
                           "minimum": float(norms.min()), "median": float(np.median(norms)),
                           "maximum": float(norms.max()), "path": str(path.relative_to(ROOT)),
                           "sha256": sha256(path)}
    return summaries


def compute_cosine_and_cross_agreement(inputs, blocks):
    target_to_idx = {t: i for i, t in enumerate(inputs["targets"])}
    cross_rows = []
    block_audits = []
    for block in blocks:
        donor, context = block["donor"], block["context"]
        targets = inputs["rep_data"]["PCA"]["block_targets"][(donor, context)]
        rows_by_rep = {}
        n = len(targets)
        n_edges = n * (n - 1) // 2
        pair_hash = key_order_hash(targets, target_to_idx)
        rep_upper = {}
        observed = []
        for rep in REPRESENTATIONS:
            name, dim = rep["name"], rep["dim"]
            data = inputs["rep_data"][name]
            targets_here = data["block_targets"][(donor, context)]
            if targets_here != targets:
                raise RuntimeError(f"Target order differs before cosine for {name}, {donor}/{context}")
            rows = data["block_rows"][(donor, context)]
            # Canonical unit rows are loaded in the deterministic target order.
            matrix = np.asarray(data["vectors"][rows], dtype=np.float64)
            if matrix.shape != (n, dim) or not np.isfinite(matrix).all():
                raise RuntimeError(f"Invalid response matrix block for {name}, {donor}/{context}")
            norms = np.linalg.norm(matrix, axis=1)
            if not np.isfinite(norms).all() or np.any(norms <= 0):
                raise RuntimeError(f"Zero/nonfinite response norm in {name}, {donor}/{context}")
            unit = matrix / norms[:, None]
            sim = unit @ unit.T
            diag_error = float(np.max(np.abs(np.diag(sim) - 1.0)))
            lo, hi = float(sim.min()), float(sim.max())
            if diag_error > 1e-10 or lo < -1.0 - 1e-10 or hi > 1.0 + 1e-10 or not np.isfinite(sim).all():
                raise RuntimeError(f"Cosine matrix sanity check failed for {name}, {donor}/{context}: diag={diag_error}, range={lo}:{hi}")
            block_path = OUT / rep["dir"] / "cosine_pairs" / f"{donor}_{context}.parquet"
            codes = np.asarray([target_to_idx[t] for t in targets], dtype=np.int32)
            edge_vals, saved_edges, _ = write_cosine_block(block_path, donor, context, codes, inputs["targets"], sim, pair_hash)
            if saved_edges != n_edges or len(edge_vals) != n_edges:
                raise RuntimeError(f"Cosine pair-count mismatch for {name}, {donor}/{context}")
            rep_upper[name] = edge_vals
            print(json.dumps({"event": "cosine_representation_complete", "representation": name,
                              "donor": donor, "context": context, "targets": n,
                              "unique_pairs": saved_edges, "parquet": str(block_path.relative_to(ROOT))}), flush=True)
            observed.append({"representation": name, "dimensions": dim, "targets": n,
                             "response_units": n, "pair_count": saved_edges,
                             "diagonal_max_abs_error": diag_error, "cosine_min": lo, "cosine_max": hi,
                             "pair_key_order_sha256": pair_hash,
                             "parquet": str(block_path.relative_to(ROOT))})
            del matrix, norms, unit, sim

        # Explicitly compare the six representation pairs over the identical target-pair order.
        for a_name, b_name in REP_PAIRS:
            a, b = rep_upper[a_name], rep_upper[b_name]
            if len(a) != n_edges or len(b) != n_edges or not np.isfinite(a).all() or not np.isfinite(b).all():
                raise RuntimeError(f"Cross-representation edge vector mismatch in {donor}/{context}")
            corr = correlation_values(a, b)
            cross_rows.append({"donor": donor, "context": context, "representation_a": a_name,
                               "representation_b": b_name, "n_pairwise_edges": n_edges,
                               "spearman_rho": corr["spearman_rho"], "spearman_p": corr["spearman_p"],
                               "pearson_r": corr["pearson_r"], "pearson_p": corr["pearson_p"],
                               "pair_key_order_sha256": pair_hash, "target_order_sha256": hashlib.sha256("\n".join(targets).encode()).hexdigest()})
            print(json.dumps({"event": "cross_rep_correlation_complete", "donor": donor, "context": context,
                              "representation_a": a_name, "representation_b": b_name,
                              "spearman_rho": corr["spearman_rho"]}), flush=True)
        block_audits.append({"donor": donor, "context": context, "target_count": n,
                             "response_unit_count": n, "unique_pair_count": n_edges,
                             "expected_pair_count": n * (n - 1) // 2,
                             "representations": observed, "pair_key_order_sha256": pair_hash,
                             "target_order_sha256": hashlib.sha256("\n".join(targets).encode()).hexdigest(),
                             "pass": len(observed) == 4 and all(x["pair_count"] == n_edges for x in observed)})
        rep_upper.clear()
        del a, b, edge_vals
        print(json.dumps({"event": "cosine_block_complete", "donor": donor, "context": context,
                          "targets": n, "unique_pairs": n_edges, "cross_rep_comparisons": 6}), flush=True)

    fields = ["donor", "context", "representation_a", "representation_b", "n_pairwise_edges",
              "spearman_rho", "spearman_p", "pearson_r", "pearson_p", "pair_key_order_sha256", "target_order_sha256"]
    out_rows = [{k: format(v, ".17g") if isinstance(v, float) else v for k, v in row.items()} for row in cross_rows]
    with (OUT / "cross_representation/agreement_by_donor_context.tsv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter="\t", lineterminator="\n")
        w.writeheader(); w.writerows(out_rows)
    context_summary = []
    for context in CONTEXTS:
        for a_name, b_name in REP_PAIRS:
            group = [x for x in cross_rows if x["context"] == context and x["representation_a"] == a_name and x["representation_b"] == b_name]
            vals = np.asarray([x["spearman_rho"] for x in group], dtype=np.float64)
            context_summary.append({"context": context, "representation_a": a_name, "representation_b": b_name,
                                    "n_donors": len(vals), "n_pairwise_edges_per_donor": ";".join(f"{x['donor']}:{x['n_pairwise_edges']}" for x in group),
                                    "donor_specific_spearman": ";".join(f"{x['donor']}:{x['spearman_rho']:.12g}" for x in group),
                                    "mean_spearman": float(vals.mean()), "median_spearman": float(np.median(vals)),
                                    "min_spearman": float(vals.min()), "max_spearman": float(vals.max())})
    csv_gz(OUT / "cross_representation/agreement_by_context_summary.tsv.gz",
           ["context", "representation_a", "representation_b", "n_donors", "n_pairwise_edges_per_donor",
            "donor_specific_spearman", "mean_spearman", "median_spearman", "min_spearman", "max_spearman"], context_summary)
    return cross_rows, context_summary, block_audits


def load_filtered_edge_values(path, allowed_targets, global_target_index, expected_edges):
    values_parts = []
    key_hasher = hashlib.sha256()
    kept = 0
    parquet = pq.ParquetFile(path, read_dictionary=["target_a", "target_b"])
    for batch in parquet.iter_batches(batch_size=PARQUET_ROW_CHUNK,
                                     columns=["target_a", "target_b", "cosine_similarity"]):
        ca, cb = batch.column(0), batch.column(1)
        if pa.types.is_dictionary(ca.type):
            da = np.asarray([global_target_index[x] for x in ca.dictionary.to_pylist()], dtype=np.int32)
            db = np.asarray([global_target_index[x] for x in cb.dictionary.to_pylist()], dtype=np.int32)
            ac = da[np.asarray(ca.indices)]
            bc = db[np.asarray(cb.indices)]
        else:
            ac = np.fromiter((global_target_index[x] for x in ca.to_pylist()), dtype=np.int32, count=len(ca))
            bc = np.fromiter((global_target_index[x] for x in cb.to_pylist()), dtype=np.int32, count=len(cb))
        keep = allowed_targets[ac] & allowed_targets[bc]
        if keep.any():
            keys = (ac[keep].astype(np.uint64) << 32) | bc[keep].astype(np.uint64)
            key_hasher.update(keys.tobytes())
            vals = batch.column(2).to_numpy(zero_copy_only=False)
            selected = np.asarray(vals[keep], dtype=np.float64)
            values_parts.append(selected)
            kept += len(selected)
    if kept != expected_edges:
        raise RuntimeError(f"Overlap edge selection yielded {kept}, expected {expected_edges}: {path}")
    return np.concatenate(values_parts), key_hasher.hexdigest()


def reproducibility(inputs):
    targets = inputs["targets"]
    target_index = {t: i for i, t in enumerate(targets)}
    n_global = len(targets)
    coverage = inputs["rep_data"]["PCA"]["block_targets"]
    donor_rows = []
    context_rows = []
    overlap_rows = []

    def compare_blocks(kind, rep, label_a, label_b, key_a, key_b, target_a, target_b, out_rows):
        common = sorted(set(target_a).intersection(target_b))
        if len(common) < 2:
            raise RuntimeError(f"Too few common targets for {kind} {rep['name']} {label_a}/{label_b}")
        edge_n = len(common) * (len(common) - 1) // 2
        allowed = np.zeros(n_global, dtype=np.bool_)
        allowed[np.asarray([target_index[x] for x in common], dtype=np.int64)] = True
        block_dir = OUT / rep["dir"] / "cosine_pairs"
        path_a = block_dir / f"{key_a[0]}_{key_a[1]}.parquet"
        path_b = block_dir / f"{key_b[0]}_{key_b[1]}.parquet"
        va, ha = load_filtered_edge_values(path_a, allowed, target_index, edge_n)
        vb, hb = load_filtered_edge_values(path_b, allowed, target_index, edge_n)
        if ha != hb or len(va) != len(vb):
            raise RuntimeError(f"Reproducibility pair keys/order did not match for {rep['name']} {kind} {label_a}/{label_b}")
        corr = correlation_values(va, vb)
        target_digest = hashlib.sha256("\n".join(common).encode()).hexdigest()
        common_id = f"{kind}|{rep['name']}|{label_a}|{label_b}"
        for t in common:
            overlap_rows.append({"comparison_id": common_id, "comparison_type": kind,
                                 "representation": rep["name"], "block_a": label_a, "block_b": label_b,
                                 "target": t})
        out_rows.append({"representation": rep["name"], "comparison_type": kind,
                         "block_a": label_a, "block_b": label_b,
                         "n_common_targets": len(common), "n_common_pairwise_edges": edge_n,
                         "spearman_rho": corr["spearman_rho"], "spearman_p": corr["spearman_p"],
                         "pearson_r": corr["pearson_r"], "pearson_p": corr["pearson_p"],
                         "common_target_set_sha256": target_digest,
                         "common_pair_order_sha256": ha})

    for rep in REPRESENTATIONS:
        for context in CONTEXTS:
            for d_a, d_b in DONOR_PAIRS:
                ta, tb = coverage[(d_a, context)], coverage[(d_b, context)]
                compare_blocks("donor", rep, f"{d_a}|{context}", f"{d_b}|{context}",
                               (d_a, context), (d_b, context), ta, tb, donor_rows)
        for donor in DONORS:
            for c_a, c_b in CONTEXT_PAIRS:
                ta, tb = coverage[(donor, c_a)], coverage[(donor, c_b)]
                compare_blocks("context", rep, f"{donor}|{c_a}", f"{donor}|{c_b}",
                               (donor, c_a), (donor, c_b), ta, tb, context_rows)
        print(json.dumps({"event": "reproducibility_complete", "representation": rep["name"],
                          "donor_comparisons": 18, "context_comparisons": 12}), flush=True)

    for rows in (donor_rows, context_rows):
        for row in rows:
            for k, v in list(row.items()):
                if isinstance(v, float):
                    row[k] = format(v, ".17g")
    csv_gz(OUT / "donor_reproducibility/donor_geometry_agreement.tsv.gz",
           ["representation", "comparison_type", "block_a", "block_b", "n_common_targets", "n_common_pairwise_edges",
            "spearman_rho", "spearman_p", "pearson_r", "pearson_p", "common_target_set_sha256", "common_pair_order_sha256"], donor_rows)
    csv_gz(OUT / "context_reproducibility/context_geometry_agreement.tsv.gz",
           ["representation", "comparison_type", "block_a", "block_b", "n_common_targets", "n_common_pairwise_edges",
            "spearman_rho", "spearman_p", "pearson_r", "pearson_p", "common_target_set_sha256", "common_pair_order_sha256"], context_rows)
    csv_gz(OUT / "audits/reproducibility_overlap_targets.tsv.gz",
           ["comparison_id", "comparison_type", "representation", "block_a", "block_b", "target"], overlap_rows)
    return donor_rows, context_rows, overlap_rows


def main():
    started = datetime.now(timezone.utc)
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    for rep in REPRESENTATIONS:
        (OUT / rep["dir"] / "cosine_pairs").mkdir(parents=True, exist_ok=True)
    for folder in ("cross_representation", "donor_reproducibility", "context_reproducibility", "reports", "audits"):
        (OUT / folder).mkdir(parents=True, exist_ok=True)
    inputs = read_and_validate_inputs()
    input_audit, blocks = do_input_audit(inputs)
    print(json.dumps({"event": "input_audit_pass", "units": EXPECTED_UNITS,
                      "targets": inputs["all_target_count"], "block_count": len(blocks),
                      "pairs_per_representation": input_audit["unique_upper_triangle_pairs_per_representation"]}), flush=True)
    if sys.argv[1:] == ["--audit-only"]:
        return
    magnitude_summary = magnitude_outputs(inputs)
    cross_rows, context_summary, block_audits = compute_cosine_and_cross_agreement(inputs, blocks)
    donor_rows, context_rows, overlap_rows = reproducibility(inputs)

    # Checksum each major block artifact after closing the Parquet writers.
    output_checksums = {}
    for rep in REPRESENTATIONS:
        name, sub = rep["name"], rep["dir"]
        for donor in DONORS:
            for context in CONTEXTS:
                path = OUT / sub / "cosine_pairs" / f"{donor}_{context}.parquet"
                output_checksums[str(path.relative_to(ROOT))] = sha256(path)
        mag_path = OUT / sub / "response_magnitudes.tsv.gz"
        output_checksums[str(mag_path.relative_to(ROOT))] = sha256(mag_path)
    for rel in ["cross_representation/agreement_by_donor_context.tsv",
                "cross_representation/agreement_by_context_summary.tsv.gz",
                "donor_reproducibility/donor_geometry_agreement.tsv.gz",
                "context_reproducibility/context_geometry_agreement.tsv.gz",
                "reports/target_coverage_by_block.tsv.gz", "reports/target_coverage_missing.tsv.gz",
                "reports/target_order.tsv", "audits/reproducibility_overlap_targets.tsv.gz"]:
        output_checksums[str((OUT / rel).relative_to(ROOT))] = sha256(OUT / rel)

    magnitude_summaries = {}
    for name, x in magnitude_summary.items():
        magnitude_summaries[name] = {k: x[k] for k in ("finite", "zero_values", "minimum", "median", "maximum")}
    cosine_sanity = all(x["pass"] and len(x["representations"]) == 4 for x in block_audits)
    cross_sanity = len(cross_rows) == 12 * 6 and all(x["n_pairwise_edges"] == next(
        b["n_unique_pairs"] for b in blocks if b["donor"] == x["donor"] and b["context"] == x["context"])
        for x in cross_rows)
    reproducibility_sanity = len(donor_rows) == 4 * 3 * 6 and len(context_rows) == 4 * 4 * 3
    if not cosine_sanity or not cross_sanity or not reproducibility_sanity:
        raise RuntimeError("One or more geometry sanity checks failed")

    # Context summary table keeps donor-specific values and descriptive summaries.
    for row in context_summary:
        row["mean_spearman"] = format(row["mean_spearman"], ".17g")
        row["median_spearman"] = format(row["median_spearman"], ".17g")
        row["min_spearman"] = format(row["min_spearman"], ".17g")
        row["max_spearman"] = format(row["max_spearman"], ".17g")

    summary = {
        "schema": "response_geometry_v1", "status": "PASS",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "inputs": {"canonical_response_root": str(RESP.relative_to(ROOT)),
                   "response_construction_summary": str(SUMMARY.relative_to(ROOT)),
                   "response_construction_summary_sha256": sha256(SUMMARY),
                   "common_cell_audit": str(INDEX_AUDIT.relative_to(ROOT)),
                   "response_specification": str((RESP / "RESPONSE_CONSTRUCTION_SPEC.md").relative_to(ROOT)),
                   "representation_inputs": {name: {"vectors": d["artifact"], "metadata": d["metadata"],
                                                   "vectors_sha256": d["artifact_sha256"],
                                                   "metadata_sha256": d["metadata_sha256"],
                                                   "shape": d["shape"], "dtype": d["dtype"], "units": len(d["records"]),
                                                   "audit": d["audit"], "audit_status": d["audit_status"]}
                                               for name, d in inputs["rep_data"].items()}},
        "target_coverage": {"total_unique_targets": inputs["all_target_count"],
                            "targets_by_donor": {d: len(set(t for c in CONTEXTS for t in inputs["rep_data"]["PCA"]["block_targets"][(d,c)])) for d in DONORS},
                            "targets_by_context": {c: len(set(t for d in DONORS for t in inputs["rep_data"]["PCA"]["block_targets"][(d,c)])) for c in CONTEXTS},
                            "donor_context_blocks": blocks,
                            "n_target_cells_by_donor": inputs["target_cells_by_donor"],
                            "n_target_cells_by_context": inputs["target_cells_by_context"],
                            "response_unit_support_distribution": {
                                "min": int(min(x["n_target_cells"] for x in inputs["rep_data"]["PCA"]["records"])),
                                "median": float(np.median([x["n_target_cells"] for x in inputs["rep_data"]["PCA"]["records"]])),
                                "mean": float(np.mean([x["n_target_cells"] for x in inputs["rep_data"]["PCA"]["records"]])),
                                "max": int(max(x["n_target_cells"] for x in inputs["rep_data"]["PCA"]["records"]))}},
        "response_units": {"expected": EXPECTED_UNITS, "found_per_representation": EXPECTED_UNITS,
                           "cross_representation_exact_equality": "PASS"},
        "cosine_geometry": {"primary_block": "donor x context", "blocks": block_audits,
                            "n_unique_upper_triangle_pairs_per_representation": input_audit["unique_upper_triangle_pairs_per_representation"],
                            "pair_rows_total_all_representations": input_audit["all_representations_unique_pair_rows_total"],
                            "pair_sanity": "PASS"},
        "response_magnitudes": magnitude_summaries,
        "cross_representation_agreement": {"primary_statistic": "Spearman", "secondary_statistic": "Pearson",
                                            "comparisons": cross_rows, "context_summary": context_summary,
                                            "status": "PASS"},
        "donor_reproducibility": {"comparisons": donor_rows, "status": "PASS"},
        "context_reproducibility": {"comparisons": context_rows, "status": "PASS"},
        "sanity_checks": {"input_audit": "PASS", "response_unit_equality": "PASS", "diagonal_and_range": "PASS",
                          "finite_cosines": "PASS", "pair_counts": "PASS", "no_duplicate_pairs": "PASS",
                          "edge_order_equal_across_representations": "PASS",
                          "magnitudes_finite_positive": "PASS", "reproducibility_overlap_documented": "PASS"},
        "output_checksums_sha256": output_checksums,
        "provenance": {"script": os.path.relpath(Path(__file__).resolve(), ROOT), "script_sha256": sha256(Path(__file__)),
                       "runtime": {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__,
                                  "pyarrow": pa.__version__, "scipy": stats.__version__ if hasattr(stats, "__version__") else __import__("scipy").__version__},
                       "created_utc": started.isoformat(timespec="seconds"), "elapsed_seconds": round(time.time()-t0,1),
                       "cosine_output_dtype": "float64", "magnitude_dtype": "float64"},
        "prohibited_downstream_analyses_started": [],
    }
    write_json(OUT / "reports/RESPONSE_GEOMETRY_SUMMARY.json", summary)
    # A readable report preserves all 72 donor/context cross-representation results.
    md = ["# Response-level geometry summary", "", "**Status: PASS**", "",
          "This stage contains response-level cosine geometry, response magnitudes, cross-representation agreement, donor/context reproducibility, and QC only. No neighborhoods, graphs, communities, spectra, cell-level geometry, DE, biological program summaries, or figures were run.", "",
          "## Input audit", "", "**PASS** — all four canonical response matrices contain 144,368 exact target × donor × context units with identical target support, target-cell counts, lane contributions, and contributing-cell ID hashes.", "",
          "## Target coverage", "", f"Unique targets represented: {inputs['all_target_count']:,}.", "",
          "| Donor | Targets across contexts | Target cells |", "|---|---:|---:|"]
    for d in DONORS:
        md.append(f"| {d} | {summary['target_coverage']['targets_by_donor'][d]:,} | {inputs['target_cells_by_donor'][d]:,} |")
    md += ["", "| Context | Targets across donors | Target cells |", "|---|---:|---:|"]
    for c in CONTEXTS:
        md.append(f"| {c} | {summary['target_coverage']['targets_by_context'][c]:,} | {inputs['target_cells_by_context'][c]:,} |")
    md += ["", "### Donor × context blocks", "", "| Donor | Context | Targets / units | Unique pairs | Missing global targets |", "|---|---|---:|---:|---:|"]
    for b in blocks:
        md.append(f"| {b['donor']} | {b['context']} | {b['n_targets']:,} | {b['n_unique_pairs']:,} | {b['missing_global_targets']:,} |")
    md += ["", "## Response magnitudes", "", "| Representation | Finite | Zero | Minimum | Median | Maximum |", "|---|---|---:|---:|---:|---:|"]
    for name, x in magnitude_summaries.items():
        md.append(f"| {name} | {x['finite']} | {x['zero_values']} | {x['minimum']:.8g} | {x['median']:.8g} | {x['maximum']:.8g} |")
    md += ["", "## Cross-representation agreement", "", "Primary statistic: Spearman correlation of the identical upper-triangle perturbation-pair cosine values; Pearson is secondary.", ""]
    for context in CONTEXTS:
        md += [f"### {context}", "", "| Donor | Pair | Edges | Spearman ρ | Pearson r |", "|---|---|---:|---:|---:|"]
        for row in cross_rows:
            if row["context"] == context:
                md.append(f"| {row['donor']} | {row['representation_a']} vs {row['representation_b']} | {row['n_pairwise_edges']:,} | {row['spearman_rho']:.6g} | {row['pearson_r']:.6g} |")
        md.append("")
        md += ["| Representation pair | Mean ρ | Median ρ | Min | Max | Donor values |", "|---|---:|---:|---:|---:|---|"]
        for row in context_summary:
            if row["context"] == context:
                md.append(f"| {row['representation_a']} vs {row['representation_b']} | {float(row['mean_spearman']):.6g} | {float(row['median_spearman']):.6g} | {float(row['min_spearman']):.6g} | {float(row['max_spearman']):.6g} | {row['donor_specific_spearman']} |")
        md.append("")
    md += ["## Donor geometry reproducibility", "", "Spearman values are descriptive; each row records the exact common target set and common pair count used.", "", "| Representation | Context | Donors | Common targets | Common edges | Spearman ρ | Pearson r |", "|---|---|---|---:|---:|---:|---:|"]
    for r in donor_rows:
        md.append(f"| {r['representation']} | {r['block_b'].split('|')[1]} | {r['block_a']} vs {r['block_b']} | {r['n_common_targets']} | {r['n_common_pairwise_edges']} | {float(r['spearman_rho']):.6g} | {float(r['pearson_r']):.6g} |")
    md += ["", "## Context geometry reproducibility", "", "| Representation | Donor | Contexts | Common targets | Common edges | Spearman ρ | Pearson r |", "|---|---|---|---:|---:|---:|---:|"]
    for r in context_rows:
        md.append(f"| {r['representation']} | {r['block_a'].split('|')[0]} | {r['block_a'].split('|')[1]} vs {r['block_b'].split('|')[1]} | {r['n_common_targets']} | {r['n_common_pairwise_edges']} | {float(r['spearman_rho']):.6g} | {float(r['pearson_r']):.6g} |")
    md += ["", "## Output locations", "", "Cosine upper-triangle pairs are stored in per-representation `cosine_pairs/<donor>_<context>.parquet` files. Response magnitudes are in `response_magnitudes.tsv.gz`; target coverage and exact reproducibility overlaps are in `reports/` and `audits/`.", "", "No biological interpretation or model ranking is provided at this stage.", ""]
    (OUT / "reports/RESPONSE_GEOMETRY_SUMMARY.md").write_text("\n".join(md))
    write_json(OUT / "audits/RESPONSE_GEOMETRY_AUDIT.json", {
        "status": "PASS", "input_audit": "PASS", "cosine_blocks": len(block_audits),
        "cross_representation_comparisons": len(cross_rows), "donor_reproducibility_comparisons": len(donor_rows),
        "context_reproducibility_comparisons": len(context_rows), "all_sanity_checks_pass": True,
        "geometry_stage_only": True, "no_downstream_geometry_started": True,
    })
    print(json.dumps({"status": "PASS", "blocks": len(block_audits),
                      "pairs_per_representation": input_audit["unique_upper_triangle_pairs_per_representation"],
                      "cross_rep_comparisons": len(cross_rows), "donor_comparisons": len(donor_rows),
                      "context_comparisons": len(context_rows), "elapsed_seconds": round(time.time()-t0,1)}), flush=True)


if __name__ == "__main__":
    main()
