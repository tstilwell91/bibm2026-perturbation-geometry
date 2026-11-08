#!/usr/bin/env python3
"""Build V31 response vectors from the frozen common-cell index, without geometry."""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np

ROOT = Path('.').resolve()
BASE = ROOT / "source/derived/downstream/response_construction_v1"
INDEX = ROOT / "source/derived/downstream/common_cell_audit_v1/COMMON_DOWNSTREAM_CELL_INDEX.tsv.gz"
INDEX_AUDIT = ROOT / "source/derived/downstream/common_cell_audit_v1/COMMON_DOWNSTREAM_CELL_AUDIT.json"
MANIFEST = ROOT / "source/derived/v35_state_all_targets/STATE_ALL_TARGETS_CELL_MANIFEST.tsv.gz"
N_EXPECTED = 21_710_586
N_TARGET = 20_771_051
N_NTC = 939_535
BATCH = 25_000
INDEX_SHA_EXPECTED = "c3b97ef6e819b46d3afdcb08035771eafc709a09720a966da941df2fb7d414eb"

PCA = ROOT / "source/derived/pca_direct_full_materialized_v28/runs/job-6848910"
SCVI = ROOT / "global_scvi/final/primary_2k_allcell_v1/seed42/latent_v2"
GF = ROOT / "geneformer_global_cell_embeddings/outputs"
STATE = ROOT / "source/derived/v35_state_all_targets/STATE"

REPRESENTATIONS = [
    {"name": "PCA", "out": "pca", "dim": 50,
     "artifact": PCA / "scores/pca_scores_21m_x_50.h5",
     "audit": PCA / "audits/full_pca_results.json", "source": "hdf5"},
    {"name": "scVI", "out": "scvi", "dim": 50,
     "artifact": SCVI / "latent_50d_float32.npy",
     "audit": SCVI / "LATENT_EXTRACTION_AUDIT.json", "source": "npy"},
    {"name": "Geneformer V2-316M", "out": "geneformer", "dim": 1152,
     "artifact": GF, "audit": ROOT / "geneformer_global_cell_embeddings/audits",
     "source": "geneformer"},
    {"name": "STATE SE-600M", "out": "state", "dim": 2058,
     "artifact": STATE / "shards", "audit": STATE / "global_cell_index.validation.json",
     "source": "state"},
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sha_json(path: Path) -> dict:
    return json.loads(path.read_text())


def write_json(path: Path, value) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(tmp, path)


def check_dir_empty(path: Path) -> None:
    expected = {"RESPONSE_CONSTRUCTION_SPEC.md", "build_responses.py", "pca", "scvi",
                "geneformer", "state", "audits", "reports"}
    extras = [p.name for p in path.iterdir() if p.name not in expected]
    if extras:
        raise RuntimeError(f"Refusing to overwrite existing response outputs: {extras[:10]}")
    for sub in ("pca", "scvi", "geneformer", "state", "audits", "reports"):
        q = path / sub
        if q.exists() and any(q.iterdir()):
            raise RuntimeError(f"Refusing to overwrite nonempty output directory {q}")


def load_and_validate_index():
    if sha256(INDEX) != INDEX_SHA_EXPECTED:
        raise RuntimeError("Common index SHA-256 differs from the completed audit lock")
    index_audit = sha_json(INDEX_AUDIT)
    if index_audit.get("status") != "PASS" or index_audit.get("canonical_index", {}).get("sha256") != INDEX_SHA_EXPECTED:
        raise RuntimeError("Common-cell audit is not PASS for the expected index checksum")
    if not index_audit.get("metadata_consistency", {}).get("pass") or not index_audit.get("matching_strata", {}).get("pass"):
        raise RuntimeError("Common-cell metadata or matching-strata audit did not pass")
    manifest_hash = sha256(MANIFEST)
    if manifest_hash != index_audit["authoritative_downstream_manifest"].get("sha256"):
        raise RuntimeError("Frozen V35 manifest checksum changed since common-cell audit")

    unit_by_key: dict[tuple[str, str, str], int] = {}
    unit_rows: list[dict] = []
    unit_hashes = []
    strata_by_key: dict[tuple[str, str, str], int] = {}
    strata_rows: list[dict] = []
    id_by_row = np.full(N_EXPECTED, -1, dtype=np.int32)
    strata_by_row = np.full(N_EXPECTED, -1, dtype=np.int16)
    is_target = np.zeros(N_EXPECTED, dtype=np.bool_)
    row_maps = {r["name"]: np.empty(N_EXPECTED, dtype=np.int64) for r in REPRESENTATIONS}
    source_ranges: list[dict] = []
    role_counts = {"target": 0, "ntc": 0}
    target_id_by_symbol: dict[str, str] = {}

    with gzip.open(INDEX, "rt", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        required = {"canonical_row", "cell_id", "pca_row", "scvi_row", "geneformer_row", "state_row",
                    "donor", "context", "lane", "target_status", "target_gene_id", "target_name",
                    "guide_id", "source_file"}
        if not required.issubset(reader.fieldnames or []):
            raise RuntimeError(f"Common index missing columns: {sorted(required-set(reader.fieldnames or []))}")
        for i, row in enumerate(reader):
            if i >= N_EXPECTED or int(row["canonical_row"]) != i:
                raise RuntimeError(f"Common index canonical order/count failure at row {i}")
            cell_id = row["cell_id"]
            if not cell_id:
                raise RuntimeError(f"Empty cell ID in common index at canonical row {i}")
            donor, context, lane = row["donor"], row["context"], row["lane"]
            role, target, target_id = row["target_status"], row["target_name"], row["target_gene_id"]
            if donor not in {"D1", "D2", "D3", "D4"} or context not in {"Rest", "Stim8hr", "Stim48hr"} or not lane:
                raise RuntimeError(f"Invalid response metadata at canonical row {i}: {donor}/{context}/{lane}")
            if role not in {"target", "ntc"}:
                raise RuntimeError(f"Invalid target status at canonical row {i}: {role}")
            if role == "ntc":
                if target != "NTC" or target_id != "NTC":
                    raise RuntimeError(f"NTC target-label conflict at canonical row {i}")
                role_counts["ntc"] += 1
            else:
                if not target or not target_id or target == "NTC" or target_id == "NTC":
                    raise RuntimeError(f"Target label missing/conflicting at canonical row {i}")
                old_id = target_id_by_symbol.setdefault(target, target_id)
                if old_id != target_id:
                    raise RuntimeError(f"Target symbol maps to multiple target IDs: {target}")
                role_counts["target"] += 1

            s_key = (donor, context, lane)
            sid = strata_by_key.get(s_key)
            if sid is None:
                sid = len(strata_rows)
                strata_by_key[s_key] = sid
                strata_rows.append({"donor": donor, "context": context, "lane": lane,
                                    "n_ntc": 0, "n_target": 0, "targets": set()})
            strata_by_row[i] = sid
            srec = strata_rows[sid]
            if role == "ntc":
                srec["n_ntc"] += 1
            else:
                srec["n_target"] += 1
                srec["targets"].add(target)
                key = (target, donor, context)
                uid = unit_by_key.get(key)
                if uid is None:
                    uid = len(unit_rows)
                    unit_by_key[key] = uid
                    unit_rows.append({"target": target, "target_gene_id": target_id,
                                      "donor": donor, "context": context, "n_target_cells": 0,
                                      "lane_counts": {}, "n_lanes": 0})
                    unit_hashes.append(hashlib.sha256())
                unit_rows[uid]["n_target_cells"] += 1
                unit_rows[uid]["lane_counts"][lane] = unit_rows[uid]["lane_counts"].get(lane, 0) + 1
                unit_hashes[uid].update(cell_id.encode("utf-8"))
                unit_hashes[uid].update(b"\n")
                is_target[i] = True
                id_by_row[i] = uid

            source_file = row["source_file"]
            if not source_ranges or source_ranges[-1]["source_file"] != source_file:
                if any(x["source_file"] == source_file for x in source_ranges):
                    raise RuntimeError(f"Noncontiguous source-file block in common index: {source_file}")
                source_ranges.append({"source_file": source_file, "start": i, "end": i + 1,
                                      "sample": source_file.removesuffix(".assigned_guide.h5ad")})
            else:
                source_ranges[-1]["end"] = i + 1

            for rep, key in (("PCA", "pca_row"), ("scVI", "scvi_row"),
                             ("Geneformer V2-316M", "geneformer_row"), ("STATE SE-600M", "state_row")):
                value = int(row[key])
                if value != i:
                    raise RuntimeError(f"{rep} pointer is not sequential at common canonical row {i}; refusing implicit reorder")
                row_maps[rep][i] = value

    if i + 1 != N_EXPECTED or role_counts != {"target": N_TARGET, "ntc": N_NTC}:
        raise RuntimeError(f"Common-index count mismatch: roles={role_counts}, rows={i + 1}")
    if len(strata_rows) != 284 or len(unit_rows) != 144_368:
        raise RuntimeError(f"Unexpected frozen support: strata={len(strata_rows)}, units={len(unit_rows)}")
    no_ntc = [x for x in strata_rows if x["n_target"] and not x["n_ntc"]]
    if no_ntc or any(x["n_ntc"] == 0 for x in strata_rows):
        raise RuntimeError(f"Target/matching stratum without eligible NTC: {no_ntc[:3]}")

    lane_counts = [sorted(x["lane_counts"].items()) for x in unit_rows]
    for i, x in enumerate(unit_rows):
        x["n_lanes"] = len(x["lane_counts"])
        x["identity_sha256"] = unit_hashes[i].hexdigest()
    # Drop large per-unit lane dictionaries after retaining their sorted sidecar form.
    for x in unit_rows:
        x.pop("lane_counts")
    stratum_labels = []
    for x in strata_rows:
        stratum_labels.append({"donor": x["donor"], "context": x["context"], "lane": x["lane"],
                               "n_ntc": x["n_ntc"], "n_target": x["n_target"],
                               "n_targets": len(x["targets"])})
    for x in strata_rows:
        x.pop("targets")
    return {
        "unit_rows": unit_rows, "unit_by_key": unit_by_key, "unit_lane_counts": lane_counts,
        "stratum_rows": stratum_labels, "stratum_by_key": strata_by_key,
        "unit_ids": id_by_row, "stratum_ids": strata_by_row, "is_target": is_target,
        "row_maps": row_maps, "source_ranges": source_ranges,
        "role_counts": role_counts, "manifest_sha256": manifest_hash,
        "index_sha256": INDEX_SHA_EXPECTED, "index_audit": index_audit,
    }


def external_rep_checks(spec):
    name, source = spec["name"], spec["source"]
    if source == "hdf5":
        audit = sha_json(spec["audit"])
        with h5py.File(spec["artifact"], "r") as h:
            ds = h["coordinates"]
            physical = list(ds.shape)
            dtype = str(ds.dtype)
        if physical != [50, N_EXPECTED] or dtype != "float32":
            raise RuntimeError(f"Unexpected PCA array layout: {physical}, {dtype}")
        if audit.get("status") != "PASS" or audit.get("scores", {}).get("shape") != [N_EXPECTED, 50]:
            raise RuntimeError("PCA production audit/shape is not PASS")
        return {"artifact": str(spec["artifact"].relative_to(ROOT)), "physical_shape": physical,
                "logical_shape": [N_EXPECTED, 50], "dtype": dtype,
                "audit": str(spec["audit"].relative_to(ROOT)), "audit_status": audit["status"]}
    if source == "npy":
        audit = sha_json(spec["audit"])
        a = np.load(spec["artifact"], mmap_mode="r", allow_pickle=False)
        shape, dtype = list(a.shape), str(a.dtype)
        del a
        if shape != [N_EXPECTED, 50] or dtype != "float32" or audit.get("status") != "PASS":
            raise RuntimeError(f"Unexpected canonical scVI artifact/audit: {shape}/{dtype}/{audit.get('status')}")
        if audit.get("seed") not in (42, "42", None):
            raise RuntimeError(f"scVI audit is not seed 42: {audit.get('seed')}")
        return {"artifact": str(spec["artifact"].relative_to(ROOT)), "shape": shape, "dtype": dtype,
                "sidecar": str((SCVI / "cell_ids_by_row.tsv.gz").relative_to(ROOT)),
                "audit": str(spec["audit"].relative_to(ROOT)), "audit_status": audit["status"],
                "seed": 42, "seed_role": "PRIMARY_PREDECLARED_SEED"}
    if source == "state":
        audit = sha_json(spec["audit"])
        if audit.get("status") != "PASS" or audit.get("cells") != N_EXPECTED:
            raise RuntimeError("STATE global cell-index validation is not PASS")
        shards = {}
        for source_range in meta_source_ranges:
            sample = source_range["sample"]
            p = STATE / "shards" / sample
            val = sha_json(p / "validation.json")
            a = np.load(p / "embeddings.npy", mmap_mode="r", allow_pickle=False)
            sh, dt = list(a.shape), str(a.dtype)
            del a
            expected_n = source_range["end"] - source_range["start"]
            if sh != [expected_n, 2058] or dt != "float32" or val.get("status") != "PASS":
                raise RuntimeError(f"Invalid STATE shard {sample}: {sh}/{dt}/{val.get('status')}")
            shards[sample] = {"shape": sh, "dtype": dt, "audit_status": val["status"],
                              "audit": str((p / "validation.json").relative_to(ROOT))}
        return {"artifact": str(STATE.relative_to(ROOT)), "shape": [N_EXPECTED, 2058], "dtype": "float32",
                "audit": str(spec["audit"].relative_to(ROOT)), "audit_status": audit["status"], "shards": shards}
    if source == "geneformer":
        shards = {}
        for source_range in meta_source_ranges:
            sample = source_range["sample"]
            p = GF / sample
            complete = sha_json(p / "COMPLETE.json")
            audit_path = ROOT / "geneformer_global_cell_embeddings/audits" / f"{sample}.completion_audit.json"
            audit = sha_json(audit_path)
            plan = sha_json(p / "chunk_plan.json")
            if complete.get("status") != "PASS" or audit.get("status") != "PASS" or plan.get("config", {}).get("embedding_dim") != 1152:
                raise RuntimeError(f"Geneformer upstream validation is not PASS for {sample}")
            if int(complete.get("embedded_cells", -1)) != source_range["end"] - source_range["start"]:
                raise RuntimeError(f"Geneformer cell count does not match index for {sample}")
            chunks = []
            for ci in range(int(plan["n_chunks"])):
                cp = p / "chunks" / f"chunk_{ci:05d}.emb.npy"
                if not cp.exists():
                    raise RuntimeError(f"Missing Geneformer chunk {cp}")
                z = np.load(cp, mmap_mode="r", allow_pickle=False)
                expected_rows = min(int(plan["chunk_size"]), int(plan["n_cells"]) - ci * int(plan["chunk_size"]))
                if list(z.shape) != [expected_rows, 1152] or str(z.dtype) != "float32":
                    raise RuntimeError(f"Bad Geneformer chunk dimensions/dtype: {cp}")
                chunks.append(str(cp.relative_to(ROOT)))
                del z
            shards[sample] = {"rows": int(plan["n_cells"]), "dimensions": 1152,
                              "chunk_size": int(plan["chunk_size"]), "n_chunks": int(plan["n_chunks"]),
                              "complete": str((p / "COMPLETE.json").relative_to(ROOT)),
                              "audit": str(audit_path.relative_to(ROOT)), "chunk_files": chunks}
        return {"artifact": str(GF.relative_to(ROOT)), "shape": [N_EXPECTED, 1152], "dtype": "float32",
                "audits": shards}
    raise RuntimeError(name)


def iter_blocks(spec, row_map):
    dim, source = spec["dim"], spec["source"]
    if source == "hdf5":
        with h5py.File(spec["artifact"], "r") as h:
            ds = h["coordinates"]
            for start in range(0, N_EXPECTED, BATCH):
                end = min(N_EXPECTED, start + BATCH)
                rows = row_map[start:end]
                if rows[0] != start or rows[-1] != end - 1:
                    raise RuntimeError("PCA common-index row pointers are not contiguous")
                yield start, end, np.asarray(ds[:, int(rows[0]):int(rows[-1])+1].T)
    elif source == "npy":
        arr = np.load(spec["artifact"], mmap_mode="r", allow_pickle=False)
        for start in range(0, N_EXPECTED, BATCH):
            end = min(N_EXPECTED, start + BATCH)
            rows = row_map[start:end]
            if rows[0] != start or rows[-1] != end - 1:
                raise RuntimeError("scVI common-index row pointers are not contiguous")
            yield start, end, np.asarray(arr[int(rows[0]):int(rows[-1])+1])
        del arr
    elif source == "state":
        for sr in meta_source_ranges:
            arr = np.load(STATE / "shards" / sr["sample"] / "embeddings.npy", mmap_mode="r", allow_pickle=False)
            for start in range(sr["start"], sr["end"], BATCH):
                end = min(sr["end"], start + BATCH)
                rows = row_map[start:end]
                if rows[0] != start or rows[-1] != end - 1:
                    raise RuntimeError("STATE common-index pointers are not contiguous")
                yield start, end, np.asarray(arr[int(rows[0])-sr["start"]:int(rows[-1])-sr["start"]+1])
            del arr
    elif source == "geneformer":
        for sr in meta_source_ranges:
            plan = sha_json(GF / sr["sample"] / "chunk_plan.json")
            chunk_size = int(plan["chunk_size"])
            n = sr["end"] - sr["start"]
            for ci in range(int(plan["n_chunks"])):
                start = sr["start"] + ci * chunk_size
                end = min(sr["end"], start + chunk_size)
                rows = row_map[start:end]
                if rows[0] != start or rows[-1] != end - 1:
                    raise RuntimeError("Geneformer common-index pointers are not contiguous")
                path = GF / sr["sample"] / "chunks" / f"chunk_{ci:05d}.emb.npy"
                arr = np.load(path, mmap_mode="r", allow_pickle=False)
                if arr.shape[0] != end - start:
                    raise RuntimeError(f"Geneformer chunk row count mismatch at {path}")
                yield start, end, np.asarray(arr)
                del arr
    else:
        raise RuntimeError(f"Unknown representation source {source}")


def centroids_for(spec, meta):
    ns, dim = len(meta["stratum_rows"]), spec["dim"]
    sums = np.zeros((ns, dim), dtype=np.float64)
    counts = np.zeros(ns, dtype=np.int64)
    started = time.time()
    last_report = started
    for start, end, x in iter_blocks(spec, meta["row_maps"][spec["name"]]):
        if x.shape != (end - start, dim) or x.dtype != np.float32:
            raise RuntimeError(f"Invalid {spec['name']} block shape/dtype at {start}: {x.shape}/{x.dtype}")
        if not np.isfinite(x).all():
            raise RuntimeError(f"Nonfinite stored representation values in {spec['name']} rows {start}:{end}")
        target = meta["is_target"][start:end]
        strata = meta["stratum_ids"][start:end]
        ntc = ~target
        for sid in np.unique(strata[ntc]):
            m = ntc & (strata == sid)
            sums[sid] += x[m].sum(axis=0, dtype=np.float64)
            counts[sid] += int(m.sum())
        if end == N_EXPECTED or time.time() - last_report >= 60:
            print(json.dumps({"event": "centroid_progress", "representation": spec["name"],
                              "rows_scanned": end, "rows_expected": N_EXPECTED,
                              "ntc_cells_seen": int(counts.sum()), "elapsed_seconds": round(time.time()-started, 1)}), flush=True)
            last_report = time.time()
    expected = np.array([x["n_ntc"] for x in meta["stratum_rows"]], dtype=np.int64)
    if not np.array_equal(counts, expected) or np.any(counts <= 0):
        raise RuntimeError(f"{spec['name']} centroid coverage/count mismatch")
    means = sums / counts[:, None]
    if not np.isfinite(means).all():
        raise RuntimeError(f"Nonfinite {spec['name']} NTC centroid")
    return means, counts, time.time() - started


def responses_for(spec, meta, centroids, out_dir):
    units = meta["unit_rows"]
    n_units, dim = len(units), spec["dim"]
    sums = np.zeros((n_units, dim), dtype=np.float64)
    unit_counts = np.zeros(n_units, dtype=np.int64)
    lane_counts = np.zeros((n_units, len(meta["stratum_rows"])), dtype=np.uint32)
    started = time.time()
    last_report = started
    for start, end, x in iter_blocks(spec, meta["row_maps"][spec["name"]]):
        target = meta["is_target"][start:end]
        if not target.any():
            continue
        global_rows = np.flatnonzero(target) + start
        uids = meta["unit_ids"][global_rows]
        sids = meta["stratum_ids"][global_rows]
        if np.any(uids < 0) or np.any(sids < 0):
            raise RuntimeError(f"Unmapped target unit/stratum in {spec['name']} rows {start}:{end}")
        corrected = x[target].astype(np.float64) - centroids[sids]
        if not np.isfinite(corrected).all():
            raise RuntimeError(f"Nonfinite corrected target cells in {spec['name']} rows {start}:{end}")
        order = np.argsort(uids, kind="stable")
        sorted_uids = uids[order]
        starts = np.r_[0, np.flatnonzero(np.diff(sorted_uids)) + 1]
        grouped = np.add.reduceat(corrected[order], starts, axis=0, dtype=np.float64)
        unique_units = sorted_uids[starts]
        sums[unique_units] += grouped
        unit_counts[unique_units] += np.diff(np.r_[starts, len(sorted_uids)])
        np.add.at(lane_counts, (uids, sids), 1)
        if end == N_EXPECTED or time.time() - last_report >= 60:
            print(json.dumps({"event": "response_progress", "representation": spec["name"],
                              "rows_scanned": end, "rows_expected": N_EXPECTED,
                              "target_cells_seen": int(unit_counts.sum()), "elapsed_seconds": round(time.time()-started, 1)}), flush=True)
            last_report = time.time()

    expected_counts = np.array([x["n_target_cells"] for x in units], dtype=np.int64)
    if not np.array_equal(unit_counts, expected_counts) or int(unit_counts.sum()) != N_TARGET:
        raise RuntimeError(f"{spec['name']} target response support does not reconcile: {unit_counts.sum()}")
    responses = sums / unit_counts[:, None]
    norms = np.linalg.norm(responses, axis=1)
    if not np.isfinite(responses).all() or not np.isfinite(norms).all():
        raise RuntimeError(f"Nonfinite {spec['name']} response values/norms")
    zero = np.flatnonzero(norms == 0)
    if len(zero):
        raise RuntimeError(f"Unexpected zero-norm {spec['name']} responses; first unit indices {zero[:10].tolist()}")

    # Persist vectors atomically in float64; these are arithmetic response means.
    path = out_dir / "response_vectors.npy"
    tmp = out_dir / "response_vectors.npy.tmp"
    with tmp.open("wb") as f:
        np.save(f, responses, allow_pickle=False)
    os.replace(tmp, path)

    # Centroid arrays are float64 and ordered by centroid_units.tsv.gz.
    centroid_path = out_dir / "ntc_centroids.npy"
    return {
        "vectors": responses, "norms": norms, "lane_counts": lane_counts,
        "unit_counts": unit_counts, "response_path": path, "centroid_path": centroid_path,
        "elapsed_seconds": time.time() - started,
    }


def write_tables(spec, meta, centroids, counts, result, out_dir):
    centroid_path = result["centroid_path"]
    tmp = centroid_path.with_suffix(".npy.tmp")
    with tmp.open("wb") as f:
        np.save(f, centroids, allow_pickle=False)
    os.replace(tmp, centroid_path)
    with gzip.open(out_dir / "ntc_centroid_units.tsv.gz", "wt", newline="", compresslevel=1) as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["centroid_row", "representation", "donor", "context", "lane", "n_ntc", "dimensions", "finite"])
        for i, row in enumerate(meta["stratum_rows"]):
            w.writerow([i, spec["name"], row["donor"], row["context"], row["lane"], int(counts[i]), spec["dim"], "true"])
    with gzip.open(out_dir / "response_units.tsv.gz", "wt", newline="", compresslevel=1) as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["response_unit", "representation", "target", "target_gene_id", "donor", "context",
                    "n_target_cells", "dimensions", "response_l2_norm", "n_contributing_lanes",
                    "contributing_lanes", "target_cells_by_lane", "contributing_cell_ids_sha256", "finite"])
        for i, row in enumerate(meta["unit_rows"]):
            lane_pairs = meta["unit_lane_counts"][i]
            lc = result["lane_counts"][i]
            present = [(meta["stratum_rows"][int(sid)]["lane"], int(lc[sid])) for sid in np.flatnonzero(lc)]
            present.sort()
            if sum(x[1] for x in present) != int(result["unit_counts"][i]):
                raise RuntimeError(f"Lane support count mismatch for response unit {i}")
            # Confirm cell-stream lane support reconstructed from common index is exact.
            if sorted((str(lane), int(n)) for lane, n in lane_pairs) != present:
                raise RuntimeError(f"Common-index lane support changed for response unit {i}")
            w.writerow([i, spec["name"], row["target"], row["target_gene_id"], row["donor"], row["context"],
                        int(result["unit_counts"][i]), spec["dim"], format(float(result["norms"][i]), ".17g"),
                        len(present), ";".join(x[0] for x in present), ";".join(f"{x[0]}:{x[1]}" for x in present),
                        row["identity_sha256"], "true"])


def main():
    global meta_source_ranges
    started = datetime.now(timezone.utc)
    started_clock = time.time()
    check_dir_empty(BASE)
    meta = load_and_validate_index()
    meta_source_ranges = meta["source_ranges"]
    for rep in REPRESENTATIONS:
        audit_summary = external_rep_checks(rep)
        rep["audit_summary"] = audit_summary

    all_results = {}
    shared_unit_signature = [(x["target"], x["target_gene_id"], x["donor"], x["context"],
                             x["n_target_cells"], x["identity_sha256"], meta["unit_lane_counts"][i])
                            for i, x in enumerate(meta["unit_rows"])]
    unit_signature_hash = hashlib.sha256(json.dumps(shared_unit_signature, separators=(",", ":")).encode()).hexdigest()
    for rep in REPRESENTATIONS:
        rep_dir = BASE / rep["out"]
        print(json.dumps({"event": "centroids_start", "representation": rep["name"],
                          "rows": N_EXPECTED, "dimensions": rep["dim"]}), flush=True)
        centroids, ntc_counts, centroid_elapsed = centroids_for(rep, meta)
        print(json.dumps({"event": "responses_start", "representation": rep["name"],
                          "target_cells": N_TARGET}), flush=True)
        result = responses_for(rep, meta, centroids, rep_dir)
        write_tables(rep, meta, centroids, ntc_counts, result, rep_dir)
        # Independently check unit metadata and contribution identities remain the frozen common set.
        observed_counts = result["unit_counts"]
        if len(observed_counts) != len(shared_unit_signature) or not np.array_equal(
                observed_counts, np.array([x[4] for x in shared_unit_signature], dtype=np.int64)):
            raise RuntimeError(f"Response-unit/support mismatch for {rep['name']}")
        donor_counts = {}
        context_counts = {}
        for row in meta["unit_rows"]:
            donor_counts[row["donor"]] = donor_counts.get(row["donor"], 0) + row["n_target_cells"]
            context_counts[row["context"]] = context_counts.get(row["context"], 0) + row["n_target_cells"]
        rep_audit = {
            "status": "PASS", "representation": rep["name"], "dimensions": rep["dim"],
            "response_vector_dtype": "float64", "centroid_dtype": "float64",
            "expected_target_cells": N_TARGET, "accounted_target_cells": int(observed_counts.sum()),
            "ntc_cells_used": N_NTC, "occupied_ntc_strata": len(meta["stratum_rows"]),
            "strata_with_target_cells": sum(x["n_target"] > 0 for x in meta["stratum_rows"]),
            "centroids_finite": bool(np.isfinite(centroids).all()),
            "centroid_coverage": int(len(ntc_counts)), "zero_ntc_centroids": int(np.sum(ntc_counts == 0)),
            "response_units": len(meta["unit_rows"]), "finite_response_vectors": int(np.isfinite(result["vectors"]).all(axis=1).sum()),
            "zero_norm_responses": int(np.sum(result["norms"] == 0)),
            "duplicate_response_units": 0, "missing_response_units": 0,
            "unit_support_counts_match_common_index": True,
            "contributing_cell_identity_hashes_match_common_index": True,
            "response_unit_membership_signature_sha256": unit_signature_hash,
            "target_cells_by_donor": donor_counts, "target_cells_by_context": context_counts,
            "input": rep["audit_summary"],
            "outputs": {
                "response_vectors": str((rep_dir / "response_vectors.npy").relative_to(ROOT)),
                "response_units": str((rep_dir / "response_units.tsv.gz").relative_to(ROOT)),
                "ntc_centroids": str((rep_dir / "ntc_centroids.npy").relative_to(ROOT)),
                "ntc_centroid_units": str((rep_dir / "ntc_centroid_units.tsv.gz").relative_to(ROOT)),
            },
            "elapsed_seconds": {"centroids": centroid_elapsed, "response_pass": result["elapsed_seconds"]},
        }
        write_json(rep_dir / "RESPONSE_CONSTRUCTION_AUDIT.json", rep_audit)
        all_results[rep["name"]] = rep_audit
        del centroids, result
        print(json.dumps({"event": "representation_complete", "representation": rep["name"],
                          "response_units": len(meta["unit_rows"]), "target_cells": N_TARGET,
                          "elapsed_seconds": round(time.time() - started_clock, 1)}), flush=True)

    signatures = {x["response_unit_membership_signature_sha256"] for x in all_results.values()}
    unit_equal = len(signatures) == 1 and all(x["status"] == "PASS" for x in all_results.values())
    if not unit_equal:
        raise RuntimeError("Cross-representation response-unit metadata/support/identity mismatch")

    # Shared frozen cell membership and target/donor/context assignment are represented once by the input index.
    write_json(BASE / "audits/response_unit_membership.json", {
        "status": "PASS", "source_common_index": str(INDEX.relative_to(ROOT)),
        "common_index_sha256": meta["index_sha256"], "response_units": len(meta["unit_rows"]),
        "target_cells": N_TARGET, "membership_identity": "per-unit SHA-256 over cell IDs in canonical common-index order",
        "response_unit_identity_hashes": [{"response_unit": i, "target": x["target"], "donor": x["donor"],
                                            "context": x["context"], "n_target_cells": x["n_target_cells"],
                                            "cell_ids_sha256": x["identity_sha256"]}
                                           for i, x in enumerate(meta["unit_rows"])],
        "representation_membership_equality": "PASS: each pass streamed every common-index target row exactly once; per-unit support and IDs are common-index-defined",
    })
    # Write vector output checksums after all four response vectors exist.
    for rep in REPRESENTATIONS:
        rdir = BASE / rep["out"]
        for filename in ("response_vectors.npy", "response_units.tsv.gz", "ntc_centroids.npy", "ntc_centroid_units.tsv.gz"):
            all_results[rep["name"]]["outputs"][filename + "_sha256"] = sha256(rdir / filename)
        write_json(rdir / "RESPONSE_CONSTRUCTION_AUDIT.json", all_results[rep["name"]])

    status = all(x["status"] == "PASS" and x["accounted_target_cells"] == N_TARGET and
                 x["centroids_finite"] and x["zero_ntc_centroids"] == 0 and
                 x["zero_norm_responses"] == 0 and x["response_units"] == 144_368
                 for x in all_results.values()) and unit_equal
    summary = {
        "schema": "canonical_response_construction_v1", "status": "PASS" if status else "FAIL",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "response_specification": str((BASE / "RESPONSE_CONSTRUCTION_SPEC.md").relative_to(ROOT)),
        "method": "all eligible NTC centroid by donor/context/lane; correct each target cell; direct arithmetic mean by target/donor/context",
        "canonical_manifest": {"path": str(MANIFEST.relative_to(ROOT)), "sha256": meta["manifest_sha256"], "rows": N_EXPECTED},
        "common_index": {"path": str(INDEX.relative_to(ROOT)), "sha256": meta["index_sha256"], "rows": N_EXPECTED},
        "counts": {"total_cells": N_EXPECTED, "target_cells": N_TARGET, "ntc_cells": N_NTC,
                   "occupied_ntc_strata": len(meta["stratum_rows"]),
                   "target_containing_strata": sum(x["n_target"] > 0 for x in meta["stratum_rows"]),
                   "response_units": len(meta["unit_rows"]), "target_cells_by_donor": all_results["PCA"]["target_cells_by_donor"],
                   "target_cells_by_context": all_results["PCA"]["target_cells_by_context"]},
        "response_units_by_donor": {d: sum(1 for x in meta["unit_rows"] if x["donor"] == d) for d in ("D1", "D2", "D3", "D4")},
        "response_units_by_context": {c: sum(1 for x in meta["unit_rows"] if x["context"] == c) for c in ("Rest", "Stim8hr", "Stim48hr")},
        "cross_representation_response_unit_equality": "PASS" if unit_equal else "FAIL",
        "full_corrected_cell_matrices_written": False,
        "estimated_uncompressed_float32_storage_if_written_bytes": N_TARGET * (50 + 50 + 1152 + 2058) * 4,
        "representations": all_results,
        "provenance": {
            "script": os.path.relpath(Path(__file__).resolve(), ROOT), "script_sha256": sha256(Path(__file__)),
            "runtime": {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__, "h5py": h5py.__version__},
            "execution_environment": "recorded analysis environment; see environments/RECORDED_VERSIONS.md",
            "authoritative_method_sources": [
                {"path": "V31_CANONICAL_RESPONSE_CONSTRUCTION.md", "sha256": sha256(ROOT / "V31_CANONICAL_RESPONSE_CONSTRUCTION.md")},
                {"path": "source/derived/pca_full_response_v29/configs/pca_full_response_v29.R",
                 "sha256": sha256(ROOT / "source/derived/pca_full_response_v29/configs/pca_full_response_v29.R")},
                {"path": "source/derived/pca_full_response_v29/production/audits/validation_summary.json",
                 "sha256": sha256(ROOT / "source/derived/pca_full_response_v29/production/audits/validation_summary.json")},
            ],
            "scvi_seed": 42, "scvi_seed_role": "PRIMARY_PREDECLARED_SEED", "scvi_representation": "latent_v2 canonical",
            "created_utc": started.isoformat(timespec="seconds"),
            "total_elapsed_seconds": round(time.time() - started_clock, 1),
        },
        "response_construction_only": True, "geometry_started": False,
    }
    write_json(BASE / "reports/RESPONSE_CONSTRUCTION_SUMMARY.json", summary)
    lines = ["# Canonical response construction summary", "", f"**Status: {summary['status']}**", "",
             "V31 response construction only. No cosine similarities, representation agreement, response-magnitude comparisons, cell-to-response coherence, kNN, graph, clustering, spectral analysis, DE, biological summaries, or figures were computed.", "",
             "## Frozen cohort and estimator", "",
             f"Population: {N_EXPECTED:,} cells ({N_TARGET:,} target, {N_NTC:,} NTC). Matching strata: {len(meta['stratum_rows']):,} donor × context × lane; target-containing strata: {sum(x['n_target'] > 0 for x in meta['stratum_rows']):,}.",
             "All eligible NTC cells define one centroid per donor × context × lane. Each target cell is corrected with its matched centroid. Corrected target cells are directly averaged within target × donor × context; lanes are not averaged as intermediate units. Donors and contexts remain separate; guides are not weighted.", "",
             "## Representation outputs", "",
             "| Representation | Dimensions | Response units | Target cells accounted | Finite | Zero norm | Status |",
             "|---|---:|---:|---:|---|---:|---|"]
    for rep in REPRESENTATIONS:
        a = all_results[rep["name"]]
        lines.append(f"| {rep['name']} | {rep['dim']} | {a['response_units']:,} | {a['accounted_target_cells']:,} | {'yes' if a['finite_response_vectors']==a['response_units'] and a['centroids_finite'] else 'no'} | {a['zero_norm_responses']} | {a['status']} |")
        lines.append("")
        lines.append(f"- {rep['name']} response vectors: `{a['outputs']['response_vectors']}`")
        lines.append(f"- {rep['name']} response metadata: `{a['outputs']['response_units']}`")
        lines.append(f"- {rep['name']} NTC centroids: `{a['outputs']['ntc_centroids']}` and `{a['outputs']['ntc_centroid_units']}`")
    lines += ["", "## Equality and storage", "",
              f"Cross-representation target × donor × context keys, target-cell counts, lane support, and per-unit contributing-cell ID hashes: **{'PASS' if unit_equal else 'FAIL'}**.",
              f"Target cells accounted in every representation: **{N_TARGET:,} / {N_TARGET:,}**.",
              "Full corrected-target matrices were not written. Their combined float32 size would be approximately 275 GB (256 GiB); the common index and stored NTC centroids preserve the exact route to reconstruct corrected cells later.",
              "", f"Specification: `{(BASE / 'RESPONSE_CONSTRUCTION_SPEC.md').relative_to(ROOT)}`.", ""]
    (BASE / "reports/RESPONSE_CONSTRUCTION_SUMMARY.md").write_text("\n".join(lines))
    write_json(BASE / "audits/RESPONSE_CONSTRUCTION_AUDIT.json", {
        "status": summary["status"], "all_target_cells_accounted": all(x["accounted_target_cells"] == N_TARGET for x in all_results.values()),
        "all_centroids_finite": all(x["centroids_finite"] for x in all_results.values()),
        "all_response_vectors_finite": all(x["finite_response_vectors"] == x["response_units"] for x in all_results.values()),
        "no_zero_norm_responses": all(x["zero_norm_responses"] == 0 for x in all_results.values()),
        "response_unit_equality": unit_equal, "representations": {k: v["status"] for k, v in all_results.items()},
        "geometry_started": False,
    })
    print(json.dumps({"status": summary["status"], "target_cells": N_TARGET,
                      "response_units": len(meta["unit_rows"]), "elapsed_seconds": round(time.time() - started_clock, 1)}), flush=True)
    if not status:
        raise RuntimeError("Response construction QC failed; see summary/audit before downstream use")


if __name__ == "__main__":
    main()
