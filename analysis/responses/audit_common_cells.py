#!/usr/bin/env python3
"""Identity/provenance audit and canonical index map for V31 cell representations."""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import pyarrow.parquet as pq

ROOT = Path('.').resolve()
N_EXPECTED = 21_710_586
CANON = ROOT / "source/derived/v35_state_all_targets/STATE_ALL_TARGETS_CELL_MANIFEST.tsv.gz"
V29_SHARED = ROOT / "source/derived/shared_eligible_cell_manifest.h5"
V29_PCA_AUDIT = ROOT / "source/derived/pca_full_response_v29/production/audits/shared_eligible_cell_manifest_r_compatible.h5"
PCA_DIR = ROOT / "source/derived/pca_direct_full_materialized_v28/runs/job-6848910"
SCVI_DIR = ROOT / "global_scvi/final/primary_2k_allcell_v1/seed42/latent_v2"
GF_DIR = ROOT / "geneformer_global_cell_embeddings/outputs"
STATE_DIR = ROOT / "source/derived/v35_state_all_targets/STATE"
OUT = ROOT / "source/derived/downstream/common_cell_audit_v1"

REP = {
    "PCA": {"artifact": str((PCA_DIR / "scores/pca_scores_21m_x_50.h5").relative_to(ROOT)), "dim": 50},
    "scVI": {"artifact": str((SCVI_DIR / "latent_50d_float32.npy").relative_to(ROOT)), "dim": 50},
    "Geneformer V2-316M": {"artifact": str(GF_DIR.relative_to(ROOT)), "dim": 1152},
    "STATE SE-600M": {"artifact": str((STATE_DIR / "shards").relative_to(ROOT)), "dim": 2058},
}

META_FIELDS = ["source_file", "source_row_index", "donor", "context", "lane_id",
               "guide_id", "cohort_role", "curated_target_gene_id", "curated_target_gene_symbol"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def dump_json(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")
    os.replace(tmp, path)


def fail_example(rows: list, value, limit=10):
    if len(rows) < limit:
        rows.append(value)


class Canonical:
    def __init__(self):
        self.ids: dict[str, int] = {}
        self.id_by_row: list[str | None] = [None] * N_EXPECTED
        self.cats: dict[str, dict[str, int]] = {x: {} for x in META_FIELDS if x != "source_row_index"}
        self.codes: dict[str, np.ndarray] = {
            x: np.zeros(N_EXPECTED, dtype=np.uint32) for x in self.cats
        }
        self.source_rows = np.zeros(N_EXPECTED, dtype=np.int32)
        self.counts_by_role = Counter()
        self.counts_by_donor = Counter()
        self.counts_by_context = Counter()
        self.counts_by_role_donor = Counter()
        self.counts_by_role_context = Counter()
        self.strata = defaultdict(lambda: {"target_cells": 0, "ntc_cells": 0, "targets": set()})
        self.file_counts = Counter()
        self.invalid = Counter()
        self.invalid_examples = []
        self.duplicate_examples = []
        self.conflict_examples = []

    def catcode(self, field, value):
        book = self.cats[field]
        if value not in book:
            book[value] = len(book)
        return book[value]

    def ingest(self):
        n = 0
        with gzip.open(CANON, "rt", newline="") as f:
            r = csv.DictReader(f, delimiter="\t")
            required = {"cell_id", "source_file", "source_row_index", "donor", "context", "lane_id",
                        "guide_id", "cohort_role", "curated_target_gene_id", "curated_target_gene_symbol", "global_order"}
            if not required.issubset(r.fieldnames or []):
                raise RuntimeError(f"Frozen manifest is missing fields: {sorted(required-set(r.fieldnames or []))}")
            for row in r:
                if n >= N_EXPECTED:
                    raise RuntimeError("Frozen manifest contains more than the locked population")
                cell_id = row["cell_id"]
                if not cell_id:
                    self.invalid["empty_cell_id"] += 1
                if cell_id in self.ids:
                    self.invalid["duplicate_manifest_cell_id"] += 1
                    old = self.ids[cell_id]
                    fail_example(self.duplicate_examples, {"cell_id": cell_id, "first_row": old, "duplicate_row": n})
                    if old < n and any(self.codes[k][old] != self.catcode(k, row[k]) for k in META_FIELDS if k != "source_row_index"):
                        self.invalid["conflicting_metadata_for_duplicate_cell"] += 1
                        fail_example(self.conflict_examples, {"cell_id": cell_id, "first_row": old, "conflict_row": n})
                else:
                    self.ids[cell_id] = n
                self.id_by_row[n] = cell_id
                try:
                    global_order = int(row["global_order"])
                    source_row = int(row["source_row_index"])
                except Exception:
                    global_order, source_row = -1, -1
                if global_order != n:
                    self.invalid["noncontiguous_global_order"] += 1
                self.source_rows[n] = source_row
                for field in self.cats:
                    value = row[field]
                    self.codes[field][n] = self.catcode(field, value)
                    if not value or value.strip() in {"NA", "NaN", "None"}:
                        self.invalid[f"missing_{field}"] += 1
                donor, context, lane = row["donor"], row["context"], row["lane_id"]
                role = row["cohort_role"]
                target_id, target_symbol = row["curated_target_gene_id"], row["curated_target_gene_symbol"]
                if donor not in {"D1", "D2", "D3", "D4"}:
                    self.invalid["invalid_donor"] += 1
                    fail_example(self.invalid_examples, {"row": n, "cell_id": cell_id, "field": "donor", "value": donor})
                if context not in {"Rest", "Stim8hr", "Stim48hr"}:
                    self.invalid["invalid_context"] += 1
                    fail_example(self.invalid_examples, {"row": n, "cell_id": cell_id, "field": "context", "value": context})
                if not lane or lane.strip() in {"NA", "NaN", "None"}:
                    self.invalid["invalid_lane"] += 1
                    fail_example(self.invalid_examples, {"row": n, "cell_id": cell_id, "field": "lane_id", "value": lane})
                if role not in {"target", "ntc"}:
                    self.invalid["invalid_target_status"] += 1
                    fail_example(self.invalid_examples, {"row": n, "cell_id": cell_id, "field": "cohort_role", "value": role})
                if role == "ntc" and (target_id != "NTC" or target_symbol != "NTC"):
                    self.invalid["ntc_target_label_conflict"] += 1
                    fail_example(self.invalid_examples, {"row": n, "cell_id": cell_id, "role": role,
                                                          "target_id": target_id, "target_symbol": target_symbol})
                if role == "target" and (target_id == "NTC" or target_symbol == "NTC" or not target_id or not target_symbol):
                    self.invalid["target_ntc_label_conflict"] += 1
                    fail_example(self.invalid_examples, {"row": n, "cell_id": cell_id, "role": role,
                                                          "target_id": target_id, "target_symbol": target_symbol})
                self.file_counts[row["source_file"]] += 1
                self.counts_by_role[role] += 1
                self.counts_by_donor[donor] += 1
                self.counts_by_context[context] += 1
                self.counts_by_role_donor[(role, donor)] += 1
                self.counts_by_role_context[(role, context)] += 1
                s = self.strata[(donor, context, lane)]
                if role == "target":
                    s["target_cells"] += 1
                    s["targets"].add(target_symbol)
                elif role == "ntc":
                    s["ntc_cells"] += 1
                n += 1
        if n != N_EXPECTED:
            self.invalid["manifest_row_count_mismatch"] += abs(n - N_EXPECTED) or 1
        return n


def compare_v29_manifest(canon: Canonical):
    errors, examples = 0, []
    with h5py.File(V29_SHARED, "r") as h:
        ids = h["cells/cell_id"]
        if ids.shape != (N_EXPECTED,):
            return {"status": "FAIL", "v29_rows": int(ids.shape[0]), "mismatches": 1}
        for block_start in range(0, N_EXPECTED, 200_000):
            block_n = min(200_000, N_EXPECTED - block_start)
            block = ids[block_start:block_start + block_n]
            for local, value in enumerate(block):
                row = block_start + local
                observed = value.decode("utf-8").rstrip("\x00")
                expected = canon.id_by_row[row]
                if expected != observed:
                    errors += 1
                    fail_example(examples, {"row": row, "v35_cell_id": expected, "v29_cell_id": observed})
    return {"status": "PASS" if errors == 0 else "FAIL", "rows_compared": N_EXPECTED,
            "mismatches": errors, "examples": examples,
            "v29_manifest": str(V29_SHARED.relative_to(ROOT)),
            "v29_r_compatible_manifest": str(V29_PCA_AUDIT.relative_to(ROOT))}


def metadata_mismatch(c: Canonical, canonical_row: int, fields: dict, rep_name: str,
                      counts: Counter, examples: list, source_row_field="source_row_index"):
    for field in ("source_file", "donor", "context", "lane_id", "guide_id",
                  "cohort_role", "curated_target_gene_id", "curated_target_gene_symbol"):
        value = fields.get(field)
        if value is None:
            continue
        expected_code = int(c.codes[field][canonical_row])
        actual_code = c.cats[field].get(str(value), -1)
        if expected_code != actual_code:
            counts[f"{rep_name}:{field}"] += 1
            fail_example(examples, {"representation": rep_name, "cell_id": c.id_by_row[canonical_row],
                                    "field": field, "expected": list(c.cats[field])[expected_code] if expected_code >= 0 else None,
                                    "observed": value})
    actual_sr = fields.get(source_row_field)
    if actual_sr is not None:
        try:
            actual_sr = int(actual_sr)
        except Exception:
            actual_sr = -1
        if actual_sr != int(c.source_rows[canonical_row]):
            counts[f"{rep_name}:source_row_index"] += 1
            fail_example(examples, {"representation": rep_name,
                                    "cell_id": c.id_by_row[canonical_row],
                                    "field": "source_row_index", "expected": int(c.source_rows[canonical_row]),
                                    "observed": actual_sr})


def audit_one(c: Canonical, name: str, total_rows: int, records, meta_counts: Counter, meta_examples: list):
    seen = np.zeros(N_EXPECTED, dtype=np.bool_)
    row_seen = np.zeros(total_rows, dtype=np.bool_)
    row_map = np.full(N_EXPECTED, -1, dtype=np.int64)
    unexpected_ids = set()
    duplicate_values = set()
    dup_occurrences = dup_row_indices = out_of_range_rows = n_rows = 0
    for cell_id, rep_row, meta in records:
        n_rows += 1
        row = c.ids.get(cell_id)
        if row is None:
            if cell_id in unexpected_ids:
                dup_occurrences += 1
                duplicate_values.add(cell_id)
            else:
                unexpected_ids.add(cell_id)
            continue
        if seen[row]:
            dup_occurrences += 1
            duplicate_values.add(cell_id)
        else:
            seen[row] = True
            row_map[row] = int(rep_row)
        if meta:
            metadata_mismatch(c, row, meta, name, meta_counts, meta_examples)
        if rep_row < 0 or rep_row >= total_rows:
            out_of_range_rows += 1
        elif row_seen[rep_row]:
            dup_row_indices += 1
        else:
            row_seen[rep_row] = True
    missing_mask = ~seen
    expected_found = int(seen.sum())
    missing = int(missing_mask.sum())
    unexpected = len(unexpected_ids)
    def by_code(field, mask):
        code_values = c.codes[field][mask]
        ix, ct = np.unique(code_values, return_counts=True)
        return {list(c.cats[field])[int(i)]: int(n) for i, n in zip(ix, ct)}
    by_status = by_code("cohort_role", seen)
    by_donor = by_code("donor", seen)
    by_context = by_code("context", seen)
    miss_status = by_code("cohort_role", missing_mask)
    miss_donor = by_code("donor", missing_mask)
    miss_context = by_code("context", missing_mask)
    return row_map, {
        "total_representation_rows": total_rows,
        "rows_scanned": n_rows,
        "expected_downstream_cells": N_EXPECTED,
        "found_unique_expected_cells": expected_found,
        "missing_expected_cells": missing,
        "unexpected_unique_cells": unexpected,
        "duplicate_id_occurrences": dup_occurrences,
        "duplicate_id_values": len(duplicate_values),
        "duplicate_representation_row_indices": dup_row_indices,
        "out_of_range_row_indices": out_of_range_rows,
        "expected_found_by_target_status": by_status,
        "expected_missing_by_target_status": miss_status,
        "expected_found_by_donor": by_donor,
        "expected_missing_by_donor": miss_donor,
        "expected_found_by_context": by_context,
        "expected_missing_by_context": miss_context,
        "unexpected_examples": list(sorted(unexpected_ids))[:10],
        "pass": (n_rows == total_rows == N_EXPECTED and expected_found == N_EXPECTED and missing == 0
                 and unexpected == 0 and dup_occurrences == 0 and dup_row_indices == 0 and out_of_range_rows == 0
                 and bool(row_seen.all())),
    }


def representation_records(c: Canonical, file_offsets: dict, gf_totals: dict):
    # PCA HDF5 is stored as components x cells; the logical cell matrix is cells x 50.
    pca_ids = PCA_DIR / "audits/full_population_cell_ids.txt"
    def pca_iter():
        with pca_ids.open("rt") as f:
            for row, line in enumerate(f):
                yield line.rstrip("\r\n"), row, None
    pca_shape = None
    with h5py.File(PCA_DIR / "scores/pca_scores_21m_x_50.h5", "r") as h:
        pca_shape = tuple(h["coordinates"].shape)
    pca_summary = {"logical_shape": [N_EXPECTED, 50], "physical_hdf5_shape": list(pca_shape),
                   "dtype": "float32", "sidecar": str(pca_ids.relative_to(ROOT)),
                   "audit": str((PCA_DIR / "audits/full_pca_results.json").relative_to(ROOT)),
                   "total_cells": N_EXPECTED}

    scvi_ids = SCVI_DIR / "cell_ids_by_row.tsv.gz"
    def scvi_iter():
        with gzip.open(scvi_ids, "rt", newline="") as f:
            r = csv.DictReader(f, delimiter="\t")
            for row in r:
                yield row["cell_id"], int(row["global_row_index"]), None
    scvi_arr = np.load(SCVI_DIR / "latent_50d_float32.npy", mmap_mode="r", allow_pickle=False)
    scvi_summary = {"logical_shape": list(scvi_arr.shape), "dtype": str(scvi_arr.dtype),
                    "sidecar": str(scvi_ids.relative_to(ROOT)),
                    "audit": str((SCVI_DIR / "LATENT_EXTRACTION_AUDIT.json").relative_to(ROOT)),
                    "total_cells": int(scvi_arr.shape[0])}
    del scvi_arr

    def gf_iter():
        for sample, nrows in gf_totals.items():
            path = GF_DIR / sample / "cell_index.parquet"
            chunk_plan = json.loads((GF_DIR / sample / "chunk_plan.json").read_text())
            chunk_size = int(chunk_plan["chunk_size"])
            pf = pq.ParquetFile(path)
            for batch in pf.iter_batches(batch_size=250_000, columns=[
                "embedding_row", "manifest_row", "chunk", "cell_id", "source_file", "source_row_index",
                "donor", "context", "lane_id", "guide_id", "assignment_status",
                "curated_target_gene_id", "curated_target_gene_symbol", "is_ntc"]):
                d = batch.to_pydict()
                for i, cell_id in enumerate(d["cell_id"]):
                    role = "ntc" if d["is_ntc"][i] else "target"
                    meta = {"source_file": d["source_file"][i], "source_row_index": d["source_row_index"][i],
                            "donor": d["donor"][i], "context": d["context"][i], "lane_id": d["lane_id"][i],
                            "guide_id": d["guide_id"][i], "cohort_role": role,
                            "curated_target_gene_id": d["curated_target_gene_id"][i],
                            "curated_target_gene_symbol": d["curated_target_gene_symbol"][i]}
                    if (d["assignment_status"][i] not in {"NTC", "targeting"}
                            or (d["assignment_status"][i] == "NTC") != bool(d["is_ntc"][i])):
                        meta["cohort_role"] = "METADATA_CONFLICT"
                    # embedding_row is local to its chunk; manifest_row is the
                    # row in the complete per-sample embedding matrix.
                    # Check the producer's chunk mapping while using the
                    # full-matrix row for the durable representation index.
                    manifest_row = int(d["manifest_row"][i])
                    chunk = int(d["chunk"][i])
                    embedding_row = int(d["embedding_row"][i])
                    if manifest_row != chunk * chunk_size + embedding_row:
                        meta["cohort_role"] = "GENEFORMER_ROW_ORDER_CONFLICT"
                    global_row = file_offsets[sample] + manifest_row
                    yield cell_id, global_row, meta
    gf_totals_row = sum(gf_totals.values())
    gf_summary = {"logical_shape": [gf_totals_row, 1152], "dtype": "float32",
                  "cell_index": f"{GF_DIR.relative_to(ROOT)}/<sample>/cell_index.parquet",
                  "audits": str((ROOT / "geneformer_global_cell_embeddings/audits").relative_to(ROOT)),
                  "total_cells": gf_totals_row}

    state_index = STATE_DIR / "global_cell_index.tsv.gz"
    def state_iter():
        with gzip.open(state_index, "rt", newline="") as f:
            r = csv.DictReader(f, delimiter="\t")
            for d in r:
                source = d["source_file"]
                sample = source.removesuffix(".assigned_guide.h5ad")
                local_row = int(d["embedding_row"])
                state_row = file_offsets[sample] + local_row
                meta = {k: d[k] for k in META_FIELDS if k != "source_row_index"}
                meta["source_row_index"] = int(d["source_row_index"])
                if int(d["global_order"]) != state_row:
                    meta["cohort_role"] = "STATE_ROW_ORDER_CONFLICT"
                yield d["cell_id"], state_row, meta
    shard_validation = {}
    for sample, nrows in file_offsets.items():
        shard_dir = STATE_DIR / "shards" / sample
        val = json.loads((shard_dir / "validation.json").read_text())
        mat = np.load(shard_dir / "embeddings.npy", mmap_mode="r", allow_pickle=False)
        shard_validation[sample] = {"shape": list(mat.shape), "dtype": str(mat.dtype),
                                    "validation_status": val.get("status"),
                                    "cell_id_order_matches_manifest": val.get("cell_id_order_matches_manifest"),
                                    "cell_ids_unique": val.get("cell_ids_unique"),
                                    "finite": val.get("all_values_finite")}
        del mat
    state_total = sum(x["shape"][0] for x in shard_validation.values())
    state_summary = {"logical_shape": [state_total, 2058], "dtype": "float32",
                     "cell_index": str(state_index.relative_to(ROOT)),
                     "audit": str((STATE_DIR / "global_cell_index.validation.json").relative_to(ROOT)),
                     "shards": shard_validation, "total_cells": state_total}
    return [("PCA", N_EXPECTED, pca_iter, pca_summary),
            ("scVI", N_EXPECTED, scvi_iter, scvi_summary),
            ("Geneformer V2-316M", gf_totals_row, gf_iter, gf_summary),
            ("STATE SE-600M", state_total, state_iter, state_summary)]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    c = Canonical()
    n_manifest = c.ingest()
    v29_crosscheck = compare_v29_manifest(c)
    file_offsets = {}
    offset = 0
    for source, count in c.file_counts.items():
        sample = source.removesuffix(".assigned_guide.h5ad")
        file_offsets[sample] = offset
        offset += count
    if offset != N_EXPECTED:
        c.invalid["source_file_partition_count_mismatch"] += 1

    # Cross-check the canonical V35 freeze against the exact PCA input order.
    pca_order_audit = {"status": "PASS", "pca_order_sha256": "0777439e5e03a07d39204af3ef4e707e6db359a71638472906c703eb7e16d56d",
                       "frozen_manifest": str(CANON.relative_to(ROOT)), "rows_compared_to_v29_shared_manifest": v29_crosscheck["rows_compared"]}
    with (PCA_DIR / "audits/full_population_cell_ids.txt").open("rt") as pf:
        mismatch, pca_rows = 0, 0
        for pca_id in pf:
            pca_id = pca_id.rstrip("\r\n")
            if pca_rows >= N_EXPECTED or pca_id != c.id_by_row[pca_rows]:
                mismatch += 1
            pca_rows += 1
    pca_order_audit.update({"pca_rows": pca_rows, "rowwise_v35_pca_v29_id_mismatches": mismatch,
                            "status": "PASS" if pca_rows == N_EXPECTED and mismatch == 0 else "FAIL"})

    # Verify physical representation shapes, dtypes, and manifests before ID mapping.
    results = {}
    metadata_counts = Counter()
    metadata_examples = []
    rep_maps = {}
    summaries = {}
    gf_totals = {}
    for sample in file_offsets:
        complete_path = GF_DIR / sample / "COMPLETE.json"
        complete = json.loads(complete_path.read_text())
        if complete.get("status") != "PASS":
            raise RuntimeError(f"Geneformer {sample} COMPLETE marker not PASS")
        n = int(complete.get("embedded_cells", complete.get("expected_cells", -1)))
        gf_totals[sample] = n
    reps = representation_records(c, file_offsets, gf_totals)
    # Confirm primary shape/dtype evidence and canonical cohort totals from reports.
    pca_audit = json.loads((PCA_DIR / "audits/full_pca_results.json").read_text())
    scvi_audit = json.loads((SCVI_DIR / "LATENT_EXTRACTION_AUDIT.json").read_text())
    state_audit = json.loads((STATE_DIR / "global_cell_index.validation.json").read_text())
    gf_audits = [json.loads((ROOT / "geneformer_global_cell_embeddings/audits" / f"{s}.completion_audit.json").read_text()) for s in gf_totals]
    external_checks = {
        "PCA": pca_audit.get("status") == "PASS" and pca_audit.get("scores", {}).get("shape") == [N_EXPECTED, 50]
              and pca_audit.get("scores", {}).get("dtype") == "float32",
        "scVI": scvi_audit.get("status") == "PASS" and scvi_audit.get("rows") == N_EXPECTED
               and scvi_audit.get("dimensions") == 50 and scvi_audit.get("dtype") == "float32",
        "STATE SE-600M": state_audit.get("status") == "PASS" and state_audit.get("cells") == N_EXPECTED,
        "Geneformer V2-316M": len(gf_audits) == 12 and all(a.get("status") == "PASS" for a in gf_audits),
    }
    if n_manifest != N_EXPECTED:
        c.invalid["manifest_count_mismatch"] += 1

    for name, total_rows, iterator, summary in reps:
        row_map, audit = audit_one(c, name, total_rows, iterator(), metadata_counts, metadata_examples)
        audit["upstream_artifact_audit_pass"] = bool(external_checks[name])
        audit["artifact"] = summary
        audit["pass"] = audit["pass"] and bool(external_checks[name])
        results[name] = audit
        rep_maps[name] = row_map
        summaries[name] = summary

    target_counts, ntc_counts, no_ntc_strata = {}, {}, []
    for key, value in sorted(c.strata.items()):
        donor, context, lane = key
        row = {"donor": donor, "context": context, "lane": lane,
               "target_cells": value["target_cells"], "ntc_cells": value["ntc_cells"],
               "perturbation_targets": len(value["targets"]),
               "matched_ntc_available": value["ntc_cells"] > 0}
        target_counts[(donor, context, lane)] = row["target_cells"]
        ntc_counts[(donor, context, lane)] = row["ntc_cells"]
        if row["target_cells"] > 0 and row["ntc_cells"] == 0:
            no_ntc_strata.append(row)
    matching = {
        "strata_count": len(c.strata),
        "strata_with_targets": sum(v["target_cells"] > 0 for v in c.strata.values()),
        "strata_with_ntc": sum(v["ntc_cells"] > 0 for v in c.strata.values()),
        "target_cells_without_matched_ntc": sum(x["target_cells"] for x in no_ntc_strata),
        "strata_missing_required_ntc": no_ntc_strata,
        "pass": not no_ntc_strata and len(c.strata) > 0,
    }
    strata_rows = [{"donor": d, "context": ctx, "lane": lane,
                    "target_cells": values["target_cells"], "ntc_cells": values["ntc_cells"],
                    "perturbation_targets": len(values["targets"]),
                    "matched_ntc_available": values["ntc_cells"] > 0}
                   for (d, ctx, lane), values in sorted(c.strata.items())]

    metadata_report = {
        "required_fields": META_FIELDS,
        "manifest_metadata_errors": dict(c.invalid),
        "cross_representation_metadata_mismatch_counts": dict(metadata_counts),
        "examples": {"manifest": c.invalid_examples, "duplicate_cells": c.duplicate_examples,
                     "conflicting_metadata": c.conflict_examples, "representation_mismatches": metadata_examples},
        "pass": not c.invalid and not metadata_counts,
    }
    identity_pass = all(x["pass"] for x in results.values()) and v29_crosscheck["status"] == "PASS" and pca_order_audit["status"] == "PASS"
    cross_sets = {
        "PCA_equals_frozen_cohort": results["PCA"]["pass"],
        "scVI_equals_frozen_cohort": results["scVI"]["pass"],
        "Geneformer_equals_frozen_cohort": results["Geneformer V2-316M"]["pass"],
        "STATE_equals_frozen_cohort": results["STATE SE-600M"]["pass"],
        "all_four_representation_sets_equal_each_other_and_frozen_cohort": identity_pass,
        "pass": identity_pass,
    }
    overall = identity_pass and metadata_report["pass"] and matching["pass"]

    role_counts = dict(c.counts_by_role)
    if role_counts != {"target": 20_771_051, "ntc": 939_535}:
        c.invalid["frozen_population_target_ntc_totals_mismatch"] += 1
        metadata_report["manifest_metadata_errors"] = dict(c.invalid)
        metadata_report["pass"] = False
        overall = False

    # Write the canonical index only after every identity, metadata, and support check passes.
    if overall:
        map_path = OUT / "COMMON_DOWNSTREAM_CELL_INDEX.tsv.gz"
        tmp_path = map_path.with_suffix(map_path.suffix + ".tmp")
        with gzip.open(CANON, "rt", newline="") as src, gzip.open(tmp_path, "wt", newline="", compresslevel=1) as dst:
            r = csv.DictReader(src, delimiter="\t")
            fields = ["canonical_row", "cell_id", "pca_row", "scvi_row", "geneformer_row", "state_row",
                      "donor", "context", "lane", "target_status", "target_gene_id", "target_name",
                      "guide_id", "source_file", "source_row_index"]
            w = csv.DictWriter(dst, fieldnames=fields, delimiter="\t", lineterminator="\n")
            w.writeheader()
            for row in r:
                i = int(row["global_order"])
                w.writerow({"canonical_row": i, "cell_id": row["cell_id"],
                            "pca_row": int(rep_maps["PCA"][i]), "scvi_row": int(rep_maps["scVI"][i]),
                            "geneformer_row": int(rep_maps["Geneformer V2-316M"][i]),
                            "state_row": int(rep_maps["STATE SE-600M"][i]),
                            "donor": row["donor"], "context": row["context"], "lane": row["lane_id"],
                            "target_status": row["cohort_role"], "target_gene_id": row["curated_target_gene_id"],
                            "target_name": row["curated_target_gene_symbol"], "guide_id": row["guide_id"],
                            "source_file": row["source_file"], "source_row_index": row["source_row_index"]})
        os.replace(tmp_path, map_path)
        index_info = {"path": str(map_path.relative_to(ROOT)), "rows": N_EXPECTED,
                      "sha256": sha256(map_path), "canonical_order": "global_order in V35 frozen all-target manifest"}
    else:
        index_info = {"path": None, "written": False, "reason": "One or more pass criteria failed; no intersection/index was produced."}

    counts_report = {
        "target_vs_ntc": role_counts,
        "donor": dict(c.counts_by_donor),
        "context": dict(c.counts_by_context),
        "target_vs_ntc_by_donor": {f"{a}:{b}": n for (a, b), n in c.counts_by_role_donor.items()},
        "target_vs_ntc_by_context": {f"{a}:{b}": n for (a, b), n in c.counts_by_role_context.items()},
        "donor_context_lane": strata_rows,
    }
    report = {
        "schema": "common_downstream_cell_audit_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": "PASS" if overall else "FAIL",
        "methodology": "Identity/provenance audit only; no corrected cells, centroids, response vectors, or geometry computed.",
        "authoritative_downstream_manifest": {
            "path": str(CANON.relative_to(ROOT)), "sha256": sha256(CANON), "rows": n_manifest,
            "order_field": "global_order", "order_contiguous": not c.invalid.get("noncontiguous_global_order", 0),
            "target_cells": role_counts.get("target", 0), "ntc_cells": role_counts.get("ntc", 0),
            "distinct_target_symbols": len(c.cats["curated_target_gene_symbol"]) - (1 if "NTC" in c.cats["curated_target_gene_symbol"] else 0),
            "metadata": META_FIELDS,
            "cross_checked_against_v29_shared_eligible_manifest": v29_crosscheck,
            "historical_subset_used": False,
        },
        "representations": results,
        "representation_artifact_summaries": summaries,
        "cross_representation_set_equality": cross_sets,
        "metadata_consistency": metadata_report,
        "counts_and_strata": counts_report,
        "matching_strata": matching,
        "canonical_index": index_info,
        "pass_criteria": {
            "all_frozen_ids_exactly_once_in_each_representation": identity_pass,
            "all_required_metadata_consistent": metadata_report["pass"],
            "all_target_strata_have_matched_ntc": matching["pass"],
            "index_written_only_if_all_pass": True,
        },
    }
    dump_json(OUT / "COMMON_DOWNSTREAM_CELL_AUDIT.json", report)
    with (OUT / "MATCHING_STRATA.tsv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["donor", "context", "lane", "target_cells", "ntc_cells",
                                         "perturbation_targets", "matched_ntc_available"], delimiter="\t", lineterminator="\n")
        w.writeheader(); w.writerows(strata_rows)
    with (OUT / "DOWNSTREAM_COUNTS.tsv").open("w", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["axis", "group", "expected_cells"])
        for label, count in sorted(role_counts.items()): w.writerow(["target_status", label, count])
        for label, count in sorted(c.counts_by_donor.items()): w.writerow(["donor", label, count])
        for label, count in sorted(c.counts_by_context.items()): w.writerow(["context", label, count])
        for (label, group), count in sorted(c.counts_by_role_donor.items()): w.writerow([f"target_status_by_donor:{label}", group, count])
        for (label, group), count in sorted(c.counts_by_role_context.items()): w.writerow([f"target_status_by_context:{label}", group, count])

    lines = ["# Common downstream cell identity audit", "",
             f"**Status: {'PASS' if overall else 'FAIL'}**", "",
             "This audit validates frozen identities, metadata, representation row mappings, and matching strata only. It does not construct corrected cells, NTC centroids, response vectors, or geometry.", "",
             "## Frozen cohort", "",
             f"Authoritative cohort: `{CANON.relative_to(ROOT)}`; {n_manifest:,} rows, target {role_counts.get('target',0):,}, NTC {role_counts.get('ntc',0):,}; order is the frozen `global_order`.",
             f"V29 shared eligible manifest row comparison: **{v29_crosscheck['status']}**; V35/PCA/V29 row-order comparison: **{pca_order_audit['status']}**.", "",
             "The V35 manifest is the newest frozen all-target manifest and carries donor, context, lane, guide, target assignment, and target ID/symbol. It was cross-checked against the V29 shared eligible-cell manifest. No historical 15-target or pilot cohort was used.", "",
             "## Representation identity audit", "",
             "| Representation | Total represented | Expected | Found | Missing | Unexpected | Duplicate IDs | Dimensions | Status |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    for name, audit in results.items():
        lines.append(f"| {name} | {audit['total_representation_rows']:,} | {N_EXPECTED:,} | {audit['found_unique_expected_cells']:,} | {audit['missing_expected_cells']:,} | {audit['unexpected_unique_cells']:,} | {audit['duplicate_id_occurrences']:,} | {REP[name]['dim']} | {'PASS' if audit['pass'] else 'FAIL'} |")
    lines += ["", "## Downstream counts", "", "### Target and NTC", "", "| Status | Cells |", "|---|---:|"]
    for key, count in sorted(role_counts.items()): lines.append(f"| {key} | {count:,} |")
    lines += ["", "### Donor", "", "| Donor | Cells |", "|---|---:|"]
    for key, count in sorted(c.counts_by_donor.items()): lines.append(f"| {key} | {count:,} |")
    lines += ["", "### Context", "", "| Context | Cells |", "|---|---:|"]
    for key, count in sorted(c.counts_by_context.items()): lines.append(f"| {key} | {count:,} |")
    lines += ["", "### Donor × context × lane matching strata", "",
              f"Occupied strata: {matching['strata_count']}; strata containing targets: {matching['strata_with_targets']}; strata with eligible NTCs: {matching['strata_with_ntc']}; target cells without matched NTC: {matching['target_cells_without_matched_ntc']:,}.",
              f"Matching-stratum status: **{'PASS' if matching['pass'] else 'FAIL'}**. Full per-stratum counts are in `MATCHING_STRATA.tsv`.", "",
              "## Metadata and common ordering", "",
              f"Metadata consistency: **{'PASS' if metadata_report['pass'] else 'FAIL'}**; manifest/metadata error counts: `{json.dumps(dict(c.invalid), sort_keys=True)}`; representation metadata mismatches: `{json.dumps(dict(metadata_counts), sort_keys=True)}`.",
              f"Cross-representation set equality: **{'PASS' if identity_pass else 'FAIL'}**.",
              f"Canonical order is frozen V35 `global_order`. Common row map: `{index_info.get('path') if index_info.get('path') else 'not written because audit failed'}`.", "",
              "No response construction or geometry was started.", ""]
    (OUT / "COMMON_DOWNSTREAM_CELL_AUDIT.md").write_text("\n".join(lines))
    print(json.dumps({"status": report["status"], "rows": n_manifest,
                      "representations": {k: {"pass": v["pass"], "found": v["found_unique_expected_cells"],
                                               "missing": v["missing_expected_cells"], "unexpected": v["unexpected_unique_cells"],
                                               "duplicates": v["duplicate_id_occurrences"]} for k, v in results.items()},
                      "metadata_pass": metadata_report["pass"], "matching_pass": matching["pass"],
                      "index": index_info}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
