"""Generate target-conditioned DSSM candidates with exact FAISS FlatIP.

Purpose
-------
Turn a trained DSSM checkpoint into a measurable recall source without loading
all query sessions or candidate rows into memory.

Workflow
--------
1. Verify that checkpoint, item vocabulary and point-in-time snapshot match.
2. L2-normalize real item embeddings (item_id >= 2) and build IndexFlatIP.
3. Stream query histories, apply last-N truncation and map unseen aids to UNK.
4. Encode one session vector for each click/cart/order target and search Top-K.
5. Write compact fixed-size candidate lists and evaluate Recall@K from labels.

Inputs are a prepared DSSM directory, its checkpoint, query-prefix Parquet and
future-label Parquet. Output is ``candidates/part-*.parquet`` plus ``report.json``.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import shutil
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Sequence

import duckdb
import faiss
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.dssm_attention import TargetAwareAttentionDSSM
from models.dssm_baseline import FixedPositionDSSM
from utils.config import resolve_project_config


TARGET_TYPES = (1, 2, 3)
DSSMModel = FixedPositionDSSM | TargetAwareAttentionDSSM


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _quote(path: str | Path) -> str:
    return str(path).replace("'", "''")


def _candidate_schema(topk: int) -> pa.Schema:
    return pa.schema(
        [
            pa.field("session", pa.int64(), nullable=False),
            pa.field("target_type", pa.int8(), nullable=False),
            pa.field("aids", pa.list_(pa.int64(), topk), nullable=False),
            pa.field("scores", pa.list_(pa.float32(), topk), nullable=False),
        ]
    )


class _RollingWriter:
    def __init__(
        self,
        output_dir: Path,
        schema: pa.Schema,
        *,
        compression: str,
        rows_per_file: int,
    ) -> None:
        self.output_dir = output_dir
        self.schema = schema
        self.compression = compression
        self.rows_per_file = rows_per_file
        self.file_index = 0
        self.rows_in_file = 0
        self.total_rows = 0
        self.writer: pq.ParquetWriter | None = None
        output_dir.mkdir(parents=True, exist_ok=True)

    def _open(self) -> None:
        path = self.output_dir / f"part-{self.file_index:05d}.parquet"
        self.writer = pq.ParquetWriter(path, self.schema, compression=self.compression)
        self.file_index += 1
        self.rows_in_file = 0

    def write(self, table: pa.Table) -> None:
        if table.num_rows == 0:
            return
        if self.writer is None or self.rows_in_file + table.num_rows > self.rows_per_file:
            self.close_current()
            self._open()
        assert self.writer is not None
        self.writer.write_table(table)
        self.rows_in_file += table.num_rows
        self.total_rows += table.num_rows

    def close_current(self) -> None:
        if self.writer is not None:
            self.writer.close()
            self.writer = None

    def close(self) -> None:
        self.close_current()


def _load_vocab(data_dir: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    report_path = data_dir / "build_report.json"
    if not report_path.is_file():
        raise FileNotFoundError(report_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    vocab_path = data_dir / report["vocabulary"]["path"]
    if _sha256(vocab_path) != report["vocabulary"]["sha256"]:
        raise AssertionError("Vocabulary hash differs from DSSM build report")
    table = pq.read_table(vocab_path, columns=["aid", "item_id"]).sort_by("item_id")
    aids = table["aid"].combine_chunks().to_numpy(zero_copy_only=False).astype(np.int64)
    item_ids = table["item_id"].combine_chunks().to_numpy(zero_copy_only=False).astype(np.int64)
    expected = np.arange(2, len(item_ids) + 2, dtype=np.int64)
    if not np.array_equal(item_ids, expected):
        raise AssertionError("DSSM item IDs must be contiguous from 2")
    if len(aids) == 0 or np.any(aids[1:] <= aids[:-1]):
        raise AssertionError("DSSM vocabulary aids must be strictly increasing")
    return aids, item_ids, report


def _load_model(
    checkpoint_path: Path,
    *,
    vocab_hash: str,
    snapshot: str,
    device: torch.device,
) -> tuple[DSSMModel, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint["vocab_sha256"] != vocab_hash:
        raise AssertionError("Checkpoint and item vocabulary hashes differ")
    if checkpoint["snapshot"] != snapshot:
        raise AssertionError("Checkpoint was trained from a different point-in-time snapshot")
    architecture = checkpoint["architecture"]
    architecture_name = architecture.get("name")
    common_arguments = {
        "num_items": int(architecture["num_items"]),
        "embedding_dim": int(architecture["embedding_dim"]),
        "num_types": int(architecture["num_types"]),
        "temperature": float(architecture["temperature"]),
        "sparse_item_gradients": bool(
            architecture.get("sparse_item_gradients", True)
        ),
    }
    if architecture_name == "fixed_position_dssm":
        model: DSSMModel = FixedPositionDSSM(**common_arguments)
    elif architecture_name == "target_aware_attention_dssm":
        model = TargetAwareAttentionDSSM(
            **common_arguments,
            max_sequence_length=int(architecture["max_sequence_length"]),
        )
    else:
        raise ValueError(f"Unsupported DSSM architecture: {architecture_name!r}")
    model.load_state_dict(checkpoint["model_state_dict"])
    metadata = {
        "checkpoint_format_version": checkpoint.get("format_version", 1),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "architecture": architecture,
        "snapshot_max_ts": int(checkpoint["snapshot_max_ts"]),
        "config_hash": checkpoint["config_hash"],
    }
    del checkpoint
    gc.collect()
    return model.to(device).eval(), metadata


def _normalized_item_vectors(model: DSSMModel) -> np.ndarray:
    weight = model.item_embedding.weight.detach()[2:].float().cpu().numpy().copy()
    norms = np.linalg.norm(weight, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise AssertionError("A real item has a zero DSSM embedding")
    weight /= norms
    return np.ascontiguousarray(weight, dtype=np.float32)


def _build_flat_index(
    vectors: np.ndarray,
    *,
    use_gpu: bool,
) -> tuple[Any, Any | None, float]:
    started = time.perf_counter()
    cpu_index = faiss.IndexFlatIP(vectors.shape[1])
    cpu_index.add(vectors)
    resources = None
    index: Any = cpu_index
    if use_gpu:
        if not hasattr(faiss, "get_num_gpus") or faiss.get_num_gpus() < 1:
            raise RuntimeError("FAISS GPU was requested but no FAISS GPU device is available")
        resources = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(resources, 0, cpu_index)
    return index, resources, time.perf_counter() - started


def _encode_raw_aids(
    raw_aids: Sequence[int],
    vocab_aids: np.ndarray,
    vocab_item_ids: np.ndarray,
) -> tuple[np.ndarray, int]:
    raw = np.asarray(raw_aids, dtype=np.int64)
    positions = np.searchsorted(vocab_aids, raw)
    inside = positions < len(vocab_aids)
    matched = np.zeros(len(raw), dtype=bool)
    matched[inside] = vocab_aids[positions[inside]] == raw[inside]
    encoded = np.ones(len(raw), dtype=np.int64)
    encoded[matched] = vocab_item_ids[positions[matched]]
    return encoded, int((~matched).sum())


def _collate_queries(
    rows: dict[str, list[Any]],
    *,
    vocab_aids: np.ndarray,
    vocab_item_ids: np.ndarray,
    max_history: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int, int]:
    sessions = np.asarray(rows["session"], dtype=np.int64)
    histories = rows["aids"]
    event_types = rows["event_types"]
    timestamps = rows["timestamps"]
    width = max(1, max((min(len(values), max_history) for values in histories), default=0))
    items = np.zeros((len(sessions), width), dtype=np.int64)
    types = np.zeros((len(sessions), width), dtype=np.int64)
    unknown = 0
    history_events = 0
    min_ts = 2**63 - 1
    for row_index, (raw_items, raw_types, raw_ts) in enumerate(
        zip(histories, event_types, timestamps, strict=True)
    ):
        if len(raw_items) != len(raw_types) or len(raw_items) != len(raw_ts):
            raise ValueError(f"Misaligned query history for session {sessions[row_index]}")
        raw_items = raw_items[-max_history:]
        raw_types = raw_types[-max_history:]
        raw_ts = raw_ts[-max_history:]
        encoded, row_unknown = _encode_raw_aids(raw_items, vocab_aids, vocab_item_ids)
        if any(kind not in TARGET_TYPES for kind in raw_types):
            raise ValueError(f"Invalid event type in session {sessions[row_index]}")
        start = width - len(encoded)
        items[row_index, start:] = encoded
        types[row_index, start:] = np.asarray(raw_types, dtype=np.int64)
        unknown += row_unknown
        history_events += len(encoded)
        if raw_ts:
            min_ts = min(min_ts, int(min(raw_ts)))
    return sessions, items, types, unknown, history_events, min_ts


def _result_table(
    sessions: np.ndarray,
    target_type: int,
    candidate_aids: np.ndarray,
    scores: np.ndarray,
    schema: pa.Schema,
) -> pa.Table:
    topk = candidate_aids.shape[1]
    aid_lists = pa.FixedSizeListArray.from_arrays(
        pa.array(candidate_aids.reshape(-1), type=pa.int64()), topk
    )
    score_lists = pa.FixedSizeListArray.from_arrays(
        pa.array(scores.reshape(-1), type=pa.float32()), topk
    )
    return pa.Table.from_arrays(
        [
            pa.array(sessions, type=pa.int64()),
            pa.array(np.full(len(sessions), target_type, dtype=np.int8), type=pa.int8()),
            aid_lists,
            score_lists,
        ],
        schema=schema,
    )


def _evaluate(
    candidate_glob: Path,
    labels_path: Path,
    eval_ks: Sequence[int],
    target_weights: dict[int, float],
    *,
    workers: int,
    memory_limit_gb: int,
    temp_dir: Path,
) -> dict[str, Any]:
    """Run streaming DSSM recall and return the persisted evaluation report."""
    connection = duckdb.connect()
    connection.execute(f"SET threads={int(workers)}")
    connection.execute(f"SET memory_limit='{int(memory_limit_gb)}GB'")
    connection.execute(f"SET temp_directory='{_quote(temp_dir)}'")
    metrics: dict[str, Any] = {}
    try:
        for k in eval_ks:
            rows = connection.execute(
                f"""
                WITH labels AS (
                    SELECT DISTINCT session, target_type, aid
                    FROM read_parquet('{_quote(labels_path.resolve())}')
                ), predictions AS (
                    SELECT session, target_type, aids
                    FROM read_parquet('{_quote(candidate_glob.resolve())}')
                )
                SELECT
                    labels.target_type::TINYINT,
                    count(*)::BIGINT AS denominator,
                    sum(
                        CASE WHEN predictions.aids IS NOT NULL
                                  AND list_contains(list_slice(predictions.aids, 1, {int(k)}), labels.aid)
                             THEN 1 ELSE 0 END
                    )::BIGINT AS hits
                FROM labels
                INNER JOIN predictions USING (session, target_type)
                GROUP BY labels.target_type
                ORDER BY labels.target_type
                """
            ).fetchall()
            denominators = {int(kind): int(total) for kind, total, _ in rows}
            hits = {int(kind): int(value) for kind, _, value in rows}
            recalls = {
                kind: hits.get(kind, 0) / denominators[kind]
                for kind in TARGET_TYPES
                if denominators.get(kind, 0) > 0
            }
            weighted = sum(target_weights[kind] * recalls.get(kind, 0.0) for kind in TARGET_TYPES)
            metrics[f"recall_at_{k}"] = {
                "by_target_type": {str(kind): round(recalls.get(kind, 0.0), 8) for kind in TARGET_TYPES},
                "denominators": {str(kind): denominators.get(kind, 0) for kind in TARGET_TYPES},
                "hits": {str(kind): hits.get(kind, 0) for kind in TARGET_TYPES},
                "weighted": round(weighted, 8),
            }
    finally:
        connection.close()
    return metrics


def generate_dssm_recall(
    data_dir: str | Path,
    checkpoint_path: str | Path,
    queries_path: str | Path,
    labels_path: str | Path,
    output_dir: str | Path,
    config: dict[str, Any],
    *,
    dataset_name: str,
    topk: int | None = None,
    max_sessions: int | None = None,
) -> dict[str, Any]:
    source = Path(data_dir).resolve()
    checkpoint_file = Path(checkpoint_path).resolve()
    queries_file = Path(queries_path).resolve()
    labels_file = Path(labels_path).resolve()
    destination = Path(output_dir).resolve()
    for path in (checkpoint_file, queries_file, labels_file):
        if not path.is_file():
            raise FileNotFoundError(path)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite DSSM recall output: {destination}")

    requested_topk = int(topk or config["faiss"]["topk"])
    if requested_topk <= 0:
        raise ValueError("topk must be positive")
    if max_sessions is not None and max_sessions <= 0:
        raise ValueError("max_sessions must be positive")
    vocab_aids, vocab_item_ids, build_report = _load_vocab(source)
    if build_report.get("dataset") != dataset_name:
        raise AssertionError(
            f"DSSM data is for {build_report.get('dataset')!r}, not {dataset_name!r}"
        )
    actual_topk = min(requested_topk, len(vocab_aids))
    device = torch.device("cuda" if config["faiss"]["use_gpu"] else "cpu")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA inference was requested but PyTorch CUDA is unavailable")
    model, checkpoint_metadata = _load_model(
        checkpoint_file,
        vocab_hash=build_report["vocabulary"]["sha256"],
        snapshot=build_report["snapshot"],
        device=device,
    )
    if model.num_items != len(vocab_aids) + 2:
        raise AssertionError("Checkpoint item rows do not match item vocabulary")
    configured_max_history = int(config["dssm"]["max_sequence_length"])
    checkpoint_max_history = int(
        checkpoint_metadata["architecture"]["max_sequence_length"]
    )
    if configured_max_history != checkpoint_max_history:
        raise AssertionError(
            "Checkpoint max_sequence_length differs from inference configuration"
        )

    token = f"{os.getpid()}-{uuid.uuid4().hex[:8]}"
    staging = destination.parent / f".{destination.name}.building-{token}"
    candidate_dir = staging / "candidates"
    temp_dir = staging / "duckdb_tmp"
    staging.mkdir(parents=True, exist_ok=False)
    temp_dir.mkdir()
    started = time.perf_counter()
    writer = _RollingWriter(
        candidate_dir,
        _candidate_schema(actual_topk),
        compression=config["runtime"]["parquet_compression"],
        rows_per_file=100_000,
    )
    resources = None
    try:
        item_vectors = _normalized_item_vectors(model)
        index, resources, index_build_seconds = _build_flat_index(
            item_vectors,
            use_gpu=bool(config["faiss"]["use_gpu"]),
        )
        query_batch_size = int(config["faiss"]["query_batch_size"])
        max_history = configured_max_history
        schema = _candidate_schema(actual_topk)
        query_rows = 0
        query_groups = 0
        history_events = 0
        unknown_events = 0
        query_min_ts = 2**63 - 1
        search_seconds = 0.0
        exact_overlap: float | None = None

        parquet = pq.ParquetFile(queries_file)
        if parquet.metadata.num_rows == 0:
            raise ValueError("DSSM recall queries are empty")
        with torch.inference_mode():
            for record_batch in parquet.iter_batches(
                batch_size=query_batch_size,
                columns=["session", "aids", "timestamps", "event_types"],
                use_threads=False,
            ):
                if max_sessions is not None:
                    remaining = max_sessions - query_rows
                    if remaining <= 0:
                        break
                    if record_batch.num_rows > remaining:
                        record_batch = record_batch.slice(0, remaining)
                rows = record_batch.to_pydict()
                sessions, history_items, history_types, unknown, event_count, min_ts = _collate_queries(
                    rows,
                    vocab_aids=vocab_aids,
                    vocab_item_ids=vocab_item_ids,
                    max_history=max_history,
                )
                history_item_tensor = torch.from_numpy(history_items).to(device, non_blocking=True)
                history_type_tensor = torch.from_numpy(history_types).to(device, non_blocking=True)
                batch_tables: list[pa.Table] = []
                for target_type in TARGET_TYPES:
                    target_types = torch.full(
                        (len(sessions),), target_type, dtype=torch.long, device=device
                    )
                    with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                        query_vectors = model.encode_session(
                            history_item_tensor, history_type_tensor, target_types
                        )
                    query_numpy = np.ascontiguousarray(
                        query_vectors.float().cpu().numpy(), dtype=np.float32
                    )
                    search_started = time.perf_counter()
                    scores, indices = index.search(query_numpy, actual_topk)
                    search_seconds += time.perf_counter() - search_started
                    if np.any(indices < 0) or np.any(indices >= len(vocab_aids)):
                        raise AssertionError("FAISS returned an invalid item index")
                    if np.any(scores[:, 1:] > scores[:, :-1] + 1e-5):
                        raise AssertionError("FAISS scores are not sorted in descending order")
                    candidate_aids = vocab_aids[indices]
                    if exact_overlap is None:
                        sample_size = min(8, len(query_numpy))
                        exact_queries = torch.from_numpy(query_numpy[:sample_size]).to(device)
                        exact_items = torch.from_numpy(item_vectors).to(device)
                        exact_indices = torch.topk(
                            exact_queries @ exact_items.T,
                            k=min(20, actual_topk),
                            dim=1,
                        ).indices.cpu().numpy()
                        compare_k = exact_indices.shape[1]
                        overlap = [
                            len(set(exact_indices[row]) & set(indices[row, :compare_k])) / compare_k
                            for row in range(sample_size)
                        ]
                        exact_overlap = float(np.mean(overlap))
                        del exact_queries, exact_items
                        if exact_overlap < 0.99:
                            raise AssertionError(
                                f"FAISS FlatIP differs from exact search: overlap={exact_overlap}"
                            )
                    batch_tables.append(
                        _result_table(
                            sessions,
                            target_type,
                            candidate_aids,
                            scores.astype(np.float32, copy=False),
                            schema,
                        )
                    )
                writer.write(pa.concat_tables(batch_tables))
                query_rows += len(sessions)
                query_groups += len(sessions) * len(TARGET_TYPES)
                history_events += event_count
                unknown_events += unknown
                query_min_ts = min(query_min_ts, min_ts)
                if query_rows % 100_000 < len(sessions):
                    print(
                        f"[dssm-recall] dataset={dataset_name} sessions={query_rows:,} "
                        f"groups={query_groups:,} elapsed={time.perf_counter() - started:.1f}s",
                        flush=True,
                    )
        writer.close()
        if query_groups != query_rows * len(TARGET_TYPES):
            raise AssertionError("DSSM query-group count is inconsistent")
        if writer.total_rows != query_groups:
            raise AssertionError("Every DSSM query group must have one stored candidate list")
        del index, resources
        resources = None
        gc.collect()

        snapshot_max_ts = int(checkpoint_metadata["snapshot_max_ts"])
        leakage_passed = snapshot_max_ts < query_min_ts
        if not leakage_passed:
            raise AssertionError(
                f"DSSM snapshot max ts {snapshot_max_ts} is not before query min ts {query_min_ts}"
            )
        eval_ks = [int(k) for k in config["recall"]["eval_ks"] if int(k) <= actual_topk]
        target_weights = {
            index + 1: float(config["recall"]["type_weights"][name])
            for index, name in enumerate(("clicks", "carts", "orders"))
        }
        metrics_started = time.perf_counter()
        metrics = _evaluate(
            candidate_dir / "part-*.parquet",
            labels_file,
            eval_ks,
            target_weights,
            workers=int(config["runtime"]["workers"]),
            memory_limit_gb=int(config["runtime"]["duckdb_memory_limit_gb"]),
            temp_dir=temp_dir,
        )
        report = {
            "dataset": dataset_name,
            "source": "dssm",
            "inputs": {
                "data_dir": str(source),
                "checkpoint": str(checkpoint_file),
                "queries": str(queries_file),
                "labels": str(labels_file),
                "snapshot": build_report["snapshot"],
                "snapshot_max_ts": snapshot_max_ts,
                "query_min_ts": query_min_ts,
            },
            "configuration": {
                "index": "IndexFlatIP",
                "use_gpu": bool(config["faiss"]["use_gpu"]),
                "topk": actual_topk,
                "query_batch_size": query_batch_size,
                "max_history": max_history,
                "storage": "one target-conditioned fixed-size candidate list per query group",
            },
            "model": checkpoint_metadata,
            "index": {
                "items": len(vocab_aids),
                "embedding_dim": item_vectors.shape[1],
                "build_seconds": round(index_build_seconds, 6),
                "flatip_exact_top20_overlap": round(exact_overlap or 0.0, 8),
            },
            "queries": {
                "sessions": query_rows,
                "groups": query_groups,
                "limited": max_sessions is not None,
                "max_sessions": max_sessions,
                "history_events_after_last_n": history_events,
                "unknown_history_events": unknown_events,
                "unknown_history_rate": round(unknown_events / max(history_events, 1), 8),
            },
            "candidates": {
                "rows": writer.total_rows,
                "logical_candidate_count": writer.total_rows * actual_topk,
                "files": writer.file_index,
                "disk_bytes": _directory_size(candidate_dir),
            },
            "metrics": metrics,
            "timing": {
                "faiss_search_seconds": round(search_seconds, 6),
                "metrics_seconds": round(time.perf_counter() - metrics_started, 6),
                "total_seconds": round(time.perf_counter() - started, 6),
                "query_groups_per_second": round(query_groups / max(search_seconds, 1e-9), 3),
            },
            "assertions": {
                "passed": True,
                "snapshot_before_queries": leakage_passed,
                "flatip_matches_exact": (exact_overlap or 0.0) >= 0.99,
                "pad_unk_excluded_from_index": True,
                "one_row_per_query_group": writer.total_rows == query_groups,
            },
        }
        shutil.rmtree(temp_dir, ignore_errors=True)
        (staging / "report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        staging.replace(destination)
        return report
    except Exception:
        writer.close()
        del resources
        shutil.rmtree(staging, ignore_errors=True)
        raise


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate exact FAISS DSSM candidates.")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--dataset-name", choices=("ranker", "valid"), required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--topk", type=int)
    parser.add_argument("--max-sessions", type=int)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    config_path = args.config or os.environ.get("OTTO_CONFIG_PATH")
    if not config_path:
        raise ValueError("Provide --config or run through the experiment runner")
    experiment_dir = os.environ.get("OTTO_EXPERIMENT_DIR")
    if args.output_dir:
        output_dir = args.output_dir
    elif experiment_dir:
        output_dir = Path(experiment_dir) / "recall" / args.dataset_name / "dssm"
    else:
        raise ValueError("Provide --output-dir or run through the experiment runner")
    report = generate_dssm_recall(
        args.data_dir,
        args.checkpoint,
        args.queries,
        args.labels,
        output_dir,
        resolve_project_config(config_path),
        dataset_name=args.dataset_name,
        topk=args.topk,
        max_sessions=args.max_sessions,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
