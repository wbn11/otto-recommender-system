"""Evaluate DSSM recall by effective query-history length.

Purpose
-------
Diagnose whether sparse query histories are the weak point of the current DSSM
before changing its architecture.  The evaluator reuses persisted candidates;
it does not train a model or run FAISS again.

Workflow
--------
1. Load the point-in-time item vocabulary used by the DSSM checkpoint.
2. Stream query lists and count known items within the same last-N window used
   by DSSM inference.  PAD does not exist in raw queries and aids missing from
   the vocabulary count as UNK, so neither contributes to effective length.
3. Assign sessions to empty, short, medium and long buckets.
4. Join the lightweight bucket table to labels, the DSSM vocabulary and
   persisted candidates with DuckDB.  This separates labels that the FAISS
   item index can retrieve from model-ranking misses, then computes per-target
   and weighted Recall@K.

Inputs are a prepared DSSM directory, an existing DSSM candidate directory,
query-prefix Parquet and future-label Parquet.  Outputs are ``report.json``, a
tidy ``metrics.csv`` and an auditable ``session_history.parquet`` table.
"""

from __future__ import annotations

import argparse
import csv
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
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.config import resolve_project_config


TARGET_TYPES = (1, 2, 3)
TARGET_NAMES = {1: "clicks", 2: "carts", 3: "orders"}
BUCKET_NAMES = {0: "empty", 1: "short", 2: "medium", 3: "long"}
BUCKET_ORDER = ("overall", "empty", "short", "medium", "long")

HISTORY_SCHEMA = pa.schema(
    [
        pa.field("session", pa.int64(), nullable=False),
        pa.field("raw_history_length", pa.int16(), nullable=False),
        pa.field("effective_history_length", pa.int16(), nullable=False),
        pa.field("unknown_history_events", pa.int16(), nullable=False),
        pa.field("history_bucket_id", pa.int8(), nullable=False),
    ]
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _quote(path: str | Path) -> str:
    return str(path).replace("'", "''")


def _sql_path(path: Path) -> str:
    return _quote(path.resolve().as_posix())


def _load_vocab_aids(data_dir: Path, dataset_name: str) -> tuple[np.ndarray, dict[str, Any]]:
    report_path = data_dir / "build_report.json"
    if not report_path.is_file():
        raise FileNotFoundError(report_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("dataset") != dataset_name:
        raise AssertionError(
            f"DSSM data is for {report.get('dataset')!r}, not {dataset_name!r}"
        )
    vocab_path = data_dir / report["vocabulary"]["path"]
    if not vocab_path.is_file():
        raise FileNotFoundError(vocab_path)
    expected_hash = report["vocabulary"].get("sha256")
    if expected_hash and _sha256(vocab_path) != expected_hash:
        raise AssertionError("Vocabulary hash differs from DSSM build report")

    table = pq.read_table(vocab_path, columns=["aid", "item_id"]).sort_by("item_id")
    aids = table["aid"].combine_chunks().to_numpy(zero_copy_only=False).astype(np.int64)
    item_ids = table["item_id"].combine_chunks().to_numpy(zero_copy_only=False).astype(np.int64)
    if not np.array_equal(item_ids, np.arange(2, len(item_ids) + 2, dtype=np.int64)):
        raise AssertionError("DSSM item IDs must be contiguous from 2")
    if len(aids) == 0 or np.any(aids[1:] <= aids[:-1]):
        raise AssertionError("DSSM vocabulary aids must be strictly increasing")
    return aids, report


def _candidate_files(candidate_path: Path) -> tuple[list[Path], Path, Path | None]:
    """Return candidate files, their glob path and an optional source report."""

    if candidate_path.is_file():
        if candidate_path.suffix != ".parquet":
            raise ValueError(f"Candidate file must be Parquet: {candidate_path}")
        return [candidate_path], candidate_path, None
    if not candidate_path.is_dir():
        raise FileNotFoundError(candidate_path)

    if (candidate_path / "candidates").is_dir():
        data_dir = candidate_path / "candidates"
        report_path = candidate_path / "report.json"
    else:
        data_dir = candidate_path
        report_path = candidate_path.parent / "report.json"
    files = sorted(data_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No candidate Parquet files under {data_dir}")
    return files, data_dir / "*.parquet", report_path if report_path.is_file() else None


def _candidate_topk(first_file: Path) -> int:
    aid_type = pq.read_schema(first_file).field("aids").type
    if not pa.types.is_fixed_size_list(aid_type):
        raise TypeError("DSSM candidates must store aids as a fixed-size list")
    return int(aid_type.list_size)


def _bucket_ids(effective_lengths: np.ndarray) -> np.ndarray:
    buckets = np.full(len(effective_lengths), 3, dtype=np.int8)
    buckets[effective_lengths == 0] = 0
    buckets[(effective_lengths >= 1) & (effective_lengths <= 2)] = 1
    buckets[(effective_lengths >= 3) & (effective_lengths <= 5)] = 2
    return buckets


def _write_session_history(
    queries_path: Path,
    output_path: Path,
    vocab_aids: np.ndarray,
    *,
    max_history: int,
    batch_size: int,
) -> dict[str, int]:
    if max_history <= 0 or max_history > np.iinfo(np.int16).max:
        raise ValueError("max_history must fit in int16 and be positive")
    writer = pq.ParquetWriter(output_path, HISTORY_SCHEMA, compression="zstd")
    rows_written = 0
    raw_events = 0
    effective_events = 0
    unknown_events = 0
    try:
        parquet = pq.ParquetFile(queries_path)
        for batch in parquet.iter_batches(
            batch_size=batch_size,
            columns=["session", "aids"],
            use_threads=False,
        ):
            sessions = batch.column(0).to_numpy(zero_copy_only=False).astype(np.int64)
            histories = batch.column(1)
            offsets = histories.offsets.to_numpy(zero_copy_only=False).astype(np.int64)
            base_offset = int(offsets[0])
            offsets = offsets - base_offset
            flat_size = int(offsets[-1])
            raw_aids = histories.values.slice(base_offset, flat_size).to_numpy(
                zero_copy_only=False
            ).astype(np.int64)

            positions = np.searchsorted(vocab_aids, raw_aids)
            inside = positions < len(vocab_aids)
            known = np.zeros(len(raw_aids), dtype=np.bool_)
            known[inside] = vocab_aids[positions[inside]] == raw_aids[inside]
            prefix = np.empty(len(known) + 1, dtype=np.int64)
            prefix[0] = 0
            np.cumsum(known, dtype=np.int64, out=prefix[1:])

            ends = offsets[1:]
            starts = np.maximum(offsets[:-1], ends - max_history)
            raw_lengths = ends - starts
            effective_lengths = prefix[ends] - prefix[starts]
            row_unknown = raw_lengths - effective_lengths
            bucket_ids = _bucket_ids(effective_lengths)

            if np.any(effective_lengths > raw_lengths) or np.any(raw_lengths > max_history):
                raise AssertionError("Invalid DSSM history lengths")
            table = pa.Table.from_arrays(
                [
                    pa.array(sessions, type=pa.int64()),
                    pa.array(raw_lengths.astype(np.int16), type=pa.int16()),
                    pa.array(effective_lengths.astype(np.int16), type=pa.int16()),
                    pa.array(row_unknown.astype(np.int16), type=pa.int16()),
                    pa.array(bucket_ids, type=pa.int8()),
                ],
                schema=HISTORY_SCHEMA,
            )
            writer.write_table(table)
            rows_written += len(sessions)
            raw_events += int(raw_lengths.sum())
            effective_events += int(effective_lengths.sum())
            unknown_events += int(row_unknown.sum())
    finally:
        writer.close()
    return {
        "sessions": rows_written,
        "raw_history_events": raw_events,
        "effective_history_events": effective_events,
        "unknown_history_events": unknown_events,
    }


def _empty_metric() -> dict[str, dict[int, int]]:
    return {
        "denominators": {target_type: 0 for target_type in TARGET_TYPES},
        "retrievable_denominators": {
            target_type: 0 for target_type in TARGET_TYPES
        },
        "hits": {target_type: 0 for target_type in TARGET_TYPES},
    }


def _format_metric(
    counts: dict[str, dict[int, int]], target_weights: dict[int, float]
) -> dict[str, Any]:
    denominators = counts["denominators"]
    retrievable_denominators = counts["retrievable_denominators"]
    hits = counts["hits"]
    recalls = {
        target_type: hits[target_type] / denominators[target_type]
        for target_type in TARGET_TYPES
        if denominators[target_type] > 0
    }
    retrievable_rates = {
        target_type: retrievable_denominators[target_type] / denominators[target_type]
        for target_type in TARGET_TYPES
        if denominators[target_type] > 0
    }
    conditional_recalls = {
        target_type: hits[target_type] / retrievable_denominators[target_type]
        for target_type in TARGET_TYPES
        if retrievable_denominators[target_type] > 0
    }
    has_all_targets = all(denominators[target_type] > 0 for target_type in TARGET_TYPES)
    has_all_retrievable_targets = all(
        retrievable_denominators[target_type] > 0 for target_type in TARGET_TYPES
    )
    weighted = (
        sum(target_weights[target_type] * recalls[target_type] for target_type in TARGET_TYPES)
        if has_all_targets
        else None
    )
    weighted_retrievable_ceiling = (
        sum(
            target_weights[target_type] * retrievable_rates[target_type]
            for target_type in TARGET_TYPES
        )
        if has_all_targets
        else None
    )
    weighted_conditional = (
        sum(
            target_weights[target_type] * conditional_recalls[target_type]
            for target_type in TARGET_TYPES
        )
        if has_all_retrievable_targets
        else None
    )
    return {
        "by_target_type": {
            str(target_type): (
                round(recalls[target_type], 8) if target_type in recalls else None
            )
            for target_type in TARGET_TYPES
        },
        "denominators": {
            str(target_type): denominators[target_type] for target_type in TARGET_TYPES
        },
        "retrievable_denominators": {
            str(target_type): retrievable_denominators[target_type]
            for target_type in TARGET_TYPES
        },
        "retrievable_rate_by_target_type": {
            str(target_type): (
                round(retrievable_rates[target_type], 8)
                if target_type in retrievable_rates
                else None
            )
            for target_type in TARGET_TYPES
        },
        "recall_given_retrievable_by_target_type": {
            str(target_type): (
                round(conditional_recalls[target_type], 8)
                if target_type in conditional_recalls
                else None
            )
            for target_type in TARGET_TYPES
        },
        "hits": {str(target_type): hits[target_type] for target_type in TARGET_TYPES},
        "weighted": round(weighted, 8) if weighted is not None else None,
        "weighted_retrievable_ceiling": (
            round(weighted_retrievable_ceiling, 8)
            if weighted_retrievable_ceiling is not None
            else None
        ),
        "weighted_recall_given_retrievable": (
            round(weighted_conditional, 8) if weighted_conditional is not None else None
        ),
        "has_all_target_types": has_all_targets,
        "has_all_retrievable_target_types": has_all_retrievable_targets,
    }


def _evaluate(
    connection: duckdb.DuckDBPyConnection,
    *,
    eval_ks: Sequence[int],
    target_weights: dict[int, float],
) -> dict[str, dict[str, Any]]:
    raw_counts: dict[str, dict[int, dict[str, dict[int, int]]]] = {
        scope: {int(k): _empty_metric() for k in eval_ks} for scope in BUCKET_ORDER
    }
    for k in eval_ks:
        rows = connection.execute(
            f"""
            SELECT
                history.history_bucket_id::TINYINT,
                labels.target_type::TINYINT,
                count(*)::BIGINT AS denominator,
                count(dssm_vocab.aid)::BIGINT AS retrievable_denominator,
                sum(
                    CASE WHEN list_contains(
                                  list_slice(predictions.aids, 1, {int(k)}), labels.aid
                              )
                         THEN 1 ELSE 0 END
                )::BIGINT AS hits
            FROM labels
            INNER JOIN predictions USING (session, target_type)
            INNER JOIN history USING (session)
            LEFT JOIN dssm_vocab ON labels.aid = dssm_vocab.aid
            GROUP BY history.history_bucket_id, labels.target_type
            ORDER BY history.history_bucket_id, labels.target_type
            """
        ).fetchall()
        for bucket_id, target_type, denominator, retrievable_denominator, hits in rows:
            scope = BUCKET_NAMES[int(bucket_id)]
            target_type = int(target_type)
            raw_counts[scope][int(k)]["denominators"][target_type] = int(denominator)
            raw_counts[scope][int(k)]["retrievable_denominators"][target_type] = int(
                retrievable_denominator
            )
            raw_counts[scope][int(k)]["hits"][target_type] = int(hits)
        for target_type in TARGET_TYPES:
            raw_counts["overall"][int(k)]["denominators"][target_type] = sum(
                raw_counts[scope][int(k)]["denominators"][target_type]
                for scope in BUCKET_ORDER[1:]
            )
            raw_counts["overall"][int(k)]["retrievable_denominators"][target_type] = sum(
                raw_counts[scope][int(k)]["retrievable_denominators"][target_type]
                for scope in BUCKET_ORDER[1:]
            )
            raw_counts["overall"][int(k)]["hits"][target_type] = sum(
                raw_counts[scope][int(k)]["hits"][target_type]
                for scope in BUCKET_ORDER[1:]
            )

    return {
        scope: {
            f"recall_at_{int(k)}": _format_metric(raw_counts[scope][int(k)], target_weights)
            for k in eval_ks
        }
        for scope in BUCKET_ORDER
    }


def _history_distribution(
    connection: duckdb.DuckDBPyConnection,
) -> tuple[dict[str, dict[str, Any]], int]:
    rows = connection.execute(
        """
        WITH candidate_sessions AS (
            SELECT DISTINCT session FROM predictions
        )
        SELECT
            history.history_bucket_id::TINYINT,
            count(*)::BIGINT AS sessions,
            sum(history.raw_history_length)::BIGINT AS raw_events,
            sum(history.effective_history_length)::BIGINT AS effective_events,
            sum(history.unknown_history_events)::BIGINT AS unknown_events,
            avg(history.raw_history_length)::DOUBLE AS raw_mean,
            avg(history.effective_history_length)::DOUBLE AS effective_mean,
            min(history.effective_history_length)::INTEGER AS effective_min,
            max(history.effective_history_length)::INTEGER AS effective_max
        FROM candidate_sessions
        INNER JOIN history USING (session)
        GROUP BY history.history_bucket_id
        ORDER BY history.history_bucket_id
        """
    ).fetchall()
    result: dict[str, dict[str, Any]] = {
        name: {
            "sessions": 0,
            "session_share": 0.0,
            "raw_history_events": 0,
            "effective_history_events": 0,
            "unknown_history_events": 0,
            "unknown_history_rate": 0.0,
            "raw_history_length_mean": None,
            "effective_history_length_mean": None,
            "effective_history_length_min": None,
            "effective_history_length_max": None,
        }
        for name in BUCKET_ORDER[1:]
    }
    total_sessions = sum(int(row[1]) for row in rows)
    for (
        bucket_id,
        sessions,
        raw_events,
        effective_events,
        unknown_events,
        raw_mean,
        effective_mean,
        effective_min,
        effective_max,
    ) in rows:
        name = BUCKET_NAMES[int(bucket_id)]
        result[name] = {
            "sessions": int(sessions),
            "session_share": round(int(sessions) / max(total_sessions, 1), 8),
            "raw_history_events": int(raw_events),
            "effective_history_events": int(effective_events),
            "unknown_history_events": int(unknown_events),
            "unknown_history_rate": round(int(unknown_events) / max(int(raw_events), 1), 8),
            "raw_history_length_mean": round(float(raw_mean), 6),
            "effective_history_length_mean": round(float(effective_mean), 6),
            "effective_history_length_min": int(effective_min),
            "effective_history_length_max": int(effective_max),
        }
    return result, total_sessions


def _write_metrics_csv(
    path: Path,
    *,
    dataset_name: str,
    distributions: dict[str, dict[str, Any]],
    total_sessions: int,
    metrics: dict[str, dict[str, Any]],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "dataset",
                "history_bucket",
                "sessions",
                "k",
                "target_type",
                "target_type_id",
                "denominator",
                "retrievable_labels",
                "retrievable_rate",
                "hits",
                "recall",
                "recall_given_retrievable",
            ]
        )
        for scope in BUCKET_ORDER:
            sessions = total_sessions if scope == "overall" else distributions[scope]["sessions"]
            for metric_name, metric in metrics[scope].items():
                k = int(metric_name.rsplit("_", 1)[1])
                for target_type in TARGET_TYPES:
                    writer.writerow(
                        [
                            dataset_name,
                            scope,
                            sessions,
                            k,
                            TARGET_NAMES[target_type],
                            target_type,
                            metric["denominators"][str(target_type)],
                            metric["retrievable_denominators"][str(target_type)],
                            metric["retrievable_rate_by_target_type"][str(target_type)],
                            metric["hits"][str(target_type)],
                            metric["by_target_type"][str(target_type)],
                            metric["recall_given_retrievable_by_target_type"][
                                str(target_type)
                            ],
                        ]
                    )
                writer.writerow(
                    [
                        dataset_name,
                        scope,
                        sessions,
                        k,
                        "weighted",
                        "",
                        "",
                        "",
                        metric["weighted_retrievable_ceiling"],
                        "",
                        metric["weighted"],
                        metric["weighted_recall_given_retrievable"],
                    ]
                )


def _label_retrievability_summary(
    metrics: dict[str, dict[str, Any]], *, reference_k: int
) -> dict[str, dict[str, Any]]:
    """Extract the K-independent label/index coverage diagnostic."""

    result: dict[str, dict[str, Any]] = {}
    for scope in BUCKET_ORDER:
        metric = metrics[scope][f"recall_at_{reference_k}"]
        result[scope] = {
            "by_target_type": {
                str(target_type): {
                    "labels": metric["denominators"][str(target_type)],
                    "retrievable_labels": metric["retrievable_denominators"][
                        str(target_type)
                    ],
                    "retrievable_rate": metric["retrievable_rate_by_target_type"][
                        str(target_type)
                    ],
                }
                for target_type in TARGET_TYPES
            },
            "weighted_retrievable_ceiling": metric["weighted_retrievable_ceiling"],
        }
    return result


def analyze_dssm_history(
    data_dir: str | Path,
    candidate_path: str | Path,
    queries_path: str | Path,
    labels_path: str | Path,
    output_dir: str | Path,
    config: dict[str, Any],
    *,
    dataset_name: str,
) -> dict[str, Any]:
    source = Path(data_dir).resolve()
    candidates = Path(candidate_path).resolve()
    queries = Path(queries_path).resolve()
    labels = Path(labels_path).resolve()
    destination = Path(output_dir).resolve()
    for path in (queries, labels):
        if not path.is_file():
            raise FileNotFoundError(path)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite DSSM history analysis: {destination}")

    vocab_aids, build_report = _load_vocab_aids(source, dataset_name)
    candidate_files, candidate_glob, candidate_report_path = _candidate_files(candidates)
    topk = _candidate_topk(candidate_files[0])
    max_history = int(config["dssm"]["max_sequence_length"])
    candidate_report: dict[str, Any] | None = None
    if candidate_report_path is not None:
        candidate_report = json.loads(candidate_report_path.read_text(encoding="utf-8"))
        if candidate_report.get("dataset") != dataset_name:
            raise AssertionError("Candidate report belongs to a different dataset")
        candidate_max_history = int(candidate_report["configuration"]["max_history"])
        if candidate_max_history != max_history:
            raise AssertionError(
                f"Candidate max_history={candidate_max_history} differs from config={max_history}"
            )
        if int(candidate_report["configuration"]["topk"]) != topk:
            raise AssertionError("Candidate report Top-K differs from its Parquet schema")
        candidate_items = candidate_report.get("index", {}).get("items")
        if candidate_items is not None and int(candidate_items) != len(vocab_aids):
            raise AssertionError("DSSM candidates were built from a different vocabulary")
        candidate_snapshot = candidate_report.get("inputs", {}).get("snapshot")
        if candidate_snapshot is not None and candidate_snapshot != build_report.get("snapshot"):
            raise AssertionError("DSSM candidates were built from a different snapshot")

    eval_ks = sorted(
        {int(k) for k in config["recall"]["eval_ks"] if 0 < int(k) <= topk}
    )
    if not eval_ks:
        raise ValueError(f"No configured Recall@K fits candidate Top-{topk}")
    target_weights = {
        target_type: float(config["recall"]["type_weights"][name])
        for target_type, name in TARGET_NAMES.items()
    }
    if abs(sum(target_weights.values()) - 1.0) > 1e-9:
        raise ValueError("Target metric weights must sum to one")

    destination.parent.mkdir(parents=True, exist_ok=True)
    token = f"{os.getpid()}-{uuid.uuid4().hex[:8]}"
    staging = destination.parent / f".{destination.name}.building-{token}"
    staging.mkdir(parents=False, exist_ok=False)
    history_path = staging / "session_history.parquet"
    temp_dir = staging / "duckdb_tmp"
    temp_dir.mkdir()
    started = time.perf_counter()
    connection: duckdb.DuckDBPyConnection | None = None
    try:
        query_summary = _write_session_history(
            queries,
            history_path,
            vocab_aids,
            max_history=max_history,
            batch_size=max(4096, int(config["faiss"]["query_batch_size"])),
        )

        connection = duckdb.connect()
        connection.execute(f"SET threads={int(config['runtime']['workers'])}")
        connection.execute(
            f"SET memory_limit='{int(config['runtime']['duckdb_memory_limit_gb'])}GB'"
        )
        connection.execute("SET preserve_insertion_order=false")
        connection.execute(f"SET temp_directory='{_sql_path(temp_dir)}'")
        connection.execute(
            f"CREATE VIEW history AS SELECT * FROM read_parquet('{_sql_path(history_path)}')"
        )
        connection.execute(
            f"CREATE VIEW predictions AS SELECT session, target_type, aids "
            f"FROM read_parquet('{_sql_path(candidate_glob)}')"
        )
        connection.execute(
            f"CREATE VIEW labels AS SELECT DISTINCT session, target_type, aid "
            f"FROM read_parquet('{_sql_path(labels)}')"
        )
        connection.register(
            "dssm_vocab",
            pa.table({"aid": pa.array(vocab_aids, type=pa.int64())}),
        )

        validation = connection.execute(
            """
            WITH prediction_counts AS (
                SELECT
                    count(*)::BIGINT AS rows,
                    count(DISTINCT (session, target_type))::BIGINT AS unique_groups,
                    count(DISTINCT session)::BIGINT AS sessions
                FROM predictions
            ), history_counts AS (
                SELECT
                    count(*)::BIGINT AS rows,
                    count(DISTINCT session)::BIGINT AS unique_sessions,
                    sum(CASE WHEN effective_history_length > raw_history_length
                                  OR raw_history_length < 0
                                  OR unknown_history_events !=
                                     raw_history_length - effective_history_length
                             THEN 1 ELSE 0 END)::BIGINT AS invalid_rows
                FROM history
            ), missing AS (
                SELECT count(*)::BIGINT AS sessions
                FROM (SELECT DISTINCT session FROM predictions) predicted
                LEFT JOIN history USING (session)
                WHERE history.session IS NULL
            )
            SELECT
                prediction_counts.rows,
                prediction_counts.unique_groups,
                prediction_counts.sessions,
                history_counts.rows,
                history_counts.unique_sessions,
                history_counts.invalid_rows,
                missing.sessions
            FROM prediction_counts, history_counts, missing
            """
        ).fetchone()
        assert validation is not None
        (
            prediction_rows,
            unique_prediction_groups,
            prediction_sessions,
            history_rows,
            unique_history_sessions,
            invalid_history_rows,
            missing_history_sessions,
        ) = (int(value) for value in validation)
        if prediction_rows != unique_prediction_groups:
            raise AssertionError("DSSM candidate groups contain duplicates")
        if prediction_rows != prediction_sessions * len(TARGET_TYPES):
            raise AssertionError("Each evaluated session must have three DSSM target groups")
        if history_rows != unique_history_sessions:
            raise AssertionError("Query history contains duplicate sessions")
        if invalid_history_rows or missing_history_sessions:
            raise AssertionError("Invalid or missing DSSM history bucket rows")

        distributions, evaluated_sessions = _history_distribution(connection)
        if evaluated_sessions != prediction_sessions:
            raise AssertionError("History bucket sessions do not cover DSSM candidate sessions")
        metrics = _evaluate(
            connection,
            eval_ks=eval_ks,
            target_weights=target_weights,
        )
        label_retrievability = _label_retrievability_summary(
            metrics, reference_k=eval_ks[0]
        )
        overall_matches_candidate_report = True
        retrievability_is_k_independent = True
        hits_only_on_retrievable_labels = True
        for k in eval_ks:
            overall = metrics["overall"][f"recall_at_{k}"]
            for target_type in TARGET_TYPES:
                denominator_sum = sum(
                    metrics[scope][f"recall_at_{k}"]["denominators"][str(target_type)]
                    for scope in BUCKET_ORDER[1:]
                )
                hit_sum = sum(
                    metrics[scope][f"recall_at_{k}"]["hits"][str(target_type)]
                    for scope in BUCKET_ORDER[1:]
                )
                retrievable_sum = sum(
                    metrics[scope][f"recall_at_{k}"]["retrievable_denominators"][
                        str(target_type)
                    ]
                    for scope in BUCKET_ORDER[1:]
                )
                if denominator_sum != overall["denominators"][str(target_type)]:
                    raise AssertionError("Bucket denominators do not reproduce overall metrics")
                if retrievable_sum != overall["retrievable_denominators"][str(target_type)]:
                    raise AssertionError(
                        "Bucket retrievable labels do not reproduce overall coverage"
                    )
                if hit_sum != overall["hits"][str(target_type)]:
                    raise AssertionError("Bucket hits do not reproduce overall metrics")
                for scope in BUCKET_ORDER:
                    metric = metrics[scope][f"recall_at_{k}"]
                    target = str(target_type)
                    if metric["hits"][target] > metric["retrievable_denominators"][target]:
                        hits_only_on_retrievable_labels = False
                    reference = metrics[scope][f"recall_at_{eval_ks[0]}"]
                    if (
                        metric["retrievable_denominators"][target]
                        != reference["retrievable_denominators"][target]
                    ):
                        retrievability_is_k_independent = False
        if not hits_only_on_retrievable_labels:
            raise AssertionError("A DSSM hit refers to an item outside its own vocabulary")
        if not retrievability_is_k_independent:
            raise AssertionError("Label/index coverage unexpectedly changes with K")

        for k in eval_ks:
            overall = metrics["overall"][f"recall_at_{k}"]
            if candidate_report is not None:
                expected = candidate_report.get("metrics", {}).get(f"recall_at_{k}")
                if expected is not None:
                    overall_matches_candidate_report = (
                        overall["denominators"] == expected["denominators"]
                        and overall["hits"] == expected["hits"]
                        and abs(float(overall["weighted"]) - float(expected["weighted"])) < 1e-8
                    )
                    if not overall_matches_candidate_report:
                        raise AssertionError(
                            "History-bucket Overall differs from the original DSSM report"
                        )

        _write_metrics_csv(
            staging / "metrics.csv",
            dataset_name=dataset_name,
            distributions=distributions,
            total_sessions=evaluated_sessions,
            metrics=metrics,
        )
        connection.close()
        connection = None
        shutil.rmtree(temp_dir, ignore_errors=True)

        report = {
            "dataset": dataset_name,
            "purpose": "diagnose DSSM recall by effective known-item history length",
            "inputs": {
                "data_dir": str(source),
                "vocabulary": str(source / build_report["vocabulary"]["path"]),
                "queries": str(queries),
                "labels": str(labels),
                "candidates": str(candidates),
                "candidate_files": len(candidate_files),
            },
            "configuration": {
                "max_history": max_history,
                "candidate_topk": topk,
                "eval_ks": eval_ks,
                "target_weights": {
                    str(target_type): target_weights[target_type]
                    for target_type in TARGET_TYPES
                },
                "effective_length": (
                    "known vocabulary events after last-N truncation; PAD and UNK excluded"
                ),
                "history_buckets": {
                    "empty": "n = 0",
                    "short": "1 <= n <= 2",
                    "medium": "3 <= n <= 5",
                    "long": "n > 5",
                },
            },
            "queries": query_summary,
            "evaluated_sessions": evaluated_sessions,
            "history_distribution": distributions,
            "label_retrievability": label_retrievability,
            "metrics": metrics,
            "artifacts": {
                "session_history": "session_history.parquet",
                "metrics_csv": "metrics.csv",
            },
            "timing": {"total_seconds": round(time.perf_counter() - started, 6)},
            "assertions": {
                "passed": True,
                "one_history_row_per_query_session": history_rows == unique_history_sessions,
                "one_candidate_row_per_query_group": prediction_rows == unique_prediction_groups,
                "three_target_groups_per_evaluated_session": (
                    prediction_rows == prediction_sessions * len(TARGET_TYPES)
                ),
                "all_candidate_sessions_have_history": missing_history_sessions == 0,
                "valid_effective_lengths": invalid_history_rows == 0,
                "candidate_matches_vocabulary_and_snapshot": True,
                "bucket_sessions_sum_to_overall": (
                    sum(distributions[name]["sessions"] for name in BUCKET_ORDER[1:])
                    == evaluated_sessions
                ),
                "bucket_metrics_sum_to_overall": True,
                "hits_only_on_retrievable_labels": hits_only_on_retrievable_labels,
                "retrievability_is_k_independent": retrievability_is_k_independent,
                "overall_matches_candidate_report": overall_matches_candidate_report,
            },
        }
        (staging / "report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        staging.replace(destination)
        return report
    except Exception:
        if connection is not None:
            connection.close()
        shutil.rmtree(staging, ignore_errors=True)
        raise


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate DSSM recall by history length.")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--dataset-name", choices=("ranker", "valid"), required=True)
    parser.add_argument("--output-dir", type=Path)
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
        output_dir = Path(experiment_dir) / "metrics" / args.dataset_name / "dssm_history"
    else:
        raise ValueError("Provide --output-dir or run through the experiment runner")
    report = analyze_dssm_history(
        args.data_dir,
        args.candidates,
        args.queries,
        args.labels,
        output_dir,
        resolve_project_config(config_path),
        dataset_name=args.dataset_name,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
