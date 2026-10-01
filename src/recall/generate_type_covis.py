"""Generate and evaluate session-level candidates from a type-CoVis matrix."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

import duckdb
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.schemas import (
    COVIS_MATRIX_SCHEMA,
    EVENT_TYPE_TO_ID,
    LABEL_SCHEMA,
    QUERY_SCHEMA,
    SESSION_RECALL_SCHEMA,
)
from utils.config import resolve_project_config


TARGET_TYPES = tuple(sorted(EVENT_TYPE_TO_ID.values()))


def _quote(value: str | Path) -> str:
    return str(value).replace("'", "''")


def _path(path: Path) -> str:
    return path.resolve().as_posix()


def _sql_path_list(paths: Sequence[Path]) -> str:
    return "[" + ",".join(f"'{_quote(_path(path))}'" for path in paths) + "]"


def _directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _parquet_rows(paths: Sequence[Path]) -> int:
    return sum(pq.ParquetFile(path).metadata.num_rows for path in paths)


def _check_schema(path: Path, expected, name: str) -> None:
    actual = pq.read_schema(path)
    if actual.names != expected.names or actual.types != expected.types:
        raise AssertionError(f"Unexpected {name} schema in {path}: {actual}")


def _weight_case(column: str, event_type_weights: Mapping[int, float]) -> str:
    clauses = " ".join(
        f"WHEN {int(event_type)} THEN {float(weight)}"
        for event_type, weight in sorted(event_type_weights.items())
    )
    return f"CASE {column} {clauses} ELSE NULL END"


def _write_query_seeds(
    connection: duckdb.DuckDBPyConnection,
    queries_path: Path,
    output_dir: Path,
    *,
    recent_events: int,
    recency_scale_events: float,
    event_type_weights: Mapping[int, float],
    aid_buckets: int,
    compression: str,
) -> None:
    behavior_weight = _weight_case("event_type", event_type_weights)
    connection.execute(
        f"""
        COPY (
            WITH exploded AS (
                SELECT
                    session::BIGINT AS session,
                    unnest(aids)::BIGINT AS aid,
                    unnest(event_types)::TINYINT AS event_type,
                    generate_subscripts(aids, 1)::INTEGER AS position,
                    array_length(aids)::INTEGER AS history_length
                FROM read_parquet('{_quote(_path(queries_path))}')
            ),
            recent AS (
                SELECT *
                FROM exploded
                WHERE position > history_length - {int(recent_events)}
            ),
            session_items AS (
                SELECT
                    session,
                    aid,
                    max(position)::INTEGER AS last_position,
                    max(history_length)::INTEGER AS history_length,
                    max({behavior_weight})::DOUBLE AS strongest_behavior_weight
                FROM recent
                GROUP BY session, aid
            )
            SELECT
                (aid % {int(aid_buckets)})::INTEGER AS aid_bucket,
                session::BIGINT AS session,
                aid::BIGINT AS seed_aid,
                (
                    exp(
                        -(history_length - last_position)::DOUBLE
                        / {float(recency_scale_events)}
                    ) * strongest_behavior_weight
                )::DOUBLE AS seed_weight
            FROM session_items
        ) TO '{_quote(_path(output_dir))}' (
            FORMAT PARQUET,
            PARTITION_BY (aid_bucket),
            COMPRESSION {compression.upper()}
        )
        """
    )


def _write_candidate_partial(
    connection: duckdb.DuckDBPyConnection,
    seed_files: Sequence[Path],
    matrix_files: Sequence[Path],
    output_dir: Path,
    *,
    session_buckets: int,
    compression: str,
) -> None:
    connection.execute(
        f"""
        COPY (
            SELECT
                (seeds.session % {int(session_buckets)})::INTEGER AS session_bucket,
                seeds.session::BIGINT AS session,
                matrix.neighbor_aid::BIGINT AS aid,
                sum(seeds.seed_weight * matrix.source_score)::DOUBLE AS partial_score
            FROM read_parquet({_sql_path_list(seed_files)}, union_by_name=true) seeds
            INNER JOIN read_parquet({_sql_path_list(matrix_files)}, union_by_name=true) matrix
              ON seeds.seed_aid = matrix.aid
            GROUP BY session_bucket, seeds.session, matrix.neighbor_aid
        ) TO '{_quote(_path(output_dir))}' (
            FORMAT PARQUET,
            PARTITION_BY (session_bucket),
            COMPRESSION {compression.upper()}
        )
        """
    )


def _finalize_session_bucket(
    connection: duckdb.DuckDBPyConnection,
    partial_files: Sequence[Path],
    output_path: Path,
    *,
    source_name: str,
    topk: int,
    compression: str,
    row_group_size: int,
) -> int:
    connection.execute(
        f"""
        COPY (
            WITH aggregated AS (
                SELECT
                    session::BIGINT AS session,
                    aid::BIGINT AS aid,
                    sum(partial_score)::DOUBLE AS source_score
                FROM read_parquet({_sql_path_list(partial_files)}, union_by_name=true)
                GROUP BY session, aid
            ),
            ranked AS (
                SELECT
                    session,
                    aid,
                    row_number() OVER (
                        PARTITION BY session
                        ORDER BY source_score DESC, aid ASC
                    )::INTEGER AS source_rank,
                    source_score::FLOAT AS source_score
                FROM aggregated
            )
            SELECT
                session::BIGINT AS session,
                aid::BIGINT AS aid,
                '{_quote(source_name)}'::VARCHAR AS source,
                source_rank::INTEGER AS source_rank,
                source_score::FLOAT AS source_score
            FROM ranked
            WHERE source_rank <= {int(topk)}
            ORDER BY session, source_rank
        ) TO '{_quote(_path(output_path))}' (
            FORMAT PARQUET,
            COMPRESSION {compression.upper()},
            ROW_GROUP_SIZE {int(row_group_size)}
        )
        """
    )
    _check_schema(output_path, SESSION_RECALL_SCHEMA, f"{source_name} candidate")
    return pq.ParquetFile(output_path).metadata.num_rows


def _denominators(
    connection: duckdb.DuckDBPyConnection,
    labels_path: Path,
    k: int,
) -> dict[int, int]:
    rows = connection.execute(
        f"""
        SELECT target_type, sum(least(label_count, {int(k)}))::BIGINT
        FROM (
            SELECT session, target_type, count(*)::BIGINT AS label_count
            FROM read_parquet('{_quote(_path(labels_path))}')
            GROUP BY session, target_type
        )
        GROUP BY target_type ORDER BY target_type
        """
    ).fetchall()
    return {int(target): int(value) for target, value in rows}


def _evaluate(
    connection: duckdb.DuckDBPyConnection,
    labels_path: Path,
    candidate_glob: str,
    *,
    eval_ks: Sequence[int],
    target_weights: Mapping[int, float],
) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for k in sorted(set(int(value) for value in eval_ks)):
        denominators = _denominators(connection, labels_path, k)
        hit_rows = connection.execute(
            f"""
            SELECT labels.target_type, count(*)::BIGINT
            FROM read_parquet('{_quote(_path(labels_path))}') labels
            INNER JOIN read_parquet('{_quote(candidate_glob)}') candidates
              ON labels.session = candidates.session AND labels.aid = candidates.aid
            WHERE candidates.source_rank <= {int(k)}
            GROUP BY labels.target_type ORDER BY labels.target_type
            """
        ).fetchall()
        hits = {int(target): int(value) for target, value in hit_rows}
        recalls: dict[str, float | None] = {}
        weighted = 0.0
        has_all_targets = True
        for target_type in TARGET_TYPES:
            denominator = denominators.get(target_type, 0)
            recall = hits.get(target_type, 0) / denominator if denominator else None
            recalls[str(target_type)] = round(recall, 8) if recall is not None else None
            if recall is None:
                has_all_targets = False
            else:
                weighted += target_weights[target_type] * recall
        metrics[f"recall_at_{k}"] = {
            "by_target_type": recalls,
            "weighted": round(weighted, 8) if has_all_targets else None,
            "hits": {str(key): value for key, value in hits.items()},
            "denominators": {str(key): value for key, value in denominators.items()},
        }
    return metrics


def _candidate_statistics(
    connection: duckdb.DuckDBPyConnection,
    queries_path: Path,
    candidate_glob: str,
    matrix_glob: str,
) -> dict[str, Any]:
    summary = connection.execute(
        f"""
        WITH candidate_counts AS (
            SELECT session, count(*)::BIGINT AS candidate_count
            FROM read_parquet('{_quote(candidate_glob)}')
            GROUP BY session
        ), all_counts AS (
            SELECT queries.session, coalesce(candidate_count, 0)::BIGINT AS candidate_count
            FROM read_parquet('{_quote(_path(queries_path))}') queries
            LEFT JOIN candidate_counts USING (session)
        )
        SELECT
            min(candidate_count)::BIGINT,
            max(candidate_count)::BIGINT,
            avg(candidate_count)::DOUBLE,
            quantile_cont(candidate_count, 0.50)::DOUBLE,
            quantile_cont(candidate_count, 0.90)::DOUBLE,
            quantile_cont(candidate_count, 0.95)::DOUBLE,
            quantile_cont(candidate_count, 0.99)::DOUBLE,
            sum(CASE WHEN candidate_count = 0 THEN 1 ELSE 0 END)::BIGINT,
            sum(CASE WHEN candidate_count < 5 THEN 1 ELSE 0 END)::BIGINT,
            sum(CASE WHEN candidate_count < 20 THEN 1 ELSE 0 END)::BIGINT,
            count(*)::BIGINT
        FROM all_counts
        """
    ).fetchone()
    unique_candidates = int(connection.execute(
        f"SELECT count(DISTINCT aid) FROM read_parquet('{_quote(candidate_glob)}')"
    ).fetchone()[0])
    matrix_items = int(connection.execute(
        f"""SELECT count(*) FROM (
                SELECT aid FROM read_parquet('{_quote(matrix_glob)}')
                UNION
                SELECT neighbor_aid AS aid FROM read_parquet('{_quote(matrix_glob)}')
             )"""
    ).fetchone()[0])
    sessions = int(summary[10])
    zero_sessions = int(summary[7])
    return {
        "candidate_count_per_session": {
            "min": int(summary[0]),
            "max": int(summary[1]),
            "mean": round(float(summary[2]), 6),
            "p50": float(summary[3]),
            "p90": float(summary[4]),
            "p95": float(summary[5]),
            "p99": float(summary[6]),
        },
        "sessions": sessions,
        "sessions_without_candidates": zero_sessions,
        "session_coverage": round((sessions - zero_sessions) / sessions, 8) if sessions else None,
        "sessions_below_5_candidates": int(summary[8]),
        "sessions_below_20_candidates": int(summary[9]),
        "logical_query_groups": sessions * len(TARGET_TYPES),
        "unique_candidate_items": unique_candidates,
        "matrix_items": matrix_items,
        "matrix_item_coverage": (
            round(unique_candidates / matrix_items, 8) if matrix_items else None
        ),
    }


def generate_type_covis(
    matrix_dir: str | Path,
    queries_path: str | Path,
    labels_path: str | Path,
    output_dir: str | Path,
    *,
    dataset_name: str,
    recent_events: int = 30,
    recency_scale_events: float = 10.0,
    event_type_weights: Mapping[int, float] | None = None,
    topk: int = 200,
    eval_ks: Sequence[int] = (20, 50, 100),
    target_weights: Mapping[int, float] | None = None,
    aid_buckets: int = 256,
    matrix_buckets_per_pass: int = 16,
    compression: str = "zstd",
    row_group_size: int = 250_000,
    workers: int = 8,
    memory_limit_gb: int = 8,
    source_name: str = "type_covis",
) -> dict[str, Any]:
    """Join recent query seeds to the matrix with bounded disk-backed aggregation."""

    matrix_source = Path(matrix_dir).resolve()
    query_source = Path(queries_path).resolve()
    label_source = Path(labels_path).resolve()
    target = Path(output_dir).resolve()
    matrix_files = sorted(matrix_source.glob("bucket-*.parquet"))
    if not matrix_files:
        raise FileNotFoundError(f"No {source_name} matrix buckets found in {matrix_source}")
    for matrix_file in matrix_files:
        _check_schema(matrix_file, COVIS_MATRIX_SCHEMA, f"{source_name} matrix")
    if not query_source.is_file() or not label_source.is_file():
        raise FileNotFoundError("Query and label Parquet files are required")
    _check_schema(query_source, QUERY_SCHEMA, "query")
    _check_schema(label_source, LABEL_SCHEMA, "label")
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite {source_name} recall: {target}")
    if recent_events <= 0 or recency_scale_events <= 0 or topk <= 0:
        raise ValueError("recent_events, recency_scale_events and topk must be positive")
    if aid_buckets <= 0 or matrix_buckets_per_pass <= 0:
        raise ValueError("Bucket settings must be positive")
    if workers <= 0 or memory_limit_gb <= 0:
        raise ValueError("workers and memory_limit_gb must be positive")
    if any(int(value) <= 0 or int(value) > topk for value in eval_ks):
        raise ValueError("Evaluation K values must be positive and no greater than topk")

    behavior_weights = dict(event_type_weights or {1: 1.0, 2: 3.0, 3: 6.0})
    metric_weights = dict(target_weights or {1: 0.1, 2: 0.3, 3: 0.6})
    if set(behavior_weights) != set(TARGET_TYPES) or set(metric_weights) != set(TARGET_TYPES):
        raise ValueError("Weights must define event types 1, 2 and 3")
    if any(value <= 0 or not math.isfinite(value) for value in behavior_weights.values()):
        raise ValueError("Behavior weights must be positive and finite")
    if any(value < 0 or not math.isfinite(value) for value in metric_weights.values()):
        raise ValueError("Target metric weights must be non-negative and finite")
    if not math.isclose(sum(metric_weights.values()), 1.0, abs_tol=1e-9):
        raise ValueError("Target metric weights must sum to 1.0")

    matrix_buckets = {
        int(path.stem.split("-", 1)[1]): path for path in matrix_files
    }
    if any(bucket < 0 or bucket >= aid_buckets for bucket in matrix_buckets):
        raise ValueError("Matrix bucket ids do not match configured aid_buckets")

    target.parent.mkdir(parents=True, exist_ok=True)
    token = f"{os.getpid()}-{uuid.uuid4().hex[:8]}"
    staging = target.parent / f".{target.name}.building-{token}"
    staging.mkdir(parents=True, exist_ok=False)
    seed_dir = staging / "query_seeds"
    partial_dir = staging / "candidate_partials"
    candidate_dir = staging / "candidates"
    temp_dir = staging / "duckdb_tmp"
    partial_dir.mkdir()
    candidate_dir.mkdir()
    temp_dir.mkdir()
    started = time.perf_counter()
    connection = duckdb.connect()

    try:
        connection.execute(f"SET threads={int(workers)}")
        connection.execute(f"SET memory_limit='{int(memory_limit_gb)}GB'")
        connection.execute(f"SET temp_directory='{_quote(_path(temp_dir))}'")
        connection.execute("SET preserve_insertion_order=false")
        invalid_queries = int(connection.execute(
            f"""SELECT count(*) FROM read_parquet('{_quote(_path(query_source))}')
                 WHERE array_length(aids) < 1
                    OR array_length(aids) != array_length(timestamps)
                    OR array_length(aids) != array_length(event_types)
                    OR list_min(event_types) < 1
                    OR list_max(event_types) > 3"""
        ).fetchone()[0])
        if invalid_queries:
            raise ValueError(f"Found {invalid_queries} invalid query rows")
        query_min_ts = int(connection.execute(
            f"SELECT min(list_min(timestamps)) FROM read_parquet('{_quote(_path(query_source))}')"
        ).fetchone()[0])
        build_report_path = matrix_source.parent / "build_report.json"
        matrix_snapshot_max_ts = None
        if build_report_path.is_file():
            build_report = json.loads(build_report_path.read_text(encoding="utf-8"))
            matrix_source_name = build_report.get("source")
            if matrix_source_name is not None and matrix_source_name != source_name:
                raise ValueError(
                    f"Expected a {source_name} matrix, found {matrix_source_name}"
                )
            matrix_snapshot_max_ts = int(build_report["snapshot_stats"]["max_ts"])
            matrix_pair_buckets = build_report.get("configuration", {}).get("pair_buckets")
            if matrix_pair_buckets is not None and int(matrix_pair_buckets) != aid_buckets:
                raise ValueError(
                    "Configured aid_buckets does not match the matrix build configuration"
                )
            if matrix_snapshot_max_ts >= query_min_ts:
                raise AssertionError(
                    "Point-in-time violation: matrix snapshot must precede query history"
                )

        seed_started = time.perf_counter()
        _write_query_seeds(
            connection,
            query_source,
            seed_dir,
            recent_events=recent_events,
            recency_scale_events=recency_scale_events,
            event_type_weights=behavior_weights,
            aid_buckets=aid_buckets,
            compression=compression,
        )
        seed_files = sorted(seed_dir.rglob("*.parquet"))
        seed_rows = _parquet_rows(seed_files)
        seed_runtime = round(time.perf_counter() - seed_started, 6)

        session_buckets = min(aid_buckets, 64)
        join_started = time.perf_counter()
        bucket_ids = sorted(matrix_buckets)
        passes = [
            bucket_ids[index:index + matrix_buckets_per_pass]
            for index in range(0, len(bucket_ids), matrix_buckets_per_pass)
        ]
        completed_passes = 0
        for pass_index, bucket_group in enumerate(passes):
            group_seed_files = [
                file
                for bucket in bucket_group
                for file in sorted((seed_dir / f"aid_bucket={bucket}").glob("*.parquet"))
            ]
            if not group_seed_files:
                continue
            group_matrix_files = [matrix_buckets[bucket] for bucket in bucket_group]
            pass_output = partial_dir / f"pass-{pass_index:05d}"
            _write_candidate_partial(
                connection,
                group_seed_files,
                group_matrix_files,
                pass_output,
                session_buckets=session_buckets,
                compression=compression,
            )
            completed_passes += 1
            print(
                f"[{source_name}-recall] matrix_pass={pass_index + 1}/{len(passes)} "
                f"elapsed={time.perf_counter() - join_started:.1f}s",
                flush=True,
            )
        partial_files = sorted(partial_dir.rglob("*.parquet"))
        if not partial_files:
            raise ValueError(f"{source_name} matrix produced no candidates for these queries")
        partial_rows = _parquet_rows(partial_files)
        partial_disk_bytes = _directory_size(partial_dir)
        join_runtime = round(time.perf_counter() - join_started, 6)

        finalize_started = time.perf_counter()
        candidate_rows = 0
        candidate_files: list[Path] = []
        session_bucket_values = sorted({
            int(path.name.split("=", 1)[1])
            for pass_dir in partial_dir.glob("pass-*")
            for path in pass_dir.glob("session_bucket=*")
        })
        for position, session_bucket in enumerate(session_bucket_values, start=1):
            bucket_partials = sorted(
                partial_dir.glob(f"pass-*/session_bucket={session_bucket}/*.parquet")
            )
            output_path = candidate_dir / f"session-bucket-{session_bucket:05d}.parquet"
            candidate_rows += _finalize_session_bucket(
                connection,
                bucket_partials,
                output_path,
                source_name=source_name,
                topk=topk,
                compression=compression,
                row_group_size=row_group_size,
            )
            candidate_files.append(output_path)
            if position % 16 == 0 or position == len(session_bucket_values):
                print(
                    f"[{source_name}-recall] finalized={position}/{len(session_bucket_values)} "
                    f"candidate_rows={candidate_rows:,}",
                    flush=True,
                )
        candidate_glob = _path(candidate_dir / "session-bucket-*.parquet")
        matrix_glob = _path(matrix_source / "bucket-*.parquet")
        violations = {
            "duplicate_candidates": int(connection.execute(
                f"""SELECT count(*) FROM (
                        SELECT session, aid, count(*) AS n
                        FROM read_parquet('{_quote(candidate_glob)}')
                        GROUP BY session, aid HAVING count(*) > 1
                     )"""
            ).fetchone()[0]),
            "sessions_over_topk": int(connection.execute(
                f"""SELECT count(*) FROM (
                        SELECT session, count(*) AS n
                        FROM read_parquet('{_quote(candidate_glob)}')
                        GROUP BY session HAVING count(*) > {int(topk)}
                     )"""
            ).fetchone()[0]),
            "invalid_rank": int(connection.execute(
                f"""SELECT count(*) FROM read_parquet('{_quote(candidate_glob)}')
                     WHERE source_rank < 1 OR source_rank > {int(topk)}"""
            ).fetchone()[0]),
            "non_positive_score": int(connection.execute(
                f"""SELECT count(*) FROM read_parquet('{_quote(candidate_glob)}')
                     WHERE source_score <= 0 OR source_score IS NULL"""
            ).fetchone()[0]),
        }
        if any(violations.values()):
            raise AssertionError(f"Invalid {source_name} candidates: {violations}")
        finalize_runtime = round(time.perf_counter() - finalize_started, 6)

        metrics_started = time.perf_counter()
        metrics = _evaluate(
            connection,
            label_source,
            candidate_glob,
            eval_ks=eval_ks,
            target_weights=metric_weights,
        )
        statistics = _candidate_statistics(
            connection,
            query_source,
            candidate_glob,
            matrix_glob,
        )
        metrics_runtime = round(time.perf_counter() - metrics_started, 6)
        report = {
            "dataset": dataset_name,
            "source": source_name,
            "inputs": {
                "matrix": str(matrix_source),
                "matrix_files": len(matrix_files),
                "queries": str(query_source),
                "query_rows": pq.ParquetFile(query_source).metadata.num_rows,
                "labels": str(label_source),
                "label_rows": pq.ParquetFile(label_source).metadata.num_rows,
                "matrix_snapshot_max_ts": matrix_snapshot_max_ts,
                "query_min_ts": query_min_ts,
            },
            "output": str(target),
            "configuration": {
                "recent_events": recent_events,
                "recency_scale_events": recency_scale_events,
                "event_type_weights": {
                    str(key): value for key, value in behavior_weights.items()
                },
                "topk": topk,
                "eval_ks": sorted(set(int(value) for value in eval_ks)),
                "aid_buckets": aid_buckets,
                "session_buckets": session_buckets,
                "matrix_buckets_per_pass": matrix_buckets_per_pass,
                "storage": "one target-agnostic candidate list per session",
            },
            "query_seeds": {
                "rows": seed_rows,
                "files": len(seed_files),
                "runtime_seconds": seed_runtime,
                "retained": False,
            },
            "candidate_partials": {
                "completed_passes": completed_passes,
                "files": len(partial_files),
                "rows": partial_rows,
                "peak_disk_bytes": partial_disk_bytes,
                "runtime_seconds": join_runtime,
                "retained": False,
            },
            "candidates": {
                "files": len(candidate_files),
                "stored_rows": candidate_rows,
                "logical_rows_across_three_targets": candidate_rows * len(TARGET_TYPES),
                "disk_bytes": _directory_size(candidate_dir),
                "runtime_seconds": finalize_runtime,
            },
            "candidate_statistics": statistics,
            "metrics": metrics,
            "assertions": {"passed": not any(violations.values()), "violations": violations},
            "leakage_assertions": {
                "checked": matrix_snapshot_max_ts is not None,
                "passed": (
                    matrix_snapshot_max_ts is not None
                    and matrix_snapshot_max_ts < query_min_ts
                ),
            },
            "metrics_runtime_seconds": metrics_runtime,
            "runtime_seconds": round(time.perf_counter() - started, 6),
        }
        (staging / "metrics.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    except Exception:
        connection.close()
        shutil.rmtree(staging, ignore_errors=True)
        raise
    else:
        connection.close()

    shutil.rmtree(seed_dir, ignore_errors=True)
    shutil.rmtree(partial_dir, ignore_errors=True)
    shutil.rmtree(temp_dir, ignore_errors=True)
    (staging / "metrics.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    staging.replace(target)
    return report


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate candidates from type-CoVis Top-K.")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--matrix-dir", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--dataset-name", choices=("ranker", "valid"), required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--topk", type=int)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    config_path = args.config or os.environ.get("OTTO_CONFIG_PATH")
    if not config_path:
        raise ValueError("Provide --config or run through the experiment runner")
    config = resolve_project_config(config_path)
    experiment_dir = os.environ.get("OTTO_EXPERIMENT_DIR")
    if args.output_dir:
        output_dir = args.output_dir
    elif experiment_dir:
        output_dir = (
            Path(experiment_dir) / "recall" / args.dataset_name / "type_covis" / "recall"
        )
    else:
        raise ValueError("Provide --output-dir or run through the experiment runner")

    recall_config = config["recall"]
    covis_config = recall_config["type_covis"]
    behavior_weights = {
        EVENT_TYPE_TO_ID[name]: float(value)
        for name, value in covis_config["event_type_weights"].items()
    }
    target_weights = {
        EVENT_TYPE_TO_ID[name]: float(value)
        for name, value in recall_config["type_weights"].items()
    }
    report = generate_type_covis(
        args.matrix_dir,
        args.queries,
        args.labels,
        output_dir,
        dataset_name=args.dataset_name,
        recent_events=covis_config["recent_events_per_session"],
        recency_scale_events=recall_config["revisit"]["recency_scale_events"],
        event_type_weights=behavior_weights,
        topk=args.topk or recall_config["topk_per_source"],
        eval_ks=recall_config["eval_ks"],
        target_weights=target_weights,
        aid_buckets=covis_config["pair_buckets"],
        compression=config["runtime"]["parquet_compression"],
        row_group_size=config["runtime"]["parquet_row_group_size"],
        workers=config["runtime"]["workers"],
        memory_limit_gb=config["runtime"]["duckdb_memory_limit_gb"],
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
