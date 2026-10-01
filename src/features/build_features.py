"""Build the versioned LightGBM feature table without future leakage.

Purpose
-------
Turn a bounded fused candidate pool into numeric ranking evidence.  Snapshot
events provide global item priors, query prefixes provide session and
session-candidate features, and point-in-time CoVis matrices provide local
affinity.  Future labels are joined only after every feature is computed.

Workflow
--------
1. Validate candidate/query/snapshot/matrix schemas and the time boundary.
2. Select a deterministic session sample for ranker training (full validation
   is kept by default).
3. Aggregate snapshot-safe item and recent-popularity lookups.
4. Process one session hash bucket at a time, deriving query and interaction
   features and joining the 49-feature registry.
5. Join labels, write compact Parquet shards, and save a reproducible report.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.schemas import (
    COVIS_MATRIX_SCHEMA,
    EVENT_SCHEMA,
    FUSED_CANDIDATE_SCHEMA,
    LABEL_SCHEMA,
    QUERY_SCHEMA,
    RANKER_FEATURE_SCHEMA,
    RANKER_FEATURE_SCHEMA_VERSION,
)
from features.registry import FEATURE_GROUPS, FEATURE_NAMES, registry_payload
from utils.config import resolve_project_config


DAY_MS = 86_400_000
TARGET_TYPES = (1, 2, 3)
_CANDIDATE_RE = re.compile(r"session-bucket-(\d+)-target-([123])\.parquet$")

ITEM_STATS_SCHEMA = pa.schema([
    pa.field("aid", pa.int64(), nullable=False),
    pa.field("item_total_count", pa.int32(), nullable=False),
    pa.field("item_click_count", pa.int32(), nullable=False),
    pa.field("item_cart_count", pa.int32(), nullable=False),
    pa.field("item_order_count", pa.int32(), nullable=False),
    pa.field("item_recent_1d_count", pa.int32(), nullable=False),
    pa.field("item_recent_7d_count", pa.int32(), nullable=False),
])

SELECTED_SESSION_SCHEMA = pa.schema([
    pa.field("session", pa.int64(), nullable=False),
])


def _quote(value: str | Path) -> str:
    return str(value).replace("'", "''")


def _path(path: Path) -> str:
    return path.resolve().as_posix()


def _sql_files(paths: Sequence[Path]) -> str:
    if not paths:
        raise ValueError("At least one Parquet file is required")
    return "[" + ",".join(f"'{_quote(_path(path))}'" for path in paths) + "]"


def _directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _check_schema(path: Path, expected: pa.Schema, name: str) -> None:
    actual = pq.read_schema(path)
    if actual.names != expected.names or actual.types != expected.types:
        raise AssertionError(f"Unexpected {name} schema in {path}: {actual}")


def _files(path: str | Path, pattern: str, name: str) -> list[Path]:
    source = Path(path).resolve()
    if source.is_file():
        result = [source]
    elif source.is_dir():
        result = sorted(source.rglob(pattern))
    else:
        raise FileNotFoundError(source)
    if not result:
        raise FileNotFoundError(f"No {pattern} files below {source} for {name}")
    return result


def _candidate_file_map(path: str | Path) -> tuple[dict[tuple[int, int], Path], int]:
    files = _files(path, "session-bucket-*-target-*.parquet", "fused candidates")
    result: dict[tuple[int, int], Path] = {}
    for file in files:
        match = _CANDIDATE_RE.fullmatch(file.name)
        if match is None:
            continue
        key = (int(match.group(1)), int(match.group(2)))
        if key in result:
            raise ValueError(f"Duplicate candidate shard for bucket and target {key}")
        _check_schema(file, FUSED_CANDIDATE_SCHEMA, "fused candidate")
        result[key] = file
    if not result:
        raise ValueError("No canonical fused candidate shards were found")
    buckets = sorted({bucket for bucket, _ in result})
    if buckets != list(range(max(buckets) + 1)):
        raise ValueError(f"Candidate session buckets are not contiguous: {buckets}")
    missing = [
        (bucket, target)
        for bucket in buckets
        for target in TARGET_TYPES
        if (bucket, target) not in result
    ]
    if missing:
        raise ValueError(f"Missing candidate shards: {missing[:10]}")
    return result, len(buckets)


def _matrix_files(path: str | Path, name: str) -> list[Path]:
    result = _files(path, "bucket-*.parquet", name)
    for file in result:
        _check_schema(file, COVIS_MATRIX_SCHEMA, f"{name} matrix")
    return result


def _configure_connection(
    connection: duckdb.DuckDBPyConnection,
    *,
    workers: int,
    memory_limit_gb: int,
    temp_dir: Path,
) -> None:
    temp_dir.mkdir(parents=True, exist_ok=True)
    connection.execute(f"SET threads={int(workers)}")
    connection.execute(f"SET memory_limit='{int(memory_limit_gb)}GB'")
    connection.execute(f"SET temp_directory='{_quote(_path(temp_dir))}'")
    connection.execute("SET preserve_insertion_order=false")


def _point_in_time_cutoff(
    connection: duckdb.DuckDBPyConnection,
    queries: Path,
    snapshot_files: Sequence[Path],
    explicit_cutoff_ts: int | None,
) -> tuple[int, int, int]:
    query_min_ts = int(
        connection.execute(
            f"""
            SELECT min(ts)::BIGINT
            FROM (
                SELECT unnest(timestamps)::BIGINT AS ts
                FROM read_parquet('{_quote(_path(queries))}')
            )
            """
        ).fetchone()[0]
    )
    snapshot_min_ts, snapshot_max_ts = connection.execute(
        f"""
        SELECT min(ts)::BIGINT, max(ts)::BIGINT
        FROM read_parquet({_sql_files(snapshot_files)}, union_by_name=true)
        """
    ).fetchone()
    cutoff_ts = int(explicit_cutoff_ts or query_min_ts)
    if int(snapshot_max_ts) >= cutoff_ts:
        raise AssertionError(
            f"Snapshot leakage: max ts {snapshot_max_ts} is not before cutoff {cutoff_ts}"
        )
    if cutoff_ts > query_min_ts:
        raise AssertionError(
            f"Feature cutoff {cutoff_ts} is after the earliest query event {query_min_ts}"
        )
    return cutoff_ts, int(snapshot_min_ts), int(snapshot_max_ts)


def _create_selected_sessions(
    connection: duckdb.DuckDBPyConnection,
    queries: Path,
    output_path: Path,
    *,
    limit: int | None,
    seed: int,
    compression: str,
) -> dict[str, int | bool | None]:
    total_sessions = int(
        connection.execute(
            f"SELECT count(*) FROM read_parquet('{_quote(_path(queries))}')"
        ).fetchone()[0]
    )
    effective_limit = min(total_sessions, int(limit)) if limit else total_sessions
    connection.execute("DROP TABLE IF EXISTS selected_sessions")
    connection.execute(
        f"""
        CREATE TEMP TABLE selected_sessions AS
        SELECT session::BIGINT AS session
        FROM read_parquet('{_quote(_path(queries))}')
        ORDER BY hash(session, {int(seed)}), session
        LIMIT {effective_limit}
        """
    )
    connection.execute(
        f"""
        COPY (
            SELECT session::BIGINT AS session
            FROM selected_sessions ORDER BY session
        ) TO '{_quote(_path(output_path))}' (
            FORMAT PARQUET,
            COMPRESSION {compression.upper()}
        )
        """
    )
    _check_schema(output_path, SELECTED_SESSION_SCHEMA, "selected session")
    return {
        "available_sessions": total_sessions,
        "selected_sessions": effective_limit,
        "limit": limit,
        "sampled": effective_limit < total_sessions,
        "seed": seed,
    }


def _build_item_stats(
    connection: duckdb.DuckDBPyConnection,
    snapshot_files: Sequence[Path],
    output_path: Path,
    *,
    cutoff_ts: int,
    compression: str,
    row_group_size: int,
) -> dict[str, int | float]:
    started = time.perf_counter()
    connection.execute(
        f"""
        COPY (
            SELECT
                aid::BIGINT AS aid,
                count(*)::INTEGER AS item_total_count,
                count(*) FILTER (WHERE event_type = 1)::INTEGER AS item_click_count,
                count(*) FILTER (WHERE event_type = 2)::INTEGER AS item_cart_count,
                count(*) FILTER (WHERE event_type = 3)::INTEGER AS item_order_count,
                count(*) FILTER (
                    WHERE ts >= {int(cutoff_ts - DAY_MS)} AND ts < {int(cutoff_ts)}
                )::INTEGER AS item_recent_1d_count,
                count(*) FILTER (
                    WHERE ts >= {int(cutoff_ts - 7 * DAY_MS)} AND ts < {int(cutoff_ts)}
                )::INTEGER AS item_recent_7d_count
            FROM read_parquet({_sql_files(snapshot_files)}, union_by_name=true)
            GROUP BY aid
            ORDER BY aid
        ) TO '{_quote(_path(output_path))}' (
            FORMAT PARQUET,
            COMPRESSION {compression.upper()},
            ROW_GROUP_SIZE {int(row_group_size)}
        )
        """
    )
    _check_schema(output_path, ITEM_STATS_SCHEMA, "item feature lookup")
    return {
        "items": pq.ParquetFile(output_path).metadata.num_rows,
        "disk_bytes": output_path.stat().st_size,
        "runtime_seconds": round(time.perf_counter() - started, 6),
    }


def _create_bucket_inputs(
    connection: duckdb.DuckDBPyConnection,
    queries: Path,
    candidate_files: Sequence[Path],
    *,
    bucket: int,
    session_buckets: int,
) -> None:
    for table in (
        "bucket_queries",
        "bucket_query_events",
        "bucket_session_stats",
        "bucket_history_stats",
        "bucket_recent_seeds",
        "bucket_candidates",
        "bucket_candidate_keys",
    ):
        connection.execute(f"DROP TABLE IF EXISTS {table}")
    connection.execute(
        f"""
        CREATE TEMP TABLE bucket_queries AS
        SELECT q.*
        FROM read_parquet('{_quote(_path(queries))}') q
        INNER JOIN selected_sessions selected USING (session)
        WHERE q.session % {int(session_buckets)} = {int(bucket)}
        """
    )
    connection.execute(
        """
        CREATE TEMP TABLE bucket_query_events AS
        SELECT
            session::BIGINT AS session,
            unnest(aids)::BIGINT AS aid,
            unnest(timestamps)::BIGINT AS ts,
            unnest(event_types)::TINYINT AS event_type,
            generate_subscripts(aids, 1)::SMALLINT AS position
        FROM bucket_queries
        """
    )
    connection.execute(
        f"""
        CREATE TEMP TABLE bucket_candidates AS
        SELECT c.*
        FROM read_parquet({_sql_files(candidate_files)}, union_by_name=true) c
        INNER JOIN selected_sessions selected USING (session)
        """
    )
    connection.execute(
        """
        CREATE TEMP TABLE bucket_session_stats AS
        SELECT
            session,
            count(*)::SMALLINT AS session_length,
            count(*) FILTER (WHERE event_type = 1)::SMALLINT AS session_click_count,
            count(*) FILTER (WHERE event_type = 2)::SMALLINT AS session_cart_count,
            count(*) FILTER (WHERE event_type = 3)::SMALLINT AS session_order_count,
            (max(ts) - min(ts))::BIGINT AS session_duration_ms,
            arg_max(event_type, position)::TINYINT AS session_last_event_type,
            (floor(arg_max(ts, position) / 3600000) % 24)::TINYINT AS session_hour_utc,
            ((floor(arg_max(ts, position) / 86400000) + 3) % 7)::TINYINT
                AS session_weekday_utc
        FROM bucket_query_events
        GROUP BY session
        """
    )
    connection.execute(
        """
        CREATE TEMP TABLE bucket_history_stats AS
        SELECT
            session,
            aid,
            count(*)::SMALLINT AS candidate_occurrence_count,
            max(position)::SMALLINT AS candidate_last_position,
            arg_max(event_type, position)::TINYINT AS candidate_last_event_type
        FROM bucket_query_events
        GROUP BY session, aid
        """
    )
    connection.execute(
        """
        CREATE TEMP TABLE bucket_recent_seeds AS
        SELECT session, aid AS seed_aid, recent_rank
        FROM (
            SELECT
                session,
                aid,
                row_number() OVER (
                    PARTITION BY session ORDER BY position DESC
                )::INTEGER AS recent_rank
            FROM bucket_query_events
        )
        WHERE recent_rank <= 5
        """
    )
    connection.execute(
        """
        CREATE TEMP TABLE bucket_candidate_keys AS
        SELECT DISTINCT session, aid FROM bucket_candidates
        """
    )


def _create_interaction(
    connection: duckdb.DuckDBPyConnection,
    matrix_files: Sequence[Path],
    source: str,
) -> None:
    table = f"interaction_{source}"
    connection.execute(f"DROP TABLE IF EXISTS {table}")
    connection.execute(
        f"""
        CREATE TEMP TABLE {table} AS
        WITH candidate_seed AS (
            SELECT
                candidates.session,
                candidates.aid,
                seeds.recent_rank,
                coalesce(matrix.source_score, 0.0)::FLOAT AS source_score
            FROM bucket_candidate_keys candidates
            INNER JOIN bucket_recent_seeds seeds USING (session)
            LEFT JOIN read_parquet({_sql_files(matrix_files)}, union_by_name=true) matrix
              ON matrix.aid = seeds.seed_aid
             AND matrix.neighbor_aid = candidates.aid
        )
        SELECT
            session,
            aid,
            max(CASE WHEN recent_rank = 1 THEN source_score ELSE 0.0 END)::FLOAT
                AS last_item_score,
            max(source_score)::FLOAT AS recent5_max,
            avg(source_score)::FLOAT AS recent5_mean
        FROM candidate_seed
        GROUP BY session, aid
        """
    )


def _create_bucket_features(
    connection: duckdb.DuckDBPyConnection,
    labels: Path | None,
    item_stats: Path,
) -> None:
    connection.execute("DROP TABLE IF EXISTS bucket_features")
    if labels is None:
        label_join = ""
        label_expression = "0::TINYINT"
    else:
        label_join = f"""
        LEFT JOIN read_parquet('{_quote(_path(labels))}') labels
          ON labels.session = candidates.session
         AND labels.target_type = candidates.target_type
         AND labels.aid = candidates.aid
        """
        label_expression = "(labels.aid IS NOT NULL)::TINYINT"
    connection.execute(
        f"""
        CREATE TEMP TABLE bucket_features AS
        SELECT
            candidates.session::BIGINT AS session,
            candidates.target_type::TINYINT AS target_type,
            candidates.aid::BIGINT AS aid,
            candidates.candidate_rank::SMALLINT AS candidate_rank,
            candidates.selected_source::TINYINT AS selected_source,
            {label_expression} AS label,
            candidates.target_type::TINYINT AS target_type_id,
            sessions.session_length::SMALLINT AS session_length,
            sessions.session_click_count::SMALLINT AS session_click_count,
            sessions.session_cart_count::SMALLINT AS session_cart_count,
            sessions.session_order_count::SMALLINT AS session_order_count,
            sessions.session_duration_ms::BIGINT AS session_duration_ms,
            sessions.session_last_event_type::TINYINT AS session_last_event_type,
            sessions.session_hour_utc::TINYINT AS session_hour_utc,
            sessions.session_weekday_utc::TINYINT AS session_weekday_utc,
            coalesce(items.item_total_count, 0)::INTEGER AS item_total_count,
            coalesce(items.item_click_count, 0)::INTEGER AS item_click_count,
            coalesce(items.item_cart_count, 0)::INTEGER AS item_cart_count,
            coalesce(items.item_order_count, 0)::INTEGER AS item_order_count,
            candidates.from_popular::TINYINT AS from_popular,
            candidates.popular_rank::SMALLINT AS popular_rank,
            candidates.popular_score::FLOAT AS popular_score,
            candidates.from_revisit::TINYINT AS from_revisit,
            candidates.revisit_rank::SMALLINT AS revisit_rank,
            candidates.revisit_score::FLOAT AS revisit_score,
            candidates.from_type_covis::TINYINT AS from_type_covis,
            candidates.type_covis_rank::SMALLINT AS type_covis_rank,
            candidates.type_covis_score::FLOAT AS type_covis_score,
            candidates.from_buy2buy::TINYINT AS from_buy2buy,
            candidates.buy2buy_rank::SMALLINT AS buy2buy_rank,
            candidates.buy2buy_score::FLOAT AS buy2buy_score,
            candidates.from_time_covis::TINYINT AS from_time_covis,
            candidates.time_covis_rank::SMALLINT AS time_covis_rank,
            candidates.time_covis_score::FLOAT AS time_covis_score,
            candidates.from_dssm::TINYINT AS from_dssm,
            candidates.dssm_rank::SMALLINT AS dssm_rank,
            candidates.dssm_score::FLOAT AS dssm_score,
            candidates.source_count::TINYINT AS source_count,
            candidates.best_source_rank::SMALLINT AS best_source_rank,
            (history.aid IS NOT NULL)::TINYINT AS candidate_seen,
            coalesce(history.candidate_occurrence_count, 0)::SMALLINT
                AS candidate_occurrence_count,
            coalesce(history.candidate_last_position, 0)::SMALLINT
                AS candidate_last_position,
            CASE
                WHEN history.aid IS NULL THEN -1
                ELSE sessions.session_length - history.candidate_last_position
            END::SMALLINT AS candidate_distance_to_end,
            coalesce(history.candidate_last_event_type, 0)::TINYINT
                AS candidate_last_event_type,
            coalesce(type_interaction.last_item_score, 0.0)::FLOAT
                AS last_item_type_covis_score,
            coalesce(buy_interaction.last_item_score, 0.0)::FLOAT
                AS last_item_buy2buy_score,
            coalesce(time_interaction.last_item_score, 0.0)::FLOAT
                AS last_item_time_covis_score,
            coalesce(type_interaction.recent5_max, 0.0)::FLOAT
                AS recent5_type_covis_max,
            coalesce(type_interaction.recent5_mean, 0.0)::FLOAT
                AS recent5_type_covis_mean,
            coalesce(buy_interaction.recent5_max, 0.0)::FLOAT
                AS recent5_buy2buy_max,
            coalesce(buy_interaction.recent5_mean, 0.0)::FLOAT
                AS recent5_buy2buy_mean,
            coalesce(time_interaction.recent5_max, 0.0)::FLOAT
                AS recent5_time_covis_max,
            coalesce(time_interaction.recent5_mean, 0.0)::FLOAT
                AS recent5_time_covis_mean,
            coalesce(items.item_recent_1d_count, 0)::INTEGER AS item_recent_1d_count,
            coalesce(items.item_recent_7d_count, 0)::INTEGER AS item_recent_7d_count
        FROM bucket_candidates candidates
        INNER JOIN bucket_session_stats sessions USING (session)
        LEFT JOIN read_parquet('{_quote(_path(item_stats))}') items USING (aid)
        LEFT JOIN bucket_history_stats history USING (session, aid)
        LEFT JOIN interaction_type_covis type_interaction USING (session, aid)
        LEFT JOIN interaction_buy2buy buy_interaction USING (session, aid)
        LEFT JOIN interaction_time_covis time_interaction USING (session, aid)
        {label_join}
        """
    )


def _write_bucket(
    connection: duckdb.DuckDBPyConnection,
    output_dir: Path,
    *,
    bucket: int,
    compression: str,
    row_group_size: int,
) -> tuple[list[Path], dict[str, int]]:
    output_files: list[Path] = []
    rows = 0
    for target_type in TARGET_TYPES:
        output = output_dir / f"session-bucket-{bucket:05d}-target-{target_type}.parquet"
        connection.execute(
            f"""
            COPY (
                SELECT * FROM bucket_features
                WHERE target_type = {target_type}
                ORDER BY session, candidate_rank
            ) TO '{_quote(_path(output))}' (
                FORMAT PARQUET,
                COMPRESSION {compression.upper()},
                ROW_GROUP_SIZE {int(row_group_size)}
            )
            """
        )
        _check_schema(output, RANKER_FEATURE_SCHEMA, "ranker feature")
        output_files.append(output)
        rows += pq.ParquetFile(output).metadata.num_rows
    positive_rows = int(
        connection.execute("SELECT sum(label)::BIGINT FROM bucket_features").fetchone()[0] or 0
    )
    groups_with_positive = int(
        connection.execute(
            """
            SELECT count(*) FROM (
                SELECT session, target_type
                FROM bucket_features
                GROUP BY session, target_type
                HAVING max(label) = 1
            )
            """
        ).fetchone()[0]
    )
    return output_files, {
        "rows": rows,
        "positive_rows": positive_rows,
        "groups_with_positive": groups_with_positive,
    }


def build_features(
    candidates_dir: str | Path,
    queries_path: str | Path,
    labels_path: str | Path | None,
    snapshot_dir: str | Path,
    type_covis_matrix: str | Path,
    buy2buy_matrix: str | Path,
    time_covis_matrix: str | Path,
    output_dir: str | Path,
    config: Mapping[str, Any],
    *,
    dataset_name: str,
    max_sessions: int | None = None,
    cutoff_ts: int | None = None,
) -> dict[str, Any]:
    """Build one train/validation/test feature dataset with a shared schema."""

    if dataset_name not in {"ranker", "valid", "test"}:
        raise ValueError("dataset_name must be ranker, valid or test")
    queries = Path(queries_path).resolve()
    labels = Path(labels_path).resolve() if labels_path else None
    destination = Path(output_dir).resolve()
    if not queries.is_file():
        raise FileNotFoundError(queries)
    if labels is not None and not labels.is_file():
        raise FileNotFoundError(labels)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite feature output: {destination}")
    _check_schema(queries, QUERY_SCHEMA, "query")
    if labels is not None:
        _check_schema(labels, LABEL_SCHEMA, "label")

    candidate_map, session_buckets = _candidate_file_map(candidates_dir)
    snapshot_files = _files(snapshot_dir, "part-*.parquet", "snapshot events")
    for file in snapshot_files:
        _check_schema(file, EVENT_SCHEMA, "snapshot event")
    matrices = {
        "type_covis": _matrix_files(type_covis_matrix, "Type-CoVis"),
        "buy2buy": _matrix_files(buy2buy_matrix, "Buy2Buy"),
        "time_covis": _matrix_files(time_covis_matrix, "Time-CoVis"),
    }

    runtime = config["runtime"]
    compression = str(runtime["parquet_compression"])
    row_group_size = int(runtime["parquet_row_group_size"])
    workers = int(runtime["workers"])
    memory_limit_gb = int(runtime["duckdb_memory_limit_gb"])
    seed = int(config["seed"])
    if max_sessions is not None:
        session_limit = int(max_sessions)
    elif dataset_name == "ranker":
        configured_limit = config["ranker"].get("train_session_limit")
        session_limit = int(configured_limit) if configured_limit else None
    else:
        session_limit = None
    if session_limit is not None and session_limit <= 0:
        raise ValueError("Session limit must be positive")

    enabled_groups = tuple(config["feature_groups"])
    registry = registry_payload(enabled_groups)
    if set(enabled_groups) != set(FEATURE_GROUPS):
        raise ValueError(
            "The feature builder materializes the canonical full schema; "
            "enable all six feature groups. "
            "Feature ablation is performed by the ranker without rebuilding Parquet."
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    token = f"{os.getpid()}-{uuid.uuid4().hex[:8]}"
    staging = destination.parent / f".{destination.name}.building-{token}"
    parts_dir = staging / "parts"
    lookups_dir = staging / "lookups"
    work_dir = staging / "work"
    parts_dir.mkdir(parents=True)
    lookups_dir.mkdir()
    work_dir.mkdir()
    connection = duckdb.connect()
    _configure_connection(
        connection,
        workers=workers,
        memory_limit_gb=memory_limit_gb,
        temp_dir=work_dir / "duckdb_tmp",
    )
    started = time.perf_counter()
    try:
        point_in_time_cutoff, snapshot_min_ts, snapshot_max_ts = _point_in_time_cutoff(
            connection, queries, snapshot_files, cutoff_ts
        )
        selection = _create_selected_sessions(
            connection,
            queries,
            staging / "selected_sessions.parquet",
            limit=session_limit,
            seed=seed,
            compression=compression,
        )
        item_stats_path = lookups_dir / "item_statistics.parquet"
        item_report = _build_item_stats(
            connection,
            snapshot_files,
            item_stats_path,
            cutoff_ts=point_in_time_cutoff,
            compression=compression,
            row_group_size=row_group_size,
        )

        output_files: list[Path] = []
        total_rows = 0
        positive_rows = 0
        groups_with_positive = 0
        processed_sessions = 0
        bucket_reports: list[dict[str, int | float]] = []
        feature_started = time.perf_counter()
        for bucket in range(session_buckets):
            bucket_started = time.perf_counter()
            input_files = [candidate_map[(bucket, target)] for target in TARGET_TYPES]
            _create_bucket_inputs(
                connection,
                queries,
                input_files,
                bucket=bucket,
                session_buckets=session_buckets,
            )
            bucket_sessions = int(
                connection.execute("SELECT count(*) FROM bucket_queries").fetchone()[0]
            )
            if bucket_sessions == 0:
                continue
            processed_sessions += bucket_sessions
            for source, files in matrices.items():
                _create_interaction(connection, files, source)
            _create_bucket_features(connection, labels, item_stats_path)
            files, stats = _write_bucket(
                connection,
                parts_dir,
                bucket=bucket,
                compression=compression,
                row_group_size=row_group_size,
            )
            output_files.extend(files)
            total_rows += stats["rows"]
            positive_rows += stats["positive_rows"]
            groups_with_positive += stats["groups_with_positive"]
            bucket_reports.append({
                "bucket": bucket,
                "sessions": bucket_sessions,
                "rows": stats["rows"],
                "positive_rows": stats["positive_rows"],
                "runtime_seconds": round(time.perf_counter() - bucket_started, 6),
            })
            for table in (
                "bucket_features",
                "interaction_type_covis",
                "interaction_buy2buy",
                "interaction_time_covis",
                "bucket_candidate_keys",
                "bucket_candidates",
                "bucket_recent_seeds",
                "bucket_history_stats",
                "bucket_session_stats",
                "bucket_query_events",
                "bucket_queries",
            ):
                connection.execute(f"DROP TABLE IF EXISTS {table}")
            print(
                f"[features] dataset={dataset_name} bucket={bucket + 1}/{session_buckets} "
                f"sessions={processed_sessions:,} rows={total_rows:,} "
                f"elapsed={time.perf_counter() - feature_started:.1f}s",
                flush=True,
            )

        expected_sessions = int(selection["selected_sessions"])
        expected_groups = expected_sessions * len(TARGET_TYPES)
        candidate_k = int(config["candidate_k"])
        expected_rows = expected_groups * candidate_k
        if processed_sessions != expected_sessions:
            raise AssertionError(
                f"Processed {processed_sessions} sessions; expected {expected_sessions}"
            )
        if total_rows != expected_rows:
            raise AssertionError(f"Wrote {total_rows} feature rows; expected {expected_rows}")
        if len(output_files) != session_buckets * len(TARGET_TYPES):
            raise AssertionError("Feature output does not contain every bucket/target shard")

        registry_path = staging / "feature_registry.json"
        registry_path.write_text(
            json.dumps(registry, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        report = {
            "dataset": dataset_name,
            "output": str(destination),
            "schema_version": RANKER_FEATURE_SCHEMA_VERSION,
            "inputs": {
                "candidates": str(Path(candidates_dir).resolve()),
                "queries": str(queries),
                "labels": str(labels) if labels else None,
                "snapshot": str(Path(snapshot_dir).resolve()),
                "type_covis_matrix": str(Path(type_covis_matrix).resolve()),
                "buy2buy_matrix": str(Path(buy2buy_matrix).resolve()),
                "time_covis_matrix": str(Path(time_covis_matrix).resolve()),
            },
            "point_in_time": {
                "cutoff_ts": point_in_time_cutoff,
                "snapshot_min_ts": snapshot_min_ts,
                "snapshot_max_ts": snapshot_max_ts,
                "snapshot_before_cutoff": snapshot_max_ts < point_in_time_cutoff,
            },
            "session_selection": selection,
            "features": {
                "count": len(FEATURE_NAMES),
                "enabled_groups": list(enabled_groups),
                "group_counts": {
                    name: len(values) for name, values in FEATURE_GROUPS.items()
                },
                "registry": "feature_registry.json",
                "labels_joined_after_feature_computation": True,
                "recent5_semantics": (
                    "last five events including repeats; missing CoVis edges score zero"
                ),
            },
            "item_lookup": item_report,
            "output_data": {
                "files": len(output_files),
                "rows": total_rows,
                "groups": expected_groups,
                "candidate_k": candidate_k,
                "positive_rows": positive_rows,
                "groups_with_positive": groups_with_positive,
                "groups_without_positive": expected_groups - groups_with_positive,
                "disk_bytes": _directory_size(parts_dir),
            },
            "assertions": {
                "passed": True,
                "snapshot_before_queries": snapshot_max_ts < point_in_time_cutoff,
                "schema_matches_registry": tuple(RANKER_FEATURE_SCHEMA.names[6:])
                == FEATURE_NAMES,
                "rows_match_candidate_budget": total_rows == expected_rows,
                "same_builder_for_train_and_inference": True,
                "future_labels_not_used_as_features": True,
            },
            "bucket_reports": bucket_reports,
            "timing": {
                "feature_seconds": round(time.perf_counter() - feature_started, 6),
                "total_seconds": round(time.perf_counter() - started, 6),
            },
        }
        connection.close()
        shutil.rmtree(work_dir, ignore_errors=True)
        (staging / "report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        staging.replace(destination)
        return report
    except Exception:
        connection.close()
        shutil.rmtree(staging, ignore_errors=True)
        raise


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build point-in-time LightGBM features.")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--snapshot-dir", type=Path, required=True)
    parser.add_argument("--type-covis-matrix", type=Path, required=True)
    parser.add_argument("--buy2buy-matrix", type=Path, required=True)
    parser.add_argument("--time-covis-matrix", type=Path, required=True)
    parser.add_argument("--dataset-name", choices=("ranker", "valid", "test"), required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-sessions", type=int)
    parser.add_argument("--cutoff-ts", type=int)
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
        output_dir = Path(experiment_dir) / "features" / args.dataset_name
    else:
        raise ValueError("Provide --output-dir or run through the experiment runner")
    report = build_features(
        args.candidates,
        args.queries,
        args.labels,
        args.snapshot_dir,
        args.type_covis_matrix,
        args.buy2buy_matrix,
        args.time_covis_matrix,
        output_dir,
        resolve_project_config(config_path),
        dataset_name=args.dataset_name,
        max_sessions=args.max_sessions,
        cutoff_ts=args.cutoff_ts,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
