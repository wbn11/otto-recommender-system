"""Build a disk-backed, type-weighted item co-visitation matrix."""

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
from typing import Any, Mapping

import duckdb
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.schemas import COVIS_MATRIX_SCHEMA, EVENT_SCHEMA, EVENT_TYPE_TO_ID
from utils.config import resolve_project_config


def _quote(value: str | Path) -> str:
    return str(value).replace("'", "''")


def _glob(path: Path) -> str:
    return path.resolve().as_posix()


def _directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _parquet_rows(paths: list[Path]) -> int:
    return sum(pq.ParquetFile(path).metadata.num_rows for path in paths)


def _check_event_schema(path: Path) -> None:
    actual = pq.read_schema(path)
    if actual.names != EVENT_SCHEMA.names or actual.types != EVENT_SCHEMA.types:
        raise AssertionError(f"Unexpected event schema in {path}: {actual}")


def _check_matrix_schema(path: Path) -> None:
    actual = pq.read_schema(path)
    if actual.names != COVIS_MATRIX_SCHEMA.names or actual.types != COVIS_MATRIX_SCHEMA.types:
        raise AssertionError(f"Unexpected co-visitation matrix schema in {path}: {actual}")


def _weight_case(column: str, event_type_weights: Mapping[int, float]) -> str:
    clauses = " ".join(
        f"WHEN {int(event_type)} THEN {float(weight)}"
        for event_type, weight in sorted(event_type_weights.items())
    )
    return f"CASE {column} {clauses} ELSE NULL END"


def _write_partial_pairs(
    connection: duckdb.DuckDBPyConnection,
    event_file: Path,
    output_dir: Path,
    *,
    recent_events_per_session: int,
    pair_buckets: int,
    event_type_weights: Mapping[int, float],
    allowed_event_types: tuple[int, ...],
    pair_score_mode: str,
    max_time_difference_ms: int | None,
    time_decay_ms: int | None,
    compression: str,
) -> None:
    candidate_weight = _weight_case("event_type", event_type_weights)
    event_type_filter = ", ".join(str(value) for value in allowed_event_types)
    if pair_score_mode == "neighbor_event_weight":
        pair_query = f"""
            WITH ordered AS (
                SELECT
                    session::BIGINT AS session,
                    aid::BIGINT AS aid,
                    event_type::TINYINT AS event_type,
                    row_number() OVER (
                        PARTITION BY session
                        ORDER BY ts DESC, file_row_number DESC
                    )::INTEGER AS recent_rank
                FROM read_parquet(
                    '{_quote(_glob(event_file))}',
                    file_row_number=true
                )
                WHERE event_type IN ({event_type_filter})
            ),
            recent AS (
                SELECT session, aid, event_type, recent_rank
                FROM ordered
                WHERE recent_rank <= {int(recent_events_per_session)}
            ),
            session_items AS (
                SELECT
                    session,
                    aid,
                    max({candidate_weight})::DOUBLE AS item_weight
                FROM recent
                GROUP BY session, aid
            )
            SELECT
                (left_item.aid % {int(pair_buckets)})::INTEGER AS bucket,
                left_item.aid::BIGINT AS aid,
                right_item.aid::BIGINT AS neighbor_aid,
                sum(right_item.item_weight)::DOUBLE AS pair_score
            FROM session_items left_item
            INNER JOIN session_items right_item
              ON left_item.session = right_item.session
             AND left_item.aid != right_item.aid
            GROUP BY bucket, left_item.aid, right_item.aid
        """
    elif pair_score_mode == "time_decay":
        if max_time_difference_ms is None or time_decay_ms is None:
            raise ValueError("Time-decay pairs require time-window and decay parameters")
        pair_query = f"""
            WITH ordered AS (
                SELECT
                    session::BIGINT AS session,
                    aid::BIGINT AS aid,
                    ts::BIGINT AS ts,
                    row_number() OVER (
                        PARTITION BY session
                        ORDER BY ts DESC, file_row_number DESC
                    )::INTEGER AS recent_rank
                FROM read_parquet(
                    '{_quote(_glob(event_file))}',
                    file_row_number=true
                )
                WHERE event_type IN ({event_type_filter})
            ),
            recent AS (
                SELECT session, aid, ts
                FROM ordered
                WHERE recent_rank <= {int(recent_events_per_session)}
            ),
            session_pairs AS (
                SELECT
                    left_event.session,
                    left_event.aid,
                    right_event.aid AS neighbor_aid,
                    max(
                        exp(
                            -abs(left_event.ts - right_event.ts)::DOUBLE
                            / {int(time_decay_ms)}
                        )
                    )::DOUBLE AS session_pair_score
                FROM recent left_event
                INNER JOIN recent right_event
                  ON left_event.session = right_event.session
                 AND left_event.aid != right_event.aid
                 AND abs(left_event.ts - right_event.ts)
                     <= {int(max_time_difference_ms)}
                GROUP BY left_event.session, left_event.aid, right_event.aid
            )
            SELECT
                (aid % {int(pair_buckets)})::INTEGER AS bucket,
                aid::BIGINT AS aid,
                neighbor_aid::BIGINT AS neighbor_aid,
                sum(session_pair_score)::DOUBLE AS pair_score
            FROM session_pairs
            GROUP BY bucket, aid, neighbor_aid
        """
    else:
        raise ValueError(f"Unsupported pair_score_mode: {pair_score_mode}")
    connection.execute(
        f"""
        COPY (
            {pair_query}
        ) TO '{_quote(_glob(output_dir))}' (
            FORMAT PARQUET,
            PARTITION_BY (bucket),
            COMPRESSION {compression.upper()}
        )
        """
    )


def _finalize_bucket(
    connection: duckdb.DuckDBPyConnection,
    partial_glob: str,
    output_path: Path,
    *,
    topk_neighbors: int,
    compression: str,
    row_group_size: int,
) -> dict[str, int]:
    connection.execute(
        f"""
        COPY (
            WITH aggregated AS (
                SELECT
                    aid::BIGINT AS aid,
                    neighbor_aid::BIGINT AS neighbor_aid,
                    sum(pair_score)::DOUBLE AS source_score
                FROM read_parquet('{_quote(partial_glob)}', union_by_name=true)
                GROUP BY aid, neighbor_aid
            ),
            ranked AS (
                SELECT
                    aid,
                    neighbor_aid,
                    row_number() OVER (
                        PARTITION BY aid
                        ORDER BY source_score DESC, neighbor_aid ASC
                    )::INTEGER AS source_rank,
                    source_score::FLOAT AS source_score
                FROM aggregated
            )
            SELECT
                aid::BIGINT AS aid,
                neighbor_aid::BIGINT AS neighbor_aid,
                source_rank::INTEGER AS source_rank,
                source_score::FLOAT AS source_score
            FROM ranked
            WHERE source_rank <= {int(topk_neighbors)}
            ORDER BY aid, source_rank
        ) TO '{_quote(_glob(output_path))}' (
            FORMAT PARQUET,
            COMPRESSION {compression.upper()},
            ROW_GROUP_SIZE {int(row_group_size)}
        )
        """
    )
    _check_matrix_schema(output_path)
    rows, source_items = connection.execute(
        f"""SELECT count(*)::BIGINT, count(DISTINCT aid)::BIGINT
             FROM read_parquet('{_quote(_glob(output_path))}')"""
    ).fetchone()
    return {"rows": int(rows), "source_items": int(source_items)}


def build_type_covis(
    snapshot_dir: str | Path,
    output_dir: str | Path,
    *,
    dataset_name: str,
    recent_events_per_session: int = 30,
    topk_neighbors: int = 200,
    pair_buckets: int = 256,
    event_type_weights: Mapping[int, float] | None = None,
    compression: str = "zstd",
    row_group_size: int = 250_000,
    workers: int = 8,
    memory_limit_gb: int = 8,
    source_name: str = "type_covis",
    allowed_event_types: tuple[int, ...] = (1, 2, 3),
    pair_score_mode: str = "neighbor_event_weight",
    max_time_difference_ms: int | None = None,
    time_decay_ms: int | None = None,
    pair_semantics: str = (
        "directed unique item pairs per session; neighbor uses strongest "
        "event-type weight within the recent window"
    ),
) -> dict[str, Any]:
    """Build Top-K directed neighbors while bounding memory by source-aid bucket."""

    snapshot_source = Path(snapshot_dir).resolve()
    target = Path(output_dir).resolve()
    event_files = sorted(snapshot_source.glob("part-*.parquet"))
    if not event_files:
        event_files = sorted((snapshot_source / "events").glob("part-*.parquet"))
    if not event_files:
        raise FileNotFoundError(f"No snapshot event shards found in {snapshot_source}")
    for event_file in event_files:
        _check_event_schema(event_file)
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite existing {source_name} matrix: {target}")
    if recent_events_per_session <= 1:
        raise ValueError("recent_events_per_session must be greater than one")
    if topk_neighbors <= 0 or pair_buckets <= 0:
        raise ValueError("topk_neighbors and pair_buckets must be positive")
    if workers <= 0 or memory_limit_gb <= 0:
        raise ValueError("workers and memory_limit_gb must be positive")
    if pair_score_mode not in {"neighbor_event_weight", "time_decay"}:
        raise ValueError("pair_score_mode must be neighbor_event_weight or time_decay")
    if pair_score_mode == "time_decay" and (
        max_time_difference_ms is None
        or time_decay_ms is None
        or max_time_difference_ms <= 0
        or time_decay_ms <= 0
    ):
        raise ValueError("Time-decay mode requires positive time-window and decay values")

    event_types = tuple(sorted(set(int(value) for value in allowed_event_types)))
    if not event_types or not set(event_types).issubset(EVENT_TYPE_TO_ID.values()):
        raise ValueError("allowed_event_types must be a non-empty subset of 1, 2 and 3")
    weights = dict(event_type_weights or {1: 1.0, 2: 3.0, 3: 6.0})
    if set(weights) != set(event_types) or any(
        not math.isfinite(value) or value <= 0 for value in weights.values()
    ):
        raise ValueError("Positive finite weights are required for every allowed event type")

    target.parent.mkdir(parents=True, exist_ok=True)
    token = f"{os.getpid()}-{uuid.uuid4().hex[:8]}"
    staging = target.parent / f".{target.name}.building-{token}"
    staging.mkdir(parents=True, exist_ok=False)
    partial_dir = staging / "partial_pairs"
    matrix_dir = staging / "matrix"
    temp_dir = staging / "duckdb_tmp"
    partial_dir.mkdir()
    matrix_dir.mkdir()
    temp_dir.mkdir()
    started = time.perf_counter()
    connection = duckdb.connect()

    try:
        connection.execute(f"SET threads={int(workers)}")
        connection.execute(f"SET memory_limit='{int(memory_limit_gb)}GB'")
        connection.execute(f"SET temp_directory='{_quote(_glob(temp_dir))}'")
        connection.execute("SET preserve_insertion_order=false")
        source_rows = _parquet_rows(event_files)
        event_type_filter = ", ".join(str(value) for value in event_types)
        source_summary = connection.execute(
            f"""SELECT min(ts), max(ts), count(DISTINCT session), count(DISTINCT aid), count(*)
                 FROM read_parquet(
                    '{_quote(_glob(event_files[0].parent / 'part-*.parquet'))}',
                    union_by_name=true
                 )
                 WHERE event_type IN ({event_type_filter})"""
        ).fetchone()
        if int(source_summary[4]) == 0:
            raise ValueError(f"Snapshot contains no eligible events for {source_name}")

        pair_started = time.perf_counter()
        for index, event_file in enumerate(event_files):
            output = partial_dir / f"source-{index:05d}"
            _write_partial_pairs(
                connection,
                event_file,
                output,
                recent_events_per_session=recent_events_per_session,
                pair_buckets=pair_buckets,
                event_type_weights=weights,
                allowed_event_types=event_types,
                pair_score_mode=pair_score_mode,
                max_time_difference_ms=max_time_difference_ms,
                time_decay_ms=time_decay_ms,
                compression=compression,
            )
            print(
                f"[{source_name}] partial={index + 1}/{len(event_files)} "
                f"source={event_file.name} elapsed={time.perf_counter() - pair_started:.1f}s",
                flush=True,
            )

        partial_files = sorted(partial_dir.rglob("*.parquet"))
        if not partial_files:
            raise ValueError("No co-visitation pairs were generated")
        partial_rows = _parquet_rows(partial_files)
        partial_disk_bytes = _directory_size(partial_dir)
        pair_runtime = round(time.perf_counter() - pair_started, 6)

        finalize_started = time.perf_counter()
        bucket_values = sorted({
            int(path.name.split("=", 1)[1])
            for source_dir in partial_dir.glob("source-*")
            for path in source_dir.glob("bucket=*")
        })
        matrix_rows = 0
        matrix_source_items = 0
        for position, bucket in enumerate(bucket_values, start=1):
            partial_glob = _glob(
                partial_dir / "source-*" / f"bucket={bucket}" / "*.parquet"
            )
            bucket_output = matrix_dir / f"bucket-{bucket:05d}.parquet"
            details = _finalize_bucket(
                connection,
                partial_glob,
                bucket_output,
                topk_neighbors=topk_neighbors,
                compression=compression,
                row_group_size=row_group_size,
            )
            matrix_rows += details["rows"]
            matrix_source_items += details["source_items"]
            if position % 16 == 0 or position == len(bucket_values):
                print(
                    f"[{source_name}] finalized_buckets={position}/{len(bucket_values)} "
                    f"matrix_rows={matrix_rows:,}",
                    flush=True,
                )

        matrix_files = sorted(matrix_dir.glob("bucket-*.parquet"))
        matrix_glob = _glob(matrix_dir / "bucket-*.parquet")
        violations = {
            "self_pairs": int(connection.execute(
                f"""SELECT count(*) FROM read_parquet('{_quote(matrix_glob)}')
                     WHERE aid = neighbor_aid"""
            ).fetchone()[0]),
            "duplicate_pairs": int(connection.execute(
                f"""SELECT count(*) FROM (
                        SELECT aid, neighbor_aid, count(*) AS n
                        FROM read_parquet('{_quote(matrix_glob)}')
                        GROUP BY aid, neighbor_aid HAVING count(*) > 1
                     )"""
            ).fetchone()[0]),
            "source_items_over_topk": int(connection.execute(
                f"""SELECT count(*) FROM (
                        SELECT aid, count(*) AS n
                        FROM read_parquet('{_quote(matrix_glob)}')
                        GROUP BY aid HAVING count(*) > {int(topk_neighbors)}
                     )"""
            ).fetchone()[0]),
            "invalid_rank": int(connection.execute(
                f"""SELECT count(*) FROM read_parquet('{_quote(matrix_glob)}')
                     WHERE source_rank < 1 OR source_rank > {int(topk_neighbors)}"""
            ).fetchone()[0]),
            "non_positive_score": int(connection.execute(
                f"""SELECT count(*) FROM read_parquet('{_quote(matrix_glob)}')
                     WHERE source_score <= 0 OR source_score IS NULL"""
            ).fetchone()[0]),
        }
        if any(violations.values()):
            raise AssertionError(f"Invalid {source_name} matrix: {violations}")

        neighbor_summary = connection.execute(
            f"""
            WITH neighbor_counts AS (
                SELECT aid, count(*)::BIGINT AS neighbor_count
                FROM read_parquet('{_quote(matrix_glob)}')
                GROUP BY aid
            )
            SELECT
                min(neighbor_count)::BIGINT,
                max(neighbor_count)::BIGINT,
                avg(neighbor_count)::DOUBLE,
                quantile_cont(neighbor_count, 0.50)::DOUBLE,
                quantile_cont(neighbor_count, 0.90)::DOUBLE,
                quantile_cont(neighbor_count, 0.95)::DOUBLE,
                quantile_cont(neighbor_count, 0.99)::DOUBLE,
                sum(CASE WHEN neighbor_count < 20 THEN 1 ELSE 0 END)::BIGINT,
                sum(CASE WHEN neighbor_count = {int(topk_neighbors)} THEN 1 ELSE 0 END)::BIGINT
            FROM neighbor_counts
            """
        ).fetchone()
        neighbor_statistics = {
            "min": int(neighbor_summary[0]),
            "max": int(neighbor_summary[1]),
            "mean": round(float(neighbor_summary[2]), 6),
            "p50": float(neighbor_summary[3]),
            "p90": float(neighbor_summary[4]),
            "p95": float(neighbor_summary[5]),
            "p99": float(neighbor_summary[6]),
            "source_items_below_20": int(neighbor_summary[7]),
            "source_items_at_topk": int(neighbor_summary[8]),
        }

        finalize_runtime = round(time.perf_counter() - finalize_started, 6)
        matrix_disk_bytes = _directory_size(matrix_dir)
        report = {
            "dataset": dataset_name,
            "source": source_name,
            "snapshot": str(snapshot_source),
            "output": str(target),
            "configuration": {
                "recent_events_per_session": recent_events_per_session,
                "topk_neighbors": topk_neighbors,
                "pair_buckets": pair_buckets,
                "allowed_event_types": list(event_types),
                "event_type_weights": {str(key): value for key, value in weights.items()},
                "pair_score_mode": pair_score_mode,
                "max_time_difference_ms": max_time_difference_ms,
                "time_decay_ms": time_decay_ms,
                "pair_semantics": pair_semantics,
            },
            "snapshot_stats": {
                "event_files": len(event_files),
                "event_rows": source_rows,
                "eligible_event_rows": int(source_summary[4]),
                "sessions": int(source_summary[2]),
                "unique_items": int(source_summary[3]),
                "min_ts": int(source_summary[0]),
                "max_ts": int(source_summary[1]),
            },
            "partial_pairs": {
                "files": len(partial_files),
                "rows_after_per_shard_aggregation": partial_rows,
                "peak_disk_bytes": partial_disk_bytes,
                "runtime_seconds": pair_runtime,
                "retained": False,
            },
            "matrix": {
                "files": len(matrix_files),
                "rows": matrix_rows,
                "source_items": matrix_source_items,
                "source_item_coverage": round(
                    matrix_source_items / int(source_summary[3]), 8
                ),
                "neighbors_per_source_item": neighbor_statistics,
                "disk_bytes": matrix_disk_bytes,
                "runtime_seconds": finalize_runtime,
            },
            "assertions": {"passed": not any(violations.values()), "violations": violations},
            "runtime_seconds": round(time.perf_counter() - started, 6),
        }
        (staging / "build_report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    except Exception:
        connection.close()
        shutil.rmtree(staging, ignore_errors=True)
        raise
    else:
        connection.close()

    shutil.rmtree(partial_dir, ignore_errors=True)
    shutil.rmtree(temp_dir, ignore_errors=True)
    (staging / "build_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    staging.replace(target)
    return report


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build disk-backed type-weighted CoVis Top-K.")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--snapshot-dir", type=Path, required=True)
    parser.add_argument("--dataset-name", choices=("ranker", "valid"), required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--recent-events", type=int)
    parser.add_argument("--topk-neighbors", type=int)
    parser.add_argument("--pair-buckets", type=int)
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
        output_dir = Path(experiment_dir) / "recall" / args.dataset_name / "type_covis"
    else:
        raise ValueError("Provide --output-dir or run through the experiment runner")

    covis_config = config["recall"]["type_covis"]
    weights = {
        EVENT_TYPE_TO_ID[name]: float(value)
        for name, value in covis_config["event_type_weights"].items()
    }
    report = build_type_covis(
        args.snapshot_dir,
        output_dir,
        dataset_name=args.dataset_name,
        recent_events_per_session=(
            args.recent_events or covis_config["recent_events_per_session"]
        ),
        topk_neighbors=args.topk_neighbors or covis_config["topk_neighbors"],
        pair_buckets=args.pair_buckets or covis_config["pair_buckets"],
        event_type_weights=weights,
        compression=config["runtime"]["parquet_compression"],
        row_group_size=config["runtime"]["parquet_row_group_size"],
        workers=config["runtime"]["workers"],
        memory_limit_gb=config["runtime"]["duckdb_memory_limit_gb"],
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
