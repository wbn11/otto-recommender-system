"""Build and evaluate snapshot-safe Popular and Revisit recall sources."""

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
    EVENT_TYPE_TO_ID,
    LABEL_SCHEMA,
    POPULAR_SCHEMA,
    QUERY_SCHEMA,
    RECALL_SCHEMA,
)
from utils.config import resolve_project_config


ROOT = Path(__file__).resolve().parents[2]
TARGET_TYPES = tuple(sorted(EVENT_TYPE_TO_ID.values()))


def _quote(value: str | Path) -> str:
    return str(value).replace("'", "''")


def _parquet_rows(path: Path) -> int:
    return pq.ParquetFile(path).metadata.num_rows


def _directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def compute_revisit_candidates(
    aids: Sequence[int],
    event_types: Sequence[int],
    *,
    topk: int = 200,
    recency_scale_events: float = 10.0,
    event_type_weights: Mapping[int, float] | None = None,
) -> list[tuple[int, float]]:
    """Return unique history items ranked by recency, last type and frequency.

    This pure-Python reference implementation is intentionally kept beside the
    DuckDB production query so the scoring rule can be verified by a hand-made
    unit test.
    """

    if len(aids) != len(event_types):
        raise ValueError("aids and event_types must have equal lengths")
    if topk <= 0:
        raise ValueError("topk must be positive")
    if recency_scale_events <= 0:
        raise ValueError("recency_scale_events must be positive")
    weights = dict(event_type_weights or {1: 1.0, 2: 3.0, 3: 6.0})
    stats: dict[int, list[int]] = {}
    for position, (aid, event_type) in enumerate(zip(aids, event_types, strict=True)):
        if int(event_type) not in weights:
            raise ValueError(f"Unknown event type: {event_type}")
        item = int(aid)
        if item in stats:
            stats[item][0] += 1
            stats[item][1] = position
            stats[item][2] = int(event_type)
        else:
            stats[item] = [1, position, int(event_type)]

    history_length = len(aids)
    scored = []
    for aid, (frequency, last_position, last_event_type) in stats.items():
        distance_to_end = history_length - 1 - last_position
        score = (
            math.exp(-distance_to_end / recency_scale_events)
            * weights[last_event_type]
            * math.log1p(frequency)
        )
        scored.append((aid, score))
    scored.sort(key=lambda pair: (-pair[1], pair[0]))
    return scored[:topk]


def _validate_inputs(snapshot_dir: Path, queries_path: Path, labels_path: Path) -> list[Path]:
    event_files = sorted(snapshot_dir.glob("part-*.parquet"))
    if not event_files:
        event_files = sorted((snapshot_dir / "events").glob("part-*.parquet"))
    if not event_files:
        raise FileNotFoundError(f"No snapshot event shards found in {snapshot_dir}")
    if not queries_path.is_file():
        raise FileNotFoundError(queries_path)
    if not labels_path.is_file():
        raise FileNotFoundError(labels_path)
    _check_parquet_schema(queries_path, QUERY_SCHEMA, "query")
    _check_parquet_schema(labels_path, LABEL_SCHEMA, "label")
    return event_files


def _check_parquet_schema(path: Path, expected, name: str) -> None:
    actual = pq.read_schema(path)
    if actual.names != expected.names or actual.types != expected.types:
        raise AssertionError(f"Unexpected {name} schema: {actual}")


def _build_popular(
    connection: duckdb.DuckDBPyConnection,
    event_glob: str,
    output_path: Path,
    *,
    topk: int,
    compression: str,
    row_group_size: int,
) -> dict[str, Any]:
    connection.execute(
        f"""
        CREATE TEMP TABLE popular_counts AS
        SELECT
            event_type::TINYINT AS target_type,
            aid::BIGINT AS aid,
            count(*)::BIGINT AS event_count
        FROM read_parquet('{_quote(event_glob)}', union_by_name=true)
        WHERE event_type BETWEEN 1 AND 3
        GROUP BY event_type, aid
        """
    )
    connection.execute(
        f"""
        COPY (
            WITH ranked AS (
                SELECT
                    target_type,
                    aid,
                    row_number() OVER (
                        PARTITION BY target_type
                        ORDER BY event_count DESC, aid ASC
                    ) AS source_rank,
                    event_count::FLOAT AS source_score
                FROM popular_counts
            )
            SELECT
                target_type::TINYINT AS target_type,
                aid::BIGINT AS aid,
                'popular'::VARCHAR AS source,
                source_rank::INTEGER AS source_rank,
                source_score::FLOAT AS source_score
            FROM ranked
            WHERE source_rank <= {int(topk)}
            ORDER BY target_type, source_rank
        ) TO '{_quote(output_path)}' (
            FORMAT PARQUET,
            COMPRESSION {compression.upper()},
            ROW_GROUP_SIZE {int(row_group_size)}
        )
        """
    )
    _check_parquet_schema(output_path, POPULAR_SCHEMA, "Popular")
    counts = connection.execute(
        f"""SELECT target_type, count(*)
             FROM read_parquet('{_quote(output_path)}')
             GROUP BY target_type ORDER BY target_type"""
    ).fetchall()
    return {
        "rows": _parquet_rows(output_path),
        "rows_by_target_type": {str(int(target)): int(rows) for target, rows in counts},
    }


def _weight_case(event_type_weights: Mapping[int, float]) -> str:
    clauses = " ".join(
        f"WHEN {int(event_type)} THEN {float(weight)}"
        for event_type, weight in sorted(event_type_weights.items())
    )
    return f"CASE last_event_type {clauses} ELSE NULL END"


def _build_revisit(
    connection: duckdb.DuckDBPyConnection,
    queries_path: Path,
    output_path: Path,
    *,
    topk: int,
    recency_scale_events: float,
    event_type_weights: Mapping[int, float],
    compression: str,
    row_group_size: int,
) -> dict[str, Any]:
    invalid_types = int(connection.execute(
        f"""SELECT count(*) FROM (
                SELECT unnest(event_types) AS event_type
                FROM read_parquet('{_quote(queries_path)}')
             ) WHERE event_type NOT BETWEEN 1 AND 3"""
    ).fetchone()[0])
    if invalid_types:
        raise ValueError(f"Found {invalid_types} query events with an invalid event type")

    weight_case = _weight_case(event_type_weights)
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
                FROM read_parquet('{_quote(queries_path)}')
            ),
            item_history AS (
                SELECT
                    session,
                    aid,
                    count(*)::INTEGER AS frequency,
                    max(position)::INTEGER AS last_position,
                    arg_max(event_type, position)::TINYINT AS last_event_type,
                    max(history_length)::INTEGER AS history_length
                FROM exploded
                GROUP BY session, aid
            ),
            scored AS (
                SELECT
                    session,
                    aid,
                    (
                        exp(
                            -(history_length - last_position)::DOUBLE
                            / {float(recency_scale_events)}
                        )
                        * ({weight_case})
                        * ln(1.0 + frequency)
                    )::FLOAT AS source_score
                FROM item_history
            ),
            ranked AS (
                SELECT
                    session,
                    aid,
                    source_score,
                    row_number() OVER (
                        PARTITION BY session
                        ORDER BY source_score DESC, aid ASC
                    )::INTEGER AS source_rank
                FROM scored
            )
            SELECT
                ranked.session::BIGINT AS session,
                targets.target_type::TINYINT AS target_type,
                ranked.aid::BIGINT AS aid,
                'revisit'::VARCHAR AS source,
                ranked.source_rank::INTEGER AS source_rank,
                ranked.source_score::FLOAT AS source_score
            FROM ranked
            CROSS JOIN (VALUES (1), (2), (3)) AS targets(target_type)
            WHERE ranked.source_rank <= {int(topk)}
        ) TO '{_quote(output_path)}' (
            FORMAT PARQUET,
            COMPRESSION {compression.upper()},
            ROW_GROUP_SIZE {int(row_group_size)}
        )
        """
    )
    _check_parquet_schema(output_path, RECALL_SCHEMA, "Revisit")
    duplicate_rows = int(connection.execute(
        f"""SELECT count(*) FROM (
                SELECT session, target_type, aid, count(*) AS n
                FROM read_parquet('{_quote(output_path)}')
                GROUP BY session, target_type, aid HAVING count(*) > 1
             )"""
    ).fetchone()[0])
    over_budget_groups = int(connection.execute(
        f"""SELECT count(*) FROM (
                SELECT session, target_type, count(*) AS n
                FROM read_parquet('{_quote(output_path)}')
                GROUP BY session, target_type HAVING count(*) > {int(topk)}
             )"""
    ).fetchone()[0])
    if duplicate_rows or over_budget_groups:
        raise AssertionError(
            f"Invalid Revisit candidates: duplicates={duplicate_rows}, "
            f"over_budget_groups={over_budget_groups}"
        )
    return {
        "rows": _parquet_rows(output_path),
        "duplicate_candidates": duplicate_rows,
        "groups_over_topk": over_budget_groups,
    }


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
            FROM read_parquet('{_quote(labels_path)}')
            GROUP BY session, target_type
        )
        GROUP BY target_type ORDER BY target_type
        """
    ).fetchall()
    return {int(target): int(value) for target, value in rows}


def _hits(
    connection: duckdb.DuckDBPyConnection,
    labels_path: Path,
    candidates_path: Path,
    source: str,
    k: int,
) -> dict[int, int]:
    session_join = "l.session = c.session AND" if source == "revisit" else ""
    rows = connection.execute(
        f"""
        SELECT l.target_type, count(*)::BIGINT
        FROM read_parquet('{_quote(labels_path)}') l
        INNER JOIN read_parquet('{_quote(candidates_path)}') c
          ON {session_join} l.target_type = c.target_type AND l.aid = c.aid
        WHERE c.source_rank <= {int(k)}
        GROUP BY l.target_type ORDER BY l.target_type
        """
    ).fetchall()
    return {int(target): int(value) for target, value in rows}


def _evaluate_sources(
    connection: duckdb.DuckDBPyConnection,
    labels_path: Path,
    popular_path: Path,
    revisit_path: Path,
    *,
    eval_ks: Sequence[int],
    target_weights: Mapping[int, float],
) -> dict[str, Any]:
    result: dict[str, Any] = {"popular": {}, "revisit": {}}
    for k in sorted(set(int(value) for value in eval_ks)):
        denominators = _denominators(connection, labels_path, k)
        for source, candidate_path in (
            ("popular", popular_path),
            ("revisit", revisit_path),
        ):
            hits = _hits(connection, labels_path, candidate_path, source, k)
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
            result[source][f"recall_at_{k}"] = {
                "by_target_type": recalls,
                "weighted": round(weighted, 8) if has_all_targets else None,
                "hits": {str(key): value for key, value in hits.items()},
                "denominators": {str(key): value for key, value in denominators.items()},
            }
    return result


def _coverage(
    connection: duckdb.DuckDBPyConnection,
    queries_path: Path,
    popular_path: Path,
    revisit_path: Path,
) -> dict[str, Any]:
    catalog_items = int(connection.execute(
        "SELECT count(DISTINCT aid) FROM popular_counts"
    ).fetchone()[0])
    query_rows = _parquet_rows(queries_path)
    expected_groups = query_rows * len(TARGET_TYPES)
    result: dict[str, Any] = {}
    for source, candidate_path in (
        ("popular", popular_path),
        ("revisit", revisit_path),
    ):
        rows, unique_items = connection.execute(
            f"""SELECT count(*)::BIGINT, count(DISTINCT aid)::BIGINT
                 FROM read_parquet('{_quote(candidate_path)}')"""
        ).fetchone()
        if source == "popular":
            groups = expected_groups if rows else 0
            average_candidates = rows / len(TARGET_TYPES) if query_rows else 0.0
        else:
            groups = int(connection.execute(
                f"""SELECT count(*) FROM (
                        SELECT session, target_type
                        FROM read_parquet('{_quote(candidate_path)}')
                        GROUP BY session, target_type
                     )"""
            ).fetchone()[0])
            average_candidates = rows / expected_groups if expected_groups else 0.0
        result[source] = {
            "candidate_rows_stored": int(rows),
            "unique_candidate_items": int(unique_items),
            "catalog_item_coverage": round(unique_items / catalog_items, 8) if catalog_items else None,
            "query_group_coverage": round(groups / expected_groups, 8) if expected_groups else None,
            "average_candidates_per_group": round(float(average_candidates), 6),
        }
    return {"catalog_items": catalog_items, "query_groups": expected_groups, **result}


def build_popular_revisit(
    snapshot_dir: str | Path,
    queries_path: str | Path,
    labels_path: str | Path,
    output_dir: str | Path,
    *,
    dataset_name: str,
    topk: int = 200,
    eval_ks: Sequence[int] = (20, 50, 100),
    recency_scale_events: float = 10.0,
    event_type_weights: Mapping[int, float] | None = None,
    target_weights: Mapping[int, float] | None = None,
    compression: str = "zstd",
    row_group_size: int = 250_000,
    workers: int = 8,
    memory_limit_gb: int = 8,
) -> dict[str, Any]:
    """Build both sources from one point-in-time snapshot and query dataset."""

    snapshot_source = Path(snapshot_dir).resolve()
    query_source = Path(queries_path).resolve()
    label_source = Path(labels_path).resolve()
    target = Path(output_dir).resolve()
    event_files = _validate_inputs(snapshot_source, query_source, label_source)
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite existing recall dataset: {target}")
    if topk <= 0 or any(int(value) <= 0 or int(value) > topk for value in eval_ks):
        raise ValueError("Evaluation K values must be positive and no greater than topk")
    if recency_scale_events <= 0:
        raise ValueError("recency_scale_events must be positive")
    if workers <= 0 or memory_limit_gb <= 0:
        raise ValueError("workers and memory_limit_gb must be positive")

    item_weights = dict(event_type_weights or {1: 1.0, 2: 3.0, 3: 6.0})
    metric_weights = dict(target_weights or {1: 0.1, 2: 0.3, 3: 0.6})
    if set(item_weights) != set(TARGET_TYPES) or set(metric_weights) != set(TARGET_TYPES):
        raise ValueError("Weights must define event types 1, 2 and 3")
    if any(value <= 0 for value in item_weights.values()):
        raise ValueError("Revisit event type weights must be positive")
    if any(value < 0 for value in metric_weights.values()):
        raise ValueError("Target metric weights must be non-negative")
    if not math.isclose(sum(metric_weights.values()), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("Target metric weights must sum to 1.0")

    target.parent.mkdir(parents=True, exist_ok=True)
    token = f"{os.getpid()}-{uuid.uuid4().hex[:8]}"
    staging = target.parent / f".{target.name}.building-{token}"
    staging.mkdir(parents=True, exist_ok=False)
    popular_dir = staging / "popular"
    revisit_dir = staging / "revisit"
    popular_dir.mkdir()
    revisit_dir.mkdir()
    temp_dir = staging / "duckdb_tmp"
    temp_dir.mkdir()
    popular_path = popular_dir / "items.parquet"
    revisit_path = revisit_dir / "candidates.parquet"
    event_glob = str(event_files[0].parent / "part-*.parquet")
    started = time.perf_counter()
    connection = duckdb.connect()

    try:
        connection.execute(f"SET threads={int(workers)}")
        connection.execute(f"SET memory_limit='{int(memory_limit_gb)}GB'")
        connection.execute(f"SET temp_directory='{_quote(temp_dir)}'")
        connection.execute("SET preserve_insertion_order=false")
        snapshot_max_ts = int(connection.execute(
            f"SELECT max(ts) FROM read_parquet('{_quote(event_glob)}', union_by_name=true)"
        ).fetchone()[0])
        query_min_ts = int(connection.execute(
            f"SELECT min(list_min(timestamps)) FROM read_parquet('{_quote(query_source)}')"
        ).fetchone()[0])
        if snapshot_max_ts >= query_min_ts:
            raise AssertionError(
                "Point-in-time violation: snapshot maximum timestamp must be earlier "
                "than every supervised query timestamp"
            )
        popular_started = time.perf_counter()
        popular = _build_popular(
            connection,
            event_glob,
            popular_path,
            topk=topk,
            compression=compression,
            row_group_size=row_group_size,
        )
        popular["runtime_seconds"] = round(time.perf_counter() - popular_started, 6)
        print(
            f"[recall] dataset={dataset_name} source=popular rows={popular['rows']:,} "
            f"elapsed={popular['runtime_seconds']:.1f}s",
            flush=True,
        )

        revisit_started = time.perf_counter()
        revisit = _build_revisit(
            connection,
            query_source,
            revisit_path,
            topk=topk,
            recency_scale_events=recency_scale_events,
            event_type_weights=item_weights,
            compression=compression,
            row_group_size=row_group_size,
        )
        revisit["runtime_seconds"] = round(time.perf_counter() - revisit_started, 6)
        print(
            f"[recall] dataset={dataset_name} source=revisit rows={revisit['rows']:,} "
            f"elapsed={revisit['runtime_seconds']:.1f}s",
            flush=True,
        )

        metrics_started = time.perf_counter()
        recall_metrics = _evaluate_sources(
            connection,
            label_source,
            popular_path,
            revisit_path,
            eval_ks=eval_ks,
            target_weights=metric_weights,
        )
        coverage = _coverage(connection, query_source, popular_path, revisit_path)
        metrics_runtime = round(time.perf_counter() - metrics_started, 6)
        report = {
            "dataset": dataset_name,
            "sources": {
                "snapshot": str(snapshot_source),
                "snapshot_event_files": len(event_files),
                "queries": str(query_source),
                "query_rows": _parquet_rows(query_source),
                "labels": str(label_source),
                "label_rows": _parquet_rows(label_source),
                "snapshot_max_ts": snapshot_max_ts,
                "query_min_ts": query_min_ts,
            },
            "output": str(target),
            "configuration": {
                "topk_per_source": topk,
                "eval_ks": sorted(set(int(value) for value in eval_ks)),
                "revisit_score": "exp(-distance_to_end/scale)*last_event_type_weight*log1p(frequency)",
                "recency_scale_events": recency_scale_events,
                "event_type_weights": {str(key): value for key, value in item_weights.items()},
                "target_metric_weights": {str(key): value for key, value in metric_weights.items()},
            },
            "artifacts": {"popular": popular, "revisit": revisit},
            "metrics": recall_metrics,
            "coverage": coverage,
            "leakage_assertions": {
                "passed": snapshot_max_ts < query_min_ts,
                "snapshot_max_ts_before_query_min_ts": snapshot_max_ts < query_min_ts,
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

    shutil.rmtree(temp_dir, ignore_errors=True)
    report["disk_bytes"] = _directory_size(staging)
    (staging / "metrics.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    staging.replace(target)
    return report


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build snapshot-safe Popular and Revisit recall.")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--snapshot-dir", type=Path, required=True)
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
        output_dir = Path(experiment_dir) / "recall" / args.dataset_name
    else:
        raise ValueError("Provide --output-dir or run through the experiment runner")

    recall_config = config["recall"]
    event_type_weights = {
        EVENT_TYPE_TO_ID[name]: float(value)
        for name, value in recall_config["revisit"]["event_type_weights"].items()
    }
    target_weights = {
        EVENT_TYPE_TO_ID[name]: float(value)
        for name, value in recall_config["type_weights"].items()
    }
    report = build_popular_revisit(
        args.snapshot_dir,
        args.queries,
        args.labels,
        output_dir,
        dataset_name=args.dataset_name,
        topk=args.topk or recall_config["topk_per_source"],
        eval_ks=recall_config["eval_ks"],
        recency_scale_events=recall_config["revisit"]["recency_scale_events"],
        event_type_weights=event_type_weights,
        target_weights=target_weights,
        compression=config["runtime"]["parquet_compression"],
        row_group_size=config["runtime"]["parquet_row_group_size"],
        workers=config["runtime"]["workers"],
        memory_limit_gb=config["runtime"]["duckdb_memory_limit_gb"],
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
