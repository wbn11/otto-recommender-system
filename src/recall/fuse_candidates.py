"""Fuse six recall sources into a bounded, feature-rich candidate table.

Purpose
-------
Produce exactly K unique candidates for every ``(session, target_type)`` while
preserving each source's rank and score for the downstream feature builder.

Workflow
--------
1. Repartition queries, Revisit and compact DSSM lists by session hash.
2. Process one session bucket and one target type at a time.
3. Union Revisit, three CoVis sources and DSSM, then deduplicate by aid.
4. Select candidates in source-balanced rank order without score calibration.
5. Fill short groups from the target-specific Popular list.
6. Backfill all source flags/ranks/scores and evaluate candidate Recall@K.

The output is a long Parquet table partitioned into deterministic session
buckets. It is intentionally ready for feature generation and LightGBM.
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
    FUSED_CANDIDATE_SCHEMA,
    LABEL_SCHEMA,
    POPULAR_SCHEMA,
    QUERY_SCHEMA,
    RECALL_SCHEMA,
    SESSION_RECALL_SCHEMA,
)
from utils.config import resolve_project_config


TARGET_TYPES = (1, 2, 3)
SOURCE_NAMES = (
    "popular",
    "revisit",
    "type_covis",
    "buy2buy",
    "time_covis",
    "dssm",
)
SOURCE_IDS = {
    "revisit": 1,
    "type_covis": 2,
    "buy2buy": 3,
    "time_covis": 4,
    "dssm": 5,
    "popular": 6,
}
_SESSION_BUCKET_RE = re.compile(r"session-bucket-(\d+)\.parquet$")


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


def _parquet_rows(paths: Sequence[Path]) -> int:
    return sum(pq.ParquetFile(path).metadata.num_rows for path in paths)


def _check_schema(path: Path, expected: pa.Schema, name: str) -> None:
    actual = pq.read_schema(path)
    if actual.names != expected.names or actual.types != expected.types:
        raise AssertionError(f"Unexpected {name} schema in {path}: {actual}")


def _candidate_files(path: str | Path, pattern: str) -> list[Path]:
    source = Path(path).resolve()
    if source.is_file():
        files = [source]
    elif source.is_dir():
        files = sorted(source.rglob(pattern))
    else:
        raise FileNotFoundError(source)
    if not files:
        raise FileNotFoundError(f"No {pattern} files below {source}")
    return files


def _bucket_file_map(path: str | Path, source_name: str) -> dict[int, Path]:
    files = _candidate_files(path, "session-bucket-*.parquet")
    result: dict[int, Path] = {}
    for file in files:
        match = _SESSION_BUCKET_RE.search(file.name)
        if match is None:
            continue
        bucket = int(match.group(1))
        if bucket in result:
            raise ValueError(f"Duplicate {source_name} session bucket {bucket}")
        _check_schema(file, SESSION_RECALL_SCHEMA, source_name)
        result[bucket] = file
    if not result:
        raise ValueError(f"No final {source_name} candidate buckets found")
    return result


def _validate_dssm_schema(path: Path) -> None:
    schema = pq.read_schema(path)
    if schema.names != ["session", "target_type", "aids", "scores"]:
        raise AssertionError(f"Unexpected DSSM candidate columns in {path}: {schema}")
    if schema.field("session").type != pa.int64() or schema.field("target_type").type != pa.int8():
        raise AssertionError(f"Unexpected DSSM key types in {path}: {schema}")
    aids = schema.field("aids").type
    scores = schema.field("scores").type
    if not pa.types.is_fixed_size_list(aids) or aids.value_type != pa.int64():
        raise AssertionError(f"DSSM aids must be fixed-size int64 lists: {schema}")
    if not pa.types.is_fixed_size_list(scores) or scores.value_type != pa.float32():
        raise AssertionError(f"DSSM scores must be fixed-size float32 lists: {schema}")
    if aids.list_size != scores.list_size:
        raise AssertionError("DSSM aid and score list sizes differ")


def _partition_inputs(
    connection: duckdb.DuckDBPyConnection,
    queries_path: Path,
    revisit_path: Path,
    dssm_files: Sequence[Path],
    output_dir: Path,
    *,
    session_buckets: int,
    compression: str,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=False)
    groups_dir = output_dir / "groups"
    revisit_dir = output_dir / "revisit"
    dssm_dir = output_dir / "dssm"
    started = time.perf_counter()
    connection.execute(
        f"""
        COPY (
            SELECT
                (session % {int(session_buckets)})::INTEGER AS session_bucket,
                session::BIGINT AS session
            FROM read_parquet('{_quote(_path(queries_path))}')
        ) TO '{_quote(_path(groups_dir))}' (
            FORMAT PARQUET,
            PARTITION_BY (session_bucket),
            COMPRESSION {compression.upper()}
        )
        """
    )
    connection.execute(
        f"""
        COPY (
            SELECT
                (session % {int(session_buckets)})::INTEGER AS session_bucket,
                session::BIGINT AS session,
                target_type::TINYINT AS target_type,
                aid::BIGINT AS aid,
                source_rank::INTEGER AS source_rank,
                source_score::FLOAT AS source_score
            FROM read_parquet('{_quote(_path(revisit_path))}')
        ) TO '{_quote(_path(revisit_dir))}' (
            FORMAT PARQUET,
            PARTITION_BY (session_bucket),
            COMPRESSION {compression.upper()}
        )
        """
    )
    connection.execute(
        f"""
        COPY (
            SELECT
                (session % {int(session_buckets)})::INTEGER AS session_bucket,
                session::BIGINT AS session,
                target_type::TINYINT AS target_type,
                aids,
                scores
            FROM read_parquet({_sql_files(dssm_files)}, union_by_name=true)
        ) TO '{_quote(_path(dssm_dir))}' (
            FORMAT PARQUET,
            PARTITION_BY (session_bucket),
            COMPRESSION {compression.upper()}
        )
        """
    )
    return {
        "runtime_seconds": round(time.perf_counter() - started, 6),
        "disk_bytes": _directory_size(output_dir),
        "groups_files": len(list(groups_dir.rglob("*.parquet"))),
        "revisit_files": len(list(revisit_dir.rglob("*.parquet"))),
        "dssm_files": len(list(dssm_dir.rglob("*.parquet"))),
    }


def _bucket_parts(root: Path, bucket: int) -> list[Path]:
    return sorted((root / f"session_bucket={bucket}").glob("*.parquet"))


def _source_selects(
    *,
    target_type: int,
    revisit_files: Sequence[Path],
    type_covis_file: Path | None,
    buy2buy_file: Path | None,
    time_covis_file: Path | None,
    dssm_files: Sequence[Path],
) -> list[str]:
    selects = [
        f"""
        SELECT session, aid, source_rank, source_score, 1::TINYINT AS source_id
        FROM read_parquet({_sql_files(revisit_files)}, union_by_name=true)
        WHERE target_type = {int(target_type)}
        """
    ]
    for source_id, candidate_file in (
        (2, type_covis_file),
        (3, buy2buy_file),
        (4, time_covis_file),
    ):
        if candidate_file is None:
            continue
        selects.append(
            f"""
            SELECT session, aid, source_rank, source_score,
                   {source_id}::TINYINT AS source_id
            FROM read_parquet('{_quote(_path(candidate_file))}')
            """
        )
    selects.append(
        f"""
        SELECT
            session::BIGINT AS session,
            unnest(aids)::BIGINT AS aid,
            generate_subscripts(aids, 1)::INTEGER AS source_rank,
            unnest(scores)::FLOAT AS source_score,
            5::TINYINT AS source_id
        FROM read_parquet({_sql_files(dssm_files)}, union_by_name=true)
        WHERE target_type = {int(target_type)}
        """
    )
    return selects


def _create_personalized(
    connection: duckdb.DuckDBPyConnection,
    source_selects: Sequence[str],
    *,
    candidate_k: int,
) -> None:
    union_sql = " UNION ALL ".join(source_selects)
    connection.execute("DROP TABLE IF EXISTS personalized")
    connection.execute(
        f"""
        CREATE TEMP TABLE personalized AS
        WITH unioned AS (
            {union_sql}
        ), features AS (
            SELECT
                session::BIGINT AS session,
                aid::BIGINT AS aid,
                min(source_rank) FILTER (WHERE source_id = 1)::INTEGER AS revisit_rank,
                max(source_score) FILTER (WHERE source_id = 1)::FLOAT AS revisit_score,
                min(source_rank) FILTER (WHERE source_id = 2)::INTEGER AS type_covis_rank,
                max(source_score) FILTER (WHERE source_id = 2)::FLOAT AS type_covis_score,
                min(source_rank) FILTER (WHERE source_id = 3)::INTEGER AS buy2buy_rank,
                max(source_score) FILTER (WHERE source_id = 3)::FLOAT AS buy2buy_score,
                min(source_rank) FILTER (WHERE source_id = 4)::INTEGER AS time_covis_rank,
                max(source_score) FILTER (WHERE source_id = 4)::FLOAT AS time_covis_score,
                min(source_rank) FILTER (WHERE source_id = 5)::INTEGER AS dssm_rank,
                max(source_score) FILTER (WHERE source_id = 5)::FLOAT AS dssm_score
            FROM unioned
            GROUP BY session, aid
        ), selection_round AS (
            SELECT
                *,
                least(
                    coalesce(revisit_rank, 32767),
                    coalesce(type_covis_rank, 32767),
                    coalesce(buy2buy_rank, 32767),
                    coalesce(time_covis_rank, 32767),
                    coalesce(dssm_rank, 32767)
                )::INTEGER AS first_round
            FROM features
        ), selection_source AS (
            SELECT
                *,
                CASE
                    WHEN revisit_rank = first_round THEN 1
                    WHEN type_covis_rank = first_round THEN 2
                    WHEN buy2buy_rank = first_round THEN 3
                    WHEN time_covis_rank = first_round THEN 4
                    ELSE 5
                END::TINYINT AS selected_source
            FROM selection_round
        ), ranked AS (
            SELECT
                *,
                row_number() OVER (
                    PARTITION BY session
                    ORDER BY first_round, selected_source, aid
                )::INTEGER AS candidate_rank
            FROM selection_source
        )
        SELECT * FROM ranked WHERE candidate_rank <= {int(candidate_k)}
        """
    )


def _write_final_target_bucket(
    connection: duckdb.DuckDBPyConnection,
    query_files: Sequence[Path],
    output_path: Path,
    *,
    target_type: int,
    candidate_k: int,
    compression: str,
    row_group_size: int,
) -> dict[str, int]:
    connection.execute(
        f"""
        COPY (
            WITH selected_counts AS (
                SELECT session, count(*)::INTEGER AS selected_count
                FROM personalized
                GROUP BY session
            ), query_groups AS (
                SELECT
                    queries.session::BIGINT AS session,
                    coalesce(selected_counts.selected_count, 0)::INTEGER AS selected_count
                FROM read_parquet({_sql_files(query_files)}, union_by_name=true) queries
                LEFT JOIN selected_counts USING (session)
            ), fallback_ranked AS (
                SELECT
                    query_groups.session,
                    popular.aid,
                    popular.source_rank AS popular_rank,
                    popular.source_score AS popular_score,
                    query_groups.selected_count,
                    row_number() OVER (
                        PARTITION BY query_groups.session
                        ORDER BY popular.source_rank, popular.aid
                    )::INTEGER AS fallback_order
                FROM query_groups
                INNER JOIN popular_lookup popular
                  ON popular.target_type = {int(target_type)}
                LEFT JOIN personalized selected
                  ON selected.session = query_groups.session
                 AND selected.aid = popular.aid
                WHERE selected.aid IS NULL
            ), fallback AS (
                SELECT *
                FROM fallback_ranked
                WHERE fallback_order <= {int(candidate_k)} - selected_count
            ), selected_with_popular AS (
                SELECT
                    selected.*,
                    popular.source_rank AS popular_rank,
                    popular.source_score AS popular_score
                FROM personalized selected
                LEFT JOIN popular_lookup popular
                  ON popular.target_type = {int(target_type)}
                 AND popular.aid = selected.aid
            ), combined AS (
                SELECT
                    session,
                    aid,
                    candidate_rank,
                    selected_source,
                    popular_rank,
                    popular_score,
                    revisit_rank,
                    revisit_score,
                    type_covis_rank,
                    type_covis_score,
                    buy2buy_rank,
                    buy2buy_score,
                    time_covis_rank,
                    time_covis_score,
                    dssm_rank,
                    dssm_score
                FROM selected_with_popular
                UNION ALL
                SELECT
                    session,
                    aid,
                    selected_count + fallback_order AS candidate_rank,
                    6::TINYINT AS selected_source,
                    popular_rank,
                    popular_score,
                    NULL::INTEGER AS revisit_rank,
                    NULL::FLOAT AS revisit_score,
                    NULL::INTEGER AS type_covis_rank,
                    NULL::FLOAT AS type_covis_score,
                    NULL::INTEGER AS buy2buy_rank,
                    NULL::FLOAT AS buy2buy_score,
                    NULL::INTEGER AS time_covis_rank,
                    NULL::FLOAT AS time_covis_score,
                    NULL::INTEGER AS dssm_rank,
                    NULL::FLOAT AS dssm_score
                FROM fallback
            ), flags AS (
                SELECT
                    *,
                    (popular_rank IS NOT NULL)::TINYINT AS from_popular,
                    (revisit_rank IS NOT NULL)::TINYINT AS from_revisit,
                    (type_covis_rank IS NOT NULL)::TINYINT AS from_type_covis,
                    (buy2buy_rank IS NOT NULL)::TINYINT AS from_buy2buy,
                    (time_covis_rank IS NOT NULL)::TINYINT AS from_time_covis,
                    (dssm_rank IS NOT NULL)::TINYINT AS from_dssm
                FROM combined
            )
            SELECT
                session::BIGINT AS session,
                {int(target_type)}::TINYINT AS target_type,
                aid::BIGINT AS aid,
                candidate_rank::SMALLINT AS candidate_rank,
                selected_source::TINYINT AS selected_source,
                from_popular,
                coalesce(popular_rank, 0)::SMALLINT AS popular_rank,
                coalesce(popular_score, 0.0)::FLOAT AS popular_score,
                from_revisit,
                coalesce(revisit_rank, 0)::SMALLINT AS revisit_rank,
                coalesce(revisit_score, 0.0)::FLOAT AS revisit_score,
                from_type_covis,
                coalesce(type_covis_rank, 0)::SMALLINT AS type_covis_rank,
                coalesce(type_covis_score, 0.0)::FLOAT AS type_covis_score,
                from_buy2buy,
                coalesce(buy2buy_rank, 0)::SMALLINT AS buy2buy_rank,
                coalesce(buy2buy_score, 0.0)::FLOAT AS buy2buy_score,
                from_time_covis,
                coalesce(time_covis_rank, 0)::SMALLINT AS time_covis_rank,
                coalesce(time_covis_score, 0.0)::FLOAT AS time_covis_score,
                from_dssm,
                coalesce(dssm_rank, 0)::SMALLINT AS dssm_rank,
                coalesce(dssm_score, 0.0)::FLOAT AS dssm_score,
                (
                    from_popular + from_revisit + from_type_covis
                    + from_buy2buy + from_time_covis + from_dssm
                )::TINYINT AS source_count,
                least(
                    coalesce(popular_rank, 32767),
                    coalesce(revisit_rank, 32767),
                    coalesce(type_covis_rank, 32767),
                    coalesce(buy2buy_rank, 32767),
                    coalesce(time_covis_rank, 32767),
                    coalesce(dssm_rank, 32767)
                )::SMALLINT AS best_source_rank
            FROM flags
            ORDER BY session, candidate_rank
        ) TO '{_quote(_path(output_path))}' (
            FORMAT PARQUET,
            COMPRESSION {compression.upper()},
            ROW_GROUP_SIZE {int(row_group_size)}
        )
        """
    )
    _check_schema(output_path, FUSED_CANDIDATE_SCHEMA, "fused candidate")
    query_sessions = int(
        connection.execute(
            f"SELECT count(*) FROM read_parquet({_sql_files(query_files)}, union_by_name=true)"
        ).fetchone()[0]
    )
    rows = pq.ParquetFile(output_path).metadata.num_rows
    invalid_groups = int(
        connection.execute(
            f"""
            SELECT count(*) FROM (
                SELECT session
                FROM read_parquet('{_quote(_path(output_path))}')
                GROUP BY session
                HAVING count(*) != {int(candidate_k)}
                    OR count(DISTINCT aid) != {int(candidate_k)}
                    OR count(DISTINCT candidate_rank) != {int(candidate_k)}
                    OR min(candidate_rank) != 1
                    OR max(candidate_rank) != {int(candidate_k)}
            )
            """
        ).fetchone()[0]
    )
    invalid_features = int(
        connection.execute(
            f"""
            SELECT count(*)
            FROM read_parquet('{_quote(_path(output_path))}')
            WHERE source_count != (
                    from_popular + from_revisit + from_type_covis
                    + from_buy2buy + from_time_covis + from_dssm
                  )
               OR source_count < 1 OR source_count > 6
               OR selected_source < 1 OR selected_source > 6
               OR best_source_rank < 1
            """
        ).fetchone()[0]
    )
    if rows != query_sessions * candidate_k or invalid_groups or invalid_features:
        raise AssertionError(
            "Invalid fused candidate bucket: "
            f"rows={rows}, expected={query_sessions * candidate_k}, "
            f"groups={invalid_groups}, features={invalid_features}"
        )
    return {
        "query_sessions": query_sessions,
        "rows": rows,
        "invalid_groups": invalid_groups,
        "invalid_features": invalid_features,
    }


def _denominators(
    connection: duckdb.DuckDBPyConnection,
    labels_path: Path,
    ks: Sequence[int],
) -> dict[int, dict[int, int]]:
    result: dict[int, dict[int, int]] = {}
    for k in ks:
        rows = connection.execute(
            f"""
            SELECT target_type, sum(least(label_count, {int(k)}))::BIGINT
            FROM (
                SELECT session, target_type, count(DISTINCT aid)::BIGINT AS label_count
                FROM read_parquet('{_quote(_path(labels_path))}')
                GROUP BY session, target_type
            )
            GROUP BY target_type ORDER BY target_type
            """
        ).fetchall()
        result[int(k)] = {int(target): int(value) for target, value in rows}
    return result


def _evaluate_and_analyze(
    connection: duckdb.DuckDBPyConnection,
    candidate_files: Sequence[Path],
    labels_path: Path,
    *,
    eval_ks: Sequence[int],
    candidate_k: int,
    target_weights: Mapping[int, float],
) -> tuple[dict[str, Any], dict[str, Any]]:
    candidate_relation = f"read_parquet({_sql_files(candidate_files)}, union_by_name=true)"
    ks = sorted({int(k) for k in eval_ks if int(k) <= candidate_k} | {candidate_k})
    denominators = _denominators(connection, labels_path, ks)
    hit_columns = ",\n".join(
        f"count(*) FILTER (WHERE candidates.candidate_rank <= {k})::BIGINT AS hit_{k}"
        for k in ks
    )
    source_hit_columns = ",\n".join(
        f"sum(candidates.from_{source})::BIGINT AS {source}_hits, "
        f"sum(CASE WHEN candidates.from_{source} = 1 AND candidates.source_count = 1 "
        f"THEN 1 ELSE 0 END)::BIGINT AS {source}_unique_hits"
        for source in SOURCE_NAMES
    )
    hit_rows = connection.execute(
        f"""
        WITH labels AS (
            SELECT DISTINCT session, target_type, aid
            FROM read_parquet('{_quote(_path(labels_path))}')
        )
        SELECT
            labels.target_type::TINYINT AS target_type,
            {hit_columns},
            {source_hit_columns}
        FROM labels
        INNER JOIN {candidate_relation} candidates
          USING (session, target_type, aid)
        GROUP BY labels.target_type
        ORDER BY labels.target_type
        """
    ).fetchall()
    metrics: dict[str, Any] = {}
    for k_index, k in enumerate(ks):
        hits = {int(row[0]): int(row[1 + k_index]) for row in hit_rows}
        recalls = {
            target: hits.get(target, 0) / denominators[k].get(target, 1)
            for target in TARGET_TYPES
        }
        metrics[f"candidate_recall_at_{k}"] = {
            "by_target_type": {
                str(target): round(recalls[target], 8) for target in TARGET_TYPES
            },
            "weighted": round(
                sum(target_weights[target] * recalls[target] for target in TARGET_TYPES), 8
            ),
            "hits": {str(target): hits.get(target, 0) for target in TARGET_TYPES},
            "denominators": {
                str(target): denominators[k].get(target, 0) for target in TARGET_TYPES
            },
        }

    source_offset = 1 + len(ks)
    individual: dict[str, Any] = {}
    unique_hits: dict[str, Any] = {}
    for source_index, source in enumerate(SOURCE_NAMES):
        hit_col = source_offset + source_index * 2
        unique_col = hit_col + 1
        source_hits = {int(row[0]): int(row[hit_col]) for row in hit_rows}
        source_unique = {int(row[0]): int(row[unique_col]) for row in hit_rows}
        source_recalls = {
            target: source_hits.get(target, 0) / denominators[candidate_k].get(target, 1)
            for target in TARGET_TYPES
        }
        individual[source] = {
            "scope": "source membership among the final fused candidates",
            "by_target_type": {
                str(target): round(source_recalls[target], 8) for target in TARGET_TYPES
            },
            "weighted": round(
                sum(
                    target_weights[target] * source_recalls[target]
                    for target in TARGET_TYPES
                ),
                8,
            ),
        }
        unique_hits[source] = {
            str(target): source_unique.get(target, 0) for target in TARGET_TYPES
        }

    aggregate_columns = []
    for source in SOURCE_NAMES:
        aggregate_columns.extend(
            [
                f"sum(from_{source})::BIGINT AS {source}_presence",
                f"sum((selected_source = {SOURCE_IDS[source]})::INTEGER)::BIGINT "
                f"AS {source}_selected",
            ]
        )
    pairs: list[tuple[str, str]] = []
    for left_index, left in enumerate(SOURCE_NAMES):
        for right in SOURCE_NAMES[left_index + 1 :]:
            pairs.append((left, right))
            aggregate_columns.extend(
                [
                    f"sum((from_{left} = 1 AND from_{right} = 1)::INTEGER)::BIGINT "
                    f"AS {left}_{right}_both",
                    f"sum((from_{left} = 1 OR from_{right} = 1)::INTEGER)::BIGINT "
                    f"AS {left}_{right}_union",
                ]
            )
    aggregate = connection.execute(
        f"SELECT {','.join(aggregate_columns)} FROM {candidate_relation}"
    ).fetchone()
    cursor = 0
    presence: dict[str, int] = {}
    selected: dict[str, int] = {}
    for source in SOURCE_NAMES:
        presence[source] = int(aggregate[cursor])
        selected[source] = int(aggregate[cursor + 1])
        cursor += 2
    overlap: dict[str, dict[str, float]] = {source: {} for source in SOURCE_NAMES}
    for left, right in pairs:
        both = int(aggregate[cursor])
        union = int(aggregate[cursor + 1])
        value = both / union if union else 0.0
        overlap[left][right] = round(value, 8)
        overlap[right][left] = round(value, 8)
        cursor += 2
    for source in SOURCE_NAMES:
        overlap[source][source] = 1.0
    source_count_rows = connection.execute(
        f"""
        SELECT source_count, count(*)::BIGINT
        FROM {candidate_relation}
        GROUP BY source_count ORDER BY source_count
        """
    ).fetchall()
    complementarity = {
        "individual_recall_within_fused_pool": individual,
        "unique_label_hits": unique_hits,
        "candidate_presence_rows": presence,
        "selection_rows": selected,
        "pairwise_jaccard_within_fused_pool": overlap,
        "source_count_distribution": {
            str(int(count)): int(rows) for count, rows in source_count_rows
        },
    }
    return metrics, complementarity


def fuse_candidates(
    queries_path: str | Path,
    labels_path: str | Path,
    popular_path: str | Path,
    revisit_path: str | Path,
    type_covis_path: str | Path,
    buy2buy_path: str | Path,
    time_covis_path: str | Path,
    dssm_path: str | Path,
    output_dir: str | Path,
    config: Mapping[str, Any],
    *,
    dataset_name: str,
    candidate_k: int | None = None,
    session_buckets: int = 64,
) -> dict[str, Any]:
    """Build fused candidates for one point-in-time supervised dataset."""

    queries = Path(queries_path).resolve()
    labels = Path(labels_path).resolve()
    popular = Path(popular_path).resolve()
    revisit = Path(revisit_path).resolve()
    destination = Path(output_dir).resolve()
    for path in (queries, labels, popular, revisit):
        if not path.is_file():
            raise FileNotFoundError(path)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite candidate fusion output: {destination}")
    k = int(candidate_k or config["candidate_k"])
    if k <= 0 or k > 200 or session_buckets <= 0:
        raise ValueError("candidate_k must be in [1, 200] and session_buckets must be positive")
    _check_schema(queries, QUERY_SCHEMA, "query")
    _check_schema(labels, LABEL_SCHEMA, "label")
    _check_schema(popular, POPULAR_SCHEMA, "Popular")
    _check_schema(revisit, RECALL_SCHEMA, "Revisit")
    type_buckets = _bucket_file_map(type_covis_path, "type_covis")
    buy_buckets = _bucket_file_map(buy2buy_path, "buy2buy")
    time_buckets = _bucket_file_map(time_covis_path, "time_covis")
    dssm_files = _candidate_files(dssm_path, "part-*.parquet")
    for file in dssm_files:
        _validate_dssm_schema(file)

    compression = str(config["runtime"]["parquet_compression"])
    row_group_size = int(config["runtime"]["parquet_row_group_size"])
    workers = int(config["runtime"]["workers"])
    memory_limit_gb = int(config["runtime"]["duckdb_memory_limit_gb"])
    target_weights = {
        index + 1: float(config["recall"]["type_weights"][name])
        for index, name in enumerate(("clicks", "carts", "orders"))
    }
    eval_ks = [int(value) for value in config["recall"]["eval_ks"]]

    destination.parent.mkdir(parents=True, exist_ok=True)
    token = f"{os.getpid()}-{uuid.uuid4().hex[:8]}"
    staging = destination.parent / f".{destination.name}.building-{token}"
    work_dir = staging / "work"
    candidate_dir = staging / "parts"
    duckdb_temp = work_dir / "duckdb_tmp"
    work_dir.mkdir(parents=True)
    candidate_dir.mkdir()
    duckdb_temp.mkdir()
    connection = duckdb.connect()
    started = time.perf_counter()
    output_files: list[Path] = []
    total_rows = 0
    invalid_groups = 0
    invalid_features = 0
    try:
        connection.execute(f"SET threads={workers}")
        connection.execute(f"SET memory_limit='{memory_limit_gb}GB'")
        connection.execute(f"SET temp_directory='{_quote(_path(duckdb_temp))}'")
        connection.execute("SET preserve_insertion_order=false")
        connection.execute(
            f"""
            CREATE TEMP TABLE popular_lookup AS
            SELECT target_type, aid, source_rank, source_score
            FROM read_parquet('{_quote(_path(popular))}')
            """
        )
        popular_counts = {
            int(target): int(count)
            for target, count in connection.execute(
                "SELECT target_type, count(*) FROM popular_lookup GROUP BY target_type"
            ).fetchall()
        }
        if any(popular_counts.get(target, 0) < k for target in TARGET_TYPES):
            raise ValueError(f"Popular fallback has fewer than K items: {popular_counts}")

        partition_report = _partition_inputs(
            connection,
            queries,
            revisit,
            dssm_files,
            work_dir / "partitioned",
            session_buckets=session_buckets,
            compression=compression,
        )
        partition_root = work_dir / "partitioned"
        query_rows = pq.ParquetFile(queries).metadata.num_rows
        processed_sessions = 0
        fusion_started = time.perf_counter()
        for bucket in range(session_buckets):
            query_parts = _bucket_parts(partition_root / "groups", bucket)
            if not query_parts:
                continue
            revisit_parts = _bucket_parts(partition_root / "revisit", bucket)
            dssm_parts = _bucket_parts(partition_root / "dssm", bucket)
            if not revisit_parts or not dssm_parts:
                raise AssertionError(f"Missing Revisit or DSSM data for session bucket {bucket}")
            bucket_sessions = _parquet_rows(query_parts)
            processed_sessions += bucket_sessions
            for target_type in TARGET_TYPES:
                selects = _source_selects(
                    target_type=target_type,
                    revisit_files=revisit_parts,
                    type_covis_file=type_buckets.get(bucket),
                    buy2buy_file=buy_buckets.get(bucket),
                    time_covis_file=time_buckets.get(bucket),
                    dssm_files=dssm_parts,
                )
                _create_personalized(connection, selects, candidate_k=k)
                output_path = (
                    candidate_dir
                    / f"session-bucket-{bucket:05d}-target-{target_type}.parquet"
                )
                stats = _write_final_target_bucket(
                    connection,
                    query_parts,
                    output_path,
                    target_type=target_type,
                    candidate_k=k,
                    compression=compression,
                    row_group_size=row_group_size,
                )
                output_files.append(output_path)
                total_rows += stats["rows"]
                invalid_groups += stats["invalid_groups"]
                invalid_features += stats["invalid_features"]
                connection.execute("DROP TABLE personalized")
            print(
                f"[candidate-fusion] dataset={dataset_name} "
                f"bucket={bucket + 1}/{session_buckets} sessions={processed_sessions:,} "
                f"rows={total_rows:,} elapsed={time.perf_counter() - fusion_started:.1f}s",
                flush=True,
            )
        if processed_sessions != query_rows:
            raise AssertionError(
                f"Fusion consumed {processed_sessions} sessions; expected {query_rows}"
            )
        expected_rows = query_rows * len(TARGET_TYPES) * k
        if total_rows != expected_rows:
            raise AssertionError(f"Fusion wrote {total_rows} rows; expected {expected_rows}")

        fusion_seconds = time.perf_counter() - fusion_started
        evaluation_started = time.perf_counter()
        metrics, complementarity = _evaluate_and_analyze(
            connection,
            output_files,
            labels,
            eval_ks=eval_ks,
            candidate_k=k,
            target_weights=target_weights,
        )
        evaluation_seconds = time.perf_counter() - evaluation_started
        report = {
            "dataset": dataset_name,
            "output": str(destination),
            "inputs": {
                "queries": str(queries),
                "query_rows": query_rows,
                "labels": str(labels),
                "label_rows": pq.ParquetFile(labels).metadata.num_rows,
                "popular": str(popular),
                "revisit": str(revisit),
                "type_covis": str(Path(type_covis_path).resolve()),
                "buy2buy": str(Path(buy2buy_path).resolve()),
                "time_covis": str(Path(time_covis_path).resolve()),
                "dssm": str(Path(dssm_path).resolve()),
            },
            "configuration": {
                "candidate_k": k,
                "session_buckets": session_buckets,
                "selection": "rank-wise round-robin over five personalized sources",
                "personalized_source_order": [
                    "revisit",
                    "type_covis",
                    "buy2buy",
                    "time_covis",
                    "dssm",
                ],
                "fallback": "target-specific Popular",
                "source_ids": SOURCE_IDS,
            },
            "candidates": {
                "rows": total_rows,
                "query_groups": query_rows * len(TARGET_TYPES),
                "average_per_group": k,
                "files": len(output_files),
                "disk_bytes": _directory_size(candidate_dir),
            },
            "metrics": metrics,
            "complementarity": complementarity,
            "assertions": {
                "passed": invalid_groups == 0 and invalid_features == 0,
                "invalid_groups": invalid_groups,
                "invalid_features": invalid_features,
                "rows_match_exact_budget": total_rows == expected_rows,
                "no_duplicate_aids_within_group": invalid_groups == 0,
                "source_features_consistent": invalid_features == 0,
            },
            "partitioning": partition_report,
            "timing": {
                "fusion_seconds": round(fusion_seconds, 6),
                "evaluation_seconds": round(evaluation_seconds, 6),
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
    parser = argparse.ArgumentParser(description="Fuse six OTTO recall sources.")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--popular", type=Path, required=True)
    parser.add_argument("--revisit", type=Path, required=True)
    parser.add_argument("--type-covis", type=Path, required=True)
    parser.add_argument("--buy2buy", type=Path, required=True)
    parser.add_argument("--time-covis", type=Path, required=True)
    parser.add_argument("--dssm", type=Path, required=True)
    parser.add_argument("--dataset-name", choices=("ranker", "valid"), required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--candidate-k", type=int)
    parser.add_argument("--session-buckets", type=int, default=64)
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
        output_dir = Path(experiment_dir) / "candidates" / args.dataset_name
    else:
        raise ValueError("Provide --output-dir or run through the experiment runner")
    report = fuse_candidates(
        args.queries,
        args.labels,
        args.popular,
        args.revisit,
        args.type_covis,
        args.buy2buy,
        args.time_covis,
        args.dssm,
        output_dir,
        resolve_project_config(config_path),
        dataset_name=args.dataset_name,
        candidate_k=args.candidate_k,
        session_buckets=args.session_buckets,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
