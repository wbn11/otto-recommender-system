"""Build point-in-time snapshots plus prefix queries and future labels."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import duckdb
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.schemas import EVENT_SCHEMA
from utils.config import resolve_project_config


DAY_MS = 24 * 60 * 60 * 1000
SESSION_COLUMNS = (
    "session",
    "start_ts",
    "end_ts",
    "event_count",
    "click_count",
    "cart_count",
    "order_count",
)


@dataclass(frozen=True)
class TimeBoundaries:
    data_max_ts: int
    window_end: int
    t1: int
    t2: int
    ranker_window_days: int
    validation_window_days: int
    shift_days: int


def compute_boundaries(
    data_max_ts: int,
    ranker_window_days: int,
    validation_window_days: int,
    shift_days: int = 0,
) -> TimeBoundaries:
    if ranker_window_days <= 0 or validation_window_days <= 0:
        raise ValueError("Window lengths must be positive")
    if shift_days < 0:
        raise ValueError("shift_days must be non-negative")
    window_end = data_max_ts - shift_days * DAY_MS
    t2 = window_end - validation_window_days * DAY_MS
    t1 = t2 - ranker_window_days * DAY_MS
    return TimeBoundaries(
        data_max_ts=data_max_ts,
        window_end=window_end,
        t1=t1,
        t2=t2,
        ranker_window_days=ranker_window_days,
        validation_window_days=validation_window_days,
        shift_days=shift_days,
    )


def classify_supervised_session(start_ts: int, boundaries: TimeBoundaries) -> str:
    """Classify a session by start time; snapshot membership is event-based."""

    if start_ts < boundaries.t1:
        return "history_only"
    if start_ts < boundaries.t2:
        return "ranker_train"
    if start_ts <= boundaries.window_end:
        return "final_valid"
    return "outside_window"


def _quote(value: str | Path) -> str:
    return str(value).replace("'", "''")


def _iso_utc(timestamp_ms: int | None) -> str | None:
    if timestamp_ms is None:
        return None
    return datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc).isoformat()


def _parquet_rows(path: Path) -> int:
    return pq.ParquetFile(path).metadata.num_rows


def _typed_batch(batch: pa.RecordBatch) -> pa.RecordBatch:
    arrays = [
        pc.cast(batch.column(batch.schema.get_field_index(field.name)), field.type)
        for field in EVENT_SCHEMA
    ]
    return pa.RecordBatch.from_arrays(arrays, schema=EVENT_SCHEMA)


class _SnapshotPartWriter:
    """Write one filtered part while preserving source row order."""

    def __init__(self, path: Path, compression: str, row_group_size: int):
        self.path = path
        self.temporary = path.with_suffix(".parquet.tmp")
        self.compression = compression
        self.row_group_size = row_group_size
        self.writer: pq.ParquetWriter | None = None
        self.rows = 0
        self.min_ts: int | None = None
        self.max_ts: int | None = None

    def append(self, batch: pa.RecordBatch) -> None:
        if batch.num_rows == 0:
            return
        typed = _typed_batch(batch)
        if self.writer is None:
            self.writer = pq.ParquetWriter(
                self.temporary,
                EVENT_SCHEMA,
                compression=self.compression,
                use_dictionary=False,
                write_statistics=True,
            )
        self.writer.write_batch(typed, row_group_size=self.row_group_size)
        ts_column = typed.column(typed.schema.get_field_index("ts"))
        batch_min = int(pc.min(ts_column).as_py())
        batch_max = int(pc.max(ts_column).as_py())
        self.min_ts = batch_min if self.min_ts is None else min(self.min_ts, batch_min)
        self.max_ts = batch_max if self.max_ts is None else max(self.max_ts, batch_max)
        self.rows += typed.num_rows

    def close(self) -> dict[str, int | None] | None:
        if self.writer is None:
            return None
        self.writer.close()
        self.temporary.replace(self.path)
        return {"rows": self.rows, "min_ts": self.min_ts, "max_ts": self.max_ts}


def _combine_time_and_session_filter(
    batch: pa.RecordBatch,
    cutoff: int,
    selected_sessions: pa.Array | None,
) -> pa.Array:
    ts_column = batch.column(batch.schema.get_field_index("ts"))
    mask = pc.less(ts_column, pa.scalar(cutoff, pa.int64()))
    if selected_sessions is not None:
        session_column = batch.column(batch.schema.get_field_index("session"))
        mask = pc.and_(mask, pc.is_in(session_column, value_set=selected_sessions))
    return mask


def _write_event_snapshots(
    event_files: list[Path],
    ranker_output: Path,
    valid_output: Path,
    *,
    t1: int,
    t2: int,
    ranker_selected_sessions: set[int] | None,
    valid_selected_sessions: set[int] | None,
    compression: str,
    row_group_size: int,
) -> dict[str, dict[str, Any]]:
    ranker_output.mkdir(parents=True, exist_ok=False)
    valid_output.mkdir(parents=True, exist_ok=False)
    ranker_values = (
        pa.array(sorted(ranker_selected_sessions), type=pa.int64())
        if ranker_selected_sessions is not None
        else None
    )
    valid_values = (
        pa.array(sorted(valid_selected_sessions), type=pa.int64())
        if valid_selected_sessions is not None
        else None
    )
    totals: dict[str, dict[str, Any]] = {
        "ranker_snapshot": {"event_rows": 0, "files": 0, "min_ts": None, "max_ts": None},
        "valid_snapshot": {"event_rows": 0, "files": 0, "min_ts": None, "max_ts": None},
    }

    for source_path in event_files:
        ranker_writer = _SnapshotPartWriter(
            ranker_output / source_path.name,
            compression,
            row_group_size,
        )
        valid_writer = _SnapshotPartWriter(
            valid_output / source_path.name,
            compression,
            row_group_size,
        )
        parquet_file = pq.ParquetFile(source_path)
        for batch in parquet_file.iter_batches(
            batch_size=row_group_size,
            columns=EVENT_SCHEMA.names,
            use_threads=True,
        ):
            ranker_mask = _combine_time_and_session_filter(batch, t1, ranker_values)
            valid_mask = _combine_time_and_session_filter(batch, t2, valid_values)
            ranker_writer.append(batch.filter(ranker_mask))
            valid_writer.append(batch.filter(valid_mask))

        for name, result in (
            ("ranker_snapshot", ranker_writer.close()),
            ("valid_snapshot", valid_writer.close()),
        ):
            if result is None:
                continue
            totals[name]["event_rows"] += int(result["rows"])
            totals[name]["files"] += 1
            for key, function in (("min_ts", min), ("max_ts", max)):
                value = int(result[key])
                current = totals[name][key]
                totals[name][key] = value if current is None else function(current, value)

    if totals["ranker_snapshot"]["event_rows"] == 0:
        raise ValueError("ranker_snapshot is empty")
    if totals["valid_snapshot"]["event_rows"] == 0:
        raise ValueError("valid_snapshot is empty")
    return totals


def _selected_ids_sql(predicate: str, max_sessions: int | None, seed: int) -> str:
    sampling = ""
    if max_sessions is not None:
        sampling = f"ORDER BY hash(session, {int(seed)}), session LIMIT {int(max_sessions)}"
    return f"SELECT session FROM source_sessions WHERE {predicate} {sampling}"


def _write_query_and_labels(
    connection: duckdb.DuckDBPyConnection,
    dataset_name: str,
    id_relation: str,
    event_predicate: str,
    query_path: Path,
    label_path: Path,
    history_ratio: float,
    compression: str,
    row_group_size: int,
) -> dict[str, Any]:
    if not 0.0 < history_ratio < 1.0:
        raise ValueError("history_ratio must be between 0 and 1")
    relation = f"{dataset_name}_sequences"
    connection.execute(
        f"""
        CREATE TEMP TABLE {relation} AS
        WITH grouped AS (
            SELECT
                e.session::BIGINT AS session,
                list(
                    struct_pack(
                        aid := e.aid,
                        ts := e.ts,
                        event_type := e.event_type
                    )
                    ORDER BY e.ts, e.file_row_number
                ) AS events
            FROM source_events e
            INNER JOIN {id_relation} selected USING (session)
            WHERE {event_predicate}
            GROUP BY e.session
            HAVING count(*) >= 2
        )
        SELECT
            session,
            events,
            greatest(
                1,
                least(
                    array_length(events) - 1,
                    floor(array_length(events) * {float(history_ratio)})::INTEGER
                )
            )::INTEGER AS split_index
        FROM grouped
        """
    )
    connection.execute(
        f"""
        COPY (
            SELECT
                session,
                list_transform(
                    list_slice(events, 1, split_index), lambda event: event.aid
                ) AS aids,
                list_transform(
                    list_slice(events, 1, split_index), lambda event: event.ts
                ) AS timestamps,
                list_transform(
                    list_slice(events, 1, split_index), lambda event: event.event_type
                ) AS event_types
            FROM {relation}
            ORDER BY session
        ) TO '{_quote(query_path)}' (
            FORMAT PARQUET,
            COMPRESSION {compression.upper()},
            ROW_GROUP_SIZE {int(row_group_size)}
        )
        """
    )
    connection.execute(
        f"""
        COPY (
            SELECT
                session::BIGINT AS session,
                future_event.event_type::TINYINT AS target_type,
                future_event.aid::BIGINT AS aid
            FROM (
                SELECT
                    session,
                    unnest(
                        list_slice(events, split_index + 1, array_length(events))
                    ) AS future_event
                FROM {relation}
            )
            GROUP BY session, future_event.event_type, future_event.aid
            ORDER BY session, future_event.event_type, future_event.aid
        ) TO '{_quote(label_path)}' (
            FORMAT PARQUET,
            COMPRESSION {compression.upper()},
            ROW_GROUP_SIZE {int(row_group_size)}
        )
        """
    )
    sequence_stats = connection.execute(
        f"""
        SELECT
            count(*)::BIGINT,
            min(split_index)::BIGINT,
            max(split_index)::BIGINT,
            avg(split_index)::DOUBLE,
            min(array_length(events) - split_index)::BIGINT,
            max(array_length(events) - split_index)::BIGINT,
            avg(array_length(events) - split_index)::DOUBLE
        FROM {relation}
        """
    ).fetchone()
    if int(sequence_stats[0]) == 0:
        raise ValueError(f"{dataset_name} contains no sessions with at least two events")
    query_rows = _parquet_rows(query_path)
    label_rows = _parquet_rows(label_path)
    connection.execute(f"DROP TABLE {relation}")
    return {
        "written_sessions": query_rows,
        "label_rows": label_rows,
        "query_length": {
            "min": int(sequence_stats[1]),
            "max": int(sequence_stats[2]),
            "mean": round(float(sequence_stats[3]), 6),
        },
        "future_length": {
            "min": int(sequence_stats[4]),
            "max": int(sequence_stats[5]),
            "mean": round(float(sequence_stats[6]), 6),
        },
    }


def build_time_splits(
    events_dir: str | Path,
    sessions_dir: str | Path,
    output_dir: str | Path,
    snapshot_dir: str | Path,
    *,
    ranker_window_days: int = 7,
    validation_window_days: int = 7,
    history_ratio: float = 0.8,
    shift_days: int = 0,
    max_sessions_per_window: int | None = None,
    session_hash_seed: int = 2024,
    compression: str = "zstd",
    row_group_size: int = 250_000,
    workers: int = 8,
    memory_limit_gb: int = 8,
) -> dict[str, Any]:
    events_source = Path(events_dir).resolve()
    sessions_source = Path(sessions_dir).resolve()
    target = Path(output_dir).resolve()
    snapshot_target = Path(snapshot_dir).resolve()
    event_files = sorted(events_source.glob("part-*.parquet"))
    session_files = sorted(sessions_source.glob("part-*.parquet"))
    if not event_files:
        raise FileNotFoundError(f"No event Parquet shards found in {events_source}")
    if not session_files:
        raise FileNotFoundError(f"No session Parquet shards found in {sessions_source}")
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite existing splits: {target}")
    if snapshot_target.exists():
        raise FileExistsError(f"Refusing to overwrite existing snapshots: {snapshot_target}")
    if max_sessions_per_window is not None and max_sessions_per_window <= 0:
        raise ValueError("max_sessions_per_window must be positive or null")

    target.parent.mkdir(parents=True, exist_ok=True)
    snapshot_target.parent.mkdir(parents=True, exist_ok=True)
    token = f"{os.getpid()}-{uuid.uuid4().hex[:8]}"
    target_staging = target.parent / f".{target.name}.building-{token}"
    snapshot_staging = snapshot_target.parent / f".{snapshot_target.name}.building-{token}"
    target_staging.mkdir(parents=True, exist_ok=False)
    snapshot_staging.mkdir(parents=True, exist_ok=False)
    temp_dir = target_staging / "duckdb_tmp"
    temp_dir.mkdir()
    started = time.perf_counter()
    connection = duckdb.connect()
    snapshot_moved = False

    try:
        connection.execute(f"SET threads={int(workers)}")
        connection.execute(f"SET memory_limit='{int(memory_limit_gb)}GB'")
        connection.execute(f"SET temp_directory='{_quote(temp_dir)}'")
        connection.execute("SET preserve_insertion_order=false")
        event_glob = _quote(events_source / "part-*.parquet")
        session_glob = _quote(sessions_source / "part-*.parquet")
        connection.execute(
            f"CREATE TEMP VIEW source_events AS SELECT "
            f"session, aid, ts, event_type, file_row_number "
            f"FROM read_parquet('{event_glob}', union_by_name=true, "
            f"file_row_number=true)"
        )
        connection.execute(
            f"CREATE TEMP VIEW source_sessions AS SELECT {', '.join(SESSION_COLUMNS)} "
            f"FROM read_parquet('{session_glob}', union_by_name=true)"
        )

        source_summary = connection.execute(
            """
            SELECT count(*), count(DISTINCT session), min(start_ts), max(end_ts),
                   sum(CASE WHEN start_ts > end_ts THEN 1 ELSE 0 END)
            FROM source_sessions
            """
        ).fetchone()
        total_sessions, unique_sessions, min_start_ts, max_end_ts, invalid_time_order = source_summary
        if total_sessions == 0:
            raise ValueError("Session dataset is empty")
        if unique_sessions != total_sessions:
            raise AssertionError("Duplicate session ids found in canonical session data")
        if invalid_time_order:
            raise AssertionError(f"Found {invalid_time_order} sessions with start_ts > end_ts")

        boundaries = compute_boundaries(
            int(max_end_ts), ranker_window_days, validation_window_days, shift_days
        )
        ranker_predicate = f"start_ts >= {boundaries.t1} AND start_ts < {boundaries.t2}"
        final_predicate = f"start_ts >= {boundaries.t2} AND start_ts <= {boundaries.window_end}"
        connection.execute(
            "CREATE TEMP TABLE ranker_target_ids AS "
            + _selected_ids_sql(ranker_predicate, max_sessions_per_window, session_hash_seed)
        )
        connection.execute(
            "CREATE TEMP TABLE final_valid_target_ids AS "
            + _selected_ids_sql(final_predicate, max_sessions_per_window, session_hash_seed)
        )

        eligible_ranker = int(connection.execute(
            f"SELECT count(*) FROM source_sessions WHERE {ranker_predicate}"
        ).fetchone()[0])
        eligible_final = int(connection.execute(
            f"SELECT count(*) FROM source_sessions WHERE {final_predicate}"
        ).fetchone()[0])
        selected_ranker = int(connection.execute("SELECT count(*) FROM ranker_target_ids").fetchone()[0])
        selected_final = int(connection.execute("SELECT count(*) FROM final_valid_target_ids").fetchone()[0])

        ranker_query_path = target_staging / "ranker_queries.parquet"
        ranker_label_path = target_staging / "ranker_labels.parquet"
        valid_query_path = target_staging / "valid_queries.parquet"
        valid_label_path = target_staging / "valid_labels.parquet"
        ranker_data = _write_query_and_labels(
            connection,
            "ranker",
            "ranker_target_ids",
            f"e.ts < {boundaries.t2}",
            ranker_query_path,
            ranker_label_path,
            history_ratio,
            compression,
            row_group_size,
        )
        valid_data = _write_query_and_labels(
            connection,
            "valid",
            "final_valid_target_ids",
            f"e.ts <= {boundaries.window_end}",
            valid_query_path,
            valid_label_path,
            history_ratio,
            compression,
            row_group_size,
        )

        ranker_snapshot_sessions: set[int] | None = None
        valid_snapshot_sessions: set[int] | None = None
        if max_sessions_per_window is not None:
            ranker_snapshot_sessions = {
                int(row[0]) for row in connection.execute(_selected_ids_sql(
                    f"start_ts < {boundaries.t1}", max_sessions_per_window, session_hash_seed
                )).fetchall()
            }
            valid_snapshot_sessions = {
                int(row[0]) for row in connection.execute(_selected_ids_sql(
                    f"start_ts < {boundaries.t2}", max_sessions_per_window, session_hash_seed
                )).fetchall()
            }

        snapshot_stats = _write_event_snapshots(
            event_files,
            snapshot_staging / "ranker_snapshot" / "events",
            snapshot_staging / "valid_snapshot" / "events",
            t1=boundaries.t1,
            t2=boundaries.t2,
            ranker_selected_sessions=ranker_snapshot_sessions,
            valid_selected_sessions=valid_snapshot_sessions,
            compression=compression,
            row_group_size=row_group_size,
        )

        boundary_counts = connection.execute(
            f"""
            SELECT
                sum(CASE WHEN start_ts < {boundaries.t1} AND end_ts >= {boundaries.t1} THEN 1 ELSE 0 END),
                sum(CASE WHEN start_ts < {boundaries.t2} AND end_ts >= {boundaries.t2} THEN 1 ELSE 0 END),
                sum(CASE WHEN start_ts <= {boundaries.window_end} AND end_ts > {boundaries.window_end}
                         THEN 1 ELSE 0 END)
            FROM source_sessions
            """
        ).fetchone()

        violations = {
            "ranker_snapshot_at_or_after_t1": int(
                snapshot_stats["ranker_snapshot"]["max_ts"] >= boundaries.t1
            ),
            "valid_snapshot_at_or_after_t2": int(
                snapshot_stats["valid_snapshot"]["max_ts"] >= boundaries.t2
            ),
            "ranker_query_invalid": int(connection.execute(
                f"""SELECT count(*) FROM read_parquet('{_quote(ranker_query_path)}')
                    WHERE array_length(aids) < 1
                       OR array_length(aids) != array_length(timestamps)
                       OR array_length(aids) != array_length(event_types)
                       OR list_min(timestamps) < {boundaries.t1}
                       OR list_max(timestamps) >= {boundaries.t2}"""
            ).fetchone()[0]),
            "valid_query_invalid": int(connection.execute(
                f"""SELECT count(*) FROM read_parquet('{_quote(valid_query_path)}')
                    WHERE array_length(aids) < 1
                       OR array_length(aids) != array_length(timestamps)
                       OR array_length(aids) != array_length(event_types)
                       OR list_min(timestamps) < {boundaries.t2}
                       OR list_max(timestamps) > {boundaries.window_end}"""
            ).fetchone()[0]),
            "ranker_valid_session_overlap": int(connection.execute(
                f"""SELECT count(*) FROM read_parquet('{_quote(ranker_query_path)}') r
                    INNER JOIN read_parquet('{_quote(valid_query_path)}') v USING (session)"""
            ).fetchone()[0]),
            "ranker_label_duplicates": int(connection.execute(
                f"""SELECT count(*) FROM (
                    SELECT session, target_type, aid, count(*) AS n
                    FROM read_parquet('{_quote(ranker_label_path)}')
                    GROUP BY session, target_type, aid HAVING count(*) > 1)"""
            ).fetchone()[0]),
            "valid_label_duplicates": int(connection.execute(
                f"""SELECT count(*) FROM (
                    SELECT session, target_type, aid, count(*) AS n
                    FROM read_parquet('{_quote(valid_label_path)}')
                    GROUP BY session, target_type, aid HAVING count(*) > 1)"""
            ).fetchone()[0]),
            "ranker_label_without_query": int(connection.execute(
                f"""SELECT count(*) FROM (
                    SELECT DISTINCT session FROM read_parquet('{_quote(ranker_label_path)}')) l
                    LEFT JOIN read_parquet('{_quote(ranker_query_path)}') q USING (session)
                    WHERE q.session IS NULL"""
            ).fetchone()[0]),
            "valid_label_without_query": int(connection.execute(
                f"""SELECT count(*) FROM (
                    SELECT DISTINCT session FROM read_parquet('{_quote(valid_label_path)}')) l
                    LEFT JOIN read_parquet('{_quote(valid_query_path)}') q USING (session)
                    WHERE q.session IS NULL"""
            ).fetchone()[0]),
            "ranker_query_without_label": int(connection.execute(
                f"""SELECT count(*) FROM read_parquet('{_quote(ranker_query_path)}') q
                    LEFT JOIN (SELECT DISTINCT session FROM read_parquet(
                        '{_quote(ranker_label_path)}')) l USING (session)
                    WHERE l.session IS NULL"""
            ).fetchone()[0]),
            "valid_query_without_label": int(connection.execute(
                f"""SELECT count(*) FROM read_parquet('{_quote(valid_query_path)}') q
                    LEFT JOIN (SELECT DISTINCT session FROM read_parquet(
                        '{_quote(valid_label_path)}')) l USING (session)
                    WHERE l.session IS NULL"""
            ).fetchone()[0]),
            "invalid_target_type": int(connection.execute(
                f"""SELECT
                    (SELECT count(*) FROM read_parquet('{_quote(ranker_label_path)}')
                        WHERE target_type NOT BETWEEN 1 AND 3)
                    +
                    (SELECT count(*) FROM read_parquet('{_quote(valid_label_path)}')
                        WHERE target_type NOT BETWEEN 1 AND 3)"""
            ).fetchone()[0]),
        }
        if any(violations.values()):
            raise AssertionError(f"Leakage assertion failed: {violations}")

        for details in snapshot_stats.values():
            details["min_ts_iso"] = _iso_utc(details["min_ts"])
            details["max_ts_iso"] = _iso_utc(details["max_ts"])
        snapshot_stats["ranker_snapshot"].update({
            "rule": f"ts < {boundaries.t1}",
            "cutoff_ts": boundaries.t1,
            "cutoff_ts_iso": _iso_utc(boundaries.t1),
            "sampled_sessions": len(ranker_snapshot_sessions) if ranker_snapshot_sessions is not None else None,
        })
        snapshot_stats["valid_snapshot"].update({
            "rule": f"ts < {boundaries.t2}",
            "cutoff_ts": boundaries.t2,
            "cutoff_ts_iso": _iso_utc(boundaries.t2),
            "sampled_sessions": len(valid_snapshot_sessions) if valid_snapshot_sessions is not None else None,
        })

        report = {
            "strategy": "point_in_time",
            "sources": {
                "events": str(events_source), "event_files": len(event_files),
                "sessions": str(sessions_source), "session_files": len(session_files),
            },
            "output": str(target),
            "snapshot_output": str(snapshot_target),
            "total_sessions": int(total_sessions),
            "unique_sessions": int(unique_sessions),
            "sampled": max_sessions_per_window is not None,
            "max_sessions_per_window": max_sessions_per_window,
            "session_hash_seed": session_hash_seed,
            "history_ratio": history_ratio,
            "boundaries": {
                "data_min_ts": int(min_start_ts), "data_max_ts": boundaries.data_max_ts,
                "window_end": boundaries.window_end, "t1": boundaries.t1, "t2": boundaries.t2,
                "data_min_ts_iso": _iso_utc(int(min_start_ts)),
                "data_max_ts_iso": _iso_utc(boundaries.data_max_ts),
                "window_end_iso": _iso_utc(boundaries.window_end),
                "t1_iso": _iso_utc(boundaries.t1), "t2_iso": _iso_utc(boundaries.t2),
                "ranker_window_days": ranker_window_days,
                "validation_window_days": validation_window_days, "shift_days": shift_days,
            },
            "snapshots": snapshot_stats,
            "supervised_datasets": {
                "ranker": {
                    "rule": f"{boundaries.t1} <= session_start < {boundaries.t2}; events ts < T2",
                    "eligible_sessions": eligible_ranker, "selected_sessions": selected_ranker,
                    **ranker_data,
                    "dropped_after_truncation_lt_2_events": (
                        selected_ranker - ranker_data["written_sessions"]
                    ),
                },
                "valid": {
                    "rule": f"{boundaries.t2} <= session_start <= {boundaries.window_end}; events ts <= window_end",
                    "eligible_sessions": eligible_final, "selected_sessions": selected_final,
                    **valid_data,
                    "dropped_after_truncation_lt_2_events": (
                        selected_final - valid_data["written_sessions"]
                    ),
                },
            },
            "cross_boundary_sessions": {
                "cross_t1": int(boundary_counts[0]), "cross_t2": int(boundary_counts[1]),
                "cross_window_end": int(boundary_counts[2]),
                "policy": "Events before each cutoff enter that snapshot; pre-window sessions are not primary labels.",
            },
            "leakage_assertions": {"passed": not any(violations.values()), "violations": violations},
            "runtime_seconds": round(time.perf_counter() - started, 6),
        }
        (target_staging / "split_report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    except Exception:
        connection.close()
        shutil.rmtree(target_staging, ignore_errors=True)
        shutil.rmtree(snapshot_staging, ignore_errors=True)
        raise
    else:
        connection.close()

    shutil.rmtree(temp_dir, ignore_errors=True)
    try:
        snapshot_staging.replace(snapshot_target)
        snapshot_moved = True
        target_staging.replace(target)
    except Exception:
        if snapshot_moved:
            shutil.rmtree(snapshot_target, ignore_errors=True)
        shutil.rmtree(target_staging, ignore_errors=True)
        shutil.rmtree(snapshot_staging, ignore_errors=True)
        raise
    return report


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build OTTO point-in-time snapshots, prefix queries and future labels."
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument("--events-dir", type=Path, required=True)
    parser.add_argument("--sessions-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--snapshot-dir", type=Path)
    parser.add_argument("--window-name", default="primary")
    parser.add_argument("--shift-days", type=int, default=0)
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
        output_dir = Path(experiment_dir) / "data" / "splits" / args.window_name
    else:
        raise ValueError("Provide --output-dir or run through the experiment runner")
    if args.snapshot_dir:
        snapshot_dir = args.snapshot_dir
    elif experiment_dir:
        snapshot_dir = Path(experiment_dir) / "snapshots" / args.window_name
    else:
        raise ValueError("Provide --snapshot-dir or run through the experiment runner")

    report = build_time_splits(
        args.events_dir, args.sessions_dir, output_dir, snapshot_dir,
        ranker_window_days=config["split"]["ranker_window_days"],
        validation_window_days=config["split"]["validation_window_days"],
        history_ratio=config["split"]["history_ratio"],
        shift_days=args.shift_days,
        max_sessions_per_window=config["data"]["max_sessions_per_window"],
        session_hash_seed=config["data"]["session_hash_seed"],
        compression=config["runtime"]["parquet_compression"],
        row_group_size=config["runtime"]["parquet_row_group_size"],
        workers=config["runtime"]["workers"],
        memory_limit_gb=config["runtime"]["duckdb_memory_limit_gb"],
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
