"""Prepare deterministic vocabulary and session-sequence Parquet for DSSM training."""

from __future__ import annotations

import argparse
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
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.schemas import (
    DSSM_SEQUENCE_SCHEMA,
    EVENT_SCHEMA,
    ITEM_VOCAB_SCHEMA,
    PAD_ID,
    UNK_ID,
)
from utils.config import resolve_project_config


def _quote(value: str | Path) -> str:
    return str(value).replace("'", "''")


def _path(path: Path) -> str:
    return path.resolve().as_posix()


def _parquet_rows(paths: Sequence[Path]) -> int:
    return sum(pq.ParquetFile(path).metadata.num_rows for path in paths)


def _directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check_schema(path: Path, expected, name: str) -> None:
    actual = pq.read_schema(path)
    if actual.names != expected.names or actual.types != expected.types:
        raise AssertionError(f"Unexpected {name} schema in {path}: {actual}")


def _sql_path_list(paths: Sequence[Path]) -> str:
    return "[" + ",".join(f"'{_quote(_path(path))}'" for path in paths) + "]"


def prepare_dssm_data(
    snapshot_dir: str | Path,
    output_dir: str | Path,
    *,
    dataset_name: str,
    compression: str = "zstd",
    row_group_size: int = 250_000,
    workers: int = 8,
    memory_limit_gb: int = 8,
) -> dict[str, Any]:
    """Build vocab once, then group one snapshot shard at a time into sequences."""

    snapshot_source = Path(snapshot_dir).resolve()
    target = Path(output_dir).resolve()
    event_files = sorted(snapshot_source.glob("part-*.parquet"))
    if not event_files:
        event_files = sorted((snapshot_source / "events").glob("part-*.parquet"))
    if not event_files:
        raise FileNotFoundError(f"No snapshot event shards found in {snapshot_source}")
    for event_file in event_files:
        _check_schema(event_file, EVENT_SCHEMA, "event")
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite DSSM data: {target}")
    if workers <= 0 or memory_limit_gb <= 0 or row_group_size <= 0:
        raise ValueError("workers, memory_limit_gb and row_group_size must be positive")

    target.parent.mkdir(parents=True, exist_ok=True)
    token = f"{os.getpid()}-{uuid.uuid4().hex[:8]}"
    staging = target.parent / f".{target.name}.building-{token}"
    sequence_dir = staging / "sequences"
    temp_dir = staging / "duckdb_tmp"
    vocab_path = staging / "item_vocab.parquet"
    staging.mkdir(parents=True, exist_ok=False)
    sequence_dir.mkdir()
    temp_dir.mkdir()
    started = time.perf_counter()
    connection = duckdb.connect()

    try:
        connection.execute(f"SET threads={int(workers)}")
        connection.execute(f"SET memory_limit='{int(memory_limit_gb)}GB'")
        connection.execute(f"SET temp_directory='{_quote(_path(temp_dir))}'")
        connection.execute("SET preserve_insertion_order=false")
        event_paths = _sql_path_list(event_files)

        vocab_started = time.perf_counter()
        connection.execute(
            f"""
            COPY (
                WITH unique_items AS (
                    SELECT DISTINCT aid::BIGINT AS aid
                    FROM read_parquet({event_paths}, union_by_name=true)
                )
                SELECT
                    aid::BIGINT AS aid,
                    (row_number() OVER (ORDER BY aid) + 1)::INTEGER AS item_id
                FROM unique_items
                ORDER BY aid
            ) TO '{_quote(_path(vocab_path))}' (
                FORMAT PARQUET,
                COMPRESSION {compression.upper()},
                ROW_GROUP_SIZE {int(row_group_size)}
            )
            """
        )
        _check_schema(vocab_path, ITEM_VOCAB_SCHEMA, "item vocabulary")
        vocab_runtime = round(time.perf_counter() - vocab_started, 6)

        source_rows = _parquet_rows(event_files)
        source_sessions = 0
        single_event_sessions = 0
        source_min_ts: int | None = None
        source_max_ts: int | None = None
        sequence_started = time.perf_counter()
        for index, event_file in enumerate(event_files):
            shard_summary = connection.execute(
                f"""
                WITH lengths AS (
                    SELECT session, count(*)::BIGINT AS n
                    FROM read_parquet('{_quote(_path(event_file))}')
                    GROUP BY session
                )
                SELECT
                    count(*)::BIGINT,
                    sum(CASE WHEN n = 1 THEN 1 ELSE 0 END)::BIGINT
                FROM lengths
                """
            ).fetchone()
            min_ts, max_ts = connection.execute(
                f"""SELECT min(ts)::BIGINT, max(ts)::BIGINT
                     FROM read_parquet('{_quote(_path(event_file))}')"""
            ).fetchone()
            source_sessions += int(shard_summary[0])
            single_event_sessions += int(shard_summary[1])
            source_min_ts = int(min_ts) if source_min_ts is None else min(source_min_ts, int(min_ts))
            source_max_ts = int(max_ts) if source_max_ts is None else max(source_max_ts, int(max_ts))

            output_path = sequence_dir / f"part-{index:05d}.parquet"
            connection.execute(
                f"""
                COPY (
                    WITH encoded AS (
                        SELECT
                            events.session::BIGINT AS session,
                            vocab.item_id::INTEGER AS item_id,
                            events.event_type::TINYINT AS event_type,
                            events.ts::BIGINT AS ts,
                            events.file_row_number::BIGINT AS file_row_number
                        FROM read_parquet(
                            '{_quote(_path(event_file))}',
                            file_row_number=true
                        ) events
                        INNER JOIN read_parquet('{_quote(_path(vocab_path))}') vocab
                          ON events.aid = vocab.aid
                    )
                    SELECT
                        session::BIGINT AS session,
                        list(item_id ORDER BY ts, file_row_number)::INTEGER[] AS item_ids,
                        list(event_type ORDER BY ts, file_row_number)::TINYINT[] AS event_types
                    FROM encoded
                    GROUP BY session
                    HAVING count(*) >= 2
                    ORDER BY session
                ) TO '{_quote(_path(output_path))}' (
                    FORMAT PARQUET,
                    COMPRESSION {compression.upper()},
                    ROW_GROUP_SIZE {int(row_group_size)}
                )
                """
            )
            _check_schema(output_path, DSSM_SEQUENCE_SCHEMA, "DSSM sequence")
            print(
                f"[prepare-dssm-data] shard={index + 1}/{len(event_files)} "
                f"source={event_file.name} elapsed={time.perf_counter() - sequence_started:.1f}s",
                flush=True,
            )

        sequence_files = sorted(sequence_dir.glob("part-*.parquet"))
        sequence_glob = _path(sequence_dir / "part-*.parquet")
        sequence_summary = connection.execute(
            f"""
            WITH sequences AS (
                SELECT
                    session,
                    array_length(item_ids)::BIGINT AS sequence_length,
                    array_length(event_types)::BIGINT AS type_length,
                    list_min(item_ids)::INTEGER AS min_item_id,
                    list_max(item_ids)::INTEGER AS max_item_id,
                    list_min(event_types)::INTEGER AS min_event_type,
                    list_max(event_types)::INTEGER AS max_event_type
                FROM read_parquet('{_quote(sequence_glob)}', union_by_name=true)
            )
            SELECT
                count(*)::BIGINT,
                count(DISTINCT session)::BIGINT,
                sum(sequence_length)::BIGINT,
                sum(sequence_length - 1)::BIGINT,
                min(sequence_length)::BIGINT,
                max(sequence_length)::BIGINT,
                avg(sequence_length)::DOUBLE,
                quantile_cont(sequence_length, 0.50)::DOUBLE,
                quantile_cont(sequence_length, 0.90)::DOUBLE,
                quantile_cont(sequence_length, 0.95)::DOUBLE,
                quantile_cont(sequence_length, 0.99)::DOUBLE,
                sum(CASE WHEN sequence_length != type_length THEN 1 ELSE 0 END)::BIGINT,
                sum(CASE WHEN sequence_length < 2 THEN 1 ELSE 0 END)::BIGINT,
                sum(CASE WHEN min_item_id < 2 THEN 1 ELSE 0 END)::BIGINT,
                sum(CASE WHEN min_event_type < 1 OR max_event_type > 3 THEN 1 ELSE 0 END)::BIGINT,
                max(max_item_id)::INTEGER
            FROM sequences
            """
        ).fetchone()
        sequence_runtime = round(time.perf_counter() - sequence_started, 6)

        vocab_summary = connection.execute(
            f"""
            SELECT
                count(*)::BIGINT,
                count(DISTINCT aid)::BIGINT,
                count(DISTINCT item_id)::BIGINT,
                min(item_id)::INTEGER,
                max(item_id)::INTEGER,
                sum(CASE WHEN item_id IN ({PAD_ID}, {UNK_ID}) THEN 1 ELSE 0 END)::BIGINT
            FROM read_parquet('{_quote(_path(vocab_path))}')
            """
        ).fetchone()
        vocab_rows = int(vocab_summary[0])
        expected_max_item_id = vocab_rows + 1
        encoded_events = int(sequence_summary[2])
        sequence_rows = int(sequence_summary[0])
        violations = {
            "duplicate_vocab_aids": vocab_rows - int(vocab_summary[1]),
            "duplicate_item_ids": vocab_rows - int(vocab_summary[2]),
            "invalid_vocab_start": int(int(vocab_summary[3]) != 2),
            "non_contiguous_item_ids": int(int(vocab_summary[4]) != expected_max_item_id),
            "reserved_ids_in_vocab": int(vocab_summary[5]),
            "duplicate_sequence_sessions": sequence_rows - int(sequence_summary[1]),
            "misaligned_item_type_lists": int(sequence_summary[11]),
            "sequences_shorter_than_two": int(sequence_summary[12]),
            "reserved_ids_in_sequences": int(sequence_summary[13]),
            "invalid_event_types": int(sequence_summary[14]),
            "sequence_item_id_over_vocab": int(
                int(sequence_summary[15]) > expected_max_item_id
            ),
            "unexpected_sequence_session_count": int(
                sequence_rows != source_sessions - single_event_sessions
            ),
            "unexpected_encoded_event_count": int(
                encoded_events != source_rows - single_event_sessions
            ),
        }
        if any(violations.values()):
            raise AssertionError(f"Invalid DSSM data: {violations}")

        report = {
            "dataset": dataset_name,
            "snapshot": str(snapshot_source),
            "output": str(target),
            "reserved_item_ids": {"pad": PAD_ID, "unk": UNK_ID, "first_real": 2},
            "source": {
                "event_files": len(event_files),
                "event_rows": source_rows,
                "sessions": source_sessions,
                "single_event_sessions_dropped": single_event_sessions,
                "min_ts": source_min_ts,
                "max_ts": source_max_ts,
            },
            "vocabulary": {
                "path": "item_vocab.parquet",
                "real_items": vocab_rows,
                "embedding_rows": vocab_rows + 2,
                "min_item_id": int(vocab_summary[3]),
                "max_item_id": int(vocab_summary[4]),
                "sha256": _sha256(vocab_path),
                "disk_bytes": vocab_path.stat().st_size,
                "runtime_seconds": vocab_runtime,
            },
            "sequences": {
                "path": "sequences",
                "files": len(sequence_files),
                "sessions": sequence_rows,
                "events": encoded_events,
                "next_item_pairs": int(sequence_summary[3]),
                "length": {
                    "min": int(sequence_summary[4]),
                    "max": int(sequence_summary[5]),
                    "mean": round(float(sequence_summary[6]), 6),
                    "p50": float(sequence_summary[7]),
                    "p90": float(sequence_summary[8]),
                    "p95": float(sequence_summary[9]),
                    "p99": float(sequence_summary[10]),
                },
                "disk_bytes": _directory_size(sequence_dir),
                "runtime_seconds": sequence_runtime,
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

    shutil.rmtree(temp_dir, ignore_errors=True)
    staging.replace(target)
    return report


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare streaming DSSM training data.")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--snapshot-dir", type=Path, required=True)
    parser.add_argument("--dataset-name", choices=("ranker", "valid"), required=True)
    parser.add_argument("--output-dir", type=Path)
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
        output_dir = Path(experiment_dir) / "data" / "dssm" / args.dataset_name
    else:
        raise ValueError("Provide --output-dir or run through the experiment runner")

    report = prepare_dssm_data(
        args.snapshot_dir,
        output_dir,
        dataset_name=args.dataset_name,
        compression=config["runtime"]["parquet_compression"],
        row_group_size=config["runtime"]["parquet_row_group_size"],
        workers=config["runtime"]["workers"],
        memory_limit_gb=config["runtime"]["duckdb_memory_limit_gb"],
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
