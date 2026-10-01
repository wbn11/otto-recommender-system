"""Stream raw OTTO JSONL sessions into canonical partitioned Parquet datasets."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.schemas import EVENT_SCHEMA, EVENT_TYPE_TO_ID, SESSION_SCHEMA
from utils.config import resolve_project_config


ROOT = Path(__file__).resolve().parents[2]


def json_loader() -> Callable[[bytes], Any]:
    try:
        import orjson

        return orjson.loads
    except ImportError:
        return json.loads


class ParquetShardWriter:
    """Bounded in-memory row buffer that emits immutable Parquet shards."""

    def __init__(
        self,
        output_dir: Path,
        schema: pa.Schema,
        rows_per_file: int,
        compression: str,
        row_group_size: int = 250_000,
    ):
        if rows_per_file <= 0:
            raise ValueError("rows_per_file must be positive")
        self.output_dir = output_dir
        self.schema = schema
        self.rows_per_file = rows_per_file
        self.compression = compression
        self.row_group_size = row_group_size
        self.buffer = {field.name: [] for field in schema}
        self.buffered_rows = 0
        self.total_rows = 0
        self.part_count = 0
        output_dir.mkdir(parents=True, exist_ok=False)

    def append_columns(self, columns: dict[str, list[Any]]) -> None:
        lengths = {len(values) for values in columns.values()}
        if lengths == {0}:
            return
        if len(lengths) != 1 or set(columns) != set(self.buffer):
            raise ValueError("Columns must match the writer schema and have equal lengths")
        row_count = lengths.pop()
        for name, values in columns.items():
            self.buffer[name].extend(values)
        self.buffered_rows += row_count
        if self.buffered_rows >= self.rows_per_file:
            self.flush()

    def append_row(self, row: dict[str, Any]) -> None:
        if set(row) != set(self.buffer):
            raise ValueError("Row must match the writer schema")
        for name, value in row.items():
            self.buffer[name].append(value)
        self.buffered_rows += 1
        if self.buffered_rows >= self.rows_per_file:
            self.flush()

    def flush(self) -> None:
        if not self.buffered_rows:
            return
        table = pa.Table.from_pydict(self.buffer, schema=self.schema)
        path = self.output_dir / f"part-{self.part_count:05d}.parquet"
        temporary = path.with_suffix(".parquet.tmp")
        pq.write_table(
            table,
            temporary,
            compression=self.compression,
            row_group_size=self.row_group_size,
            use_dictionary=False,
            write_statistics=True,
        )
        temporary.replace(path)
        self.total_rows += self.buffered_rows
        self.part_count += 1
        self.buffer = {field.name: [] for field in self.schema}
        self.buffered_rows = 0

    def close(self) -> None:
        self.flush()


@dataclass
class IngestionResult:
    output_dir: Path
    report: dict[str, Any]


def _empty_event_columns() -> dict[str, list[Any]]:
    return {field.name: [] for field in EVENT_SCHEMA}


def _histogram_quantile(histogram: Counter[int], total: int, quantile: float) -> float | None:
    """Return an exact linearly interpolated quantile from an integer histogram."""

    if total == 0:
        return None
    position = (total - 1) * quantile
    lower_rank = int(position)
    upper_rank = lower_rank if position.is_integer() else lower_rank + 1
    values: dict[int, int] = {}
    cumulative = 0
    for value, frequency in sorted(histogram.items()):
        next_cumulative = cumulative + frequency
        for rank in (lower_rank, upper_rank):
            if rank not in values and cumulative <= rank < next_cumulative:
                values[rank] = value
        if len(values) == (1 if lower_rank == upper_rank else 2):
            break
        cumulative = next_cumulative
    fraction = position - lower_rank
    return round(values[lower_rank] + (values[upper_rank] - values[lower_rank]) * fraction, 6)


def _session_length_summary(histogram: Counter[int], total: int, events: int) -> dict[str, Any]:
    if total == 0:
        return {
            "min": None,
            "max": None,
            "mean": None,
            "p50": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "buckets": {},
        }

    buckets = {
        "1": 0,
        "2": 0,
        "3-5": 0,
        "6-10": 0,
        "11-20": 0,
        "21-50": 0,
        "51-100": 0,
        "101+": 0,
    }
    for length, frequency in histogram.items():
        if length == 1:
            bucket = "1"
        elif length == 2:
            bucket = "2"
        elif length <= 5:
            bucket = "3-5"
        elif length <= 10:
            bucket = "6-10"
        elif length <= 20:
            bucket = "11-20"
        elif length <= 50:
            bucket = "21-50"
        elif length <= 100:
            bucket = "51-100"
        else:
            bucket = "101+"
        buckets[bucket] += frequency

    return {
        "min": min(histogram),
        "max": max(histogram),
        "mean": round(events / total, 6),
        "p50": _histogram_quantile(histogram, total, 0.50),
        "p90": _histogram_quantile(histogram, total, 0.90),
        "p95": _histogram_quantile(histogram, total, 0.95),
        "p99": _histogram_quantile(histogram, total, 0.99),
        "buckets": buckets,
    }


def ingest_jsonl(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    rows_per_file: int = 2_000_000,
    session_rows_per_file: int = 250_000,
    row_group_size: int = 250_000,
    compression: str = "zstd",
    max_sessions: int | None = None,
    progress_every: int = 100_000,
) -> IngestionResult:
    """Ingest JSONL without materializing the complete event table."""

    source = Path(input_path).resolve()
    target = Path(output_dir).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite existing dataset: {target}")

    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.parent / f".{target.name}.ingesting-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    staging.mkdir(parents=True, exist_ok=False)
    event_writer = ParquetShardWriter(
        staging / "events",
        EVENT_SCHEMA,
        rows_per_file,
        compression,
        row_group_size,
    )
    session_writer = ParquetShardWriter(
        staging / "sessions",
        SESSION_SCHEMA,
        session_rows_per_file,
        compression,
        row_group_size,
    )

    loads = json_loader()
    started = time.perf_counter()
    counts: Counter[str] = Counter()
    session_length_histogram: Counter[int] = Counter()
    unique_items: set[int] = set()
    seen_sessions: set[int] = set()
    min_ts: int | None = None
    max_ts: int | None = None
    source_size = source.stat().st_size

    with source.open("rb") as handle:
        for line in handle:
            if max_sessions is not None and counts["sessions_written"] >= max_sessions:
                break
            counts["input_lines"] += 1
            counts["bytes_read"] += len(line)
            try:
                record = loads(line)
            except (json.JSONDecodeError, TypeError, ValueError):
                counts["invalid_json_lines"] += 1
                continue
            if not isinstance(record, dict) or "session" not in record or "events" not in record:
                counts["invalid_session_records"] += 1
                continue
            try:
                session = int(record["session"])
            except (TypeError, ValueError):
                counts["invalid_session_records"] += 1
                continue
            if session in seen_sessions:
                counts["duplicate_session_ids"] += 1
            else:
                seen_sessions.add(session)

            raw_events = record["events"]
            if not isinstance(raw_events, list):
                counts["invalid_session_records"] += 1
                continue

            columns = _empty_event_columns()
            type_counts: Counter[int] = Counter()
            session_event_keys: set[tuple[int, int, int]] = set()
            session_min_ts: int | None = None
            session_max_ts: int | None = None

            for event in raw_events:
                counts["raw_events"] += 1
                if not isinstance(event, dict):
                    counts["invalid_events"] += 1
                    continue
                event_type = EVENT_TYPE_TO_ID.get(event.get("type"))
                if event_type is None:
                    counts["invalid_event_types"] += 1
                    continue
                try:
                    aid = int(event["aid"])
                    timestamp = int(event["ts"])
                except (KeyError, TypeError, ValueError):
                    counts["invalid_events"] += 1
                    continue
                if aid < 0 or timestamp < 0:
                    counts["invalid_events"] += 1
                    continue

                event_key = (aid, timestamp, event_type)
                if event_key in session_event_keys:
                    counts["duplicate_events"] += 1
                else:
                    session_event_keys.add(event_key)

                columns["session"].append(session)
                columns["aid"].append(aid)
                columns["ts"].append(timestamp)
                columns["event_type"].append(event_type)
                unique_items.add(aid)
                type_counts[event_type] += 1
                session_min_ts = timestamp if session_min_ts is None else min(session_min_ts, timestamp)
                session_max_ts = timestamp if session_max_ts is None else max(session_max_ts, timestamp)

            event_count = len(columns["aid"])
            if event_count == 0:
                counts["empty_sessions"] += 1
                continue

            event_writer.append_columns(columns)
            session_writer.append_row({
                "session": session,
                "start_ts": session_min_ts,
                "end_ts": session_max_ts,
                "event_count": event_count,
                "click_count": type_counts[EVENT_TYPE_TO_ID["clicks"]],
                "cart_count": type_counts[EVENT_TYPE_TO_ID["carts"]],
                "order_count": type_counts[EVENT_TYPE_TO_ID["orders"]],
            })
            counts["sessions_written"] += 1
            counts["events_written"] += event_count
            session_length_histogram[event_count] += 1
            counts["click_events"] += type_counts[EVENT_TYPE_TO_ID["clicks"]]
            counts["cart_events"] += type_counts[EVENT_TYPE_TO_ID["carts"]]
            counts["order_events"] += type_counts[EVENT_TYPE_TO_ID["orders"]]
            min_ts = session_min_ts if min_ts is None else min(min_ts, session_min_ts)
            max_ts = session_max_ts if max_ts is None else max(max_ts, session_max_ts)

            if progress_every and counts["sessions_written"] % progress_every == 0:
                elapsed = max(time.perf_counter() - started, 1e-9)
                input_percent = 100.0 * counts["bytes_read"] / source_size
                print(
                    "[ingest] "
                    f"sessions={counts['sessions_written']:,} "
                    f"events={counts['events_written']:,} "
                    f"input={input_percent:.2f}% "
                    f"event_parts={event_writer.part_count} "
                    f"elapsed={elapsed:.1f}s "
                    f"sessions_per_second={counts['sessions_written'] / elapsed:,.1f}",
                    flush=True,
                )

    event_writer.close()
    session_writer.close()
    for name in (
        "input_lines",
        "bytes_read",
        "sessions_written",
        "raw_events",
        "events_written",
        "click_events",
        "cart_events",
        "order_events",
        "invalid_json_lines",
        "invalid_session_records",
        "invalid_events",
        "invalid_event_types",
        "duplicate_session_ids",
        "duplicate_events",
        "empty_sessions",
    ):
        counts[name] += 0
    report = {
        "source": str(source),
        "source_size_bytes": source_size,
        "output": str(target),
        "limited": max_sessions is not None,
        "max_sessions": max_sessions,
        "rows_per_file": rows_per_file,
        "session_rows_per_file": session_rows_per_file,
        "row_group_size": row_group_size,
        "compression": compression,
        "unique_items": len(unique_items),
        "min_ts": min_ts,
        "max_ts": max_ts,
        "event_parts": event_writer.part_count,
        "session_parts": session_writer.part_count,
        "session_length": _session_length_summary(
            session_length_histogram,
            counts["sessions_written"],
            counts["events_written"],
        ),
        "runtime_seconds": round(time.perf_counter() - started, 6),
        **dict(sorted(counts.items())),
    }
    (staging / "quality_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    staging.replace(target)
    return IngestionResult(target, report)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stream raw OTTO JSONL into canonical Parquet shards.")
    parser.add_argument("--dataset", choices=("train", "test"), default="train")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--rows-per-file", type=int)
    parser.add_argument("--session-rows-per-file", type=int)
    parser.add_argument("--row-group-size", type=int)
    parser.add_argument("--progress-every", type=int)
    parser.add_argument("--max-sessions", type=int, help="Smoke-test only; canonical ingestion uses all sessions.")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    config_path = args.config or os.environ.get("OTTO_CONFIG_PATH")
    if not config_path:
        raise ValueError("Provide --config or run through the experiment runner.")
    config = resolve_project_config(config_path)
    input_path = args.input or ROOT / config["paths"][f"{args.dataset}_jsonl"]
    experiment_dir = os.environ.get("OTTO_EXPERIMENT_DIR")
    if args.output_dir:
        output_dir = args.output_dir
    elif experiment_dir:
        output_dir = Path(experiment_dir) / "data" / args.dataset
    else:
        raise ValueError("Provide --output-dir or run through the experiment runner.")

    result = ingest_jsonl(
        input_path,
        output_dir,
        rows_per_file=args.rows_per_file or config["data"]["parquet_rows_per_file"],
        session_rows_per_file=(
            args.session_rows_per_file or config["data"]["session_rows_per_file"]
        ),
        row_group_size=args.row_group_size or config["runtime"]["parquet_row_group_size"],
        compression=config["runtime"]["parquet_compression"],
        max_sessions=args.max_sessions,
        progress_every=(
            args.progress_every
            if args.progress_every is not None
            else config["runtime"]["progress_every_sessions"]
        ),
    )
    print(json.dumps(result.report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
