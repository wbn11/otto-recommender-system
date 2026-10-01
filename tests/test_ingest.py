import json
from pathlib import Path

import pyarrow.dataset as ds
import pyarrow.parquet as pq

from data.ingest import ingest_jsonl
from data.schemas import EVENT_SCHEMA, SESSION_SCHEMA


def write_jsonl(path: Path, records):
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            if isinstance(record, str):
                handle.write(record + "\n")
            else:
                handle.write(json.dumps(record) + "\n")


def test_streaming_ingestion_writes_canonical_shards_and_quality_report(tmp_path):
    source = tmp_path / "sample.jsonl"
    target = tmp_path / "canonical"
    write_jsonl(source, [
        {
            "session": 10,
            "events": [
                {"aid": 100, "ts": 1000, "type": "clicks"},
                {"aid": 200, "ts": 2000, "type": "carts"},
                {"aid": 200, "ts": 2000, "type": "carts"},
            ],
        },
        {
            "session": 11,
            "events": [
                {"aid": 300, "ts": 3000, "type": "orders"},
                {"aid": 999, "ts": 4000, "type": "unknown"},
            ],
        },
        "{not-json",
    ])

    result = ingest_jsonl(
        source,
        target,
        rows_per_file=2,
        session_rows_per_file=1,
    )

    events = ds.dataset(target / "events", format="parquet").to_table()
    sessions = ds.dataset(target / "sessions", format="parquet").to_table()
    assert events.schema == EVENT_SCHEMA
    assert sessions.schema == SESSION_SCHEMA
    assert events.num_rows == 4
    assert sessions.num_rows == 2
    assert len(list((target / "events").glob("part-*.parquet"))) == 2
    assert result.report["unique_items"] == 3
    assert result.report["bytes_read"] == source.stat().st_size
    assert result.report["duplicate_events"] == 1
    assert result.report["invalid_event_types"] == 1
    assert result.report["invalid_json_lines"] == 1
    assert result.report["click_events"] == 1
    assert result.report["cart_events"] == 2
    assert result.report["order_events"] == 1
    assert result.report["session_length"] == {
        "min": 1,
        "max": 3,
        "mean": 2.0,
        "p50": 2.0,
        "p90": 2.8,
        "p95": 2.9,
        "p99": 2.98,
        "buckets": {
            "1": 1,
            "2": 0,
            "3-5": 1,
            "6-10": 0,
            "11-20": 0,
            "21-50": 0,
            "51-100": 0,
            "101+": 0,
        },
    }

    first_session_part = pq.read_table(target / "sessions" / "part-00000.parquet")
    assert first_session_part.column("start_ts").to_pylist() == [1000]
    assert first_session_part.column("end_ts").to_pylist() == [2000]


def test_ingestion_refuses_to_overwrite_existing_dataset(tmp_path):
    source = tmp_path / "sample.jsonl"
    target = tmp_path / "canonical"
    write_jsonl(source, [{"session": 1, "events": [{"aid": 1, "ts": 1, "type": "clicks"}]}])
    ingest_jsonl(source, target, rows_per_file=10)

    try:
        ingest_jsonl(source, target, rows_per_file=10)
    except FileExistsError as exc:
        assert "Refusing to overwrite" in str(exc)
    else:
        raise AssertionError("Expected FileExistsError")


def test_max_sessions_is_explicitly_marked_as_limited(tmp_path):
    source = tmp_path / "sample.jsonl"
    target = tmp_path / "canonical"
    write_jsonl(source, [
        {"session": index, "events": [{"aid": index, "ts": index + 1, "type": "clicks"}]}
        for index in range(3)
    ])

    result = ingest_jsonl(source, target, rows_per_file=10, max_sessions=2)

    assert result.report["limited"] is True
    assert result.report["sessions_written"] == 2
