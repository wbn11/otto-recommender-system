import json

import pyarrow as pa
import pyarrow.parquet as pq

from data.build_time_splits import (
    DAY_MS,
    build_time_splits,
    classify_supervised_session,
    compute_boundaries,
)
from data.schemas import EVENT_SCHEMA, LABEL_SCHEMA, QUERY_SCHEMA, SESSION_SCHEMA


def _event(session, aid, ts, event_type=1):
    return {"session": session, "aid": aid, "ts": ts, "event_type": event_type}


def _session_row(session, events):
    counts = {1: 0, 2: 0, 3: 0}
    for event in events:
        counts[event["event_type"]] += 1
    return {
        "session": session,
        "start_ts": min(event["ts"] for event in events),
        "end_ts": max(event["ts"] for event in events),
        "event_count": len(events),
        "click_count": counts[1],
        "cart_count": counts[2],
        "order_count": counts[3],
    }


def _read_event_dataset(path):
    files = sorted(path.glob("part-*.parquet"))
    return pa.concat_tables([pq.read_table(file) for file in files])


def test_supervised_session_classification_uses_start_time():
    boundaries = compute_boundaries(30 * DAY_MS, 7, 7)
    assert boundaries.t1 == 16 * DAY_MS
    assert boundaries.t2 == 23 * DAY_MS
    assert classify_supervised_session(boundaries.t1 - 1, boundaries) == "history_only"
    assert classify_supervised_session(boundaries.t1, boundaries) == "ranker_train"
    assert classify_supervised_session(boundaries.t2 - 1, boundaries) == "ranker_train"
    assert classify_supervised_session(boundaries.t2, boundaries) == "final_valid"
    assert classify_supervised_session(boundaries.window_end + 1, boundaries) == "outside_window"


def test_build_time_splits_writes_snapshots_queries_and_labels(tmp_path):
    events_dir = tmp_path / "events"
    sessions_dir = tmp_path / "sessions"
    events_dir.mkdir()
    sessions_dir.mkdir()
    max_ts = 30 * DAY_MS
    boundaries = compute_boundaries(max_ts, 7, 7)
    events_by_session = {
        1: [_event(1, 10, 1 * DAY_MS), _event(1, 11, 2 * DAY_MS)],
        2: [_event(2, 20, 15 * DAY_MS), _event(2, 21, 17 * DAY_MS)],
        3: [
            _event(3, 30, 18 * DAY_MS),
            _event(3, 31, 19 * DAY_MS, 2),
            _event(3, 32, 20 * DAY_MS, 3),
        ],
        4: [
            _event(4, 40, 22 * DAY_MS),
            _event(4, 41, 22 * DAY_MS + 1),
            _event(4, 42, 24 * DAY_MS),
        ],
        5: [_event(5, 50, 24 * DAY_MS), _event(5, 51, 25 * DAY_MS)],
        6: [
            _event(6, 60, 15 * DAY_MS),
            _event(6, 61, 20 * DAY_MS),
            _event(6, 62, 24 * DAY_MS),
        ],
        7: [_event(7, 70, 29 * DAY_MS), _event(7, 71, max_ts)],
    }
    all_events = [event for events in events_by_session.values() for event in events]
    session_rows = [
        _session_row(session, events) for session, events in events_by_session.items()
    ]
    pq.write_table(
        pa.Table.from_pylist(all_events, schema=EVENT_SCHEMA),
        events_dir / "part-00000.parquet",
    )
    pq.write_table(
        pa.Table.from_pylist(session_rows, schema=SESSION_SCHEMA),
        sessions_dir / "part-00000.parquet",
    )

    target = tmp_path / "splits"
    snapshots = tmp_path / "snapshots"
    report = build_time_splits(
        events_dir, sessions_dir, target, snapshots, workers=1, memory_limit_gb=1
    )

    ranker_snapshot = _read_event_dataset(
        snapshots / "ranker_snapshot" / "events"
    ).to_pylist()
    valid_snapshot = _read_event_dataset(
        snapshots / "valid_snapshot" / "events"
    ).to_pylist()
    assert all(row["ts"] < boundaries.t1 for row in ranker_snapshot)
    assert all(row["ts"] < boundaries.t2 for row in valid_snapshot)
    assert {(row["session"], row["aid"]) for row in ranker_snapshot} == {
        (1, 10), (1, 11), (2, 20), (6, 60)
    }
    valid_pairs = {(row["session"], row["aid"]) for row in valid_snapshot}
    assert (2, 21) in valid_pairs
    assert (6, 61) in valid_pairs
    assert (4, 42) not in valid_pairs

    ranker_query_table = pq.read_table(target / "ranker_queries.parquet")
    ranker_label_table = pq.read_table(target / "ranker_labels.parquet")
    valid_query_table = pq.read_table(target / "valid_queries.parquet")
    valid_label_table = pq.read_table(target / "valid_labels.parquet")
    assert ranker_query_table.column_names == QUERY_SCHEMA.names
    assert ranker_query_table.schema.types == QUERY_SCHEMA.types
    assert ranker_label_table.column_names == LABEL_SCHEMA.names
    assert ranker_label_table.schema.types == LABEL_SCHEMA.types
    assert valid_query_table.column_names == QUERY_SCHEMA.names
    assert valid_query_table.schema.types == QUERY_SCHEMA.types
    assert valid_label_table.column_names == LABEL_SCHEMA.names
    assert valid_label_table.schema.types == LABEL_SCHEMA.types

    ranker_queries = {row["session"]: row for row in ranker_query_table.to_pylist()}
    valid_queries = {row["session"]: row for row in valid_query_table.to_pylist()}
    assert set(ranker_queries) == {3, 4}
    assert set(valid_queries) == {5, 7}
    assert ranker_queries[3]["aids"] == [30, 31]
    assert ranker_queries[3]["event_types"] == [1, 2]
    assert ranker_queries[4]["aids"] == [40]
    assert all(ts < boundaries.t2 for ts in ranker_queries[4]["timestamps"])
    assert valid_queries[5]["aids"] == [50]

    assert ranker_label_table.to_pylist() == [
        {"session": 3, "target_type": 3, "aid": 32},
        {"session": 4, "target_type": 1, "aid": 41},
    ]
    assert valid_label_table.to_pylist() == [
        {"session": 5, "target_type": 1, "aid": 51},
        {"session": 7, "target_type": 1, "aid": 71},
    ]

    assert report["cross_boundary_sessions"]["cross_t1"] == 2
    assert report["cross_boundary_sessions"]["cross_t2"] == 2
    assert report["leakage_assertions"]["passed"] is True
    saved_report = json.loads((target / "split_report.json").read_text())
    assert saved_report["strategy"] == "point_in_time"
    assert saved_report["supervised_datasets"]["ranker"]["written_sessions"] == 2
    assert saved_report["supervised_datasets"]["valid"]["written_sessions"] == 2
