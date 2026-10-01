import json
import math

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from data.schemas import EVENT_SCHEMA, LABEL_SCHEMA, POPULAR_SCHEMA, QUERY_SCHEMA, RECALL_SCHEMA
from recall.build_popular_revisit import build_popular_revisit, compute_revisit_candidates


def test_compute_revisit_candidates_matches_hand_calculation():
    candidates = compute_revisit_candidates(
        [10, 20, 10],
        [1, 2, 3],
        recency_scale_events=10.0,
        event_type_weights={1: 1.0, 2: 3.0, 3: 6.0},
    )
    assert [aid for aid, _ in candidates] == [10, 20]
    assert candidates[0][1] == pytest.approx(6.0 * math.log(3.0))
    assert candidates[1][1] == pytest.approx(math.exp(-0.1) * 3.0 * math.log(2.0))


def test_build_popular_revisit_writes_candidates_and_metrics(tmp_path):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    events = [
        {"session": 1, "aid": 10, "ts": 1, "event_type": 1},
        {"session": 1, "aid": 10, "ts": 2, "event_type": 1},
        {"session": 2, "aid": 20, "ts": 3, "event_type": 1},
        {"session": 2, "aid": 20, "ts": 4, "event_type": 2},
        {"session": 3, "aid": 30, "ts": 5, "event_type": 2},
        {"session": 3, "aid": 30, "ts": 6, "event_type": 3},
        {"session": 3, "aid": 30, "ts": 7, "event_type": 3},
    ]
    pq.write_table(
        pa.Table.from_pylist(events, schema=EVENT_SCHEMA),
        snapshot / "part-00000.parquet",
    )
    queries_path = tmp_path / "queries.parquet"
    labels_path = tmp_path / "labels.parquet"
    queries = [
        {"session": 100, "aids": [10, 20, 10], "timestamps": [10, 11, 12], "event_types": [1, 2, 3]},
        {"session": 200, "aids": [30], "timestamps": [13], "event_types": [1]},
    ]
    labels = [
        {"session": 100, "target_type": 1, "aid": 10},
        {"session": 100, "target_type": 2, "aid": 20},
        {"session": 100, "target_type": 3, "aid": 10},
        {"session": 200, "target_type": 1, "aid": 30},
        {"session": 200, "target_type": 2, "aid": 30},
        {"session": 200, "target_type": 3, "aid": 30},
    ]
    pq.write_table(pa.Table.from_pylist(queries, schema=QUERY_SCHEMA), queries_path)
    pq.write_table(pa.Table.from_pylist(labels, schema=LABEL_SCHEMA), labels_path)

    output = tmp_path / "recall"
    report = build_popular_revisit(
        snapshot,
        queries_path,
        labels_path,
        output,
        dataset_name="ranker",
        topk=2,
        eval_ks=(1, 2),
        workers=1,
        memory_limit_gb=1,
    )

    popular = pq.read_table(output / "popular" / "items.parquet")
    revisit = pq.read_table(output / "revisit" / "candidates.parquet")
    assert popular.schema.names == POPULAR_SCHEMA.names
    assert popular.schema.types == POPULAR_SCHEMA.types
    assert revisit.schema.names == RECALL_SCHEMA.names
    assert revisit.schema.types == RECALL_SCHEMA.types
    popular_rows = popular.to_pylist()
    assert [row["aid"] for row in popular_rows if row["target_type"] == 1] == [10, 20]
    assert [row["aid"] for row in popular_rows if row["target_type"] == 2] == [20, 30]
    assert [row["aid"] for row in popular_rows if row["target_type"] == 3] == [30]

    revisit_rows = revisit.to_pylist()
    session_100_clicks = [
        row for row in revisit_rows if row["session"] == 100 and row["target_type"] == 1
    ]
    session_100_clicks.sort(key=lambda row: row["source_rank"])
    assert [(row["aid"], row["source_rank"]) for row in session_100_clicks] == [(10, 1), (20, 2)]
    candidate_keys = {
        (row["session"], row["target_type"], row["aid"]) for row in revisit_rows
    }
    assert len(candidate_keys) == len(revisit_rows)
    assert report["metrics"]["revisit"]["recall_at_2"]["weighted"] == 1.0
    saved = json.loads((output / "metrics.json").read_text())
    assert saved["artifacts"]["popular"]["rows"] == 5
