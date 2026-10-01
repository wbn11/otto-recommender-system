import pyarrow as pa
import pyarrow.parquet as pq

from data.schemas import EVENT_SCHEMA
from recall.build_type_covis import build_type_covis


def test_buy2buy_filters_clicks_and_counts_each_session_once(tmp_path):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    events = [
        {"session": 1, "aid": 10, "ts": 1, "event_type": 1},
        {"session": 1, "aid": 20, "ts": 2, "event_type": 2},
        {"session": 1, "aid": 20, "ts": 3, "event_type": 2},
        {"session": 1, "aid": 30, "ts": 4, "event_type": 3},
        {"session": 2, "aid": 20, "ts": 5, "event_type": 2},
        {"session": 2, "aid": 40, "ts": 6, "event_type": 1},
        {"session": 2, "aid": 50, "ts": 7, "event_type": 3},
        {"session": 3, "aid": 60, "ts": 8, "event_type": 1},
    ]
    pq.write_table(
        pa.Table.from_pylist(events, schema=EVENT_SCHEMA),
        snapshot / "part-00000.parquet",
    )

    output = tmp_path / "buy2buy"
    report = build_type_covis(
        snapshot,
        output,
        dataset_name="ranker",
        recent_events_per_session=30,
        topk_neighbors=2,
        pair_buckets=2,
        event_type_weights={2: 1.0, 3: 1.0},
        workers=1,
        memory_limit_gb=1,
        source_name="buy2buy",
        allowed_event_types=(2, 3),
        pair_semantics="test",
    )

    files = sorted((output / "matrix").glob("bucket-*.parquet"))
    matrix = pa.concat_tables([pq.read_table(path) for path in files]).to_pylist()
    pairs = {
        (row["aid"], row["neighbor_aid"]): row["source_score"] for row in matrix
    }
    assert pairs == {
        (20, 30): 1.0,
        (30, 20): 1.0,
        (20, 50): 1.0,
        (50, 20): 1.0,
    }
    assert all(10 not in pair and 40 not in pair and 60 not in pair for pair in pairs)
    assert report["source"] == "buy2buy"
    assert report["configuration"]["allowed_event_types"] == [2, 3]
    assert report["assertions"]["passed"] is True
