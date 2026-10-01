import pyarrow as pa
import pyarrow.parquet as pq

from data.prepare_dssm_data import prepare_dssm_data
from data.schemas import DSSM_SEQUENCE_SCHEMA, EVENT_SCHEMA, ITEM_VOCAB_SCHEMA


def test_prepare_dssm_data_builds_deterministic_vocab_and_sequences(tmp_path):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    events = [
        {"session": 1, "aid": 30, "ts": 3, "event_type": 3},
        {"session": 1, "aid": 10, "ts": 1, "event_type": 1},
        {"session": 1, "aid": 20, "ts": 2, "event_type": 2},
        {"session": 2, "aid": 40, "ts": 4, "event_type": 1},
        {"session": 3, "aid": 20, "ts": 6, "event_type": 1},
        {"session": 3, "aid": 20, "ts": 5, "event_type": 2},
    ]
    pq.write_table(
        pa.Table.from_pylist(events, schema=EVENT_SCHEMA),
        snapshot / "part-00000.parquet",
    )

    output = tmp_path / "dssm"
    report = prepare_dssm_data(
        snapshot,
        output,
        dataset_name="ranker",
        workers=1,
        memory_limit_gb=1,
    )

    vocab = pq.read_table(output / "item_vocab.parquet")
    assert vocab.schema.names == ITEM_VOCAB_SCHEMA.names
    assert vocab.schema.types == ITEM_VOCAB_SCHEMA.types
    assert vocab.to_pylist() == [
        {"aid": 10, "item_id": 2},
        {"aid": 20, "item_id": 3},
        {"aid": 30, "item_id": 4},
        {"aid": 40, "item_id": 5},
    ]

    sequence_file = output / "sequences" / "part-00000.parquet"
    sequences = pq.read_table(sequence_file)
    assert sequences.schema.names == DSSM_SEQUENCE_SCHEMA.names
    assert sequences.schema.types == DSSM_SEQUENCE_SCHEMA.types
    assert sequences.to_pylist() == [
        {"session": 1, "item_ids": [2, 3, 4], "event_types": [1, 2, 3]},
        {"session": 3, "item_ids": [3, 3], "event_types": [2, 1]},
    ]
    assert report["reserved_item_ids"] == {"pad": 0, "unk": 1, "first_real": 2}
    assert report["source"]["single_event_sessions_dropped"] == 1
    assert report["sequences"]["next_item_pairs"] == 3
    assert report["assertions"]["passed"] is True
