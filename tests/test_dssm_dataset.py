from functools import partial

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader

from data.schemas import DSSM_SEQUENCE_SCHEMA
from models.dssm_dataset import NextItemSample, ParquetNextItemDataset, collate_next_item


def test_streaming_dataset_generates_every_next_item_pair_once(tmp_path):
    sequences = tmp_path / "sequences"
    sequences.mkdir()
    rows = [
        {"session": 10, "item_ids": [2, 3, 2, 4], "event_types": [1, 2, 1, 3]},
        {"session": 20, "item_ids": [1, 5], "event_types": [1, 2]},
    ]
    pq.write_table(
        pa.Table.from_pylist(rows, schema=DSSM_SEQUENCE_SCHEMA),
        sequences / "part-00000.parquet",
        row_group_size=1,
    )
    dataset = ParquetNextItemDataset(sequences, max_history=2, sessions_per_batch=1)
    examples = list(dataset)
    assert len(dataset.row_groups) == 2
    assert [(row.session, row.target_position, row.target_item_id) for row in examples] == [
        (10, 1, 3),
        (10, 2, 2),
        (10, 3, 4),
        (20, 1, 5),
    ]
    assert [row.history_item_ids for row in examples] == [
        (2,),
        (2, 3),
        (3, 2),
        (1,),
    ]
    assert [row.history_event_types for row in examples] == [
        (1,),
        (1, 2),
        (2, 1),
        (1,),
    ]

    loader = DataLoader(
        dataset,
        batch_size=2,
        num_workers=2,
        multiprocessing_context="spawn",
        collate_fn=partial(collate_next_item, type_weights={1: 1.0, 2: 3.0, 3: 6.0}),
    )
    streamed = sorted(
        (int(session), int(position), int(item))
        for batch in loader
        for session, position, item in zip(
            batch["sessions"], batch["target_positions"], batch["target_item_ids"]
        )
    )
    assert streamed == sorted(
        (row.session, row.target_position, row.target_item_id) for row in examples
    )


def test_collate_handles_empty_and_unknown_histories():
    examples = [
        NextItemSample(1, 0, (), (), 2, 1),
        NextItemSample(2, 1, (1, 3), (1, 2), 4, 3),
    ]
    batch = collate_next_item(examples)
    assert torch.equal(batch["history_item_ids"], torch.tensor([[0, 0], [1, 3]]))
    assert torch.equal(batch["history_event_types"], torch.tensor([[0, 0], [1, 2]]))
    assert batch["history_lengths"].tolist() == [0, 2]
    assert batch["sample_weights"].tolist() == [1.0, 6.0]
