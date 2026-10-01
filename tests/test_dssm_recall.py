import hashlib
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

pytest.importorskip("faiss")

from data.schemas import ITEM_VOCAB_SCHEMA, LABEL_SCHEMA, QUERY_SCHEMA
from models.dssm_baseline import FixedPositionDSSM
from recall.generate_dssm import generate_dssm_recall
from utils.config import resolve_project_config


def test_flatip_dssm_recall_writes_target_conditioned_lists(tmp_path):
    data_dir = tmp_path / "dssm-data"
    data_dir.mkdir()
    vocab_path = data_dir / "item_vocab.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {"aid": 10, "item_id": 2},
                {"aid": 20, "item_id": 3},
                {"aid": 30, "item_id": 4},
            ],
            schema=ITEM_VOCAB_SCHEMA,
        ),
        vocab_path,
    )
    vocab_hash = hashlib.sha256(vocab_path.read_bytes()).hexdigest()
    snapshot = "/example/ranker_snapshot/events"
    (data_dir / "build_report.json").write_text(
        json.dumps(
            {
                "dataset": "ranker",
                "snapshot": snapshot,
                "assertions": {"passed": True},
                "vocabulary": {
                    "path": "item_vocab.parquet",
                    "sha256": vocab_hash,
                    "embedding_rows": 5,
                },
            }
        ),
        encoding="utf-8",
    )

    model = FixedPositionDSSM(num_items=5, embedding_dim=4)
    checkpoint = tmp_path / "model.pt"
    torch.save(
        {
            "format_version": 2,
            "architecture": {
                "name": "fixed_position_dssm",
                "num_items": 5,
                "embedding_dim": 4,
                "num_types": 4,
                "temperature": 0.1,
                "max_sequence_length": 50,
                "sparse_item_gradients": True,
            },
            "model_state_dict": model.state_dict(),
            "epoch": 1,
            "vocab_sha256": vocab_hash,
            "snapshot": snapshot,
            "snapshot_max_ts": 100,
            "config_hash": "test",
        },
        checkpoint,
    )

    queries = tmp_path / "queries.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "session": 7,
                    "aids": [10, 999],
                    "timestamps": [101, 102],
                    "event_types": [1, 2],
                }
            ],
            schema=QUERY_SCHEMA,
        ),
        queries,
    )
    labels = tmp_path / "labels.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {"session": 7, "target_type": 1, "aid": 10},
                {"session": 7, "target_type": 2, "aid": 20},
                {"session": 7, "target_type": 3, "aid": 30},
            ],
            schema=LABEL_SCHEMA,
        ),
        labels,
    )

    config = resolve_project_config("configs/experiments/debug.yaml")
    config["faiss"].update({"topk": 2, "query_batch_size": 1, "use_gpu": False})
    config["recall"]["eval_ks"] = [1, 2]
    config["runtime"].update({"workers": 1, "duckdb_memory_limit_gb": 1})
    output = tmp_path / "recall"
    report = generate_dssm_recall(
        data_dir,
        checkpoint,
        queries,
        labels,
        output,
        config,
        dataset_name="ranker",
    )

    table = pq.read_table(output / "candidates")
    assert table.num_rows == 3
    assert table["session"].to_pylist() == [7, 7, 7]
    assert table["target_type"].to_pylist() == [1, 2, 3]
    assert all(len(values) == 2 for values in table["aids"].to_pylist())
    assert all(0 not in values and 1 not in values for values in table["aids"].to_pylist())
    assert report["queries"]["unknown_history_events"] == 1
    assert report["candidates"]["rows"] == 3
    assert report["assertions"]["passed"] is True
    assert report["index"]["flatip_exact_top20_overlap"] == 1.0
    assert set(report["metrics"]) == {"recall_at_1", "recall_at_2"}


def test_dssm_recall_rejects_the_wrong_snapshot_dataset(tmp_path):
    data_dir = tmp_path / "dssm-data"
    data_dir.mkdir()
    vocab_path = data_dir / "item_vocab.parquet"
    pq.write_table(
        pa.Table.from_pylist([{"aid": 10, "item_id": 2}], schema=ITEM_VOCAB_SCHEMA),
        vocab_path,
    )
    (data_dir / "build_report.json").write_text(
        json.dumps(
            {
                "dataset": "valid",
                "snapshot": "/valid",
                "vocabulary": {
                    "path": "item_vocab.parquet",
                    "sha256": hashlib.sha256(vocab_path.read_bytes()).hexdigest(),
                },
            }
        ),
        encoding="utf-8",
    )
    empty = tmp_path / "placeholder"
    empty.write_bytes(b"x")
    config = resolve_project_config("configs/experiments/debug.yaml")
    with pytest.raises(AssertionError, match="not 'ranker'"):
        generate_dssm_recall(
            data_dir,
            empty,
            empty,
            empty,
            tmp_path / "output",
            config,
            dataset_name="ranker",
        )
