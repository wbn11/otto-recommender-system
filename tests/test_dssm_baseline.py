import hashlib
import json
import math

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
import torch.nn.functional as F

from data.schemas import DSSM_SEQUENCE_SCHEMA, ITEM_VOCAB_SCHEMA
from models.dssm_baseline import FixedPositionDSSM, masked_in_batch_loss
from models.train_dssm_baseline import train_baseline
from utils.config import resolve_project_config


def test_fixed_position_pooling_and_empty_unknown_histories():
    model = FixedPositionDSSM(num_items=5, embedding_dim=2)
    with torch.no_grad():
        model.item_embedding.weight.zero_()
        model.type_embedding.weight.zero_()
        model.item_embedding.weight[2] = torch.tensor([1.0, 0.0])
        model.item_embedding.weight[3] = torch.tensor([0.0, 1.0])
        model.type_embedding.weight[1] = torch.tensor([1.0, 1.0])
    pooled = model.encode_session(
        torch.tensor([[0, 2, 3]]),
        torch.tensor([[0, 1, 1]]),
        torch.tensor([1]),
    )
    assert torch.allclose(
        pooled,
        F.normalize(torch.tensor([[2.0 + 1 / 3, 2.0 + 2 / 3]]), dim=1),
    )
    unpadded = model.encode_session(
        torch.tensor([[2, 3]]), torch.tensor([[1, 1]]), torch.tensor([1])
    )
    assert torch.allclose(pooled, unpadded)
    empty = model.encode_session(torch.tensor([[0]]), torch.tensor([[0]]), torch.tensor([1]))
    unknown = model.encode_session(torch.tensor([[1]]), torch.tensor([[1]]), torch.tensor([1]))
    assert torch.isfinite(empty).all() and torch.isfinite(unknown).all()
    assert torch.allclose(empty, F.normalize(torch.tensor([[1.0, 1.0]]), dim=1))


def test_duplicate_targets_are_not_in_batch_negatives():
    logits = torch.tensor(
        [[2.0, 10.0, 0.0], [10.0, 2.0, 0.0], [0.0, 0.0, 2.0]],
        requires_grad=True,
    )
    targets = torch.tensor([2, 2, 3])
    weights = torch.tensor([1.0, 3.0, 6.0])
    loss, masked = masked_in_batch_loss(logits, targets, weights)
    expected = (
        4 * math.log1p(math.exp(-2))
        + 6 * math.log1p(2 * math.exp(-2))
    ) / 10
    assert masked == 2
    assert loss.item() == pytest.approx(expected)
    loss.backward()
    assert logits.grad[0, 1].item() == 0
    assert logits.grad[1, 0].item() == 0


def test_item_embedding_uses_sparse_gradients():
    model = FixedPositionDSSM(num_items=6, embedding_dim=4)
    logits = model(
        torch.tensor([[0, 2], [3, 4]]),
        torch.tensor([[0, 1], [2, 3]]),
        torch.tensor([3, 5]),
        torch.tensor([1, 2]),
    )
    logits.sum().backward()
    assert model.item_embedding.weight.grad is not None
    assert model.item_embedding.weight.grad.is_sparse


def test_full_profile_keeps_128_dimensions():
    config = resolve_project_config("configs/experiments/full.yaml")
    assert config["dssm"]["embedding_dim"] == 128


def test_tiny_streaming_training_writes_reproducible_checkpoint(tmp_path):
    source = tmp_path / "prepared"
    sequences = source / "sequences"
    sequences.mkdir(parents=True)
    vocab = source / "item_vocab.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [{"aid": 10, "item_id": 2}, {"aid": 20, "item_id": 3}, {"aid": 30, "item_id": 4}],
            schema=ITEM_VOCAB_SCHEMA,
        ),
        vocab,
    )
    pq.write_table(
        pa.Table.from_pylist(
            [
                {"session": 1, "item_ids": [2, 3, 4], "event_types": [1, 2, 3]},
                {"session": 2, "item_ids": [2, 3, 2], "event_types": [1, 1, 1]},
            ],
            schema=DSSM_SEQUENCE_SCHEMA,
        ),
        sequences / "part-00000.parquet",
        row_group_size=1,
    )
    vocab_hash = hashlib.sha256(vocab.read_bytes()).hexdigest()
    (source / "build_report.json").write_text(
        json.dumps(
            {
                "dataset": "ranker",
                "snapshot": "/example/ranker_snapshot/events",
                "source": {"max_ts": 123},
                "assertions": {"passed": True},
                "vocabulary": {
                    "path": "item_vocab.parquet",
                    "sha256": vocab_hash,
                    "embedding_rows": 5,
                },
                "sequences": {"next_item_pairs": 4},
            }
        ),
        encoding="utf-8",
    )
    config = resolve_project_config("configs/experiments/debug.yaml")
    config["runtime"]["workers"] = 0
    config["dssm"].update({"embedding_dim": 8, "batch_size": 2, "epochs": 1, "amp": False})
    experiment = tmp_path / "experiment"
    report = train_baseline(source, experiment, config, device_name="cpu")
    assert report["partial_smoke"] is False
    assert report["epochs"][0]["pairs"] == 4
    assert report["vocab_sha256"] == vocab_hash
    assert len(report["sequence_sha256"]) == 64
    checkpoint = torch.load(report["checkpoint"], map_location="cpu", weights_only=True)
    assert checkpoint["architecture"]["embedding_dim"] == 8
    assert checkpoint["vocab_sha256"] == vocab_hash
    assert checkpoint["sequence_sha256"] == report["sequence_sha256"]
    assert checkpoint["epoch"] == 1
    with pytest.raises(FileExistsError):
        train_baseline(source, experiment, config, device_name="cpu")


def test_valid_snapshot_is_a_supported_point_in_time_training_source(tmp_path):
    source = tmp_path / "valid-prepared"
    sequences = source / "sequences"
    sequences.mkdir(parents=True)
    vocab = source / "item_vocab.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [{"aid": 10, "item_id": 2}, {"aid": 20, "item_id": 3}],
            schema=ITEM_VOCAB_SCHEMA,
        ),
        vocab,
    )
    pq.write_table(
        pa.Table.from_pylist(
            [{"session": 1, "item_ids": [2, 3], "event_types": [1, 2]}],
            schema=DSSM_SEQUENCE_SCHEMA,
        ),
        sequences / "part-00000.parquet",
    )
    vocab_hash = hashlib.sha256(vocab.read_bytes()).hexdigest()
    (source / "build_report.json").write_text(
        json.dumps(
            {
                "dataset": "valid",
                "snapshot": "/example/valid_snapshot/events",
                "source": {"max_ts": 456},
                "assertions": {"passed": True},
                "vocabulary": {
                    "path": "item_vocab.parquet",
                    "sha256": vocab_hash,
                    "embedding_rows": 4,
                },
                "sequences": {"next_item_pairs": 1},
            }
        ),
        encoding="utf-8",
    )
    config = resolve_project_config("configs/experiments/debug.yaml")
    config["runtime"]["workers"] = 0
    config["dssm"].update({"embedding_dim": 4, "batch_size": 1, "epochs": 1, "amp": False})
    report = train_baseline(source, tmp_path / "valid-experiment", config, device_name="cpu")
    assert report["snapshot"] == "/example/valid_snapshot/events"
