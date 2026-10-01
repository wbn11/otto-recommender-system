import hashlib
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from data.schemas import DSSM_SEQUENCE_SCHEMA, ITEM_VOCAB_SCHEMA
from models.dssm_attention import TargetAwareAttentionDSSM
from models.train_dssm_attention import train_attention
from utils.config import resolve_project_config


def test_target_type_changes_attention_and_padding_is_masked():
    model = TargetAwareAttentionDSSM(
        num_items=5,
        embedding_dim=2,
        max_sequence_length=3,
        sparse_item_gradients=False,
    )
    with torch.no_grad():
        model.item_embedding.weight.zero_()
        model.event_type_embedding.weight.zero_()
        model.target_type_embedding.weight.zero_()
        model.recency_embedding.weight.zero_()
        model.item_embedding.weight[2] = torch.tensor([1.0, 0.0])
        model.item_embedding.weight[3] = torch.tensor([0.0, 1.0])
        model.target_type_embedding.weight[1] = torch.tensor([1.0, 0.0])
        model.target_type_embedding.weight[2] = torch.tensor([0.0, 1.0])
        model.query_projection.weight.copy_(torch.eye(2))
        model.key_projection.weight.copy_(torch.eye(2))
        model.value_projection.weight.copy_(torch.eye(2))

    items = torch.tensor([[0, 2, 3], [0, 2, 3]])
    event_types = torch.tensor([[0, 1, 1], [0, 1, 1]])
    targets = torch.tensor([1, 2])
    weights = model.attention_weights(items, event_types, targets)

    assert torch.equal(weights[:, 0], torch.zeros(2))
    assert torch.allclose(weights.sum(dim=1), torch.ones(2))
    assert weights[0, 1] > weights[0, 2]
    assert weights[1, 2] > weights[1, 1]
    sessions = model.encode_session(items, event_types, targets)
    assert not torch.allclose(sessions[0], sessions[1])


def test_attention_handles_all_padding_and_keeps_sparse_item_gradients():
    model = TargetAwareAttentionDSSM(
        num_items=6,
        embedding_dim=4,
        max_sequence_length=3,
    )
    empty_items = torch.tensor([[0, 0]])
    empty_types = torch.tensor([[0, 0]])
    weights = model.attention_weights(empty_items, empty_types, torch.tensor([1]))
    session = model.encode_session(empty_items, empty_types, torch.tensor([1]))
    assert torch.equal(weights, torch.zeros_like(weights))
    assert torch.isfinite(session).all()

    logits = model(
        torch.tensor([[0, 2], [3, 4]]),
        torch.tensor([[0, 1], [2, 3]]),
        torch.tensor([3, 5]),
        torch.tensor([1, 2]),
    )
    logits.sum().backward()
    assert model.item_embedding.weight.grad is not None
    assert model.item_embedding.weight.grad.is_sparse


def test_tiny_attention_training_writes_loadable_checkpoint(tmp_path):
    source = tmp_path / "prepared"
    sequences = source / "sequences"
    sequences.mkdir(parents=True)
    vocab = source / "item_vocab.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {"aid": 10, "item_id": 2},
                {"aid": 20, "item_id": 3},
                {"aid": 30, "item_id": 4},
            ],
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
    snapshot = "/example/ranker_snapshot/events"
    (source / "build_report.json").write_text(
        json.dumps(
            {
                "dataset": "ranker",
                "snapshot": snapshot,
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
    config["dssm"].update(
        {
            "embedding_dim": 8,
            "batch_size": 2,
            "epochs": 1,
            "amp": False,
            "attention": True,
            "default_embedding": False,
            "history_dropout": 0.0,
            "hard_negatives": 0,
        }
    )
    report = train_attention(
        source,
        tmp_path / "experiment",
        config,
        device_name="cpu",
    )

    assert report["architecture"]["name"] == "target_aware_attention_dssm"
    assert report["architecture"]["default_session_embedding"] is False
    assert report["epochs"][0]["pairs"] == 4
    checkpoint = torch.load(report["checkpoint"], map_location="cpu", weights_only=True)
    assert checkpoint["format_version"] == 3


def test_attention_checkpoint_is_supported_by_recall_loader(tmp_path):
    pytest.importorskip("faiss")
    from recall.generate_dssm import _load_model

    model = TargetAwareAttentionDSSM(
        num_items=5,
        embedding_dim=4,
        max_sequence_length=3,
    )
    checkpoint = tmp_path / "attention.pt"
    torch.save(
        {
            "format_version": 3,
            "architecture": {
                "name": "target_aware_attention_dssm",
                "num_items": 5,
                "embedding_dim": 4,
                "num_types": 4,
                "temperature": 0.1,
                "max_sequence_length": 3,
                "sparse_item_gradients": True,
            },
            "model_state_dict": model.state_dict(),
            "epoch": 1,
            "vocab_sha256": "vocab-hash",
            "snapshot": "/snapshot",
            "snapshot_max_ts": 123,
            "config_hash": "config-hash",
        },
        checkpoint,
    )
    loaded, metadata = _load_model(
        checkpoint,
        vocab_hash="vocab-hash",
        snapshot="/snapshot",
        device=torch.device("cpu"),
    )
    assert isinstance(loaded, TargetAwareAttentionDSSM)
    assert metadata["checkpoint_format_version"] == 3
