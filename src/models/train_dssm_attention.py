"""Train target-aware attention DSSM from prepared streaming sequences."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from functools import partial
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.schemas import EVENT_TYPE_TO_ID
from models.dssm_attention import TargetAwareAttentionDSSM
from models.dssm_baseline import masked_in_batch_loss
from models.dssm_dataset import ParquetNextItemDataset, collate_next_item
from models.train_dssm_baseline import _save_checkpoint, _sequence_sha256, _sha256
from utils.config import config_hash, resolve_project_config


def train_attention(
    data_dir: str | Path,
    experiment_dir: str | Path,
    config: dict,
    *,
    max_batches: int | None = None,
    device_name: str | None = None,
) -> dict:
    """Train one snapshot-bound attention model with resumable checkpoints."""

    source = Path(data_dir).resolve()
    output_root = Path(experiment_dir).resolve()
    model_dir = output_root / "models" / "dssm_attention"
    report_path = output_root / "metrics" / "dssm_attention_train_report.json"
    build_report_path = source / "build_report.json"
    if not build_report_path.is_file():
        raise FileNotFoundError(build_report_path)
    if report_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite attention DSSM training report: {report_path}"
        )
    if max_batches is not None and max_batches <= 0:
        raise ValueError("max_batches must be positive")

    build_report = json.loads(build_report_path.read_text(encoding="utf-8"))
    if build_report["dataset"] not in {"ranker", "valid"}:
        raise ValueError("DSSM data must identify a ranker or valid snapshot")
    if not build_report["assertions"]["passed"]:
        raise AssertionError("DSSM input data did not pass its build checks")
    vocab_path = source / build_report["vocabulary"]["path"]
    vocab_hash = _sha256(vocab_path)
    if vocab_hash != build_report["vocabulary"]["sha256"]:
        raise AssertionError("DSSM item vocabulary differs from its build report")
    sequence_hash = _sequence_sha256(source / "sequences")

    dssm_config = config["dssm"]
    if not bool(dssm_config.get("attention", False)):
        raise ValueError("Attention DSSM requires dssm.attention=true")
    if bool(dssm_config.get("default_embedding", False)):
        raise ValueError("Attention-only ablation requires default_embedding=false")
    if float(dssm_config.get("history_dropout", 0.0)) != 0.0:
        raise ValueError("Attention-only ablation requires history_dropout=0")
    if int(dssm_config.get("hard_negatives", 0)) != 0:
        raise ValueError("Attention-only ablation requires hard_negatives=0")
    epochs = int(dssm_config["epochs"])
    batch_size = int(dssm_config["batch_size"])
    max_history = int(dssm_config["max_sequence_length"])
    num_workers = int(config["runtime"]["workers"])
    seed = int(config["seed"])
    temperature = float(dssm_config["temperature"])
    if epochs <= 0 or batch_size <= 0 or max_history <= 0 or num_workers < 0:
        raise ValueError("Invalid attention DSSM training configuration")
    type_weights = {
        EVENT_TYPE_TO_ID[name]: float(weight)
        for name, weight in dssm_config["type_loss_weights"].items()
    }
    if set(type_weights) != {1, 2, 3} or any(
        weight <= 0 for weight in type_weights.values()
    ):
        raise ValueError("DSSM loss weights must cover all targets and be positive")

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    use_amp = bool(dssm_config["amp"]) and device.type == "cuda"

    dataset = ParquetNextItemDataset(source / "sequences", max_history=max_history)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=partial(collate_next_item, type_weights=type_weights),
        multiprocessing_context="spawn" if num_workers > 0 else None,
        pin_memory=device.type == "cuda",
    )
    architecture = {
        "name": "target_aware_attention_dssm",
        "num_items": int(build_report["vocabulary"]["embedding_rows"]),
        "embedding_dim": int(dssm_config["embedding_dim"]),
        "num_types": 4,
        "temperature": temperature,
        "max_sequence_length": max_history,
        "sparse_item_gradients": True,
        "attention": "scaled_dot_product",
        "learned_recency_embedding": True,
        "default_session_embedding": False,
        "history_dropout": 0.0,
    }
    model = TargetAwareAttentionDSSM(
        num_items=architecture["num_items"],
        embedding_dim=architecture["embedding_dim"],
        num_types=architecture["num_types"],
        max_sequence_length=max_history,
        temperature=temperature,
        sparse_item_gradients=architecture["sparse_item_gradients"],
    ).to(device)
    learning_rate = float(dssm_config["learning_rate"])
    item_optimizer = torch.optim.SparseAdam(
        [model.item_embedding.weight],
        lr=learning_rate,
    )
    dense_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if name != "item_embedding.weight"
    ]
    parameter_counts = {
        "item_embedding": int(model.item_embedding.weight.numel()),
        "dense_session_tower": int(
            sum(parameter.numel() for parameter in dense_parameters)
        ),
        "total": int(sum(parameter.numel() for parameter in model.parameters())),
    }
    dense_optimizer = torch.optim.AdamW(
        dense_parameters,
        lr=learning_rate,
        weight_decay=float(dssm_config["weight_decay"]),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    model_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    effective_config = {
        "batch_size": batch_size,
        "epochs": epochs,
        "max_batches_per_epoch": max_batches,
        "num_workers": num_workers,
        "amp": use_amp,
        "type_loss_weights": type_weights,
        "item_optimizer": "SparseAdam",
        "dense_optimizer": "AdamW",
        "seed": seed,
        "ablation_contract": {
            "attention": True,
            "default_embedding": False,
            "history_dropout": 0.0,
            "hard_negatives": 0,
        },
    }
    digest = config_hash(config)
    history: list[dict] = []
    global_step = 0
    start_epoch = 1
    existing = sorted(model_dir.glob("epoch-*.pt"))
    if existing:
        latest = existing[-1]
        checkpoint = torch.load(latest, map_location="cpu", weights_only=True)
        if (
            checkpoint["vocab_sha256"] != vocab_hash
            or checkpoint["sequence_sha256"] != sequence_hash
            or checkpoint["config_hash"] != digest
            or checkpoint["architecture"] != architecture
            or checkpoint["effective_config"] != effective_config
            or checkpoint["snapshot"] != build_report["snapshot"]
        ):
            raise ValueError(
                f"Cannot resume an incompatible attention DSSM checkpoint: {latest}"
            )
        model.load_state_dict(checkpoint["model_state_dict"])
        item_optimizer.load_state_dict(checkpoint["item_optimizer_state_dict"])
        dense_optimizer.load_state_dict(checkpoint["dense_optimizer_state_dict"])
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        torch.set_rng_state(checkpoint["torch_rng_state"])
        if device.type == "cuda" and checkpoint["cuda_rng_state_all"] is not None:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state_all"])
        history = checkpoint["history"]
        global_step = int(checkpoint["global_step"])
        start_epoch = int(checkpoint["epoch"]) + 1

    started = time.perf_counter()
    expected_pairs = int(build_report["sequences"]["next_item_pairs"])
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        epoch_start = time.perf_counter()
        weighted_loss_sum = 0.0
        weight_sum = 0.0
        pair_count = 0
        batch_count = 0
        masked_false_negatives = 0
        for batch in loader:
            history_items = batch["history_item_ids"].to(device, non_blocking=True)
            history_types = batch["history_event_types"].to(device, non_blocking=True)
            target_items = batch["target_item_ids"].to(device, non_blocking=True)
            target_types = batch["target_types"].to(device, non_blocking=True)
            sample_weights = batch["sample_weights"].to(device, non_blocking=True)
            item_optimizer.zero_grad(set_to_none=True)
            dense_optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(
                    history_items, history_types, target_items, target_types
                )
                loss, masked = masked_in_batch_loss(
                    logits, target_items, sample_weights
                )
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite attention DSSM loss at epoch {epoch}, batch {batch_count}"
                )
            scaler.scale(loss).backward()
            item_gradient = model.item_embedding.weight.grad
            if item_gradient is None or not item_gradient.is_sparse:
                raise AssertionError("Item embedding must retain sparse gradients")
            scaler.step(item_optimizer)
            scaler.step(dense_optimizer)
            scaler.update()

            batch_weight = float(sample_weights.sum().item())
            weighted_loss_sum += float(loss.item()) * batch_weight
            weight_sum += batch_weight
            pair_count += int(target_items.numel())
            batch_count += 1
            masked_false_negatives += masked
            global_step += 1
            if batch_count % 100 == 0:
                print(
                    f"[train-dssm-attention] epoch={epoch}/{epochs} "
                    f"batches={batch_count} pairs={pair_count} loss={loss.item():.5f}",
                    flush=True,
                )
            if max_batches is not None and batch_count >= max_batches:
                break

        if batch_count == 0 or (max_batches is None and pair_count != expected_pairs):
            raise AssertionError(
                f"Attention DSSM epoch {epoch} consumed {pair_count} pairs; "
                f"expected {expected_pairs}"
            )
        epoch_record = {
            "epoch": epoch,
            "batches": batch_count,
            "pairs": pair_count,
            "weighted_mean_loss": round(weighted_loss_sum / weight_sum, 7),
            "masked_false_negative_cells": masked_false_negatives,
            "runtime_seconds": round(time.perf_counter() - epoch_start, 6),
        }
        history.append(epoch_record)
        checkpoint_path = model_dir / f"epoch-{epoch:03d}.pt"
        _save_checkpoint(
            checkpoint_path,
            {
                "format_version": 3,
                "architecture": architecture,
                "model_state_dict": model.state_dict(),
                "item_optimizer_state_dict": item_optimizer.state_dict(),
                "dense_optimizer_state_dict": dense_optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "epoch": epoch,
                "global_step": global_step,
                "history": history,
                "vocab_sha256": vocab_hash,
                "sequence_sha256": sequence_hash,
                "snapshot": build_report["snapshot"],
                "snapshot_max_ts": int(build_report["source"]["max_ts"]),
                "config_hash": digest,
                "effective_config": effective_config,
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state_all": (
                    torch.cuda.get_rng_state_all() if device.type == "cuda" else None
                ),
            },
        )
        print(
            f"[train-dssm-attention] {json.dumps(epoch_record, sort_keys=True)}",
            flush=True,
        )

    report = {
        "architecture": architecture,
        "checkpoint": str(model_dir / f"epoch-{epochs:03d}.pt"),
        "config_hash": digest,
        "data_dir": str(source),
        "device": str(device),
        "amp": use_amp,
        "vocab_sha256": vocab_hash,
        "sequence_sha256": sequence_hash,
        "snapshot": build_report["snapshot"],
        "snapshot_max_ts": int(build_report["source"]["max_ts"]),
        "expected_pairs_per_epoch": expected_pairs,
        "parameter_counts": parameter_counts,
        "partial_smoke": max_batches is not None,
        "epochs": history,
        "training_runtime_seconds": round(
            sum(epoch["runtime_seconds"] for epoch in history), 6
        ),
        "current_run_runtime_seconds": round(time.perf_counter() - started, 6),
    }
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train target-aware attention DSSM.")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    config_path = args.config or os.environ.get("OTTO_CONFIG_PATH")
    if not config_path:
        raise ValueError("Provide --config or run through the experiment runner")
    output_dir = args.output_dir or os.environ.get("OTTO_EXPERIMENT_DIR")
    if not output_dir:
        raise ValueError("Provide --output-dir or run through the experiment runner")
    report = train_attention(
        args.data_dir,
        output_dir,
        resolve_project_config(config_path),
        max_batches=args.max_batches,
        device_name=args.device,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
