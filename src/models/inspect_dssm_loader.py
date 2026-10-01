"""Consume DSSM Parquet batches and verify online next-item sample counts."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from functools import partial
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.schemas import EVENT_TYPE_TO_ID
from models.dssm_dataset import ParquetNextItemDataset, collate_next_item
from utils.config import resolve_project_config


def inspect_loader(
    data_dir: str | Path,
    report_path: str | Path,
    *,
    batch_size: int,
    max_history: int,
    num_workers: int,
    type_weights: dict[int, float],
) -> dict:
    source = Path(data_dir).resolve()
    destination = Path(report_path).resolve()
    build_report_path = source / "build_report.json"
    if not build_report_path.is_file():
        raise FileNotFoundError(build_report_path)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite DSSM loader report: {destination}")
    if batch_size <= 0 or num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers non-negative")

    build_report = json.loads(build_report_path.read_text(encoding="utf-8"))
    dataset = ParquetNextItemDataset(source / "sequences", max_history=max_history)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=partial(collate_next_item, type_weights=type_weights),
        multiprocessing_context="spawn" if num_workers > 0 else None,
    )
    started = time.perf_counter()
    counts: Counter[int] = Counter()
    pair_count = 0
    batch_count = 0
    history_events = 0
    max_observed_history = 0
    empty_histories = 0
    unk_history_events = 0
    min_target = None
    max_target = None
    for batch in loader:
        target_ids = batch["target_item_ids"]
        target_types = batch["target_types"]
        histories = batch["history_item_ids"]
        lengths = batch["history_lengths"]
        size = int(target_ids.numel())
        pair_count += size
        batch_count += 1
        history_events += int(lengths.sum().item())
        max_observed_history = max(max_observed_history, int(lengths.max().item()))
        empty_histories += int((lengths == 0).sum().item())
        unk_history_events += int((histories == 1).sum().item())
        min_target = int(target_ids.min().item()) if min_target is None else min(
            min_target, int(target_ids.min().item())
        )
        max_target = int(target_ids.max().item()) if max_target is None else max(
            max_target, int(target_ids.max().item())
        )
        unique_types, type_counts = torch.unique(target_types, return_counts=True)
        counts.update({int(kind): int(n) for kind, n in zip(unique_types, type_counts)})

    expected_pairs = int(build_report["sequences"]["next_item_pairs"])
    vocab_max = int(build_report["vocabulary"]["max_item_id"])
    violations = {
        "pair_count_mismatch": int(pair_count != expected_pairs),
        "history_over_max_length": int(max_observed_history > max_history),
        "invalid_target_item_id": int(min_target is None or min_target < 2),
        "target_over_vocab": int(max_target is None or max_target > vocab_max),
        "invalid_target_type": sum(
            value for kind, value in counts.items() if kind not in (1, 2, 3)
        ),
        "type_count_mismatch": int(sum(counts.values()) != pair_count),
    }
    if any(violations.values()):
        raise AssertionError(f"Invalid DSSM loader output: {violations}")

    report = {
        "input": str(source),
        "output": str(destination),
        "vocab_sha256": build_report["vocabulary"]["sha256"],
        "configuration": {
            "batch_size": batch_size,
            "max_history": max_history,
            "num_workers": num_workers,
            "multiprocessing_context": "spawn" if num_workers > 0 else None,
            "type_weights": {str(key): value for key, value in type_weights.items()},
        },
        "row_groups": len(dataset.row_groups),
        "sequence_files": len(dataset.files),
        "batches": batch_count,
        "next_item_pairs": pair_count,
        "expected_next_item_pairs": expected_pairs,
        "target_type_counts": {str(key): counts.get(key, 0) for key in (1, 2, 3)},
        "mean_history_length": round(history_events / pair_count, 6),
        "max_history_length": max_observed_history,
        "empty_histories": empty_histories,
        "unk_history_events": unk_history_events,
        "min_target_item_id": min_target,
        "max_target_item_id": max_target,
        "assertions": {"passed": True, "violations": violations},
        "runtime_seconds": round(time.perf_counter() - started, 6),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect streaming DSSM next-item batches.")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--report-path", type=Path)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    config_path = args.config or os.environ.get("OTTO_CONFIG_PATH")
    if not config_path:
        raise ValueError("Provide --config or run through the experiment runner")
    config = resolve_project_config(config_path)
    experiment_dir = os.environ.get("OTTO_EXPERIMENT_DIR")
    if args.report_path:
        report_path = args.report_path
    elif experiment_dir:
        report_path = Path(experiment_dir) / "metrics" / "dssm_loader_report.json"
    else:
        raise ValueError("Provide --report-path or run through the experiment runner")
    type_weights = {
        EVENT_TYPE_TO_ID[name]: float(value)
        for name, value in config["dssm"]["type_loss_weights"].items()
    }
    report = inspect_loader(
        args.data_dir,
        report_path,
        batch_size=int(config["dssm"]["batch_size"]),
        max_history=int(config["dssm"]["max_sequence_length"]),
        num_workers=int(config["runtime"]["workers"]),
        type_weights=type_weights,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
