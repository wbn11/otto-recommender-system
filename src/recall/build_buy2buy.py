"""Build a Buy2Buy matrix from cart and order events only."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.schemas import EVENT_TYPE_TO_ID
from recall.build_type_covis import build_type_covis
from utils.config import resolve_project_config


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a cart/order-only Buy2Buy Top-K matrix.")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--snapshot-dir", type=Path, required=True)
    parser.add_argument("--dataset-name", choices=("ranker", "valid"), required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--recent-events", type=int)
    parser.add_argument("--topk-neighbors", type=int)
    parser.add_argument("--pair-buckets", type=int)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    config_path = args.config or os.environ.get("OTTO_CONFIG_PATH")
    if not config_path:
        raise ValueError("Provide --config or run through the experiment runner")
    config = resolve_project_config(config_path)
    experiment_dir = os.environ.get("OTTO_EXPERIMENT_DIR")
    if args.output_dir:
        output_dir = args.output_dir
    elif experiment_dir:
        output_dir = Path(experiment_dir) / "recall" / args.dataset_name / "buy2buy"
    else:
        raise ValueError("Provide --output-dir or run through the experiment runner")

    buy_config = config["recall"]["buy2buy"]
    event_type_weights = {
        EVENT_TYPE_TO_ID[name]: float(value)
        for name, value in buy_config["event_type_weights"].items()
    }
    report = build_type_covis(
        args.snapshot_dir,
        output_dir,
        dataset_name=args.dataset_name,
        recent_events_per_session=(
            args.recent_events or buy_config["recent_events_per_session"]
        ),
        topk_neighbors=args.topk_neighbors or buy_config["topk_neighbors"],
        pair_buckets=args.pair_buckets or buy_config["pair_buckets"],
        event_type_weights=event_type_weights,
        compression=config["runtime"]["parquet_compression"],
        row_group_size=config["runtime"]["parquet_row_group_size"],
        workers=config["runtime"]["workers"],
        memory_limit_gb=config["runtime"]["duckdb_memory_limit_gb"],
        source_name="buy2buy",
        allowed_event_types=(2, 3),
        pair_semantics=(
            "directed unique item pairs per session after filtering to cart/order events; "
            "each neighboring item contributes one vote per session"
        ),
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
