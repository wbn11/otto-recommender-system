import argparse
import pickle
from collections import defaultdict
from pathlib import Path

import pandas as pd
from tqdm import tqdm


EVENT_TYPES = ("clicks", "carts", "orders")
DEFAULT_TRAIN_FILE = "multi_target_train_events.csv"
DEFAULT_OUTPUT_FILE = "multi_target_covis_topk_by_type.pkl"
DEFAULT_SESSION_LIMIT = 30
DEFAULT_TOP_K = 20


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Build type-specific multi-target co-visitation matrices.")
    parser.add_argument(
        "--train-file",
        default=DEFAULT_TRAIN_FILE,
        help=f"Train events file under outputs/. Default: {DEFAULT_TRAIN_FILE}",
    )
    parser.add_argument(
        "--output-file",
        default=DEFAULT_OUTPUT_FILE,
        help=f"Output pickle file under outputs/. Default: {DEFAULT_OUTPUT_FILE}",
    )
    parser.add_argument(
        "--session-limit",
        type=int,
        default=DEFAULT_SESSION_LIMIT,
        help=f"Number of latest events per session used for co-visitation. Default: {DEFAULT_SESSION_LIMIT}",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"Number of neighbors stored per item and target type. Default: {DEFAULT_TOP_K}",
    )
    return parser.parse_args(argv)


def build_covis_topk_by_type(train_events, session_limit, top_k):
    covis_by_type = {
        event_type: defaultdict(lambda: defaultdict(int))
        for event_type in EVENT_TYPES
    }
    groups = train_events.sort_values("ts").groupby("session")

    print("Building type-specific multi-target covis matrices...")
    for _, group in tqdm(groups, total=train_events["session"].nunique()):
        events = group[["aid", "type"]].tail(session_limit).itertuples(index=False, name=None)
        events = list(events)

        for i, (aid1, type1) in enumerate(events):
            for aid2, type2 in events[i + 1:]:
                if aid1 == aid2:
                    continue

                if type2 in covis_by_type:
                    covis_by_type[type2][aid1][aid2] += 1
                if type1 in covis_by_type:
                    covis_by_type[type1][aid2][aid1] += 1

    covis_topk_by_type = {}
    for event_type, covis in covis_by_type.items():
        print(f"Building {event_type} topk...")
        covis_topk_by_type[event_type] = {
            aid: sorted(neighbors.items(), key=lambda x: x[1], reverse=True)[:top_k]
            for aid, neighbors in tqdm(covis.items())
        }

    return covis_topk_by_type


def main(argv=None):
    args = parse_args(argv)
    root = Path(__file__).resolve().parent.parent.parent
    output_dir = root / "outputs"
    train_path = output_dir / args.train_file

    if not train_path.exists():
        raise FileNotFoundError(f"Train events file not found: {train_path}")

    train_events = pd.read_csv(train_path)
    covis_topk_by_type = build_covis_topk_by_type(train_events, args.session_limit, args.top_k)

    output_path = output_dir / args.output_file
    with open(output_path, "wb") as f:
        pickle.dump(covis_topk_by_type, f)

    print(f"saved to {output_path}")
    for event_type in EVENT_TYPES:
        print(f"{event_type}: {len(covis_topk_by_type[event_type]):,} source items")
