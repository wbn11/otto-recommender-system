import argparse
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.config import load_config


_CFG = load_config()
DEFAULT_TRAIN_FILE = "multi_target_train_events.parquet"
DEFAULT_OUTPUT_FILE = "multi_target_covis_topk.pkl"
DEFAULT_SESSION_LIMIT = _CFG.get("covis", {}).get("session_limit", 30)
DEFAULT_TOP_K = _CFG.get("covis", {}).get("top_k", 20)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Build multi-target co-visitation top-k matrix.")
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
        help=f"Number of latest items per session used for co-visitation. Default: {DEFAULT_SESSION_LIMIT}",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"Number of neighbors stored per item. Default: {DEFAULT_TOP_K}",
    )
    return parser.parse_args(argv)


def build_covis_topk(train_events, session_limit, top_k):
    covis = defaultdict(lambda: defaultdict(int))
    groups = train_events.sort_values("ts").groupby("session")

    print("Building multi-target covis matrix...")
    for _, group in tqdm(groups, total=train_events["session"].nunique()):
        aids = group["aid"].tolist()[-session_limit:]
        for i, aid1 in enumerate(aids):
            for aid2 in aids[i + 1:]:
                if aid1 == aid2:
                    continue

                covis[aid1][aid2] += 1
                covis[aid2][aid1] += 1

    print("Building topk...")
    covis_topk = {}
    for aid, neighbors in tqdm(covis.items()):
        covis_topk[aid] = sorted(neighbors.items(), key=lambda x: x[1], reverse=True)[:top_k]

    return covis_topk


def main(argv=None):
    args = parse_args(argv)
    root = Path(__file__).resolve().parent.parent.parent
    output_dir = root / "outputs"
    train_path = output_dir / args.train_file

    if not train_path.exists():
        raise FileNotFoundError(f"Train events file not found: {train_path}")

    train_events = pd.read_parquet(train_path)
    covis_topk = build_covis_topk(train_events, args.session_limit, args.top_k)

    output_path = output_dir / args.output_file
    with open(output_path, "wb") as f:
        pickle.dump(covis_topk, f)

    print(f"saved to {output_path}")
