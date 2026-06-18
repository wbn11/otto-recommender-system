import argparse
from pathlib import Path

import polars as pl


DEFAULT_NROWS = 100000
DEFAULT_OUTPUT_FILE = "test_events.parquet"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Build multi-target test events from OTTO test jsonl.")
    parser.add_argument(
        "--nrows",
        type=int,
        default=DEFAULT_NROWS,
        help=f"Number of test sessions to read. 0 = all sessions. Default: {DEFAULT_NROWS}",
    )
    parser.add_argument(
        "--output-file",
        default=DEFAULT_OUTPUT_FILE,
        help=f"Output events parquet under outputs/. Default: {DEFAULT_OUTPUT_FILE}",
    )
    return parser.parse_args(argv)


def load_test_events(data_file, nrows):
    lf = pl.scan_ndjson(data_file)
    if nrows > 0:
        lf = lf.head(nrows)

    return (
        lf.explode("events")
        .unnest("events")
        .select(["session", "aid", "ts", "type"])
        .collect()
    )


def main(argv=None):
    args = parse_args(argv)
    if args.nrows < 0:
        raise ValueError("--nrows must be >= 0")

    root = Path(__file__).resolve().parents[2]
    data_file = root / "data" / "otto-recsys-test.jsonl"
    output_dir = root / "outputs"
    output_dir.mkdir(exist_ok=True)

    if not data_file.exists():
        raise FileNotFoundError(f"Test jsonl not found: {data_file}")

    events = load_test_events(data_file, args.nrows)
    events.write_parquet(output_dir / args.output_file)

    print(f"Test events saved to {args.output_file}")
    print(f"Rows: {events.height:,}")
    print(f"Sessions: {events['session'].n_unique():,}")
    print(f"Mode: {'all sessions' if args.nrows == 0 else f'{args.nrows:,} sessions'}")
