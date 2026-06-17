import argparse
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.target_rows import load_target_rows_from_file


DEFAULT_LABELS_FILE = "multi_target_valid_labels.parquet"
DEFAULT_POPULAR_FILE = "multi_target_popular_predictions.csv"
DEFAULT_COVIS_FILE = "multi_target_covisitation_predictions.csv"
DEFAULT_DSSM_FILE = "multi_target_dssm_predictions.csv"
DEFAULT_OUTPUT_FILE = "multi_target_fusion_predictions.csv"
DEFAULT_GRID_OUTPUT_FILE = "multi_target_fusion_grid_search.csv"
DEFAULT_K = 20
TYPE_WEIGHTS = {
    "clicks": 0.10,
    "carts": 0.30,
    "orders": 0.60,
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Fuse multi-target recall predictions.")
    target_group = parser.add_mutually_exclusive_group()
    target_group.add_argument("--labels-file",
                              help=f"Validation labels file under outputs/. Default: {DEFAULT_LABELS_FILE}")
    target_group.add_argument("--test-events-file",
                              help="Test events file under outputs/. Target rows are expanded to all types.")
    parser.add_argument("--popular-file", default=DEFAULT_POPULAR_FILE)
    parser.add_argument("--covis-file", default=DEFAULT_COVIS_FILE)
    parser.add_argument("--dssm-file", default=DEFAULT_DSSM_FILE)
    parser.add_argument("--output-file", default=DEFAULT_OUTPUT_FILE)
    parser.add_argument("--grid-output-file", default=DEFAULT_GRID_OUTPUT_FILE)
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--popular-weight", type=float, default=0.1)
    parser.add_argument("--covis-weight", type=float, default=2.5)
    parser.add_argument("--dssm-weight", type=float, default=0.6)
    parser.add_argument("--grid-search", action="store_true")
    parser.add_argument("--popular-grid", default="0.1,0.2")
    parser.add_argument("--covis-grid", default="1.0,1.5,2.0")
    parser.add_argument("--dssm-grid", default="0.5,0.8,1.0,1.5")
    return parser.parse_args(argv)


def parse_items(value):
    if pd.isna(value):
        return []

    items = []
    seen = set()
    for token in str(value).split():
        if not token.isdigit() or token in seen:
            continue
        items.append(token)
        seen.add(token)
    return items


def parse_float_grid(value):
    return [float(token.strip()) for token in value.split(",") if token.strip()]


def load_predictions(output_dir, file_name):
    path = output_dir / file_name
    if not path.exists():
        raise FileNotFoundError(f"Prediction file not found: {path}")

    df = pd.read_csv(path)
    required_columns = {"session", "type", "predictions"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"{path} missing columns: {sorted(missing_columns)}")

    predictions = {}
    empty_rows = 0
    for session, event_type, prediction in zip(df["session"], df["type"], df["predictions"]):
        items = parse_items(prediction)
        if not items:
            empty_rows += 1
        predictions[(session, event_type)] = items

    return predictions, empty_rows


def load_inputs(output_dir, args):
    labels_file = args.labels_file if args.test_events_file else (args.labels_file or DEFAULT_LABELS_FILE)
    target_rows = load_target_rows_from_file(output_dir, labels_file, args.test_events_file)
    labels = None
    if not args.test_events_file:
        target_frame = pd.read_parquet(output_dir / labels_file)
        labels = target_frame if "labels" in target_frame.columns else None

    source_predictions = {}
    empty_counts = {}
    for source_name, file_name in {
        "popular": args.popular_file,
        "covis": args.covis_file,
        "dssm": args.dssm_file,
    }.items():
        source_predictions[source_name], empty_counts[source_name] = load_predictions(output_dir, file_name)

    return target_rows, labels, source_predictions, empty_counts


def add_source_scores(items, source_weight, scores):
    for rank, item in enumerate(items):
        scores[item] += source_weight / (rank + 1)


def build_fallback_items(source_predictions):
    for source_name in ("popular", "dssm", "covis"):
        for items in source_predictions[source_name].values():
            if items:
                return items
    return []


def fill_to_k(items, fallback_items, k):
    filled = list(items)
    seen = set(filled)

    for item in fallback_items:
        if len(filled) >= k:
            break
        if item in seen:
            continue
        filled.append(item)
        seen.add(item)

    return filled


def fuse_predictions(target_rows, source_predictions, weights, k):
    fallback_items = build_fallback_items(source_predictions)
    rows = []
    underfilled_before_fallback = 0
    underfilled_after_fallback = 0

    for session, event_type in tqdm(
        zip(target_rows["session"], target_rows["type"]),
        total=len(target_rows),
        desc="Fusing multi-target recalls",
        leave=False,
    ):
        key = (session, event_type)
        scores = defaultdict(float)

        for source_name, source_weight in weights.items():
            add_source_scores(source_predictions[source_name].get(key, []), source_weight, scores)

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        fused = [item for item, _ in ranked[:k]]

        if len(fused) < k:
            underfilled_before_fallback += 1
            fused = fill_to_k(fused, fallback_items, k)
        if len(fused) < k:
            underfilled_after_fallback += 1

        rows.append({
            "session": session,
            "type": event_type,
            "predictions": " ".join(fused[:k]),
        })

    predictions = pd.DataFrame(rows, columns=["session", "type", "predictions"])
    return predictions, underfilled_before_fallback, underfilled_after_fallback


def evaluate_predictions(labels, predictions, k):
    merged = labels.merge(predictions, on=["session", "type"], how="left")
    summary = {}

    for event_type, type_weight in TYPE_WEIGHTS.items():
        type_df = merged[merged["type"] == event_type]
        hits = 0
        denominator = 0

        for _, row in type_df.iterrows():
            true_items = set(int(item) for item in parse_items(row["labels"]))
            pred_items = [int(item) for item in parse_items(row["predictions"])[:k]]
            hits += len(true_items.intersection(pred_items))
            denominator += min(len(true_items), k)

        recall = hits / denominator if denominator else 0.0
        summary[event_type] = {
            "hits": hits,
            "denominator": denominator,
            "recall": recall,
            "contribution": recall * type_weight,
        }

    weighted_score = sum(row["contribution"] for row in summary.values())
    return summary, weighted_score


def print_eval_summary(weights, summary, weighted_score):
    print(
        "weights: "
        f"popular={weights['popular']}, covis={weights['covis']}, dssm={weights['dssm']}"
    )
    for event_type in ("clicks", "carts", "orders"):
        row = summary[event_type]
        print(f"{event_type}: recall={row['recall']:.4f}, hits={row['hits']:,}, total={row['denominator']:,}")
    print(f"Weighted Score: {weighted_score:.4f}")


def run_single(target_rows, labels, source_predictions, weights, output_path, k):
    predictions, underfilled_before, underfilled_after = fuse_predictions(target_rows, source_predictions, weights, k)
    predictions.to_csv(output_path, index=False)

    print(f"Fusion predictions saved to {output_path.name}")
    print(f"Output rows: {len(predictions):,}")
    print(f"Underfilled before fallback: {underfilled_before:,}")
    print(f"Underfilled after fallback: {underfilled_after:,}")
    if labels is None:
        print("No labels column found; skipped fusion evaluation.")
        return predictions, None, None

    summary, weighted_score = evaluate_predictions(labels, predictions, k)
    print_eval_summary(weights, summary, weighted_score)
    return predictions, summary, weighted_score


def run_grid_search(target_rows, labels, source_predictions, args, output_dir):
    if labels is None:
        raise ValueError("--grid-search requires a target file with a labels column.")

    rows = []
    best = None

    for popular_weight in parse_float_grid(args.popular_grid):
        for covis_weight in parse_float_grid(args.covis_grid):
            for dssm_weight in parse_float_grid(args.dssm_grid):
                weights = {
                    "popular": popular_weight,
                    "covis": covis_weight,
                    "dssm": dssm_weight,
                }
                predictions, underfilled_before, underfilled_after = fuse_predictions(
                    target_rows,
                    source_predictions,
                    weights,
                    args.k,
                )
                summary, weighted_score = evaluate_predictions(labels, predictions, args.k)
                row = {
                    "popular_weight": popular_weight,
                    "covis_weight": covis_weight,
                    "dssm_weight": dssm_weight,
                    "clicks_recall": summary["clicks"]["recall"],
                    "carts_recall": summary["carts"]["recall"],
                    "orders_recall": summary["orders"]["recall"],
                    "weighted_score": weighted_score,
                    "underfilled_before_fallback": underfilled_before,
                    "underfilled_after_fallback": underfilled_after,
                }
                rows.append(row)

                if best is None or weighted_score > best["weighted_score"]:
                    best = {
                        "weights": weights,
                        "predictions": predictions,
                        "summary": summary,
                        "weighted_score": weighted_score,
                    }

    grid_df = pd.DataFrame(rows).sort_values("weighted_score", ascending=False)
    grid_df.to_csv(output_dir / args.grid_output_file, index=False)
    best["predictions"].to_csv(output_dir / args.output_file, index=False)

    print(f"Grid search saved to {args.grid_output_file}")
    print(f"Best fusion predictions saved to {args.output_file}")
    print_eval_summary(best["weights"], best["summary"], best["weighted_score"])
    print()
    print("Top 5 weight combinations:")
    print(grid_df.head(5).to_string(index=False))


def main(argv=None):
    args = parse_args(argv)
    output_dir = Path(__file__).resolve().parents[2] / "outputs"
    target_rows, labels, source_predictions, empty_counts = load_inputs(output_dir, args)

    print(f"Target rows: {len(target_rows):,}")
    for source_name, empty_count in empty_counts.items():
        print(f"{source_name} empty prediction rows: {empty_count:,}")

    if args.grid_search:
        run_grid_search(target_rows, labels, source_predictions, args, output_dir)
    else:
        weights = {
            "popular": args.popular_weight,
            "covis": args.covis_weight,
            "dssm": args.dssm_weight,
        }
        run_single(target_rows, labels, source_predictions, weights, output_dir / args.output_file, args.k)
