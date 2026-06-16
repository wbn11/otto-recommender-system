import argparse
from pathlib import Path

import pandas as pd


TYPE_WEIGHTS = {
    "clicks": 0.10,
    "carts": 0.30,
    "orders": 0.60,
}
DEFAULT_CANDIDATES_FILE = "multi_target_recall_candidates.parquet"
DEFAULT_LABELS_FILE = "multi_target_valid_labels.parquet"
DEFAULT_K = 20


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Analyze multi-target recall candidate oracle recall.")
    parser.add_argument("--candidates-file", default=DEFAULT_CANDIDATES_FILE)
    parser.add_argument("--labels-file", default=DEFAULT_LABELS_FILE)
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    return parser.parse_args(argv)


def parse_items(value):
    if pd.isna(value):
        return []

    items = []
    seen = set()
    for token in str(value).split():
        if not token.isdigit() or token in seen:
            continue
        item = int(token)
        items.append(item)
        seen.add(token)
    return items


def load_inputs(output_dir, candidates_file, labels_file):
    candidates_path = output_dir / candidates_file
    labels_path = output_dir / labels_file
    if not candidates_path.exists():
        raise FileNotFoundError(f"Candidates file not found: {candidates_path}")
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_path}")

    candidates = pd.read_parquet(candidates_path)
    labels = pd.read_parquet(labels_path)

    required_candidate_columns = {"session", "type", "aid"}
    required_label_columns = {"session", "type", "labels"}
    missing_candidate_columns = required_candidate_columns - set(candidates.columns)
    missing_label_columns = required_label_columns - set(labels.columns)
    if missing_candidate_columns:
        raise ValueError(f"{candidates_path} missing columns: {sorted(missing_candidate_columns)}")
    if missing_label_columns:
        raise ValueError(f"{labels_path} missing columns: {sorted(missing_label_columns)}")

    return candidates, labels


def build_candidate_sets(candidates):
    candidate_sets = {}
    for (session, event_type), group in candidates.groupby(["session", "type"], sort=False, observed=True):
        candidate_sets[(int(session), event_type)] = set(group["aid"].astype(int).tolist())
    return candidate_sets


def analyze_oracle(candidates, labels, k):
    candidate_sets = build_candidate_sets(candidates)
    summary = {}

    for event_type, type_weight in TYPE_WEIGHTS.items():
        type_labels = labels[labels["type"] == event_type]
        hits = 0
        denominator = 0
        covered_groups = 0

        for session, label_value in zip(type_labels["session"], type_labels["labels"]):
            true_items = set(parse_items(label_value))
            candidate_items = candidate_sets.get((int(session), event_type), set())
            hit_count = len(true_items.intersection(candidate_items))
            if candidate_items:
                covered_groups += 1

            hits += min(hit_count, k)
            denominator += min(len(true_items), k)

        recall = hits / denominator if denominator else 0.0
        summary[event_type] = {
            "rows": len(type_labels),
            "covered_groups": covered_groups,
            "hits": hits,
            "denominator": denominator,
            "recall": recall,
            "contribution": recall * type_weight,
        }

    weighted_score = sum(row["contribution"] for row in summary.values())
    return summary, weighted_score


def print_candidate_summary(candidates, labels):
    row_count = len(candidates)
    group_count = candidates[["session", "type"]].drop_duplicates().shape[0] if row_count else 0
    label_group_count = labels[["session", "type"]].drop_duplicates().shape[0]
    unique_items = candidates["aid"].nunique() if row_count else 0
    duplicate_count = int(candidates.duplicated(["session", "type", "aid"]).sum()) if row_count else 0
    avg_candidates = row_count / group_count if group_count else 0.0

    print("Recall candidate pool")
    print(f"Rows: {row_count:,}")
    print(f"Groups: {group_count:,}")
    print(f"Label groups: {label_group_count:,}")
    print(f"Unique items: {unique_items:,}")
    print(f"Average candidates per group: {avg_candidates:.2f}")
    print(f"Duplicate (session,type,aid) rows: {duplicate_count:,}")


def print_oracle_summary(summary, weighted_score, k):
    print()
    print(f"Candidate oracle Recall@{k}")
    print(f"{'type':<8} {'rows':>8} {'covered':>8} {'hits':>8} {'total':>8} {f'recall@{k}':>10} {'weight':>8} {'contribution':>13}")
    for event_type, type_weight in TYPE_WEIGHTS.items():
        row = summary[event_type]
        print(
            f"{event_type:<8} "
            f"{row['rows']:>8,} "
            f"{row['covered_groups']:>8,} "
            f"{row['hits']:>8,} "
            f"{row['denominator']:>8,} "
            f"{row['recall']:>10.4f} "
            f"{type_weight:>8.2f} "
            f"{row['contribution']:>13.4f}"
        )
    print()
    print(f"Weighted Score: {weighted_score:.4f}")


def main(argv=None):
    args = parse_args(argv)
    output_dir = Path(__file__).resolve().parents[2] / "outputs"
    candidates, labels = load_inputs(output_dir, args.candidates_file, args.labels_file)

    print_candidate_summary(candidates, labels)
    summary, weighted_score = analyze_oracle(candidates, labels, args.k)
    print_oracle_summary(summary, weighted_score, args.k)
