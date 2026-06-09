from collections import defaultdict
from pathlib import Path

import pandas as pd


TOP_K = 20

RECALL_SOURCES = {
    "popular": {
        "file": "popular_predictions.csv",
        "weight": 1.0,
    },
    "covisitation": {
        "file": "covisitation_limit30_predictions.csv",
        "weight": 2.0,
    },
    "dssm": {
        "file": "dssm_predictions.csv",
        "weight": 5.0,
    },
}


def parse_items(prediction):
    if pd.isna(prediction):
        return []

    items = []
    seen = set()

    for token in str(prediction).split():
        if not token.isdigit() or token in seen:
            continue

        items.append(token)
        seen.add(token)

    return items


def load_predictions(path):
    if not path.exists():
        raise FileNotFoundError(f"Prediction file not found: {path}")

    df = pd.read_csv(path)
    required_columns = {"session", "predictions"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(f"{path} missing columns: {sorted(missing_columns)}")

    predictions = {}

    for session, prediction in zip(df["session"].astype(str), df["predictions"]):
        predictions[session] = parse_items(prediction)

    return predictions


def add_score(items, weight, scores):
    for rank, item in enumerate(items):
        scores[item] += (weight / (rank + 1))


def build_fallback_items(popular_preds):
    for items in popular_preds.values():
        if items:
            return items[:TOP_K]

    return []


def fill_with_fallback(items, fallback_items, k):
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


def main():
    ROOT = Path(__file__).resolve().parent.parent.parent
    output_dir = ROOT / "outputs"

    source_predictions = {
        name: load_predictions(output_dir / config["file"])
        for name, config in RECALL_SOURCES.items()
    }

    valid_labels = pd.read_csv(output_dir / "valid_labels.csv")
    valid_sessions = valid_labels["session"].astype(str).tolist()
    fallback_items = build_fallback_items(source_predictions["popular"])

    predictions = []
    underfilled_before_fallback = 0
    underfilled_after_fallback = 0
    output_lengths = []

    for session in valid_sessions:
        scores = defaultdict(float)

        for source_name, config in RECALL_SOURCES.items():
            items = source_predictions[source_name].get(session, [])
            add_score(items, config["weight"], scores)

        ranked_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        fused = [item for item, _ in ranked_items[:TOP_K]]

        if len(fused) < TOP_K:
            underfilled_before_fallback += 1
            fused = fill_with_fallback(fused, fallback_items, TOP_K)

        if len(fused) < TOP_K:
            underfilled_after_fallback += 1

        output_lengths.append(len(fused))
        predictions.append({
            "session": session,
            "predictions": " ".join(fused)
        })

    pred_df = pd.DataFrame(predictions)
    pred_df.to_csv(output_dir / "fusion_predictions.csv", index=False)

    avg_output_length = sum(output_lengths) / len(output_lengths) if output_lengths else 0.0

    print("Fusion predictions saved to fusion_predictions.csv")
    print(f"Output sessions: {len(predictions):,}")
    print(f"Average output length: {avg_output_length:.2f}")
    print(f"Underfilled before fallback: {underfilled_before_fallback:,}")
    print(f"Underfilled after fallback: {underfilled_after_fallback:,}")

    for source_name, preds in source_predictions.items():
        missing_or_empty = sum(1 for session in valid_sessions if not preds.get(session))
        print(f"{source_name} missing or empty sessions: {missing_or_empty:,}")


