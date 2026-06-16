import argparse
import gc
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rank.common import (
    build_predictions_from_scores,
    evaluate_predictions,
    get_feature_columns,
    get_output_dir,
    print_eval_summary,
)


DEFAULT_CANDIDATES_FILE = "multi_target_ranker_train_data.parquet"
DEFAULT_LABELS_FILE = "multi_target_valid_labels.parquet"
DEFAULT_MODEL_FILE = "lgbm_ranker.txt"
DEFAULT_IMPORTANCE_FILE = "ranker_feature_importance.csv"
DEFAULT_VALID_LABELS_FILE = "multi_target_ranker_valid_labels.parquet"
DEFAULT_VALID_PRED_FILE = "multi_target_ranker_valid_predictions.csv"
DEFAULT_SPLIT_FILE = "ranker_valid_sessions.pkl"
DEFAULT_K = 20
DEFAULT_SEED = 2024


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train a LightGBM multi-target ranker.")
    parser.add_argument("--candidates-file", default=DEFAULT_CANDIDATES_FILE)
    parser.add_argument("--labels-file", default=DEFAULT_LABELS_FILE)
    parser.add_argument("--model-file", default=DEFAULT_MODEL_FILE)
    parser.add_argument("--importance-file", default=DEFAULT_IMPORTANCE_FILE)
    parser.add_argument("--valid-labels-file", default=DEFAULT_VALID_LABELS_FILE)
    parser.add_argument("--valid-pred-file", default=DEFAULT_VALID_PRED_FILE)
    parser.add_argument("--split-file", default=DEFAULT_SPLIT_FILE)
    parser.add_argument("--valid-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--num-boost-round", type=int, default=300)
    parser.add_argument("--early-stopping-rounds", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--num-leaves", type=int, default=63)
    parser.add_argument("--min-data-in-leaf", type=int, default=50)
    parser.add_argument("--feature-fraction", type=float, default=0.85)
    parser.add_argument("--bagging-fraction", type=float, default=0.85)
    parser.add_argument("--bagging-freq", type=int, default=1)
    parser.add_argument("--lambdarank-truncation-level", type=int)
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    return parser.parse_args(argv)


def load_lightgbm():
    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise ImportError(
            "lightgbm is not installed. Install it in the OTTO environment first: "
            "D:\\anaconda3\\envs\\OTTO\\python.exe -m pip install lightgbm"
        ) from exc
    return lgb


def split_sessions(candidates, valid_ratio, seed):
    sessions = candidates["session"].drop_duplicates().to_numpy()
    rng = np.random.default_rng(seed)
    rng.shuffle(sessions)
    valid_size = max(1, int(len(sessions) * valid_ratio))
    valid_sessions = set(int(session) for session in sessions[:valid_size])
    return valid_sessions


def prepare_lgb_dataset(lgb, df, feature_columns):
    df = df.sort_values(["session", "type"], kind="mergesort")
    groups = df.groupby(["session", "type"], sort=False, observed=True).size().to_numpy()
    features = df[feature_columns]
    labels = df["label"].astype("int8", copy=False)
    dataset = lgb.Dataset(
        features,
        label=labels,
        group=groups,
        feature_name=feature_columns,
        free_raw_data=False,
    )
    return dataset, df, len(df), len(groups)


def main(argv=None):
    args = parse_args(argv)
    lgb = load_lightgbm()
    output_dir = get_output_dir()

    candidates_path = output_dir / args.candidates_file
    labels_path = output_dir / args.labels_file
    if not candidates_path.exists():
        raise FileNotFoundError(f"Candidates file not found: {candidates_path}")
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_path}")

    candidates = pd.read_parquet(candidates_path)
    labels = pd.read_parquet(labels_path)
    feature_columns = get_feature_columns(candidates)
    valid_sessions = split_sessions(candidates, args.valid_ratio, args.seed)

    is_valid = candidates["session"].isin(valid_sessions)
    train_df = candidates.loc[~is_valid]
    valid_df = candidates.loc[is_valid]

    train_data, train_df, train_row_count, train_group_count = prepare_lgb_dataset(lgb, train_df, feature_columns)
    valid_data, valid_df, valid_row_count, valid_group_count = prepare_lgb_dataset(lgb, valid_df, feature_columns)
    del candidates
    del train_df
    gc.collect()

    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [args.k],
        "learning_rate": args.learning_rate,
        "num_leaves": args.num_leaves,
        "min_data_in_leaf": args.min_data_in_leaf,
        "feature_fraction": args.feature_fraction,
        "bagging_fraction": args.bagging_fraction,
        "bagging_freq": args.bagging_freq,
        "label_gain": [0, 1],
        "seed": args.seed,
        "verbosity": -1,
        "force_col_wise": True,
    }
    if args.lambdarank_truncation_level:
        params["lambdarank_truncation_level"] = args.lambdarank_truncation_level

    model = lgb.train(
        params=params,
        train_set=train_data,
        valid_sets=[valid_data],
        valid_names=["valid"],
        num_boost_round=args.num_boost_round,
        callbacks=[
            lgb.early_stopping(args.early_stopping_rounds),
            lgb.log_evaluation(period=20),
        ],
    )

    model.save_model(output_dir / args.model_file)
    importance = pd.DataFrame({
        "feature": feature_columns,
        "importance": model.feature_importance(importance_type="gain"),
    }).sort_values("importance", ascending=False)
    importance.to_csv(output_dir / args.importance_file, index=False)

    valid_df["ranker_score"] = model.predict(valid_df[feature_columns].fillna(0), num_iteration=model.best_iteration)
    valid_labels = labels[labels["session"].isin(valid_sessions)].copy()
    valid_predictions = build_predictions_from_scores(valid_df, valid_labels, "ranker_score", args.k)
    valid_labels.to_parquet(output_dir / args.valid_labels_file, index=False)
    valid_predictions.to_csv(output_dir / args.valid_pred_file, index=False)
    with open(output_dir / args.split_file, "wb") as f:
        pickle.dump(sorted(valid_sessions), f)

    summary, weighted_score = evaluate_predictions(valid_labels, valid_predictions, args.k)
    print(f"Model saved to {args.model_file}")
    print(f"Feature importance saved to {args.importance_file}")
    print(f"Train rows: {train_row_count:,}  Valid rows: {valid_row_count:,}")
    print(f"Train groups: {train_group_count:,}")
    print(f"Valid groups: {valid_group_count:,}")
    print_eval_summary("Holdout ranker Recall@20", summary, weighted_score)
