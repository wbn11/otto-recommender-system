from pathlib import Path
import pandas as pd
from collections import Counter
#PRED_FILE = "popular_predictions.csv"
PRED_FILE = "covisitation_limit30_predictions.csv"
#PRED_FILE = "dssm_predictions.csv"
#PRED_FILE = "fusion_predictions.csv"
def parse_predictions(pred_str):
    """字符串转商品列表"""
    if pd.isna(pred_str):
        return []
    return list(map(int, pred_str.split()))

def recall_at_k(labels_df,predictions_df,k):
    """计算 Recall@K"""
    merged = labels_df.merge(predictions_df,on="session",how="left")

    hits = 0
    for _, row in merged.iterrows():
        label_aid = row["label_aid"]
        pred_list = parse_predictions(row["predictions"])
        if label_aid in pred_list[:k]:
            hits += 1
    recall = hits / len(merged)
    return recall

def main():
    ROOT = Path(__file__).resolve().parent.parent.parent
    labels = pd.read_csv(ROOT / "outputs" / "valid_labels.csv")
    predictions = pd.read_csv(ROOT / "outputs" / PRED_FILE)

    recall = recall_at_k(labels,predictions,k=20)
    print(f"Evaluating {PRED_FILE}...")
    print(f"Recall@20: {recall:.4f}")

