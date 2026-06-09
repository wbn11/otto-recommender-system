import pandas as pd
from pathlib import Path
from collections import defaultdict
def load_predictions(path):
    df = pd.read_csv(path)
    pred_dict = dict(zip(df["session"].astype(str),df["predictions"].astype(str).str.split()))
    return pred_dict


def add_score(items, weight, scores):
    for rank, item in enumerate(items):
        scores[item] += (weight / (rank + 1))

def main():
    ROOT = Path(__file__).resolve().parent.parent.parent
    dssm_preds = load_predictions(ROOT / "outputs" / "dssm_predictions.csv")
    covis_preds = load_predictions(ROOT / "outputs" / "covisitation_predictions.csv")
    popular_preds = load_predictions(ROOT / "outputs" / "popular_predictions.csv")

    valid_labels = pd.read_csv(ROOT / "outputs" / "valid_labels.csv")
    predictions = []
    for session in valid_labels["session"].astype(str):
        fused = []
        scores = defaultdict(float)
        add_score(popular_preds.get(session, []),1,scores)
        add_score(covis_preds.get(session, []),2,scores)
        add_score(dssm_preds.get(session, []),5,scores)
        ranked_items = sorted(scores.items(),key=lambda x: x[1],reverse=True)
        fused = [item for item, score in ranked_items[:20]]
        predictions.append({
            "session": session,
            "predictions": " ".join(fused)
        })
    pred_df = pd.DataFrame(predictions)
    pred_df.to_csv(ROOT / "outputs" / "fusion_predictions.csv", index=False)
    print("Fusion predictions saved to fusion_predictions.csv")
        

if __name__ == "__main__":
    main()
