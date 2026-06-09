from pathlib import Path
import pandas as pd
import pickle
from collections import defaultdict
def recommend(session_items,covis_topk,k=20):
    scores = defaultdict(float)
    for idx,aid in enumerate(reversed(session_items)):
        weight = 1.0 / (idx + 1)
        for neighbor, count in (covis_topk.get(aid, [])):
            scores[neighbor] += weight*count
    ranked = sorted(scores.items(),key=lambda x: x[1],reverse=True)

    return [aid for aid, _ in ranked[:k]]

def main():
    ROOT = Path(__file__).resolve().parent.parent.parent

    train_events = pd.read_csv(ROOT / "outputs" / "train_events.csv")
    valid_labels = pd.read_csv(ROOT / "outputs" / "valid_labels.csv")
    with open(ROOT / "outputs" / "covis_topk_limit30.pkl","rb") as f:
        covis_topk = pickle.load(f)

    session_items = (train_events.groupby("session")["aid"].apply(list).to_dict())

    predictions = []

    for session in valid_labels["session"]:

        recs = recommend(session_items[session],covis_topk,20)

        predictions.append({
            "session": session,
            "predictions":" ".join(map(str, recs))
        })

    pd.DataFrame(predictions).to_csv(ROOT / "outputs" /"covisitation_limit30_predictions.csv",index=False)
    print("Covisitation-based recommendations saved to covisitation_predictions.csv")
