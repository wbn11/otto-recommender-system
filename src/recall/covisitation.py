
###运行一次太花时间，拆分为build_covis_matrix.py和covisitation_recall.py

from pathlib import Path
import pandas as pd
from collections import defaultdict
from tqdm import tqdm

# 根据共现矩阵生成推荐结果
def recommend(session_items, covis_topk, k):
    scores = defaultdict(float)
    for aid in session_items:
        for neighbor, count in covis_topk.get(aid, []):
            # 不推荐已经出现过的商品
            # if neighbor in session_items:
            #     continue
            scores[neighbor] += count
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [aid for aid, _ in ranked[:k]]

def main():
    ROOT = Path(__file__).resolve().parent.parent.parent
    train_events = pd.read_csv(ROOT / "outputs" / "train_events.csv")
    valid_labels = pd.read_csv(ROOT / "outputs" / "valid_labels.csv")
    # 构建商品共现矩阵
    print("Building covis matrix...")
    covis = defaultdict(lambda: defaultdict(int))
    groups = train_events.groupby("session")
    for session, group in tqdm(
        groups,
        total=train_events["session"].nunique(),
        desc="Building Co-visitation"):
        aids = group["aid"].tolist()
        for i, aid1 in enumerate(aids):
            for aid2 in aids[i+1:]:
                if aid1 == aid2:
                    continue
                covis[aid1][aid2] += 1
                covis[aid2][aid1] += 1
    print("Covis matrix done!")

    # 保留每个商品Top20共现邻居
    print("Building topk...")
    covis_topk = {}
    for aid, neighbors in covis.items():
        top_neighbors = sorted(neighbors.items(), key=lambda x: x[1], reverse=True)[:20]
        covis_topk[aid] = top_neighbors

    print("Topk done!")
    # 获取每个session历史行为并生成推荐结果
    print("Building session_items...")
    session_items = (train_events.groupby("session")["aid"].apply(list).to_dict())
    print("session_items done!")

    print("Start testing recommend...")

    predictions = []
    for session in valid_labels["session"]:
        recs = recommend(session_items[session],covis_topk,k=20)
        predictions.append({
            "session": session,
            "predictions": " ".join(map(str, recs))
        })
    pred_df = pd.DataFrame(predictions)
    pred_df.to_csv(ROOT / "outputs" / "covisitation_predictions.csv", index=False)
    print("Covisitation-based recommendations saved to covisitation_predictions.csv")


