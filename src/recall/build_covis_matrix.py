from pathlib import Path
import pandas as pd
import pickle
from collections import defaultdict
from tqdm import tqdm

def main():
    ROOT = Path(__file__).resolve().parent.parent.parent
    train_events = pd.read_csv(ROOT / "outputs" / "train_events.csv")

    print("Building covis matrix...")
    covis = defaultdict(lambda: defaultdict(int))
    groups = train_events.groupby("session")
    for _, group in tqdm(groups,total=train_events["session"].nunique()):
        aids = group["aid"].tolist()[-30:]
        for i, aid1 in enumerate(aids):
            for aid2 in aids[i + 1:]:
                if aid1 == aid2:
                    continue
                covis[aid1][aid2] += 1
                covis[aid2][aid1] += 1

    print("Building topk...")
    covis_topk = {}
    for aid, neighbors in tqdm(covis.items()):
        covis_topk[aid] = sorted(neighbors.items(),key=lambda x: x[1],reverse=True)[:20]
    output_path = (ROOT / "outputs" / "covis_topk.pkl")
    with open(output_path, "wb") as f:
        pickle.dump(covis_topk, f)

    print(f"saved to {output_path}")

