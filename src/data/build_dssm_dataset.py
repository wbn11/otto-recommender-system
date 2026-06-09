from pathlib import Path
import pandas as pd
from torch.utils.data import Dataset, DataLoader

class DSSMDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]
    
def main(): 
    ROOT = Path(__file__).resolve().parent.parent.parent
    train_events = pd.read_csv(ROOT / "outputs" / "train_events.csv")
    
    #构建item2id映射
    unique_items = train_events["aid"].unique()
    item2id = {aid: idx+1 for idx, aid in enumerate(unique_items)}
    
    #Session聚合
    sessions = (train_events.sort_values("ts").groupby("session")["aid"].apply(list).to_dict())

    #构建训练对
    pairs = []
    for items in sessions.values():
        if len(items) < 2:
            continue
        for i in range(1, len(items)):
            history = items[:i]
            target = items[i]
            history_ids = [item2id[x] for x in history]
            target_id = item2id[target]
            pairs.append((history_ids, target_id))

    # print(train_events.columns)
    # print(train_events.shape)
    # print("Num items:", len(item2id))
    # print("Num training pairs:", len(pairs))
    # #最大Session长度
    # session_lengths = [len(items)for items in sessions.values()]
    # print("Max session length:",max(session_lengths))
    dataset = DSSMDataset(pairs)
    print(len(dataset))
    print(dataset[0])
    print(dataset[1])
    print(dataset[2])


if __name__ == "__main__":
    main()
