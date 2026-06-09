from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import time
import pickle

class DSSMDataset(Dataset):
    # 训练样本：(history_ids, target_id)
    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]
    
class DSSM(nn.Module):
    # 商品Embedding，0用于Padding
    def __init__(self,num_items,embedding_dim):
        super().__init__()
        self.item_embedding = nn.Embedding(num_items,embedding_dim,padding_idx=0)
    
    def encode_item(self,items):
        # Item Tower：Embedding + L2 Normalize
        item_emb = self.item_embedding(items)
        item_emb = F.normalize(item_emb,p=2,dim=1)
        return item_emb

    def encode_session(self,histories):
        # Position-weighted Pooling + L2 Normalize。
        history_emb = self.item_embedding(histories)
        # 过滤Padding位置
        mask = (histories != 0).float()

        seq_len = histories.size(1)
        weights = torch.arange(1,seq_len + 1,device=histories.device,dtype=torch.float)
        weights = weights.unsqueeze(0)
        weights = weights * mask
        weights = weights / weights.sum(dim=1,keepdim=True).clamp(min=1e-8)
        session_emb = (history_emb* weights.unsqueeze(-1)).sum(dim=1)
        # L2归一化
        session_emb = F.normalize(session_emb,p=2,dim=1)
        return session_emb
    
    def forward(self,histories,targets):
        # Session向量
        session_emb = self.encode_session(histories)
        # Target商品向量
        target_emb = self.encode_item(targets)
        # In-Batch Negative
        scores = session_emb @ target_emb.T
        scores = scores / 0.1
        return scores
    
def build_pairs(train_events):
    # 构建item2id映射aid -> 连续id
    unique_items = train_events["aid"].unique()
    item2id = {aid: idx+1 for idx, aid in enumerate(unique_items)}
    
    # Session聚合,按时间排序
    sessions = (train_events.sort_values("ts").groupby("session")["aid"].apply(list).to_dict())

    # 构建训练对A B C D -> [A]->B, [A,B]->C, [A,B,C]->D
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

    return pairs, item2id

def collate_fn(batch):
    histories = []
    targets = []
    for history, target in batch:
        histories.append(history)
        targets.append(target)

    # Padding histories补齐到当前Batch最大长度
    max_len = max(len(history) for history in histories)
    padded_histories = []
    for history in histories:
        padded = [0] * (max_len - len(history)) + history
        padded_histories.append(padded)

    history_tensor = torch.LongTensor(padded_histories)
    target_tensor = torch.LongTensor(targets)

    return history_tensor, target_tensor

def main():
    ROOT = Path(__file__).resolve().parent.parent.parent
    train_events = pd.read_csv(ROOT / "outputs" / "train_events.csv")

    pairs, item2id = build_pairs(train_events)
    pairs = pairs[:100000]  # 取前10万对进行测试

    dataset = DSSMDataset(pairs)
    data_loader = DataLoader(dataset,batch_size=256,shuffle=True,collate_fn=collate_fn)
    device = torch.device("cuda" if torch.cuda.is_available()else "cpu")
    model = DSSM(num_items=len(item2id)+1,embedding_dim=128).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-5)
    for epoch in range(20):
        total_loss = 0
        tqdm_loader = tqdm(data_loader,desc=f"Epoch {epoch}")
        start_time = time.time()

        for histories, targets in tqdm_loader:
            histories = histories.to(device)
            targets = targets.to(device)
            # 第i个Session匹配第i个Target
            labels = torch.arange(len(targets),device=device)

            scores = model(histories,targets)
            loss = criterion(scores,labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tqdm_loader.set_postfix({"loss": loss.item()})
            total_loss += loss.item()

        epoch_time = time.time() - start_time
        print(f"Epoch {epoch}: "f"avg_loss={total_loss/len(data_loader):.4f}, epoch_time={epoch_time:.2f}s")

    torch.save(model.state_dict(),ROOT / "outputs" / "dssm_model.pth")
    with open(ROOT / "outputs" / "item2id.pkl","wb") as f:
        pickle.dump(item2id,f)
    print("Model saved.")

if __name__ == "__main__":
    main()
