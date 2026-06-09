from pathlib import Path
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm

class DSSM(nn.Module):
    def __init__(self,num_items,embedding_dim):
        super().__init__()
        # 商品Embedding，0用于Padding
        self.item_embedding = nn.Embedding(num_items,embedding_dim,padding_idx=0)
    
    def encode_item(self,items):
        # Item Tower
        item_emb = self.item_embedding(items)
        item_emb = F.normalize(item_emb,p=2,dim=1)
        return item_emb

    def encode_session(self,histories):
        history_emb = self.item_embedding(histories)
        mask = (histories != 0).float()
        seq_len = histories.size(1)
        weights = torch.arange(1,seq_len + 1,device=histories.device,dtype=torch.float)
        weights = weights.unsqueeze(0)
        weights = weights * mask
        weights = weights / weights.sum(dim=1,keepdim=True).clamp(min=1e-8)
        session_emb = (history_emb* weights.unsqueeze(-1)).sum(dim=1)
        session_emb = F.normalize(session_emb,p=2,dim=1)
        return session_emb
    
    def forward(self,histories,targets):
        session_emb = self.encode_session(histories)
        target_emb = self.encode_item(targets)

        scores = session_emb @ target_emb.T
        #scores = scores / 0.05
        return scores
    
def main():
    ROOT = Path(__file__).resolve().parent.parent.parent
    with open(ROOT / "outputs" / "item2id.pkl", "rb") as f:
        item2id = pickle.load(f)
    # 连续id -> 原始aid
    id2item = {idx: aid for aid, idx in item2id.items()}

    device = torch.device("cuda" if torch.cuda.is_available()else "cpu")
    model = DSSM(num_items=len(item2id)+1,embedding_dim=128).to(device)
    # 加载训练好的参数
    model.load_state_dict(torch.load(ROOT / "outputs" / "dssm_model.pth",map_location=device))
    model.eval()
    with torch.no_grad():
        # 所有商品Embedding L2归一化
        item_embs = model.item_embedding.weight
        item_embs = F.normalize(item_embs,p=2,dim=1)

    train_events = pd.read_csv(ROOT / "outputs" / "train_events.csv")
    valid_labels = pd.read_csv(ROOT / "outputs" / "valid_labels.csv")
    # session -> 商品序列
    session_items = (train_events.groupby("session")["aid"].apply(list).to_dict())
    valid_sessions = valid_labels["session"].tolist()

    predictions = []
    for session in tqdm(valid_sessions,desc="Generating DSSM Recall"):
        items = session_items[session]
        # aid -> 连续id
        history_ids = [item2id[x] for x in items]
        histories = torch.tensor([history_ids],dtype=torch.long,device=device)

        with torch.no_grad():
            session_emb = model.encode_session(histories)
            # 与所有商品计算相似度
            scores = session_emb @ item_embs.T
        scores[:, 0] = -1e9

        # for hid in history_ids:
        #     scores[0,hid] = -1e9

        topk = torch.topk(scores, k=20, dim=1).indices[0]
        recs = [id2item[idx.item()]for idx in topk]
        predictions.append({
            "session": session,
            "predictions": " ".join(map(str,recs))
        })
    pred_df = pd.DataFrame(predictions)
    pred_df.to_csv(ROOT / "outputs" / "dssm_predictions.csv",index=False)
    print("DSSM predictions saved.")
