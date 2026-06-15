import argparse
import pickle
import sys
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.config import load_config


_CFG = load_config()
_DSSM = _CFG.get("dssm", {})
DEFAULT_TRAIN_FILE = "multi_target_train_events.parquet"
DEFAULT_MODEL_FILE = "dssm_model_mt.pth"
DEFAULT_ITEM2ID_FILE = "item2id_mt.pkl"
DEFAULT_EMBEDDING_DIM = _DSSM.get("embedding_dim", 128)
DEFAULT_EPOCHS = _DSSM.get("epochs", 20)
DEFAULT_BATCH_SIZE = _DSSM.get("batch_size", 256)
DEFAULT_LR = _DSSM.get("lr", 1e-3)
DEFAULT_WEIGHT_DECAY = _DSSM.get("weight_decay", 1e-5)
DEFAULT_MAX_PAIRS = _DSSM.get("max_pairs", 100000)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train the multi-target DSSM recall model.")
    parser.add_argument("--train-file", default=DEFAULT_TRAIN_FILE,
                        help=f"Train events parquet under outputs/. Default: {DEFAULT_TRAIN_FILE}")
    parser.add_argument("--model-file", default=DEFAULT_MODEL_FILE,
                        help=f"Output model weights under outputs/. Default: {DEFAULT_MODEL_FILE}")
    parser.add_argument("--item2id-file", default=DEFAULT_ITEM2ID_FILE,
                        help=f"Output aid->id mapping under outputs/. Default: {DEFAULT_ITEM2ID_FILE}")
    parser.add_argument("--embedding-dim", type=int, default=DEFAULT_EMBEDDING_DIM)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--max-pairs", type=int, default=DEFAULT_MAX_PAIRS,
                        help="Cap number of training pairs. 0 = use all.")
    return parser.parse_args(argv)


class DSSMDataset(Dataset):
    # 训练样本：(history_ids, target_id)
    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]


class DSSM(nn.Module):
    # 商品 Embedding，0 用于 Padding
    def __init__(self, num_items, embedding_dim):
        super().__init__()
        self.item_embedding = nn.Embedding(num_items, embedding_dim, padding_idx=0)

    def encode_item(self, items):
        # Item Tower：Embedding + L2 Normalize
        item_emb = self.item_embedding(items)
        return F.normalize(item_emb, p=2, dim=1)

    def encode_session(self, histories):
        # 位置加权 Pooling（越靠后的行为权重越大）+ L2 Normalize
        history_emb = self.item_embedding(histories)
        mask = (histories != 0).float()
        seq_len = histories.size(1)
        weights = torch.arange(1, seq_len + 1, device=histories.device, dtype=torch.float).unsqueeze(0)
        weights = weights * mask
        weights = weights / weights.sum(dim=1, keepdim=True).clamp(min=1e-8)
        session_emb = (history_emb * weights.unsqueeze(-1)).sum(dim=1)
        return F.normalize(session_emb, p=2, dim=1)

    def forward(self, histories, targets):
        # In-Batch Negative：batch 内第 i 个 session 只与第 i 个 target 为正样本
        session_emb = self.encode_session(histories)
        target_emb = self.encode_item(targets)
        scores = session_emb @ target_emb.T
        return scores / 0.1  # 温度系数


def build_pairs(train_events, max_pairs=0):
    # 构建 item2id 映射：原始 aid -> 连续 id（0 留给 padding）
    unique_items = train_events["aid"].unique()
    item2id = {aid: idx + 1 for idx, aid in enumerate(unique_items)}

    # 每个 session 按时间排序，滑窗成 [a]->b, [a,b]->c, ...
    sessions = train_events.sort_values("ts").groupby("session")["aid"].apply(list).to_dict()
    pairs = []
    for items in sessions.values():
        if len(items) < 2:
            continue
        for i in range(1, len(items)):
            history_ids = [item2id[x] for x in items[:i]]
            target_id = item2id[items[i]]
            pairs.append((history_ids, target_id))
            if max_pairs and len(pairs) >= max_pairs:
                return pairs, item2id

    return pairs, item2id


def collate_fn(batch):
    # 把 batch 内 history 左侧 padding 到当前 batch 的最大长度
    histories = [history for history, _ in batch]
    targets = [target for _, target in batch]
    max_len = max(len(history) for history in histories)
    padded = [[0] * (max_len - len(history)) + history for history in histories]
    return torch.LongTensor(padded), torch.LongTensor(targets)


def main(argv=None):
    args = parse_args(argv)
    output_dir = Path(__file__).resolve().parents[2] / "outputs"
    train_events = pd.read_parquet(output_dir / args.train_file)

    pairs, item2id = build_pairs(train_events, args.max_pairs)
    print(f"items={len(item2id):,}  training pairs={len(pairs):,}")

    dataset = DSSMDataset(pairs)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    model = DSSM(num_items=len(item2id) + 1, embedding_dim=args.embedding_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    for epoch in range(args.epochs):
        total_loss = 0.0
        start_time = time.time()
        loader = tqdm(data_loader, desc=f"Epoch {epoch}")
        for histories, targets in loader:
            histories = histories.to(device)
            targets = targets.to(device)
            labels = torch.arange(len(targets), device=device)  # in-batch 正样本在对角线

            scores = model(histories, targets)
            loss = criterion(scores, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loader.set_postfix({"loss": loss.item()})
            total_loss += loss.item()

        print(f"Epoch {epoch}: avg_loss={total_loss / len(data_loader):.4f}, time={time.time() - start_time:.1f}s")

    torch.save(model.state_dict(), output_dir / args.model_file)
    with open(output_dir / args.item2id_file, "wb") as f:
        pickle.dump(item2id, f)
    print(f"Saved model -> {args.model_file}, item2id -> {args.item2id_file}")
