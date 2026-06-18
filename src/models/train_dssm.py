"""训练 type-aware DSSM 召回模型。

将 session 历史 item 和行为类型编码为 session 向量，
用 batch 内负样本学习 item embedding，供后续向量召回使用。
"""

import argparse
import pickle
import random
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


TYPE2ID = {
    "clicks": 1,
    "carts": 2,
    "orders": 3,
}

TYPE_LOSS_WEIGHTS = {
    TYPE2ID["clicks"]: 1.0,
    TYPE2ID["carts"]: 3.0,
    TYPE2ID["orders"]: 6.0,
}

_CFG = load_config()
_DSSM = _CFG.get("dssm", {})
DEFAULT_TRAIN_FILE = "train_events.parquet"
DEFAULT_MODEL_FILE = "dssm_model.pth"
DEFAULT_ITEM2ID_FILE = "item2id.pkl"
DEFAULT_EMBEDDING_DIM = _DSSM.get("embedding_dim", 128)
DEFAULT_EPOCHS = _DSSM.get("epochs", 20)
DEFAULT_BATCH_SIZE = _DSSM.get("batch_size", 256)
DEFAULT_LR = _DSSM.get("lr", 1e-3)
DEFAULT_WEIGHT_DECAY = _DSSM.get("weight_decay", 1e-5)
DEFAULT_MAX_PAIRS = 500000
DEFAULT_MAX_SEQ_LEN = 50
DEFAULT_SEED = 2024


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train the multi-target DSSM recall model.")
    parser.add_argument("--train-file", default=DEFAULT_TRAIN_FILE)
    parser.add_argument("--model-file", default=DEFAULT_MODEL_FILE)
    parser.add_argument("--item2id-file", default=DEFAULT_ITEM2ID_FILE)
    parser.add_argument("--embedding-dim", type=int, default=DEFAULT_EMBEDDING_DIM)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--max-pairs", type=int, default=DEFAULT_MAX_PAIRS,
                        help="Randomly sample this many training pairs. 0 = use all pairs.")
    parser.add_argument("--max-seq-len", type=int, default=DEFAULT_MAX_SEQ_LEN,
                        help="Keep only the latest N history events. 0 = keep full history.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args(argv)


class DSSMDataset(Dataset):
    def __init__(self, sessions, pair_indices, max_seq_len):
        self.sessions = sessions
        self.pair_indices = pair_indices
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.pair_indices)

    def __getitem__(self, idx):
        session_idx, target_pos = self.pair_indices[idx]
        items, event_types = self.sessions[session_idx]
        start = 0 if not self.max_seq_len else max(0, target_pos - self.max_seq_len)

        history_ids = items[start:target_pos]
        history_type_ids = event_types[start:target_pos]
        target_id = items[target_pos]
        target_type_id = event_types[target_pos]
        sample_weight = TYPE_LOSS_WEIGHTS.get(target_type_id, 1.0)

        return history_ids, history_type_ids, target_id, target_type_id, sample_weight


class DSSM(nn.Module):
    def __init__(self, num_items, embedding_dim, num_types=4):
        super().__init__()
        self.item_embedding = nn.Embedding(num_items, embedding_dim, padding_idx=0)
        self.type_embedding = nn.Embedding(num_types, embedding_dim, padding_idx=0)

    def encode_item(self, items):
        item_emb = self.item_embedding(items)
        return F.normalize(item_emb, p=2, dim=1)

    def encode_session(self, histories, history_types, target_types):
        # Combine item identity and behavior type in the sequence representation.
        history_emb = self.item_embedding(histories) + self.type_embedding(history_types)
        mask = (histories != 0).float()

        # Later actions get larger weights because they are closer to the target.
        seq_len = histories.size(1)
        weights = torch.arange(1, seq_len + 1, device=histories.device, dtype=torch.float).unsqueeze(0)
        weights = weights * mask
        weights = weights / weights.sum(dim=1, keepdim=True).clamp(min=1e-8)

        session_emb = (history_emb * weights.unsqueeze(-1)).sum(dim=1)
        # Inject target type so clicks/carts/orders can produce different recall vectors.
        session_emb = session_emb + self.type_embedding(target_types)
        return F.normalize(session_emb, p=2, dim=1)

    def forward(self, histories, history_types, targets, target_types):
        session_emb = self.encode_session(histories, history_types, target_types)
        target_emb = self.encode_item(targets)
        # In-batch targets act as negatives for each other.
        scores = session_emb @ target_emb.T
        return scores / 0.1


def build_item2id(train_events):
    unique_items = train_events["aid"].unique()
    return {aid: idx + 1 for idx, aid in enumerate(unique_items)}


def build_sessions(train_events, item2id):
    sessions = []
    groups = train_events.sort_values("ts").groupby("session")

    for _, group in groups:
        item_ids = [item2id[aid] for aid in group["aid"].tolist()]
        type_ids = [TYPE2ID.get(event_type, 0) for event_type in group["type"].tolist()]
        if len(item_ids) >= 2:
            sessions.append((item_ids, type_ids))

    return sessions


def sample_pair_indices(sessions, max_pairs, seed):
    pair_counts = [len(items) - 1 for items, _ in sessions]
    total_pairs = sum(pair_counts)

    if max_pairs and max_pairs < total_pairs:
        # Sample over flattened pair offsets to avoid materializing all pairs first.
        sampled_offsets = sorted(random.Random(seed).sample(range(total_pairs), max_pairs))
        pair_indices = []
        offset_cursor = 0
        sample_cursor = 0

        for session_idx, count in enumerate(pair_counts):
            next_offset = offset_cursor + count
            while sample_cursor < len(sampled_offsets) and sampled_offsets[sample_cursor] < next_offset:
                target_pos = sampled_offsets[sample_cursor] - offset_cursor + 1
                pair_indices.append((session_idx, target_pos))
                sample_cursor += 1
            offset_cursor = next_offset
            if sample_cursor >= len(sampled_offsets):
                break
    else:
        pair_indices = [
            (session_idx, target_pos)
            for session_idx, count in enumerate(pair_counts)
            for target_pos in range(1, count + 1)
        ]

    return pair_indices, total_pairs


def print_pair_stats(sessions, pair_indices, total_pairs):
    target_items = set()
    target_type_counts = {type_id: 0 for type_id in TYPE2ID.values()}

    for session_idx, target_pos in pair_indices:
        items, event_types = sessions[session_idx]
        target_items.add(items[target_pos])
        target_type_counts[event_types[target_pos]] = target_type_counts.get(event_types[target_pos], 0) + 1

    print(f"available pairs={total_pairs:,}  sampled pairs={len(pair_indices):,}")
    print(f"unique target items in sampled pairs={len(target_items):,}")
    print(
        "sampled target types: "
        + ", ".join(f"{name}={target_type_counts.get(type_id, 0):,}" for name, type_id in TYPE2ID.items())
    )


def collate_fn(batch):
    histories = [sample[0] for sample in batch]
    history_types = [sample[1] for sample in batch]
    targets = [sample[2] for sample in batch]
    target_types = [sample[3] for sample in batch]
    sample_weights = [sample[4] for sample in batch]

    max_len = max(len(history) for history in histories)
    # Left padding keeps the most recent actions aligned at the sequence end.
    padded_histories = [[0] * (max_len - len(history)) + history for history in histories]
    padded_history_types = [[0] * (max_len - len(types)) + types for types in history_types]

    return (
        torch.LongTensor(padded_histories),
        torch.LongTensor(padded_history_types),
        torch.LongTensor(targets),
        torch.LongTensor(target_types),
        torch.FloatTensor(sample_weights),
    )


def save_checkpoint(output_path, model, args):
    checkpoint = {
        "state_dict": model.state_dict(),
        "embedding_dim": args.embedding_dim,
        "type2id": TYPE2ID,
        "type_loss_weights": TYPE_LOSS_WEIGHTS,
        "max_seq_len": args.max_seq_len,
    }
    torch.save(checkpoint, output_path)


def main(argv=None):
    args = parse_args(argv)
    output_dir = Path(__file__).resolve().parents[2] / "outputs"
    train_events = pd.read_parquet(output_dir / args.train_file)

    item2id = build_item2id(train_events)
    sessions = build_sessions(train_events, item2id)
    pair_indices, total_pairs = sample_pair_indices(sessions, args.max_pairs, args.seed)

    print(f"items={len(item2id):,}  sessions={len(sessions):,}")
    print_pair_stats(sessions, pair_indices, total_pairs)

    dataset = DSSMDataset(sessions, pair_indices, args.max_seq_len)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    model = DSSM(
        num_items=len(item2id) + 1,
        embedding_dim=args.embedding_dim,
        num_types=len(TYPE2ID) + 1,
    ).to(device)
    criterion = nn.CrossEntropyLoss(reduction="none")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    for epoch in range(args.epochs):
        total_loss = 0.0
        start_time = time.time()
        loader = tqdm(data_loader, desc=f"Epoch {epoch}")
        for histories, history_types, targets, target_types, sample_weights in loader:
            histories = histories.to(device)
            history_types = history_types.to(device)
            targets = targets.to(device)
            target_types = target_types.to(device)
            sample_weights = sample_weights.to(device)
            labels = torch.arange(len(targets), device=device)

            scores = model(histories, history_types, targets, target_types)
            loss_per_sample = criterion(scores, labels)
            # Normalize by weight sum so batch composition does not change loss scale.
            loss = (loss_per_sample * sample_weights).sum() / sample_weights.sum().clamp(min=1e-8)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loader.set_postfix({"loss": loss.item()})
            total_loss += loss.item()

        print(f"Epoch {epoch}: avg_loss={total_loss / len(data_loader):.4f}, time={time.time() - start_time:.1f}s")

    save_checkpoint(output_dir / args.model_file, model, args)
    with open(output_dir / args.item2id_file, "wb") as f:
        pickle.dump(item2id, f)
    print(f"Saved model -> {args.model_file}, item2id -> {args.item2id_file}")
