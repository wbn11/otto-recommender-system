import argparse
import pickle
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.train_dssm_multi_target import DSSM, TYPE2ID
from utils.config import load_config


_CFG = load_config()
_DSSM = _CFG.get("dssm", {})
DEFAULT_TRAIN_FILE = "multi_target_train_events.parquet"
DEFAULT_LABELS_FILE = "multi_target_valid_labels.parquet"
DEFAULT_MODEL_FILE = "dssm_model_mt.pth"
DEFAULT_ITEM2ID_FILE = "item2id_mt.pkl"
DEFAULT_OUTPUT_FILE = "multi_target_dssm_predictions.csv"
DEFAULT_K = _CFG.get("eval", {}).get("k", 20)
DEFAULT_BATCH_SIZE = _DSSM.get("batch_size", 256)
DEFAULT_EMBEDDING_DIM = _DSSM.get("embedding_dim", 128)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Generate multi-target DSSM recall predictions.")
    parser.add_argument("--train-file", default=DEFAULT_TRAIN_FILE)
    parser.add_argument("--labels-file", default=DEFAULT_LABELS_FILE)
    parser.add_argument("--model-file", default=DEFAULT_MODEL_FILE)
    parser.add_argument("--item2id-file", default=DEFAULT_ITEM2ID_FILE)
    parser.add_argument("--output-file", default=DEFAULT_OUTPUT_FILE)
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--embedding-dim", type=int, default=DEFAULT_EMBEDDING_DIM)
    return parser.parse_args(argv)


def load_inputs(output_dir, args):
    train_path = output_dir / args.train_file
    labels_path = output_dir / args.labels_file
    model_path = output_dir / args.model_file
    item2id_path = output_dir / args.item2id_file

    for path in [train_path, labels_path, model_path, item2id_path]:
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    train_events = pd.read_parquet(train_path)
    valid_labels = pd.read_parquet(labels_path)
    with open(item2id_path, "rb") as f:
        item2id = pickle.load(f)

    return train_events, valid_labels, model_path, item2id


def load_model(model_path, item2id, args, device):
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    embedding_dim = checkpoint.get("embedding_dim", args.embedding_dim) if isinstance(checkpoint, dict) else args.embedding_dim

    model = DSSM(
        num_items=len(item2id) + 1,
        embedding_dim=embedding_dim,
        num_types=len(TYPE2ID) + 1,
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def build_session_histories(train_events, item2id):
    groups = train_events.sort_values("ts").groupby("session")[["aid", "type"]]
    session_histories = {}
    session_history_types = {}
    skipped_items = 0

    for session, group in groups:
        history = []
        history_types = []
        for aid, event_type in zip(group["aid"], group["type"]):
            item_id = item2id.get(aid)
            if item_id is None:
                skipped_items += 1
                continue
            history.append(item_id)
            history_types.append(TYPE2ID.get(event_type, 0))

        session_histories[session] = history
        session_history_types[session] = history_types

    return session_histories, session_history_types, skipped_items


def pad(values):
    max_len = max(len(row) for row in values)
    padded = [[0] * (max_len - len(row)) + row for row in values]
    return torch.LongTensor(padded)


def recommend_rows(model, item_embs, id2item, session_histories, session_history_types,
                   valid_labels, batch_size, k, device):
    rows = []
    empty_rows = 0
    topk = min(k, item_embs.size(0) - 1)
    label_rows = list(zip(valid_labels["session"], valid_labels["type"]))

    for start in tqdm(range(0, len(label_rows), batch_size), desc="Generating multi-target DSSM recall"):
        batch_rows = label_rows[start:start + batch_size]
        known_rows = []
        known_histories = []
        known_history_types = []
        target_type_ids = []

        for session, event_type in batch_rows:
            history = session_histories.get(session, [])
            if not history:
                rows.append({"session": session, "type": event_type, "predictions": ""})
                empty_rows += 1
                continue

            known_rows.append((session, event_type))
            known_histories.append(history)
            known_history_types.append(session_history_types.get(session, [0] * len(history)))
            target_type_ids.append(TYPE2ID.get(event_type, 0))

        if not known_rows:
            continue

        histories = pad(known_histories).to(device)
        history_types = pad(known_history_types).to(device)
        target_types = torch.LongTensor(target_type_ids).to(device)

        with torch.no_grad():
            session_embs = model.encode_session(histories, history_types, target_types)
            scores = session_embs @ item_embs.T
            scores[:, 0] = -1e9
            topk_ids = torch.topk(scores, k=topk, dim=1).indices.cpu().tolist()

        for (session, event_type), ids in zip(known_rows, topk_ids):
            recs = [id2item[item_id] for item_id in ids if item_id in id2item]
            rows.append({
                "session": session,
                "type": event_type,
                "predictions": " ".join(map(str, recs[:k])),
            })

    return pd.DataFrame(rows, columns=["session", "type", "predictions"]), empty_rows


def main(argv=None):
    args = parse_args(argv)
    output_dir = Path(__file__).resolve().parents[2] / "outputs"

    train_events, valid_labels, model_path, item2id = load_inputs(output_dir, args)
    id2item = {idx: aid for aid, idx in item2id.items()}
    print(f"items={len(item2id):,}  label rows={len(valid_labels):,}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")
    model = load_model(model_path, item2id, args, device)
    with torch.no_grad():
        item_embs = F.normalize(model.item_embedding.weight, p=2, dim=1)

    session_histories, session_history_types, skipped_items = build_session_histories(train_events, item2id)
    predictions, empty_rows = recommend_rows(
        model=model,
        item_embs=item_embs,
        id2item=id2item,
        session_histories=session_histories,
        session_history_types=session_history_types,
        valid_labels=valid_labels,
        batch_size=args.batch_size,
        k=args.k,
        device=device,
    )
    predictions.to_csv(output_dir / args.output_file, index=False)

    print(f"Multi-target DSSM predictions saved to {args.output_file}")
    print(f"Output rows: {len(predictions):,}")
    print(f"Skipped unknown history items: {skipped_items:,}")
    print(f"Empty prediction rows: {empty_rows:,}")

