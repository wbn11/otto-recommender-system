import argparse
import pickle
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.train_dssm import DSSM, TYPE2ID
from utils.config import load_config
from utils.target_rows import load_target_rows_from_file


_CFG = load_config()
_DSSM = _CFG.get("dssm", {})
DEFAULT_TRAIN_FILE = "train_events.parquet"
DEFAULT_LABELS_FILE = "valid_labels.parquet"
DEFAULT_MODEL_FILE = "dssm_model.pth"
DEFAULT_ITEM2ID_FILE = "item2id.pkl"
DEFAULT_OUTPUT_FILE = "dssm_predictions.csv"
DEFAULT_K = _CFG.get("eval", {}).get("k", 20)
DEFAULT_BATCH_SIZE = _DSSM.get("batch_size", 256)
DEFAULT_EMBEDDING_DIM = _DSSM.get("embedding_dim", 128)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Generate multi-target DSSM recall predictions.")
    parser.add_argument("--train-file", default=DEFAULT_TRAIN_FILE)
    target_group = parser.add_mutually_exclusive_group()
    target_group.add_argument("--labels-file",
                              help=f"Validation labels file under outputs/. Default: {DEFAULT_LABELS_FILE}")
    target_group.add_argument("--test-events-file",
                              help="Test events file under outputs/. Target rows are expanded to all types.")
    parser.add_argument("--model-file", default=DEFAULT_MODEL_FILE)
    parser.add_argument("--item2id-file", default=DEFAULT_ITEM2ID_FILE)
    parser.add_argument("--output-file", default=DEFAULT_OUTPUT_FILE)
    parser.add_argument("--detail-file", help="Optional parquet output with columns session,type,aid,rank,dssm_score.")
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--embedding-dim", type=int, default=DEFAULT_EMBEDDING_DIM)
    return parser.parse_args(argv)


def load_inputs(output_dir, args):
    train_path = output_dir / args.train_file
    labels_file = args.labels_file if args.test_events_file else (args.labels_file or DEFAULT_LABELS_FILE)
    model_path = output_dir / args.model_file
    item2id_path = output_dir / args.item2id_file

    for path in [train_path, model_path, item2id_path]:
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    train_events = pd.read_parquet(train_path)
    target_rows = load_target_rows_from_file(output_dir, labels_file, args.test_events_file)
    with open(item2id_path, "rb") as f:
        item2id = pickle.load(f)

    return train_events, target_rows, model_path, item2id


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
                   target_rows, batch_size, k, device):
    rows = []
    detail_rows = {"session": [], "type": [], "aid": [], "rank": [], "dssm_score": []}
    empty_rows = 0
    topk = min(k, item_embs.size(0) - 1)
    target_row_values = list(zip(target_rows["session"], target_rows["type"]))

    for start in tqdm(range(0, len(target_row_values), batch_size), desc="Generating multi-target DSSM recall"):
        batch_rows = target_row_values[start:start + batch_size]
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
            topk_values, topk_indices = torch.topk(scores, k=topk, dim=1)
            topk_scores = topk_values.cpu().tolist()
            topk_ids = topk_indices.cpu().tolist()

        for (session, event_type), ids, score_values in zip(known_rows, topk_ids, topk_scores):
            recs = []
            for rank, (item_id, score) in enumerate(zip(ids, score_values), start=1):
                if item_id not in id2item:
                    continue
                aid = id2item[item_id]
                recs.append(aid)
                detail_rows["session"].append(session)
                detail_rows["type"].append(event_type)
                detail_rows["aid"].append(aid)
                detail_rows["rank"].append(rank)
                detail_rows["dssm_score"].append(score)
            rows.append({
                "session": session,
                "type": event_type,
                "predictions": " ".join(map(str, recs[:k])),
            })

    predictions = pd.DataFrame(rows, columns=["session", "type", "predictions"])
    details = pd.DataFrame(detail_rows, columns=["session", "type", "aid", "rank", "dssm_score"])
    if not details.empty:
        details["rank"] = details["rank"].astype("int16")
        details["dssm_score"] = details["dssm_score"].astype("float32")
    return predictions, details, empty_rows


def main(argv=None):
    args = parse_args(argv)
    output_dir = Path(__file__).resolve().parents[2] / "outputs"

    train_events, target_rows, model_path, item2id = load_inputs(output_dir, args)
    id2item = {idx: aid for aid, idx in item2id.items()}
    print(f"items={len(item2id):,}  target rows={len(target_rows):,}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")
    model = load_model(model_path, item2id, args, device)
    with torch.no_grad():
        item_embs = F.normalize(model.item_embedding.weight, p=2, dim=1)

    session_histories, session_history_types, skipped_items = build_session_histories(train_events, item2id)
    predictions, details, empty_rows = recommend_rows(
        model=model,
        item_embs=item_embs,
        id2item=id2item,
        session_histories=session_histories,
        session_history_types=session_history_types,
        target_rows=target_rows,
        batch_size=args.batch_size,
        k=args.k,
        device=device,
    )
    predictions.to_csv(output_dir / args.output_file, index=False)
    if args.detail_file:
        details.to_parquet(output_dir / args.detail_file, index=False)
        print(f"Multi-target DSSM details saved to {args.detail_file}")
        print(f"Detail rows: {len(details):,}")

    print(f"Multi-target DSSM predictions saved to {args.output_file}")
    print(f"Output rows: {len(predictions):,}")
    print(f"Skipped unknown history items: {skipped_items:,}")
    print(f"Empty prediction rows: {empty_rows:,}")
