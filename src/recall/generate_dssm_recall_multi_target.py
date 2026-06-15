import argparse
import pickle
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.train_dssm_multi_target import DSSM
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
    parser.add_argument("--train-file", default=DEFAULT_TRAIN_FILE,
                        help=f"Train events parquet under outputs/. Default: {DEFAULT_TRAIN_FILE}")
    parser.add_argument("--labels-file", default=DEFAULT_LABELS_FILE,
                        help=f"Validation labels parquet under outputs/. Default: {DEFAULT_LABELS_FILE}")
    parser.add_argument("--model-file", default=DEFAULT_MODEL_FILE,
                        help=f"Model weights under outputs/. Default: {DEFAULT_MODEL_FILE}")
    parser.add_argument("--item2id-file", default=DEFAULT_ITEM2ID_FILE,
                        help=f"aid->id mapping under outputs/. Default: {DEFAULT_ITEM2ID_FILE}")
    parser.add_argument("--output-file", default=DEFAULT_OUTPUT_FILE,
                        help=f"Prediction CSV under outputs/. Default: {DEFAULT_OUTPUT_FILE}")
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


def build_session_histories(train_events, item2id):
    session_items = (
        train_events.sort_values("ts")
        .groupby("session")["aid"]
        .apply(list)
        .to_dict()
    )

    histories = {}
    skipped_items = 0
    for session, items in session_items.items():
        history_ids = []
        for aid in items:
            item_id = item2id.get(aid)
            if item_id is None:
                skipped_items += 1
                continue
            history_ids.append(item_id)
        histories[session] = history_ids

    return histories, skipped_items


def pad_histories(histories):
    max_len = max(len(history) for history in histories)
    padded = [[0] * (max_len - len(history)) + history for history in histories]
    return torch.LongTensor(padded)


def recommend_sessions(model, item_embs, id2item, session_histories, valid_sessions, batch_size, k, device):
    recs_by_session = {}
    empty_sessions = 0
    topk = min(k, item_embs.size(0) - 1)

    for start in tqdm(range(0, len(valid_sessions), batch_size), desc="Generating multi-target DSSM recall"):
        batch_sessions = valid_sessions[start:start + batch_size]
        known_sessions = []
        known_histories = []

        for session in batch_sessions:
            history = session_histories.get(session, [])
            if not history:
                recs_by_session[session] = ""
                empty_sessions += 1
                continue
            known_sessions.append(session)
            known_histories.append(history)

        if not known_sessions:
            continue

        histories = pad_histories(known_histories).to(device)
        with torch.no_grad():
            session_embs = model.encode_session(histories)
            scores = session_embs @ item_embs.T
            scores[:, 0] = -1e9
            topk_ids = torch.topk(scores, k=topk, dim=1).indices.cpu().tolist()

        for session, ids in zip(known_sessions, topk_ids):
            recs = [id2item[item_id] for item_id in ids if item_id in id2item]
            recs_by_session[session] = " ".join(map(str, recs[:k]))

    return recs_by_session, empty_sessions


def build_predictions(valid_labels, recs_by_session):
    rows = []
    for session, event_type in zip(valid_labels["session"], valid_labels["type"]):
        rows.append({
            "session": session,
            "type": event_type,
            "predictions": recs_by_session.get(session, ""),
        })
    return pd.DataFrame(rows, columns=["session", "type", "predictions"])


def main(argv=None):
    args = parse_args(argv)
    output_dir = Path(__file__).resolve().parents[2] / "outputs"

    train_events, valid_labels, model_path, item2id = load_inputs(output_dir, args)
    id2item = {idx: aid for aid, idx in item2id.items()}
    print(f"items={len(item2id):,}  label rows={len(valid_labels):,}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")
    model = DSSM(num_items=len(item2id) + 1, embedding_dim=args.embedding_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    with torch.no_grad():
        item_embs = F.normalize(model.item_embedding.weight, p=2, dim=1)

    session_histories, skipped_items = build_session_histories(train_events, item2id)
    valid_sessions = valid_labels["session"].drop_duplicates().tolist()
    recs_by_session, empty_sessions = recommend_sessions(
        model=model,
        item_embs=item_embs,
        id2item=id2item,
        session_histories=session_histories,
        valid_sessions=valid_sessions,
        batch_size=args.batch_size,
        k=args.k,
        device=device,
    )

    predictions = build_predictions(valid_labels, recs_by_session)
    predictions.to_csv(output_dir / args.output_file, index=False)

    print(f"Multi-target DSSM predictions saved to {args.output_file}")
    print(f"Output rows: {len(predictions):,}")
    print(f"Unique sessions: {len(valid_sessions):,}")
    print(f"Skipped unknown history items: {skipped_items:,}")
    print(f"Empty session predictions: {empty_sessions:,}")


if __name__ == "__main__":
    main()
