from pathlib import Path
import pandas as pd

def load_data(data_file, nrows=None):
    """读取 OTTO 原始 jsonl 数据"""
    return pd.read_json(
        data_file,
        lines=True,
        nrows=nrows
    )

def expand_events(df):
    """展开嵌套 events"""
    rows = []
    for _, row in df.iterrows():
        session_id = row["session"]

        for event in row["events"]:
            rows.append({
                "session": session_id,
                "aid": event["aid"],
                "ts": event["ts"],
                "type": event["type"]
            })
    return pd.DataFrame(rows)

def main():
    ROOT = Path(__file__).resolve().parent.parent.parent
    data_file = ROOT / "data" / "otto-recsys-train.jsonl"
    df = load_data(data_file,nrows=100000)
    events_df = expand_events(df)

    # Leave-One-Out 验证集
    train_parts = []
    valid_labels = []

    for session, group in events_df.groupby("session"):

        # 长度为1的 Session 无法切分
        if len(group) < 2:
            continue

        # 历史行为作为训练数据
        history = group.iloc[:-1]

        # 最后一次行为作为标签
        label = group.iloc[-1]

        train_parts.append(history)

        valid_labels.append({
            "session": session,
            "label_aid": label["aid"],
            "label_type": label["type"]
        })

    train_events = pd.concat(train_parts,ignore_index=True)
    valid_labels = pd.DataFrame(valid_labels)

    output_dir = ROOT / "outputs"
    output_dir.mkdir(exist_ok=True)

    train_events.to_csv(output_dir / "train_events.csv",index=False)
    valid_labels.to_csv(output_dir / "valid_labels.csv",index=False)
    print(f"Train events: {len(train_events):,}")
    print(f"Valid labels: {len(valid_labels):,}")

