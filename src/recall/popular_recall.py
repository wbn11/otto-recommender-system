from pathlib import Path
import pandas as pd

def main():
    ROOT = Path(__file__).resolve().parent.parent.parent
    train_events = pd.read_csv(ROOT / "outputs" / "train_events.csv")
    valid_labels = pd.read_csv(ROOT / "outputs" / "valid_labels.csv")

    top20_items = (train_events["aid"].value_counts().head(20).index.tolist())
    top20_str = " ".join(map(str, top20_items)
                         )
    predictions = pd.DataFrame({
        "session": valid_labels["session"]
        })
    predictions["predictions"] = top20_str
    predictions.to_csv(ROOT / "outputs" / "popular_predictions.csv",index=False)


if __name__ == "__main__":
    main()
