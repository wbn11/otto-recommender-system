from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def load_data(data_file: Path, nrows: int) -> pd.DataFrame:
    """
    读取OTTO原始jsonl数据

    Parameters
    ----------
    data_file : Path
        数据文件路径
    nrows : int
        仅读取前nrows个session，便于调试

    Returns
    -------
    pd.DataFrame
        原始DataFrame
    """
    return pd.read_json(
        data_file,
        lines=True,
        nrows=nrows
    )


def expand_events(df: pd.DataFrame) -> pd.DataFrame:
    """
    将嵌套events展开为标准格式

    原始:
    session | events

    转换后:
    session | aid | ts | type
    """

    rows = []

    for _, row in df.iterrows():

        session_id = row["session"]

        for event in row["events"]:

            rows.append(
                {
                    "session": session_id,
                    "aid": event["aid"],
                    "ts": event["ts"],
                    "type": event["type"]
                }
            )

    return pd.DataFrame(rows)


def basic_statistics(events_df: pd.DataFrame):
    """
    输出基础统计信息
    """

    print("=" * 60)
    print("OTTO Dataset EDA")
    print("=" * 60)

    print("\n前5条记录:")
    print(events_df.head())

    print("\n数据维度:")
    print(events_df.shape)

    print("\nSession数量:")
    print(events_df["session"].nunique())

    print("\n商品数量(AID):")
    print(events_df["aid"].nunique())

    print("\n行为类型统计:")
    print(events_df["type"].value_counts())

    print("\n行为类型占比(%):")
    behavior_ratio = (
        events_df["type"]
        .value_counts(normalize=True)
        * 100
    ).round(2)

    print(behavior_ratio)

    # =========================
    # Session长度统计
    # =========================

    session_len = events_df.groupby("session").size()

    print("\nSession长度统计:")
    print(session_len.describe())

    print("\n最长的10个Session:")
    print(
        session_len
        .sort_values(ascending=False)
        .head(10)
    )

    # =========================
    # 商品热度统计
    # =========================

    aid_cnt = events_df["aid"].value_counts()

    print("\n商品热度统计:")
    print(aid_cnt.describe())

    print("\nTop20热门商品:")
    print(aid_cnt.head(20))

    return session_len, aid_cnt, behavior_ratio


def plot_session_length(session_len: pd.Series):
    """
    Session长度分布图
    """

    plt.figure(figsize=(8, 5))

    session_len.hist(bins=50)

    plt.title("Session Length Distribution")
    plt.xlabel("Session Length")
    plt.ylabel("Count")

    plt.tight_layout()
    plt.show()


def plot_session_boxplot(session_len: pd.Series):
    """
    Session长度箱线图
    """

    plt.figure(figsize=(6, 5))

    plt.boxplot(session_len)

    plt.title("Session Length Boxplot")

    plt.tight_layout()
    plt.show()


def plot_item_popularity(aid_cnt: pd.Series):
    """
    商品热度分布图
    """

    plt.figure(figsize=(8, 5))

    # 截断极热门商品
    aid_cnt.clip(upper=50).hist(bins=50)

    plt.title("Item Popularity Distribution")
    plt.xlabel("Interaction Count")
    plt.ylabel("Item Count")

    plt.tight_layout()
    plt.show()


def plot_behavior_ratio(behavior_ratio: pd.Series):
    """
    行为类型占比图
    """

    plt.figure(figsize=(6, 6))

    behavior_ratio.plot(
        kind="pie",
        autopct="%1.1f%%"
    )

    plt.ylabel("")
    plt.title("Behavior Type Distribution")

    plt.tight_layout()
    plt.show()


def main():

    # 项目根目录
    ROOT = Path(__file__).resolve().parent.parent.parent

    # 数据路径
    data_file = ROOT / "data" / "otto-recsys-train.jsonl"

    print(f"读取数据: {data_file}")

    # 读取前1000个Session
    df = load_data(
        data_file,
        nrows=10000
    )

    # 展开events
    events_df = expand_events(df)

    # 基础统计
    session_len, aid_cnt, behavior_ratio = basic_statistics(events_df)

    # Session长度直方图
    plot_session_length(session_len)

    # Session长度箱线图
    plot_session_boxplot(session_len)

    # 商品热度分布图
    plot_item_popularity(aid_cnt)

    # 行为类型占比图
    plot_behavior_ratio(behavior_ratio)


