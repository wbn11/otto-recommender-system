# OTTO 多目标推荐系统架构

本文档说明当前项目的主流程、数据契约和各模块职责。项目目标是基于 OTTO 数据完成 `clicks / carts / orders` 三类目标的推荐，评估指标为加权 Recall@20。

```text
clicks: 0.10
carts:  0.30
orders: 0.60
```

## 1. 总体架构

项目采用工业推荐系统里常见的两阶段结构：

```text
数据构建
  -> 多路召回
  -> 召回候选池合并
  -> LightGBM 精排
  -> 预测 / 评估 / 提交
```

召回阶段负责扩大候选覆盖率，排序阶段负责在候选池内部把更可能命中的 item 排到前 20。

## 2. Validation 流程

validation 有真实未来 label，因此可以做离线评估和排序训练。

```mermaid
flowchart TD
    subgraph S1["1. Data split"]
        A["Raw train jsonl<br/>100000 sessions"] --> B["build_validation.py<br/>session time split 8:2"]
        B --> C["train_events.parquet<br/>history events"]
        B --> D["valid_labels.parquet<br/>future labels"]
    end

    subgraph S2["2. Multi-channel recall"]
        C --> E1["popular_recall.py<br/>global popularity"]
        C --> E2["build_covis_matrix.py<br/>item co-visitation matrix"]
        E2 --> E3["covisitation_recall.py<br/>Top50 candidates"]
        C --> E4["dssm_recall.py<br/>Top50 candidates"]
    end

    subgraph S3["3. Candidate pool"]
        E1 --> F["build_recall_candidates.py<br/>merge popular + covis + DSSM"]
        E3 --> F
        E4 --> F
        F --> G["recall_candidates.parquet"]
    end

    subgraph S4["4. Optional analysis"]
        G --> H["analyze_recall_candidates.py<br/>candidate oracle"]
        D --> H
    end

    subgraph S5["5. Ranker"]
        G --> I["build_ranker_train_data.py<br/>add labels + features"]
        C --> I
        D --> I
        I --> J["train_ranker.py<br/>LightGBM LambdaRank<br/>session-level 8:2 holdout"]
        J --> K["predict_ranker.py<br/>Top20 validation predictions"]
        K --> L["evaluate.py<br/>Weighted Recall@20"]
    end

    classDef data fill:#eaf4ff,stroke:#3b82f6,color:#0f172a;
    classDef recall fill:#ecfdf3,stroke:#16a34a,color:#0f172a;
    classDef candidate fill:#f0fdfa,stroke:#0f766e,color:#0f172a;
    classDef rank fill:#fff7ed,stroke:#f97316,color:#0f172a;
    classDef eval fill:#f5f3ff,stroke:#7c3aed,color:#0f172a;

    class A,B,C,D data;
    class E1,E2,E3,E4 recall;
    class F,G candidate;
    class I,J,K rank;
    class H,L eval;
```

说明：

- 召回脚本只需要历史行为和目标行，不依赖真实 label。
- `build_ranker_train_data.py` 才会读取 `valid_labels.parquet`，给候选 item 打 `label`。
- `analyze_recall_candidates.py` 不属于主训练链路，它只用于观察候选池 oracle 上限。

## 3. Test / Submission 流程

test 没有真实 label，只能做推理和提交文件生成。

```mermaid
flowchart TD
    subgraph T1["1. Build test events"]
        A["otto-recsys-test.jsonl"] --> B["build_test_events.py"]
        B --> C["test_events.parquet"]
    end

    subgraph T2["2. Test recall"]
        C --> D1["popular_recall.py<br/>--test-events-file"]
        C --> D2["covisitation_recall.py<br/>--test-events-file, Top50"]
        C --> D3["dssm_recall.py<br/>--test-events-file, Top50"]
    end

    subgraph T3["3. Candidate and features"]
        D1 --> E["build_recall_candidates.py<br/>test candidate pool"]
        D2 --> E
        D3 --> E
        E --> F["build_ranker_inference_data.py<br/>same features, no labels"]
    end

    subgraph T4["4. Rank and submit"]
        G["lgbm_ranker.txt<br/>trained model"] --> H["predict_ranker.py<br/>Top20 test predictions"]
        F --> H
        H --> I["build_submission.py<br/>session_type, labels"]
    end

    classDef data fill:#eaf4ff,stroke:#3b82f6,color:#0f172a;
    classDef recall fill:#ecfdf3,stroke:#16a34a,color:#0f172a;
    classDef rank fill:#fff7ed,stroke:#f97316,color:#0f172a;
    classDef output fill:#f5f3ff,stroke:#7c3aed,color:#0f172a;

    class A,B,C data;
    class D1,D2,D3,E recall;
    class F,G,H rank;
    class I output;
```

test 侧不会执行评估，也不会生成 `label` 列。目标行由 test events 自动展开为：

```text
session,clicks
session,carts
session,orders
```

## 4. 数据契约

训练事件：

```text
session, aid, ts, type
```

验证标签：

```text
session, type, labels
```

召回和排序预测：

```text
session, type, predictions
```

Kaggle submission：

```text
session_type, labels
```

## 5. Pipeline 入口

所有脚本都通过 `src/pipeline/run.py` 统一执行。推荐优先使用 workflow：

```powershell
D:\anaconda3\envs\OTTO\python.exe src\pipeline\run.py --workflow validation
D:\anaconda3\envs\OTTO\python.exe src\pipeline\run.py --workflow ranker
D:\anaconda3\envs\OTTO\python.exe src\pipeline\run.py --workflow test
D:\anaconda3\envs\OTTO\python.exe src\pipeline\run.py --workflow all
```

其中：

- `validation`: 构建 validation 召回候选池，并分析 candidate oracle。
- `ranker`: 构建排序训练数据、训练 LightGBM、生成 validation 预测并评估。
- `test`: 生成 test 预测和 `submission.csv`。
- `all`: 执行 `validation + ranker`。

查看所有 workflow 和 task：

```powershell
D:\anaconda3\envs\OTTO\python.exe src\pipeline\run.py --list
```

`--list` 会按 Data / Recall / Ranker / Evaluation 分组显示，并在每一项后给出示例命令。

## 6. 召回候选池字段

`build_recall_candidates.py` 合并 popular、covisitation、DSSM 三路召回，输出一行一个候选：

```text
session, type, aid,
from_popular, popular_rank, popular_score,
from_covis, covis_rank, covis_score, covis_raw_score_norm,
from_dssm, dssm_rank, dssm_score, dssm_raw_score_norm,
source_count, min_rank, rrf_score, target_type_id
```

字段含义：

- `from_*`: 该候选是否来自对应召回源。
- `*_rank`: 该候选在对应召回源中的名次。
- `popular_score / covis_score / dssm_score`: 基于 rank 的 `1 / rank` 分数，用于 RRF 类融合特征。
- `covis_raw_score_norm`: co-visitation 原始累积分数在同一个 `(session,type)` 内的归一化值，需要传入 covis detail 文件才会生成。
- `dssm_raw_score_norm`: DSSM cosine similarity 在同一个 `(session,type)` 内的归一化值，需要传入 DSSM detail 文件才会生成。
- `source_count`: 候选被多少路召回同时命中。
- `min_rank`: 候选在所有来源中的最好名次。
- `rrf_score`: reciprocal rank fusion 分数。
- `target_type_id`: `type` 的数值编码，方便 LightGBM 使用。

## 7. 排序训练数据字段

`build_ranker_train_data.py` 在召回候选池基础上增加监督信号和统计特征：

```text
label,
item_popularity, item_click_count, item_cart_count, item_order_count,
session_len, session_click_count, session_cart_count, session_order_count,
in_session_history, session_aid_count, aid_last_pos_from_end, aid_last_type_id
```

其中：

- `label`: 当前候选 `aid` 是否在该 `(session,type)` 的未来真实 labels 中。
- `item_*`: item 在训练历史中的全局统计。
- `session_*`: 当前 session 的长度和行为类型计数。
- `history_*`: 候选 item 是否在当前 session 历史中出现过，以及最近一次出现的位置和类型。

## 8. 模块职责

### 数据层

- `build_validation.py`: 从原始训练数据切分 history 和 future labels。
- `build_test_events.py`: 读取 test jsonl，展开为事件表。

### 召回层

- `popular_recall.py`: 全局热门召回。
- `build_covis_matrix.py`: 构建 co-visitation top-k 矩阵。
- `covisitation_recall.py`: 根据 session 历史和共现矩阵召回。
- `train_dssm.py`: 训练 type-aware DSSM。
- `dssm_recall.py`: 用 DSSM session 向量做全库相似度检索。
- `fusion_recall.py`: 固定权重 RRF 融合，作为召回基线。
- `build_recall_candidates.py`: 合并多路召回结果，输出统一候选池。

### 排序层

- `build_ranker_train_data.py`: 为 validation 候选池打 label 并补充特征。
- `build_ranker_inference_data.py`: 为 test 候选池补充同样的特征，不生成 label。
- `train_ranker.py`: 训练 LightGBM LambdaRank 模型。
- `predict_ranker.py`: 对候选池打分，每个 `(session,type)` 取 Top20。

### 评估与提交

- `analyze_recall_candidates.py`: 计算候选池 oracle Recall@20。
- `evaluate.py`: 评估预测文件的 Recall@20。
- `build_submission.py`: 生成 Kaggle submission 格式。

## 9. 当前结果

| 阶段 | Weighted Recall@20 |
|---|---:|
| Popular | 0.0096 |
| Covisitation | 0.2656 |
| DSSM | 0.1792 |
| Fixed fusion | 0.3028 |
| Top50 candidate oracle | 0.4058 |
| LightGBM holdout | 0.3793 |
| LightGBM full validation | 0.3858 |

当前主结果是 `LightGBM full validation = 0.3858`。

## 10. 后续扩展

TIGER 或其他生成式召回可以作为第四路召回源加入。只要输出仍保持：

```text
session,type,predictions
```

就可以接入 `build_recall_candidates.py`，再进入现有 LightGBM 精排流程。
