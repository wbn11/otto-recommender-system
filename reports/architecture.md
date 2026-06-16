# OTTO 多目标推荐系统架构

> 目标: 围绕 OTTO 的 `clicks / carts / orders` 多目标任务, 构建从多路召回到 LightGBM 精排的两阶段推荐系统。评估指标为加权 Recall@20, 权重为 clicks 0.10 / carts 0.30 / orders 0.60。

## 当前主链路

```text
multi_target_train_events.parquet
multi_target_valid_labels.parquet
        |
        v
Popular / Covisitation / DSSM multi-target recall
        |
        v
build_recall_candidates_multi_target.py
        |
        v
build_ranker_train_data_multi_target.py
        |
        v
train_ranker_multi_target.py
        |
        v
predict_ranker_multi_target.py
        |
        v
multi_target_ranker_predictions.csv
```

## 数据契约

- 训练事件: `outputs/multi_target_train_events.parquet`
  - 列: `session, aid, ts, type`
- 验证标签: `outputs/multi_target_valid_labels.parquet`
  - 列: `session, type, labels`
  - `labels` 是空格分隔的未来 aid
- 召回/排序预测输出:
  - 列: `session, type, predictions`
  - `predictions` 是空格分隔的 Top-K aid

## 召回层

- `src/recall/popular_recall_multi_target.py`
  - 生成全局热门召回。
- `src/recall/build_covis_matrix_multi_target.py`
  - 构建多目标共现矩阵。
- `src/recall/covisitation_recall_multi_target.py`
  - 根据共现矩阵生成 `(session,type)` 召回。
- `src/models/train_dssm_multi_target.py`
  - 训练 type-aware DSSM。
- `src/recall/generate_dssm_recall_multi_target.py`
  - 用 DSSM session 向量和 item embedding 做全库相似度检索。
- `src/recall/fusion_recall_multi_target.py`
  - 用 reciprocal-rank 融合 popular / covis / DSSM。
- `src/recall/build_recall_candidates_multi_target.py`
  - 合并 popular / covis / DSSM Top50 召回结果, 输出候选池。

## 精排层

- `src/rank/build_ranker_train_data_multi_target.py`
  - 读取统一候选池, 根据验证标签打 `label`, 加 item/session 统计特征和 session history 特征。
- `src/rank/train_ranker_multi_target.py`
  - 使用 LightGBM `lambdarank`, 按 `(session,type)` 分组排序, 按 session 做 train/holdout split。
- `src/rank/predict_ranker_multi_target.py`
  - 对候选池打 `ranker_score`, 每个 `(session,type)` 取 Top20。

## 离线分析

- `src/evaluation/analyze_recall_candidates_multi_target.py`
  - 输入候选池和验证标签, 计算 candidate oracle Recall@20。
- `src/evaluation/evaluate_multi_target.py`
  - 输入预测 CSV 和验证标签, 计算加权 Recall@20。

## 当前结果

- popular: Weighted Recall@20 = 0.0096
- covisitation: Weighted Recall@20 = 0.2656
- DSSM: Weighted Recall@20 = 0.1792
- fusion: Weighted Recall@20 = 0.3028
- Top50 candidate oracle: Weighted Recall@20 = 0.4058
- LightGBM ranker holdout: Weighted Recall@20 = 0.3793
- LightGBM ranker full validation: Weighted Recall@20 = 0.3858

## 后续扩展

TIGER 或其他生成式召回可以作为第四路召回源加入候选池。只要输出继续保持:

```text
session,type,predictions
```

就可以接入现有 recall candidates 和 ranker 流程。
