# OTTO 多目标推荐系统架构

> 目标:围绕 OTTO 的 `clicks / carts / orders` 多目标任务,构建从多路召回到 LightGBM 精排的两阶段推荐系统。评估指标为加权 Recall@20,权重为 clicks 0.10、carts 0.30、orders 0.60。

## 当前主链路

```text
multi_target_train_events.parquet
multi_target_valid_labels.parquet
        |
        v
Popular / Covisitation / DSSM 多目标召回
        |
        v
多路候选合并 + 召回源特征 + session history 特征
        |
        v
LightGBM LambdaRank 精排
        |
        v
multi_target_ranker_predictions.csv
```

## 数据契约

- 训练事件:`outputs/multi_target_train_events.parquet`
  - 列:`session, aid, ts, type`
- 验证标签:`outputs/multi_target_valid_labels.parquet`
  - 列:`session, type, labels`
  - `labels` 是空格分隔的未来 aid
- 所有召回/排序预测输出:
  - 列:`session, type, predictions`
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
  - 用 reciprocal-rank 融合 popular、covis、DSSM。

## 精排层

- `src/rank/build_ranker_dataset_multi_target.py`
  - 将多路召回结果合并为候选表。
  - 每行是一个 `(session,type,aid)`。
  - `label` 表示该 aid 是否命中对应未来标签。
  - 特征包括召回源 rank/score、item/session 统计、DSSM raw score、session history。
- `src/rank/train_ranker_multi_target.py`
  - 使用 LightGBM `lambdarank`。
  - 按 `(session,type)` 分组排序。
  - 按 session 做 train/holdout split。
- `src/rank/predict_ranker_multi_target.py`
  - 对候选表打 `ranker_score`。
  - 每个 `(session,type)` 取 Top20。

## 当前结果

- popular:Weighted Recall@20 = 0.0096
- covisitation:Weighted Recall@20 = 0.2656
- DSSM:Weighted Recall@20 = 0.1792
- fusion:Weighted Recall@20 = 0.3028
- LightGBM ranker:Weighted Recall@20 = 0.3856

## 后续扩展

TIGER 或其他生成式召回可以作为第四路召回源加入候选表。只要输出仍保持:

```text
session,type,predictions
```

就可以接入现有 fusion 和 ranker 流程。
