# LightGBM 特征选择

## 目的

验证 49 维排序特征中各语义组的增量价值。该阶段只使用 `ranker_train` 窗口内部划分，
不读取 `final_valid`，避免用最终验证集反复调特征。

## 原理

单纯按 LightGBM Gain 删除特征不可靠：Gain 会偏向可切分点较多的连续特征，而且强相关
特征会互相分摊重要性。因此采用“Gain 决定尝试顺序 + 固定验证集消融决定去留”：

1. 使用全部 49 维特征训练 Full 模型；
2. 汇总 Full 模型各特征组的 Gain，低 Gain 组优先尝试删除；
3. 每次只在当前入选集合上删除一个组并重新训练；
4. 所有试验固定候选、标签、哈希采样、内部训练/验证划分、随机种子和 LightGBM 参数；
5. 若内部 `NDCG@20` 相对 Full 下降不超过 `0.0002`，接受删除，否则保留；
6. `base` 组中的 `target_type_id` 始终保留；
7. 只对最终模型运行一次完整 `final_valid`，最终仍以 `Weighted Recall@20` 决策。

使用 Full 模型作为统一基准，可防止多次微小下降累积成明显退化。

## 输入

- `artifacts/features-attention/features/ranker`
  - 固定哈希采样的 100,000 个 ranker-train Session；
  - 每个 `(session, target_type)` 100 个候选；
  - 49 维 Base、Session、Item、Recall、Interaction、Temporal 特征；
  - 标签只来自该 Session 的 future 部分。
- `configs/experiments/dssm_attention.yaml`
  - 固定随机种子、LightGBM 参数、早停规则和资源参数。

## 工作流程

```text
49 维 Full 模型
  -> 汇总各特征组 Gain
  -> 从低 Gain 到高 Gain 逐组尝试删除
  -> 固定内部验证 NDCG@20 判断接受/拒绝
  -> 保存 selected_model
  -> 在完整 final_valid 上流式预测
```

## 输出

`artifacts/feature-selection/feature_selection/lambdarank/`：

- `report.json`：完整选择过程、入选特征组、NDCG 变化和断言；
- `selection_steps.csv`：每次删除试验及接受/拒绝结果；
- `group_importance.csv`：Full 模型的特征组 Gain；
- `runs/`：Full 和每个删除试验的模型、schema、重要性与训练报告；
- `selected_model/`：最终模型、实际特征顺序、重要性和选择来源。

最终实验中删除任一非 Base 特征组都会使 NDCG@20 下降，因此保留全部 49 维特征。

## 验收

- 每个试验使用相同训练 group 与内部验证 group；
- `final_valid_not_used=true`；
- `base_group_retained=true`；
- 最终内部 NDCG 不低于 `Full NDCG - 0.0002`；
- 推理读取模型自己的特征子集和顺序；
- 完整 final validation 的每个 group 恰有 20 个不重复 aid。
