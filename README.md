# OTTO Multi-Target Session Recommendation

基于 Kaggle [OTTO Recommender System](https://www.kaggle.com/competitions/otto-recommender-system)
数据构建的全量多目标会话推荐系统。系统为每个 Session 分别预测未来的点击、加购和购买商品，
采用严格时间快照避免统计与模型训练使用未来信息。

```text
Streaming JSONL / Parquet
        ↓
Point-in-time snapshots + query/future labels
        ↓
Revisit / Popular / Type-CoVis / Buy2Buy / Time-CoVis / Attention DSSM
        ↓
Source-balanced Top100 candidate fusion
        ↓
49-dimensional point-in-time feature engineering
        ↓
Unified LightGBM LambdaRank
        ↓
click / cart / order Top20
```

## 最终结果

最终指标来自完全未参与召回统计、DSSM 或 LambdaRank 训练的 `final_valid` 窗口。

| 指标 | Click | Cart | Order | Weighted |
| :--- | ---: | ---: | ---: | ---: |
| Attention DSSM Recall@20 | 0.2691 | 0.4213 | 0.5912 | **0.5080** |
| Candidate Recall@100 | 0.5154 | 0.6707 | 0.9035 | **0.7948** |
| LambdaRank Recall@20 | 0.4489 | 0.6263 | 0.8899 | **0.7667** |

比赛指标权重为 `click=0.1, cart=0.3, order=0.6`。候选融合与最终精排均使用完整
`1,801,251` 个 final-validation Session。LightGBM 为适配 48–60GB 内存，从 ranker 窗口固定
哈希采样 100,000 个 Session 训练；全量数据仍用于召回模型、统计特征和最终评估。

历史仓库中的 `0.3858` 来自原 100k Session、Session 内 8:2 切分的玩具流程，和当前严格
时间窗口结果不可直接比较。

## 数据与时间切分

原始训练数据约 11GB：

| 统计 | 数量 |
| :--- | ---: |
| Session | 12,899,779 |
| Event | 216,716,096 |
| Unique item | 1,855,603 |
| Click | 194,720,954 |
| Cart | 16,896,191 |
| Order | 5,098,951 |

数据最大时间记为 `Tmax`，使用两个全局边界：

```text
T2 = Tmax - 7 days
T1 = T2 - 7 days

ts < T1                retrieval/ranker snapshot
T1 <= session_start<T2 LambdaRank training cohort
T2 <= session_start    final validation cohort
```

监督窗口内按照时间顺序将每个 Session 的前 80% 作为 query history，后 20% 作为 future
labels。跨越边界的 Session 按快照时间截断；任何 `ts >= cutoff` 的事件都不会进入对应快照。
所有 Popular、CoVis、DSSM vocabulary、item statistics 和 temporal features 均绑定其
point-in-time snapshot。

## 多路召回

- **Revisit**：根据历史商品的行为强度、出现频率与距 Session 结尾的距离召回重复兴趣商品。
- **Popular**：分别统计 click/cart/order 热门商品，为短历史和候选不足提供兜底。
- **Type-CoVis**：使用 click/cart/order 权重构建最近 30 个行为内的商品共现矩阵。
- **Buy2Buy**：只使用 cart/order 行为构建购买意图更强的共现关系。
- **Time-CoVis**：在 24 小时窗口内使用 `exp(-|Δt|/1h)` 对共现关系进行时间衰减。
- **Attention DSSM**：融合 item、event type、learned recency embedding，并通过目标类型条件化
  attention 生成 Session 表征；使用带重复正样本屏蔽的 batch 内负样本训练。

DSSM 使用 128 维向量、最长 50 个历史行为和 FAISS `IndexFlatIP` 精确内积检索。相较固定位置
池化 DSSM，Attention DSSM 在 final validation 的 Weighted Recall@20 从 `0.4455` 提升到
`0.5080`；主要增益来自长 Session。

六路召回先各自保留 Top200，再通过 source-balanced round-robin 构建每个
`(session, target_type)` 的 Top100 总候选池。融合结果保留每一路的 flag、rank、raw score，
候选不足时仅使用相应目标的 Popular 补齐。

## 特征与排序

统一 LambdaRank 使用 49 个特征：

| 特征组 | 数量 | 示例 |
| :--- | ---: | :--- |
| Base | 1 | target type |
| Session | 8 | 长度、行为计数、持续时间、最近行为、hour、weekday |
| Item | 4 | 总热度与 click/cart/order 计数 |
| Recall | 20 | 六路 flag/rank/score、source count、best rank |
| Interaction | 14 | revisit、位置、频次、last/recent5 CoVis 关系 |
| Temporal | 2 | 最近 1 天和 7 天热度 |

排序 group 为 `(session, target_type)`，候选命中 future label 时标记为正样本。训练前删除没有
任何正候选的 group，因为它们不能为 LambdaRank 提供成对排序监督。最终使用一个统一模型，
通过 `target_type_id` 学习三个目标的差异。

特征选择采用固定数据、固定参数的后向组消融。删除 Interaction、Recall、Item、Session、
Temporal 均使内部 NDCG@20 下降，最终保留全部 49 维；其中 Interaction 与 Recall 的消融损失
最大。详细方法见 [M10 特征选择说明](docs/m10_feature_selection.md)。

## 工程实现

- 流式解析 JSONL，统一写入紧凑类型的 ZSTD Parquet，避免完整事件表常驻内存。
- DuckDB 分桶聚合 CoVis pair、候选和特征，并允许聚合中间结果 spill-to-disk。
- 每次运行写入独立 `artifacts/{experiment_id}`，记录配置哈希、输入指纹、运行时间、峰值内存、
  软件和硬件环境，并支持安全缓存与失败续跑。
- 大规模候选、特征和预测按 Session hash 分区；最终验证以单分片为单位流式推理。
- A6000 用于 DSSM、embedding 导出和 FAISS；CoVis、特征和 LightGBM 主要使用 CPU、内存与 SSD。

## 环境与运行

目标环境为 Ubuntu/Linux、Python venv、单卡 NVIDIA RTX A6000。PyTorch 和 FAISS GPU 由安装
脚本单独安装，避免 `pip` 自动选择 CPU 版本。

```bash
bash scripts/setup_a6000.sh
source .venv/bin/activate
python src/pipeline/run.py check-environment --require-gpu
python -m pytest
python src/pipeline/run.py --list
```

分层配置位于 `configs/`，正式 Attention DSSM 与 LambdaRank 使用：

```text
configs/experiments/dssm_attention.yaml
```

所有正式任务建议通过实验运行器执行：

```bash
python src/pipeline/experiment.py run \
  --config configs/experiments/dssm_attention.yaml \
  --experiment-id EXPERIMENT_ID \
  --stage STAGE_NAME \
  --input INPUT_PATH \
  -- \
  python src/pipeline/run.py TASK [TASK_ARGS]
```

环境、SSH 复制和实验生命周期详见 [开发说明](docs/development.md)。

## 项目结构

```text
configs/        layered data and experiment configuration
docs/           development and experiment methodology
scripts/        A6000 venv bootstrap
src/data/       streaming ingestion, schemas and temporal split
src/recall/     traditional recall, DSSM retrieval and candidate fusion
src/models/     fixed-position and target-aware attention DSSM
src/features/   49-feature registry and partitioned feature builder
src/rank/       LambdaRank training, feature selection and streaming inference
src/evaluation/ offline analysis and metrics
src/pipeline/   task entrypoint and reproducible experiment runner
tests/          unit and debug integration tests
```

## 限制

- Kaggle 比赛已经结束，本项目报告严格离线时间验证结果，不宣称线上榜单成绩。
- 为控制 48–60GB 主机内存，LambdaRank 使用固定 100k Session 样本训练，而非全部 ranker cohort。
- FAISS 当前使用 FlatIP 做精确检索；尚未将 IVF Recall–Latency benchmark 纳入最终主结果。
- 未进入最终模型的 hard negative、default session embedding 和更细粒度特征搜索不作为项目结论。
- `data/`、`artifacts/`、`outputs/`、模型 checkpoint 与预测文件均不提交 Git。
