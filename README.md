# 基于多路召回与学习排序的多目标电商会话推荐系统

基于 Kaggle [OTTO Recommender System](https://www.kaggle.com/competitions/otto-recommender-system)
全量行为数据构建的两阶段推荐系统。系统根据 Session 已发生的点击、加购和购买行为，分别预测后续
最可能发生的 20 个商品，并通过严格的时间快照保证召回统计、模型训练和离线评估之间不存在未来信息泄漏。

最终链路由六路召回、Top100 候选融合、49 维特征和统一 LightGBM LambdaRank 组成。DSSM 使用
目标类型条件化注意力建模不同任务下的历史兴趣，并通过 FAISS 完成全库向量检索。

## 1. 最终结果

最终指标来自完全未参与召回统计、DSSM 训练或 LambdaRank 训练的 `final_valid` 窗口，覆盖
`1,801,251` 个 Session。比赛指标定义为：

```text
Weighted Recall@20
= 0.1 × Click Recall@20
+ 0.3 × Cart Recall@20
+ 0.6 × Order Recall@20
```

| 阶段 | Click | Cart | Order | Weighted |
| :--- | ---: | ---: | ---: | ---: |
| Attention DSSM Recall@20 | 0.2691 | 0.4213 | 0.5912 | **0.5080** |
| Candidate Recall@100 | 0.5154 | 0.6707 | 0.9035 | **0.7948** |
| LambdaRank Recall@20 | **0.4489** | **0.6263** | **0.8899** | **0.7667** |

`Candidate Recall@100` 表示真实标签是否进入最终候选池，是精排能够达到的召回上界；
`LambdaRank Recall@20` 表示候选经过排序后真正保留在前 20 位的结果。

## 2. 系统架构

```text
11 GB OTTO JSONL
        │
        ▼
Streaming ingestion
events Parquet + session metadata Parquet
        │
        ▼
Point-in-time snapshots
ranker_snapshot (ts < T1) / valid_snapshot (ts < T2)
        │
        ├──────── Popular / Revisit
        ├──────── Type-CoVis / Buy2Buy / Time-CoVis
        └──────── Target-aware Attention DSSM + FAISS FlatIP
                         │
                         ▼
              Source-balanced Top100 fusion
                         │
                         ▼
        49 point-in-time ranking features
                         │
                         ▼
             Unified LightGBM LambdaRank
                         │
                         ▼
             click / cart / order Top20
```

两套快照使用完全相同的代码和配置：

- `ranker_snapshot` 为 ranker cohort 生成候选和特征，用于训练 LambdaRank。
- `valid_snapshot` 为 final-valid cohort 生成候选和特征，只用于最终离线评估。

## 3. 数据规模与数据契约

### 3.1 全量数据

| 统计项 | 数量 |
| :--- | ---: |
| Session | 12,899,779 |
| Event | 216,716,096 |
| Unique item | 1,855,603 |
| Click | 194,720,954 |
| Cart | 16,896,191 |
| Order | 5,098,951 |

原始 JSONL 以 Session 为一行，事件嵌套在列表中。数据层逐行解析并输出两类 ZSTD Parquet：

```text
events:
session:int64, aid:int64, ts:int64, event_type:int8

sessions:
session:int64, start_ts:int64, end_ts:int64, event_count:int32,
click_count:int32, cart_count:int32, order_count:int32
```

监督数据使用列表形式的 query history 和长表标签：

```text
queries:
session:int64, aids:list<int64>, timestamps:list<int64>, event_types:list<int8>

labels:
session:int64, target_type:int8, aid:int64
```

`event_type/target_type` 使用 `1=click`、`2=cart`、`3=order`。

紧凑数值类型避免 pandas object 常驻内存；分片 Parquet 为 DuckDB 聚合、流式特征生成和失败续跑提供统一接口。

### 3.2 严格时间切分

设数据最大时间为 `Tmax`：

```text
T2 = Tmax - 7 days
T1 = T2 - 7 days

snapshot events:    ts < cutoff
ranker cohort:      T1 <= session_start < T2
final-valid cohort: T2 <= session_start <= Tmax
```

本次数据对应的边界为：

| 边界 | UTC 时间 |
| :--- | :--- |
| T1 | 2022-08-14 21:59:59.984 |
| T2 | 2022-08-21 21:59:59.984 |
| Tmax | 2022-08-28 21:59:59.984 |

快照按事件时间 `ts` 取数，监督 cohort 按 `session_start` 归属，两者职责不同。每个监督 Session
按事件时间排序，前 80% 作为 query history，后 20% 作为 future labels，切点保证 history 和 future
均至少包含一个事件。跨越时间边界的 Session 只保留当前 cutoff 之前的事件；future label 不会进入
对应的 Popular、CoVis、DSSM 词表、Item 统计或时间特征。

| 监督数据 | 可用 Session | 最终写入 Session | 用途 |
| :--- | ---: | ---: | :--- |
| Ranker cohort | 2,455,308 | 2,237,925 | 生成精排训练样本 |
| Final-valid cohort | 1,801,251 | 1,801,251 | 最终独立评估 |

切分阶段强制检查快照最大时间、两个监督集合的 Session 交集、query/label 完整性、标签重复及非法行为类型；
任一检查失败时任务立即终止。

## 4. 六路召回

每一路召回独立保存 `source_score`、`source_rank` 和来源标记。不同方法的原始分数不直接相加，
由候选融合和排序模型学习来源价值。

| 召回源 | 主要输入 | 打分方式 | 主要作用 |
| :--- | :--- | :--- | :--- |
| Popular | 时间快照内的事件 | 分目标统计 click/cart/order 热度 | 空历史、稀疏历史和候选补齐 |
| Revisit | 当前 query history | 行为权重 × 频次 × 位置衰减 | 捕获重复浏览和重复购买意图 |
| Type-CoVis | 最近 30 个行为 | click/cart/order 加权共现 | 建模通用行为关联 |
| Buy2Buy | 最近 30 个 cart/order | 每个 Session 对商品对投票一次 | 强化加购和购买关联 |
| Time-CoVis | 24 小时内的事件对 | `exp(-abs(Δt)/1h)` | 捕获时间邻近的商品关系 |
| Attention DSSM | 行为序列与目标类型 | 归一化向量内积 | 补充稠密表征和长程兴趣 |

### 4.1 Revisit

同一商品在 query history 中只形成一个候选，默认分数为：

```text
exp(-distance_to_end / 10)
× last_event_type_weight
× log1p(frequency)
```

行为权重为 `click=1`、`cart=3`、`order=6`。该召回对 OTTO 很有效，因为 future 行为中存在大量
历史商品的再次点击、加购或购买；在 final-valid 上，Revisit 单路 Weighted Recall@20 为 `0.7118`。

### 4.2 三种 CoVis

CoVis 构建采用定向商品对 `(source_aid, neighbor_aid)`，跳过 self-pair，并为每个 source item
保留 Top200 邻居。Pair 按 `source_aid` 分桶写入临时 Parquet，再由 DuckDB 分桶聚合，避免数亿
商品对同时进入内存。

- Type-CoVis 在 Session 内使用最强行为类型贡献，防止重复事件无限放大同一商品对。
- Buy2Buy 过滤 click，只保留 cart/order，用更稀疏但意图更强的关系服务 carts/orders。
- Time-CoVis 为同一 Session 内的重复商品对保留最大时间衰减贡献，突出时间上最近的一次共现。

## 5. Target-aware Attention DSSM

### 5.1 Item ID 与训练样本

每个快照独立构建词表：

```text
0 = PAD
1 = UNK
2 ... N+1 = snapshot 中出现的真实商品
```

PAD、UNK 不进入候选和 FAISS 索引。后续 query history 中未出现在训练快照的商品映射为 UNK；
未知 future label 保留原始 `aid` 参与 Recall 分母，不能伪装成 UNK 正样本。

DSSM 序列以 Parquet list column 保存。`IterableDataset` 按 row group 流式读取，并在线生成每一个
next-item 训练对；历史最多保留最近 50 个事件。

### 5.2 网络结构

对历史位置 `j` 构造：

```text
h_j = item_embedding(aid_j)
    + event_type_embedding(type_j)
    + learned_recency_embedding(recency_j)
```

目标类型 click/cart/order 的 Embedding 经线性投影形成 query，历史向量分别投影为 key/value，使用
scaled dot-product attention 得到目标条件化的历史表示：

```text
q = Wq × target_type_embedding
k_j = Wk × h_j
v_j = Wv × h_j
attention_j = softmax(q · k_j / sqrt(128))
session = L2Norm(sum(attention_j × v_j) + target_type_embedding)
item    = L2Norm(item_embedding(candidate))
score   = session · item / temperature
```

同一段历史会为 click、cart、order 生成三个不同的 Session 向量，使模型能够关注不同的行为位置和商品。

### 5.3 训练与检索

- Embedding 维度 128，batch size 2048，训练 1 epoch，启用 AMP。
- Item Embedding 使用 sparse gradient 和 SparseAdam；Attention 等稠密参数使用 AdamW。
- batch 内其他正样本作为负样本，形成 `[batch, batch]` logits。
- batch 内重复 target item 会被屏蔽，避免把另一个样本的真实正例当作负例。
- click/cart/order loss 权重为 `1/3/6`。
- 推理使用 GPU FAISS `IndexFlatIP`，索引只包含真实商品。

FlatIP 与 PyTorch exact Top20 的重合率为 `1.0`。valid 阶段包含 1,825,499 个索引商品和
5,403,753 个 `(session, target_type)` 查询组，纯 FAISS 检索耗时 241 秒，总吞吐约
22,393 query groups/s。

### 5.4 固定位置池化对比

对比实验保持词表、Item Tower、Embedding 维度、训练样本、batch 内负样本和 FAISS 检索一致。
Session Tower 从固定位置加权改为融合事件类型与 learned recency 的目标条件化注意力：

| 模型 | Pooling | Final-valid Weighted Recall@20 |
| :--- | :--- | ---: |
| Fixed-position DSSM | 越靠近序列末尾，固定权重越大 | 0.4455 |
| Target-aware Attention DSSM | 由目标类型动态计算历史权重 | **0.5080** |

Attention 相比固定位置池化提高 `6.25` 个百分点，最终候选池采用 Attention DSSM。

## 6. 候选融合

Revisit、Type-CoVis、Buy2Buy、Time-CoVis 和 DSSM 五路个性化召回各自产生 Top200，最终候选预算是
每个 `(session, target_type)` 总计 100 个，而不是每路各保留 100 个。融合使用
source-balanced round-robin：按轮次从五路结果中依次取候选，去重后继续补充；候选不足时使用
对应目标的 Popular 补齐。Popular 是否命中及其排名、分数仍作为召回来源特征保留。

融合后回填候选在所有来源中的：

- `from_source`、`source_rank`、`source_score`
- `source_count`
- `best_source_rank`

该策略不要求人为统一不同 raw score 的量纲，并保证各个性化召回源有机会进入候选池。

| Candidate K | Click | Cart | Order | Weighted |
| :--- | ---: | ---: | ---: | ---: |
| 20 | 0.4011 | 0.5794 | 0.7779 | 0.6807 |
| 50 | 0.4656 | 0.6336 | 0.8609 | 0.7532 |
| 100 | **0.5154** | **0.6707** | **0.9035** | **0.7948** |

## 7. 49 维特征与 LambdaRank

训练和推理共用同一份版本化 feature registry，并在写入和读取时检查字段名称、顺序与类型。
全局统计只读取当前 point-in-time snapshot，Session 和交互特征只读取 query history。

| 特征组 | 数量 | 代表特征 |
| :--- | ---: | :--- |
| Base | 1 | `target_type_id` |
| Session | 8 | 长度、三类行为计数、持续时间、最后行为、小时、星期 |
| Item | 4 | 总热度、click/cart/order 次数 |
| Recall | 20 | 六路 flag/rank/score、来源数、最佳来源排名 |
| Interaction | 14 | 是否见过、频次、位置、距末尾距离、last/recent5 CoVis |
| Temporal | 2 | cutoff 前最近 1 天和 7 天热度 |

排序 group 为 `(session, target_type)`，Top100 候选中命中 future label 的商品标记为 `1`，其他商品
标记为 `0`。没有任何正候选的 group 无法形成有效的 pairwise 排序监督，只在训练时删除；最终验证
仍覆盖全部 Session 和三种目标。

LambdaRank 使用一个统一模型，通过 `target_type_id` 学习 click/cart/order 的差异。为控制
48–60 GB 主机内存，从 ranker cohort 固定哈希采样 100,000 个 Session，即 300,000 个原始
ranking groups；内部训练/验证继续按 Session hash 做 9:1 划分，避免同一 Session 跨集合。

最终模型的 best iteration 为 `198`，对完整 final-valid 的 540,375,300 条候选流式打分，只在内存中
保留每组 Top20，耗时约 165 秒。

### 7.1 特征组消融

特征选择固定候选、数据、seed 和 LightGBM 参数，逐组删除特征并比较内部 NDCG@20。完整 49 维模型
的 NDCG@20 为 `0.75268791`。

| 删除特征组 | 特征数 | NDCG@20 | 相对完整模型变化 |
| :--- | ---: | ---: | ---: |
| Temporal | 47 | 0.75188473 | -0.00080318 |
| Session | 41 | 0.74933746 | -0.00335045 |
| Item | 45 | 0.74858512 | -0.00410280 |
| Recall | 29 | 0.74765680 | -0.00503111 |
| Interaction | 35 | 0.74296241 | -0.00972550 |

所有特征组均带来正向增益，最终模型保留全部 49 维。Interaction 和 Recall 特征贡献最大，说明
Session 内重复兴趣、近期商品关系以及多路召回的一致性是精排的主要信号。详细规则见
[特征选择说明](docs/feature_selection.md)。

## 8. 工程实现与运行成本

- JSONL 全程流式解析，完整事件表不进入 pandas 内存。
- CoVis pair、候选融合和特征统计按 aid/session hash 分桶，DuckDB 聚合允许 spill-to-disk。
- 候选、特征和预测按 Session 分区写 Parquet，支持分区推理和失败恢复。
- 实验目录记录配置哈希、输入指纹、Git commit、依赖版本、运行时间和峰值 RSS。
- 相同配置、命令和输入指纹命中缓存；输入内容或配置变化时自动失效。
- 新实验只创建实际使用的输出目录，不生成无关的空文件夹。

以下数据来自单张 RTX A6000 48 GB、约 60 GB 系统内存的完整运行：

| 阶段 | Ranker / Valid 耗时 | 峰值 RSS或主要设备 |
| :--- | :--- | :--- |
| 11 GB JSONL → Parquet | 293 秒 | 2.0 GB RSS |
| 全量时间切分与快照 | 31 秒 | 7.9 GB RSS |
| Attention DSSM 训练 | 409 / 723 秒 | A6000，AMP |
| Attention DSSM + FAISS 召回 | 362 / 300 秒 | A6000 |
| Top100 候选融合 | 600 / 489 秒 | 9.3 / 8.8 GB RSS |
| 49 维特征生成 | 149 / 499 秒 | 4.1 / 7.0 GB RSS |
| Final-valid LambdaRank 推理 | 165 秒 | 3.1 GB RSS |

## 9. 环境与运行

目标环境：Ubuntu/Linux、Python venv、NVIDIA RTX A6000。推荐准备至少 300 GB 可用 SSD。

### 9.1 创建环境

```bash
bash scripts/setup_a6000.sh
source .venv/bin/activate

python src/pipeline/run.py check-environment --require-gpu
python -m pytest -q
```

PyTorch CUDA 和 FAISS GPU 由安装脚本单独安装，避免 pip 自动选择 CPU wheel。项目不依赖 Conda。

### 9.2 数据位置

```text
data/otto-recsys-train.jsonl
```

比赛 test JSONL 没有标签，不参与当前严格离线实验。

### 9.3 配置

```text
configs/base.yaml                              公共参数
configs/data/debug.yaml                        10k Session smoke test
configs/data/full.yaml                         全量数据
configs/experiments/pipeline_smoke.yaml        全链路小样本检查
configs/experiments/dssm_baseline.yaml         固定位置 DSSM 对比
configs/experiments/dssm_attention.yaml        最终 Attention 方案
```

### 9.4 任务入口

```bash
python src/pipeline/run.py --list
```

主流程顺序：

```text
ingest-events
→ build-time-splits
→ build-popular-revisit
→ build-type-covis → type-covis-recall
→ build-buy2buy → buy2buy-recall
→ build-time-covis → time-covis-recall
→ prepare-dssm-data
→ train-dssm-attention
→ dssm-recall
→ fuse-candidates
→ build-features
→ train-lambdarank
→ predict-lambdarank
```

正式任务通过实验运行器记录配置、输入和日志。例如重新训练一个 Attention DSSM：

```bash
python src/pipeline/experiment.py run \
  --config configs/experiments/dssm_attention.yaml \
  --experiment-id dssm-attention-v2 \
  --stage train-attention \
  --input artifacts/dssm-data/data/dssm/ranker \
  -- \
  python src/pipeline/run.py train-dssm-attention \
  --data-dir artifacts/dssm-data/data/dssm/ranker
```

环境部署、Windows 到服务器的复制方式和实验生命周期见 [开发说明](docs/development.md)。

## 10. 产物目录

正式产物使用方法名，不使用开发阶段编号：

```text
artifacts/
  data-full/
  time-split/
  popular-revisit/
  type-covis/
  buy2buy/
  time-covis/
  dssm-data/
  dssm-baseline-ranker/
  dssm-baseline-valid/
  dssm-baseline-recall/
  dssm-attention-ranker/
  dssm-attention-valid/
  candidates-baseline/
  candidates-attention/
  features-baseline-ranker/
  features-baseline-valid/
  features-attention/
  lambdarank-baseline/
  lambdarank-attention/
  feature-selection/
```

已有正式产物视为只读；新实验使用 `-v2` 等版本后缀，避免覆盖模型、指标和运行记录。

## 11. 代码结构

```text
configs/        分层数据与实验配置
docs/           开发、部署和实验说明
scripts/        A6000 venv 环境脚本
src/data/       流式 ingestion、schema、时间切分和 DSSM 数据准备
src/recall/     Popular、Revisit、Multi-CoVis、DSSM 召回和候选融合
src/models/     Fixed-position DSSM、Attention DSSM 和流式 Dataset
src/features/   49 维特征注册表与分区特征生成
src/rank/       LambdaRank 训练、特征选择和流式推理
src/evaluation/ DSSM 历史长度诊断
src/pipeline/   统一任务入口与实验生命周期
src/utils/      配置、指纹、日志和缓存
tests/          单元测试与 debug 集成测试
```

## 12. 限制

- Kaggle 比赛已经结束，项目只报告严格时间切分的离线结果，不将其表述为线上榜单成绩。
- LambdaRank 使用固定 100k Session 训练；召回统计、DSSM 和最终评估仍使用相应窗口的全量数据。
- FAISS 使用 FlatIP 做精确检索；IVF 等近似索引未纳入最终结果。
- DSSM 无法召回训练快照词表之外的 future item，此类标签仍计入 Recall 分母，并由传统召回补充覆盖。
- Hard negative、Default Session Embedding 和 history dropout 未进入最终方案，不作为实验结论。
- `data/`、`artifacts/`、模型 checkpoint、特征和预测文件不提交 Git。
