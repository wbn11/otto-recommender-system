# OTTO 推荐系统架构设计:从多路召回到生成式检索

> 目标:把当前"多路召回 + 启发式融合"的系统,先补全 **DSSM 多目标闭环**,再加入 **TIGER 生成式检索** 作为核心亮点,最终(可选)用 **LightGBM 精排** 补全工业级两阶段架构。
> 硬件:联想 R7000 / RTX 4060 Laptop(8GB 显存)。评估指标:加权 Recall@20(clicks 0.10 / carts 0.30 / orders 0.60)。

---

## 1. 现状与缺口

| 模块 | 状态 | 说明 |
|---|---|---|
| 数据 / 验证集 | ✅ | 单目标(leave-one-out)+ 多目标(`multi_target_*`)两套验证集 |
| Popular 召回 | ✅ | 单目标 + 多目标 |
| Covisitation 召回 | ✅ | 全局,多目标(分类型/融合实验收益甚微,已移除) |
| DSSM 双塔召回 | ⚠️ 仅单目标 | 只吃 `train_events.csv`,输出 `(session, predictions)`,**未接入多目标闭环** |
| 多源融合 | ❌ | 多目标侧暂无;旧 `fusion_recall.py` 单目标且硬编码 dssm |
| 精排(LTR) | ❌ | `src/rank/` 空目录,固定权重融合 ≠ 学习型精排 |
| 生成式推荐 | ❌ | 本次新增 |

**接口契约(必须遵守)**

- 训练事件:`outputs/multi_target_train_events.csv` 列 `session, aid, ts, type`
- 验证标签:`outputs/multi_target_valid_labels.csv` 列 `session, type, labels`(labels 为空格分隔 aid;每个有未来事件的 `(session,type)` 一行)
- 预测输出:`(session, type, predictions)`,predictions 为空格分隔 aid
- 评估:`evaluate-multi-target --pred-file <xxx>.csv`,按 `[session, type]` 合并算 Recall@20 加权分

---

## 2. 目标架构(三阶段)

```
会话 session
   │
   ▼
┌──────────── 多路召回 Recall ────────────┐
│ Popular   Covisitation   DSSM(多目标)   │   ← 已有 + DSSM 改造(P1)
│ TIGER 生成式检索                          │   ← 新增(P2,亮点)
└──────────────────┬───────────────────────┘
                   ▼
            候选集合并 (per session,type)
                   ▼
         LightGBM 精排 (LambdaRank)            ← 可选(P3,补全两阶段)
                   ▼
         Top-20  →  加权 Recall@20
```

生成式不是孤岛:**TIGER 既作为一路新召回,又把它的生成分数作为精排特征**,从而能在同一套 Recall@20 上做可量化的 A/B 对比。

---

## 3. P1 — DSSM 多目标改造(当务之急)

**目的**:让 DSSM 成为多目标管线里一等公民的召回通道,并进入多源融合 + 评估。

### 3.1 改造 `src/models/train_dssm.py`
- 加 `argparse`:`--train-file`(默认 `multi_target_train_events.csv`)、`--max-pairs`、`--epochs`、`--embedding-dim`、`--batch-size`、`--max-history`(截断到最近 N 个,省显存)。
- 产物改名,避免覆盖单目标:`outputs/dssm_model_mt.pth`、`outputs/item2id_mt.pkl`。
- `build_pairs` 逻辑可复用,仅把读取文件参数化。

### 3.2 新增 `src/recall/generate_dssm_recall_multi_target.py`
- 从 `multi_target_train_events.csv` 聚合 `session -> 商品序列`(按 ts 排序)。
- 每个 session 算一次 `encode_session` 向量,与全量 item embedding 算相似度取 Top-K,再按 `valid_labels` 展开到 `(session, type)`(参照 [covisitation_recall_multi_target.py:91](../src/recall/covisitation_recall_multi_target.py) 的 `build_predictions`)。
- **冷启动防护**:`item2id.get(aid)`,跳过未登录 aid(单目标版用 `item2id[x]` 会 KeyError)。
- 默认 **type-agnostic**:同一 session 三种 type 复用同一候选列表(评估按 type 分别对标签)。
- 输出 `outputs/multi_target_dssm_predictions.csv`,列 `(session, type, predictions)`。

### 3.3 新增多源多目标融合 `src/recall/fusion_recall_multi_target.py`
- 镜像 [fusion_recall.py](../src/recall/fusion_recall.py) 的 reciprocal-rank 打分(`score += weight/(rank+1)`),但 key 改为 `(session, type)`。
- 融合通道:`popular_mt` + `covis_mt`(全局多目标 covis) + `dssm_mt`,权重做成 CLI 参数。
- Popular 兜底补满 20 个。输出 `outputs/multi_target_fusion_predictions.csv`。

### 3.4 注册任务(`src/pipeline/run.py`)
新增:`train-dssm-multi-target`、`dssm-recall-multi-target`、`fusion-recall-multi-target`。

### 3.5 进阶(可选)
- **type-aware DSSM**:decoder/打分时拼一个 type embedding,或按 type 对历史加权,做出分类型候选。
- 可学习温度、采样负样本、序列加位置编码。

---

## 4. P2 — TIGER 生成式检索(核心亮点)

> 关键前提:OTTO 的 item 只有整数 aid、**无文本/类目**,因此走"行为协同 embedding → 语义ID → 生成"路线(而非文本 LLM)。

### 4.1 四个子模块
1. **Item Embedding**:先复用 DSSM 训好的 item embedding(已是协同语义);后续可升级为 gensim word2vec/item2vec(把 session 当句子)。产物 `outputs/item_emb.npy`。
2. **RQ-VAE 语义ID**(`src/models/train_rqvae.py`):
   - Encoder(MLP)→ L 级残差向量量化,每级码本大小 K(建议 L=3, K=256)。
   - 每个 item → 语义ID 三元组 `(c1,c2,c3)`;行为相似的 item 共享前缀码。
   - 冲突处理:同码 item 追加一个去重位。
   - 损失:重构 + commitment。产物 `outputs/item2semid.pkl` 及反查表。
   - 显存:模型很小,8GB 绰绰有余。
3. **生成式 seq2seq**(`src/models/train_tiger.py`):
   - 小型 Transformer(T5 风格 encoder-decoder 或 decoder-only)。
   - 输入:session 历史展平成语义ID token 序列(每个 item = L 个 token);输出:下一个 item 的 L 个语义ID token,自回归 teacher forcing。
   - 词表 = L×K + 特殊符 ≈ 数百~1k;模型 ~10–40M 参数。
   - 训练:batch 128–256,序列长 ~(30 items×3)+ 特殊符,8GB 可跑。
4. **推理**(`src/recall/generate_tiger_recall.py`):
   - beam search(beam 20–50)生成 Top-K 语义ID → 反查回 aid(一码多 item 则展开,非法码跳过)。
   - 输出 `outputs/multi_target_tiger_predictions.csv`,`(session, type, predictions)`。
   - 默认 type-agnostic;可在 decoder 起始拼一个 type token 做分类型生成。

### 4.2 集成与评估
- 单独评估 TIGER:`evaluate-multi-target --pred-file multi_target_tiger_predictions.csv`。
- 把 TIGER 作为第 4 路接入 `fusion_recall_multi_target.py`,看融合后加权分提升。
- 注册任务:`build-item-emb`、`train-rqvae`、`build-semantic-id`、`train-tiger`、`tiger-recall`。

### 4.3 RTX 4060 资源预算
- item 1.8M × emb 64 ≈ 460MB(fp32),Adam 双倍 ≈ 1.4GB,主体模型 < 0.5GB,**8GB 富余**。
- 数据子采样(如 50万~100万 session)起步,跑通后再扩。

---

## 5. P3 — LightGBM 精排(可选,补全工业两阶段)

- 每个 `(session, type)` 合并各路候选;label = 该 type 未来标签命中与否。
- 特征:各通道分数/排名、item 流行度、recency、session 长度/类型计数、item 分类型 CTR/CVR、历史交互次数、时间间隔。
- LightGBM `lambdarank`,按 `(session,type)` 分组;对 Top-N 重排。
- 这是推荐实习面试几乎必问的环节,建议在 DSSM-mt 跑通后补上。

---

## 6. P0 — 工程地基(随手可做的高性价比项)

- **polars + parquet** 替代 pandas + CSV 处理 11GB 原始数据(`expand_events` 用 `iterrows` 很慢),预处理可从几十分钟降到几分钟。
- 新建 `configs/` 统一管理超参(现全硬编码,做消融很痛苦)。
- 模型/产物统一命名规范,避免单目标与多目标产物互相覆盖。

---

## 7. 推进路线图

| 阶段 | 内容 | 预计 |
|---|---|---|
| P1 | DSSM 多目标改造 + 多源多目标融合 + 评估 | ~3–5 天 |
| P2 | TIGER:item emb → RQ-VAE 语义ID → seq2seq → beam 召回 → 融合 | ~1.5–2 周 |
| P3(可选) | LightGBM LambdaRank 精排 + 特征工程 | ~1 周 |
| P0(并行) | polars/parquet、configs、命名规范 | 随手 |

---

## 8. 验证方式(端到端)

```bash
# P1:DSSM 多目标
python src/pipeline/run.py build-multi-target-validation --nrows 100000
python src/pipeline/run.py train-dssm-multi-target
python src/pipeline/run.py dssm-recall-multi-target
python src/pipeline/run.py evaluate-multi-target --pred-file multi_target_dssm_predictions.csv
python src/pipeline/run.py fusion-recall-multi-target
python src/pipeline/run.py evaluate-multi-target --pred-file multi_target_fusion_predictions.csv

# P2:TIGER 生成式检索
python src/pipeline/run.py build-item-emb
python src/pipeline/run.py train-rqvae
python src/pipeline/run.py build-semantic-id
python src/pipeline/run.py train-tiger
python src/pipeline/run.py tiger-recall
python src/pipeline/run.py evaluate-multi-target --pred-file multi_target_tiger_predictions.csv
```

每步看 `Weighted Score`,记录到消融表:`popular → +covis → +dssm → +tiger`,展示每路召回与生成式的边际收益。
