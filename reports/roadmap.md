# OTTO 项目进度跟踪 Roadmap

> 配套设计文档:[architecture.md](architecture.md)
> 指标:加权 Recall@20(clicks 0.10 / carts 0.30 / orders 0.60)
> 硬件:RTX 4060 Laptop(8GB)
> 运行环境:conda env `OTTO`(Python 3.10,torch 2.12+cu126),用 `D:\anaconda3\envs\OTTO\python.exe` 跑

## 总体进度

- [x] **P0** 工程地基(提速 + 配置)
- [x] **P1** DSSM 多目标改造 + 多源融合
- [ ] **P2** TIGER 生成式检索(核心亮点)
- [x] **P3** LightGBM 精排(补全两阶段,可选)
- [ ] **P4** 收尾:消融实验 + 文档 + 可视化

| 阶段 | 预计 | 依赖 | 状态 |
|---|---|---|---|
| P0 | 1–2天 | 无 | ✅ 完成 |
| P1 | 3–5天 | P0(可选) | ✅ 完成 |
| P2 | 1.5–2周 | P1 | ⬜ 未开始 |
| P3 | ~1周 | P1 | ✅ 完成 |
| P4 | 3–5天 | P1–P3 | ⬜ 未开始 |

---

## 已完成(基线)

- [x] 单目标 + 多目标验证集构建
- [x] Popular 召回(单 / 多目标)
- [x] Covisitation 召回(全局;分类型/融合已移除)
- [x] DSSM 双塔召回(单目标 + 多目标 type-aware)
- [x] 加权 Recall@20 评估

---

## P0 · 工程地基 ✅(多目标主链路)
> 目标:消除 I/O 瓶颈、统一超参管理。已完成多目标主链路;单目标(leave-one-out)旧链路暂留 CSV,待 P1 重做 DSSM 时一并迁移。

- [x] ① `expand_events` 的 `iterrows` → polars `scan_ndjson` 向量化展开 + 写 parquet(`src/data/build_multi_target_validation.py`)
- [x] ① 新增 `polars` / `pyarrow` / `pyyaml` 依赖到 `requirements.txt`
- [x] ① 多目标下游脚本改读 parquet(popular / covis build / covis recall / evaluate)
- [x] ② 新建 `configs/default.yaml` + `src/utils/config.py` 加载器,接入多目标 build / covis / evaluate 默认值
- [x] ③ 命名规范:中间表 `.parquet`、预测 `.csv`、矩阵 `.pkl`、多目标加 `multi_target_` 前缀
- [ ] (留待 P1)单目标 leave-one-out 链路迁移 parquet

**产出**:`outputs/multi_target_*.parquet`、`configs/default.yaml`、`src/utils/config.py`
**验证**:`--nrows 2000` 隔离冒烟测试,build→popular→evaluate→covis→evaluate 全链路通过(covis 加权分 0.34 ≫ popular 0.005)

---

## P1 · DSSM 多目标改造 ✅
> 目标:让 DSSM 进入多目标召回 + 融合 + 评估闭环。

- [x] 新增 `src/models/train_dssm_multi_target.py`:type-aware DSSM,支持多目标样本权重
- [x] 产物改名 `dssm_model_mt.pth` / `item2id_mt.pkl`(防覆盖单目标)
- [x] 新增 `src/recall/generate_dssm_recall_multi_target.py`:输出 `(session, type, predictions)`
- [x] 加冷启动防护(`item2id.get`,跳过未登录 aid)
- [x] 新增 `src/recall/fusion_recall_multi_target.py`:popular + covis(全局) + dssm 三路按 `(session,type)` 融合
- [x] `src/pipeline/run.py` 注册:`train-dssm-multi-target` / `dssm-recall-multi-target` / `fusion-recall-multi-target`
- [x] 评估并记录:DSSM 单路 & 融合后加权 Recall@20

**当前结果**:
- DSSM 单路(type-aware,weight-normalized loss,1M pairs / 5 epochs):加权 Recall@20 = 0.1792
- 三路融合(popular=0.1,covis=2.5,dssm=0.6):加权 Recall@20 = 0.3028

**产出**:`multi_target_dssm_predictions.csv`、`multi_target_fusion_predictions.csv`

---

## P2 · TIGER 生成式检索(核心亮点)
> 目标:加入"真·生成式"召回。OTTO 无商品文本 → 走协同 embedding → 语义ID → 生成路线。

- [ ] ① Item Embedding:复用 DSSM item embedding(后续可升级 word2vec)→ `item_emb.npy`
- [ ] ② `src/models/train_rqvae.py`:RQ-VAE(L=3 / K=256)→ `item2semid.pkl`
- [ ] ② 处理语义ID 冲突(同码追加去重位)
- [ ] ③ `src/models/train_tiger.py`:小 T5,历史展平成语义ID token,自回归生成
- [ ] ④ `src/recall/generate_tiger_recall.py`:beam search(20–50)→ 反查 aid → `(session,type,predictions)`
- [ ] 单独评估 TIGER 加权 Recall@20
- [ ] 接入 `fusion_recall_multi_target.py` 作为第 4 路,看融合提升
- [ ] `run.py` 注册:`build-item-emb` / `train-rqvae` / `build-semantic-id` / `train-tiger` / `tiger-recall`

**产出**:`item2semid.pkl`、`tiger_model.pth`、`multi_target_tiger_predictions.csv`

---

## P3 · LightGBM 精排 ✅(补全两阶段)
> 目标:把"召回 + 固定权重融合"升级为"召回 → 学习型精排"。

- [x] 候选合并:每个 `(session,type)` 汇总 popular / covis / DSSM 候选,label = 是否命中未来标签
- [x] 特征工程:各通道分数/排名、item 流行度、分类型行为计数、session 统计、target type
- [x] LightGBM `lambdarank`,按 `(session,type)` 分组训练
- [x] Top-N 重排,对比精排前后加权 Recall@20
- [x] 新增 lightgbm 依赖

**当前结果**:
- 候选池 oracle:加权 Recall@20 = 0.3286
- LightGBM ranker(full prediction):加权 Recall@20 = 0.3153
- group-level holdout:ranker = 0.3094,fusion = 0.2985

**产出**:`lgbm_ranker.txt`、`ranker_feature_importance.csv`

---

## P4 · 收尾:实验 + 文档 + 可视化
> 目标:把工作变成可展示的成果。

- [ ] 消融表:`popular → +covis → +dssm → +tiger → +精排` 每步加权分
- [ ] 更新 `README.md`(目前为空)+ 架构图
- [ ] 一页"项目亮点"总结(传统召回 → 生成式检索 → 精排的演进故事)
- [ ] (可选)生成测试集提交文件,跑 Kaggle 实际分数

**产出**:`README.md`、消融表、项目总结

---

## 实验记录(随做随填)

| 日期 | 配置 / 改动 | clicks | carts | orders | 加权分 | 备注 |
|---|---|---|---|---|---|---|
| 2026-06-15 | popular 基线 | 0.0091 | 0.0092 | 0.0099 | 0.0096 | 多目标 Top20 |
| 2026-06-15 | covis(全局) | 0.1204 | 0.1511 | 0.3471 | 0.2656 | 当前最强单路传统召回 |
| 2026-06-15 | dssm(多目标单路) | 0.0866 | 0.1052 | 0.2316 | 0.1792 | type-aware,weight-normalized loss,1M pairs / 5 epochs |
| 2026-06-15 | fusion(popular+covis+dssm) | 0.1443 | 0.1778 | 0.3917 | 0.3028 | 权重 0.1 / 2.5 / 0.6,popular 为正的当前网格最高分 |
|  | + tiger |  |  |  |  |  |
| 2026-06-16 | + LightGBM 精排(full prediction) | 0.1492 | 0.1846 | 0.4083 | 0.3153 | 候选池 oracle 0.3286;holdout ranker 0.3094 vs fusion 0.2985 |
