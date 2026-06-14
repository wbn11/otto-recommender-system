# OTTO 项目进度跟踪 Roadmap

> 配套设计文档:[architecture.md](architecture.md)
> 指标:加权 Recall@20(clicks 0.10 / carts 0.30 / orders 0.60)
> 硬件:RTX 4060 Laptop(8GB)

## 总体进度

- [ ] **P0** 工程地基(提速 + 配置)
- [ ] **P1** DSSM 多目标改造 + 多源融合
- [ ] **P2** TIGER 生成式检索(核心亮点)
- [ ] **P3** LightGBM 精排(补全两阶段,可选)
- [ ] **P4** 收尾:消融实验 + 文档 + 可视化

| 阶段 | 预计 | 依赖 | 状态 |
|---|---|---|---|
| P0 | 1–2天 | 无 | ⬜ 未开始 |
| P1 | 3–5天 | P0(可选) | ⬜ 未开始 |
| P2 | 1.5–2周 | P1 | ⬜ 未开始 |
| P3 | ~1周 | P1 | ⬜ 未开始 |
| P4 | 3–5天 | P1–P3 | ⬜ 未开始 |

---

## 已完成(基线)

- [x] 单目标 + 多目标验证集构建
- [x] Popular 召回(单 / 多目标)
- [x] Covisitation 召回(全局;分类型/融合已移除)
- [x] DSSM 双塔召回(**仅单目标**)
- [x] 加权 Recall@20 评估

---

## P0 · 工程地基
> 目标:消除 I/O 瓶颈、统一超参管理。不阻塞主线,赶时间可跳过。

- [ ] ① `expand_events` 的 `iterrows` → polars 向量化 + parquet(`src/data/build_multi_target_validation.py`)
- [ ] ① 新增 polars 依赖到 `requirements.txt`
- [ ] ① 下游脚本改读 parquet
- [ ] ② 新建 `configs/default.yaml`,抽出硬编码超参(emb_dim、top_k、融合权重、history_ratio 等)
- [ ] ③ 统一产物命名规范(单目标 vs 多目标不互相覆盖)

**产出**:`*.parquet`、`configs/default.yaml`

---

## P1 · DSSM 多目标改造
> 目标:让 DSSM 进入多目标召回 + 融合 + 评估闭环。

- [ ] 改 `src/models/train_dssm.py`:参数化输入文件(默认 `multi_target_train_events.csv`)
- [ ] 产物改名 `dssm_model_mt.pth` / `item2id_mt.pkl`(防覆盖单目标)
- [ ] 新增 `src/recall/generate_dssm_recall_multi_target.py`:输出 `(session, type, predictions)`
- [ ] 加冷启动防护(`item2id.get`,跳过未登录 aid)
- [ ] 新增 `src/recall/fusion_recall_multi_target.py`:popular + covis(全局) + dssm 三路按 `(session,type)` 融合
- [ ] `src/pipeline/run.py` 注册:`train-dssm-multi-target` / `dssm-recall-multi-target` / `fusion-recall-multi-target`
- [ ] 评估并记录:DSSM 单路 & 融合后加权 Recall@20

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

## P3 · LightGBM 精排(可选,补全两阶段)
> 目标:把"召回 + 固定权重融合"升级为"召回 → 学习型精排"。

- [ ] 候选合并:每个 `(session,type)` 汇总各路候选,label = 是否命中未来标签
- [ ] 特征工程:各通道分数/排名、item 流行度/分类型 CTR-CVR、recency、session 统计、TIGER 生成分数
- [ ] LightGBM `lambdarank`,按 `(session,type)` 分组训练
- [ ] Top-N 重排,对比精排前后加权 Recall@20
- [ ] 新增 lightgbm 依赖

**产出**:`ranker.txt`、特征重要性表

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
|  | popular 基线 |  |  |  |  |  |
|  | + covis(全局) |  |  |  |  |  |
|  | + dssm(多目标) |  |  |  |  |  |
|  | + tiger |  |  |  |  |  |
|  | + 精排 |  |  |  |  |  |
