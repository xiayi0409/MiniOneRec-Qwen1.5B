# MiniOneRec：语义 ID（SID）码本量化实验报告

> 实验仓库：本仓库  
> 官方算法来源：[AkaliKong/MiniOneRec](https://github.com/AkaliKong/MiniOneRec)  
> 数据集：Amazon `Industrial_and_Scientific`  
> 更新时间：2026-08-13  
> 范围：商品 embedding → 三层 Semantic ID（SID）→ SID 质量统计 → SFT → constrained-beam 推理 → `calc.py` → SID embedding 类目一致性召回。RL 不属于本报告范围。

## 1. 背景与目标

生成式推荐不直接预测一个平铺的 item ID，而是为每个商品构造由多个离散 token 组成的语义 ID，例如 `<a_12><b_87><c_4>`。语言模型按 token 自回归生成 SID，再通过 SID→item 映射获得推荐商品。

SID 的质量会同时影响：

1. **可区分性**：不同商品若共用同一三层 SID，生成命中 SID 不等于命中唯一商品；
2. **可学习性**：码字如果高度偏斜或出现死码，LLM 只会频繁学习少数 token；
3. **语义保真**：强行均衡分配可能降低碰撞，但也可能把商品分配到并非最近的中心；
4. **解码有效性**：最终使用 constrained decoding，要求 SID token 路径在商品字典内且唯一。

本轮比较的核心问题是：在相同商品 embedding、相同 `3 × 256` 码本规模下，哪种量化方式可以在碰撞率、利用率、负载均衡与 SFT 推荐表现之间取得更好的平衡。

## 2. 数据集与统一协议

### 2.1 数据

| 项目 | 值 |
|---|---:|
| Amazon 类别 | `Industrial_and_Scientific` |
| 商品 embedding 数 | 3,686 |
| embedding 文件 | `data/Amazon/index/Industrial_and_Scientific.emb-qwen-td.npy` |
| embedding 维度 | 2,560 |
| SID 层数 | 3 |
| 每层码本大小 | 256 |
| 理论 SID 空间 | `256³ = 16,777,216` |
| 测试样本数 | 4,533 |
| SFT 最大训练 epoch | 10（验证集 early stopping） |

三层量化码先作为“原始 SID”进行统计。若原始 SID 冲突，则下游训练数据按官方 `constrained RQ-KMeans` / `RQ-KMeans++` exporter 的规则追加第四位 ordinal token：同一三层路径内第 `r` 个商品追加 `<d_r>`。这只用于区分冲突商品，**不改变前三层量化语义**，并使最终 SID 一一映射到 3,686 个商品。

### 2.2 码本训练共同配置

除算法明确另有设定外，RQVAE 路线保持 MiniOneRec 的配置：

- `lr=1e-3`，`epochs=10000`，`batch_size=2048`，`seed=2024`；
- 6 层 MLP encoder/decoder：`2560→2048→1024→512→256→128→64→32`；
- 三层 residual quantization，`K=256`，`e_dim=32`，`beta=0.25`；
- 每 50 epoch 在全商品上统计三层 raw SID 碰撞率，使用最佳碰撞 checkpoint。

RQ-KMeans++ 遵循官方脚本：`lr=1e-4`、`epochs=10000`、`batch_size=2048`、`e_dim=2560`，以 constrained 码本 warm start，并使用残差 encoder `z=x+MLP(x)`。

### 2.3 SFT 与评测共同配置

所有新补跑的方法保持 Reproduction 的 SFT 与评测参数：

```text
Base model: Qwen2.5-1.5B
batch_size=1024, micro_batch_size=16, seed=42
freeze_LLM=False, bf16, AdamW, 10 epochs
beam_size=50, batch_size=8, max_new_tokens=256, length_penalty=0
```

训练集由三类任务拼接：序列→SID、商品特征→SID、融合序列推荐；评测采用 constrained beam search。`calc.py` 计算 NDCG@K、HR@K 和无效路径计数 CC。

## 3. 指标定义

| 指标 | 定义 | 解读 |
|---|---|---|
| `unique_full_sid` | 原始三层完整路径的去重数 | 越大表示商品区分越强 |
| collision rate | `(商品数 - unique_full_sid) / 商品数` | 越低越好；不是去重后的最终 SID 碰撞率 |
| utilization | 某层被至少一个商品使用的码字数 / 256 | 检测死码；100% 不代表负载一定均匀 |
| normalized perplexity | `exp(H(p))/256`，`p` 为该层码字使用频率 | 有效容量占比；越接近 100% 越均匀 |
| load CV | `std(count) / mean(count)` | 码字负载离散程度；越低越均衡 |
| full SID space occupancy | `unique_full_sid / 256³` | 本数据最多仅 `3686/256³=0.021970%`，主要作为辅助指标 |
| NDCG@K | 命中位置按 `1/log(rank+1)` 折损 | 更强调前排推荐 |
| HR@K | 正例是否进入 Top-K | 召回能力 |
| CC | 生成结果中不在有效商品 SID 集合的次数 | 0 表示 constrained decoding 有效 |

注意：码本利用率和完整 SID 空间占用率完全不同。三层空间巨大，因此即使所有单层码字均被使用，完整空间占用率仍会小于 0.022%。

## 4. 方法说明

### M1. FAISS RQ-KMeans（官方）

使用 FAISS `ResidualQuantizer`。第 1 层对 embedding 聚类并量化，第 `l` 层对前 `l-1` 层重构后的残差继续聚类；每层硬最近中心分配、无负载约束。优点是快速、无神经网络训练；风险是后续残差层容易出现长尾负载。

实现：`rq/rqkmeans_faiss.py`，不加 `--uniform`。

### M2. Constrained / Balanced RQ-KMeans（官方）

每一层使用 `k-means-constrained`，直接对簇容量施加约束，使每个中心分到近似相同数量的商品。原始三层 SID 仍可能冲突，官方 exporter 为冲突商品追加第四位区分 token。

实现：`rq/rqkmeans_constrained.py`。

### M3. RQ-VAE（官方）

MLP encoder 将原 embedding 压缩到 latent；三层 VQ 逐层量化残差；decoder 重构 embedding。优化目标由重构损失、codebook loss、commitment loss 构成。可学习表征可能降低碰撞，但也会发生 codebook collapse：未被选中的码字几乎没有梯度。

实现：`rq/rqvae.py`、`rq/models/vq.py`。

### M4. RQ-KMeans++（官方 MiniOneRec 变体）

不是通用 K-Means++ 初始化算法。流程为：

```text
Constrained RQ-KMeans 训练均衡 codebook
→ codebook warm start
→ 将 encoder 改成 z = x + MLP(x)
→ 将最后层零初始化，使起点 z≈x
→ 端到端微调 RQ 和重构
```

实现：`rq/rqkmeans_plus.py` 与 `rq/generate_indices_plus.py`。

### M5. Sinkhorn RQ-VAE（本实验基于仓库已有 Sinkhorn 接口）

在 RQ-VAE **训练时**，每个量化 batch 将距离矩阵经 Sinkhorn-Knopp 归一化，再从近似均衡的分配矩阵取 code。此方式会影响 encoder、codebook、decoder 的共同优化。参数：`sk_epsilons=[0.005,0.005,0.005]`、`sk_iters=50`。

### M6. Sinkhorn RQ-VAE + dead-code reset（文献启发的基线）

在 M5 基础上，若某码字在当前 batch 分配次数 `≤1`，则每 batch 从对应 residual encoded features 中采样向量替换该 code embedding。它是“重置式”策略：简单直接，但可能引入训练扰动。

本实现参数：threshold=1、interval=1 batch。它借鉴 VQ 文献的 code reset 思路，不是 CVQ-VAE 的完整复现。

### M7. FAISS RQ-KMeans + Sinkhorn uniform mapping（官方）

这是 **后处理** 版本，与 M5 不同：先训练普通 FAISS RQ-KMeans，再在全量 3,686 商品上逐层计算 Sinkhorn optimal transport，并在指定容量约束下重新分配 code ID。它不再更新 FAISS codebook，而是用更近似均匀的全局编码替换原始 ID。

实现：`rq/rqkmeans_faiss.py --uniform --iters 30`。

### M8. RQ-VAE + CVQ-VAE online clustered codebook（本轮新增，遵循论文核心算法）

CVQ-VAE 不是硬死码重置。对每个 code 维护 EMA 使用率 `embed_prob`；根据低使用程度计算连续迁移权重；再从 encoded feature 分布采样 anchor（本实验用论文默认 `probrandom`），将低使用 code 平滑迁移至 anchor。活跃 code 仍通过原 VQ loss 优化。

```text
活码：原始 VQ loss 更新
低使用码：EMA usage → anchor sampling → 连续重定位
```

参数：`anchor=probrandom`、`decay=0.99`、`scale=10`、无 contrastive loss、Sinkhorn 关闭。论文：Zheng & Vedaldi, *Online Clustered Codebook*, ICCV 2023，https://arxiv.org/abs/2307.15139 。官方代码参考：`https://github.com/lyndonzheng/CVQ-VAE`。

### M9. CRID-inspired popularity-ranked SID（SID 结构扩展，不是新量化器）

参考 *Beyond Semantic IDs: Encoding Business-Value Ranking into Document Identifiers for Generative Retrieval*，保留 M4 RQ-KMeans++ 的前两级语义聚类结果，并把最后一级替换为簇内业务价值排序：

```text
<a_semantic><b_semantic><d_rank>
```

Amazon 数据没有 GMV、CTR、CVR，因此仅使用**训练集交互次数**作为业务价值代理；验证集与测试集不参与 rank 构建。共形成 2,594 个两级语义簇，最大簇大小为 22，3,686 个最终 SID 全部唯一。M9 用于检验“语义前缀 + 有序统计先验”的 SID 设计，不纳入 M1–M8 的原始三层量化碰撞率横向排名。

## 5. 原始三层 SID 质量结果

所有数据均在同一 3,686 商品 embedding 上统计；util/perplexity/CV 按 `L1/L2/L3` 给出。

| 方法 | unique SID | 碰撞率↓ | 空间占用率 | Utilization | Norm. perplexity | Load CV |
|---|---:|---:|---:|---|---|---|
| M1 FAISS RQ-KMeans | 2,958 | 19.750% | 0.017631% | 100 / 100 / 100% | 77.40 / 14.72 / 41.84% | 0.723 / 3.660 / 2.130 |
| M2 Constrained RQ-KMeans | 3,290 | 10.743% | 0.019610% | 100 / 100 / 100% | 99.81 / 99.82 / 99.82% | 0.060 / 0.060 / 0.059 |
| M3 RQ-VAE | 3,313 | 10.119% | 0.019747% | 31.25 / 100 / 100% | 25.40 / 93.82 / 85.23% | 1.945 / 0.371 / 0.666 |
| M4 RQ-KMeans++ | 3,277 | 11.096% | 0.019532% | 100 / 100 / 100% | 95.91 / 80.80 / 73.98% | 0.291 / 0.828 / 0.974 |
| M5 Sinkhorn RQ-VAE | 3,519 | 4.531% | 0.020975% | 100 / 99.61 / 99.61% | 99.53 / 93.22 / 85.85% | 0.097 / 0.379 / 0.567 |
| M6 Sinkhorn + dead reset | 3,442 | 6.620% | 0.020516% | 100 / 100 / 99.22% | 99.27 / 89.40 / 67.38% | 0.123 / 0.505 / 1.104 |
| M7 FAISS + Sinkhorn mapping | **3,576** | **2.984%** | **0.021315%** | 100 / 100 / 100% | 98.25 / 99.71 / 99.21% | 0.248 / 0.083 / 0.148 |
| M8 CVQ-RQ-VAE | 3,334 | 9.550% | 0.019872% | 97.66 / 100 / 100% | 42.79 / 89.58 / 78.06% | 1.733 / 0.571 / 1.077 |

### 5.1 码本级结论

1. **普通 FAISS**：单层利用率虽为 100%，但 L2 perplexity 只有 14.72%、CV 高达 3.660，说明“没有死码”并不等于均匀；它也是碰撞最高的方法。
2. **Constrained RQ-KMeans**：三层均衡性最稳定（CV 约 0.06），验证了容量约束的作用；但三层路径仍有 10.74% 碰撞。
3. **原始 RQ-VAE**：L1 利用率 31.25%、perplexity 25.40%，存在典型第 1 层码本塌陷。
4. **RQ-KMeans++**：官方残差 warm-start 保持了 100% 单层覆盖，但后两层仍有明显不均衡，原始碰撞未优于 RQ-VAE。
5. **Sinkhorn RQ-VAE**：把训练内的均衡约束引入 RQ-VAE 后，碰撞降至 4.53%，并消除了明显死码。
6. **Dead-code reset**：相比纯 Sinkhorn，碰撞率变差（6.62% vs 4.53%），尤其 L3 负载 CV 升至 1.104。当前小数据集、每 batch 重置一次的激进策略带来了扰动。
7. **FAISS+Sinkhorn 后处理**：当前 SID 级别最佳。它用全量全局容量约束得到 2.98% 碰撞和近乎满 perplexity，但“均衡”并不自动保证下游推荐最佳，仍需 SFT 验证。
8. **CVQ-RQ-VAE**：成功把原 RQ-VAE L1 利用率由 31.25% 提升至 97.66%，验证 online clustered codebook 的去死码作用；但当前其碰撞率为 9.55%，没有优于 Sinkhorn RQ-VAE。这说明提升覆盖率与优化完整 SID 区分性是不同目标。

## 6. 下游 SFT + 评测结果

已完成结果均来自 4,533 条测试样本、50 beam、CC=0。NDCG/HR 为 item-level target 匹配结果。

| 方法 | SID 来源/状态 | NDCG@1 | NDCG@10 | NDCG@50 | HR@1 | HR@10 | HR@50 | CC |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| M1 FAISS RQ-KMeans | 本轮补跑完成 | 0.050518 | 0.078963 | 0.102240 | 0.050518 | 0.117362 | 0.223914 | 0 |
| M2 Constrained RQ-KMeans | 本轮补跑完成 | 0.057798 | 0.092321 | 0.114613 | 0.057798 | 0.138760 | 0.241121 | 0 |
| M3 官方 RQ-VAE | 既有 Reproduction 完整结果 | 0.066402 | 0.098743 | 0.118175 | 0.066402 | 0.140746 | 0.230752 | 0 |
| M4 RQ-KMeans++ | 既有结果；本次 SID 文件 SHA256 完全一致 | 0.065078 | 0.099909 | **0.122416** | 0.065078 | 0.147143 | **0.249062** | 0 |
| M5 Sinkhorn RQ-VAE | 本轮补跑完成 | 0.065299 | 0.097877 | 0.119944 | 0.065299 | 0.141628 | 0.242444 | 0 |
| M6 Sinkhorn + dead reset | 本轮补跑完成 | **0.068829** | 0.096665 | 0.115828 | **0.068829** | 0.132583 | 0.221046 | 0 |
| M7 FAISS + Sinkhorn mapping | 本轮补跑完成 | 0.060225 | 0.083779 | 0.104153 | 0.060225 | 0.114273 | 0.207809 | 0 |
| M8 CVQ-RQ-VAE | 本轮补跑完成 | 0.065078 | 0.092548 | 0.112904 | 0.065078 | 0.128612 | 0.223252 | 0 |
| M9 CRID-popularity | RQ-KMeans++ 两级前缀 + 训练交互频次 rank | 0.063755 | **0.101377** | 0.121678 | 0.063755 | **0.149790** | 0.241121 | 0 |

### 6.1 下游结果的阶段性解读

- M1→M2：约束均衡显著改善 NDCG@10（0.07896→0.09232）与 HR@10（0.11736→0.13876），说明普通 FAISS 的严重负载偏斜确实伤害了 LLM 学习。
- M5：SID 碰撞已显著下降至 4.53%，并获得了接近已有 RQ-VAE/M4 的 HR@10；但不必然超过 M4，说明量化几何、ID 序列分布和生成难度均影响最终指标。
- M6：前排 NDCG@1/HR@1 最高，但 Top-10/Top-50 较弱；与其后层负载不均衡相吻合。不能据此宣称 dead reset 优于 Sinkhorn。
- M4：NDCG@50、HR@50 最好，且其 SID 和此前已评测版本完全一致。
- M7：虽然其 raw SID 碰撞率最低（2.98%），但 SFT 的 NDCG@10=0.08378、HR@10=0.11427，低于 M2/M3/M4/M5；全局均衡的后处理可能牺牲了 token 与原始语义几何的一致性。
- M8：CVQ 成功显著提高了原 RQ-VAE 的 L1 覆盖率（31.25%→97.66%），但 SFT NDCG@10=0.09255、HR@10=0.12861，仍弱于原 RQ-VAE/M4/M5；CVQ 解决死码，但未自动带来最优的 item-level 推荐表现。
- M9：NDCG@10=0.10138、HR@10=0.14979，为当前九个方案最高；相对 M4 分别提高 1.47% 和 1.80%。但 HR@1 相对下降 2.03%、HR@50 相对下降 3.19%，表现为前中排统计先验增强而深层召回受损，不能表述为全面优于 M4。
- 因此，**raw SID 碰撞率并不能单独预测下游指标**；应同时报告碰撞、负载均衡、SID token 语义结构与 SFT NDCG/HR。

## 7. SID embedding 召回的类目一致性

### 7.1 标签恢复与评测协议

仓库处理后的 3,686 个商品都属于顶层 `Industrial_and_Scientific`，但细粒度 `category/categories` 字段为空。为避免顶层类目直接产生无意义的 100% 一致性，本实验下载 UCSD Amazon Review Data (2018) 官方 metadata，并以规范化标题精确匹配恢复类目：

| 项目 | 数值 |
|---|---:|
| 本地商品数 | 3,686 |
| 标题精确匹配 | 3,686（100%） |
| 有效细类目商品 | 3,561（96.61%） |
| Parent 类目数 | 25 |
| Leaf 类目数 | 707 |

对每个方法分别读取其 SFT `final_checkpoint` 中学习后的 SID token embedding。商品向量为其 SID token embedding 的均值，再进行余弦 Top-K 最近邻召回并排除 query 商品自身。`Category Precision@K` 表示召回邻居与 query 属于同一类目的比例。

该指标衡量的是 **SID 结构经过 SFT 后的局部语义一致性**，同时受到量化路径、去重 token 和 SFT token embedding 学习影响，不能单独当作原始码本的纯语义指标。

### 7.2 完整 SID embedding 召回

随机召回基线为 Parent 7.87%、Leaf 0.83%。

| 方法 | Parent P@1 | P@5 | P@10 | P@20 | Leaf P@1 | P@5 | P@10 | P@20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| M1 FAISS RQ-KMeans | 74.64% | 61.86% | 55.50% | 48.53% | 46.98% | 32.72% | 26.55% | 20.83% |
| M2 Constrained RQ-KMeans | 68.21% | 57.14% | 53.39% | 48.76% | 41.25% | 27.36% | 23.33% | 20.26% |
| M3 RQ-VAE | 60.77% | 46.05% | 37.89% | 34.81% | 31.65% | 21.35% | 15.70% | 12.50% |
| M4 RQ-KMeans++ | 74.64% | 62.74% | 56.84% | 49.18% | 46.53% | 32.22% | 26.88% | 21.94% |
| M5 Sinkhorn RQ-VAE | 56.50% | 49.45% | 46.49% | 41.65% | 29.43% | 21.42% | 19.58% | 16.98% |
| M6 Sinkhorn + reset | 55.35% | 45.22% | 41.63% | 37.99% | 28.53% | 19.61% | 17.56% | 15.81% |
| M7 FAISS + Sinkhorn | 59.45% | 51.79% | 48.17% | 43.26% | 29.57% | 21.44% | 18.61% | 16.00% |
| M8 CVQ-RQ-VAE | 66.58% | 56.90% | 53.40% | 50.42% | 37.26% | 24.76% | 21.20% | 18.71% |
| M9 CRID-popularity | **82.87%** | **75.60%** | **70.02%** | **58.13%** | **49.93%** | **38.54%** | **33.52%** | **26.12%** |

### 7.3 仅前两级语义前缀 embedding

为排除第三层量化码、冲突去重 token 或 CRID rank token 的影响，额外只平均 `<a_i><b_j>` 两级 token：

| 方法 | Parent P@1 | Parent P@10 | Leaf P@1 | Leaf P@10 |
|---|---:|---:|---:|---:|
| M1 FAISS RQ-KMeans | 81.72% | 72.96% | 47.23% | 34.72% |
| M2 Constrained RQ-KMeans | 80.57% | 72.58% | 47.66% | 33.86% |
| M3 RQ-VAE | 64.87% | 37.76% | 36.76% | 16.06% |
| M4 RQ-KMeans++ | **82.90%** | 71.93% | **50.04%** | 35.08% |
| M5 Sinkhorn RQ-VAE | 73.04% | 67.44% | 36.25% | 28.93% |
| M6 Sinkhorn + reset | 74.59% | 67.52% | 38.70% | 30.08% |
| M7 FAISS + Sinkhorn | 64.76% | 57.19% | 30.27% | 22.83% |
| M8 CVQ-RQ-VAE | 76.21% | 65.77% | 43.08% | 26.03% |
| M9 CRID-popularity | **82.90%** | **73.21%** | 49.26% | **35.49%** |

### 7.4 类目一致性与下游推荐的联合结论

1. M9 完整 SID 在 Parent/Leaf 的所有 K 上均最高，同时取得最好的 NDCG@10/HR@10，但没有取得最佳 HR@50；类目局部一致性与深层 item recall 不是同一个目标。
2. M1/M4 的完整 SID 类目一致性在传统量化方法中最好；M4 同时取得最佳 NDCG@50/HR@50，是当前综合最稳健的量化路线。
3. M7 的 raw SID 碰撞率最低、负载最均衡，但完整 SID Parent P@10=48.17%、Leaf P@10=18.61%，下游 NDCG@10/HR@10 也较弱。这支持“全局均衡后处理破坏部分语义几何”的解释。
4. M3 第一层码本塌陷对应较弱的两级前缀类目一致性；M5/M6 虽通过 Sinkhorn 改善利用率和碰撞，却没有在 embedding 类目一致性上超过 M1/M4，说明均衡约束不等于类目语义更纯。
5. M4 与 M9 的两级前缀指标几乎一致，因为 M9 直接复用了 M4 的两级语义分簇；完整 SID 的差异主要来自 M4 第三层 residual code 与 M9 popularity-rank token 的不同。

## 8. 实现与可复现文件

### 8.1 算法代码

| 内容 | 路径 |
|---|---|
| 官方 FAISS / Sinkhorn mapping | `rq/rqkmeans_faiss.py` |
| 官方 constrained RQ-KMeans | `rq/rqkmeans_constrained.py` |
| 官方 RQ-KMeans++ | `rq/rqkmeans_plus.py` |
| 官方 plus SID exporter | `rq/generate_indices_plus.py` |
| RQ-VAE / Sinkhorn / CVQ quantizer | `rq/models/vq.py` |
| RQ 量化器封装 | `rq/models/rq.py` |
| SID 变体数据准备 | `tools/prepare_sid_sft_variants.py` |
| 可恢复 SFT→评测流水线 | `tools/run_sid_variant_sft_eval.sh` |
| CRID SID 构建 | `tools/build_crid_sid.py` |
| SID embedding 类目召回 | `tools/eval_sid_category_recall.py` |

### 8.2 指标产物

| 内容 | 路径 |
|---|---|
| RQ-VAE、FAISS、constrained、Sinkhorn 统计 | `SID_RERUN_UNIFIED_METRICS.json` |
| FAISS+Sinkhorn 统计 | `data/rerun_faiss_sinkhorn/Industrial_and_Scientific/unified_metrics.json` |
| 官方 RQ-KMeans++ 重训统计 | `data/rerun_plus_official/unified_metrics.json` |
| CVQ-RQ-VAE 统计 | `data/cvq_rqvae_unified_metrics.json` |
| 五种变体 SID 输入清单 | `results/sid_sft_variant_manifest.json` |
| 每种 SFT 的完整日志/结果 | `output/logs/sid_<method>/`、`results/sid_<method>/` |
| 各方法类目召回明细 | `results/<method>/sid_category_recall.json` |
| CRID 推荐与类目召回结果 | `results/crid_popularity/` |

## 9. 当前状态与待完成事项

- M1、M2、M5、M6 的 SFT + 评测已完成。
- M3、M4 已有可复用的 SFT + `calc.py` 结果；M4 的 SID 与本次重训文件 SHA256 完全一致。
- M1–M9 的 SFT、constrained-beam 推理和 `calc.py` 均已完成；全部 CC=0。
- M1–M9 的 SID embedding Parent/Leaf 类目一致性均已完成，统一标签覆盖率 96.61%。
- 当前 9 条 SFT 可在同一训练/评测协议下横向比较。M9 不是新量化器，而是基于 M4 语义前缀的 SID 结构扩展，应与 M1–M8 分开陈述。

## 10. 结果解释的边界

1. `FAISS+Sinkhorn mapping`、`constrained RQ-KMeans` 和 `RQ-KMeans++` 为官方 MiniOneRec 实现/路线；Sinkhorn-RQ-VAE、dead reset、CVQ-RQ-VAE 是本地受控扩展。
2. CVQ 实现迁移了论文作者公开代码的核心在线更新公式，但将二维图像 VQ 改造成 MiniOneRec 的每层 residual vector quantization；因此应表述为“CVQ-style online clustered codebook adapted to RQ-VAE”，而不是声称完整复现其图像实验。
3. 所有 raw SID 指标均在去重前统计；下游训练以官方风格第四位 ordinal token 解决冲突，故训练时最终 item SID 是唯一的。
4. SID 级指标与推荐质量相关但并非单调关系；报告应同时给出碰撞/均衡指标和 NDCG/HR/CC。
5. 类目一致性使用 SFT 后的 token embedding；不同方法的 early stopping 点可能不同，因此它是端到端 SID 表示诊断，不是严格隔离量化器影响的静态指标。
6. M9 使用训练交互频次代替真实业务价值，只能表述为 CRID-inspired popularity proxy，不能声称复现淘宝 GMV/CVR 实验。
