# CRID-inspired Popularity-Ranked SID 实验

## 1. 实验目的

参考 *Beyond Semantic IDs: Encoding Business-Value Ranking into Document Identifiers for Generative Retrieval* 的 CRID 思路，将 SID 解耦为：

```text
[两级语义聚类前缀] + [簇内业务价值排名]
```

本实验没有 Amazon GMV、CTR 或 CVR，因此只使用训练集交互次数作为业务价值代理。验证集和测试集交互没有用于构造排名，以避免标签泄漏。

## 2. 实验配置

- 数据集：Amazon `Industrial_and_Scientific`
- 商品数：3,686
- Backbone：Qwen2.5-1.5B
- 语义前缀：RQ-KMeans++ 的前两级 `<a_i><b_j>`
- 排名后缀：同一语义簇内按训练交互次数降序生成 `<d_0><d_1>...`
- 语义簇数：2,594
- 最大簇大小：22
- 最终 SID 碰撞率：0
- SFT：MiniOneRec-Reproduction 原配置，batch size 1024、micro batch size 16、学习率 `3e-4`、seed 42，并采用验证集 early stopping
- 推理：约束 Beam Search，beam size 50

## 3. 推荐效果

| 指标 | RQ-KMeans++ | CRID-inspired | 绝对变化 | 相对变化 |
|---|---:|---:|---:|---:|
| NDCG@1 | 0.065078 | 0.063755 | -0.001323 | -2.03% |
| NDCG@10 | 0.099909 | 0.101377 | +0.001468 | +1.47% |
| NDCG@50 | 0.122416 | 0.121678 | -0.000738 | -0.60% |
| HR@1 | 0.065078 | 0.063755 | -0.001323 | -2.03% |
| HR@10 | 0.147143 | 0.149790 | +0.002647 | +1.80% |
| HR@50 | 0.249062 | 0.241121 | -0.007941 | -3.19% |

结论：当前 popularity proxy 在 Top-10 附近带来小幅收益，但 Top-1 和深层 HR@50 下降，不能表述为全面优于 RQ-KMeans++。这符合“业务排序强化中前排先验，但可能牺牲部分长尾/深层召回”的解释。

## 4. SID embedding 类目一致性

### 4.1 方法

处理后的仓库数据缺失细粒度 `category` 字段，因此从 UCSD Amazon Review Data (2018) 官方 `Industrial_and_Scientific` metadata 恢复标签：

- 规范化标题精确匹配率：100%（3,686 / 3,686）
- 具有非空类目的商品：3,561
- 有效标签覆盖率：96.61%
- 父类目数：25
- 叶子类目数：707

从各自 SFT checkpoint 中读取 SID token embedding，将一个商品全部 SID token 的 embedding 取均值，进行余弦 Top-K 最近邻召回。指标是召回商品与 query 商品的同类目比例，即 Category Precision@K。检索时排除商品自身。

### 4.2 完整 SID embedding

| 粒度/指标 | 随机基线 | RQ-KMeans++ | CRID-inspired | CRID 变化 |
|---|---:|---:|---:|---:|
| 父类 P@1 | 7.87% | 74.64% | 82.87% | +8.23pp |
| 父类 P@5 | 7.87% | 62.74% | 75.60% | +12.86pp |
| 父类 P@10 | 7.87% | 56.84% | 70.02% | +13.18pp |
| 父类 P@20 | 7.87% | 49.18% | 58.13% | +8.95pp |
| 叶子类 P@1 | 0.83% | 46.53% | 49.93% | +3.40pp |
| 叶子类 P@5 | 0.83% | 32.22% | 38.54% | +6.32pp |
| 叶子类 P@10 | 0.83% | 26.88% | 33.52% | +6.64pp |
| 叶子类 P@20 | 0.83% | 21.94% | 26.12% | +4.18pp |

### 4.3 仅前两级语义前缀 embedding

| 粒度/指标 | RQ-KMeans++ | CRID-inspired | CRID 变化 |
|---|---:|---:|---:|
| 父类 P@1 | 82.90% | 82.90% | 0.00pp |
| 父类 P@10 | 71.93% | 73.21% | +1.28pp |
| 叶子类 P@1 | 50.04% | 49.26% | -0.79pp |
| 叶子类 P@10 | 35.08% | 35.49% | +0.40pp |

只看前两级时两者基本一致，而完整 SID 差距明显，说明完整表示的变化主要来自最后一级 token。需要注意，这个指标同时受到 SID 结构和 SFT 后 token embedding 学习的影响，不能单独归因于量化器。

## 5. 当前结论

1. CRID-inspired SID 在本数据上实现最终零碰撞，并提高完整 SID embedding 的类目局部一致性。
2. 推荐指标不是全面上涨：NDCG@10 和 HR@10 小幅提高，HR@50 下降。
3. 交互次数只是业务价值 proxy，和论文使用的转化、点击或业务模型分数不同。
4. 后续最有价值的消融是：随机簇内 rank、流行度 rank、平滑/分桶 popularity rank。在相同语义前缀下比较，才能确认收益来自有序业务先验而不是减少第三层语义粒度。

## 6. 产物

- 构建脚本：`tools/build_crid_sid.py`
- 类目召回评估：`tools/eval_sid_category_recall.py`
- CRID 推荐结果：`results/crid_popularity/final_result_Industrial_and_Scientific.json`
- CRID 类目召回指标：`results/crid_popularity/sid_category_recall.json`
- RQ-KMeans++ 类目召回指标：`results/Industrial_and_Scientific_rqkmeans_plus_gpu5/sid_category_recall.json`
