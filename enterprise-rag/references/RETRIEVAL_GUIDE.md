# 检索策略详细说明

## 概述

本系统支持三种检索策略：向量检索、BM25 关键词检索、混合检索（RRF 融合）。

## 检索策略对比

| 维度 | 向量检索 | BM25 检索 | 混合检索 |
|------|----------|-----------|----------|
| 原理 | 语义向量相似度 | 词频+IDF统计 | 向量+BM25 |
| 优势 | 理解语义相似性 | 精确匹配关键词 | 兼顾语义和关键词 |
| 缺点 | 对关键词不敏感 | 无法理解同义词 | 计算成本较高 |
| 适用场景 | 理解性问题 | 事实性问题 | 复杂问题 |

## 向量检索

### 原理

使用 `text-embedding-v4` 模型将文本转换为向量，通过余弦相似度计算相关性。

### 适用场景

- "怎么申请报销？"
- "请解释一下年假制度"
- "什么情况下可以请事假？"

### 调用方式

```python
from enterprise_rag.scripts.retriever_vector import VectorRetriever

retriever = VectorRetriever()
results = retriever.search(query, category="HR制度", limit=5)
```

## BM25 检索

### 原理

使用 BM25Okapi 算法，基于词频和逆文档频率计算相关性得分。

### 分词处理

使用 `jieba` 进行中文分词，支持自定义词典。

### 适用场景

- "报销需要什么材料？"
- "请假审批流程是什么？"
- "差旅补贴标准是多少？"

### 调用方式

```python
from enterprise_rag.scripts.retriever_bm25 import BM25Retriever

retriever = BM25Retriever()
results = retriever.search(query, category="财务制度", limit=5)
```

## 混合检索

### 原理

同时执行向量检索和 BM25 检索，使用 RRF（Reciprocal Rank Fusion）融合结果。

### RRF 计算公式

```
RRF = Σ (1 / (k + rank))
```

其中 `k` 为常数（默认 60），`rank` 为文档在单个检索结果中的排名。

### 适用场景

- "从入职到转正的完整流程是什么？"
- "请详细说明差旅费报销的要求和流程"
- "年假和事假的区别是什么？"

### 调用方式

```python
from enterprise_rag.scripts.retriever_hybrid import HybridRetriever

retriever = HybridRetriever()
results = retriever.search_with_rrf(query, category="全部", limit=5)
```

## 检索策略选择逻辑

QueryRouter 根据问题类型自动选择检索策略：

| 问题类型 | 默认策略 | 说明 |
|----------|----------|------|
| policy_qa | hybrid | 制度问答需要兼顾精确匹配和语义理解 |
| process_guide | hybrid | 流程指引需要步骤清晰，关键词重要 |
| material_list | bm25 | 材料清单需要精确匹配关键词 |
| notice_summary | vector | 通知总结需要语义理解 |
| complex | hybrid | 复杂问题需要多种策略结合 |

## 自定义词典

BM25 检索支持自定义词典，在项目根目录创建 `custom_dict.txt`：

```txt
弹性工作制 5 n
远程办公 5 n
数字孪生 5 n
```

## 性能优化

1. **批量检索**：一次性检索多个查询，减少 API 调用次数
2. **结果缓存**：缓存高频查询的检索结果
3. **并行检索**：向量检索和 BM25 检索可并行执行
