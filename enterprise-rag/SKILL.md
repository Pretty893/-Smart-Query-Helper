# Enterprise RAG Skills 模块

## 概述

本模块实现了企业级 RAG 系统的 Skills 化架构，将检索、问答、后处理等功能解耦为独立的可插拔模块。

## 核心设计理念

1. **模块化**：每个功能独立封装，便于维护和扩展
2. **动态选择**：根据问题类型自动选择最合适的处理策略
3. **可替换性**：同一功能可有多实现，按需切换
4. **易于集成**：提供统一的接口，方便与现有系统对接

## 目录结构

```
enterprise-rag/
├── SKILL.md                    # 本文件
├── scripts/                    # 核心脚本
│   ├── query_router.py         # 查询意图分类 + 检索策略选择
│   ├── retriever_bm25.py       # BM25 关键词检索
│   ├── retriever_vector.py     # 向量语义检索
│   ├── retriever_hybrid.py     # 混合检索（RRF 融合）
│   ├── prompt_selector.py      # 根据问题类型选择 Prompt
│   ├── tool_caller.py          # 外部系统 API 调用
│   └── post_processor.py       # 摘要/格式转换/翻译
├── references/                 # 参考文档
│   ├── RETRIEVAL_GUIDE.md      # 检索策略详细说明
│   ├── PROMPT_TEMPLATES.md     # 所有 Prompt 模板
│   ├── PROMPT_INTEGRATION.md   # Prompt 对接文档
│   └── API_INTEGRATION.md      # 外部系统对接文档
└── assets/
    └── report_templates/       # 输出格式模板
```

## 使用方式

### 基本流程

```python
from enterprise_rag.scripts.query_router import QueryRouter
from enterprise_rag.scripts.retriever_hybrid import HybridRetriever
from enterprise_rag.scripts.prompt_selector import PromptSelector
from enterprise_rag.scripts.post_processor import PostProcessor

router = QueryRouter()
retriever = HybridRetriever()
prompt_selector = PromptSelector()
post_processor = PostProcessor()

# 1. 路由分析
route_result = router.route("请假流程是什么？")

# 2. 选择检索策略
if route_result["retriever_strategy"] == "hybrid":
    search_results = retriever.search_with_rrf(question, category)

# 3. 选择 Prompt
prompt = prompt_selector.select(route_result["intent_type"])

# 4. 生成回答
answer = chain.invoke({...})

# 5. 后处理
summary = post_processor.generate_summary(answer)
```

## 扩展指南

### 添加新的检索策略

1. 在 `scripts/` 目录下创建新文件，如 `retriever_custom.py`
2. 实现 `search()` 方法，返回格式与现有检索器一致
3. 在 `query_router.py` 中添加新策略的路由逻辑

### 添加新的 Prompt 模板

1. 在 `prompt_selector.py` 的 `PROMPT_TEMPLATES` 字典中添加新模板
2. 在 `get_template_names()` 和 `list_templates()` 中注册新模板

### 添加新的外部工具

1. 在 `tool_caller.py` 的 `available_tools` 字典中注册新工具
2. 实现对应的 `_execute_tool` 方法

## 性能优化建议

1. **缓存检索结果**：对于相同的查询，缓存检索结果
2. **异步执行**：多个检索策略可并发执行
3. **结果预加载**：提前加载常用文档的向量和 BM25 索引
