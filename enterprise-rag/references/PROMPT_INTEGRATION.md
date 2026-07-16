# Prompt 对接文档

## 概述

本文档描述了如何将 Prompt 模板集成到现有的问答流程中。

## 集成架构

```
用户问题 → QueryRouter → PromptSelector → LLM → 回答
              │             │
              ▼             ▼
         意图分析      选择模板
```

## 集成步骤

### 步骤一：路由分析

使用 QueryRouter 分析问题类型：

```python
from enterprise_rag.scripts.query_router import QueryRouter

router = QueryRouter()
route_result = router.route("请假流程是什么？")
# route_result["intent_type"] = "process_guide"
```

### 步骤二：选择 Prompt

使用 PromptSelector 根据意图类型选择模板：

```python
from enterprise_rag.scripts.prompt_selector import PromptSelector

selector = PromptSelector()
prompt = selector.select(route_result["intent_type"])
# prompt = 流程指引模板
```

### 步骤三：构建 Chain

将 Prompt 与 LLM 和输出解析器组合：

```python
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models.tongyi import ChatTongyi

chat_model = ChatTongyi(model="qwen3-max")
chain = prompt | chat_model | StrOutputParser()
```

### 步骤四：生成回答

调用 Chain 生成回答：

```python
answer = chain.invoke({
    "history": [],
    "context": context,
    "question": question,
})
```

## 完整示例

```python
from enterprise_rag.scripts.query_router import QueryRouter
from enterprise_rag.scripts.prompt_selector import PromptSelector
from enterprise_rag.scripts.retriever_hybrid import HybridRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models.tongyi import ChatTongyi

router = QueryRouter()
selector = PromptSelector()
retriever = HybridRetriever()
chat_model = ChatTongyi(model="qwen3-max")

question = "报销流程是什么？"

# 1. 路由分析
route_result = router.route(question)

# 2. 检索
search_results = retriever.search_with_rrf(
    question,
    category=route_result.get("category", "全部"),
    limit=5
)

# 3. 构建上下文
context = "\n".join([doc for doc, meta, score in search_results])

# 4. 选择 Prompt
prompt = selector.select(route_result["intent_type"])

# 5. 构建 Chain
chain = prompt | chat_model | StrOutputParser()

# 6. 生成回答
answer = chain.invoke({
    "history": [],
    "context": context,
    "question": question,
})

print(answer)
```

## 与现有系统集成

### 替换现有 Prompt

在 `chat_service.py` 中，将原来的固定 Prompt 替换为动态选择：

```python
from enterprise_rag.scripts.prompt_selector import PromptSelector

class OfficeMateChatService:
    def __init__(self):
        self.prompt_selector = PromptSelector()
    
    def _handle_simple_question(self, question, ...):
        # 选择 Prompt
        prompt = self.prompt_selector.select(question_type)
        
        # 构建 Chain
        chain = prompt | self._get_chat_model() | StrOutputParser()
        
        # 生成回答
        answer_body = chain.invoke({...})
```

### 保持兼容性

为了保持与现有系统的兼容性，确保新模板的输出格式与原有格式一致：

```markdown
### 最终回答
[你的回答]

### 操作步骤/材料清单
[相关内容]

### 风险提示
[如有适用]
```

## 性能优化

1. **缓存 Prompt**：对于相同的意图类型，缓存编译后的 Prompt
2. **批量处理**：一次性处理多个问题，减少 LLM 调用次数
3. **异步执行**：检索和 Prompt 选择可并行执行

## 监控与日志

记录每次使用的 Prompt 类型和效果，以便优化：

```python
import logging

logging.info(f"使用 Prompt: {intent_type}")
logging.info(f"回答长度: {len(answer)}")
logging.info(f"用户满意度: {feedback_score}")
```
