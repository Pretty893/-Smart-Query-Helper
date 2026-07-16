from typing import List, Dict, Any
from pydantic import BaseModel, Field

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_community.embeddings import DashScopeEmbeddings

import config_data as config


class ExpandedQueries(BaseModel):
    expanded_queries: List[str] = Field(
        description="从不同角度生成的同义问题列表，每个问题都应该与原问题语义相关且包含关键实体"
    )


class QueryExpander:
    def __init__(self):
        self.chat_model = ChatTongyi(
            model=config.chat_model_name,
            temperature=0.1,#控制随机性
        )
        self.embedding_model = DashScopeEmbeddings(model=config.embedding_model_name)
        self.parser = PydanticOutputParser(pydantic_object=ExpandedQueries)

        self.expand_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """你是一个企业知识问答领域的查询扩展专家。请根据用户问题和上下文信息，生成高质量的同义扩展问题。

要求：
1. 扩展问题必须与原问题语义高度相关，不能偏离主题
2. 必须保留原问题的关键实体（如部门、角色、政策领域等）
3. 从不同角度重新表述，如换一种问法、添加具体场景、强调不同侧重点
4. 语言自然，符合企业员工日常提问习惯
5. 不要生成与原问题完全相同的问题

示例1：
原问题：病假可以请几天？
识别的实体：[{"type": "policy_area", "value": "病假"}, {"type": "time", "value": "天数"}]
扩展问题：
- 员工请病假的最长天数是多少？
- 病假的请假期限有什么规定？

示例2：
原问题：差旅费怎么报销？
识别的实体：[{"type": "policy_area", "value": "差旅费"}, {"type": "action", "value": "报销"}]
扩展问题：
- 差旅费报销流程是什么？
- 如何申请差旅费报销？

示例3：
原问题：试用期员工可以享受年假吗？
识别的实体：[{"type": "employee_type", "value": "试用期员工"}, {"type": "policy_area", "value": "年假"}]
扩展问题：
- 试用期员工是否享有年休假？
- 新入职试用期员工的年假政策是怎样的？

请按照上述示例的风格生成扩展问题。输出必须是严格的 JSON 格式。""",
                ),
                MessagesPlaceholder("history", optional=True),
                (
                    "human",
                    """用户问题：{question}

识别的实体信息：
- 实体列表：{entities}
- 约束条件：{constraints}

请生成{count}个不同角度的同义扩展问题。""",
                ),
            ]
        )

        self.chain = self.expand_prompt | self.chat_model | self.parser

    def expand(self, question: str, analysis: Dict[str, Any], history: List = None) -> List[str]:
        if not config.ENABLE_QUERY_EXPANSION:
            return [question]

        entities_str = self._format_entities(analysis.get("entities", []))
        constraints_str = self._format_constraints(analysis.get("constraints", []))

        history_messages = self._format_history(history) if history else []

        try:
            result = self.chain.invoke(
                {
                    "question": question,
                    "entities": entities_str,
                    "constraints": constraints_str,
                    "count": config.QUERY_EXPANSION_COUNT,
                    "history": history_messages,
                }
            )

            expanded = [q.strip() for q in result.expanded_queries if q.strip()]
            filtered = self._filter_queries(question, expanded, analysis)

            return [question] + filtered
        except Exception:
            return [question]

    def _format_entities(self, entities: List[Dict]) -> str:
        if not entities:
            return "无"
        return "\n".join([f"- {e['type']}: {e['value']}" for e in entities])

    def _format_constraints(self, constraints: List[Any]) -> str:
        if not constraints:
            return "无"
        return "\n".join([f"- {c}" for c in constraints])

    def _format_history(self, history: List) -> List:
        messages = []
        for item in history:
            if isinstance(item, HumanMessage):
                messages.append(HumanMessage(content=item.content))
            elif isinstance(item, AIMessage):
                messages.append(AIMessage(content=item.content))
            elif isinstance(item, dict):
                role = item.get("role", "")
                content = item.get("content", "")
                if role == "user":
                    messages.append(HumanMessage(content=content))
                elif role == "assistant":
                    messages.append(AIMessage(content=content))
        return messages

    def _filter_queries(self, original: str, expanded: List[str], analysis: Dict[str, Any]) -> List[str]:
        filtered = []

        original_embedding = self.embedding_model.embed_query(original)
        original_lower = original.lower()

        entity_values = [e["value"].lower() for e in analysis.get("entities", [])]

        for query in expanded:
            if not query or query.lower() == original_lower:
                continue

            if self._check_semantic_similarity(original_embedding, query) and self._check_entity_consistency(query, entity_values):
                filtered.append(query)

        return filtered

    def _check_semantic_similarity(self, original_embedding: List[float], query: str) -> bool:
        try:
            query_embedding = self.embedding_model.embed_query(query)
            similarity = self._cosine_similarity(original_embedding, query_embedding)
            return similarity >= 0.6
        except Exception:
            return True

    def _check_entity_consistency(self, query: str, entity_values: List[str]) -> bool:
        if not entity_values:
            return True

        query_lower = query.lower()
        matched_count = sum(1 for entity in entity_values if entity in query_lower)

        return matched_count >= max(1, len(entity_values) // 2)

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot_product / (norm1 * norm2)
