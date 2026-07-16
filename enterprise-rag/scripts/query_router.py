from typing import Dict, Any, Optional, Tuple
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models.tongyi import ChatTongyi

from .config_manager import get_config
config = get_config()
from .skill_registry import list_skills, get_skill_metadata
from .retriever_vector import VectorRetriever
from .retriever_bm25 import BM25Retriever
from .retriever_hybrid import HybridRetriever
from .tool_caller import ToolCaller


class QueryRouter:
    def __init__(self):
        self.chat_model = ChatTongyi(model=config.chat_model_name)
        self.vector_retriever = VectorRetriever()
        self.bm25_retriever = BM25Retriever()
        self.hybrid_retriever = HybridRetriever()
        self.tool_caller = ToolCaller()

    def route(self, query: str) -> Dict[str, Any]:
        routing_result = self._analyze_intent(query)
        return routing_result

    def _analyze_intent(self, query: str) -> Dict[str, Any]:
        skills_info = list_skills()
        
        retriever_skills = [s for s in skills_info if s.get("skill_type") == "retriever"]
        tool_skills = [s for s in skills_info if s.get("skill_type") == "tool"]

        skills_description = "\n\n".join([
            f"技能名称: {skill['name']}\n类型: {skill.get('skill_type', '未知')}\n描述: {skill['description']}\n标签: {skill.get('tags', [])}\n示例: {skill.get('examples', [])}"
            for skill in skills_info
        ])

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """你是一个智能路由专家。请分析用户问题，选择最合适的处理策略。

可用技能列表：
{skills_description}

分析步骤：
1. 判断问题类型：制度问答、流程指引、材料清单、通知总结、复杂问题、工具调用
2. 选择最合适的检索器（如果需要检索）
3. 判断是否需要调用外部工具
4. 选择合适的提示词模板类型

返回格式（JSON）：
{{
  "intent_type": "intent_type",
  "retriever_type": "retriever_type",
  "use_tool": true/false,
  "tool_name": "tool_name",
  "tool_params": {{}},
  "prompt_type": "prompt_type",
  "confidence": 0.0-1.0,
  "reasoning": "选择理由"
}}

intent_type可选值：policy_qa, process_guide, material_list, notice_summary, complex, tool_call
retriever_type可选值：vector, bm25, hybrid, none
prompt_type可选值：policy_qa, process_guide, material_list, notice_summary, complex
""",
                ),
                ("human", "用户问题：{query}"),
            ]
        )

        chain = prompt | self.chat_model | StrOutputParser()
        result = chain.invoke({"query": query, "skills_description": skills_description})

        try:
            import json
            return json.loads(result)
        except (json.JSONDecodeError, ValueError):
            return self._fallback_routing(query)

    def _fallback_routing(self, query: str) -> Dict[str, Any]:
        keywords = ["流程", "步骤", "怎么", "如何", "操作"]
        if any(k in query for k in keywords):
            return {
                "intent_type": "process_guide",
                "retriever_type": "hybrid",
                "use_tool": False,
                "tool_name": None,
                "tool_params": {},
                "prompt_type": "process_guide",
                "confidence": 0.7,
                "reasoning": "问题包含流程相关关键词，使用混合检索和流程指引模板",
            }

        keywords_qa = ["什么", "多少", "多久", "是否", "规定", "条款"]
        if any(k in query for k in keywords_qa):
            return {
                "intent_type": "policy_qa",
                "retriever_type": "bm25",
                "use_tool": False,
                "tool_name": None,
                "tool_params": {},
                "prompt_type": "policy_qa",
                "confidence": 0.7,
                "reasoning": "问题包含事实性关键词，使用BM25精确检索",
            }

        return {
            "intent_type": "policy_qa",
            "retriever_type": "hybrid",
            "use_tool": False,
            "tool_name": None,
            "tool_params": {},
            "prompt_type": "policy_qa",
            "confidence": 0.5,
            "reasoning": "无法确定意图，使用混合检索作为默认策略",
        }

    def get_retriever(self, retriever_type: str):
        if retriever_type == "vector":
            return self.vector_retriever
        elif retriever_type == "bm25":
            return self.bm25_retriever
        elif retriever_type == "hybrid":
            return self.hybrid_retriever
        return self.hybrid_retriever

    def call_tool(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        return self.tool_caller.call_tool(tool_name, **kwargs)

    def get_skill_info(self, skill_name: str) -> Optional[Dict[str, Any]]:
        return get_skill_metadata(skill_name)
