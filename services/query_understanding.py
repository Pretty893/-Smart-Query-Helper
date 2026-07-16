import json
from typing import Dict, Any, Optional

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models.tongyi import ChatTongyi

import config_data as config


QUERY_UNDERSTANDING_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """你是一个企业知识问答领域的意图分析专家。请分析用户问题并提取以下信息：

输出格式必须是 JSON，包含以下字段：
- intent_type: 问题意图类型，必须是以下之一：
  - policy_qa: 查询具体政策规定（如"病假可以请几天"）
  - process_guide: 查询办理流程（如"怎么报销差旅费"）
  - material_list: 查询所需材料（如"报销需要提交什么"）
  - notice_summary: 要求总结通知（如"这份通知的重点是什么"）
  - complex: 复杂问题，涉及多个政策或需要多步推理（如"从入职到转正的完整流程"）
- entities: 提取问题中的关键实体，是一个对象数组，每个对象包含：
  - type: 实体类型（department/role/employee_type/policy_area/time/money/action）
  - value: 实体值
- constraints: 约束条件数组，如时间限制、金额限制、角色限制等
- is_complex: 是否为复杂问题（需要多文档检索或多步推理）

请只输出 JSON，不要包含任何其他内容。""",
        ),
        ("human", "用户问题：{question}"),
    ]
)


class QueryUnderstanding:
    def __init__(self):
        self.chat_model = ChatTongyi(model=config.chat_model_name)
        self.parser = JsonOutputParser()
        self.chain = QUERY_UNDERSTANDING_PROMPT | self.chat_model | self.parser

    def analyze(self, question: str) -> Dict[str, Any]:
        try:
            result = self.chain.invoke({"question": question})
            return self._normalize_result(result)
        except Exception:
            return self._fallback_analysis(question)

    def _normalize_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        intent_type = result.get("intent_type", "policy_qa")
        entities = result.get("entities", [])
        constraints = result.get("constraints", [])
        is_complex = result.get("is_complex", False)

        if intent_type not in ["policy_qa", "process_guide", "material_list", "notice_summary", "complex"]:
            intent_type = "policy_qa"

        if not isinstance(entities, list):
            entities = []

        if not isinstance(constraints, list):
            constraints = []

        return {
            "intent_type": intent_type,
            "intent_label": config.QUESTION_TYPE_LABELS.get(intent_type, "制度问答"),
            "entities": entities,
            "constraints": constraints,
            "is_complex": is_complex,
        }

    def _fallback_analysis(self, question: str) -> Dict[str, Any]:
        lowered = question.lower()
        if any(keyword in lowered for keyword in ("材料", "附件", "提交什么", "要带什么", "需要什么")):
            intent_type = "material_list"
        elif any(keyword in lowered for keyword in ("流程", "步骤", "怎么走", "怎么发起", "如何办理")):
            intent_type = "process_guide"
        elif any(keyword in lowered for keyword in ("总结", "概括", "通知重点", "提炼")):
            intent_type = "notice_summary"
        elif any(keyword in lowered for keyword in ("怎么", "如何", "全流程", "完整", "一起", "同时")):
            intent_type = "complex"
            is_complex = True
        else:
            intent_type = "policy_qa"
            is_complex = False

        return {
            "intent_type": intent_type,
            "intent_label": config.QUESTION_TYPE_LABELS.get(intent_type, "制度问答"),
            "entities": [],
            "constraints": [],
            "is_complex": is_complex,
        }