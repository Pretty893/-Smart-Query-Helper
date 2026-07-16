import json
from typing import List, Dict, Any

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models.tongyi import ChatTongyi

import config_data as config


STRUCTURED_EXTRACTION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """你是一个企业知识问答领域的结构化信息提取专家。请从回答中提取以下结构化信息：

输出格式必须是 JSON，包含以下字段：
- approval_chain: 审批链数组，每个元素包含 role（角色）、action（动作）、required（是否必须）
- materials: 所需材料数组，每个元素包含 name（材料名称）、type（材料类型）、required（是否必须）
- deadlines: 截止日期数组，每个元素包含 event（事件）、deadline（截止时间）、unit（单位）
- prerequisites: 前置条件数组，每个元素包含 condition（条件描述）、satisfaction（满足方式）
- roles: 涉及角色数组，每个元素包含 name（角色名称）、responsibility（职责）
- amounts: 金额相关规则数组，每个元素包含 item（项目）、limit（限额）、unit（单位）
- time_limits: 时间限制数组，每个元素包含 item（项目）、limit（限制时长）、unit（单位）

如果某类信息不存在，对应数组为空。

请只输出 JSON，不要包含任何其他内容。""",
        ),
        ("human", "回答内容：\n{answer}"),
    ]
)


class StructuredOutputExtractor:
    def __init__(self):
        self.chat_model = ChatTongyi(model=config.chat_model_name)
        self.extraction_chain = STRUCTURED_EXTRACTION_PROMPT | self.chat_model | JsonOutputParser()

    def extract(self, answer: str) -> Dict[str, Any]:
        try:
            result = self.extraction_chain.invoke({"answer": answer})
            return self._normalize_result(result)
        except Exception:
            return self._fallback_extract(answer)
        
    #整理答案
    def _normalize_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        default_structured = {
            "approval_chain": [],
            "materials": [],
            "deadlines": [],
            "prerequisites": [],
            "roles": [],
            "amounts": [],
            "time_limits": [],
        }

        for key in default_structured:
            if key in result:
                if isinstance(result[key], list):
                    default_structured[key] = result[key]
                else:
                    default_structured[key] = []

        return default_structured

    def _fallback_extract(self, answer: str) -> Dict[str, Any]:
        lines = answer.split("\n")
        materials = []
        approval_chain = []
        deadlines = []
        prerequisites = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            if any(keyword in line for keyword in ("材料", "附件", "提交", "提供")):
                materials.append({"name": line, "type": "document", "required": True})

            if any(keyword in line for keyword in ("审批", "批准", "审核", "签字")):
                parts = line.split("→") if "→" in line else [line]
                for part in parts:
                    approval_chain.append({"role": part.strip(), "action": "审批", "required": True})

            if any(keyword in line for keyword in ("天", "小时", "工作日", "期限")):
                deadlines.append({"event": line, "deadline": "", "unit": "天"})

            if any(keyword in line for keyword in ("先", "之前", "前提", "条件")):
                prerequisites.append({"condition": line, "satisfaction": ""})

        return {
            "approval_chain": approval_chain,
            "materials": materials,
            "deadlines": deadlines,
            "prerequisites": prerequisites,
            "roles": [],
            "amounts": [],
            "time_limits": [],
        }