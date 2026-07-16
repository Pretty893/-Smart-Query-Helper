import json
from typing import List, Dict, Any

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models.tongyi import ChatTongyi

import config_data as config


FACT_EXTRACTION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """你是一个企业知识问答领域的事实提取专家。请从回答中提取所有事实声明。

输出格式必须是 JSON，包含一个 facts 数组，每个元素包含：
- statement: 事实声明的文本内容
- type: 事实类型（policy_rule/process_step/material_requirement/time_limit/amount_limit/role_responsibility）
- confidence: 初始置信度（0-1，基于语言确定性）

请只输出 JSON，不要包含任何其他内容。""",
        ),
        ("human", "回答内容：\n{answer}"),
    ]
)
#policy_rule/process_step/material_requirement/time_limit/amount_limit/role_responsibility
#policy_rule：政策规则
#process_step：流程步骤
#material_requirement：物料需求
#time_limit：时间限制
#amount_limit：金额限制
#role_responsibility：角色责任
BATCH_GROUNDNING_CHECK_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """你是一个企业知识问答领域的事实验证专家。请批量检查所有事实声明是否有参考材料支撑。

输出格式必须是 JSON，包含一个 results 数组，每个元素包含：
- statement: 原始事实声明
- is_supported: 是否有参考材料支撑（true/false）
- supporting_evidence: 支撑该事实的原文片段，如果没有则为空字符串
- confidence: 验证后的置信度（0-1）
- risk_level: 风险等级（low/medium/high），如果 is_supported 为 false 则为 high

请只输出 JSON，不要包含任何其他内容。""",
        ),
        ("human", "事实声明列表：\n{facts}\n\n参考材料：\n{context}"),
    ]
)


class GroundingVerifier:
    def __init__(self):
        self.chat_model = ChatTongyi(model=config.chat_model_name)
        self.extraction_chain = FACT_EXTRACTION_PROMPT | self.chat_model | JsonOutputParser()
        self.grounding_chain = BATCH_GROUNDNING_CHECK_PROMPT | self.chat_model | JsonOutputParser()

    def verify(self, answer: str, context: str) -> Dict[str, Any]:
        facts = self._extract_facts(answer)
        if not facts:
            return {"verified": False, "facts": [], "summary": "无法提取事实声明"}

        facts_json = json.dumps(facts, ensure_ascii=False, indent=2)
        try:
            verification = self.grounding_chain.invoke({
                "facts": facts_json,
                "context": context,
            })
            results = verification.get("results", []) if isinstance(verification, dict) else []
        except Exception:
            results = []
            for fact in facts:
                results.append({
                    "statement": fact["statement"],
                    "is_supported": False,
                    "supporting_evidence": "",
                    "confidence": 0.0,
                    "risk_level": "high",
                })

        return self._build_summary(results)

    def _extract_facts(self, answer: str) -> List[Dict]:
        try:
            result = self.extraction_chain.invoke({"answer": answer})
            return result.get("facts", []) if isinstance(result, dict) else []
        except Exception:
            return self._fallback_extract_facts(answer)

    def _fallback_extract_facts(self, answer: str) -> List[Dict]:
        facts = []
        lines = answer.split("\n")
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if any(keyword in line for keyword in ("必须", "需要", "应当", "不得", "可以", "禁止")):
                facts.append({
                    "statement": line,
                    "type": "policy_rule",
                    "confidence": 0.8,
                })
            elif any(keyword in line for keyword in ("步骤", "流程", "首先", "其次", "然后")):
                facts.append({
                    "statement": line,
                    "type": "process_step",
                    "confidence": 0.7,
                })
            elif any(keyword in line for keyword in ("材料", "附件", "提交", "提供")):
                facts.append({
                    "statement": line,
                    "type": "material_requirement",
                    "confidence": 0.8,
                })
        return facts

    def _build_summary(self, results: List[Dict]) -> Dict[str, Any]:
        supported_count = sum(1 for r in results if r.get("is_supported"))
        total_count = len(results)
        avg_confidence = sum(r.get("confidence", 0) for r in results) / total_count if total_count > 0 else 0
        high_risk_count = sum(1 for r in results if r.get("risk_level") == "high")

        return {
            "verified": True,
            "facts": results,
            "summary": {
                "total_facts": total_count,
                "supported_facts": supported_count,
                "unsupported_facts": total_count - supported_count,
                "avg_confidence": round(avg_confidence, 2),
                "high_risk_count": high_risk_count,
                "overall_risk_level": "high" if high_risk_count > 0 else ("medium" if avg_confidence < 0.7 else "low"),
            },
        }