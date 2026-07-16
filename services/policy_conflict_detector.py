import json
from typing import List, Dict, Any

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models.tongyi import ChatTongyi

import config_data as config
from services.storage_service import JsonStorageService
from services.document_parser import DocumentParser


RULE_EXTRACTION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """你是一个企业政策规则提取专家。请从文档内容中提取所有政策规则。

输出格式必须是 JSON，包含一个 rules 数组，每个元素包含：
- rule_id: 规则唯一标识（自动生成）
- rule_type: 规则类型（leave_policy/expense_policy/purchase_policy/approval_policy/access_policy）
- subject: 规则主题（如"病假期限"、"差旅标准"）
- condition: 适用条件（如"员工级别"、"入职年限"）
- value: 规则数值或具体要求（如"3天"、"500元/天"）
- source_document: 来源文档标题

请只输出 JSON，不要包含任何其他内容。""",
        ),
        ("human", "文档内容：\n{content}\n\n文档标题：{title}"),
    ]
)

CONFLICT_DETECTION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """你是一个企业政策冲突检测专家。请检查以下规则之间是否存在冲突。

输出格式必须是 JSON，包含一个 conflicts 数组，每个元素包含：
- conflict_id: 冲突唯一标识
- rules: 冲突的规则ID列表
- description: 冲突描述
- severity: 严重程度（critical/major/minor）
- suggestion: 解决建议

如果没有冲突，conflicts 数组为空。

请只输出 JSON，不要包含任何其他内容。""",
        ),
        ("human", "规则列表：\n{rules}"),
    ]
)


class PolicyConflictDetector:
    def __init__(self):
        self.chat_model = ChatTongyi(model=config.chat_model_name)
        self.storage = JsonStorageService()
        self.parser = DocumentParser()
        self.extraction_chain = RULE_EXTRACTION_PROMPT | self.chat_model | JsonOutputParser()
        self.detection_chain = CONFLICT_DETECTION_PROMPT | self.chat_model | JsonOutputParser()

    def _read_document_content(self, record: Dict) -> str:
        raw_path = record.get("raw_path")
        if not raw_path:
            return ""
        try:
            full_path = config.BASE_DIR / raw_path
            file_bytes = full_path.read_bytes()
            text, _ = self.parser.parse(record.get("file_name", ""), file_bytes)
            return text
        except Exception:
            return ""

    def extract_rules_from_document(self, document_id: str, content: str, title: str) -> List[Dict]:
        try:
            result = self.extraction_chain.invoke({
                "content": content,
                "title": title,
            })
            rules = result.get("rules", []) if isinstance(result, dict) else []
            for rule in rules:
                rule["source_document_id"] = document_id
            return rules
        except Exception:
            return []

    def detect_conflicts(self, rules: List[Dict]) -> List[Dict]:
        if len(rules) < 2:
            return []

        rules_json = json.dumps(rules, ensure_ascii=False, indent=2)
        try:
            result = self.detection_chain.invoke({"rules": rules_json})
            return result.get("conflicts", []) if isinstance(result, dict) else []
        except Exception:
            return []

    def check_document_conflicts(self, document_id: str) -> List[Dict]:
        record = self.storage.get_document_by_id(document_id)
        if not record or record.get("status") != "success":
            return []

        all_rules = []
        for doc in self.storage.list_documents():
            if doc["id"] == document_id:
                continue
            if doc.get("status") != "success":
                continue

            content = self._read_document_content(doc)
            if content:
                rules = self.extract_rules_from_document(
                    doc["id"],
                    content,
                    doc.get("title", ""),
                )
                all_rules.extend(rules)

        target_content = self._read_document_content(record)
        target_rules = []
        if target_content:
            target_rules = self.extract_rules_from_document(
                document_id,
                target_content,
                record.get("title", ""),
            )

        all_rules.extend(target_rules)
        return self.detect_conflicts(all_rules)

    def check_all_conflicts(self) -> Dict[str, Any]:
        all_rules = []
        for doc in self.storage.list_documents():
            if doc.get("status") != "success":
                continue

            content = self._read_document_content(doc)
            if content:
                rules = self.extract_rules_from_document(
                    doc["id"],
                    content,
                    doc.get("title", ""),
                )
                all_rules.extend(rules)

        conflicts = self.detect_conflicts(all_rules)
        return {
            "total_rules": len(all_rules),
            "total_conflicts": len(conflicts),
            "conflicts": conflicts,
            "rules": all_rules,
        }