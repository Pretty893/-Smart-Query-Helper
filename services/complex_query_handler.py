import ast
from typing import List, Dict, Any, Tuple

from langchain_community.chat_models.tongyi import ChatTongyi

import config_data as config
from services.vector_store import OfficeMateVectorStore


PLANNER_PROMPT_TEMPLATE = """
你是一个企业知识问答领域的顶级规划专家。你的任务是将用户提出的复杂问题分解成一个由多个简单子问题组成的行动计划。

请确保：
1. 每个子问题都能从知识库中独立找到答案
2. 子问题覆盖原始问题的所有方面
3. 严格按照逻辑顺序排列

问题: {question}

请严格按照以下格式输出你的计划:
```python
["子问题1", "子问题2", "子问题3", ...]
```
"""

SYNTHESIS_PROMPT_TEMPLATE = """
你是一个企业知识问答领域的顶级综合专家。你的任务是将多个子问题的回答综合成一个完整、连贯的最终答案。

原始问题: {question}

子问题与回答:
{sub_answers}

请综合以上信息，给出一个完整的最终回答。输出必须严格使用以下 Markdown 标题：
### 最终回答
### 操作步骤/材料清单
### 风险提示

如果某一部分不适用，请写"无"。
"""


class ComplexQueryHandler:
    def __init__(self):
        self.chat_model = ChatTongyi(model=config.chat_model_name)
        self.vector_store = OfficeMateVectorStore()

    def handle(self, question: str, category: str = "全部") -> Tuple[str, List[Dict]]:
        plan = self._generate_plan(question)
        if not plan:
            return "", []

        all_references = []
        sub_answers = []

        for i, sub_question in enumerate(plan, 1):
            search_results = self.vector_store.search(sub_question, category=category, limit=3)
            references = self._build_references(search_results)
            all_references.extend(references)

            context = self._build_context(search_results)
            answer = self._answer_sub_question(sub_question, context)
            sub_answers.append(f"子问题 {i}: {sub_question}\n回答: {answer}")

        final_answer = self._synthesize(question, "\n\n".join(sub_answers))

        seen_ids = set()
        unique_references = []
        for ref in all_references:
            if ref["document_id"] not in seen_ids:
                seen_ids.add(ref["document_id"])
                unique_references.append(ref)

        return final_answer, unique_references

    def _generate_plan(self, question: str) -> List[str]:
        prompt = PLANNER_PROMPT_TEMPLATE.format(question=question)
        messages = [{"role": "user", "content": prompt}]
        response_text = self.chat_model.invoke(messages) or ""

        try:
            plan_str = response_text.split("```python")[1].split("```")[0].strip()
            plan = ast.literal_eval(plan_str)
            return plan if isinstance(plan, list) else []
        except (ValueError, SyntaxError, IndexError):
            return []

    def _answer_sub_question(self, sub_question: str, context: str) -> str:
        prompt = f"""
根据以下参考材料，回答子问题。

参考材料:
{context}

子问题: {sub_question}

请直接给出答案，不要包含多余的解释。
"""
        messages = [{"role": "user", "content": prompt}]
        return self.chat_model.invoke(messages) or ""

    def _synthesize(self, question: str, sub_answers: str) -> str:
        prompt = SYNTHESIS_PROMPT_TEMPLATE.format(question=question, sub_answers=sub_answers)
        messages = [{"role": "user", "content": prompt}]
        return self.chat_model.invoke(messages) or ""

    def _build_context(self, search_results):
        blocks = []
        for index, (document, score) in enumerate(search_results[:3], start=1):
            blocks.append(
                f"[{index}] 标题：{document.metadata.get('title', document.metadata.get('file_name', '未命名文档'))}\n"
                f"分类：{document.metadata.get('category', '未分类')}\n"
                f"内容：{document.page_content[:600]}"
            )
        return "\n\n".join(blocks)

    def _build_references(self, search_results):
        references = []
        seen_document_ids = set()
        for document, score in search_results:
            document_id = document.metadata.get("document_id")
            if document_id in seen_document_ids:
                continue
            seen_document_ids.add(document_id)
            references.append(
                {
                    "document_id": document_id,
                    "title": document.metadata.get("title", document.metadata.get("file_name", "未命名文档")),
                    "category": document.metadata.get("category", "未分类"),
                    "version": document.metadata.get("version", "未填写"),
                    "file_name": document.metadata.get("file_name", ""),
                    "score": round(float(score), 4),
                }
            )
        return references