from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_models.tongyi import ChatTongyi

import config_data as config
from services.storage_service import JsonStorageService
from services.vector_store import OfficeMateVectorStore


class OfficeMateChatService:
    def __init__(self):
        self.storage = JsonStorageService()
        self.vector_store = None
        self.chat_model = None
        self.prompt_template = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "你是 OfficeMate，负责回答企业内部制度与流程问题。"
                    "你只能依据给定的参考材料回答，不能编造制度、审批规则或联系方式。"
                    "如果材料不足，请明确写“未找到明确依据”。"
                    "输出必须严格使用以下 Markdown 标题："
                    "### 最终回答\n### 操作步骤/材料清单\n### 风险提示\n"
                    "如果某一部分不适用，请写“无”。",
                ),
                MessagesPlaceholder("history"),
                (
                    "human",
                    "当前任务类型：{question_type}\n"
                    "当前分类过滤：{category}\n\n"
                    "参考材料如下：\n{context}\n\n"
                    "用户问题：{question}",
                ),
            ]
        )

    def answer_question(self, question, session_id, category="全部"):
        question_type_key = self.infer_question_type(question)
        question_type = config.QUESTION_TYPE_LABELS[question_type_key]
        search_results = self._get_vector_store().search(question, category=category)
        references = self._build_references(search_results)

        if not references:
            answer = config.NO_EVIDENCE_MESSAGE + "\n\n### 引用文档\n无"
            qa_log = self.storage.add_qa_log(
                {
                    "session_id": session_id,
                    "question": question,
                    "answer": answer,
                    "category": category,
                    "question_type": question_type,
                    "source_docs": [],
                }
            )
            return {
                "answer": answer,
                "question_type": question_type,
                "qa_log_id": qa_log["id"],
                "source_docs": [],
            }

        history = self._build_history(session_id)
        context = self._build_context(search_results)
        chain = self.prompt_template | self._get_chat_model() | StrOutputParser()
        answer_body = chain.invoke(
            {
                "question_type": question_type,
                "category": category,
                "context": context,
                "question": question,
                "history": history,
            }
        )
        full_answer = f"{answer_body.strip()}\n\n### 引用文档\n{self._format_reference_markdown(references)}"
        qa_log = self.storage.add_qa_log(
            {
                "session_id": session_id,
                "question": question,
                "answer": full_answer,
                "category": category,
                "question_type": question_type,
                "source_docs": references,
            }
        )

        return {
            "answer": full_answer,
            "question_type": question_type,
            "qa_log_id": qa_log["id"],
            "source_docs": references,
        }

    def infer_question_type(self, question):
        lowered = question.lower()
        if any(keyword in lowered for keyword in ("材料", "附件", "提交什么", "要带什么", "需要什么")):
            return "material_list"
        if any(keyword in lowered for keyword in ("流程", "步骤", "怎么走", "怎么发起", "如何办理")):
            return "process_guide"
        if any(keyword in lowered for keyword in ("总结", "概括", "通知重点", "提炼")):
            return "notice_summary"
        return "policy_qa"

    def _build_history(self, session_id):
        logs = self.storage.list_session_logs(session_id, limit=config.max_history_rounds)
        messages = []
        for log in logs:
            messages.append(HumanMessage(content=log["question"]))
            messages.append(AIMessage(content=log["answer"]))
        return messages

    def _build_context(self, search_results):
        blocks = []
        for index, (document, score) in enumerate(search_results[: config.max_reference_documents], start=1):
            blocks.append(
                f"[{index}] 标题：{document.metadata.get('title', document.metadata.get('file_name', '未命名文档'))}\n"
                f"分类：{document.metadata.get('category', '未分类')} | "
                f"版本：{document.metadata.get('version', '未填写')} | "
                f"相似度得分：{score:.4f}\n"
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
            if len(references) >= config.max_reference_documents:
                break
        return references

    def _format_reference_markdown(self, references):
        lines = []
        for index, item in enumerate(references, start=1):
            lines.append(
                f"{index}. {item['title']} | 分类：{item['category']} | "
                f"版本：{item['version']} | 文件：{item['file_name']}"
            )
        return "\n".join(lines) if lines else "无"

    def _get_vector_store(self):
        if self.vector_store is None:
            self.vector_store = OfficeMateVectorStore()
        return self.vector_store

    def _get_chat_model(self):
        if self.chat_model is None:
            self.chat_model = ChatTongyi(model=config.chat_model_name)
        return self.chat_model
