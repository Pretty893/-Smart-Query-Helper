import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models.tongyi import ChatTongyi

import config_data as config
from services.storage_service import JsonStorageService
from services.vector_store import OfficeMateVectorStore
from services.complex_query_handler import ComplexQueryHandler
from services.grounding_verifier import GroundingVerifier
from services.structured_output_extractor import StructuredOutputExtractor

enterprise_rag_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "enterprise-rag")
sys.path.insert(0, enterprise_rag_path)

from scripts.query_router import QueryRouter
from scripts.retriever_vector import VectorRetriever
from scripts.retriever_bm25 import BM25Retriever
from scripts.retriever_hybrid import HybridRetriever
from scripts.prompt_selector import PromptSelector
from scripts.tool_caller import ToolCaller
from scripts.post_processor import PostProcessor


class OfficeMateChatService:
    def __init__(self):
        self.storage = JsonStorageService()
        self.vector_store = None
        self.chat_model = None
        self.complex_handler = ComplexQueryHandler()
        self.grounding_verifier = GroundingVerifier()
        self.structured_extractor = StructuredOutputExtractor()

        self.query_router = QueryRouter()
        self.prompt_selector = PromptSelector()
        self.tool_caller = ToolCaller()
        self.post_processor = PostProcessor()

        self.retrievers = {
            "vector": VectorRetriever(),
            "bm25": BM25Retriever(),
            "hybrid": HybridRetriever(),
        }

    def _init_retrievers(self):
        vs = self._get_vector_store()
        for name, retriever in self.retrievers.items():
            retriever.vector_store = vs

    def answer_question(self, question, session_id, category="全部", enable_deep_analysis=None):
        deep_analysis_enabled = enable_deep_analysis if enable_deep_analysis is not None else config.ENABLE_DEEP_ANALYSIS

        self._init_retrievers()

        route_result = self.query_router.route(question)
        intent_type = route_result["intent_type"]
        question_type = route_result["intent_label"]
        retriever_strategy = route_result["retriever_strategy"]

        analysis = {
            "intent_type": intent_type,
            "intent_label": question_type,
            "retriever_strategy": retriever_strategy,
            "is_complex": intent_type == "complex",
            "entities": [],
            "constraints": [],
        }

        if intent_type == "complex":
            return self._handle_complex_question(
                question, session_id, category, question_type, analysis, deep_analysis_enabled
            )

        return self._handle_simple_question(
            question, session_id, category, question_type, analysis, deep_analysis_enabled
        )

    def _handle_simple_question(self, question, session_id, category, question_type, analysis, deep_analysis_enabled):
        history = self._build_history(session_id)
        retriever_strategy = analysis.get("retriever_strategy", "hybrid")

        search_results = self._skill_based_search(question, category, retriever_strategy)

        references = self._build_references_from_skill_results(search_results)

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
                    "analysis": analysis,
                    "grounding_result": {"verified": False, "summary": {"overall_risk_level": "medium"}},
                    "structured_data": {},
                }
            )
            return {
                "answer": answer,
                "question_type": question_type,
                "qa_log_id": qa_log["id"],
                "source_docs": [],
                "analysis": analysis,
                "grounding_result": {"verified": False, "summary": {"overall_risk_level": "medium"}},
                "structured_data": {},
            }

        context = self._build_context_from_skill_results(search_results)
        prompt = self.prompt_selector.select(analysis["intent_type"])
        chain = prompt | self._get_chat_model() | StrOutputParser()

        answer_body = chain.invoke(
            {
                "context": context,
                "question": question,
                "history": history,
            }
        )
        full_answer = f"{answer_body.strip()}\n\n### 引用文档\n{self._format_reference_markdown(references)}"

        grounding_result = {"verified": False, "summary": {"overall_risk_level": "medium"}}
        if deep_analysis_enabled and config.ENABLE_GROUNDING_VERIFICATION:
            grounding_result = self.grounding_verifier.verify(full_answer, context)

            if config.ENABLE_SELF_RAG and grounding_result.get("summary", {}).get("overall_risk_level") == "high":
                search_results = self._skill_based_search(question, category, "hybrid")
                references = self._build_references_from_skill_results(search_results)
                context = self._build_context_from_skill_results(search_results)
                answer_body = chain.invoke(
                    {
                        "context": context,
                        "question": question,
                        "history": history,
                    }
                )
                full_answer = f"{answer_body.strip()}\n\n### 引用文档\n{self._format_reference_markdown(references)}"
                grounding_result = self.grounding_verifier.verify(full_answer, context)

        structured_data = {}
        if deep_analysis_enabled and config.ENABLE_STRUCTURED_EXTRACTION:
            structured_data = self.structured_extractor.extract(full_answer)

        qa_log = self.storage.add_qa_log(
            {
                "session_id": session_id,
                "question": question,
                "answer": full_answer,
                "category": category,
                "question_type": question_type,
                "source_docs": references,
                "analysis": analysis,
                "grounding_result": grounding_result,
                "structured_data": structured_data,
            }
        )

        return {
            "answer": full_answer,
            "question_type": question_type,
            "qa_log_id": qa_log["id"],
            "source_docs": references,
            "analysis": analysis,
            "grounding_result": grounding_result,
            "structured_data": structured_data,
        }

    def _handle_complex_question(self, question, session_id, category, question_type, analysis, deep_analysis_enabled):
        answer_body, references = self.complex_handler.handle(question, category)

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
                    "analysis": analysis,
                    "grounding_result": {"verified": False, "summary": {"overall_risk_level": "medium"}},
                    "structured_data": {},
                }
            )
            return {
                "answer": answer,
                "question_type": question_type,
                "qa_log_id": qa_log["id"],
                "source_docs": [],
                "analysis": analysis,
                "grounding_result": {"verified": False, "summary": {"overall_risk_level": "medium"}},
                "structured_data": {},
            }

        context = self._build_context_for_references(references)
        full_answer = f"{answer_body.strip()}\n\n### 引用文档\n{self._format_reference_markdown(references)}"

        grounding_result = {"verified": False, "summary": {"overall_risk_level": "medium"}}
        if deep_analysis_enabled and config.ENABLE_GROUNDING_VERIFICATION:
            grounding_result = self.grounding_verifier.verify(full_answer, context)

        structured_data = {}
        if deep_analysis_enabled and config.ENABLE_STRUCTURED_EXTRACTION:
            structured_data = self.structured_extractor.extract(full_answer)

        qa_log = self.storage.add_qa_log(
            {
                "session_id": session_id,
                "question": question,
                "answer": full_answer,
                "category": category,
                "question_type": question_type,
                "source_docs": references,
                "analysis": analysis,
                "grounding_result": grounding_result,
                "structured_data": structured_data,
            }
        )

        return {
            "answer": full_answer,
            "question_type": question_type,
            "qa_log_id": qa_log["id"],
            "source_docs": references,
            "analysis": analysis,
            "grounding_result": grounding_result,
            "structured_data": structured_data,
        }

    def _skill_based_search(self, question, category, strategy="hybrid"):
        retriever = self.retrievers.get(strategy, self.retrievers["hybrid"])

        try:
            if strategy == "hybrid":
                return retriever.search_with_rrf(question, category=category)
            else:
                return retriever.search(question, category=category)
        except RuntimeError as e:
            if "阿里云账号欠费" in str(e):
                raise
        except Exception:
            return self._get_vector_store().search(question, category=category)

    def _build_history(self, session_id):
        logs = self.storage.list_session_logs(session_id, limit=config.max_history_rounds)
        messages = []
        for log in logs:
            messages.append(HumanMessage(content=log["question"]))
            messages.append(AIMessage(content=log["answer"]))
        return messages

    def _build_context_from_skill_results(self, search_results):
        blocks = []
        for index, (doc, meta, score) in enumerate(search_results[: config.max_reference_documents], start=1):
            blocks.append(
                f"[{index}] 标题：{meta.get('title', meta.get('file_name', '未命名文档'))}\n"
                f"分类：{meta.get('category', '未分类')} | "
                f"版本：{meta.get('version', '未填写')} | "
                f"相似度得分：{score:.4f}\n"
                f"内容：{doc[:600]}"
            )
        return "\n\n".join(blocks)

    def _build_context_for_references(self, references):
        blocks = []
        for index, ref in enumerate(references[: config.max_reference_documents], start=1):
            search_results = self._get_vector_store().search(
                ref["title"],
                category=ref["category"],
                limit=1,
            )
            content = ""
            for document, _ in search_results:
                if document.metadata.get("document_id") == ref["document_id"]:
                    content = document.page_content[:600]
                    break

            blocks.append(
                f"[{index}] 标题：{ref['title']}\n"
                f"分类：{ref['category']} | "
                f"版本：{ref['version']} | "
                f"内容：{content}"
            )
        return "\n\n".join(blocks)

    def _build_references_from_skill_results(self, search_results):
        references = []
        seen_document_ids = set()
        for doc, meta, score in search_results:
            document_id = meta.get("document_id")
            if document_id in seen_document_ids:
                continue
            seen_document_ids.add(document_id)
            references.append(
                {
                    "document_id": document_id,
                    "title": meta.get("title", meta.get("file_name", "未命名文档")),
                    "category": meta.get("category", "未分类"),
                    "version": meta.get("version", "未填写"),
                    "file_name": meta.get("file_name", ""),
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

    def call_tool(self, tool_name, **kwargs):
        return self.tool_caller.call_tool(tool_name, **kwargs)

    def post_process(self, answer, action="summary", **kwargs):
        if action == "summary":
            return self.post_processor.generate_summary(answer, **kwargs)
        elif action == "translate":
            return self.post_processor.translate(answer, **kwargs)
        elif action == "format":
            return self.post_processor.format_answer(answer, **kwargs)
        elif action == "highlight":
            return self.post_processor.add_highlights(answer, **kwargs)
        return answer
