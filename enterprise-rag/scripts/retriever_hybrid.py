from typing import List, Tuple, Optional

from .config_manager import get_config
config = get_config()
from .retriever_vector import VectorRetriever
from .retriever_bm25 import BM25Retriever
from .skill_registry import skill_metadata


@skill_metadata(
    name="hybrid_retriever",
    description="混合检索：向量语义检索 + BM25关键词检索，使用RRF（Reciprocal Rank Fusion）算法融合结果，兼顾语义理解和关键词精确匹配，适合复杂问题",
    skill_type="retriever",
    parameters={
        "query": {
            "type": "string",
            "description": "用户问题，可以是复杂问题或包含多个关键词的问题",
            "required": True
        },
        "category": {
            "type": "string",
            "description": "文档分类，用于过滤检索范围",
            "required": False,
            "default": "全部"
        },
        "limit": {
            "type": "integer",
            "description": "返回结果数量限制",
            "required": False,
            "default": 5
        }
    },
    returns={
        "type": "list",
        "description": "文档列表，每个元素包含(doc, meta, score)",
        "items": {
            "doc": {"type": "string", "description": "文档内容片段"},
            "meta": {"type": "object", "description": "文档元数据，包含document_id、title、category等"},
            "score": {"type": "float", "description": "RRF融合得分，越高越匹配"}
        }
    },
    tags=["混合检索", "RRF融合", "复杂问题", "语义+关键词", "综合匹配"],
    examples=[
        "入职半年的员工请5天年假需要什么手续",
        "试用期员工报销差旅费有什么限制",
        "从请假到销假的完整流程是什么",
        "外地出差需要准备哪些材料"
    ],
    version="1.0.0"
)
class HybridRetriever:
    def __init__(self, vector_store=None):
        self.vector_retriever = VectorRetriever(vector_store)
        self.bm25_retriever = BM25Retriever(vector_store)

    def search(self, query: str, category: str = "全部", limit: Optional[int] = None) -> List[Tuple]:
        limit = limit or config.max_reference_documents

        vector_results = self.vector_retriever.search(query, category=category, limit=limit * 2)
        bm25_results = self.bm25_retriever.search(query, category=category, limit=limit * 2)

        seen_ids = set()
        merged = []

        for doc, meta, score in vector_results:
            key = f"{meta.get('document_id', '')}-{meta.get('chunk_index', '')}"
            if key not in seen_ids:
                seen_ids.add(key)
                merged.append((doc, meta, score, "vector"))

        for doc, meta, score in bm25_results:
            key = f"{meta.get('document_id', '')}-{meta.get('chunk_index', '')}"
            if key not in seen_ids:
                seen_ids.add(key)
                merged.append((doc, meta, score, "bm25"))

        merged.sort(key=lambda x: x[2])

        return [(doc, meta, score) for doc, meta, score, _ in merged[:limit]]

    def search_with_rrf(self, query: str, category: str = "全部", limit: Optional[int] = None) -> List[Tuple]:
        limit = limit or config.max_reference_documents

        vector_results = self.vector_retriever.search(query, category=category, limit=limit * 3)
        bm25_results = self.bm25_retriever.search(query, category=category, limit=limit * 3)

        rrf_k = 60

        vector_scores = {}
        for rank, (doc, meta, _) in enumerate(vector_results):
            key = f"{meta.get('document_id', '')}-{meta.get('chunk_index', '')}"
            vector_scores[key] = 1.0 / (rrf_k + rank + 1)

        bm25_scores = {}
        for rank, (doc, meta, _) in enumerate(bm25_results):
            key = f"{meta.get('document_id', '')}-{meta.get('chunk_index', '')}"
            bm25_scores[key] = 1.0 / (rrf_k + rank + 1)

        doc_map = {}
        for doc, meta, _ in vector_results:
            key = f"{meta.get('document_id', '')}-{meta.get('chunk_index', '')}"
            doc_map[key] = (doc, meta)
        for doc, meta, _ in bm25_results:
            key = f"{meta.get('document_id', '')}-{meta.get('chunk_index', '')}"
            if key not in doc_map:
                doc_map[key] = (doc, meta)

        combined = []
        for key in doc_map:
            rrf_score = vector_scores.get(key, 0) + bm25_scores.get(key, 0)
            doc, meta = doc_map[key]
            combined.append((doc, meta, rrf_score))

        combined.sort(key=lambda x: x[2], reverse=True)

        return combined[:limit]

    def get_name(self) -> str:
        return "hybrid"

    def get_description(self) -> str:
        return self.METADATA["description"]
