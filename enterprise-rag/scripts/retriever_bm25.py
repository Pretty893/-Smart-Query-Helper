from typing import List, Tuple, Optional
import jieba

try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False

from .config_manager import get_config
config = get_config()
from .skill_registry import skill_metadata


@skill_metadata(
    name="bm25_retriever",
    description="基于关键词的BM25精确匹配检索，适合查找具体条款、材料清单、数字规定等事实性问题，对关键词敏感",
    skill_type="retriever",
    parameters={
        "query": {
            "type": "string",
            "description": "用户问题，需要包含明确的关键词",
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
            "score": {"type": "float", "description": "BM25得分，越高越匹配"}
        }
    },
    tags=["关键词", "精确匹配", "事实性问题", "条款查询", "材料清单"],
    examples=[
        "病假最长可以请几天",
        "试用期多长时间",
        "报销需要提交什么材料",
        "餐补标准是多少"
    ],
    version="1.0.0"
)
class BM25Retriever:
    def __init__(self, vector_store=None):
        self.vector_store = vector_store
        self.bm25 = None
        self.documents = []

    def search(self, query: str, category: str = "全部", limit: Optional[int] = None) -> List[Tuple]:
        if self.vector_store is None:
            return []

        if not BM25_AVAILABLE:
            return self._fallback_search(query, category, limit)

        limit = limit or config.max_reference_documents

        all_docs = self.vector_store.vector_store.get()
        docs = all_docs["documents"]
        metadatas = all_docs["metadatas"]

        if category != "全部":
            filtered = []
            for doc, meta in zip(docs, metadatas):
                if meta.get("category") == category:
                    filtered.append((doc, meta))
            docs, metadatas = zip(*filtered) if filtered else ([], [])

        if not docs:
            return []

        tokenized_docs = [list(jieba.cut(doc)) for doc in docs]
        self.bm25 = BM25Okapi(tokenized_docs)
        self.documents = list(docs)

        tokenized_query = list(jieba.cut(query))
        scores = self.bm25.get_scores(tokenized_query)

        scored = list(zip(docs, metadatas, scores))
        scored.sort(key=lambda x: x[2], reverse=True)

        results = []
        seen_ids = set()
        for doc, meta, score in scored:
            if score <= 0:
                continue
            key = f"{meta.get('document_id', '')}-{meta.get('chunk_index', '')}"
            if key not in seen_ids:
                seen_ids.add(key)
                results.append((doc, meta, score))
            if len(results) >= limit:
                break

        return results

    def _fallback_search(self, query: str, category: str = "全部", limit: Optional[int] = None) -> List[Tuple]:
        return self.vector_store.search(query, category=category, limit=limit)

    def get_name(self) -> str:
        return "bm25"

    def get_description(self) -> str:
        return self.METADATA["description"]
