from typing import List, Tuple, Optional

from .config_manager import get_config
config = get_config()
from .skill_registry import skill_metadata


@skill_metadata(
    name="vector_retriever",
    description="基于向量语义相似度检索，适合理解性问题、同义词匹配、语义关联查询，能够捕捉问题与文档之间的语义关系",
    skill_type="retriever",
    parameters={
        "query": {
            "type": "string",
            "description": "用户问题，需要进行语义理解和匹配",
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
            "score": {"type": "float", "description": "相似度得分，范围0-1，越高越相似"}
        }
    },
    tags=["语义理解", "向量检索", "模糊匹配", "同义词", "语义关联"],
    examples=[
        "年假怎么请比较合适",
        "报销政策具体是什么意思",
        "入职流程相关的规定",
        "出差补贴标准是怎样的"
    ],
    version="1.0.0"
)
class VectorRetriever:
    def __init__(self, vector_store=None):
        self.vector_store = vector_store

    def search(self, query: str, category: str = "全部", limit: Optional[int] = None) -> List[Tuple]:
        if self.vector_store is None:
            return []

        limit = limit or config.max_reference_documents

        if category != "全部":
            results = self.vector_store.vector_store.similarity_search_with_score(
                query,
                k=limit,
                filter={"category": category}
            )
        else:
            results = self.vector_store.vector_store.similarity_search_with_score(query, k=limit)

        processed = []
        seen_ids = set()
        for doc, score in results:
            meta = doc.metadata if hasattr(doc, 'metadata') else {}
            key = f"{meta.get('document_id', '')}-{meta.get('chunk_index', '')}"
            if key not in seen_ids:
                seen_ids.add(key)
                processed.append((doc.page_content if hasattr(doc, 'page_content') else str(doc), meta, score))

        return processed

    def get_name(self) -> str:
        return "vector"

    def get_description(self) -> str:
        return self.METADATA["description"]
