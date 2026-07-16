try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config_data as config

try:
    import jieba
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False


class OfficeMateVectorStore:
    def __init__(self):
        self.embedding = DashScopeEmbeddings(model=config.embedding_model_name)
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            separators=config.separators,
            length_function=len,
        )
        try:
            self.vector_store = Chroma(
                collection_name=config.collection_name,
                embedding_function=self.embedding,
                persist_directory=config.persist_directory,
            )
        except Exception:
            self.vector_store = None

    def add_document(self, document_id, text, metadata):
        chunks = self.splitter.split_text(text) if len(text) > config.max_split_char_number else [text]
        metadatas = []
        ids = []
        for index, _ in enumerate(chunks):
            metadatas.append({**metadata, "document_id": document_id, "chunk_index": index})
            ids.append(f"{document_id}-{index}")
        self.vector_store.add_texts(chunks, metadatas=metadatas, ids=ids)
        return len(chunks)#列表

    def delete_document(self, document_id, chunk_count=0):
        chunk_total = int(chunk_count or 0)
        if chunk_total > 0:
            ids = [f"{document_id}-{index}" for index in range(chunk_total)]
            self.vector_store.delete(ids=ids)
            return 
        self.vector_store.delete(where={"document_id": document_id})

    def search(self, query, category="全部", limit=None):
        if config.ENABLE_STRUCTURED_INDEX:
            return self.structured_search(query, category=category, limit=limit)
        if config.ENABLE_HYBRID_SEARCH and BM25_AVAILABLE:
            return self.hybrid_search(query, category=category, limit=limit)
        return self.vector_search(query, category=category, limit=limit)

    def vector_search(self, query, category="全部", limit=None):
        filters = None if category == "全部" else {"category": category}
        limit = limit or config.similarity_threshold
        try:
            return self.vector_store.similarity_search_with_score(
                query,
                k=limit,
                filter=filters,
            )
        except ValueError as e:
            if "Arrearage" in str(e) or "Access denied" in str(e):
                raise RuntimeError(
                    "阿里云账号欠费或权限不足，无法调用 embedding 服务。"
                    "请检查您的阿里云账号状态并充值后重试。"
                )
            return []
        except Exception:
            return []

    def bm25_search(self, query, category="全部", limit=None):
        if not BM25_AVAILABLE:
            return []
        
        limit = limit or config.similarity_threshold
        
        try:
            all_docs = self.vector_store.get(include=["documents", "metadatas"])
            docs = all_docs["documents"]
            metadatas = all_docs["metadatas"]
            
            if category != "全部":
                filtered = [
                    (doc, meta) for doc, meta in zip(docs, metadatas)
                    if meta.get("category") == category
                ]
                if not filtered:
                    return []
                docs, metadatas = zip(*filtered)
            
            tokenized_docs = [list(jieba.cut(doc)) for doc in docs]
            bm25 = BM25Okapi(tokenized_docs)
            tokenized_query = list(jieba.cut(query))
            scores = bm25.get_scores(tokenized_query)
            
            from langchain_core.documents import Document
            
            results = []
            for idx, score in enumerate(scores):
                if score > 0:
                    doc = Document(
                        page_content=docs[idx],
                        metadata=metadatas[idx]
                    )
                    results.append((doc, score))
            
            results.sort(key=lambda x: x[1], reverse=True)
            return results[:limit]
        except Exception:
            return []

    def hybrid_search(self, query, category="全部", limit=None):
        limit = limit or config.similarity_threshold
        
        vector_results = self.vector_search(query, category=category, limit=limit * 2)
        bm25_results = self.bm25_search(query, category=category, limit=limit * 2)
        
        seen_ids = set()#集合，用于存储已处理的文档ID
        merged = []
        
        for doc, score in vector_results:
            key = f"{doc.metadata.get('document_id', '')}-{doc.metadata.get('chunk_index', '')}"
            if key not in seen_ids:
                seen_ids.add(key)
                merged.append((doc, score, "vector"))
        
        for doc, score in bm25_results:
            key = f"{doc.metadata.get('document_id', '')}-{doc.metadata.get('chunk_index', '')}"
            if key not in seen_ids:
                seen_ids.add(key)
                merged.append((doc, score, "bm25"))
        
        merged.sort(key=lambda x: x[1])
        return [(doc, score) for doc, score, _ in merged[:limit]]

    def structured_search(self, query, category="全部", limit=None):
        limit = limit or config.similarity_threshold
        
        candidate_docs = self._find_relevant_documents(query, category=category)
        
        if not candidate_docs:
            if config.ENABLE_HYBRID_SEARCH and BM25_AVAILABLE:
                return self.hybrid_search(query, category=category, limit=limit)
            return self.vector_search(query, category=category, limit=limit)
        
        doc_ids = [doc["document_id"] for doc in candidate_docs]
        
        if config.ENABLE_HYBRID_SEARCH and BM25_AVAILABLE:
            results = self.hybrid_search(query, category=category, limit=limit * 2)
        else:
            results = self.vector_search(query, category=category, limit=limit * 2)
        
        filtered = [(doc, score) for doc, score in results 
                    if doc.metadata.get("document_id") in doc_ids]
        
        return filtered[:limit]

    def _find_relevant_documents(self, query, category="全部"):
        try:
            all_docs = self.vector_store.get(include=["metadatas"])
            metadatas = all_docs["metadatas"]
            
            doc_info = {}#按文档ID分组
            for meta in metadatas:
                doc_id = meta.get("document_id")
                if doc_id not in doc_info:
                    doc_info[doc_id] = {
                        "document_id": doc_id,
                        "title": meta.get("title", ""),
                        "category": meta.get("category", ""),
                        "outline": [],
                    }
                
                outline_str = meta.get("outline", "")
                if outline_str:
                    try:
                        import json
                        outline = json.loads(outline_str)
                        if isinstance(outline, list):
                            doc_info[doc_id]["outline"] = outline#层级要是列表
                    except Exception:
                        pass
            
            if category != "全部":
                doc_info = {k: v for k, v in doc_info.items() if v.get("category") == category}
            
            if BM25_AVAILABLE:
                doc_texts = []
                doc_items = []
                for doc_id, info in doc_info.items():#字典嵌套字典
                    text = info["title"]
                    for section in info["outline"]:
                        text += " " + section.get("title", "")
                        text += " " + section.get("content_summary", "")
                        for kw in section.get("keywords", []):
                            text += " " + kw
                    doc_texts.append(text)#建立层级结构
                    doc_items.append(info)
                
                #关键词搜索
                tokenized_docs = [list(jieba.cut(doc)) for doc in doc_texts]
                bm25 = BM25Okapi(tokenized_docs)
                tokenized_query = list(jieba.cut(query))
                scores = bm25.get_scores(tokenized_query)
                
                scored = [(doc, score) for doc, score in zip(doc_items, scores)]
                scored.sort(key=lambda x: x[1], reverse=True)
                
                return [doc for doc, score in scored if score > 0.1][:5]
            
            else:
                from langchain_core.embeddings import Embeddings
                
                query_embedding = self.embedding.embed_query(query)
                scored = []
                
                for doc_id, info in doc_info.items():
                    text = info["title"]
                    for section in info["outline"]:
                        text += " " + section.get("title", "")
                        text += " " + section.get("content_summary", "")
                    
                    doc_embedding = self.embedding.embed_query(text)
                    similarity = self._cosine_similarity(query_embedding, doc_embedding)
                    scored.append((info, similarity))
                
                scored.sort(key=lambda x: x[1], reverse=True)
                return [doc for doc, score in scored if score > 0.3][:5]
        
        except Exception:
            return []

    def _cosine_similarity(self, vec1, vec2):
        import math
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(b * b for b in vec2))
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        return dot_product / (magnitude1 * magnitude2)
