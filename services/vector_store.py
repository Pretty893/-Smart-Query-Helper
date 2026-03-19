from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config_data as config


class OfficeMateVectorStore:
    def __init__(self):
        self.embedding = DashScopeEmbeddings(model=config.embedding_model_name)
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            separators=config.separators,
            length_function=len,
        )
        self.vector_store = Chroma(
            collection_name=config.collection_name,
            embedding_function=self.embedding,
            persist_directory=config.persist_directory,
        )

    def add_document(self, document_id, text, metadata):
        chunks = self.splitter.split_text(text) if len(text) > config.max_split_char_number else [text]
        metadatas = []
        ids = []
        for index, _ in enumerate(chunks):
            metadatas.append({**metadata, "document_id": document_id, "chunk_index": index})
            ids.append(f"{document_id}-{index}")
        self.vector_store.add_texts(chunks, metadatas=metadatas, ids=ids)
        return len(chunks)

    def delete_document(self, document_id, chunk_count=0):
        chunk_total = int(chunk_count or 0)
        if chunk_total > 0:
            ids = [f"{document_id}-{index}" for index in range(chunk_total)]
            self.vector_store.delete(ids=ids)
            return
        self.vector_store.delete(where={"document_id": document_id})

    def search(self, query, category="全部", limit=None):
        filters = None if category == "全部" else {"category": category}
        limit = limit or config.similarity_threshold
        try:
            return self.vector_store.similarity_search_with_score(
                query,
                k=limit,
                filter=filters,
            )
        except Exception:
            return []
