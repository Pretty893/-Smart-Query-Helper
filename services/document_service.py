import hashlib
from pathlib import Path

import config_data as config
from services.document_parser import DocumentParser
from services.storage_service import JsonStorageService
from services.vector_store import OfficeMateVectorStore


class DocumentService:
    def __init__(self):
        self.storage = JsonStorageService()
        self.parser = DocumentParser()
        self.vector_store = None

    def ingest_uploaded_file(self, uploaded_file, category, version, custom_title=""):
        title = custom_title.strip() or Path(uploaded_file.name).stem
        return self.ingest_bytes(
            file_name=uploaded_file.name,
            file_bytes=uploaded_file.getvalue(),
            category=category,
            title=title,
            version=version.strip() or config.DEFAULT_VERSION,
            source_label="manual_upload",
        )

    def ingest_bytes(self, file_name, file_bytes, category, title, version, source_label):
        file_hash = hashlib.sha256(file_bytes).hexdigest()
        existing = self.storage.get_document_by_hash(file_hash)
        if existing and existing.get("status") == "success":
            return {
                "status": "duplicate",
                "message": f"《{existing['title']}》已存在，已跳过重复导入。",
                "document": existing,
            }

        text, file_suffix = self.parser.parse(file_name, file_bytes)
        raw_path = self._save_raw_file(file_hash, file_name, file_bytes)
        record_payload = {
            "file_hash": file_hash,
            "file_name": file_name,
            "file_type": file_suffix.lstrip("."),
            "title": title,
            "category": category,
            "version": version,
            "source_label": source_label,
            "raw_path": str(raw_path.relative_to(config.BASE_DIR)),
            "text_length": len(text),
            "chunk_count": 0,
            "status": "processing",
            "error": "",
        }

        if existing:
            record = self.storage.update_document(existing["id"], record_payload)
        else:
            record = self.storage.add_document(record_payload)

        try:
            chunk_count = self._get_vector_store().add_document(
                document_id=record["id"],
                text=text,
                metadata={
                    "title": title,
                    "category": category,
                    "version": version,
                    "file_name": file_name,
                    "uploaded_at": record["uploaded_at"],
                },
            )
            updated_record = self.storage.update_document(
                record["id"],
                {
                    "chunk_count": chunk_count,
                    "status": "success",
                    "error": "",
                },
            )
        except Exception as exc:
            updated_record = self.storage.update_document(
                record["id"],
                {
                    "status": "failed",
                    "error": str(exc),
                },
            )
            return {
                "status": "failed",
                "message": f"《{title}》导入失败：{exc}",
                "document": updated_record,
            }

        return {
            "status": "success",
            "message": f"《{title}》导入成功，共切分 {chunk_count} 个片段。",
            "document": updated_record,
        }

    def delete_document(self, document_id):
        record = self.storage.get_document_by_id(document_id)
        if not record:
            return {
                "status": "not_found",
                "message": "未找到对应的知识文档。",
            }

        title = record.get("title", "未命名文档")
        try:
            if record.get("status") == "success":
                self._get_vector_store().delete_document(
                    document_id=document_id,
                    chunk_count=record.get("chunk_count", 0),
                )

            raw_path = record.get("raw_path")
            if raw_path:
                raw_file = config.BASE_DIR / raw_path
                if raw_file.exists():
                    raw_file.unlink()

            deleted = self.storage.delete_document(document_id)
            if not deleted:
                return {
                    "status": "not_found",
                    "message": "未找到对应的知识文档。",
                }
        except Exception as exc:
            return {
                "status": "failed",
                "message": f"删除《{title}》失败：{exc}",
                "document": record,
            }

        return {
            "status": "success",
            "message": f"已删除《{title}》，并同步移除原始文件与知识库索引。",
            "document": record,
        }

    def seed_sample_documents(self):
        results = []
        for sample in config.SAMPLE_DOCS:
            file_path = config.SAMPLE_DOC_DIR / sample["file_name"]
            results.append(
                self.ingest_bytes(
                    file_name=sample["file_name"],
                    file_bytes=file_path.read_bytes(),
                    category=sample["category"],
                    title=sample["title"],
                    version=sample["version"],
                    source_label="sample_docs",
                )
            )
        return results

    def list_documents(self):
        return self.storage.list_documents()

    def _save_raw_file(self, file_hash, file_name, file_bytes):
        safe_name = file_name.replace(" ", "_")
        raw_path = config.RAW_DOCUMENT_DIR / f"{file_hash[:12]}_{safe_name}"
        raw_path.write_bytes(file_bytes)
        return raw_path

    def _get_vector_store(self):
        if self.vector_store is None:
            self.vector_store = OfficeMateVectorStore()
        return self.vector_store
