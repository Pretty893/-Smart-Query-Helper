import json
from datetime import datetime
from threading import Lock
from uuid import uuid4

import config_data as config


class JsonStorageService:
    _write_lock = Lock()

    def __init__(self):
        config.ensure_runtime_dirs()

    def _read_records(self, path):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            return []

    def _write_records(self, path, records):
        with self._write_lock:
            path.write_text(
                json.dumps(records, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    def _sort_desc(self, records, field):
        return sorted(records, key=lambda item: item.get(field, ""), reverse=True)

    def _now(self):
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def list_documents(self):
        return self._sort_desc(self._read_records(config.DOCUMENT_INDEX_PATH), "uploaded_at")

    def get_document_by_id(self, document_id):
        for record in self._read_records(config.DOCUMENT_INDEX_PATH):
            if record.get("id") == document_id:
                return record
        return None

    def get_document_by_hash(self, file_hash):
        for record in self._read_records(config.DOCUMENT_INDEX_PATH):
            if record.get("file_hash") == file_hash:
                return record
        return None

    def add_document(self, record):
        records = self._read_records(config.DOCUMENT_INDEX_PATH)
        if "id" not in record:
            record["id"] = uuid4().hex
        if "uploaded_at" not in record:
            record["uploaded_at"] = self._now()
        records.append(record)
        self._write_records(config.DOCUMENT_INDEX_PATH, records)
        return record

    def update_document(self, document_id, patch):
        records = self._read_records(config.DOCUMENT_INDEX_PATH)
        updated = None
        for record in records:
            if record.get("id") == document_id:
                record.update(patch)
                updated = record
                break
        self._write_records(config.DOCUMENT_INDEX_PATH, records)
        return updated

    def delete_document(self, document_id):
        records = self._read_records(config.DOCUMENT_INDEX_PATH)
        remaining_records = [record for record in records if record.get("id") != document_id]
        if len(remaining_records) == len(records):
            return False
        self._write_records(config.DOCUMENT_INDEX_PATH, remaining_records)
        return True

    def list_qa_logs(self, limit=None):
        records = self._sort_desc(self._read_records(config.QA_LOG_PATH), "created_at")
        return records[:limit] if limit else records

    def list_session_logs(self, session_id, limit=None):
        records = [
            record
            for record in self._read_records(config.QA_LOG_PATH)
            if record.get("session_id") == session_id
        ]
        records = sorted(records, key=lambda item: item.get("created_at", ""))
        return records[-limit:] if limit else records

    def add_qa_log(self, record):
        records = self._read_records(config.QA_LOG_PATH)
        if "id" not in record:
            record["id"] = uuid4().hex
        if "created_at" not in record:
            record["created_at"] = self._now()
        records.append(record)
        self._write_records(config.QA_LOG_PATH, records)
        return record

    def list_feedback(self):
        return self._sort_desc(self._read_records(config.FEEDBACK_PATH), "created_at")

    def get_feedback_by_qa_log_id(self, qa_log_id):
        for record in self._read_records(config.FEEDBACK_PATH):
            if record.get("qa_log_id") == qa_log_id:
                return record
        return None

    def upsert_feedback(self, qa_log_id, rating, comment, session_id):
        records = self._read_records(config.FEEDBACK_PATH)
        current_time = self._now()
        for record in records:
            if record.get("qa_log_id") == qa_log_id:
                record["rating"] = rating
                record["comment"] = comment
                record["updated_at"] = current_time
                self._write_records(config.FEEDBACK_PATH, records)
                return record

        new_record = {
            "id": uuid4().hex,
            "qa_log_id": qa_log_id,
            "rating": rating,
            "comment": comment,
            "session_id": session_id,
            "created_at": current_time,
            "updated_at": current_time,
        }
        records.append(new_record)
        self._write_records(config.FEEDBACK_PATH, records)
        return new_record

    def get_stats(self):
        documents = self._read_records(config.DOCUMENT_INDEX_PATH)
        qa_logs = self._read_records(config.QA_LOG_PATH)
        feedback = self._read_records(config.FEEDBACK_PATH)
        categories = {record.get("category") for record in documents if record.get("category")}
        return {
            "document_count": len(documents),
            "category_count": len(categories),
            "qa_count": len(qa_logs),
            "feedback_count": len(feedback),
        }
