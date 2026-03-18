from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
APP_NAME = "OfficeMate"

STORAGE_DIR = BASE_DIR / "storage"
RAW_DOCUMENT_DIR = STORAGE_DIR / "raw_documents"
JSON_STORE_DIR = STORAGE_DIR / "json_store"
SAMPLE_DOC_DIR = BASE_DIR / "sample_docs"

DOCUMENT_INDEX_PATH = JSON_STORE_DIR / "documents.json"
QA_LOG_PATH = JSON_STORE_DIR / "qa_logs.json"
FEEDBACK_PATH = JSON_STORE_DIR / "feedback_logs.json"

DOCUMENT_CATEGORIES = [
    "员工手册",
    "HR制度",
    "财务制度",
    "IT支持",
    "行政流程",
    "综合公告",
]
CATEGORY_FILTER_OPTIONS = ["全部"] + DOCUMENT_CATEGORIES
SUPPORTED_FILE_TYPES = ["txt", "pdf", "docx", "xlsx", "csv"]
DEFAULT_VERSION = "v1.0"

collection_name = "officemate_knowledge_base"
persist_directory = str(STORAGE_DIR / "chroma_db")
chunk_size = 800
chunk_overlap = 120
separators = ["\n\n", "\n", "。", "！", "？", ".", "!", "?", "；", ";", " ", ""]
max_split_char_number = 800
similarity_threshold = 4

embedding_model_name = "text-embedding-v4"
chat_model_name = "qwen3-max"
max_history_rounds = 4
max_reference_documents = 4
default_session_prefix = "officemate"

NO_EVIDENCE_MESSAGE = (
    "### 最终回答\n"
    "未找到明确依据，当前知识库中没有足够材料支撑这个问题。\n\n"
    "### 操作步骤/材料清单\n"
    "请先切换更准确的分类，或补充上传对应制度、流程和通知文档。\n\n"
    "### 风险提示\n"
    "在没有明确制度依据前，请不要直接执行流程，建议联系对应部门进一步确认。"
)

QUESTION_TYPE_LABELS = {
    "policy_qa": "制度问答",
    "process_guide": "流程指引",
    "material_list": "材料清单",
    "notice_summary": "通知总结",
}

SAMPLE_DOCS = [
    {
        "file_name": "员工手册.txt",
        "category": "员工手册",
        "title": "员工手册（节选）",
        "version": "v2026.03",
    },
    {
        "file_name": "请假与考勤制度.txt",
        "category": "HR制度",
        "title": "请假与考勤制度",
        "version": "v2026.01",
    },
    {
        "file_name": "差旅与报销制度.txt",
        "category": "财务制度",
        "title": "差旅与报销制度",
        "version": "v2026.02",
    },
    {
        "file_name": "采购申请流程.txt",
        "category": "行政流程",
        "title": "采购申请流程",
        "version": "v2026.01",
    },
    {
        "file_name": "IT服务台常见问题.txt",
        "category": "IT支持",
        "title": "IT服务台常见问题",
        "version": "v2026.02",
    },
    {
        "file_name": "入职与权限开通流程.txt",
        "category": "综合公告",
        "title": "入职与权限开通流程",
        "version": "v2026.01",
    },
]


def ensure_runtime_dirs() -> None:
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DOCUMENT_DIR.mkdir(parents=True, exist_ok=True)
    JSON_STORE_DIR.mkdir(parents=True, exist_ok=True)
    Path(persist_directory).mkdir(parents=True, exist_ok=True)

    for path in (DOCUMENT_INDEX_PATH, QA_LOG_PATH, FEEDBACK_PATH):
        if not path.exists():
            path.write_text("[]", encoding="utf-8")


ensure_runtime_dirs()


# legacy compatibility for the original demo modules
md5_path = str(STORAGE_DIR / "legacy_md5.txt")
session_config = {
    "configurable": {
        "session_id": f"{default_session_prefix}_default",
    }
}
