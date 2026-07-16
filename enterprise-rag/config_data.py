import os

embedding_model_name = "text-embedding-v1"
chat_model_name = "qwen-turbo"
max_reference_documents = 5
vector_db_path = "./vector_db"
bm25_index_path = "./bm25_index"

QIANFAN_AK = os.environ.get("QIANFAN_AK", "")
QIANFAN_SK = os.environ.get("QIANFAN_SK", "")
