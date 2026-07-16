import sys
sys.path.insert(0, '.')

from langchain_community.embeddings import DashScopeEmbeddings

embedding = DashScopeEmbeddings(model="text-embedding-v3")
try:
    result = embedding.embed_query("测试")
    print('Embedding successful!')
    print('Vector length:', len(result))
except Exception as e:
    print('Error:', e)
