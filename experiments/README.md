# Agentic RAG
## Overview
This repository demonstrates how to build an Agentic RAG (Retrieval-Augmented Generation) system using LangGraph with minimal code step-by-step

### Database 
Using Qdrant to store vector data (Vector DB). Code example:

```
from qdrant_client import QdrantClient
path = "our database path"
# 1. Khởi tạo client (chạy ngay trên RAM để test)
client = QdrantClient(path)

# 2. Tạo collection
client.create_collection(
    collection_name="my_docs",
    vectors_config={"size": 4, "distance": "Cosine"}
)

# 3. Tìm kiếm vector gần nhất
results = client.search(
    collection_name="my_docs",
    query_vector=[0.2, 0.1, 0.9, 0.7],
    limit=3
)
```
### LLM
I deploy Qwen3-8B-fp8 for agent and using vLLM framework and Docker to host on GPU Device (DGX Spark 128GB).
Code example:
```
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="qwen3-8b-fp8",
    base_url="http://100.67.127.53:8000/v1",
    api_key="sk-test",
    temperature=0,
)
```

### Processing Documents
I use PDF file and convert it into markdown. I divide markdown text to headings and merge or split if chunk's size is inconsonant.
