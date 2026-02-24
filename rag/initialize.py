import os
from pathlib import Path
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_qdrant.fastembed_sparse import FastEmbedSparse
# from qdrant_client import QdrantClient

DOCS_DIR = "docs"  # Directory containing your pdf files
MARKDOWN_DIR = "markdown" # Directory containing the pdfs converted to markdown
PARENT_STORE_PATH = "parent_store"  # Directory for parent chunk JSON files
CHILD_COLLECTION = "document_child_chunks"

os.makedirs(DOCS_DIR, exist_ok=True)
os.makedirs(MARKDOWN_DIR, exist_ok=True)
os.makedirs(PARENT_STORE_PATH, exist_ok=True)