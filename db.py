import chromadb
from sentence_transformers import SentenceTransformer

chroma_client = chromadb.PersistentClient(path="chroma_db")
model = SentenceTransformer("all-MiniLM-L6-v2")
collection = chroma_client.get_or_create_collection(name="python_docs")