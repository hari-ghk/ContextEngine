from pathlib import Path
from db import collection, model
from langchain_text_splitters import RecursiveCharacterTextSplitter

def ingest_docs(docs_path: str):
    for file in Path(docs_path).rglob("*.txt"):
        text = file.read_text(encoding="utf-8", errors="ignore")
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=500)
        chunks = splitter.split_text(text)
        for i, chunk in enumerate(chunks):
            chunk_id = f"{file}_{i}"
            embedding = model.encode(chunk).tolist()
            collection.add(
                ids=[chunk_id],
                embeddings=[embedding],
                documents=[chunk],
                metadatas=[{"source":str(file)}],
            )
            print(f"Ingested: {file}")