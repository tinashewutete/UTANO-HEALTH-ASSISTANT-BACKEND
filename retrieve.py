import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

INDEX_PATH = "faiss_index/index.faiss"
METADATA_PATH = "faiss_index/chunk_metadata.json"
MODEL_NAME = "all-MiniLM-L6-v2"

# Global variables for loaded resources
index = None
chunks = None
model = None

def load_retriever():
    global index, chunks, model
    if index is None or chunks is None or model is None:
        print("Loading FAISS index, metadata and model...")
        index = faiss.read_index(INDEX_PATH)
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            metadata = json.load(f)
            chunks = metadata["chunks"]
        model = SentenceTransformer(MODEL_NAME)
        print("FAISS index, metadata, and model loaded.")
    else:
        print("Retriever already loaded; reusing.")
    return index, chunks, model

def search(query, top_k=2):
    index, chunks, model = load_retriever()
    embedding = model.encode([query], convert_to_numpy=True)
    embedding = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)
    D, I = index.search(embedding, top_k)
    return [chunks[idx] for idx in I[0] if idx < len(chunks)]

def get_context_from_query(query, top_k=2):
    chunks = search(query, top_k=top_k)
    return "\n\n".join(chunks)

# CLI test mode remains the same
if __name__ == "__main__":
    q = input("Enter your question:\n")
    context = get_context_from_query(q)
    print(f"\nRetrieved Context:\n{context}")
