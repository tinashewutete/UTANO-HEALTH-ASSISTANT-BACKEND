import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# File paths
INDEX_PATH = "faiss_index/index.faiss"
METADATA_PATH = "faiss_index/chunk_metadata.json"
MODEL_NAME = "all-MiniLM-L6-v2"

# Load FAISS index, metadata, and encoder model
def load_retriever():
    index = faiss.read_index(INDEX_PATH)
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    model = SentenceTransformer(MODEL_NAME)
    return index, metadata["chunks"], model

# Search top_k matching chunks from Gale encyclopedia
def search(query, top_k=2):
    index, chunks, model = load_retriever()
    embedding = model.encode([query], convert_to_numpy=True)
    embedding = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)
    D, I = index.search(embedding, top_k)
    return [chunks[idx] for idx in I[0] if idx < len(chunks)]

# New function to return a single combined string for OpenAI context
def get_context_from_query(query, top_k=2):
    chunks = search(query, top_k=top_k)
    return "\n\n".join(chunks)

# Optional CLI test mode
if __name__ == "__main__":
    q = input("Enter your question:\n")
    context = get_context_from_query(q)
    print(f"\nRetrieved Context:\n{context}")
