# Minimal retriever: tries to use ChromaDB when available; otherwise falls back to TF-IDF

import os
import json
from typing import List, Dict

try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except Exception:
    CHROMADB_AVAILABLE = False

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

CHUNKS_FILE = "data/chunks.json"


def get_top_k_tfidf(query: str, k: int = 5) -> List[Dict]:
    with open(CHUNKS_FILE, "r", encoding="utf-8") as fh:
        chunks = json.load(fh)
    texts = [c["text"] for c in chunks]
    vec = TfidfVectorizer(stop_words="english").fit_transform(texts + [query])
    q_vec = vec[-1]
    doc_vecs = vec[:-1]
    scores = cosine_similarity(q_vec, doc_vecs).flatten()
    top_idx = scores.argsort()[::-1][:k]
    return [chunks[i] for i in top_idx]


def get_retriever(k: int = 5):
    if CHROMADB_AVAILABLE:
        client = chromadb.Client(Settings())
        # This is a placeholder — the collection name and embedding function depend on your setup
        collection = client.get_collection("nora_chunks")

        def chroma_query(q: str, kk: int = k):
            results = collection.query(query_texts=[q], n_results=kk)
            # convert results to expected format
            hits = []
            for i in range(len(results["ids"][0])):
                hits.append({
                    "doc_id": results["metadatas"][0][i].get("doc_id"),
                    "chunk_id": results["ids"][0][i],
                    "text": results["documents"][0][i],
                })
            return hits

        return chroma_query
    else:
        return get_top_k_tfidf
