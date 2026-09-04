# Streamlit chat app (minimal placeholder)

import os
import streamlit as st
from typing import List

from retriever import get_retriever

st.set_page_config(page_title="Nora — NeuroTrace Health", layout="wide")
st.title("Nora — NeuroTrace Health RAG Chatbot")

retriever = get_retriever(k=5)

if "history" not in st.session_state:
    st.session_state.history = []

with st.form("query"):
    query = st.text_input("Ask a clinical-ops question:")
    submitted = st.form_submit_button("Ask")

if submitted and query:
    with st.spinner("Retrieving sources..."):
        hits = retriever(query, 5)

    context = "\n\n".join([h["text"] for h in hits])

    # Placeholder LLM call — replace with DeepSeek API integration
    def query_deepseek(prompt: str) -> str:
        key = os.environ.get("DEEPSEEK_API_KEY")
        if not key:
            return "[DEEPSEEK_API_KEY not set — set it in your environment or .env file]"
        # Implement the real API call here
        return "(placeholder answer) — replace query_deepseek() with the real API call"

    prompt = f"Use the following context to answer the question. If the answer is not in the context, say 'I don't know' and cite nothing.\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:"

    answer = query_deepseek(prompt)

    st.session_state.history.append({"query": query, "answer": answer, "sources": hits})

for turn in reversed(st.session_state.history[-10:]):
    st.markdown("**Q:** " + turn["query"])
    st.markdown("**A:** " + turn["answer"])
    st.markdown("**Sources:**")
    for s in turn["sources"]:
        st.markdown(f"- {s.get('doc_id')} :: {s.get('chunk_id')}")
