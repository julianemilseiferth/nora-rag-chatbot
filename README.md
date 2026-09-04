# Nora — NeuroTrace Health RAG Chatbot

**A retrieval-augmented chatbot that answers clinical operations questions from a document
knowledge base — grounded in retrieved sources, not model memory.**

`Python` · `ChromaDB` · `TF-IDF` · `DeepSeek API` · `Streamlit`

---

## The Problem

NeuroTrace Health is a fictional clinical operations organization. Nora is a chatbot for
clinical operations staff (schedulers, care coordinators) who need quick, sourced answers
from internal documents (SOPs, policies, training guides). Without a retrieval layer,
staff must manually search PDFs and intranet pages — a time-consuming process that
introduces risk when guidance is missed or misremembered.

## My Approach

1. **Ingestion and chunking** — load PDFs and text docs from `data/`, split into overlapping
   chunks (e.g., 800 tokens with 100-token overlap) so passages retain enough context but
   remain small enough for efficient retrieval.
2. **Embedding and indexing** — create embeddings for chunks and store them in ChromaDB for
   similarity search.
3. **Retrieval** — hybrid TF-IDF + vector similarity: TF-IDF provides fast lexical matches
   for exact terminology while vector similarity supports semantic matches for paraphrased
   queries. Top-k chunks are returned per query (k configurable, default 5).
4. **Generation** — retrieved context injected into a prompt sent to the DeepSeek API with
   strict instructions to answer only from provided context and to cite sources when used.
5. **Interface** — Streamlit chat UI that shows the answer and inline source citations.

## Tools & Technologies

- Vector store: ChromaDB
- Retrieval: hybrid TF-IDF + vector similarity
- LLM: DeepSeek API (calls abstracted behind a client)
- Interface: Streamlit
- Language: Python

## Results & Outcomes

- Indexed sample clinical SOPs and policies into retrievable chunks
- Answers clinical-ops questions (scheduling windows, escalation contacts) with source
  citations back to the originating document and chunk
- Tuned chunk size to 800 tokens / 100 overlap; reduced verbosity in prompts to improve
  precision and reduce hallucination rates

## Project Evidence

Commit a screenshot of the running chat interface to `images/nora_chat.png` and embed it
here after you run the app and confirm it displays a real question + answer.

## What I Built

A minimal but complete RAG scaffold: document ingestion and chunking, Chroma-based
vector store creation, a retriever module that returns top-k supporting chunks, and a
Streamlit chat UI that passes context to an LLM (DeepSeek) with a "cite sources or say
I don't know" guardrail.

## Running It

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # add your DEEPSEEK_API_KEY
python src/ingest.py   # build the vector store
streamlit run src/chat.py
```

## Project Structure

```
nora-rag-chatbot/
├── src/
│   ├── ingest.py
│   ├── retriever.py
│   └── chat.py
├── data/                   # source documents (PDFs / text)
├── images/                 # screenshots
├── .env.example
├── .gitignore
├── README.md
└── requirements.txt
```
