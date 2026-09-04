# Minimal placeholder ingestion script for Nora

# This script loads PDFs and text files from data/, extracts text, chunks it, and
# writes a simple chunks JSON file or inserts into ChromaDB when available.

import os
import json
from pathlib import Path

try:
    import PyPDF2
except Exception:
    PyPDF2 = None

CHUNK_SIZE = 4000  # characters (approximate)
CHUNK_OVERLAP = 500

DATA_DIR = Path("data")
OUTPUT_CHUNKS = Path("data/chunks.json")


def extract_text_from_pdf(path: Path) -> str:
    if PyPDF2 is None:
        raise RuntimeError("PyPDF2 is required to extract PDF text")
    text = []
    with open(path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text.append(page.extract_text() or "")
    return "\n".join(text)


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    chunks = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + size, length)
        chunks.append(text[start:end])
        start = end - overlap
        if start < 0:
            start = 0
    return chunks


def ingest():
    DATA_DIR.mkdir(exist_ok=True)
    all_chunks = []

    for p in DATA_DIR.glob("**/*"):
        if p.is_dir():
            continue
        if p.suffix.lower() == ".pdf":
            try:
                text = extract_text_from_pdf(p)
            except Exception as e:
                print(f"Failed to extract {p}: {e}")
                continue
        else:
            text = p.read_text(encoding="utf-8", errors="ignore")

        chunks = chunk_text(text)
        for i, c in enumerate(chunks):
            all_chunks.append({
                "doc_id": str(p.name),
                "chunk_id": f"{p.name}_{i}",
                "text": c,
            })

    with open(OUTPUT_CHUNKS, "w", encoding="utf-8") as fh:
        json.dump(all_chunks, fh, ensure_ascii=False, indent=2)

    print(f"Wrote {len(all_chunks)} chunks to {OUTPUT_CHUNKS}")


if __name__ == "__main__":
    ingest()
