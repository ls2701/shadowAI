"""


This is the KNOWLEDGE side of the RAG (separate from your log chunks). Point it at
a folder of your actual security policy documents and it will:
  1. read every .txt / .md / .pdf file in the folder
  2. split each into overlapping chunks
  3. embed them via Ollama (nomic) and store them in the Chroma collection
     'shadow_ai_knowledge'
  4. re-running re-ingests changed files cleanly (old chunks for that file are
     removed first, so you never get stale duplicates)

shadow_ai_analyze.py then RETRIEVES from this collection — no hardcoded policy.

SETUP (run once):
    pip install chromadb ollama
    pip install pypdf          # only needed if you have .pdf policy docs

USAGE:
    1. put your policy docs in ./knowledge_docs/   (create the folder)
    2. python build_knowledge_base.py
    3. python build_knowledge_base.py --query "is ChatGPT approved?"   # test retrieval
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction


# ---------------- CONFIG ----------------
TOWER_IP = "192.168.100.100"        # where Ollama runs. "localhost" if it's this PC.
OLLAMA_BASE = f"http://{TOWER_IP}:11434"

CHROMA_PATH = "./chroma_db"         # same DB folder as your log chunks
KNOWLEDGE_COLLECTION = "shadow_ai_knowledge"
EMBEDDING_MODEL = "nomic-embed-text-v2-moe:latest"

KNOWLEDGE_DIR = Path("knowledge_docs")   # drop your policy files here
CHUNK_CHARS = 1200                        # ~300 tokens per knowledge chunk
CHUNK_OVERLAP = 150                       # chars carried between chunks


# ---------------- CHROMA ----------------
client = chromadb.PersistentClient(path=CHROMA_PATH)
embedding_function = OllamaEmbeddingFunction(
    url=f"{OLLAMA_BASE}/api/embeddings",
    model_name=EMBEDDING_MODEL,
)
knowledge = client.get_or_create_collection(
    name=KNOWLEDGE_COLLECTION,
    embedding_function=embedding_function,
)


# ---------------- FILE READING ----------------
def read_file(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in (".txt", ".md"):
        return path.read_text(encoding="utf-8", errors="replace")
    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError:
            print(f"  ! {path.name}: install pypdf to read PDFs (pip install pypdf) — skipping")
            return ""
        try:
            reader = PdfReader(str(path))
            return "\n".join((page.extract_text() or "") for page in reader.pages)
        except Exception as exc:
            print(f"  ! {path.name}: could not read PDF ({exc}) — skipping")
            return ""
    print(f"  ! {path.name}: unsupported type {suffix} — skipping")
    return ""


# ---------------- CHUNKING ----------------
def chunk_text(text: str, size: int = CHUNK_CHARS, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Paragraph-aware char chunks with overlap. Never splits mid-word where avoidable."""
    text = " ".join(text.split())          # normalize whitespace
    if not text:
        return []
    chunks, start = [], 0
    while start < len(text):
        end = start + size
        if end < len(text):
            # back off to the last space so we don't cut a word
            space = text.rfind(" ", start, end)
            if space > start:
                end = space
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = end - overlap              # overlap for context continuity
    return [c for c in chunks if c]


# ---------------- INGEST ----------------
def ingest_file(path: Path) -> int:
    text = read_file(path)
    if not text.strip():
        return 0
    source = path.name

    # remove any previous chunks for this file so re-runs don't duplicate/leave stale
    try:
        knowledge.delete(where={"source": source})
    except Exception:
        pass

    chunks = chunk_text(text)
    if not chunks:
        return 0

    ids, docs, metas = [], [], []
    for i, ch in enumerate(chunks):
        # stable id from source + content hash
        cid = f"{path.stem}-{i}-" + hashlib.sha1(ch.encode()).hexdigest()[:8]
        ids.append(cid)
        docs.append(ch)
        metas.append({"source": source, "chunk_index": i, "type": "policy"})

    knowledge.upsert(ids=ids, documents=docs, metadatas=metas)
    return len(chunks)


def build() -> None:
    if not KNOWLEDGE_DIR.exists():
        KNOWLEDGE_DIR.mkdir(parents=True)
        print(f"Created {KNOWLEDGE_DIR}/ — put your policy .txt/.md/.pdf files there, "
              f"then run this again.")
        return

    files = [p for p in sorted(KNOWLEDGE_DIR.iterdir())
             if p.suffix.lower() in (".txt", ".md", ".pdf")]
    if not files:
        print(f"No .txt/.md/.pdf files in {KNOWLEDGE_DIR}/ — add some and re-run.")
        return

    total = 0
    for path in files:
        n = ingest_file(path)
        print(f"  {path.name}: {n} chunks")
        total += n

    print(f"\nDone. {total} chunks across {len(files)} files "
          f"in collection '{KNOWLEDGE_COLLECTION}' ({knowledge.count()} total docs).")


# ---------------- QUERY (test retrieval) ----------------
def test_query(q: str, n: int = 5) -> None:
    res = knowledge.query(query_texts=[q], n_results=n)
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    print(f"\nTop {len(docs)} results for: {q!r}\n")
    for i, (d, m) in enumerate(zip(docs, metas), 1):
        print(f"[{i}] ({m.get('source')}) {d[:200]}...\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", help="test a retrieval query instead of ingesting")
    args = ap.parse_args()
    if args.query:
        test_query(args.query)
    else:
        build()