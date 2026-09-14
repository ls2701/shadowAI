"""
RUN THIS ON YOUR LAPTOP — step 2 (after upload_to_tower.py finishes).

Downloads the finished chunks from the Tower and stores them in a
ChromaDB database folder right here on your laptop.

SETUP (run once):
    pip install requests chromadb

USAGE:
    python ingest_to_chroma.py <run_id>

    (run_id is printed by upload_to_tower.py)

After this runs, you'll have a "chroma_db" folder on your laptop —
that IS your vector database. Nothing else to install or configure.
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path

import requests
import chromadb

# CHANGE THIS if your Tower PC has a different IP
TOWER_DOWNLOAD_URL = "http://192.168.100.100:8000/download/{run_id}"

LOCAL_JSONL_DIR = Path("downloaded_chunks")
CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "log_collection"
BATCH_SIZE = 5000


def download_chunks(run_id: str) -> Path:
    LOCAL_JSONL_DIR.mkdir(parents=True, exist_ok=True)
    dest = LOCAL_JSONL_DIR / f"{run_id}_chunks.jsonl"
    url = TOWER_DOWNLOAD_URL.format(run_id=run_id)

    print(f"Downloading chunks from Tower...")
    with requests.get(url, stream=True, timeout=300) as response:
        response.raise_for_status()
        with dest.open("wb") as f:
            for block in response.iter_content(chunk_size=1024 * 1024):
                f.write(block)

    print(f"Saved to {dest.resolve()}")
    return dest


def ingest_into_chroma(jsonl_path: Path) -> int:
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    docs, embeds, metas, ids = [], [], [], []
    total = 0

    def flush():
        nonlocal docs, embeds, metas, ids
        if ids:
            collection.upsert(documents=docs, embeddings=embeds, metadatas=metas, ids=ids)
        docs, embeds, metas, ids = [], [], [], []

    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            docs.append(record["text"])
            embeds.append(record["embedding"])
            metas.append({
                "sensor": record["sensor"],
                "source": record["source"],
                "start_ts": record["start_ts"],
                "end_ts": record["end_ts"],
                "event_count": record["event_count"],
            })
            ids.append(record["id"])
            total += 1
            if len(ids) >= BATCH_SIZE:
                flush()

    flush()
    return total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id", help="run_id printed by upload_to_tower.py")
    args = parser.parse_args()

    jsonl_path = download_chunks(args.run_id)
    count = ingest_into_chroma(jsonl_path)
    print(f"\nDone! Stored {count} chunks in ChromaDB collection '{COLLECTION_NAME}'")
    print(f"Database location: {Path(CHROMA_DB_PATH).resolve()}")


if __name__ == "__main__":
    main()