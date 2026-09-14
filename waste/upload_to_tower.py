"""
RUN THIS ON YOUR LAPTOP — step 1.

Sends your CSV log file to the Tower PC for chunking + embedding.

SETUP (run once):
    pip install requests

USAGE:
    python upload_to_tower.py yourlogs.csv

It will print something like:
    {'run_id': 'a1b2c3d4...', 'chunks_created': 842}

COPY that run_id — you need it for step 2 (ingest_to_chroma.py).
"""

from __future__ import annotations
import argparse
from pathlib import Path
import requests

# CHANGE THIS if your Tower PC has a different IP
TOWER_URL = "http://192.168.100.100:8000/upload"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("log_file", type=Path)
    args = parser.parse_args()

    if not args.log_file.is_file():
        parser.error(f"File not found: {args.log_file}")

    print(f"Uploading {args.log_file} to Tower... (this may take a while for large files)")
    with args.log_file.open("rb") as log_file:
        response = requests.post(
            TOWER_URL,
            files={"file": (args.log_file.name, log_file, "text/csv")},
            timeout=3600,
        )
    if response.status_code != 200:
        print(f"\nTower rejected the file (status {response.status_code}):")
        try:
            print(response.json().get("detail", response.text))
        except ValueError:
            print(response.text)
        return

    result = response.json()
    print(result)
    print(f"\n--> Now run: python ingest_to_chroma.py {result['run_id']}")


if __name__ == "__main__":
    main()