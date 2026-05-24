#!/usr/bin/env python3
"""
Verifica se uno specifico file Markdown è presente tra i chunk.

Uso:
    python tests/check_chunk.py data/processed/markdown/cc/ccd9ae5100ec12a8.md
    python tests/check_chunk.py data/processed/markdown/cc/ccd9ae5100ec12a8.md --show-text
"""
import sys
import json
import gzip
from pathlib import Path

CHUNKS_FILE = Path("data/processed/chunks/chunks.jsonl")


def normalize_md_path(md_path: str) -> str:
    """Normalizza il path del markdown per il matching."""
    return str(md_path).replace("\\", "/").strip()


def find_chunks_for_markdown(target_md: str, show_text: bool = False):
    target_md = normalize_md_path(target_md)
    found = []

    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            chunk = json.loads(line)
            provenance_path = normalize_md_path(
                chunk.get("provenance", {}).get("index_markdown_path", "")
            )
            if provenance_path == target_md:
                found.append(chunk)

    return found


def main():
    if len(sys.argv) < 2:
        print(f"Uso: {sys.argv[0]} <path-markdown> [--show-text]")
        sys.exit(1)

    target_md = sys.argv[1]
    show_text = "--show-text" in sys.argv

    if not Path(target_md).exists():
        print(f"❌ File non trovato: {target_md}")
        sys.exit(1)

    found = find_chunks_for_markdown(target_md, show_text=show_text)

    if not found:
        print(f"❌ Nessun chunk trovato per il markdown:\n   {target_md}")
        sys.exit(1)

    print(f"✅ Trovati {len(found)} chunk per:\n   {target_md}")
    print()

    for i, chunk in enumerate(found, 1):
        print(f"─── Chunk {i}/{len(found)} ───────────────────────")
        print(f"  chunk_id : {chunk.get('chunk_id', 'N/D')}")
        print(f"  chars    : {chunk.get('chars', 'N/D')}")
        print(f"  title    : {chunk.get('retrieval_metadata', {}).get('title', 'N/D')}")
        print(f"  url      : {chunk.get('retrieval_metadata', {}).get('source_url', 'N/D')}")
        if show_text:
            text = chunk.get("text", "")
            print(f"  text ({len(text)} caratteri):")
            print(f"  {text}")
        print()

    print(f"Riepilogo: {len(found)} chunk(s) trovati")


if __name__ == "__main__":
    main()
