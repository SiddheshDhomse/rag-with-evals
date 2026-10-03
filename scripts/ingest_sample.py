"""
Seed Script: Ingest sample documents or benchmark datasets into ChromaDB.
Usage:
    python scripts/ingest_sample.py --dataset amnesty_qa
    python scripts/ingest_sample.py --dir data/raw
"""

import sys
import argparse
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import settings
from src.vectorstore import VectorStoreManager
from src.utils import load_file_to_documents, load_amnesty_qa_dataset


def ingest_amnesty_dataset(vectorstore: VectorStoreManager, limit: int = 50):
    print("Loading 'explodinggradients/amnesty_qa' dataset from Hugging Face...")
    data = load_amnesty_qa_dataset(split="eval")
    docs = data["documents"][:limit]

    print(f"Ingesting {len(docs)} documents into ChromaDB ({vectorstore.collection_name})...")
    num_chunks = vectorstore.add_documents(docs, chunk_size=500, chunk_overlap=80)
    print(f"Successfully indexed {num_chunks} chunks!")

    stats = vectorstore.get_collection_stats()
    print(f"Collection status: {stats['total_chunks']} total chunks across {stats['sources_count']} sources.")


def ingest_directory(vectorstore: VectorStoreManager, dir_path: Path):
    if not dir_path.exists():
        print(f"Directory {dir_path} does not exist.")
        return

    supported_exts = [".pdf", ".txt", ".md", ".docx"]
    files = [f for f in dir_path.glob("**/*") if f.suffix.lower() in supported_exts]

    if not files:
        print(f"No supported documents found in {dir_path}")
        return

    print(f"Found {len(files)} files in {dir_path}: {[f.name for f in files]}")
    all_docs = []
    for f in files:
        docs = load_file_to_documents(f, f.name)
        all_docs.extend(docs)

    num_chunks = vectorstore.add_documents(all_docs, chunk_size=600, chunk_overlap=100)
    print(f"Successfully added {num_chunks} chunks to ChromaDB!")


def main():
    parser = argparse.ArgumentParser(description="Ingest documents or dataset into ChromaDB")
    parser.add_argument("--dataset", choices=["amnesty_qa"], default="amnesty_qa", help="Preconfigured benchmark dataset")
    parser.add_argument("--dir", type=str, default=None, help="Local directory containing PDF, TXT, MD, DOCX files")
    parser.add_argument("--clear", action="store_true", help="Clear existing collection before ingesting")
    parser.add_argument("--limit", type=int, default=50, help="Max items to ingest from dataset")

    args = parser.parse_args()

    vectorstore = VectorStoreManager()

    if args.clear:
        print("Clearing existing ChromaDB collection...")
        vectorstore.clear_collection()

    if args.dir:
        ingest_directory(vectorstore, Path(args.dir))
    else:
        ingest_amnesty_dataset(vectorstore, limit=args.limit)


if __name__ == "__main__":
    main()
