"""
Prepare Amnesty QA Benchmark:
1. Extracts all unique passages across the dataset into ChromaDB (creating a realistic corpus with distractors).
2. Extracts the 20 benchmark test samples (Question, Ground Truth, Multi-Context references).
3. Saves the golden testset to data/testsets/amnesty_qa_eval.json.
"""

import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from datasets import load_dataset
from langchain_core.documents import Document
from src.vectorstore import VectorStoreManager


def setup_amnesty_benchmark():
    print("=" * 60)
    print("PREPARING OFFICIAL AMNESTY QA BENCHMARK DATASET")
    print("=" * 60)

    print("Downloading 'explodinggradients/amnesty_qa' (english_v2)...")
    ds = load_dataset("explodinggradients/amnesty_qa", "english_v2", split="eval")

    all_docs = []
    seen_texts = set()
    eval_samples = []

    for idx, row in enumerate(ds):
        q = row["question"]
        gt = row["ground_truth"]
        contexts = row["contexts"]

        eval_samples.append({
            "id": f"amnesty_{idx+1:02d}",
            "question": q,
            "ground_truth": gt,
            "reference_context": "\n\n".join(contexts),
            "reference_contexts_list": contexts,
            "question_type": "multi_context_reasoning"
        })

        for c_idx, ctx_text in enumerate(contexts):
            clean_text = ctx_text.strip()
            if clean_text not in seen_texts:
                seen_texts.add(clean_text)
                all_docs.append(Document(
                    page_content=clean_text,
                    metadata={"source": f"amnesty_doc_{idx+1}", "chunk_id": f"{idx}_{c_idx}"}
                ))

    # Save golden testset JSON
    testset_path = PROJECT_ROOT / "data" / "testsets" / "amnesty_qa_eval.json"
    testset_path.parent.mkdir(parents=True, exist_ok=True)
    with open(testset_path, "w", encoding="utf-8") as f:
        json.dump(eval_samples, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(eval_samples)} multi-context test questions to: {testset_path}")

    # Index into ChromaDB
    print(f"\nIndexing {len(all_docs)} unique passages into ChromaDB...")
    vsm = VectorStoreManager()
    num_added = vsm.add_documents(all_docs, chunk_size=500, chunk_overlap=80)
    print(f"Successfully indexed {num_added} chunks into knowledge base!")

    stats = vsm.get_collection_stats()
    print(f"Updated Vector Store: {stats['total_chunks']} total chunks across {stats['sources_count']} sources.")


if __name__ == "__main__":
    setup_amnesty_benchmark()
