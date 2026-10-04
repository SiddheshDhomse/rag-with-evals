"""
Generate Golden Evaluation Dataset for any indexed document in ChromaDB.
Uses the indexed ChromaDB chunks and LLM to synthesize realistic QA pairs with
ground-truth contexts and answers.

Output: data/testsets/<dataset_name>_eval.json
"""

import os
import sys
import json
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import settings
from src.models import get_chat_llm
from src.vectorstore import VectorStoreManager

PROMPT_TEMPLATE = """You are an expert evaluator constructing a benchmark test dataset for evaluating a RAG (Retrieval-Augmented Generation) system.

Given the following text passage extracted from a reference document, generate 1 high-quality question and ground-truth answer pair based STRICTLY on the facts in the passage.

Passage:
\"\"\"{passage}\"\"\"

Generate JSON with the following format:
{{
  "question": "A clear, realistic question a user would ask about this document",
  "ground_truth": "The accurate, complete factual answer grounded entirely in the passage",
  "question_type": "factual" or "conceptual" or "comparative"
}}

Respond ONLY with valid JSON. Do not include markdown codeblocks or extra conversational text.
"""


def generate_dataset(source_filter: str = "", num_samples: int = 15, output_name: str = "custom_doc_eval"):
    output_dir = settings.project_root / "data" / "testsets"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{output_name}.json"

    print("Fetching document chunks from ChromaDB...")
    vsm = VectorStoreManager()
    col = vsm.chroma_client.get_collection(vsm.collection_name)
    results = col.get(include=["documents", "metadatas"])

    docs = results.get("documents", [])
    metas = results.get("metadatas", [])

    # Filter chunks belonging to the document with substantial content
    target_chunks = []
    for d, m in zip(docs, metas):
        src = m.get("source", "")
        if (not source_filter or source_filter.lower() in src.lower()) and len(d.strip()) > 180:
            target_chunks.append((d, m))

    print(f"Found {len(target_chunks)} eligible chunks from the document.")

    if not target_chunks:
        print("No eligible chunks found. Please ensure the PDF is indexed.")
        return

    # Select evenly distributed chunks across the document
    step = max(1, len(target_chunks) // num_samples)
    selected_chunks = target_chunks[::step][:num_samples]

    llm = get_chat_llm(provider="groq", temperature=0.1)

    eval_items = []
    print(f"Synthesizing {len(selected_chunks)} golden QA pairs using Groq...")

    for i, (chunk_text, meta) in enumerate(selected_chunks):
        prompt = PROMPT_TEMPLATE.format(passage=chunk_text)
        try:
            resp = llm.invoke(prompt)
            content = resp.content.strip()
            # Clean up potential markdown formatting
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:].strip()

            parsed = json.loads(content)
            item = {
                "id": f"q_{i+1:02d}",
                "question": parsed.get("question"),
                "ground_truth": parsed.get("ground_truth"),
                "question_type": parsed.get("question_type", "factual"),
                "reference_context": chunk_text,
                "source": meta.get("source", "reference_document"),
                "page": meta.get("page", None)
            }
            eval_items.append(item)
            print(f"  [{i+1}/{len(selected_chunks)}] Generated: \"{item['question'][:60]}...\"")
        except Exception as e:
            print(f"  [{i+1}/{len(selected_chunks)}] Failed on chunk: {e}")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(eval_items, f, indent=2, ensure_ascii=False)

    print(f"\nSuccessfully generated and saved {len(eval_items)} golden test samples to:")
    print(f" -> {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate golden evaluation dataset from indexed ChromaDB chunks")
    parser.add_argument("--source", type=str, default="", help="Filter chunks by source document name substring")
    parser.add_argument("--samples", type=int, default=15, help="Number of evaluation samples to generate")
    parser.add_argument("--output", type=str, default="custom_doc_eval", help="Output filename slug (without .json)")
    args = parser.parse_args()

    generate_dataset(source_filter=args.source, num_samples=args.samples, output_name=args.output)
