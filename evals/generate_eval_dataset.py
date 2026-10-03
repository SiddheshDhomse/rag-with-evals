"""
Generate Golden Evaluation Dataset for 'Siddhesh_Dhomse_Life_Forecast_2026-2030.pdf'.
Uses the indexed ChromaDB chunks and Groq LLM to synthesize realistic QA pairs with
ground-truth contexts and answers.

Output: data/testsets/siddhesh_forecast_eval.json
"""

import os
import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import settings
from src.models import get_chat_llm
from src.vectorstore import VectorStoreManager

PROMPT_TEMPLATE = """You are an expert evaluator constructing a benchmark test dataset for evaluating a RAG (Retrieval-Augmented Generation) system.

Given the following text passage extracted from a life forecast document (2026-2030), generate 1 high-quality question and ground-truth answer pair based STRICTLY on the facts in the passage.

Passage:
\"\"\"{passage}\"\"\"

Generate JSON with the following format:
{{
  "question": "A clear, realistic question a user would ask about this forecast",
  "ground_truth": "The accurate, complete factual answer grounded entirely in the passage",
  "question_type": "factual" or "timeline" or "comparative" or "advice"
}}

Respond ONLY with valid JSON. Do not include markdown codeblocks or extra conversational text.
"""


def generate_dataset(num_samples: int = 12):
    output_dir = settings.project_root / "data" / "testsets"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "siddhesh_forecast_eval.json"

    print("Fetching document chunks from ChromaDB...")
    vsm = VectorStoreManager()
    col = vsm.chroma_client.get_collection(vsm.collection_name)
    results = col.get(include=["documents", "metadatas"])

    docs = results.get("documents", [])
    metas = results.get("metadatas", [])

    # Filter chunks belonging to the forecast PDF with substantial content
    target_chunks = []
    for d, m in zip(docs, metas):
        if "Siddhesh" in m.get("source", "") and len(d.strip()) > 180:
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
                "source": meta.get("source", "Siddhesh_Dhomse_Life_Forecast_2026-2030.pdf"),
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
    generate_dataset()
