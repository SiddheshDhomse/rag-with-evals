"""
Native Ragas Evaluation Script.
Directly executes the official `from ragas import evaluate` library using:
  - Metrics: faithfulness, answer_relevancy, context_precision, context_recall
  - Evaluator LLM: LangchainLLMWrapper wrapping our project LLM
  - Benchmark Dataset: data/testsets/amnesty_qa_eval.json

Usage:
    python evals/run_native_ragas.py --limit 3 --provider ollama
    python evals/run_native_ragas.py --limit 3 --provider groq
"""

import sys
import json
import time
import argparse
from pathlib import Path

# Ensure UTF-8 output on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from src.config import settings
from src.models import get_chat_llm, get_embedding_model
from src.vectorstore import VectorStoreManager
from src.memory import ChatHistoryManager
from src.chain import ConversationalRAGChain


def run_native_ragas(provider: str = "ollama", limit: int = 3):
    print("=" * 70)
    print("🚀 RUNNING OFFICIAL NATIVE RAGAS EVALUATION (`ragas.evaluate`)")
    print("=" * 70)
    print(f"Evaluator Provider: {provider.upper()} | Samples to Evaluate: {limit}")

    testset_path = PROJECT_ROOT / "data" / "testsets" / "amnesty_qa_eval.json"
    with open(testset_path, "r", encoding="utf-8") as f:
        samples = json.load(f)[:limit]

    vsm = VectorStoreManager()
    memory = ChatHistoryManager()
    llm = get_chat_llm(provider=provider, temperature=0.1)

    rag_chain = ConversationalRAGChain(
        llm=llm,
        vectorstore_manager=vsm,
        memory_manager=memory,
        k=4
    )

    questions = []
    answers = []
    contexts = []
    ground_truths = []

    print("\n1. Generating RAG pipeline responses for evaluation set...")
    for idx, sample in enumerate(samples):
        q = sample["question"]
        gt = sample["ground_truth"]
        print(f"   [{idx+1}/{len(samples)}] Querying: \"{q[:55]}...\"")

        standalone_q, sources_info, context_str = rag_chain.retrieve_context(q, session_id="ragas_eval")
        rag_output = rag_chain.invoke(q, session_id="ragas_eval")

        retrieved_texts = [s["content"] for s in sources_info] if sources_info else ["No context retrieved"]

        questions.append(q)
        answers.append(rag_output["answer"])
        contexts.append(retrieved_texts)
        ground_truths.append(gt)

    # Clear memory
    memory.clear_session("ragas_eval")

    # Create HuggingFace Dataset required by native Ragas
    eval_dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    })

    print("\n2. Executing official `ragas.evaluate()` with LangchainLLMWrapper...")
    evaluator_llm = LangchainLLMWrapper(llm)
    evaluator_embeddings = LangchainEmbeddingsWrapper(get_embedding_model())

    metrics = [
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall
    ]

    t0 = time.time()
    results = evaluate(
        dataset=eval_dataset,
        metrics=metrics,
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
        raise_exceptions=False
    )
    elapsed = round(time.time() - t0, 2)

    print("\n" + "=" * 70)
    print("📊 OFFICIAL RAGAS EVALUATION RESULTS")
    print("=" * 70)
    print(results)
    print(f"\nEvaluation completed in {elapsed}s")

    # Save to CSV
    results_dir = PROJECT_ROOT / "evals" / "benchmark_results"
    results_dir.mkdir(parents=True, exist_ok=True)
    df = results.to_pandas()
    csv_path = results_dir / f"native_ragas_scores_{provider}.csv"
    df.to_csv(csv_path, index=False)
    print(f"Scores exported to: {csv_path}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", type=str, default="ollama", choices=["ollama", "groq", "openrouter", "nvidia"])
    parser.add_argument("--limit", type=int, default=3)
    args = parser.parse_args()

    run_native_ragas(provider=args.provider, limit=args.limit)
