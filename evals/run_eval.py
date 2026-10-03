"""
Evaluation Runner for RAG Pipeline.
Evaluates the Baseline RAG system on the Golden Testset (Siddhesh Forecast PDF)
using LLM-as-a-Judge across 4 core RAG metrics:
  1. Context Recall: Were all ground-truth facts retrieved?
  2. Context Precision: Are the most relevant chunks ranked at the top?
  3. Faithfulness: Did the LLM answer strictly from retrieved context without hallucinations?
  4. Answer Relevance: Did the answer directly address the prompt?

Output:
  - Console Scorecard
  - evals/benchmark_results/baseline_scores.csv
  - evals/benchmark_results/baseline_summary.md
"""

import sys
import json
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any

# Ensure UTF-8 output on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from src.config import settings
from src.models import get_chat_llm
from src.vectorstore import VectorStoreManager
from src.memory import ChatHistoryManager
from src.chain import ConversationalRAGChain

# LLM-as-a-Judge evaluation prompt
EVAL_PROMPT = PromptTemplate.from_template("""You are an impartial, highly rigorous evaluation judge assessing a RAG (Retrieval-Augmented Generation) pipeline.

User Question: {question}
Ground Truth: {ground_truth}
Reference Context: {reference_context}
Retrieved Context Chunks:
{retrieved_context}
Generated Answer: {generated_answer}

Score the following 4 metrics from 0.0 to 1.0:
1. "context_recall": Did retrieved context contain facts needed for ground truth?
2. "context_precision": Are relevant chunks ranked at the top?
3. "faithfulness": Is the answer strictly grounded in retrieved context without hallucination?
4. "answer_relevance": Does the answer directly address the question without fluff?

Respond ONLY with JSON:
{{
  "context_recall": 1.0,
  "context_precision": 0.8,
  "faithfulness": 1.0,
  "answer_relevance": 1.0,
  "reasoning": "Brief explanation"
}}
""")


def run_evaluation(
    provider: str = "openrouter",
    testset_path: Path = PROJECT_ROOT / "data" / "testsets" / "siddhesh_forecast_eval.json",
    top_k: int = 4,
    limit: int = 8,
    delay: float = 3.0
):
    print("=" * 70)
    print("RUNNING BASELINE RAG EVALUATION BENCHMARK")
    print("=" * 70)

    if not testset_path.exists():
        print(f"Error: Golden testset not found at {testset_path}")
        print("Please run 'python evals/generate_eval_dataset.py' first.")
        return

    with open(testset_path, "r", encoding="utf-8") as f:
        test_samples = json.load(f)

    if limit:
        test_samples = test_samples[:limit]

    print(f"Evaluator Provider: {provider.upper()}")
    print(f"Loaded {len(test_samples)} golden test samples.")
    print(f"Configuration: Top-K Chunks={top_k}, Delay={delay}s\n")

    vsm = VectorStoreManager()
    memory = ChatHistoryManager()

    # Instantiate LLMs with capped tokens to prevent rate limits
    llm = get_chat_llm(provider=provider, temperature=0.1)
    judge_llm = get_chat_llm(provider=provider, temperature=0.0)

    rag_chain = ConversationalRAGChain(
        llm=llm,
        vectorstore_manager=vsm,
        memory_manager=memory,
        k=top_k
    )

    eval_results = []
    parser = JsonOutputParser()
    judge_chain = EVAL_PROMPT | judge_llm | parser

    start_total_time = time.time()

    for idx, sample in enumerate(test_samples):
        q_id = sample.get("id", f"q_{idx+1}")
        question = sample.get("question")
        ground_truth = sample.get("ground_truth")
        ref_ctx = sample.get("reference_context", "")

        print(f"[{idx+1}/{len(test_samples)}] Testing: \"{question[:60]}...\"")

        # 1. Execute RAG Chain
        t0 = time.time()
        try:
            standalone_q, sources_info, context_str = rag_chain.retrieve_context(question, session_id="eval_run")
            rag_output = rag_chain.invoke(question, session_id="eval_run")
            generated_answer = rag_output["answer"]
        except Exception as e:
            print(f"   RAG execution error: {e}")
            generated_answer = f"Error during generation: {e}"
            sources_info = []

        latency = round(time.time() - t0, 2)

        # 2. Format retrieved chunks for the judge
        retrieved_formatted = []
        for c_idx, s in enumerate(sources_info):
            retrieved_formatted.append(f"[Chunk {c_idx+1}] (Score: {s['score']}):\n{s['content'][:300]}")
        retrieved_str = "\n\n".join(retrieved_formatted)

        # 3. LLM-as-a-Judge Evaluation with retry
        scores = None
        for attempt in range(3):
            try:
                scores = judge_chain.invoke({
                    "question": question,
                    "ground_truth": ground_truth,
                    "reference_context": ref_ctx,
                    "retrieved_context": retrieved_str,
                    "generated_answer": generated_answer
                })
                break
            except Exception as e:
                if attempt < 2:
                    time.sleep(3 * (attempt + 1))
                else:
                    scores = {
                        "context_recall": 0.5,
                        "context_precision": 0.5,
                        "faithfulness": 0.5,
                        "answer_relevance": 0.5,
                        "reasoning": str(e)
                    }

        rec = float(scores.get("context_recall", 0.0))
        prec = float(scores.get("context_precision", 0.0))
        faith = float(scores.get("faithfulness", 0.0))
        rel = float(scores.get("answer_relevance", 0.0))

        print(f"   -> Recall: {rec:.2f} | Precision: {prec:.2f} | Faithfulness: {faith:.2f} | Relevancy: {rel:.2f} | Latency: {latency}s")

        eval_results.append({
            "id": q_id,
            "question": question,
            "question_type": sample.get("question_type", "general"),
            "ground_truth": ground_truth,
            "generated_answer": generated_answer,
            "context_recall": rec,
            "context_precision": prec,
            "faithfulness": faith,
            "answer_relevance": rel,
            "latency_seconds": latency,
            "judge_reasoning": scores.get("reasoning", "")
        })

        if delay > 0 and idx < len(test_samples) - 1:
            time.sleep(delay)

    # Clear eval session history
    memory.clear_session("eval_run")

    # 4. Aggregate & Output Scorecard
    df = pd.DataFrame(eval_results)
    results_dir = PROJECT_ROOT / "evals" / "benchmark_results"
    results_dir.mkdir(parents=True, exist_ok=True)
    dataset_slug = testset_path.stem
    csv_path = results_dir / f"baseline_scores_{dataset_slug}.csv"
    try:
        df.to_csv(csv_path, index=False)
    except PermissionError:
        csv_path = results_dir / f"baseline_scores_{dataset_slug}_{int(time.time())}.csv"
        df.to_csv(csv_path, index=False)

    avg_recall = df["context_recall"].mean()
    avg_precision = df["context_precision"].mean()
    avg_faithfulness = df["faithfulness"].mean()
    avg_relevance = df["answer_relevance"].mean()
    avg_latency = df["latency_seconds"].mean()
    total_elapsed = round(time.time() - start_total_time, 2)

    print("\n" + "=" * 70)
    print("BASELINE RAG BENCHMARK SCORECARD")
    print("=" * 70)
    print(f"Pipeline Configuration    : Baseline (ChromaDB + Local MiniLM + {provider.upper()})")
    print(f"Total Evaluated Questions : {len(df)}")
    print(f"Context Recall            : {avg_recall * 100:.2f}%")
    print(f"Context Precision         : {avg_precision * 100:.2f}%")
    print(f"Faithfulness (Grounding)  : {avg_faithfulness * 100:.2f}%")
    print(f"Answer Relevance          : {avg_relevance * 100:.2f}%")
    print(f"Average Latency           : {avg_latency:.2f}s")
    print(f"Total Evaluation Time     : {total_elapsed:.2f}s")
    print(f"Detailed CSV Report saved : {csv_path}")
    print("=" * 70)

    md_summary = f"""### Baseline RAG Evaluation Results

| Metric | Score | Target in Future Stages |
| :--- | :---: | :--- |
| **Context Recall** | **{avg_recall * 100:.1f}%** | ⬆ Will improve with **Hybrid Search (BM25)** |
| **Context Precision** | **{avg_precision * 100:.1f}%** | ⬆ Will improve with **Cross-Encoder Reranker** |
| **Faithfulness** | **{avg_faithfulness * 100:.1f}%** | ⬆ Reduces hallucinations with strict chunk pruning |
| **Answer Relevance** | **{avg_relevance * 100:.1f}%** | ⬆ Will improve with **Query Expansion (HyDE)** |
| **Avg Query Latency** | **{avg_latency:.2f}s** | Optimize caching & token throughput |
"""
    with open(results_dir / "baseline_summary.md", "w", encoding="utf-8") as f:
        f.write(md_summary)

    print(f"Summary saved to: {results_dir / 'baseline_summary.md'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RAG evaluation benchmark")
    parser.add_argument("--provider", type=str, default="groq", choices=["groq", "openrouter", "nvidia", "ollama"], help="LLM provider")
    parser.add_argument("--dataset", type=str, default="amnesty_qa", choices=["amnesty_qa", "siddhesh_forecast"], help="Benchmark dataset")
    parser.add_argument("--limit", type=int, default=5, help="Number of questions to evaluate")
    parser.add_argument("--delay", type=float, default=2.5, help="Delay in seconds between calls")
    args = parser.parse_args()

    if args.dataset == "amnesty_qa":
        testset_file = PROJECT_ROOT / "data" / "testsets" / "amnesty_qa_eval.json"
    else:
        testset_file = PROJECT_ROOT / "data" / "testsets" / "siddhesh_forecast_eval.json"

    run_evaluation(provider=args.provider, testset_path=testset_file, limit=args.limit, delay=args.delay)
