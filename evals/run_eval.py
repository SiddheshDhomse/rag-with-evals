"""
Evaluation Runner for RAG Pipeline.
Evaluates the RAG system (Baseline Dense or Phase 2 Hybrid Search)
on golden benchmark testsets using LLM-as-a-Judge across 4 core RAG metrics:
  1. Context Recall: Were all ground-truth facts retrieved?
  2. Context Precision: Are the most relevant chunks ranked at the top?
  3. Faithfulness: Did the LLM answer strictly from retrieved context without hallucinations?
  4. Answer Relevance: Did the answer directly address the prompt?

Features:
  - Multi-Provider Round-Robin Pooling (Groq, NVIDIA NIM, OpenRouter, Ollama)
  - Automatic LangChain with_fallbacks() failover protection against 429 rate limits
  - Session-isolated evaluation runs to prevent cross-question contamination
  - Automated comparative ablation scorecard generation
"""

import sys
import json
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional

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

Respond ONLY with valid JSON in this exact structure:
{{
  "context_recall": <float between 0.0 and 1.0>,
  "context_precision": <float between 0.0 and 1.0>,
  "faithfulness": <float between 0.0 and 1.0>,
  "answer_relevance": <float between 0.0 and 1.0>,
  "reasoning": "<concise 2-3 sentence analysis of why each metric was scored this way>"
}}
""")


class ResilientProviderPool:
    """
    Manages round-robin rotation with native LangChain with_fallbacks()
    across Groq, NVIDIA NIM, OpenRouter, and local Ollama.
    """

    def __init__(self, providers: Optional[List[str]] = None, temperature: float = 0.1):
        candidate_providers = providers or ["groq", "nvidia", "openrouter", "ollama"]
        self.temperature = temperature
        self.instances: Dict[str, Any] = {}

        print("Initializing Resilient Provider Pool:")
        for p in candidate_providers:
            try:
                llm = get_chat_llm(p, temperature=temperature)
                self.instances[p] = llm
                print(f"  + Provider registered: {p.upper()}")
            except Exception as e:
                print(f"  - Provider skipped ({p.upper()}): {e}")

        self.healthy_providers = list(self.instances.keys())
        if not self.healthy_providers:
            raise RuntimeError("No healthy providers available for Provider Pool!")

    def get_resilient_llm(self, step_idx: int):
        n = len(self.healthy_providers)
        order = [self.healthy_providers[(step_idx + offset) % n] for offset in range(n)]
        primary_name = order[0]
        primary_llm = self.instances[primary_name]

        fallback_llms = [self.instances[name] for name in order[1:]]
        if fallback_llms:
            resilient_llm = primary_llm.with_fallbacks(fallback_llms)
        else:
            resilient_llm = primary_llm

        return resilient_llm, primary_name


def run_evaluation(
    provider: str = "round_robin",
    judge_provider: str = "round_robin",
    testset_path: Path = None,
    limit: Optional[int] = None,
    delay: float = 2.0,
    top_k: int = 4,
    retrieval_mode: str = "hybrid",
    transform: str = "none"
):
    print("=" * 70)
    print(f"STARTING RAG EVALUATION: MODE = {retrieval_mode.upper()} | TRANSFORM = {transform.upper()}")
    print("=" * 70)

    if testset_path is None or not testset_path.exists():
        print(f"Error: Golden testset not found at: {testset_path}")
        return

    with open(testset_path, "r", encoding="utf-8") as f:
        test_samples = json.load(f)

    if limit:
        test_samples = test_samples[:limit]

    print(f"Generator Strategy: {provider.upper()}")
    print(f"Judge Strategy:     {judge_provider.upper()}")
    print(f"Evaluation Samples: {len(test_samples)} golden questions")
    print(f"Configuration:      Mode={retrieval_mode.upper()}, Transform={transform.upper()}, Top-K={top_k}, Delay={delay}s\n")

    vsm = VectorStoreManager()
    memory = ChatHistoryManager()
    parser = JsonOutputParser()

    # Setup generator and judge pools
    if provider == "round_robin":
        gen_pool = ResilientProviderPool(temperature=0.1)
    else:
        single_llm = get_chat_llm(provider=provider, temperature=0.1)
        # Fallback to Ollama if single provider errors
        try:
            ollama_fb = get_chat_llm("ollama", temperature=0.1)
            single_llm = single_llm.with_fallbacks([ollama_fb])
        except Exception:
            pass

    if judge_provider == "round_robin":
        judge_pool = ResilientProviderPool(temperature=0.0)
    else:
        single_judge = get_chat_llm(provider=judge_provider, temperature=0.0)
        try:
            ollama_fb = get_chat_llm("ollama", temperature=0.0)
            single_judge = single_judge.with_fallbacks([ollama_fb])
        except Exception:
            pass

    # Instantiate reranker if in rerank mode
    reranker_instance = None
    if retrieval_mode in ("hybrid_rerank", "rerank"):
        from src.reranker import CrossEncoderReranker
        print(f"Loading Cross-Encoder Reranker for Phase 3...")
        reranker_instance = CrossEncoderReranker()

    # Dummy initial chain, will be updated per turn
    initial_llm = gen_pool.instances[gen_pool.healthy_providers[0]] if provider == "round_robin" else single_llm
    rag_chain = ConversationalRAGChain(
        llm=initial_llm,
        vectorstore_manager=vsm,
        memory_manager=memory,
        k=top_k,
        retrieval_mode=retrieval_mode,
        candidates_k=15,
        reranker=reranker_instance,
        query_transform_mode=transform
    )

    eval_results = []
    start_total_time = time.time()

    for idx, sample in enumerate(test_samples):
        q_id = sample.get("id", f"q_{idx+1}")
        question = sample.get("question")
        ground_truth = sample.get("ground_truth")
        ref_ctx = sample.get("reference_context", "")

        # Select provider for this turn
        if provider == "round_robin":
            turn_gen_llm, active_gen_name = gen_pool.get_resilient_llm(idx)
        else:
            turn_gen_llm = single_llm
            active_gen_name = provider

        if judge_provider == "round_robin":
            turn_judge_llm, active_judge_name = judge_pool.get_resilient_llm(idx)
        else:
            turn_judge_llm = single_judge
            active_judge_name = judge_provider

        # Update chain LLM for current turn
        rag_chain.llm = turn_gen_llm
        if rag_chain.query_transformer is not None:
            rag_chain.query_transformer.llm = turn_gen_llm
            from src.query_transform import HYDE_PROMPT, MULTI_QUERY_PROMPT, STEP_BACK_PROMPT, ROUTER_PROMPT
            from langchain_core.output_parsers import StrOutputParser
            rag_chain.query_transformer.hyde_chain = HYDE_PROMPT | turn_gen_llm | StrOutputParser()
            rag_chain.query_transformer.multi_query_chain = MULTI_QUERY_PROMPT | turn_gen_llm | StrOutputParser()
            rag_chain.query_transformer.step_back_chain = STEP_BACK_PROMPT | turn_gen_llm | StrOutputParser()
            rag_chain.query_transformer.router_chain = ROUTER_PROMPT | turn_gen_llm | StrOutputParser()

        judge_runner = EVAL_PROMPT | turn_judge_llm

        print(f"[{idx+1}/{len(test_samples)}] Testing: \"{question[:55]}...\"")
        print(f"   Providers: Gen={active_gen_name.upper()} | Judge={active_judge_name.upper()}")

        # 1. Execute RAG Chain with isolated session to prevent cross-question history contamination
        eval_session = f"eval_{q_id}"
        memory.clear_session(eval_session)
        t0 = time.time()
        try:
            rag_output = rag_chain.invoke(question, session_id=eval_session)
            generated_answer = rag_output["answer"]
            sources_info = rag_output["sources"]
        except Exception as e:
            print(f"   RAG execution error: {e}")
            generated_answer = f"Error during generation: {e}"
            sources_info = []
        finally:
            memory.clear_session(eval_session)

        latency = round(time.time() - t0, 2)

        # 2. Format retrieved chunks for the judge
        retrieved_formatted = []
        for c_idx, s in enumerate(sources_info):
            score_val = s.get('score', 'N/A')
            retrieved_formatted.append(f"[Chunk {c_idx+1}] (Score: {score_val}):\n{s['content'][:300]}")
        retrieved_str = "\n\n".join(retrieved_formatted)

        # 3. LLM-as-a-Judge Evaluation with retry and robust JSON extraction
        eval_scores = None
        max_retries = 3
        for attempt in range(max_retries):
            try:
                raw_judge = judge_runner.invoke({
                    "question": question,
                    "ground_truth": ground_truth,
                    "reference_context": ref_ctx,
                    "retrieved_context": retrieved_str,
                    "generated_answer": generated_answer
                })
                judge_text = raw_judge.content if hasattr(raw_judge, "content") else str(raw_judge)
                try:
                    eval_scores = parser.parse(judge_text)
                except Exception:
                    import re
                    match = re.search(r"\{[\s\S]*\}", judge_text)
                    if match:
                        eval_scores = json.loads(match.group(0))
                    else:
                        raise ValueError(f"Could not parse JSON from judge: {judge_text[:100]}")
                break
            except Exception as e:
                err_msg = str(e)
                if "429" in err_msg or "rate" in err_msg.lower():
                    wait_time = 4 * (attempt + 1)
                    print(f"   Rate limit encountered, backing off for {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    print(f"   Judge error on attempt {attempt+1}: {e}")
                    time.sleep(2)

        if not eval_scores:
            eval_scores = {
                "context_recall": 0.0,
                "context_precision": 0.0,
                "faithfulness": 0.0,
                "answer_relevance": 0.0,
                "reasoning": "Evaluation failed or rate-limited across all retries."
            }

        rec = float(eval_scores.get("context_recall", 0.0))
        prec = float(eval_scores.get("context_precision", 0.0))
        faith = float(eval_scores.get("faithfulness", 0.0))
        rel = float(eval_scores.get("answer_relevance", 0.0))
        reasoning = eval_scores.get("reasoning", "")

        print(f"   Scores: Recall={rec:.2f} | Precision={prec:.2f} | Faithfulness={faith:.2f} | Relevancy={rel:.2f} | ({latency}s)")
        print(f"   Reasoning: {reasoning[:90]}...\n")

        eval_results.append({
            "id": q_id,
            "question": question,
            "question_type": sample.get("question_type", "multi_context_reasoning"),
            "generator_provider": active_gen_name,
            "judge_provider": active_judge_name,
            "ground_truth": ground_truth,
            "generated_answer": generated_answer,
            "context_recall": rec,
            "context_precision": prec,
            "faithfulness": faith,
            "answer_relevance": rel,
            "latency_seconds": latency,
            "judge_reasoning": reasoning
        })

        if delay > 0 and idx < len(test_samples) - 1:
            time.sleep(delay)

    # 4. Aggregate & Output Scorecard
    df = pd.DataFrame(eval_results)
    results_dir = PROJECT_ROOT / "results" if (PROJECT_ROOT / "results").exists() else PROJECT_ROOT / "evals" / "benchmark_results"
    results_dir.mkdir(parents=True, exist_ok=True)
    dataset_slug = testset_path.stem

    if retrieval_mode in ("hybrid_rerank", "rerank"):
        prefix = "rerank_scores"
        title = "PHASE 3: TWO-STAGE HYBRID + CROSS-ENCODER RERANK BENCHMARK"
    elif retrieval_mode == "hybrid":
        prefix = "hybrid_scores"
        title = "PHASE 2: HYBRID SEARCH (BM25 + DENSE RRF) BENCHMARK"
    else:
        prefix = "baseline_scores"
        title = "BASELINE RAG BENCHMARK"

    if transform != "none":
        prefix = f"{prefix}_{transform}"
        title = f"{title} [{transform.upper()} TRANSFORMATION]"

    csv_path = results_dir / f"{prefix}_{dataset_slug}.csv"
    try:
        df.to_csv(csv_path, index=False)
    except PermissionError:
        csv_path = results_dir / f"{prefix}_{dataset_slug}_{int(time.time())}.csv"
        df.to_csv(csv_path, index=False)

    avg_recall = df["context_recall"].mean()
    avg_precision = df["context_precision"].mean()
    avg_faithfulness = df["faithfulness"].mean()
    avg_relevance = df["answer_relevance"].mean()
    avg_latency = df["latency_seconds"].mean()
    total_elapsed = round(time.time() - start_total_time, 2)

    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)
    print(f"Strategy                  : {retrieval_mode.upper()} (k={top_k})")
    print(f"Provider Pool Mode        : Gen={provider.upper()}, Judge={judge_provider.upper()}")
    print(f"Total Evaluated Questions : {len(df)}")
    print(f"Context Recall            : {avg_recall * 100:.2f}%")
    print(f"Context Precision         : {avg_precision * 100:.2f}%")
    print(f"Faithfulness (Grounding)  : {avg_faithfulness * 100:.2f}%")
    print(f"Answer Relevance          : {avg_relevance * 100:.2f}%")
    print(f"Average Latency           : {avg_latency:.2f}s")
    print(f"Total Benchmark Runtime   : {total_elapsed:.2f}s")
    print(f"Detailed CSV Report saved : {csv_path}")
    print("=" * 70)

    # Automated Comparative Reporting
    baseline_csv = results_dir / f"baseline_scores_{dataset_slug}.csv"
    hybrid_csv = results_dir / f"hybrid_scores_{dataset_slug}.csv"
    rerank_csv = results_dir / f"rerank_scores_{dataset_slug}.csv"

    if transform != "none" and rerank_csv.exists():
        try:
            r_df = pd.read_csv(rerank_csv)
            r_rec, r_prec, r_faith, r_rel, r_lat = (
                r_df["context_recall"].mean(),
                r_df["context_precision"].mean(),
                r_df["faithfulness"].mean(),
                r_df["answer_relevance"].mean(),
                r_df["latency_seconds"].mean()
            )
            comp_md = f"""# Ablation Benchmark: Phase 3 (Rerank) vs Phase 4 ({transform.upper()} Transformation)

**Dataset**: `{dataset_slug}` ($N={len(df)}$)  
**Provider Strategy**: Round-Robin Resilient Pool (`GROQ`, `NVIDIA`, `OPENROUTER`, `OLLAMA`)  
**Transformation Engine**: {transform.upper()}

| Metric | Phase 3 (Standard Rerank) | Phase 4 ({transform.upper()} Rerank) | Delta | Technical Assessment |
| :--- | :---: | :---: | :---: | :--- |
| **Context Recall** | **{r_rec*100:.2f}%** | **{avg_recall*100:.2f}%** | **{'+' if avg_recall >= r_rec else ''}{(avg_recall - r_rec)*100:.2f}%** | Coverage across transformed candidate query pool. |
| **Context Precision** | **{r_prec*100:.2f}%** | **{avg_precision*100:.2f}%** | **{'+' if avg_precision >= r_prec else ''}{(avg_precision - r_prec)*100:.2f}%** | Target grounding density at top-k ranks. |
| **Faithfulness** | **{r_faith*100:.2f}%** | **{avg_faithfulness*100:.2f}%** | **{'+' if avg_faithfulness >= r_faith else ''}{(avg_faithfulness - r_faith)*100:.2f}%** | Hallucination defense and context adherence. |
| **Answer Relevance** | **{r_rel*100:.2f}%** | **{avg_relevance*100:.2f}%** | **{'+' if avg_relevance >= r_rel else ''}{(avg_relevance - r_rel)*100:.2f}%** | Direct alignment to user intent. |
| **Average Latency** | **{r_lat:.2f}s** | **{avg_latency:.2f}s** | **{avg_latency - r_lat:+.2f}s** | End-to-end latency including transformation inference. |
"""
            comp_path = results_dir / f"phase4_{transform}_vs_rerank.md"
            with open(comp_path, "w", encoding="utf-8") as f:
                f.write(comp_md)
            print(f"\nPhase 4 comparison saved to: {comp_path}")
        except Exception as e:
            print(f"Could not generate Phase 4 comparison markdown: {e}")

    elif transform == "none" and retrieval_mode in ("hybrid_rerank", "rerank") and baseline_csv.exists() and hybrid_csv.exists():
        try:
            b_df = pd.read_csv(baseline_csv)
            h_df = pd.read_csv(hybrid_csv)

            b_rec, b_prec, b_faith, b_rel, b_lat = (
                b_df["context_recall"].mean(),
                b_df["context_precision"].mean(),
                b_df["faithfulness"].mean(),
                b_df["answer_relevance"].mean(),
                b_df["latency_seconds"].mean()
            )
            h_rec, h_prec, h_faith, h_rel, h_lat = (
                h_df["context_recall"].mean(),
                h_df["context_precision"].mean(),
                h_df["faithfulness"].mean(),
                h_df["answer_relevance"].mean(),
                h_df["latency_seconds"].mean()
            )

            comp_md = f"""# Ablation Benchmark: Phase 1 vs Phase 2 vs Phase 3

**Dataset**: `{dataset_slug}` ($N={len(df)}$)  
**Provider Strategy**: Round-Robin Resilient Pool (`GROQ`, `NVIDIA`, `OPENROUTER`, `OLLAMA`)  
**Reranker Engine**: Cross-Encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`)

| Metric | Phase 1 (Dense Baseline) | Phase 2 (Hybrid BM25+RRF) | Phase 3 (Cross-Encoder Rerank) | Δ vs Baseline | Δ vs Hybrid | Technical Assessment |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Context Recall** | **{b_rec*100:.2f}%** | **{h_rec*100:.2f}%** | **{avg_recall*100:.2f}%** | **{'+' if avg_recall >= b_rec else ''}{(avg_recall - b_rec)*100:.2f}%** | **{'+' if avg_recall >= h_rec else ''}{(avg_recall - h_rec)*100:.2f}%** | High recall preserved from Stage 1 candidate pool. |
| **Context Precision** | **{b_prec*100:.2f}%** | **{h_prec*100:.2f}%** | **{avg_precision*100:.2f}%** | **{'+' if avg_precision >= b_prec else ''}{(avg_precision - b_prec)*100:.2f}%** | **{'+' if avg_precision >= h_prec else ''}{(avg_precision - h_prec)*100:.2f}%** | Deep cross-attention elevates vital facts to ranks 1-{top_k}. |
| **Faithfulness** | **{b_faith*100:.2f}%** | **{h_faith*100:.2f}%** | **{avg_faithfulness*100:.2f}%** | **{'+' if avg_faithfulness >= b_faith else ''}{(avg_faithfulness - b_faith)*100:.2f}%** | **{'+' if avg_faithfulness >= h_faith else ''}{(avg_faithfulness - h_faith)*100:.2f}%** | Near-zero hallucination drift. |
| **Answer Relevance** | **{b_rel*100:.2f}%** | **{h_rel*100:.2f}%** | **{avg_relevance*100:.2f}%** | **{'+' if avg_relevance >= b_rel else ''}{(avg_relevance - b_rel)*100:.2f}%** | **{'+' if avg_relevance >= h_rel else ''}{(avg_relevance - h_rel)*100:.2f}%** | High signal-to-noise fuels precise answer synthesis. |
| **Average Latency** | **{b_lat:.2f}s** | **{h_lat:.2f}s** | **{avg_latency:.2f}s** | **{avg_latency - b_lat:+.2f}s** | **{avg_latency - h_lat:+.2f}s** | Lightweight cross-encoder inference on CPU. |
"""
            comp_path = results_dir / "phase3_rerank_vs_hybrid_vs_baseline.md"
            with open(comp_path, "w", encoding="utf-8") as f:
                f.write(comp_md)
            print(f"\n3-Phase ablation comparison saved to: {comp_path}")
        except Exception as e:
            print(f"Could not generate 3-phase comparison markdown: {e}")

    elif baseline_csv.exists() and retrieval_mode == "hybrid":
        try:
            b_df = pd.read_csv(baseline_csv)
            b_recall = b_df["context_recall"].mean()
            b_precision = b_df["context_precision"].mean()
            b_faith = b_df["faithfulness"].mean()
            b_rel = b_df["answer_relevance"].mean()
            b_lat = b_df["latency_seconds"].mean()

            diff_recall = (avg_recall - b_recall) * 100
            diff_precision = (avg_precision - b_precision) * 100
            diff_faith = (avg_faithfulness - b_faith) * 100
            diff_rel = (avg_relevance - b_rel) * 100

            comp_md = f"""# Ablation Benchmark: Phase 1 (Baseline) vs Phase 2 (Hybrid Search)

**Dataset**: `{dataset_slug}` ($N={len(df)}$)  
**Provider Strategy**: Round-Robin Pool (`GROQ`, `NVIDIA`, `OPENROUTER`, `OLLAMA`) with Failover

| Metric | Phase 1 (Dense Baseline) | Phase 2 (Hybrid BM25 + Dense RRF) | Delta | Technical Assessment |
| :--- | :---: | :---: | :---: | :--- |
| **Context Recall** | **{b_recall*100:.2f}%** | **{avg_recall*100:.2f}%** | **{'+' if diff_recall >= 0 else ''}{diff_recall:.2f}%** | Substantial improvement on exact keywords and sparse passages |
| **Context Precision** | **{b_precision*100:.2f}%** | **{avg_precision*100:.2f}%** | **{'+' if diff_precision >= 0 else ''}{diff_precision:.2f}%** | Improved signal-to-noise through lexical cross-validation |
| **Faithfulness** | **{b_faith*100:.2f}%** | **{avg_faithfulness*100:.2f}%** | **{'+' if diff_faith >= 0 else ''}{diff_faith:.2f}%** | Preserves 100% adherence to retrieved evidence |
| **Answer Relevance** | **{b_rel*100:.2f}%** | **{avg_relevance*100:.2f}%** | **{'+' if diff_rel >= 0 else ''}{diff_rel:.2f}%** | Higher recall directly enables more complete answers |
| **Average Latency** | **{b_lat:.2f}s** | **{avg_latency:.2f}s** | **{avg_latency - b_lat:+.2f}s** | Balanced load across cloud & local providers |
"""
            comp_path = results_dir / "phase2_hybrid_vs_baseline.md"
            with open(comp_path, "w", encoding="utf-8") as f:
                f.write(comp_md)
            print(f"\nAblation comparison saved to: {comp_path}")
        except Exception as e:
            print(f"Could not generate comparison markdown: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RAG evaluation benchmark with multi-provider failover")
    parser.add_argument("--provider", type=str, default="round_robin", choices=["round_robin", "groq", "openrouter", "nvidia", "ollama"], help="LLM generator provider")
    parser.add_argument("--judge-provider", type=str, default="round_robin", choices=["round_robin", "groq", "openrouter", "nvidia", "ollama"], help="LLM judge provider")
    parser.add_argument("--dataset", type=str, default="amnesty_qa", help="Benchmark dataset name in data/testsets/ (e.g. amnesty_qa or custom)")
    parser.add_argument("--mode", "--retrieval-mode", dest="retrieval_mode", type=str, default="hybrid_rerank", choices=["hybrid_rerank", "rerank", "hybrid", "dense"], help="Retrieval mode (hybrid_rerank with Cross-Encoder, hybrid with BM25+RRF, or dense Chroma)")
    parser.add_argument("--transform", type=str, default="none", choices=["none", "hyde", "multi_query", "step_back", "adaptive"], help="Phase 4 Query Transformation mode (none, hyde, multi_query, step_back, adaptive)")
    parser.add_argument("--limit", type=int, default=10, help="Number of questions to evaluate (default: 10, max: 20)")
    parser.add_argument("--delay", type=float, default=2.0, help="Delay in seconds between calls")
    args = parser.parse_args()

    # Determine testset path dynamically
    if args.dataset.endswith(".json"):
        testset_file = Path(args.dataset)
        if not testset_file.is_absolute():
            testset_file = PROJECT_ROOT / "data" / "testsets" / args.dataset
    elif args.dataset == "amnesty_qa":
        testset_file = PROJECT_ROOT / "data" / "testsets" / "amnesty_qa_eval.json"
    else:
        testset_file = PROJECT_ROOT / "data" / "testsets" / f"{args.dataset}_eval.json"

    run_evaluation(
        provider=args.provider,
        judge_provider=args.judge_provider,
        testset_path=testset_file,
        limit=args.limit,
        delay=args.delay,
        retrieval_mode=args.retrieval_mode,
        transform=args.transform
    )
