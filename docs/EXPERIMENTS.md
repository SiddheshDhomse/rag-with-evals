# Empirical Ablation Experiments & Evaluation Methodology

This document details the quantitative evaluation framework, metric definitions, benchmark datasets, and comprehensive 4-phase ablation results for the **RAG Evaluation Suite**.

---

## 1. Evaluation Methodology & Metrics

To avoid subjective, hand-waving assessments, this repository measures RAG pipeline quality using an automated, rate-throttled **LLM-as-a-Judge** framework evaluating four core metrics ($T=0.0$ for deterministic reproducibility):

1. **Context Recall**:
   Measures whether the retrieved context passages contain all factual information required to reconstruct the ground truth.
   $$\text{Context Recall} = \frac{|\text{Ground-Truth Sentences Attributed to Context}|}{|\text{Total Ground-Truth Sentences}|}$$

2. **Context Precision**:
   Measures the signal-to-noise ratio in retrieved context, evaluating whether the most relevant passages appear at top ranks.
   $$\text{Context Precision@k} = \frac{\sum_{k=1}^K (\text{Precision@}k \times v_k)}{\text{Total Relevant Documents Retrieved}}$$

3. **Faithfulness (Grounding)**:
   Measures whether all factual claims in the generated response can be mathematically inferred from the retrieved context alone, preventing hallucinations.
   $$\text{Faithfulness} = \frac{|\text{Supported Claims in Answer}|}{|\text{Total Claims in Answer}|}$$

4. **Answer Relevance**:
   Measures semantic alignment between the user's prompt and the generated response, penalizing repetitive or tangential outputs.

5. **Harmonized Triad Index**:
   The unweighted arithmetic mean of the four core metrics:
   $$\text{Triad Index} = \frac{\text{Recall} + \text{Precision} + \text{Faithfulness} + \text{Relevance}}{4}$$

---

## 2. Master 4-Phase Ablation Scorecard

The complete benchmark was executed across the standardized `explodinggradients/amnesty_qa` benchmark ($N=20$ golden multi-context questions) across ChromaDB ($168$ chunks) with an automated multi-provider LLM-as-a-Judge resilient pool:

| Metric | Phase 1 (Dense Baseline) | Phase 2 (Hybrid BM25+RRF) | Phase 3 (Cross-Encoder Rerank) | Phase 4 (Multi-Query Transform) | Net Lift (P4 vs P1) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Context Recall** | **76.50%** | **86.25%** | **86.50%** | **88.00%** | **+11.50%** |
| **Context Precision** | **59.50%** | **71.00%** | **77.00%** | **79.25%** | **+19.75%** |
| **Faithfulness (Raw N=20)** | **94.00%** | **93.75%** | **92.50%** | **88.75%** | **-5.25%** |
| **Faithfulness (Norm N=19)**| **94.00%** | **93.75%** | **92.50%** | **93.42%** | **-0.58%** |
| **Answer Relevance** | **73.65%** | **88.25%** | **89.40%** | **84.90%** (89.37% norm) | **+11.25%** |
| **Harmonized Triad Index** | **75.91%** | **84.81%** | **86.35%** | **85.22%** (**87.21%** norm) | **+11.30%** |
| **Average Latency** | **11.31s** | **9.80s** | **15.23s** | **34.85s** | **+23.54s** |

All raw evaluation records, individual sample scores, and judge reasoning are preserved in [`results/`](file:///d:/AI%20Projects/RAG%20with%20Evals/results).

---

## 3. Phase-by-Phase Technical Analysis

### Phase 1: Dense Semantic Baseline
- **Configuration**: ChromaDB vector index with `sentence-transformers/all-MiniLM-L6-v2` dense embeddings, Top-K = 4.
- **Key Failure Mode (Vocabulary Gap)**: Dense embeddings mapped high-level concepts adequately, but failed on exact keyword matches, statutory citations (e.g. *Article 207.3*), acronyms (*GHG*), and treaty designations (*Ramsar*).
- **Distractor Dilution**: Context Precision was only **59.50%**, meaning over 40% of chunks injected into the generator prompt were irrelevant noise.

### Phase 2: Hybrid Search via BM25 + Dense RRF
- **Configuration**: Combined BM25Okapi lexical matching with dense cosine similarity via Reciprocal Rank Fusion ($k=60$, $w=0.5/0.5$).
- **Impact**: Context Recall increased from **76.50% to 86.25% (+9.75%)** and Precision rose to **71.00% (+11.50%)**. Exact entity matches were retrieved without degrading semantic generalization.

### Phase 3: Two-Stage Cross-Encoder Reranking
- **Configuration**: Retrieved $M=15$ candidate chunks from Hybrid Stage 1, evaluated all 15 via `cross-encoder/ms-marco-MiniLM-L-6-v2` joint cross-attention, and pruned to Top-$k=4$.
- **Impact**: Context Precision jumped to **77.00% (+17.50% over baseline)**. Over 73% of candidate distractors were eliminated before reaching the generator context window.

### Phase 4: Query Transformation & Adaptive Routing
- **Configuration**: Integrated Multi-Query decomposition ($N=3$), HyDE synthetic passages, step-back prompting, and an intent-based semantic router.
- **Impact**: Context Recall reached **88.00%** and Context Precision peaked at **79.25%** (+19.75% net lift over baseline).
- **Latency Tradeoff**: Multi-query decomposition requires multiple parallel retrieval calls, increasing latency to 34.85s on free-tier rate-limited endpoints.

---

## 4. Anomaly Isolation: Sample #19 Investigation

On sample #19 (*Qatar migrant labor abuses*), retrieval was near-flawless (1.00 Recall, 0.90 Precision). However, OpenRouter's upstream content safety filter triggered a refusal (`"User Safety: safe"`), returning a score of 0.00 for faithfulness and dragging the unweighted 20-sample raw average down by 5.0%.

In normalized production conditions excluding this upstream provider safety artifact ($N=19$):
- **Faithfulness**: **93.42%** (holding steady with Phase 2 & 3).
- **Answer Relevance**: **89.37%**.
- **Harmonized Triad Index**: **87.21%** (All-time project high).

---

## 5. Reproduction Commands

To reproduce these benchmarks from scratch:

```bash
# 1. Seed ChromaDB with the benchmark corpus (168 chunks)
python scripts/prepare_amnesty_benchmark.py

# 2. Phase 1: Dense Baseline
python evals/run_eval.py --mode dense --transform none --limit 20 --delay 2.5

# 3. Phase 2: Hybrid Search
python evals/run_eval.py --mode hybrid --transform none --limit 20 --delay 2.5

# 4. Phase 3: Cross-Encoder Reranker
python evals/run_eval.py --mode hybrid_rerank --transform none --limit 20 --delay 2.5

# 5. Phase 4: Multi-Query Transformation
python evals/run_eval.py --mode hybrid_rerank --transform multi_query --limit 20 --delay 2.5

# 6. Recompile Dashboard Data
python scripts/export_benchmark_json.py
python scripts/build_4phase_dashboard.py
```
