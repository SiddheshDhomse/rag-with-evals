# 🚀 Engineering a Production RAG System: An Empirical Ablation Study

> **Repository Benchmark Data & LinkedIn Case Study Draft**  
> **Track:** Retrieval-Augmented Generation (RAG) & LLM Evaluations  
> **Benchmark Dataset:** `explodinggradients/amnesty_qa` (Multi-Context Evaluation Set)  
> **Corpus Size:** 168 chunks across 23 distinct documents in ChromaDB  
> **Judge Evaluator:** LLM-as-a-Judge (`qwen/qwen3.8-27b` on Groq) + Ragas Framework  

---

## 📊 1. Master Ablation Benchmark Scorecard

| Milestone / Pipeline Stage | Context Recall | Context Precision | Faithfulness | Answer Relevance | Avg Latency | Notes |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Stage 0: Naive Baseline**<br>*(ChromaDB + all-MiniLM-L6-v2 + Groq, k=4)* | **46.25%** | **57.50%** | **100.00%** | **80.00%** | **10.35s** | Established empirical baseline. High hallucination resistance, but struggles with multi-hop context and keyword queries. |
| **Stage 1: + Hybrid Search**<br>*(BM25 + Dense RRF)* | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* | Target: Fixes exact keyword misses (boost Recall to >75%). |
| **Stage 2: + Cross-Encoder Reranker**<br>*(FlashRank / BGE)* | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* | Target: Prunes distractor noise (boost Precision to >85%). |
| **Stage 3: + Query Expansion**<br>*(HyDE / Multi-Query)* | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* | Target: Resolves complex multi-hop questions. |

---

## 🔍 2. Deep Dive: Why Naive RAG Broke Down (Empirical Failure Modes)

A naive RAG pipeline with pure cosine similarity search revealed critical failure modes when tested against multi-context queries:

### Case Study 1: The Keyword Mismatch Failure
- **Question**: *"Which companies are the main contributors to GHG emissions and their role in global warming according to the Carbon Majors database?"*
- **Ground Truth**: Identified fossil fuel producers from the Carbon Majors database.
- **What Happened**:
  - **Context Recall: 0.00% \| Context Precision: 0.00%**
  - **Root Cause**: Dense vector embeddings map semantic concepts well, but failed to prioritize the exact keyword entity *"Carbon Majors database"*. The retriever pulled general prose about global warming rather than the specific tabular report chunk.
- **Fix**: **BM25 Hybrid Search** with Reciprocal Rank Fusion (RRF).

### Case Study 2: The Multi-Hop / Distributed Facts Failure
- **Question**: *"What are the global implications of the USA Supreme Court ruling on abortion?"*
- **Ground Truth**: Requires synthesizing consequences across 3 distinct paragraphs in the report.
- **What Happened**:
  - **Context Recall: 25.00% \| Context Precision: 50.00%**
  - **Root Cause**: The dense retriever only grabbed 1 relevant chunk, filling the remaining 3 slots with general legal commentary.
- **Fix**: **Query Expansion / HyDE** + **Reranking larger candidate pool (Top-15 \(\to\) Top-4)**.

### Case Study 3: Distractor Noise
- **Overall Context Precision was 57.50%**.
- In almost every query, 2 out of the 4 retrieved chunks passed to the LLM were background noise. While the LLM maintained **100% Faithfulness** by ignoring the noise, this wastes tokens and increases latency.

---

## ✍️ 3. Ready-to-Publish LinkedIn Post Draft (Copy & Paste)

```markdown
🚨 Why Naive RAG Fails in Production (And the Data to Prove It) 📊

Most RAG tutorials show 100% accuracy on 3 toy questions. But what happens when you test your pipeline against a realistic, multi-context benchmark?

I built a production RAG system with LangChain, ChromaDB, and Groq/NVIDIA/Ollama, and ran an empirical evaluation using Ragas and LLM-as-a-Judge across 168 indexed chunks.

Here is what the baseline numbers actually look like:

📉 The Baseline Scorecard:
• Context Recall: 46.25% (Missed more than half the required facts!)
• Context Precision: 57.50% (Almost half of retrieved chunks were irrelevant noise)
• Faithfulness: 100.00% (Strict zero hallucinations)
• Answer Relevance: 80.00%

Why did it fail?
1️⃣ Keyword Mismatch: Dense vector embeddings favored general prose and completely missed specific entities like "Carbon Majors database" (0% recall on that query!).
2️⃣ Multi-hop Splitting: When an answer spans 3 non-adjacent paragraphs, top-k vector similarity only grabs 1 and fills the rest with distractors.

Over the next few days, I am running a systematic ablation study to measure the exact impact of:
👉 Stage 1: BM25 + Dense Hybrid Search (RRF)
👉 Stage 2: Cross-Encoder Reranker (FlashRank)
👉 Stage 3: Query Expansion & HyDE

Check out the full open-source repo and raw CSV benchmark logs here: [GitHub Link]

How do you tackle retrieval noise in your production pipelines? Let's discuss in the comments! 👇

#AI #MachineLearning #RAG #LangChain #ChromaDB #LLMs #NLP #GenerativeAI #Evaluation
```

---

## 💼 4. Resume Bullet Points

- **Engineered an Enterprise-Grade Conversational RAG Architecture**: Designed a modular, production-deployable RAG pipeline using LangChain, persistent ChromaDB, and multi-provider LLM support (Groq, NVIDIA NIM, and Ollama) with multi-session memory and token streaming.
- **Conducted Systematic Retrieval Ablation Studies**: Benchmarked naive dense retrieval against multi-context datasets, identifying empirical bottlenecks (46.2% Context Recall, 57.5% Context Precision due to entity mismatch and multi-hop distribution).
- **Automated LLM Evaluation Pipelines**: Built an evaluation suite using Ragas and LLM-as-a-Judge metrics to track Context Precision, Recall, Faithfulness, and Latency across iterative pipeline enhancements.
