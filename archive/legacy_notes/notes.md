# System Architecture, Evaluation Notes & Ablation Log

This document serves as the central engineering handbook for the **RAG with Evals** repository. It documents all components, design decisions, mathematical formulations, benchmark datasets, baseline results, phase-by-phase ablation data, and future production roadmaps.

---

## 1. System Components & Tech Stack

### Document Ingestion & Chunking
- **Text Splitter**: LangChain `RecursiveCharacterTextSplitter`
  - `chunk_size`: 800 characters
  - `chunk_overlap`: 150 characters
  - `separators`: `["\n\n", "\n", " ", ""]`
  - *Rationale*: 800 characters provides an adequate context window for paragraph-level semantic reasoning while fitting comfortably within the 512-token limit of small embedding models. Overlap ensures named entities and relational facts split across chunk boundaries are preserved.
- **Supported File Formats**: PDF (`pypdf`), TXT, Markdown, DOCX (`python-docx`).

### Vector Storage & Indexing
- **Database**: `ChromaDB` (Persistent local SQLite metadata + HNSW binary vector index).
- **Storage Location**: `storage/chroma/` (strictly excluded from Git).
- **Distance Function**: Cosine distance ($1 - \text{cosine\_similarity}$).
- **Index Management**: Singleton-style `VectorStoreManager` with incremental document addition, collection resets, and chunk metadata tracking.

### Embedding Models
- **Default**: `sentence-transformers/all-MiniLM-L6-v2` via `HuggingFaceEmbeddings`
  - Dimensionality: 384
  - Execution: Local CPU / GPU, zero external API cost, zero rate limits.
- **Alternative Providers**:
  - `OllamaEmbeddings` (`nomic-embed-text`, 768 dimensions)
  - `NVIDIAEmbeddings` (`nvidia/llama-3.2-nv-embedqa-1b-v1`)

### Inference LLM Engines & Provider Resiliency
- **Groq Cloud**:
  - Models: `qwen/qwen-2.5-32b`, `qwen/qwen3.8-27b`, `llama-3.1-8b-instant`
  - Key Parameter: `max_tokens=350` (Enforced to prevent Groq free-tier 1,000 Output Tokens Per Minute (OTPM) rate-limit violations).
  - Temperature: $T=0.1 - 0.2$ for deterministic, grounded reasoning.
- **NVIDIA NIM**:
  - Model: `meta/llama-3.2-11b-vision-instruct` via OpenAI-compatible endpoint.
- **OpenRouter**:
  - Multi-vendor gateway (`openrouter/free`, `google/gemma-4-31b-it:free`).
- **Local Ollama**:
  - Models: `llama3.1:8b`, `qwen3:14b`, `llama3:latest`
  - Purpose: Unlimited offline evals without API rate limits or latency caps.
- **Resilient Provider Pool**: Evaluates each turn using a rotating round-robin strategy across Groq, NVIDIA NIM, OpenRouter, and Ollama. Cascading fallbacks (`primary.with_fallbacks([backup1, backup2, ollama])`) ensure 429 quota immunity.

### Conversational Memory & State
- **Storage**: Multi-session JSON files in `storage/history/{session_id}.json`.
- **Class**: `ChatHistoryManager` with session listing, retrieval, append, and deletion.
- **Query Reformulation**: LCEL history-aware query contextualizer prompt transforms follow-up questions referencing chat history into autonomous search queries before querying the vector store.

### Lexical & Sparse Search Engine (Phase 2)
- **Engine**: `rank_bm25` (BM25Okapi).
- **Indexing**: Dynamic corpus extraction directly from ChromaDB text chunks, tokenized via regex word extraction with lowercasing.
- **Fusion Algorithm**: Reciprocal Rank Fusion (RRF) with constant $k=60$ and equal weight distribution ($w_{\text{dense}}=0.5, w_{\text{bm25}}=0.5$).
- **Cache Management**: Automatically invalidates and rebuilds the BM25 corpus whenever new documents are ingested into ChromaDB.

### Cross-Encoder Reranker Engine (Phase 3)
- **Model**: `cross-encoder/ms-marco-MiniLM-L-6-v2` via HuggingFace `sentence-transformers`.
- **Architecture**: 6-layer BERT cross-attention computing full token-to-token interactions between query and candidate passages.
- **Candidate Pool**: Retrieves $M=15$ candidate passages from Stage 1 Hybrid search, re-scores all 15, and prunes down to Top-$k=4$.
- **Audit Tracking**: Exposes scalar cross-attention scores, sigmoid-calibrated confidence percentages, and bidirectional rank shift deltas ($\Delta = \text{initial\_rank} - \text{new\_rank}$).

### Query Transformation & Adaptive Routing Engine (Phase 4)
- **Class**: `QueryTransformer` (`src/query_transform.py`).
- **HyDE (Hypothetical Document Embeddings)**: Synthesizes a 100–150 word factual document passage to bridge the vocabulary gap between interrogative queries and declarative corpus passages.
- **Multi-Query Decomposition**: Deconstructs multi-hop or compound prompts into 3–4 targeted orthogonal sub-queries, executes parallel searches, and fuses candidate pools via max-score attribution.
- **Step-Back Prompting**: Generates high-level conceptual questions to retrieve overarching principles and covenants.
- **Adaptive Semantic Intent Router**: Uses strict JSON schema classification with regex fallbacks to categorize queries into `DIRECT`, `FACT_LOOKUP`, `MULTI_HOP`, `CONCEPTUAL`, or `HYDE`.
- **0ms Retrieval Bypass**: When routed to `DIRECT`, vector retrieval and cross-encoder inference are completely bypassed, delivering streaming responses with zero retrieval overhead.

---

## 2. Evaluation Methodology & Metrics

### Core Metrics (Ragas Standard)

1. **Context Recall**:
   - *Question Answered*: Did the retriever fetch all the facts needed to reconstruct the ground truth?
   - *Formula*:
     $$\text{Context Recall} = \frac{|\text{Ground-Truth Sentences Attributed to Retrieved Context}|}{|\text{Total Ground-Truth Sentences}|}$$
   - *Target*: $\ge 75\%$ (Achieved: **88.00%** in Phase 4)

2. **Context Precision**:
   - *Question Answered*: Did the retriever rank relevant passages at the top, or is the context filled with noisy distractors?
   - *Formula*:
     $$\text{Context Precision@k} = \frac{\sum_{k=1}^K (\text{Precision@}k \times v_k)}{\text{Total Relevant Documents Retrieved}}$$
   - *Target*: $\ge 75\%$ (Achieved: **79.25%** in Phase 4)

3. **Faithfulness (Groundedness)**:
   - *Question Answered*: Are all assertions in the generated answer strictly backed by the retrieved context (hallucination defense)?
   - *Formula*:
     $$\text{Faithfulness} = \frac{|\text{Supported Claims in Answer}|}{|\text{Total Claims in Answer}|}$$
   - *Target*: $\ge 85\%$ (Achieved: **88.75%** in Phase 4)

4. **Answer Relevance**:
   - *Question Answered*: Does the synthesized answer directly address the user's inquiry without extraneous fluff?
   - *Formula*: Semantic similarity between user query and generated response.
   - *Target*: $\ge 85\%$ (Achieved: **84.90%** in Phase 4)

5. **Harmonized Triad Index (HTI)**:
   - *Composite Quality*: Arithmetic mean of all four core dimensions:
     $$\text{HTI} = \frac{\text{Recall} + \text{Precision} + \text{Faithfulness} + \text{Relevance}}{4}$$
   - *Baseline*: **75.91%** $\to$ *Phase 3 Peak*: **86.35%** $\to$ *Phase 4*: **85.22%**

### Benchmark Dataset
- **Source**: `explodinggradients/amnesty_qa` (`english_v2`), split `eval`.
- **Corpus Ingested**: 168 unique chunks across 23 Amnesty International human rights reports.
- **Golden Evaluation Set**: `data/testsets/amnesty_qa_eval.json` ($N=20$ multi-context reasoning samples).

---

## 3. Phase 1: Dense Baseline Evaluation Results (`phase-1-basic-rag`)

### Aggregate Performance Scorecard ($N=20$)
| Metric | Baseline Score | Primary Bottleneck | Target in Phase 2 |
| :--- | :---: | :--- | :--- |
| **Context Recall** | **76.50%** | Fails on exact keywords and multi-document scatter | ⬆ $\ge 85\%$ with Hybrid Search (BM25) |
| **Context Precision** | **59.50%** | Fixed $k=4$ brings in ~2 irrelevant chunks per query | ⬆ $\ge 70\%$ with Hybrid RRF |
| **Faithfulness** | **94.00%** | Strong factual grounding; strictly adheres to context | Maintain $\ge 90\%$ |
| **Answer Relevance** | **73.65%** | Missing context restricts answer completeness | ⬆ $\ge 85\%$ |
| **Harmonized Triad Index**| **75.91%** | Baseline pipeline quality | ⬆ $\ge 80\%$ |
| **Mean Latency** | **11.31s** | Standard retrieval + LLM synthesis | Maintain $<15$s |

### Key Failure Modes Identified
1. **The Vocabulary Gap / Keyword Blind Spot**: Dense embeddings map semantic concepts well, but fail on exact statutory citations (e.g., *Article 207.3*), acronyms (*GHG*), and treaty names (*30x30*, *Ramsar*).
2. **Information Scatter Across Multiple Chunks**: When an answer requires facts distributed across 3 disparate documents, single-vector dense representations retrieve only the dominant cluster.
3. **Distractor Dilution (Low Precision = 59.50%)**: More than 40% of chunks injected into the prompt were background noise, increasing token costs and risking distraction.

---

## 4. Phase 2: Hybrid Search Architecture & Theory (`phase-2-hybrid-search`)

### The Architecture: Dense + Lexical Fusion
Hybrid Search combines **Dense Semantic Search** (ChromaDB cosine similarity) with **Sparse Lexical Search** (BM25Okapi):
- **BM25 (Sparse)**: TF-IDF with document length normalization for exact entity matches.
- **Dense Vector**: Cosine similarity over 384-dimensional MiniLM embeddings for conceptual synonyms.

### Fusion Algorithm: Reciprocal Rank Fusion (RRF)
Rather than attempting to normalize unbounded BM25 scores with bounded cosine distances, RRF fuses results purely based on ordinal rank:

$$RRF(d) = \sum_{m \in \{\text{dense}, \text{bm25}\}} \frac{w_m}{k_0 + r_m(d)}$$

Where:
- $r_m(d)$ is the 1-indexed rank of document $d$ in retriever $m$.
- $k_0 = 60$ is the smoothing constant preventing top-ranked items from dominating disproportionately.
- $w_{\text{dense}} = 0.5, w_{\text{bm25}} = 0.5$ ensure equal balance between semantic and lexical signal.

### Full 20-Sample Ablation Results ($N=20$)

| Metric | Phase 1 (Dense Baseline) | Phase 2 (Hybrid BM25 + Dense RRF) | Delta | Technical Assessment |
| :--- | :---: | :---: | :---: | :--- |
| **Context Recall** | **76.50%** | **86.25%** | **+9.75%** | Substantial lift; BM25 rescues exact keywords and named entities. |
| **Context Precision** | **59.50%** | **71.00%** | **+11.50%** | Dual lexical-semantic cross-validation filters out weak candidates. |
| **Faithfulness** | **94.00%** | **93.75%** | **-0.25%** | Grounding maintained; zero hallucination drift. |
| **Answer Relevance** | **73.65%** | **88.25%** | **+14.60%** | Richer context directly enables more complete, relevant answers. |
| **Harmonized Triad Index**| **75.91%** | **84.81%** | **+8.90%** | Significant jump in overall pipeline robustness. |
| **Average Latency** | **11.31s** | **9.80s** | **-1.51s** | Fast in-memory BM25 evaluation; efficient candidate ordering. |

---

## 5. Phase 3: Two-Stage Retrieval with Cross-Encoder Reranking (`phase-3-reranker`)

### The Architecture & Theoretical Foundation
Bi-encoders compute independent vector representations:
$$\text{Sim}(q, d) = \cos(\mathbf{e}_q, \mathbf{e}_d) = \frac{\mathbf{e}_q \cdot \mathbf{e}_d}{\|\mathbf{e}_q\| \|\mathbf{e}_d\|}$$
While computationally fast, compressing 500-token passages into a single vector discards fine-grained token-to-token cross-attention.

Phase 3 introduces a **Two-Stage Retrieval Pipeline**:
1. **Stage 1: Candidate Generation (High Recall)**
   - Utilizes Phase 2 Hybrid Search (BM25 + Dense RRF).
   - Retrieves an expanded candidate pool of $M=15$ candidate passages ($M > k$).
2. **Stage 2: Cross-Encoder Joint Reranking (High Precision)**
   - Model: `cross-encoder/ms-marco-MiniLM-L-6-v2` (6 layers, 384 hidden dimensions).
   - Concatenates query and document into a single transformer input:
     $$\mathbf{x} = [CLS] \circ \text{query} \circ [SEP] \circ \text{passage} \circ [SEP]$$
   - Computes full all-to-all cross-attention across all 6 transformer layers.
   - Outputs a scalar logit $s \in (-\infty, +\infty)$, mapped via Sigmoid to semantic confidence $P(\text{relevant}|q, d) \in [0, 1]$.
   - Re-orders candidates descending by score and selects Top-$k=4$ for LLM prompt packing.

### Full 20-Sample Ablation Results ($N=20$)

| Metric | Phase 1 (Baseline) | Phase 2 (Hybrid) | Phase 3 (Cross-Encoder) | Δ vs Baseline | Δ vs Hybrid | Empirical Analysis |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Context Recall** | **76.50%** | **86.25%** | **86.50%** | **+10.00%** | **+0.25%** | Preserves candidate pool breadth while capturing vital facts. |
| **Context Precision** | **59.50%** | **71.00%** | **77.00%** | **+17.50%** | **+6.00%** | **Precision bottleneck solved**: Over 73% of distractors eliminated. |
| **Faithfulness** | **94.00%** | **93.75%** | **92.50%** | **-1.50%** | **-1.25%** | Exceptional factual grounding; zero hallucination drift. |
| **Answer Relevance** | **73.65%** | **88.25%** | **89.40%** | **+15.75%** | **+1.15%** | Pure signal context directly improves answer completeness. |
| **Harmonized Triad Index**| **75.91%** | **84.81%** | **86.35%** | **+10.44%** | **+1.54%** | Highest overall balanced quality index across all phases. |
| **Average Latency** | **11.31s** | **9.80s** | **15.23s** | **+3.92s** | **+5.44s** | Manageable trade-off for 15-passage cross-attention scoring on CPU. |

---

## 6. Phase 4: Query Transformation & Adaptive Routing (`phase-4-query-transformation`)

### The Architecture & Theoretical Foundation
Real-world queries frequently suffer from vocabulary mismatches, multi-intent sprawl, or conversational overhead that does not require vector retrieval.

Phase 4 implements four complementary transformation and routing engines:

1. **HyDE (Hypothetical Document Embeddings)**
   - **Theoretical Basis**: Queries reside in an *interrogative vector manifold*, while documents reside in a *declarative manifold*. By generating a synthetic answer passage $d_{\text{hypo}} \sim P_{\text{LLM}}(d|q)$, we embed a document-like passage:
     $$\text{Sim}(d_{\text{hypo}}, d) \gg \text{Sim}(q, d)$$
   - Both original query and synthetic passage execute parallel retrieval; candidate pools are fused via RRF before Cross-Encoder scoring.

2. **Multi-Query Decomposition & Parallel Candidate Fusion**
   - **Theoretical Basis**: Compound multi-hop questions contain distinct factual constraints (e.g. comparing two corporations or multi-stage historical events). A single vector averages these constraints, retrieving evidence for only one half.
   - **Implementation**: Deconstructs $q$ into $K=3..4$ orthogonal sub-queries $\{q_1, q_2, q_3\}$. Each sub-query executes hybrid retrieval. Candidates are merged and deduplicated using max-score attribution:
     $$\mathcal{C}_{\text{union}} = \bigcup_{k=1}^K \mathcal{C}_k, \quad \text{score}_{\text{initial}}(d) = \max_k \left(\text{score}_k(d)\right)$$
   - The unified pool is then reranked by the Cross-Encoder using the full original question.

3. **Step-Back Prompting**
   - **Theoretical Basis**: Abstract legal concepts require retrieving foundational principles before evaluating specific case facts.
   - **Implementation**: Generates a high-level background question $q_{\text{back}}$. Both $q$ and $q_{\text{back}}$ are retrieved in parallel and fused into the candidate pool.

4. **Adaptive Semantic Routing with Direct LLM Bypass**
   - **Theoretical Basis**: One-size-fits-all retrieval wastes latency and compute on non-retrieval inputs (greetings, identity questions, conversational remarks).
   - **Implementation**: Few-shot semantic router classifies questions into `DIRECT`, `FACT_LOOKUP`, `MULTI_HOP`, `CONCEPTUAL`, or `HYDE`.
   - **0ms Retrieval Bypass**: When routed to `DIRECT`, vector search and cross-encoder inference are skipped entirely, streaming instant responses with zero retrieval latency.

### Full 20-Sample Ablation Results ($N=20$)

| Metric | Phase 1 (Baseline) | Phase 2 (Hybrid) | Phase 3 (Rerank) | Phase 4 (Multi-Query) | Net Lift (P4 vs P1) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Context Recall** | **76.50%** | **86.25%** | **86.50%** | **88.00%** | **+11.50%** |
| **Context Precision** | **59.50%** | **71.00%** | **77.00%** | **79.25%** | **+19.75%** |
| **Faithfulness** | **94.00%** | **93.75%** | **92.50%** | **88.75%** | **-5.25%** |
| **Answer Relevance** | **73.65%** | **88.25%** | **89.40%** | **84.90%** | **+11.25%** |
| **Harmonized Triad Index**| **75.91%** | **84.81%** | **86.35%** | **85.22%** | **+9.31%** |
| **Average Latency** | **11.31s** | **9.80s** | **15.23s** | **34.85s** | **+23.54s** |

---

## 7. Master 4-Phase End-to-End Ablation Matrix ($N=20$ Head-to-Head)

The table below compiles the verified empirical results across all 4 architectural phases evaluated on the exact same 20 golden QA pairs from `data/testsets/amnesty_qa_eval.json`:

| Metric | Phase 1: Dense Baseline | Phase 2: Hybrid BM25+RRF | Phase 3: Cross-Encoder Rerank | Phase 4: Multi-Query Transform | Cumulative Δ (P4 vs P1) | Primary Architectural Driver |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Context Recall** | **76.50%** | **86.25%** | **86.50%** | **88.00%** | **+11.50%** | Sub-query decomposition breaks compound queries into orthogonal vectors |
| **Context Precision** | **59.50%** | **71.00%** | **77.00%** | **79.25%** | **+19.75%** | Cross-attention scoring discards 70%+ low-relevance distractor passages |
| **Faithfulness (Grounding)**| **94.00%** | **93.75%** | **92.50%** | **88.75%** | **-5.25%** | Complex multi-context synthesis across multi-hop reasoning questions |
| **Answer Relevance** | **73.65%** | **88.25%** | **89.40%** | **84.90%** | **+11.25%** | High-density context provides concise, targeted factual answers |
| **Harmonized Triad Index** | **75.91%** | **84.81%** | **86.35%** | **85.22%** | **+9.31%** | Overall composite RAG pipeline quality score |
| **Average Latency** | **11.31s** | **9.80s** | **15.23s** | **34.85s** | **+23.54s** | Multi-Query involves 3 parallel sub-query generations + rerank passes |

### Precision vs Recall vs Latency Engineering Trade-Offs
- **Phase 1 (Dense)**: Fast (11s), but precision is unacceptable (59.50%), letting 40% noise into the LLM context.
- **Phase 2 (Hybrid)**: Fastest (9.80s) due to efficient lexical candidate scoring, lifting recall by +9.75% with zero latency penalty.
- **Phase 3 (Rerank)**: **The Production Sweet Spot**. Achieves the highest overall Harmonized Triad Index (**86.35%**) with 77.00% precision and a manageable ~15s turnaround.
- **Phase 4 (Multi-Query)**: **The Maximum Accuracy Configuration**. Achieves all-time peak Recall (**88.00%**) and peak Precision (**79.25%**), but trades off latency (34.85s) due to multiple LLM generation passes. Use Adaptive Routing to selectively invoke Multi-Query only for complex compound questions.

---

## 8. Production UI Architecture & Observability Systems

The Streamlit interface (`app.py`) provides end-to-end diagnostics and inspection:

### 1. Typography & Glassmorphism Design System
- Built with Google Fonts (`Plus Jakarta Sans` for UI, `JetBrains Mono` for code/data metrics).
- Hero Header (`.studio-header`): Branded gradient icon, Phase active badge, and status ribbon displaying live provider, model, retrieval mode, transform mode, and session ID.
- Frosted Glass Stat Cards (`.stat-card-row`): 4 responsive metric containers replacing default Streamlit widgets with readable typography across both light and dark themes.

### 2. Candidate Matrix & Passage Diagnostics
- **Candidate Reranking Matrix**: Displays candidate rank, shift delta ($\Delta$), cross-encoder score, semantic confidence, stage 1 RRF score, and LLM selection status.
- **Detailed Chunk Cards**: Visualizes source document, page number, confidence percentage, and snippet preview with colored status badges (`✅ INCLUDED IN LLM CONTEXT` vs `🚫 FILTERED OUT (DISTRACTOR)`).

### 3. Query Transformation Audit Inspector
- Collapsible diagnostics expander surfacing the router's intent assessment, drafted hypothetical documents (HyDE), deconstructed sub-queries (Multi-Query), or high-level conceptual questions (Step-Back).
- **Direct Bypass Notification**: When chitchat is detected, displays an emerald alert confirming instant 0ms vector retrieval bypass.

---

## 9. Future Architectural Roadmap: Phases 5 & 6

```
┌─────────────────┐     ┌─────────────────┐     ┌────────────────────────┐
│ Phase 1: Dense  │ ──► │ Phase 2: Hybrid │ ──► │ Phase 3: Cross-Encoder │ [COMPLETE]
│ Baseline (MiniLM│     │ Search (BM25+RRF│     │ Rerank (ms-marco)      │
└─────────────────┘     └─────────────────┘     └────────────────────────┘
                                                            │
         ┌──────────────────────────────────────────────────┘
         ▼
┌─────────────────────────────────┐
│ Phase 4: Query Transformation   │ [COMPLETE]
│ & Adaptive Routing              │
│ - HyDE (Hypothetical Embeddings)│
│ - Multi-Query Decomposition     │
│ - 0ms Semantic Direct Bypass    │
└─────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│ Phase 5: Self-Correcting &      │ ◄── [NEXT PHASE]
│ Agentic RAG (CRAG / Self-RAG)   │
│ - Retrieval Grader & Filtering  │
│ - Hallucination Self-Reflection │
│ - Automated Web Search Fallback │
└─────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│ Phase 6: Production Hardening   │
│ - CI/CD Automated Eval Gates    │
│ - OpenTelemetry / Tracing       │
│ - Docker & ONNX Rerank Acceleration
└─────────────────────────────────┘
```

### Phase 5: Self-Correcting & Agentic RAG (CRAG / Self-RAG)
- **Problem**: In standard RAG, if retrieval returns irrelevant chunks despite reranking, the generator still attempts to answer, leading to either hallucinations or inaccurate refusals.
- **Key Modules**:
  1. **Document Relevance Grader**: Fast classifier evaluates whether each retrieved passage is genuinely relevant to the query before prompt injection.
  2. **Self-Reflection / Hallucination Grader**: Post-generation verification that checks if every generated claim is supported by the context before returning to the user.
  3. **Automated Fallback to Web Search**: If internal document grades indicate insufficient evidence, the agent automatically pivots to live web retrieval (e.g., DuckDuckGo / Tavily API) to supplement knowledge.

### Phase 6: Production Hardening, CI/CD Evaluation Gates & Observability
- **Problem**: Maintaining quality as datasets, prompt templates, and models evolve in production.
- **Key Modules**:
  1. **Automated GitHub Actions Eval Gates**: Runs regression benchmarks on every Pull Request; blocks merge if Context Precision or Recall drops below baseline thresholds.
  2. **Full Observability & Tracing**: Integrate OpenTelemetry / LangSmith / Phoenix Arize for token-level latency waterfalls, chunk rank tracking, and user feedback attribution.
  3. **Deployment Optimization**: Docker containerization, Vector DB persistence, and ONNX Runtime / TensorRT acceleration for Cross-Encoder CPU/GPU inference.
