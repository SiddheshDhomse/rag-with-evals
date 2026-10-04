# Production RAG with Quantitative Evaluation Harness

[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-3776AB.svg?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-v0.3-1C3C3C.svg?style=flat-square)](https://python.langchain.com/)
[![Ragas](https://img.shields.io/badge/Framework-Ragas-000000.svg?style=flat-square)](https://docs.ragas.io/)
[![ChromaDB](https://img.shields.io/badge/VectorDB-ChromaDB-FC521F.svg?style=flat-square)](https://www.trychroma.com/)
[![Groq Cloud](https://img.shields.io/badge/Inference-Groq_Cloud-F55036.svg?style=flat-square)](https://groq.com/)
[![NVIDIA NIM](https://img.shields.io/badge/Inference-NVIDIA_NIM-76B900.svg?style=flat-square)](https://build.nvidia.com/)
[![Ollama](https://img.shields.io/badge/Local_LLM-Ollama-000000.svg?style=flat-square)](https://ollama.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square)](LICENSE)

A modular, production-grade Retrieval-Augmented Generation (RAG) system with persistent vector indexing, multi-turn conversational memory, multi-provider inference routing (Groq, NVIDIA NIM, OpenRouter, Ollama), and an automated evaluation harness powered by Ragas and LLM-as-a-judge methodologies.

This repository tracks empirical performance across architectural iterations, replacing subjective heuristic assessments with repeatable quantitative benchmarks.

---

## Table of Contents

- [System Architecture](#system-architecture)
- [Directory Structure](#directory-structure)
- [Design Principles](#design-principles)
- [Inference & Embedding Providers](#inference--embedding-providers)
- [Evaluation Methodology & Metrics](#evaluation-methodology--metrics)
- [Baseline Benchmark Results (Phase 1)](#baseline-benchmark-results-phase-1)
- [Ablation Study Roadmap](#ablation-study-roadmap)
- [Setup & Reproducibility](#setup--reproducibility)
  - [Prerequisites](#prerequisites)
  - [Environment Configuration](#environment-configuration)
  - [Corpus Ingestion](#corpus-ingestion)
  - [Running Evaluations](#running-evaluations)
  - [Interactive Web Interface](#interactive-web-interface)
- [Branching Model](#branching-model)
- [License](#license)

---

## System Architecture

The pipeline decouples ingestion, state management, retrieval-augmented inference, and quantitative evaluation:

```
[ Documents (PDF, MD, TXT, DOCX) / Benchmark Datasets ]
                        │
                        ▼
         Recursive Character Text Splitter (Chunk: 800, Overlap: 150)
                        │
                        ▼
         Embedding Model (sentence-transformers/all-MiniLM-L6-v2)
                        │
                        ▼
         ChromaDB Persistent Vector Store (Cosine Distance Index)
                        ▲
                        │ Dense Top-k Retrieval (k = 4)
                        │
[ User Prompt ] ──► [ Query Contextualizer ] ──► [ Augmented Prompt ]
                          ▲ (Chat History)              │
                          │                             ▼
                 ChatHistoryManager            Target LLM Engine
               (Session JSON Persistence)      (Groq / NVIDIA / Ollama)
                                                        │
                                                        ▼
                                               [ Streaming Response +
                                                 Source Attribution ]
                                                        │
                                                        ▼
                                           [ Evaluation Pipeline ]
                                        (Context Recall, Precision,
                                         Faithfulness, Relevancy)
```

---

## Directory Structure

```
.
├── .env.example                       # Environment configuration template
├── requirements.txt                   # Production and evaluation dependencies
├── app.py                             # Streamlit interactive application
├── README.md                          # Engineering and architectural documentation
│
├── data/
│   ├── raw/                           # Document storage for user-uploaded files
│   └── testsets/                      # Golden evaluation datasets (Amnesty QA, custom)
│
├── storage/                           # Persistent local storage (git-ignored)
│   ├── chroma/                        # ChromaDB SQLite metadata and binary vector index
│   └── history/                       # Multi-session conversational JSON stores
│
├── src/                               # Core RAG application package
│   ├── __init__.py
│   ├── config.py                      # Pydantic-based settings validation
│   ├── models.py                      # Unified factory for LLMs and embeddings
│   ├── vectorstore.py                 # ChromaDB indexing, chunking, and similarity search
│   ├── memory.py                      # Session-isolated conversation persistence
│   ├── chain.py                       # LangChain LCEL pipeline with query contextualization
│   └── utils.py                       # Multi-format document parsers and dataset loaders
│
├── evals/                             # Quantitative evaluation framework
│   ├── run_eval.py                    # Rate-throttled LLM-as-a-judge evaluation runner
│   ├── run_native_ragas.py            # Native Ragas evaluate() execution harness
│   ├── generate_eval_dataset.py       # Synthetic QA and ground-truth generation pipeline
│   └── benchmark_results/             # Evaluation logs, CSV scorecards, and analysis
│
└── scripts/                           # Maintenance and diagnostic utilities
    ├── test_setup.py                  # Environment connectivity and API health check
    ├── prepare_amnesty_benchmark.py   # Corpus extraction and benchmark testset staging
    └── ingest_sample.py               # CLI document ingestion utility
```

---

## Design Principles

1. **Provider Agnosticism**: Unified interfaces allow switching among ultra-low latency cloud inference (Groq), high-throughput enterprise infrastructure (NVIDIA NIM), multi-model gateways (OpenRouter), and local air-gapped instances (Ollama) with zero code modifications.
2. **Reproducible Evaluation**: Every architectural modification is measured against a fixed ground-truth testset using deterministic evaluation parameters ($T=0.0$).
3. **Conversational Disambiguation**: Multi-turn questions are rewritten into fully autonomous search queries via an LCEL reformulation stage prior to index querying, resolving pronoun ambiguity.
4. **Transparent Provenance**: All synthesized responses return source chunk IDs, document metadata, and similarity distances for auditability.
5. **Zero Data Leakage**: Vector indices, session histories, raw documents, and secret keys remain strictly isolated and protected from version control.

---

## Inference & Embedding Providers

| Provider | Purpose | Default Model | Fallback / Alternative |
| :--- | :--- | :--- | :--- |
| **Groq Cloud** | High-speed LLM inference | `qwen/qwen-2.5-32b` / `qwen/qwen3.8-27b` | `llama-3.1-8b-instant` |
| **NVIDIA NIM** | Scalable enterprise cloud | `meta/llama-3.2-11b-vision-instruct` | `meta/llama-3.1-8b-instruct` |
| **OpenRouter** | Multi-vendor fallback | `openrouter/free` | Provider catalog |
| **Ollama** | Local / private offline inference | `llama3.1:8b` | `qwen3:14b`, `llama3:latest` |
| **Hugging Face** | Local semantic embedding | `sentence-transformers/all-MiniLM-L6-v2` | CPU/GPU local execution |

---

## Evaluation Methodology & Metrics

To assess retrieval and generation quality objectively, the evaluation framework implements four foundational RAG metrics:

- **Context Recall**: Evaluates whether the retrieved context passages contain all factual components required by the ground-truth answer.
  $$\text{Context Recall} = \frac{|\text{Ground-Truth Sentences Attributed to Context}|}{|\text{Total Ground-Truth Sentences}|}$$

- **Context Precision**: Evaluates the signal-to-noise ratio of the retrieved chunks and penalizes irrelevant chunks ranked above relevant ones.
  $$\text{Context Precision@k} = \frac{\sum_{k=1}^K (\text{Precision@}k \times v_k)}{\text{Total Relevant Documents Retrieved}}$$

- **Faithfulness**: Measures whether all claims made in the generated answer can be mathematically derived from the retrieved context, preventing ungrounded hallucinations.
  $$\text{Faithfulness} = \frac{|\text{Supported Claims in Answer}|}{|\text{Total Claims in Answer}|}$$

- **Answer Relevance**: Evaluates semantic alignment between the user's prompt and the generated response, penalizing incomplete or tangential outputs.

### Benchmark Dataset

The baseline benchmark is executed on the standardized `explodinggradients/amnesty_qa` (`english_v2`) dataset:
- **Corpus**: 168 extracted document chunks indexed into ChromaDB across 23 global human rights reports.
- **Evaluation Samples**: 20 multi-context reasoning queries designed specifically to test cross-document synthesis and long-tail factual retrieval.

---

## Master Ablation Benchmark Scorecard (All 4 Phases)

Evaluated on the standardized `explodinggradients/amnesty_qa` golden benchmark ($N=20$ multi-context queries) across ChromaDB ($168$ chunks) with an automated multi-provider LLM-as-a-Judge resilient pool:

| Metric | Phase 1 (Dense Baseline) | Phase 2 (Hybrid BM25+RRF) | Phase 3 (Cross-Encoder Rerank) | Phase 4 (Multi-Query Transform) | Net Lift (P4 vs P1) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Context Recall** | **76.50%** | **86.25%** | **86.50%** | **88.00%** | **+11.50%** |
| **Context Precision** | **59.50%** | **71.00%** | **77.00%** | **79.25%** | **+19.75%** |
| **Faithfulness (Grounding)** | **94.00%** | **93.75%** | **92.50%** | **88.75%** | **-5.25%** |
| **Answer Relevance** | **73.65%** | **88.25%** | **89.40%** | **84.90%** | **+11.25%** |
| **Harmonized Triad Index** | **75.91%** | **84.81%** | **86.35%** | **85.22%** | **+9.31%** |
| **Average Latency** | **11.31s** | **9.80s** | **15.23s** | **34.85s** | **+23.54s** |

### Key Architectural Takeaways

1. **Hybrid Search Resolves Exact Entity Mismatches (+9.75% Recall)**: Dense embeddings fail on exact statutory citations (e.g. *Article 207.3*) and acronyms (*GHG*). BM25 Okapi fused via Reciprocal Rank Fusion ($k=60$) instantly recovered missing keywords.
2. **Two-Stage Cross-Encoder Eliminates Distractor Noise (+17.50% Precision)**: Bi-encoder cosine distance lacks cross-token attention. Scoring $M=15$ candidate passages through `cross-encoder/ms-marco-MiniLM-L-6-v2` pruned over 73% of candidate distractors.
3. **Multi-Query Decomposition Conquers Complex Queries (88.00% Recall, 79.25% Precision)**: Breaking compound multi-hop prompts into parallel orthogonal sub-queries and fusing deduplicated candidate pools unlocks long-tail facts without losing precision.
4. **Adaptive Routing Provides 0ms Chitchat Bypass**: Greetings and general conversational turns bypass vector retrieval entirely, preserving GPU/inference bandwidth.

---

## Ablation Study Roadmap & Branching Architecture

Each architectural phase is cleanly engineered and isolated across dedicated feature branches:

| Phase | Branch | Architectural Enhancement | Status |
| :--- | :--- | :--- | :--- |
| **Phase 1** | `phase-1-basic-rag` | **Baseline Dense RAG**: ChromaDB vector index with standard LCEL chain. | ✅ **Complete** |
| **Phase 2** | `phase-2-hybrid-search` | **Hybrid Retrieval**: BM25 sparse lexical search + dense vectors via Reciprocal Rank Fusion (RRF). | ✅ **Complete** |
| **Phase 3** | `phase-3-reranker` | **Cross-Encoder Reranker**: Two-stage retrieval with `cross-encoder/ms-marco-MiniLM-L-6-v2` cross-attention. | ✅ **Complete** |
| **Phase 4** | `phase-4-query-transformation` | **Query Transformation & Routing**: Multi-Query decomposition, HyDE embeddings, and adaptive direct intent bypass. | ✅ **Complete** |

---

## Setup & Reproducibility

### Prerequisites

- Python 3.11 or higher
- Git
- Virtual environment manager (`venv` or `conda`)

### Environment Configuration

1. Clone the repository:
   ```bash
   git clone https://github.com/SiddheshDhomse/rag-with-evals.git
   cd rag-with-evals
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # Linux / macOS:
   source .venv/bin/activate
   ```

3. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Configure environment credentials:
   ```bash
   cp .env.example .env
   ```
   Edit `.env` with your active provider API keys:
   ```env
   # LLM Provider Credentials
   GROQ_API_KEY=gsk_your_groq_api_key
   NVIDIA_API_KEY=nvapi_your_nvidia_api_key
   OPEN_ROUTE_API_KEY=sk-or-your_openrouter_api_key

   # Local Ollama Endpoint (Optional)
   OLLAMA_BASE_URL=http://localhost:11434

   # System Defaults
   DEFAULT_LLM_PROVIDER=groq
   EMBEDDING_PROVIDER=huggingface
   EMBEDDING_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
   ```

5. Verify system connectivity and credentials:
   ```bash
   python scripts/test_setup.py
   ```

### Corpus Ingestion

Seed the persistent ChromaDB instance with the official 168-chunk Amnesty QA benchmark corpus:
```bash
python scripts/prepare_amnesty_benchmark.py
```

To ingest arbitrary local documents (PDF, TXT, MD, DOCX):
```bash
python scripts/ingest_sample.py --file path/to/document.pdf
```

### Running Evaluations

Run the rate-throttled LLM-as-a-judge benchmark across the golden testset:
```bash
# Evaluate full 20-sample testset using Groq
python evals/run_eval.py --provider groq --dataset amnesty_qa --limit 20 --delay 2.5

# Evaluate using local Ollama instance
python evals/run_eval.py --provider ollama --dataset amnesty_qa --limit 20
```

To execute native Ragas framework metrics:
```bash
python evals/run_native_ragas.py
```

Generated evaluation reports are persisted to:
- CSV Detail: `evals/benchmark_results/baseline_scores_amnesty_qa_eval.csv`
- Markdown Summary: `evals/benchmark_results/baseline_summary.md`

### Interactive Web Interface

Launch the Streamlit production dashboard:
```bash
streamlit run app.py
```

Features available in the interface:
- Real-time provider and model selector (Groq, NVIDIA NIM, OpenRouter, Ollama).
- Document ingestion manager supporting drag-and-drop file processing.
- Multi-session chat history browser with independent session memory.
- Retrieved context inspection drawer displaying chunk text, source filenames, and cosine distances.

---

## Branching Model

This repository follows an ablation-driven branching model where each architectural stage is isolated in a dedicated branch:

- `main`: Production-ready, fully validated baseline and latest stable release.
- `phase-1-basic-rag`: Baseline dense retrieval pipeline with evaluation harness.
- `phase-2-hybrid-search`: BM25 lexical + dense vector reciprocal rank fusion.
- `phase-3-reranker`: Cross-encoder reranking integration.
- `phase-4-query-transform`: HyDE and multi-query expansion.

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
