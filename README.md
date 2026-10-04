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

## 📑 Table of Contents

- [System Architecture](#system-architecture)
- [Directory Structure](#directory-structure)
- [Design Principles](#design-principles)
- [Inference & Embedding Providers](#inference--embedding-providers)
- [Evaluation Methodology & Metrics](#evaluation-methodology--metrics)
- [Master 4-Phase Ablation Scorecard](#master-4-phase-ablation-scorecard)
- [Phase Summaries & Architectural Lessons](#phase-summaries--architectural-lessons)
- [Setup & Reproducibility](#setup--reproducibility)
  - [Prerequisites](#prerequisites)
  - [Environment Configuration](#environment-configuration)
  - [Corpus Ingestion](#corpus-ingestion)
  - [Running Unit Tests](#running-unit-tests)
  - [Running Evaluations](#running-evaluations)
  - [Interactive Web Interface](#interactive-web-interface)
- [Limitations](#limitations)
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
         ChromaDB Persistent Vector Store + BM25Okapi Lexical Index
                        ▲
                        │ Stage 1: Hybrid Retrieval (Dense Cosine + Sparse BM25 via RRF)
                        │ Stage 2: Cross-Encoder Reranker (ms-marco-MiniLM-L-6-v2)
                        │
[ User Prompt ] ──► [ Query Contextualizer ] ──► [ Adaptive Router ]
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

For complete technical specifications, see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

---

## Directory Structure

```text
.
├── configs/                           # Declarative system and model configurations
│   └── settings.yaml
├── data/                              # Benchmark and raw document storage
│   ├── raw/                           # Document storage for user-uploaded files (.gitkeep)
│   └── testsets/                      # Golden evaluation datasets (amnesty_qa_eval.json)
├── src/                               # Core RAG application package
│   ├── __init__.py
│   ├── config.py                      # Dataclass settings and environment validator
│   ├── models.py                      # Unified factory for LLMs and embeddings
│   ├── vectorstore.py                 # ChromaDB indexing, chunking, and BM25 RRF hybrid retrieval
│   ├── memory.py                      # Multi-session conversational JSON stores
│   ├── chain.py                       # LCEL pipeline with query contextualization & routing
│   ├── reranker.py                    # Cross-encoder reranker (ms-marco-MiniLM-L-6-v2)
│   ├── query_transform.py             # HyDE, Multi-Query, Step-Back & Semantic Router
│   └── utils.py                       # Document parsers and dataset loaders
├── evals/                             # Quantitative evaluation framework
│   ├── __init__.py
│   ├── run_eval.py                    # Rate-throttled LLM-as-a-judge evaluation runner
│   ├── run_native_ragas.py            # Native Ragas evaluate() execution harness
│   ├── generate_eval_dataset.py       # Synthetic QA and ground-truth generation pipeline
│   └── benchmark_results/             # Evaluation logs and backward-compatible results
├── results/                           # Verified empirical scorecards and benchmark CSVs
│   ├── baseline_scores_amnesty_qa_eval.csv
│   ├── hybrid_scores_amnesty_qa_eval.csv
│   ├── rerank_scores_amnesty_qa_eval.csv
│   ├── rerank_scores_multi_query_amnesty_qa_eval.csv
│   ├── native_ragas_scores_groq.csv
│   └── phase4_full_ablation_scorecard.md
├── app/                               # Web interface package
│   ├── __init__.py
│   └── main.py                        # Streamlit production dashboard
├── tests/                             # Automated unit test suite
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_config.py
│   ├── test_chain.py
│   ├── test_memory.py
│   ├── test_query_transform.py
│   ├── test_reranker.py
│   └── test_vectorstore.py
├── scripts/                           # Maintenance and diagnostic utilities
│   ├── test_setup.py                  # Environment connectivity and API health check
│   ├── prepare_amnesty_benchmark.py   # Corpus extraction and benchmark testset staging
│   ├── ingest_sample.py               # CLI document ingestion utility
│   ├── export_benchmark_json.py       # Exports benchmark CSVs to unified JSON
│   └── build_4phase_dashboard.py      # Standalone 4-phase dashboard generator
├── assets/                            # Visual assets, diagrams, and figures
├── docs/                              # System documentation and experiment guides
│   ├── ARCHITECTURE.md                # System components, formulations & design
│   ├── EXPERIMENTS.md                 # Ablation methodology, results & scorecards
│   └── LINKEDIN_CASE_STUDY.md         # Publication-ready case study & analysis
├── archive/                           # Preserved legacy scripts and notes
│   ├── legacy_dashboards/
│   └── legacy_notes/
├── storage/                           # Persistent local storage (git-ignored)
│   ├── chroma/                        # ChromaDB SQLite metadata and binary vector index
│   └── history/                       # Multi-session conversational JSON stores
├── .env.example                       # Environment configuration template
├── .gitignore                         # Strict exclusion rules (secrets, DBs, histories)
├── app.py                             # Root entrypoint shim for Streamlit
├── LICENSE                            # MIT License
├── README.md                          # Repository documentation
└── requirements.txt                   # Production dependencies
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
| **Groq Cloud** | High-speed LLM inference | `qwen/qwen3.8-27b` | `llama-3.1-8b-instant`, `qwen/qwen-2.5-32b` |
| **NVIDIA NIM** | Scalable enterprise cloud | `meta/llama-3.2-11b-vision-instruct` | `meta/llama-3.1-8b-instruct` |
| **OpenRouter** | Multi-vendor fallback | `openrouter/free` | Provider catalog |
| **Ollama** | Local / private offline inference | `llama3.1:8b` | `qwen3:14b`, `llama3:latest` |
| **Hugging Face** | Local semantic embedding | `sentence-transformers/all-MiniLM-L6-v2` | Zero API cost, CPU/GPU local |

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

## Master 4-Phase Ablation Scorecard

Evaluated on the standardized `explodinggradients/amnesty_qa` golden benchmark ($N=20$ multi-context queries) across ChromaDB ($168$ chunks) with an automated multi-provider LLM-as-a-Judge resilient pool:

| Metric | Phase 1 (Dense Baseline) | Phase 2 (Hybrid BM25+RRF) | Phase 3 (Cross-Encoder Rerank) | Phase 4 (Multi-Query Transform) | Net Lift (P4 vs P1) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Context Recall** | **76.50%** | **86.25%** | **86.50%** | **88.00%** | **+11.50% (Peak)** |
| **Context Precision** | **59.50%** | **71.00%** | **77.00%** | **79.25%** | **+19.75% (Peak)** |
| **Faithfulness (Raw N=20)** | **94.00%** | **93.75%** | **92.50%** | **88.75%** | **-5.25%** |
| **Faithfulness (Norm N=19)**| **94.00%** | **93.75%** | **92.50%** | **93.42%** | **-0.58%** |
| **Answer Relevance** | **73.65%** | **88.25%** | **89.40%** | **84.90%** (89.37% norm) | **+11.25%** |
| **Harmonized Triad Index** | **75.91%** | **84.81%** | **86.35%** | **85.22%** (**87.21%** norm) | **+11.30% (Record)** |
| **Average Latency** | **11.31s** | **9.80s** | **15.23s** | **34.85s** | **+23.54s** |

---

## Phase Summaries & Architectural Lessons

1. **Phase 1 (Dense Baseline — The Vocabulary Gap)**:
   - Dense embeddings capture overarching semantic topics, but fail on statutory citations (*Article 207.3*), acronyms (*GHG*), and treaty names (*Ramsar*). Precision of 59.50% meant 40% of retrieved chunks were irrelevant distractors.

2. **Phase 2 (Hybrid BM25 + Dense RRF — The Lexical Fix)**:
   - Fusing sparse BM25 with dense vectors via Reciprocal Rank Fusion ($k=60$) lifted Context Recall from 76.50% to 86.25% (+9.75%) and Precision to 71.00% (+11.50%) with zero extra latency.

3. **Phase 3 (Two-Stage Cross-Encoder — Distractor Elimination)**:
   - Cross-encoder reranking over $M=15$ candidate passages filtered over 73% of candidate distractors, raising Context Precision to 77.00% (+17.50% over baseline).

4. **Phase 4 (Query Transformation & Adaptive Routing — Multi-Hop Mastery)**:
   - Multi-Query decomposition deconstructed compound queries into parallel sub-searches, achieving peak Context Recall (88.00%) and peak Context Precision (79.25%).
   - The Adaptive Router provided a 0ms retrieval bypass for conversational chitchat.

For detailed breakdown, see [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md).

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
   GROQ_API_KEY=your_groq_api_key_here
   NVIDIA_API_KEY=your_nvidia_api_key_here
   OPEN_ROUTE_API_KEY=your_openrouter_api_key_here

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
python scripts/ingest_sample.py --dir data/raw
```

### Running Unit Tests

Execute the automated test suite verifying settings, memory manager, query routing heuristics, reranker math, and vectorstore logic:
```bash
pytest tests
```

### Running Evaluations

Run the rate-throttled LLM-as-a-judge benchmark across the golden testset:
```bash
# Evaluate full 20-sample testset using Groq
python evals/run_eval.py --provider groq --dataset amnesty_qa --limit 20 --delay 2.5

# Evaluate Phase 4 Multi-Query Transformation
python evals/run_eval.py --provider round_robin --mode hybrid_rerank --transform multi_query --limit 20

# Evaluate using local Ollama instance
python evals/run_eval.py --provider ollama --dataset amnesty_qa --limit 20
```

To execute native Ragas framework metrics:
```bash
python evals/run_native_ragas.py
```

Generated evaluation reports are persisted to:
- CSV Detail: `results/baseline_scores_amnesty_qa_eval.csv`
- Markdown Summary: `results/phase4_full_ablation_scorecard.md`

### Interactive Web Interface

Launch the Streamlit production dashboard:
```bash
streamlit run app.py
```
*(Alternatively: `streamlit run app/main.py`)*

Features available in the interface:
- Real-time provider and model selector (Groq, NVIDIA NIM, OpenRouter, Ollama).
- Document ingestion manager supporting drag-and-drop file processing.
- Multi-session chat history browser with independent session memory.
- Retrieved context inspection drawer displaying chunk text, source filenames, and cosine distances.

---

## Limitations

1. **Free-Tier Rate Limits**: Evaluation speed is constrained by cloud free-tier RPM/TPM ceilings (Groq, OpenRouter), requiring 2.0s inter-call backoffs during large batch runs.
2. **Third-Party Moderation Filters**: Single queries covering sensitive topics (e.g. human rights abuses) can trigger external safety refusals, requiring multi-provider fallback logic.
3. **Multi-Query Latency**: Generating 3 sub-queries and executing multiple retrieval passes increases turnaround latency from ~15s to ~34s on single-threaded workers.

---

## Branching Model

This repository follows an ablation-driven branching model where each architectural stage is isolated in a dedicated branch:

- `main`: Production-ready, fully validated baseline and latest stable release.
- `phase-1-basic-rag`: Baseline dense retrieval pipeline with evaluation harness.
- `phase-2-hybrid-search`: BM25 lexical + dense vector reciprocal rank fusion.
- `phase-3-reranker`: Cross-encoder reranking integration.
- `phase-4-query-transformation`: HyDE and multi-query expansion.
- `cleanup/structure-and-docs`: Production structure cleanup, unit test suite, and engineering documentation.

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
