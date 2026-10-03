# RAG with Evals 🚀

A production-grade, modular Retrieval-Augmented Generation (RAG) system with multi-provider LLM support (**Groq Cloud**, **NVIDIA NIM**, and **Ollama**), persistent vector storage via **ChromaDB**, conversational memory, and an ablation evaluation framework powered by **Ragas**.

---

## 🏗️ Architecture

```
RAG with Evals/
├── .env.example              # Template for API keys & settings
├── .env                      # Your local environment variables
├── requirements.txt          # Python dependencies
├── app.py                    # Production Streamlit UI
├── data/
│   ├── raw/                  # Storage for uploaded/custom documents
│   └── testsets/             # Benchmark golden datasets for evaluations
├── storage/
│   ├── chroma/               # ChromaDB persistent vector database
│   └── history/              # Persistent multi-session chat histories (JSON)
├── src/
│   ├── __init__.py
│   ├── config.py             # Centralized settings and .env validator
│   ├── models.py             # LLM & Embedding factories (Groq, NVIDIA, Ollama)
│   ├── vectorstore.py        # ChromaDB manager (chunking, indexing, retrieval)
│   ├── memory.py             # Multi-session chat history manager
│   ├── chain.py              # Production LCEL Conversational RAG chain
│   └── utils.py              # File parsers (PDF, TXT, MD, DOCX) & dataset loaders
└── scripts/
    ├── test_setup.py         # System health-check script
    └── ingest_sample.py      # CLI tool to seed ChromaDB with benchmark data
```

---

## ⚡ Quick Start

### 1. Configure API Keys in `.env`
Open the `.env` file and add your credentials:
```env
# Free Groq API Key: https://console.groq.com/keys
GROQ_API_KEY=gsk_...

# Free NVIDIA NIM Key (1000 free credits): https://build.nvidia.com
NVIDIA_API_KEY=nvapi-...

# Ollama local URL (if running Ollama locally)
OLLAMA_BASE_URL=http://localhost:11434
```

### 2. Verify Your Environment
Run the automated health-check script:
```powershell
python scripts/test_setup.py
```

### 3. Launch the Streamlit Chatbot UI
```powershell
streamlit run app.py
```

---

## 🌟 Key Features

1. **Multi-Provider Flexibility**:
   - **Groq Cloud**: Lightning-fast inference on `llama-3.3-70b-versatile` & `mixtral-8x7b-32768`.
   - **NVIDIA NIM**: High-throughput cloud inference on `meta/llama-3.1-70b-instruct`.
   - **Ollama**: 100% offline, private local models (`llama3.2`, `mistral`, `deepseek-r1`).

2. **Persistent ChromaDB Vector Store**:
   - Automatic chunking with `RecursiveCharacterTextSplitter`.
   - Embeddings: Default local HuggingFace (`all-MiniLM-L6-v2`) with zero API cost.
   - 1-click seeding of standard RAG benchmark dataset (`explodinggradients/amnesty_qa`).

3. **Conversational Memory & Query Reformulation**:
   - Multi-turn conversation context awareness.
   - Automatically rewrites follow-up questions into standalone search queries.
   - Persistent session storage: switch sessions, inspect previous transcripts, or clear history.

4. **Transparent Source Attribution**:
   - Every answer displays an expandable drawer containing exact retrieved text chunks, source filenames, page numbers, and vector distance scores.

---

## 📊 Evaluation & Ablation Roadmap

This repository is designed to benchmark and track RAG performance improvements across iterative enhancements:

| Phase | Architecture Milestone | Key Hypothesis | Target Metric Improvement |
| :--- | :--- | :--- | :--- |
| **Baseline** | Naive ChromaDB + Dense Retrieval | Establish baseline performance | Baseline Context Precision & Recall |
| **Stage 1** | **Hybrid Search** (BM25 + Dense RRF) | Fixes missed exact keyword matches | ⬆ Context Recall |
| **Stage 2** | **Cross-Encoder Reranker** (FlashRank) | Prunes noise from top-15 to top-4 | ⬆ Context Precision & Faithfulness |
| **Stage 3** | **Query Expansion** (HyDE / Multi-Query) | Resolves complex/vague questions | ⬆ Answer Relevancy |
