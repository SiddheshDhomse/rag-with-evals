# Repository Audit & Proposed Cleanup Plan

**Date**: 2026-10-04  
**Project**: RAG Evaluation Suite (`RAG with Evals`)  
**Auditor**: Senior Engineering Cleanup Agent  
**Branch**: `cleanup/structure-and-docs`  
**Base Commit SHA**: `43b53a60a7e6ec35b671a533036502aa8784d0fe`  
**Tag**: `pre-cleanup`  

---

## 1. Current Repository Structure

```text
d:/AI Projects/RAG with Evals/
├── .env                              # Local environment variables (strictly git-ignored)
├── .env.example                      # Environment variables template with placeholder values
├── .gitignore                        # Git ignore rules for secrets, DBs, histories, caches
├── app.py                            # Production Streamlit UI (single-file monolithic app)
├── build_index.py                    # Standalone HTML builder script (git-ignored)
├── eval_dashboard.html               # Standalone interactive dashboard v1 (git-ignored)
├── eval_dashboard_v2.html            # Standalone interactive dashboard v2 (git-ignored)
├── index.html                        # Standalone web dashboard portal (git-ignored)
├── LICENSE                           # MIT License file
├── notes.md                          # Comprehensive engineering notes & ablation handbook (24KB)
├── rag-pipeline-animation (4).html   # Standalone HTML animation (git-ignored)
├── README.md                         # Public repository documentation & master scorecard (15KB)
├── requirements.txt                  # Python dependencies
├── samples_extracted.js              # Evaluation samples JS data file (git-ignored)
├── test_script.js                    # Evaluation metrics JS script (git-ignored)
├── data/
│   ├── raw/
│   │   └── .gitkeep                  # Empty directory tracking for document uploads
│   └── testsets/
│       └── amnesty_qa_eval.json      # 20-sample golden evaluation dataset (explodinggradients)
├── evals/
│   ├── generate_eval_dataset.py      # LLM synthesis pipeline for custom QA golden datasets
│   ├── run_eval.py                   # 4-phase evaluation harness with LLM-as-a-judge
│   ├── run_native_ragas.py           # Native Ragas evaluate() wrapper
│   └── benchmark_results/            # Raw evaluation scorecards and markdown summaries
│       ├── baseline_scores.csv       # (git-ignored early run)
│       ├── baseline_scores_amnesty_qa_eval.csv
│       ├── baseline_scores_amnesty_qa_eval_original4.csv  # (git-ignored)
│       ├── baseline_summary.md
│       ├── hybrid_scores_amnesty_qa_eval.csv
│       ├── LINKEDIN_CASE_STUDY.md
│       ├── native_ragas_scores_groq.csv
│       ├── phase2_hybrid_vs_baseline.md
│       ├── phase3_rerank_vs_hybrid_vs_baseline.md
│       ├── phase4_full_ablation_scorecard.md
│       ├── phase4_multi_query_vs_rerank.md
│       ├── rerank_scores_amnesty_qa_eval.csv
│       └── rerank_scores_multi_query_amnesty_qa_eval.csv
├── scripts/
│   ├── benchmark_data.json           # Aggregated 4-phase benchmark metrics JSON
│   ├── build_4phase_dashboard.py     # Standalone 4-phase dashboard generator
│   ├── build_dashboard.py            # Phase 3 dashboard compiler (superseded)
│   ├── export_benchmark_json.py      # Exports benchmark CSVs into unified JSON
│   ├── generate_dashboard_html.py    # Phase 3 HTML generator (superseded)
│   ├── ingest_sample.py              # CLI ingestion utility for local documents / benchmark
│   ├── prepare_amnesty_benchmark.py  # Corpus loader & ChromaDB indexing pipeline
│   └── test_setup.py                 # Multi-provider environment connectivity & health check
├── src/
│   ├── __init__.py
│   ├── chain.py                      # Conversational RAG chain with LCEL & query routing
│   ├── config.py                     # Dataclass settings, path resolution, and provider validation
│   ├── memory.py                     # Multi-session JSON chat history manager
│   ├── models.py                     # Unified LLM & embedding provider factories
│   ├── query_transform.py            # Phase 4 HyDE, Multi-Query, Step-Back & Adaptive Router
│   ├── reranker.py                   # Phase 3 Cross-Encoder reranker (ms-marco-MiniLM-L-6-v2)
│   ├── utils.py                      # Multi-format document parser & dataset loaders
│   └── vectorstore.py                # ChromaDB manager, BM25 indexer, and RRF hybrid retrieval
└── storage/                          # Local persistent storage (strictly git-ignored)
    ├── chroma/                       # ChromaDB SQLite DB and HNSW binary indices
    └── history/                      # Multi-session conversation logs (*.json)
```

---

## 2. Proposed Structure

Per guideline 5 and repo hygiene standards:

```text
configs/
└── settings.yaml                      # Optional declarative config / schema reference
data/
├── raw/
│   └── .gitkeep
└── testsets/
    └── amnesty_qa_eval.json
src/
├── __init__.py
├── chain.py
├── config.py
├── memory.py
├── models.py
├── query_transform.py
├── reranker.py
├── utils.py
└── vectorstore.py
evals/
├── __init__.py
├── generate_eval_dataset.py
├── run_eval.py
└── run_native_ragas.py
results/
├── baseline_scores_amnesty_qa_eval.csv
├── baseline_summary.md
├── hybrid_scores_amnesty_qa_eval.csv
├── LINKEDIN_CASE_STUDY.md
├── native_ragas_scores_groq.csv
├── phase2_hybrid_vs_baseline.md
├── phase3_rerank_vs_hybrid_vs_baseline.md
├── phase4_full_ablation_scorecard.md
├── phase4_multi_query_vs_rerank.md
├── rerank_scores_amnesty_qa_eval.csv
└── rerank_scores_multi_query_amnesty_qa_eval.csv
app/
├── __init__.py
└── main.py                           # Cleaned Streamlit entrypoint (or app.py symlink/alias)
tests/
├── __init__.py
├── conftest.py                       # Pytest fixtures and mocks
├── test_config.py                    # Settings and path resolution tests
├── test_memory.py                    # Session history manager tests
├── test_models.py                    # Model factory validation tests
├── test_query_transform.py           # Parsing and heuristic routing tests
├── test_reranker.py                  # Sigmoid and rank delta calculation tests
└── test_vectorstore.py               # RRF fusion logic and chunking tests
scripts/
├── benchmark_data.json
├── build_4phase_dashboard.py
├── export_benchmark_json.py
├── ingest_sample.py
├── prepare_amnesty_benchmark.py
└── test_setup.py
assets/
└── (diagrams, architecture charts, screenshots)
docs/
├── ARCHITECTURE.md                   # Full systems engineering & module architecture
├── EXPERIMENTS.md                    # 4-Phase ablation methodology, findings & scorecards
└── LINKEDIN_CASE_STUDY.md            # Case study & publication reference
archive/
├── legacy_dashboards/
│   ├── build_dashboard.py            # Phase 3 legacy builder
│   └── generate_dashboard_html.py    # Phase 3 legacy generator
└── legacy_notes/
    └── notes.md                      # Original monolithic engineering notes
```

---

## 3. Old → New File Mapping

| Old Path | New Path | Action / Rationale |
| :--- | :--- | :--- |
| `app.py` | `app/main.py` + root `app.py` wrapper | Move UI to `app/main.py`. Maintain a thin root `app.py` shim (`from app.main import ...` or `streamlit run app/main.py`) so existing commands do not break. |
| `evals/benchmark_results/*` | `results/*` | Relocate benchmark scorecards and CSV logs to top-level `results/` per target structure. Create compatibility path/symlink or update config so evaluation scripts find it seamlessly without altering numbers. |
| `notes.md` | Split into `docs/ARCHITECTURE.md`, `docs/EXPERIMENTS.md`, `archive/legacy_notes/notes.md` | Extract clean, structured documentation into `docs/` while preserving the original historical reference in `archive/`. |
| `scripts/build_dashboard.py` | `archive/legacy_dashboards/build_dashboard.py` | Phase 3 dashboard compiler superseded by `scripts/build_4phase_dashboard.py`. Archive safely without deletion. |
| `scripts/generate_dashboard_html.py` | `archive/legacy_dashboards/generate_dashboard_html.py` | Phase 3 dashboard HTML generator superseded by `scripts/build_4phase_dashboard.py`. Archive safely without deletion. |
| `evals/benchmark_results/LINKEDIN_CASE_STUDY.md` | `docs/LINKEDIN_CASE_STUDY.md` (copy to `results/` & `docs/`) | Preserved in both results and documented in docs. |
| Root ignored files (`eval_dashboard*.html`, `*.js`, `build_index.py`, `index.html`) | Retained in root (git-ignored) or moved to `assets/` / `archive/` | Retained without touching per user preferences, properly ignored in `.gitignore`. |

---

## 4. Duplicated / Dead Code

1. **Duplicate Class Definition in `src/chain.py`**:
   - Lines 138–145 declare `class ConversationalRAGChain:` with a docstring stub, immediately followed by `DIRECT_SYSTEM_PROMPT` (Line 148), and then the actual `class ConversationalRAGChain:` is declared again at Line 158.
   - *Status*: The first declaration is redundant dead code.
2. **Superseded Dashboard Scripts in `scripts/`**:
   - `build_dashboard.py` and `generate_dashboard_html.py` were written for Phase 3 (3-phase ablation) and hardcode 3 CSVs. `build_4phase_dashboard.py` and `export_benchmark_json.py` now handle the complete 4-phase system.
   - *Status*: Move the superseded Phase 3 scripts to `archive/legacy_dashboards/`.
3. **Redundant Fallback Import in `src/vectorstore.py`**:
   - Line 131 imports `from langchain_classic.retrievers import EnsembleRetriever`. If unavailable, it catches `Exception` and falls back.

---

## 5. Hard-Coded Paths & Values

1. **`src/config.py`**:
   - `PROJECT_ROOT = Path(__file__).resolve().parent.parent`: Resolves cleanly to repository root.
   - `chroma_dir`: Defaults to `storage/chroma`.
   - `history_dir`: Defaults to `storage/history`.
2. **`evals/run_eval.py`**:
   - `results_dir = PROJECT_ROOT / "evals" / "benchmark_results"`: Hardcodes output directory.
   - If results move to `results/`, `results_dir` should default to `PROJECT_ROOT / "results"` with fallback to `evals/benchmark_results`.
3. **`evals/run_native_ragas.py`**:
   - `results_dir = PROJECT_ROOT / "evals" / "benchmark_results"`: Hardcodes output directory.
4. **`scripts/export_benchmark_json.py`**:
   - `BENCH_DIR = ROOT / "evals" / "benchmark_results"`: Hardcoded paths to the 4 CSV files.
5. **`scripts/build_dashboard.py`**:
   - Hardcodes `evals/benchmark_results/...` relative paths.

---

## 6. Risks & Mitigation Plan

| Risk | Impact | Mitigation Strategy |
| :--- | :--- | :--- |
| **Breaking Benchmark Reproducibility** | CRITICAL | **Rule 3 strictly enforced**: Never modify `results/` CSVs, chunk size (800/150), models, seeds, or evaluation logic. Verify identical metrics before and after. |
| **Breaking Streamlit or CLI commands** | HIGH | Maintain root `app.py` entrypoint that delegates to `app/main.py`. Preserve all CLI argument flags in `evals/run_eval.py` and `scripts/*.py`. |
| **Breaking Module Imports** | HIGH | Add `__init__.py` to all new packages (`app/`, `evals/`, `tests/`, etc.). Run pytest and import sanity checks after restructuring. |
| **Altering Dataset or Results** | CRITICAL | Keep `data/testsets/amnesty_qa_eval.json` and all evaluation scorecards byte-exact. |

---

## 7. Bugs & Oddities Noticed (Documented, NOT Modified Without Approval)

1. **`src/chain.py` Duplicate Class Header**: Lines 138-145 contains an incomplete class header that gets overridden by the full class at line 158.
2. **`src/config.py` OpenRouter Key Environment Variable**: Line 24 checks `OPEN_ROUTE_API_KEY` or `OPENROUTER_API_KEY`, but `.env.example` lists `OPEN_ROUTE_API_KEY`. It works, but has a slight spelling anomaly (`OPEN_ROUTE` vs `OPENROUTER`).
3. **`evals/benchmark_results/native_ragas_scores_groq.csv` Incomplete Sample Count**: Only 2 rows were saved, with `NaN` in `faithfulness` and `context_recall` due to Groq output formatting on native Ragas without OpenAI parser.
4. **`evals/benchmark_results/baseline_scores_amnesty_qa_eval_original4.csv`**: Contains an early 4-sample test run left untracked in `.gitignore`.
5. **`scripts/build_4phase_dashboard.py` Output Path**: Writes to root `eval_dashboard.html`, which is ignored by `.gitignore`.

---

## 8. TODOs for Step 5 (Pending User Approval)

- [ ] Create `results/` directory and safely relocate `evals/benchmark_results/*` into `results/` (or create symbolic link / alias to preserve backward compatibility).
- [ ] Create `app/` package, relocate `app.py` to `app/main.py`, and leave backward-compatible root `app.py`.
- [ ] Create `tests/` directory with automated offline test suite verifying imports, settings, memory manager, cross-encoder sigmoid math, and query transform router heuristics.
- [ ] Archive superseded scripts (`build_dashboard.py`, `generate_dashboard_html.py`) to `archive/legacy_dashboards/`.
- [ ] Create `docs/ARCHITECTURE.md` and `docs/EXPERIMENTS.md` extracting clean, rigorous documentation from `notes.md`.
- [ ] Add `LICENSE` file verification (MIT license confirmed present).
- [ ] Verify dataset licensing in `data/testsets/` (MIT / Apache / CC by `explodinggradients`).
- [ ] Remove duplicate class definition stub in `src/chain.py`.
- [ ] Run full test suite and verify imports and experiment reproducibility.
