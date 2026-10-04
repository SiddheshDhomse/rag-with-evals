# Security & Confidentiality Audit Report

**Date**: 2026-10-04  
**Project**: RAG Evaluation Suite (`RAG with Evals`)  
**Auditor**: Senior Engineering Cleanup Agent  
**Branch**: `cleanup/structure-and-docs`  
**Base Commit SHA**: `43b53a60a7e6ec35b671a533036502aa8784d0fe`  
**Tag**: `pre-cleanup`  

---

## 1. Executive Summary

A comprehensive, read-only security audit was conducted across the entire repository, including tracked files, ignored files, local working directories, and complete Git revision history (17 commits across all branches).

**Result**: **PASSED (CLEAN FOR PUBLIC RELEASE)**.
- **Zero** API keys, credentials, or private tokens exist in tracked files.
- **Zero** credentials exist in Git commit history.
- **Zero** private database connection strings or internal corporate URLs were found.
- **Zero** customer data, PII, confidential contracts, or non-public datasets exist in the repository.
- Local secrets are strictly confined to an uncommitted, git-ignored `.env` file.

---

## 2. Scope & Methodology

The security audit inspected:
1. **Tracked Files**: All files tracked by Git (`git ls-files`).
2. **Git Commit History**: Every commit, commit message, patch diff, and historical file revision (`git log -p --all`).
3. **Environment Files**: `.env`, `.env.example`, `.env.*`.
4. **Ignored Files**: All files excluded by `.gitignore` (`git status --ignored`), including `storage/`, HTML visualizers, and `.pyc` caches.
5. **Evaluation Artifacts**: Ground-truth datasets, Ragas evaluation logs, LLM-as-a-judge scorecards, and benchmark summaries.
6. **Codebase AST & Regex Patterns**: Automated regex pattern matching for credential prefixes:
   - OpenAI / OpenRouter (`sk-...`, `sk-or-...`)
   - Groq Cloud (`gsk_...`)
   - NVIDIA NIM (`nvapi-...`)
   - GitHub Personal Access Tokens (`ghp_...`, `gho_...`)
   - AWS Access Keys (`AKIA...`, Secret keys)
   - JWT tokens (`eyJ...`)
   - Private Keys (`BEGIN RSA/EC/DSA/OPENSSH PRIVATE KEY`)
   - Generic API keys, passwords, and authorization headers (`api_key = "..."`, `Bearer ...`)

---

## 3. Findings Matrix

| Category | Status | Details / Location |
| :--- | :---: | :--- |
| **Active API Credentials in Tracked Code** | **NONE** | No active keys found in tracked files. `.env.example` contains only standard dummy placeholder strings (e.g. `your_groq_api_key_here`). |
| **Historical Credentials in Git Log** | **NONE** | Full history regex scan across all commits (`git log -p --all`) revealed zero real secret commits. Historical diffs for `README.md` contained only syntax documentation examples (e.g., `GROQ_API_KEY=gsk_...`). |
| **Local Working Tree Secrets** | **ISOLATED** | Local `.env` contains developer keys (`GROQ_API_KEY`, `NVIDIA_API_KEY`, `OPEN_ROUTE_API_KEY`). Strictly ignored by `.gitignore` and has never been staged. |
| **Private Keys & Certificates** | **NONE** | No `.key`, `.pem`, `.crt`, `.pfx`, or SSH private keys found anywhere in repo. |
| **Database Connection Strings** | **NONE** | No connection URI strings (`postgres://`, `mongodb://`, `mysql://`, `redis://`). ChromaDB is purely local file-based SQLite (`storage/chroma`), properly ignored. |
| **Internal Endpoints & Hostnames** | **NONE** | Only standard public external endpoints: `https://api.groq.com/openai/v1`, `https://integrate.api.nvidia.com/v1`, `https://openrouter.ai/api/v1`, and default localhost `http://localhost:11434`. |
| **PII & Confidential Data** | **NONE** | No customer data, employee data, or confidential enterprise documents. |
| **Evaluation Testset Licensing** | **PUBLIC** | Testset `data/testsets/amnesty_qa_eval.json` is derived entirely from the public `explodinggradients/amnesty_qa` benchmark (CC / public domain Amnesty International human rights reports). |
| **Jupyter Notebook Secrets** | **NONE** | No `.ipynb` files exist in the repository. |
| **Binary & Storage Artifacts** | **ISOLATED** | `storage/` directory contains local Chroma SQLite DB and chat session JSON files (`storage/history/*.json`). All properly ignored by `.gitignore`. |

---

## 4. Detailed Findings by Area

### 4.1 Environment Configuration (`.env` / `.env.example`)
- `.env.example` is tracked in Git. It was verified to contain only environment variable names and generic placeholder strings (`your_groq_api_key_here`, `your_nvidia_api_key_here`, `your_openrouter_api_key_here`).
- `.env` contains local API keys for Groq, NVIDIA NIM, and OpenRouter. It is excluded by `.gitignore` (line 4) and has never been committed.

### 4.2 Git Revision History
- Commits checked: 17 commits (`7e2b8d4` through `43b53a6`).
- Pattern scans:
  - `gsk_`: Occurred only as placeholder documentation in `README.md` (`GROQ_API_KEY=gsk_...`).
  - `nvapi-`: Occurred only as placeholder documentation in `README.md` (`NVIDIA_API_KEY=nvapi-...`).
  - `sk-`: Occurred only as placeholder documentation in `README.md` (`OPEN_ROUTE_API_KEY=sk-or-your_openrouter_api_key`).
  - `eyJ`: Zero occurrences.
  - `AKIA`: Zero occurrences.
  - `PRIVATE KEY`: Zero occurrences.

### 4.3 Evaluation & Benchmark Datasets
- Dataset location: `data/testsets/amnesty_qa_eval.json`.
- Contents: 20 multi-context QA items evaluating publicly available Amnesty International reports (climate emissions, corporate governance, civil liberties). No proprietary corporate secrets or confidential legal documents exist.
- Results location: `evals/benchmark_results/*.csv`. Contains evaluation logs, model answer strings, and judge reasoning. No leaked credentials or private data present.

### 4.4 Local Standalone & Ignored Files
- The following local files are ignored by `.gitignore`:
  - `eval_dashboard.html`, `eval_dashboard_v2.html`, `rag-pipeline-animation (4).html`
  - `build_index.py`, `index.html`, `samples_extracted.js`, `test_script.js`
  - `evals/benchmark_results/baseline_scores.csv`
  - `evals/benchmark_results/baseline_scores_amnesty_qa_eval_original4.csv`
- None of these files contain secret keys or sensitive private data.

---

## 5. Public-Release Assessment

| Dimension | Assessment | Recommendation |
| :--- | :--- | :--- |
| **Code Sanitization** | **CLEAN** | Safe for public release. |
| **Git History Sanitization** | **CLEAN** | No history rewriting required. |
| **Data Licensing** | **PUBLIC** | Amnesty QA is standard open benchmark data. |
| **Dependency Security** | **LOW RISK** | Standard pinned packages in `requirements.txt`. |

**Overall Recommendation**: The repository is safe for public distribution, open-source hosting on GitHub, or external evaluation. No credentials require rotation due to repository tracking.

---

## 6. Actionable Hygiene Recommendations

1. **Maintain Git Ignore Vigilance**: Preserve `.env` in `.gitignore` at all times.
2. **Pre-commit Hooks**: Recommend adding a secret scanning hook (e.g. `detect-secrets` or `gitleaks`) for local developer environments.
3. **Template Sync**: Whenever new providers or model settings are added to `src/config.py`, update `.env.example` with blank or descriptive placeholder values only.
