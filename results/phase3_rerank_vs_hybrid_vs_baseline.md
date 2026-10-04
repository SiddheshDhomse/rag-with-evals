# Ablation Benchmark: Phase 1 vs Phase 2 vs Phase 3

**Dataset**: `amnesty_qa_eval` ($N=20$)  
**Provider Strategy**: Round-Robin Resilient Pool (`GROQ`, `NVIDIA`, `OPENROUTER`, `OLLAMA`)  
**Reranker Engine**: Cross-Encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`)

| Metric | Phase 1 (Dense Baseline) | Phase 2 (Hybrid BM25+RRF) | Phase 3 (Cross-Encoder Rerank) | Δ vs Baseline | Δ vs Hybrid | Technical Assessment |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Context Recall** | **76.50%** | **86.25%** | **86.50%** | **+10.00%** | **+0.25%** | High recall preserved from Stage 1 candidate pool. |
| **Context Precision** | **59.50%** | **71.00%** | **77.00%** | **+17.50%** | **+6.00%** | Deep cross-attention elevates vital facts to top ranks. |
| **Faithfulness** | **94.00%** | **93.75%** | **92.50%** | **-1.50%** | **-1.25%** | Near-zero hallucination drift. |
| **Answer Relevance** | **73.65%** | **88.25%** | **89.40%** | **+15.75%** | **+1.15%** | High signal-to-noise fuels precise answer synthesis. |
| **Average Latency** | **11.31s** | **9.80s** | **15.23s** | **+3.92s** | **+5.44s** | Lightweight cross-encoder inference on CPU. |
