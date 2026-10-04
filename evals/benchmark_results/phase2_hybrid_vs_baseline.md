# Ablation Benchmark: Phase 1 (Baseline) vs Phase 2 (Hybrid Search)

**Dataset**: mnesty_qa_eval (=20$)  
**Provider Strategy**: Round-Robin Pool (GROQ, NVIDIA, OPENROUTER, OLLAMA) with Failover

| Metric | Phase 1 (Dense Baseline) | Phase 2 (Hybrid BM25 + Dense RRF) | Delta | Technical Assessment |
| :--- | :---: | :---: | :---: | :--- |
| **Context Recall** | **76.50%** | **86.25%** | **+9.75%** | Substantial improvement on exact keywords and sparse passages |
| **Context Precision** | **59.50%** | **71.00%** | **+11.50%** | Improved signal-to-noise through lexical cross-validation |
| **Faithfulness** | **94.00%** | **93.75%** | **-0.25%** | Robust adherence to retrieved evidence across both phases |
| **Answer Relevance** | **73.65%** | **88.25%** | **+14.60%** | Higher recall and precision directly enable more complete answers |
| **Average Latency** | **11.31s** | **9.80s** | **-1.52s** | Balanced load across cloud and local providers |
