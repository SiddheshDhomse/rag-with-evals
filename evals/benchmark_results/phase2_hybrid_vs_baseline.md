# Ablation Benchmark: Phase 1 (Baseline) vs Phase 2 (Hybrid Search)

**Dataset**: `amnesty_qa_eval` ($N=4$)  
**Evaluator**: LLM-as-a-Judge (`GROQ`)

| Metric | Phase 1 (Dense Baseline) | Phase 2 (Hybrid BM25 + Dense RRF) | Delta | Assessment |
| :--- | :---: | :---: | :---: | :--- |
| **Context Recall** | **46.25%** | **83.75%** | **+37.50%** | Substantial improvement on exact keywords and sparse passages |
| **Context Precision** | **57.50%** | **85.00%** | **+27.50%** | Improved signal-to-noise through lexical cross-validation |
| **Faithfulness** | **100.00%** | **100.00%** | **+0.00%** | Preserves 100% adherence to retrieved evidence |
| **Answer Relevance** | **80.00%** | **93.75%** | **+13.75%** | Higher recall directly enables more complete answers |
| **Average Latency** | **10.35s** | **2.33s** | **-8.02s** | Negligible overhead for in-memory BM25 index |
