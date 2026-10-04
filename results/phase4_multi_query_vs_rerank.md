# Ablation Benchmark: Phase 3 (Rerank) vs Phase 4 (MULTI_QUERY Transformation)

**Dataset**: `amnesty_qa_eval` ($N=20$)  
**Provider Strategy**: Round-Robin Resilient Pool (`GROQ`, `NVIDIA`, `OPENROUTER`, `OLLAMA`)  
**Transformation Engine**: MULTI_QUERY

| Metric | Phase 3 (Standard Rerank) | Phase 4 (MULTI_QUERY Rerank) | Delta | Technical Assessment |
| :--- | :---: | :---: | :---: | :--- |
| **Context Recall** | **86.50%** | **88.00%** | **+1.50%** | Coverage across transformed candidate query pool. |
| **Context Precision** | **77.00%** | **79.25%** | **+2.25%** | Target grounding density at top-k ranks. |
| **Faithfulness** | **92.50%** | **88.75%** | **-3.75%** | Hallucination defense and context adherence. |
| **Answer Relevance** | **89.40%** | **84.90%** | **-4.50%** | Direct alignment to user intent. |
| **Average Latency** | **15.23s** | **34.85s** | **+19.62s** | End-to-end latency including transformation inference. |
