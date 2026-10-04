### Baseline RAG Evaluation Results

| Metric | Score | Target in Future Stages |
| :--- | :---: | :--- |
| **Context Recall** | **46.2%** | ⬆ Will improve with **Hybrid Search (BM25)** |
| **Context Precision** | **57.5%** | ⬆ Will improve with **Cross-Encoder Reranker** |
| **Faithfulness** | **100.0%** | ⬆ Reduces hallucinations with strict chunk pruning |
| **Answer Relevance** | **80.0%** | ⬆ Will improve with **Query Expansion (HyDE)** |
| **Avg Query Latency** | **10.35s** | Optimize caching & token throughput |
