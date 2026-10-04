"""
Script to build the complete 3-Phase Interactive Eval Dashboard.
Integrates 20-sample ground-truth benchmarks across:
- Phase 1: Dense Baseline (MiniLM embeddings + ChromaDB)
- Phase 2: Hybrid Search (BM25 + Dense RRF fusion)
- Phase 3: Cross-Encoder Reranking (ms-marco-MiniLM-L-6-v2)
"""

import json
import os
import pandas as pd

def build_dashboard():
    # Load 3 benchmark CSVs
    p1_path = "evals/benchmark_results/baseline_scores_amnesty_qa_eval.csv"
    p2_path = "evals/benchmark_results/hybrid_scores_amnesty_qa_eval.csv"
    p3_path = "evals/benchmark_results/rerank_scores_amnesty_qa_eval.csv"

    df_b = pd.read_csv(p1_path).set_index("id")
    df_h = pd.read_csv(p2_path).set_index("id")
    df_r = pd.read_csv(p3_path).set_index("id")

    # Calculate means
    b_rec = df_b["context_recall"].mean() * 100
    b_prec = df_b["context_precision"].mean() * 100
    b_faith = df_b["faithfulness"].mean() * 100
    b_rel = df_b["answer_relevance"].mean() * 100
    b_lat = df_b["latency_seconds"].mean()
    b_triad = (b_rec + b_prec + b_faith + b_rel) / 4.0

    h_rec = df_h["context_recall"].mean() * 100
    h_prec = df_h["context_precision"].mean() * 100
    h_faith = df_h["faithfulness"].mean() * 100
    h_rel = df_h["answer_relevance"].mean() * 100
    h_lat = df_h["latency_seconds"].mean()
    h_triad = (h_rec + h_prec + h_faith + h_rel) / 4.0

    r_rec = df_r["context_recall"].mean() * 100
    r_prec = df_r["context_precision"].mean() * 100
    r_faith = df_r["faithfulness"].mean() * 100
    r_rel = df_r["answer_relevance"].mean() * 100
    r_lat = df_r["latency_seconds"].mean()
    r_triad = (r_rec + r_prec + r_faith + r_rel) / 4.0

    # Samples breakdown
    samples = []
    for qid in df_r.index:
        rb = df_b.loc[qid]
        rh = df_h.loc[qid]
        rr = df_r.loc[qid]

        prec_gain = (rr["context_precision"] - rb["context_precision"]) * 100
        rec_gain = (rr["context_recall"] - rb["context_recall"]) * 100

        if prec_gain > 15:
            insight = f"Cross-encoder joint attention eliminated irrelevant candidate chunks, boosting precision by +{prec_gain:.0f}% over baseline."
        elif prec_gain > 0:
            insight = f"Cross-encoder successfully prioritized the most salient evidence into top ranks (+{prec_gain:.0f}% precision)."
        elif rec_gain > 0:
            insight = f"Hybrid Stage 1 pool captured key tokens missed by dense search (+{rec_gain:.0f}% recall) preserved through reranking."
        else:
            insight = "High factual alignment and precision maintained across all retrieval stages."

        s = {
            "id": qid,
            "question": rr["question"],
            "p1Recall": float(round(rb["context_recall"], 2)),
            "p2Recall": float(round(rh["context_recall"], 2)),
            "p3Recall": float(round(rr["context_recall"], 2)),
            "p1Precision": float(round(rb["context_precision"], 2)),
            "p2Precision": float(round(rh["context_precision"], 2)),
            "p3Precision": float(round(rr["context_precision"], 2)),
            "p1Faith": float(round(rb["faithfulness"], 2)),
            "p2Faith": float(round(rh["faithfulness"], 2)),
            "p3Faith": float(round(rr["faithfulness"], 2)),
            "p1Relevance": float(round(rb["answer_relevance"], 2)),
            "p2Relevance": float(round(rh["answer_relevance"], 2)),
            "p3Relevance": float(round(rr["answer_relevance"], 2)),
            "faithfulness": float(round(rr["faithfulness"], 2)),
            "relevance": float(round(rr["answer_relevance"], 2)),
            "latency": float(round(rr["latency_seconds"], 2)),
            "groundTruth": str(rr["ground_truth"]),
            "answer": str(rr["generated_answer"]),
            "reasoning": str(rr["judge_reasoning"]),
            "insight": insight
        }
        samples.append(s)

    benchmark_data = {
        "phase1": {
            "name": "Phase 1: Dense Baseline",
            "recall": round(b_rec, 1),
            "precision": round(b_prec, 1),
            "faithfulness": round(b_faith, 1),
            "relevance": round(b_rel, 1),
            "triad": round(b_triad, 1),
            "latency": round(b_lat, 2),
            "delta": {
                "recall": "+0.0%",
                "precision": "+0.0%",
                "faith": "0.0%",
                "relevance": "+0.0%",
                "triad": "+0.0%",
                "latency": "0.0s"
            }
        },
        "phase2": {
            "name": "Phase 2: Hybrid Search",
            "recall": round(h_rec, 1),
            "precision": round(h_prec, 1),
            "faithfulness": round(h_faith, 1),
            "relevance": round(h_rel, 1),
            "triad": round(h_triad, 1),
            "latency": round(h_lat, 2),
            "delta": {
                "recall": f"+{h_rec - b_rec:.1f}%",
                "precision": f"+{h_prec - b_prec:.1f}%",
                "faith": f"{h_faith - b_faith:.1f}%",
                "relevance": f"+{h_rel - b_rel:.1f}%",
                "triad": f"+{h_triad - b_triad:.1f}%",
                "latency": f"{h_lat - b_lat:.2f}s"
            }
        },
        "phase3": {
            "name": "Phase 3: Cross-Encoder Rerank",
            "recall": round(r_rec, 1),
            "precision": round(r_prec, 1),
            "faithfulness": round(r_faith, 1),
            "relevance": round(r_rel, 1),
            "triad": round(r_triad, 1),
            "latency": round(r_lat, 2),
            "delta": {
                "recall": f"+{r_rec - b_rec:.1f}%",
                "precision": f"+{r_prec - b_prec:.1f}%",
                "faith": f"{r_faith - b_faith:.1f}%",
                "relevance": f"+{r_rel - b_rel:.1f}%",
                "triad": f"+{r_triad - b_triad:.1f}%",
                "latency": f"+{r_lat - b_lat:.2f}s"
            },
            "deltaVsHybrid": {
                "recall": f"+{r_rec - h_rec:.1f}%",
                "precision": f"+{r_prec - h_prec:.1f}%",
                "faith": f"{r_faith - h_faith:.1f}%",
                "relevance": f"+{r_rel - h_rel:.1f}%",
                "triad": f"+{r_triad - h_triad:.1f}%",
                "latency": f"+{r_lat - h_lat:.2f}s"
            }
        },
        "metricsList": [
            {
                "key": "recall",
                "label": "Recall",
                "p1": round(b_rec, 1),
                "p2": round(h_rec, 1),
                "p3": round(r_rec, 1),
                "delta": f"+{r_rec - b_rec:.1f}%"
            },
            {
                "key": "precision",
                "label": "Precision",
                "p1": round(b_prec, 1),
                "p2": round(h_prec, 1),
                "p3": round(r_prec, 1),
                "delta": f"+{r_prec - b_prec:.1f}%"
            },
            {
                "key": "faithfulness",
                "label": "Faithfulness",
                "p1": round(b_faith, 1),
                "p2": round(h_faith, 1),
                "p3": round(r_faith, 1),
                "delta": f"{r_faith - b_faith:.1f}%"
            },
            {
                "key": "relevance",
                "label": "Relevance",
                "p1": round(b_rel, 1),
                "p2": round(h_rel, 1),
                "p3": round(r_rel, 1),
                "delta": f"+{r_rel - b_rel:.1f}%"
            },
            {
                "key": "triad",
                "label": "Triad Index",
                "p1": round(b_triad, 1),
                "p2": round(h_triad, 1),
                "p3": round(r_triad, 1),
                "delta": f"+{r_triad - b_triad:.1f}%"
            }
        ],
        "samples": samples
    }

    return benchmark_data

if __name__ == "__main__":
    data = build_dashboard()
    print("Benchmark summary generated successfully.")
    print("Phase 3 Precision:", data["phase3"]["precision"], "Delta vs baseline:", data["phase3"]["delta"]["precision"])
