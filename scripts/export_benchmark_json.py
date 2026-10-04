import json
import pandas as pd
from pathlib import Path

# Paths
ROOT = Path(__file__).resolve().parent.parent
BENCH_DIR = ROOT / "results" if (ROOT / "results").exists() else ROOT / "evals" / "benchmark_results"

p1_path = BENCH_DIR / "baseline_scores_amnesty_qa_eval.csv"
p2_path = BENCH_DIR / "hybrid_scores_amnesty_qa_eval.csv"
p3_path = BENCH_DIR / "rerank_scores_amnesty_qa_eval.csv"
p4_path = BENCH_DIR / "rerank_scores_multi_query_amnesty_qa_eval.csv"

df1 = pd.read_csv(p1_path)
df2 = pd.read_csv(p2_path)
df3 = pd.read_csv(p3_path)
df4 = pd.read_csv(p4_path)

# Calculate phase metrics
def calc_metrics(df):
    rec = round(df["context_recall"].mean() * 100, 1)
    prec = round(df["context_precision"].mean() * 100, 1)
    faith = round(df["faithfulness"].mean() * 100, 1)
    rel = round(df["answer_relevance"].mean() * 100, 1)
    triad = round((rec + prec + faith + rel) / 4.0, 1)
    lat = round(df["latency_seconds"].mean(), 2)
    return {
        "recall": rec,
        "precision": prec,
        "faithfulness": faith,
        "relevance": rel,
        "triad": triad,
        "latency": lat
    }

m1 = calc_metrics(df1)
m2 = calc_metrics(df2)
m3 = calc_metrics(df3)
m4 = calc_metrics(df4)

# Create 20 samples list
samples = []
for i in range(len(df1)):
    row1 = df1.iloc[i]
    row2 = df2.iloc[i]
    row3 = df3.iloc[i]
    row4 = df4.iloc[i]
    
    qid = row1["id"]
    q_text = row1["question"]
    gt = row1.get("ground_truth", "")
    ans4 = row4.get("generated_answer", "")
    reason4 = row4.get("judge_reasoning", "")
    
    samples.append({
        "id": qid,
        "question": q_text,
        "p1Recall": float(row1["context_recall"]),
        "p2Recall": float(row2["context_recall"]),
        "p3Recall": float(row3["context_recall"]),
        "p4Recall": float(row4["context_recall"]),
        "p1Precision": float(row1["context_precision"]),
        "p2Precision": float(row2["context_precision"]),
        "p3Precision": float(row3["context_precision"]),
        "p4Precision": float(row4["context_precision"]),
        "p1Faith": float(row1["faithfulness"]),
        "p2Faith": float(row2["faithfulness"]),
        "p3Faith": float(row3["faithfulness"]),
        "p4Faith": float(row4["faithfulness"]),
        "p1Relevance": float(row1["answer_relevance"]),
        "p2Relevance": float(row2["answer_relevance"]),
        "p3Relevance": float(row3["answer_relevance"]),
        "p4Relevance": float(row4["answer_relevance"]),
        "latency": round(float(row4["latency_seconds"]), 2),
        "groundTruth": gt,
        "answer": ans4,
        "reasoning": reason4,
        "insight": "Multi-query decomposition broke compound clauses into targeted sub-queries, lifting context precision and recall."
    })

benchmark_data = {
    "phase1": {
        "name": "Phase 1: Dense Baseline",
        **m1,
        "delta": {"recall": "+0.0%", "precision": "+0.0%", "faith": "0.0%", "relevance": "+0.0%", "triad": "+0.0%", "latency": "0.0s"}
    },
    "phase2": {
        "name": "Phase 2: Hybrid Search (BM25+RRF)",
        **m2,
        "delta": {
            "recall": f"+{m2['recall'] - m1['recall']:.1f}%",
            "precision": f"+{m2['precision'] - m1['precision']:.1f}%",
            "faith": f"{m2['faithfulness'] - m1['faithfulness']:+.1f}%",
            "relevance": f"+{m2['relevance'] - m1['relevance']:.1f}%",
            "triad": f"+{m2['triad'] - m1['triad']:.1f}%",
            "latency": f"{m2['latency'] - m1['latency']:+.2f}s"
        }
    },
    "phase3": {
        "name": "Phase 3: Cross-Encoder Rerank",
        **m3,
        "delta": {
            "recall": f"+{m3['recall'] - m1['recall']:.1f}%",
            "precision": f"+{m3['precision'] - m1['precision']:.1f}%",
            "faith": f"{m3['faithfulness'] - m1['faithfulness']:+.1f}%",
            "relevance": f"+{m3['relevance'] - m1['relevance']:.1f}%",
            "triad": f"+{m3['triad'] - m1['triad']:.1f}%",
            "latency": f"{m3['latency'] - m1['latency']:+.2f}s"
        }
    },
    "phase4": {
        "name": "Phase 4: Multi-Query Transformation",
        **m4,
        "delta": {
            "recall": f"+{m4['recall'] - m1['recall']:.1f}%",
            "precision": f"+{m4['precision'] - m1['precision']:.1f}%",
            "faith": f"{m4['faithfulness'] - m1['faithfulness']:+.1f}%",
            "relevance": f"+{m4['relevance'] - m1['relevance']:.1f}%",
            "triad": f"+{m4['triad'] - m1['triad']:.1f}%",
            "latency": f"{m4['latency'] - m1['latency']:+.2f}s"
        }
    },
    "metricsList": [
        {"key": "recall", "label": "Recall", "p1": m1["recall"], "p2": m2["recall"], "p3": m3["recall"], "p4": m4["recall"], "delta": f"+{m4['recall'] - m1['recall']:.1f}%"},
        {"key": "precision", "label": "Precision", "p1": m1["precision"], "p2": m2["precision"], "p3": m3["precision"], "p4": m4["precision"], "delta": f"+{m4['precision'] - m1['precision']:.1f}%"},
        {"key": "faithfulness", "label": "Faithfulness", "p1": m1["faithfulness"], "p2": m2["faithfulness"], "p3": m3["faithfulness"], "p4": m4["faithfulness"], "delta": f"{m4['faithfulness'] - m1['faithfulness']:+.1f}%"},
        {"key": "relevance", "label": "Relevance", "p1": m1["relevance"], "p2": m2["relevance"], "p3": m3["relevance"], "p4": m4["relevance"], "delta": f"+{m4['relevance'] - m1['relevance']:.1f}%"},
        {"key": "triad", "label": "Triad Index", "p1": m1["triad"], "p2": m2["triad"], "p3": m3["triad"], "p4": m4["triad"], "delta": f"+{m4['triad'] - m1['triad']:.1f}%"}
    ],
    "samples": samples
}

with open(ROOT / "scripts" / "benchmark_data.json", "w", encoding="utf-8") as f:
    json.dump(benchmark_data, f, indent=2)

print("Benchmark data JSON exported successfully!")
