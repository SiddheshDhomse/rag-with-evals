"""
Generator script to compile the complete, self-contained 3-Phase Interactive Eval Dashboard.
Outputs to:
- eval_dashboard.html
- Artifact directory
"""

import json
import os
import shutil
import pandas as pd
from build_dashboard import build_dashboard

def generate_html():
    benchmark_data = build_dashboard()
    benchmark_json = json.dumps(benchmark_data, indent=2)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>RAG Evaluation Suite | 3-Phase Diagnostics & Ablation Analytics</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
  <style>
    :root {{
      --bg: #f8fafc;
      --surface: #ffffff;
      --surface-subtle: #f1f5f9;
      --border: #e2e8f0;
      --border-subtle: #edf2f7;
      
      --text-main: #090d16;
      --text-muted: #64748b;
      --text-dim: #94a3b8;

      --primary: #4f46e5;
      --primary-hover: #4338ca;
      --primary-light: #eef2ff;
      --primary-glow: rgba(79, 70, 229, 0.15);

      --emerald: #059669;
      --emerald-light: #ecfdf5;
      --emerald-border: #a7f3d0;

      --blue: #2563eb;
      --blue-light: #eff6ff;

      --purple: #7c3aed;
      --purple-light: #f5f3ff;

      --amber: #d97706;
      --amber-light: #fffbeb;

      --slate-dark: #0f172a;
      
      --radius-sm: 8px;
      --radius-md: 12px;
      --radius-lg: 16px;
      --radius-xl: 20px;

      --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.04);
      --shadow-card: 0 1px 3px rgba(15, 23, 42, 0.04), 0 6px 16px rgba(15, 23, 42, 0.02);
      --shadow-hover: 0 4px 6px -1px rgba(0, 0, 0, 0.06), 0 10px 24px -3px rgba(0, 0, 0, 0.05);
    }}

    * {{
      box-sizing: border-box;
      margin: 0;
      padding: 0;
      font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
      -webkit-font-smoothing: antialiased;
    }}

    body {{
      background-color: var(--bg);
      color: var(--text-main);
      padding: 32px 24px;
      min-height: 100vh;
    }}

    .container {{
      max-width: 1440px;
      margin: 0 auto;
    }}

    /* Top Navigation Bar */
    .topbar {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 28px;
      background: var(--surface);
      padding: 16px 24px;
      border-radius: var(--radius-xl);
      border: 1px solid var(--border);
      box-shadow: var(--shadow-sm);
    }}

    .brand-section {{
      display: flex;
      align-items: center;
      gap: 14px;
    }}

    .brand-logo {{
      width: 42px;
      height: 42px;
      background: linear-gradient(135deg, #059669 0%, #10b981 100%);
      border-radius: 12px;
      display: flex;
      align-items: center;
      justify-content: center;
      color: white;
      box-shadow: 0 4px 12px rgba(16, 185, 129, 0.28);
    }}

    .brand-title-wrap {{
      display: flex;
      align-items: center;
      gap: 10px;
    }}

    .brand-title {{
      font-size: 19px;
      font-weight: 800;
      letter-spacing: -0.025em;
      color: var(--text-main);
    }}

    .brand-meta {{
      font-size: 12.5px;
      color: var(--text-muted);
      margin-top: 1px;
    }}

    .pill-badge {{
      font-size: 11.5px;
      font-weight: 700;
      padding: 3px 9px;
      border-radius: 20px;
      letter-spacing: -0.01em;
      transition: all 0.2s ease;
    }}

    .pill-badge.active-phase {{
      background: #ecfdf5;
      color: #065f46;
      border: 1px solid #a7f3d0;
    }}

    .nav-actions {{
      display: flex;
      align-items: center;
      gap: 10px;
    }}

    .segmented-control {{
      background: var(--surface-subtle);
      padding: 4px;
      border-radius: 12px;
      display: flex;
      gap: 4px;
    }}

    .segment-btn {{
      background: transparent;
      border: none;
      padding: 7px 14px;
      border-radius: 8px;
      font-size: 12.5px;
      font-weight: 600;
      color: var(--text-muted);
      cursor: pointer;
      transition: all 0.15s ease;
    }}

    .segment-btn:hover {{
      color: var(--text-main);
    }}

    .segment-btn.active {{
      background: var(--surface);
      color: var(--text-main);
      box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }}

    .btn-run {{
      background: var(--text-main);
      color: white;
      border: none;
      padding: 8px 18px;
      border-radius: 10px;
      font-size: 12.5px;
      font-weight: 600;
      cursor: pointer;
      display: flex;
      align-items: center;
      gap: 6px;
      transition: all 0.2s;
    }}

    .btn-run:hover {{
      background: #1e293b;
      transform: translateY(-1px);
    }}

    /* Sub-header / System Status Strip */
    .system-strip {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 12px 20px;
      margin-bottom: 24px;
      background: #ffffff;
      border-radius: var(--radius-md);
      border: 1px solid var(--border);
      font-size: 12.5px;
      color: var(--text-muted);
    }}

    .strip-items {{
      display: flex;
      align-items: center;
      gap: 20px;
    }}

    .strip-item {{
      display: flex;
      align-items: center;
      gap: 6px;
    }}

    .strip-dot {{
      width: 7px;
      height: 7px;
      border-radius: 50%;
      background: var(--emerald);
    }}

    /* 6 Metric Cards Row */
    .metric-grid {{
      display: grid;
      grid-template-columns: repeat(6, 1fr);
      gap: 16px;
      margin-bottom: 24px;
    }}

    .metric-card {{
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--radius-lg);
      padding: 20px;
      box-shadow: var(--shadow-card);
      display: flex;
      flex-direction: column;
      justify-content: space-between;
      transition: all 0.2s ease;
      position: relative;
    }}

    .metric-card:hover {{
      transform: translateY(-2px);
      box-shadow: var(--shadow-hover);
    }}

    .metric-card-top {{
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      margin-bottom: 14px;
    }}

    .metric-label {{
      font-size: 11px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      color: var(--text-muted);
    }}

    .metric-icon {{
      width: 28px;
      height: 28px;
      border-radius: 8px;
      display: flex;
      align-items: center;
      justify-content: center;
    }}

    .metric-val-wrap {{
      display: flex;
      align-items: baseline;
      gap: 8px;
      margin-bottom: 10px;
    }}

    .metric-number {{
      font-size: 29px;
      font-weight: 800;
      letter-spacing: -0.03em;
      color: var(--text-main);
      line-height: 1;
    }}

    .metric-trend {{
      font-size: 11.5px;
      font-weight: 700;
      padding: 2px 7px;
      border-radius: 6px;
    }}

    .trend-pos {{
      background: #ecfdf5;
      color: #047857;
    }}

    .trend-neutral {{
      background: #f1f5f9;
      color: #64748b;
    }}

    .metric-bar-bg {{
      height: 5px;
      background: #f1f5f9;
      border-radius: 3px;
      overflow: hidden;
      margin-bottom: 8px;
    }}

    .metric-bar-fill {{
      height: 100%;
      border-radius: 3px;
      transition: width 0.7s cubic-bezier(0.16, 1, 0.3, 1);
    }}

    .metric-desc {{
      font-size: 11.5px;
      color: var(--text-muted);
      line-height: 1.35;
    }}

    /* Dark Triad Card */
    .metric-card.hero {{
      background: linear-gradient(145deg, #0f172a 0%, #064e3b 100%);
      border-color: #047857;
      color: white;
    }}

    .metric-card.hero .metric-label {{
      color: #a7f3d0;
    }}

    .metric-card.hero .metric-number {{
      color: #ffffff;
    }}

    .metric-card.hero .metric-desc {{
      color: #cbd5e1;
    }}

    .metric-card.hero .metric-bar-bg {{
      background: rgba(255, 255, 255, 0.12);
    }}

    .metric-card.hero .metric-bar-fill {{
      background: #34d399;
    }}

    /* Analytics Section: Radar + Bar Comparison */
    .analytics-grid {{
      display: grid;
      grid-template-columns: 460px 1fr;
      gap: 20px;
      margin-bottom: 24px;
    }}

    .chart-panel {{
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--radius-lg);
      padding: 24px;
      box-shadow: var(--shadow-card);
      display: flex;
      flex-direction: column;
    }}

    .panel-top {{
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      margin-bottom: 20px;
    }}

    .panel-heading {{
      font-size: 16px;
      font-weight: 800;
      letter-spacing: -0.015em;
      color: var(--text-main);
    }}

    .panel-sub {{
      font-size: 12.5px;
      color: var(--text-muted);
      margin-top: 2px;
    }}

    .legend-row {{
      display: flex;
      gap: 14px;
      font-size: 11.5px;
      font-weight: 600;
      color: var(--text-muted);
    }}

    .legend-chip {{
      display: flex;
      align-items: center;
      gap: 6px;
    }}

    .legend-marker {{
      width: 10px;
      height: 10px;
      border-radius: 3px;
    }}

    /* Radar Canvas Container */
    .radar-wrapper {{
      position: relative;
      display: flex;
      justify-content: center;
      align-items: center;
      height: 350px;
      background: radial-gradient(circle at center, rgba(5, 150, 105, 0.04) 0%, transparent 70%);
      border-radius: var(--radius-md);
    }}

    #radarCanvas {{
      width: 440px;
      height: 340px;
    }}

    /* Bar Graph Container */
    .bar-chart-wrapper {{
      display: flex;
      flex-direction: column;
      justify-content: space-between;
      height: 350px;
      padding-top: 10px;
    }}

    .bar-columns {{
      display: grid;
      grid-template-columns: repeat(5, 1fr);
      gap: 16px;
      height: 280px;
      align-items: flex-end;
      border-bottom: 1px solid var(--border);
      padding-bottom: 16px;
    }}

    .bar-group {{
      display: flex;
      flex-direction: column;
      align-items: center;
      height: 100%;
      justify-content: flex-end;
      position: relative;
    }}

    .bar-delta-tag {{
      font-size: 10.5px;
      font-weight: 700;
      padding: 2px 6px;
      border-radius: 4px;
      margin-bottom: 8px;
      background: #ecfdf5;
      color: #047857;
      white-space: nowrap;
    }}

    .bar-pair {{
      display: flex;
      align-items: flex-end;
      gap: 6px;
      height: 200px;
      width: 100%;
      justify-content: center;
    }}

    .bar-stem {{
      width: 22px;
      border-radius: 5px 5px 0 0;
      transition: height 0.8s cubic-bezier(0.16, 1, 0.3, 1);
      position: relative;
      display: flex;
      justify-content: center;
    }}

    .bar-stem.baseline {{
      background: #cbd5e1;
    }}

    .bar-stem.hybrid {{
      background: #818cf8;
    }}

    .bar-stem.rerank {{
      background: linear-gradient(180deg, #059669 0%, #10b981 100%);
      box-shadow: 0 4px 10px rgba(16, 185, 129, 0.28);
    }}

    .bar-val-text {{
      position: absolute;
      top: -20px;
      font-size: 10.5px;
      font-weight: 700;
      color: var(--text-main);
      font-family: 'JetBrains Mono', monospace;
      white-space: nowrap;
    }}

    .bar-label-caption {{
      margin-top: 10px;
      font-size: 11.5px;
      font-weight: 700;
      color: var(--text-muted);
      text-align: center;
    }}

    /* Roadmap Timeline */
    .roadmap-section {{
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--radius-lg);
      padding: 22px 24px;
      box-shadow: var(--shadow-card);
      margin-bottom: 24px;
    }}

    .roadmap-timeline {{
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 16px;
      margin-top: 16px;
    }}

    .roadmap-node {{
      border: 1px solid var(--border);
      border-radius: var(--radius-md);
      padding: 16px;
      background: #f8fafc;
      transition: all 0.2s;
    }}

    .roadmap-node.done {{
      border-color: #a7f3d0;
      background: #f0fdf4;
    }}

    .roadmap-node.active {{
      border-color: #34d399;
      background: #ecfdf5;
      box-shadow: 0 0 0 3px rgba(16, 185, 129, 0.12);
    }}

    .roadmap-node.upcoming {{
      border-style: dashed;
      background: #ffffff;
      opacity: 0.75;
    }}

    .rn-tag {{
      display: inline-block;
      font-size: 10.5px;
      font-weight: 700;
      text-transform: uppercase;
      padding: 2px 7px;
      border-radius: 4px;
      margin-bottom: 8px;
    }}

    .rn-title {{
      font-size: 13.5px;
      font-weight: 700;
      color: var(--text-main);
      margin-bottom: 4px;
    }}

    .rn-desc {{
      font-size: 11.5px;
      color: var(--text-muted);
      line-height: 1.4;
    }}

    /* Diagnostics Table Section */
    .table-card {{
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--radius-lg);
      padding: 24px;
      box-shadow: var(--shadow-card);
    }}

    .diagnostic-table {{
      width: 100%;
      border-collapse: collapse;
      margin-top: 14px;
    }}

    .diagnostic-table th {{
      text-align: left;
      padding: 12px 14px;
      background: #f8fafc;
      color: var(--text-muted);
      font-size: 11px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      border-bottom: 1px solid var(--border);
    }}

    .diagnostic-table td {{
      padding: 14px;
      border-bottom: 1px solid var(--border-subtle);
      font-size: 13px;
      vertical-align: middle;
    }}

    .diagnostic-table tr:hover td {{
      background: #fbfcfe;
    }}

    .qid-pill {{
      font-family: 'JetBrains Mono', monospace;
      font-size: 12px;
      font-weight: 700;
      color: var(--emerald);
      background: #ecfdf5;
      padding: 3px 8px;
      border-radius: 6px;
    }}

    .q-title {{
      font-weight: 600;
      color: var(--text-main);
      max-width: 440px;
      line-height: 1.4;
    }}

    .chip {{
      font-family: 'JetBrains Mono', monospace;
      font-weight: 700;
      font-size: 11.5px;
      padding: 4px 8px;
      border-radius: 6px;
      display: inline-block;
    }}

    .chip-green {{ background: #ecfdf5; color: #047857; }}
    .chip-amber {{ background: #fffbeb; color: #b45309; }}
    .chip-blue {{ background: #eff6ff; color: #1d4ed8; }}
    .chip-purple {{ background: #f5f3ff; color: #7c3aed; }}
    .chip-slate {{ background: #f1f5f9; color: #475569; }}

    .btn-inspect {{
      background: var(--surface);
      border: 1px solid var(--border);
      padding: 6px 14px;
      border-radius: 8px;
      font-size: 12px;
      font-weight: 600;
      color: var(--text-main);
      cursor: pointer;
      transition: all 0.15s;
    }}

    .btn-inspect:hover {{
      background: var(--emerald);
      color: white;
      border-color: var(--emerald);
    }}

    /* Modern Slide-over Drawer */
    .drawer-scrim {{
      position: fixed;
      inset: 0;
      background: rgba(15, 23, 42, 0.45);
      backdrop-filter: blur(4px);
      display: none;
      justify-content: flex-end;
      z-index: 999;
    }}

    .drawer-body {{
      width: 640px;
      height: 100vh;
      background: white;
      padding: 32px;
      overflow-y: auto;
      box-shadow: -8px 0 32px rgba(15, 23, 42, 0.15);
      animation: slideIn 0.22s cubic-bezier(0.16, 1, 0.3, 1);
    }}

    @keyframes slideIn {{
      from {{ transform: translateX(100%); }}
      to {{ transform: translateX(0); }}
    }}

    .drawer-top {{
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      margin-bottom: 24px;
      padding-bottom: 16px;
      border-bottom: 1px solid var(--border);
    }}

    .drawer-close {{
      width: 32px;
      height: 32px;
      border-radius: 8px;
      border: 1px solid var(--border);
      background: #f8fafc;
      color: var(--text-muted);
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
      font-weight: 700;
    }}

    .block-label {{
      font-size: 11px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      color: var(--text-muted);
      margin-bottom: 6px;
    }}

    .block-content {{
      background: #f8fafc;
      border: 1px solid var(--border);
      border-radius: var(--radius-sm);
      padding: 14px 16px;
      font-size: 13px;
      line-height: 1.55;
      color: var(--text-main);
      margin-bottom: 20px;
    }}
  </style>
</head>
<body>

<div class="container">

  <!-- Top Navigation Bar -->
  <header class="topbar">
    <div class="brand-section">
      <div class="brand-logo">
        <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
          <circle cx="12" cy="12" r="10"></circle>
          <path d="m4.93 4.93 4.24 4.24"></path>
          <path d="m14.83 9.17 4.24-4.24"></path>
          <path d="m14.83 14.83 4.24 4.24"></path>
          <path d="m9.17 14.83-4.24 4.24"></path>
          <circle cx="12" cy="12" r="4"></circle>
        </svg>
      </div>
      <div>
        <div class="brand-title-wrap">
          <h1 class="brand-title">RAG Evaluation Suite</h1>
          <span class="pill-badge active-phase" id="topActiveBadge">Phase 3: Cross-Encoder Rerank Active</span>
        </div>
        <p class="brand-meta">Continuous Retrieval-Augmented Generation Benchmarking & Empirical Ablation</p>
      </div>
    </div>

    <div class="nav-actions">
      <div class="segmented-control">
        <button class="segment-btn" id="segPhase1" onclick="setDashboardView('phase1')">Baseline (Dense)</button>
        <button class="segment-btn" id="segPhase2" onclick="setDashboardView('phase2')">Phase 2 (Hybrid)</button>
        <button class="segment-btn active" id="segPhase3" onclick="setDashboardView('phase3')">Phase 3 (Rerank)</button>
        <button class="segment-btn" id="segDiff" onclick="setDashboardView('compare')">3-Way Overlay</button>
      </div>

      <button class="btn-run" onclick="copyCliCommand()">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg>
        Copy CLI Eval Command
      </button>
    </div>
  </header>

  <!-- System Status Strip -->
  <div class="system-strip">
    <div class="strip-items">
      <div class="strip-item">
        <div class="strip-dot"></div>
        <span>Corpus Chunks: <b>168 Docs</b></span>
      </div>
      <div class="strip-item">
        <span>Benchmark: <b>explodinggradients/amnesty_qa (v2) [N=20]</b></span>
      </div>
      <div class="strip-item">
        <span>Retriever: <b>MiniLM (Dense) + BM25Okapi + ms-marco Reranker</b></span>
      </div>
    </div>
    <div>
      Judge Engine: <b>Groq Cloud / NVIDIA Resilient Pool</b> &bull; Evaluation Temperature: <b>T = 0.0</b>
    </div>
  </div>

  <!-- Metric Cards Grid -->
  <section class="metric-grid">
    <!-- Context Recall -->
    <div class="metric-card">
      <div class="metric-card-top">
        <span class="metric-label">Context Recall</span>
        <div class="metric-icon" style="background:#ecfdf5; color:#059669;">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><circle cx="12" cy="12" r="6"></circle><circle cx="12" cy="12" r="2"></circle></svg>
        </div>
      </div>
      <div class="metric-val-wrap">
        <span class="metric-number" id="valRecall">86.5%</span>
        <span class="metric-trend trend-pos" id="trendRecall">+10.0%</span>
      </div>
      <div class="metric-bar-bg">
        <div class="metric-bar-fill" id="barRecall" style="width: 86.5%; background: #059669;"></div>
      </div>
      <p class="metric-desc">Ground-truth facts captured in top chunks</p>
    </div>

    <!-- Precision -->
    <div class="metric-card">
      <div class="metric-card-top">
        <span class="metric-label">Context Precision</span>
        <div class="metric-icon" style="background:#eff6ff; color:#2563eb;">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><polygon points="22 3 2 3 10 12.46 10 19 14 21 14 12.46 22 3"></polygon></svg>
        </div>
      </div>
      <div class="metric-val-wrap">
        <span class="metric-number" id="valPrecision">77.0%</span>
        <span class="metric-trend trend-pos" id="trendPrecision">+17.5%</span>
      </div>
      <div class="metric-bar-bg">
        <div class="metric-bar-fill" id="barPrecision" style="width: 77.0%; background: #2563eb;"></div>
      </div>
      <p class="metric-desc">Cross-attention signal density at top ranks</p>
    </div>

    <!-- Faithfulness -->
    <div class="metric-card">
      <div class="metric-card-top">
        <span class="metric-label">Faithfulness</span>
        <div class="metric-icon" style="background:#f5f3ff; color:#7c3aed;">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path></svg>
        </div>
      </div>
      <div class="metric-val-wrap">
        <span class="metric-number" id="valFaith">92.5%</span>
        <span class="metric-trend trend-neutral" id="trendFaith">-1.5%</span>
      </div>
      <div class="metric-bar-bg">
        <div class="metric-bar-fill" id="barFaith" style="width: 92.5%; background: #7c3aed;"></div>
      </div>
      <p class="metric-desc">Zero-hallucination factual grounding</p>
    </div>

    <!-- Answer Relevance -->
    <div class="metric-card">
      <div class="metric-card-top">
        <span class="metric-label">Answer Relevance</span>
        <div class="metric-icon" style="background:#fffbeb; color:#d97706;">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><polyline points="12 6 12 12 14 14"></polyline></svg>
        </div>
      </div>
      <div class="metric-val-wrap">
        <span class="metric-number" id="valRelevance">89.4%</span>
        <span class="metric-trend trend-pos" id="trendRelevance">+15.8%</span>
      </div>
      <div class="metric-bar-bg">
        <div class="metric-bar-fill" id="barRelevance" style="width: 89.4%; background: #d97706;"></div>
      </div>
      <p class="metric-desc">Direct semantic alignment to user intent</p>
    </div>

    <!-- Avg Latency -->
    <div class="metric-card">
      <div class="metric-card-top">
        <span class="metric-label">Average Latency</span>
        <div class="metric-icon" style="background:#f1f5f9; color:#475569;">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><polyline points="12 6 12 12 16 14"></polyline></svg>
        </div>
      </div>
      <div class="metric-val-wrap">
        <span class="metric-number" id="valLatency">15.23s</span>
        <span class="metric-trend trend-neutral" id="trendLatency">+3.92s</span>
      </div>
      <div class="metric-bar-bg">
        <div class="metric-bar-fill" id="barLatency" style="width: 38%; background: #64748b;"></div>
      </div>
      <p class="metric-desc">End-to-end retrieval, rerank & LLM synthesis</p>
    </div>

    <!-- Triad Index (Hero) -->
    <div class="metric-card hero">
      <div class="metric-card-top">
        <span class="metric-label">Harmonized Triad</span>
        <div class="metric-icon" style="background:rgba(255,255,255,0.15); color:white;">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"></polygon></svg>
        </div>
      </div>
      <div class="metric-val-wrap">
        <span class="metric-number" id="valTriad">86.4%</span>
        <span class="metric-trend" style="background:rgba(255,255,255,0.2); color:white;" id="trendTriad">+10.4%</span>
      </div>
      <div class="metric-bar-bg">
        <div class="metric-bar-fill" id="barTriad" style="width: 86.4%;"></div>
      </div>
      <p class="metric-desc">Composite 4-dimensional quality index</p>
    </div>
  </section>

  <!-- Analytics Section: High-Contrast Radar + Comparative Bar Graph -->
  <section class="analytics-grid">
    <!-- RAG Triad Health Radar -->
    <div class="chart-panel">
      <div class="panel-top">
        <div>
          <h2 class="panel-heading">RAG Triad Health Radar</h2>
          <p class="panel-sub">Multidimensional retrieval & generation footprint</p>
        </div>
        <div class="legend-row">
          <div class="legend-chip" id="radarLegendP1">
            <div class="legend-marker" style="background: #94a3b8;"></div>
            <span>P1: Baseline</span>
          </div>
          <div class="legend-chip" id="radarLegendP2">
            <div class="legend-marker" style="background: #818cf8;"></div>
            <span>P2: Hybrid</span>
          </div>
          <div class="legend-chip" id="radarLegendP3">
            <div class="legend-marker" style="background: #059669;"></div>
            <span>P3: Rerank</span>
          </div>
        </div>
      </div>

      <div class="radar-wrapper">
        <canvas id="radarCanvas" width="880" height="680"></canvas>
      </div>
    </div>

    <!-- Comparative Ablation Bar Chart -->
    <div class="chart-panel">
      <div class="panel-top">
        <div>
          <h2 class="panel-heading">Ablation Head-to-Head Progression</h2>
          <p class="panel-sub">Phase 1 (Dense) vs Phase 2 (Hybrid) vs Phase 3 (Cross-Encoder Rerank)</p>
        </div>
        <div class="legend-row">
          <div class="legend-chip">
            <div class="legend-marker" style="background: #cbd5e1;"></div>
            <span>P1 Baseline</span>
          </div>
          <div class="legend-chip">
            <div class="legend-marker" style="background: #818cf8;"></div>
            <span>P2 Hybrid</span>
          </div>
          <div class="legend-chip">
            <div class="legend-marker" style="background: #059669;"></div>
            <span>P3 Rerank</span>
          </div>
        </div>
      </div>

      <div class="bar-chart-wrapper">
        <div class="bar-columns" id="barChartContainer">
          <!-- Rendered dynamically via JS -->
        </div>
      </div>
    </div>
  </section>

  <!-- Roadmap Lifecycle Strip -->
  <section class="roadmap-section">
    <div style="display:flex; justify-content:space-between; align-items:center;">
      <div>
        <h3 class="panel-heading" style="font-size:15px;">Pipeline Progression & Architecture Roadmap</h3>
        <p class="panel-sub">Systematic step-by-step engineering and empirical validation</p>
      </div>
      <span class="pill-badge" style="background:#ecfdf5; color:#065f46; border:1px solid #a7f3d0;">Phase 3 Complete &bull; Phase 4 Up Next</span>
    </div>

    <div class="roadmap-timeline">
      <div class="roadmap-node done">
        <span class="rn-tag" style="background:#dcfce7; color:#15803d;">&check; Phase 1 Complete</span>
        <h4 class="rn-title">Dense Baseline</h4>
        <p class="rn-desc">ChromaDB + MiniLM dense vector index. Established 76.5% recall and 59.5% precision baseline.</p>
      </div>

      <div class="roadmap-node done">
        <span class="rn-tag" style="background:#dcfce7; color:#15803d;">&check; Phase 2 Complete</span>
        <h4 class="rn-title">Hybrid Search (BM25 + RRF)</h4>
        <p class="rn-desc">Reciprocal Rank Fusion of dense vectors + sparse lexical tokens. Recall +10.0%, Precision +11.5%.</p>
      </div>

      <div class="roadmap-node active">
        <span class="rn-tag" style="background:#d1fae5; color:#065f46;">&check; Phase 3 (Production Ready)</span>
        <h4 class="rn-title">Cross-Encoder Reranking</h4>
        <p class="rn-desc">Joint query-passage attention (ms-marco-MiniLM-L-6-v2). Precision jumps to 77.0% (+17.5% net).</p>
      </div>

      <div class="roadmap-node upcoming">
        <span class="rn-tag" style="background:#e0e7ff; color:#3730a3;">🚀 Phase 4 Up Next</span>
        <h4 class="rn-title">Query Transformation & Routing</h4>
        <p class="rn-desc">HyDE (Hypothetical Document Embeddings), Step-Back prompting, and Multi-Query expansion.</p>
      </div>
    </div>
  </section>

  <!-- Diagnostics Table -->
  <section class="table-card">
    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
      <div>
        <h3 class="panel-heading" style="font-size:15px;">Sample Diagnostics & Evidence Attribution (N=20)</h3>
        <p class="panel-sub">Ground-truth validation across Amnesty QA benchmark samples</p>
      </div>
      <span style="font-size:12px; color:var(--text-muted);">Click <b>Inspect</b> to view ground truth, synthesized answer & judge reasoning</span>
    </div>

    <table class="diagnostic-table">
      <thead>
        <tr>
          <th>ID</th>
          <th>Evaluated Question</th>
          <th>P1 Recall</th>
          <th>P2 Recall</th>
          <th>P3 Recall</th>
          <th>Precision (P3)</th>
          <th>Faithfulness (P3)</th>
          <th>Relevance (P3)</th>
          <th>Latency</th>
          <th>Audit</th>
        </tr>
      </thead>
      <tbody id="evalTableBody">
        <!-- Rendered via JS -->
      </tbody>
    </table>
  </section>

</div>

<!-- Inspection Slide-over Drawer -->
<div class="drawer-scrim" id="drawerScrim" onclick="closeDrawer(event)">
  <div class="drawer-body" onclick="event.stopPropagation()">
    <div class="drawer-top">
      <div>
        <span class="qid-pill" id="drawerQid">amnesty_01</span>
        <h2 style="font-size:17px; font-weight:800; margin-top:8px; line-height:1.4;" id="drawerQuestion">Question Title</h2>
      </div>
      <button class="drawer-close" onclick="closeDrawer()">&times;</button>
    </div>

    <div style="display:grid; grid-template-columns:repeat(4, 1fr); gap:10px; margin-bottom:20px;">
      <div style="background:#f8fafc; border:1px solid var(--border); border-radius:8px; padding:10px; text-align:center;">
        <div style="font-size:10.5px; color:var(--text-muted); font-weight:700;">RECALL (P3)</div>
        <div style="font-size:18px; font-weight:800; color:var(--emerald); margin-top:2px;" id="dmRecall">0.85</div>
      </div>
      <div style="background:#f8fafc; border:1px solid var(--border); border-radius:8px; padding:10px; text-align:center;">
        <div style="font-size:10.5px; color:var(--text-muted); font-weight:700;">PRECISION (P3)</div>
        <div style="font-size:18px; font-weight:800; color:var(--blue); margin-top:2px;" id="dmPrecision">0.90</div>
      </div>
      <div style="background:#f8fafc; border:1px solid var(--border); border-radius:8px; padding:10px; text-align:center;">
        <div style="font-size:10.5px; color:var(--text-muted); font-weight:700;">FAITHFULNESS</div>
        <div style="font-size:18px; font-weight:800; color:var(--purple); margin-top:2px;" id="dmFaith">1.00</div>
      </div>
      <div style="background:#f8fafc; border:1px solid var(--border); border-radius:8px; padding:10px; text-align:center;">
        <div style="font-size:10.5px; color:var(--text-muted); font-weight:700;">RELEVANCE</div>
        <div style="font-size:18px; font-weight:800; color:var(--amber); margin-top:2px;" id="dmRelevance">0.95</div>
      </div>
    </div>

    <div class="block-label">Ground Truth Target</div>
    <div class="block-content" id="dmGroundTruth"></div>

    <div class="block-label">Synthesized Response (Phase 3 Rerank)</div>
    <div class="block-content" style="background:#ecfdf5; border-color:#a7f3d0;" id="dmAnswer"></div>

    <div class="block-label">LLM-as-a-Judge Reasoning</div>
    <div class="block-content" style="background:#fffbeb; border-color:#fde68a;" id="dmReasoning"></div>

    <div class="block-label">Ablation Root-Cause Analysis</div>
    <div class="block-content" style="background:#eff6ff; border-color:#bfdbfe;" id="dmInsight"></div>
  </div>
</div>

<script>
  const BENCHMARK = {benchmark_json};

  let activeView = "phase3";

  function setDashboardView(view) {{
    activeView = view;
    document.getElementById("segPhase1").classList.remove("active");
    document.getElementById("segPhase2").classList.remove("active");
    document.getElementById("segPhase3").classList.remove("active");
    document.getElementById("segDiff").classList.remove("active");

    const badge = document.getElementById("topActiveBadge");

    if (view === "phase1") {{
      document.getElementById("segPhase1").classList.add("active");
      badge.innerText = "Phase 1: Dense Baseline Active";
      badge.style.background = "#f1f5f9";
      badge.style.color = "#475569";
      badge.style.borderColor = "#cbd5e1";

      renderCards(BENCHMARK.phase1, false);
      document.getElementById("radarLegendP1").style.display = "flex";
      document.getElementById("radarLegendP2").style.display = "none";
      document.getElementById("radarLegendP3").style.display = "none";
      drawModernRadar([BENCHMARK.phase1.recall, BENCHMARK.phase1.precision, BENCHMARK.phase1.faithfulness, BENCHMARK.phase1.relevance], null, null);
    }} else if (view === "phase2") {{
      document.getElementById("segPhase2").classList.add("active");
      badge.innerText = "Phase 2: Hybrid Search Active";
      badge.style.background = "#eef2ff";
      badge.style.color = "#4338ca";
      badge.style.borderColor = "#c7d2fe";

      renderCards(BENCHMARK.phase2, true);
      document.getElementById("radarLegendP1").style.display = "none";
      document.getElementById("radarLegendP2").style.display = "flex";
      document.getElementById("radarLegendP3").style.display = "none";
      drawModernRadar([BENCHMARK.phase2.recall, BENCHMARK.phase2.precision, BENCHMARK.phase2.faithfulness, BENCHMARK.phase2.relevance], null, null);
    }} else if (view === "phase3") {{
      document.getElementById("segPhase3").classList.add("active");
      badge.innerText = "Phase 3: Cross-Encoder Rerank Active";
      badge.style.background = "#ecfdf5";
      badge.style.color = "#065f46";
      badge.style.borderColor = "#a7f3d0";

      renderCards(BENCHMARK.phase3, true);
      document.getElementById("radarLegendP1").style.display = "none";
      document.getElementById("radarLegendP2").style.display = "none";
      document.getElementById("radarLegendP3").style.display = "flex";
      drawModernRadar([BENCHMARK.phase3.recall, BENCHMARK.phase3.precision, BENCHMARK.phase3.faithfulness, BENCHMARK.phase3.relevance], null, null);
    }} else {{
      document.getElementById("segDiff").classList.add("active");
      badge.innerText = "3-Way Ablation Head-to-Head Overlay";
      badge.style.background = "#f5f3ff";
      badge.style.color = "#6d28d9";
      badge.style.borderColor = "#ddd6fe";

      renderCards(BENCHMARK.phase3, true);
      document.getElementById("radarLegendP1").style.display = "flex";
      document.getElementById("radarLegendP2").style.display = "flex";
      document.getElementById("radarLegendP3").style.display = "flex";
      drawModernRadar(
        [BENCHMARK.phase3.recall, BENCHMARK.phase3.precision, BENCHMARK.phase3.faithfulness, BENCHMARK.phase3.relevance],
        [BENCHMARK.phase2.recall, BENCHMARK.phase2.precision, BENCHMARK.phase2.faithfulness, BENCHMARK.phase2.relevance],
        [BENCHMARK.phase1.recall, BENCHMARK.phase1.precision, BENCHMARK.phase1.faithfulness, BENCHMARK.phase1.relevance]
      );
    }}
  }}

  function renderCards(data, showDelta) {{
    document.getElementById("valRecall").innerText = data.recall.toFixed(1) + "%";
    document.getElementById("barRecall").style.width = data.recall + "%";
    document.getElementById("trendRecall").innerText = data.delta.recall;
    document.getElementById("trendRecall").style.display = showDelta ? "inline-block" : "none";

    document.getElementById("valPrecision").innerText = data.precision.toFixed(1) + "%";
    document.getElementById("barPrecision").style.width = data.precision + "%";
    document.getElementById("trendPrecision").innerText = data.delta.precision;
    document.getElementById("trendPrecision").style.display = showDelta ? "inline-block" : "none";

    document.getElementById("valFaith").innerText = data.faithfulness.toFixed(1) + "%";
    document.getElementById("barFaith").style.width = data.faithfulness + "%";
    document.getElementById("trendFaith").innerText = data.delta.faith;

    document.getElementById("valRelevance").innerText = data.relevance.toFixed(1) + "%";
    document.getElementById("barRelevance").style.width = data.relevance + "%";
    document.getElementById("trendRelevance").innerText = data.delta.relevance;
    document.getElementById("trendRelevance").style.display = showDelta ? "inline-block" : "none";

    document.getElementById("valLatency").innerText = data.latency.toFixed(2) + "s";
    document.getElementById("barLatency").style.width = Math.min(100, (data.latency * 5.5)) + "%";
    document.getElementById("trendLatency").innerText = data.delta.latency;
    document.getElementById("trendLatency").style.display = showDelta ? "inline-block" : "none";

    document.getElementById("valTriad").innerText = data.triad.toFixed(1) + "%";
    document.getElementById("barTriad").style.width = data.triad + "%";
    document.getElementById("trendTriad").innerText = data.delta.triad;
    document.getElementById("trendTriad").style.display = showDelta ? "inline-block" : "none";
  }}

  // Draw High-Contrast, Modern Radar Chart supporting up to 3 superimposed phases
  function drawModernRadar(primaryVals, secondaryVals, tertiaryVals) {{
    const canvas = document.getElementById("radarCanvas");
    const ctx = canvas.getContext("2d");
    const dpr = window.devicePixelRatio || 2;

    const w = canvas.width / dpr;
    const h = canvas.height / dpr;

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.save();
    ctx.scale(dpr, dpr);

    const centerX = w / 2;
    const centerY = h / 2 + 6;
    const maxRadius = 120;

    const axes = [
      {{ name: "Context Recall", angle: -Math.PI / 2, align: "center", baseline: "bottom", offsetY: -16 }},
      {{ name: "Precision", angle: 0, align: "left", baseline: "middle", offsetX: 16 }},
      {{ name: "Faithfulness", angle: Math.PI / 2, align: "center", baseline: "top", offsetY: 16 }},
      {{ name: "Relevance", angle: Math.PI, align: "right", baseline: "middle", offsetX: -16 }}
    ];

    const rings = [0.25, 0.5, 0.75, 1.0];

    // Web background
    rings.forEach((r, idx) => {{
      ctx.beginPath();
      for (let i = 0; i < axes.length; i++) {{
        const x = centerX + Math.cos(axes[i].angle) * (maxRadius * r);
        const y = centerY + Math.sin(axes[i].angle) * (maxRadius * r);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }}
      ctx.closePath();
      ctx.fillStyle = idx % 2 === 0 ? "rgba(241, 245, 249, 0.65)" : "rgba(255, 255, 255, 0.9)";
      ctx.fill();
      ctx.strokeStyle = "#cbd5e1";
      ctx.lineWidth = 1.2;
      ctx.stroke();

      ctx.fillStyle = "#94a3b8";
      ctx.font = "600 10.5px 'JetBrains Mono', monospace";
      ctx.textAlign = "center";
      ctx.fillText(Math.round(r * 100) + "%", centerX, centerY - (maxRadius * r) + 12);
    }});

    // Axis spokes
    axes.forEach(axis => {{
      ctx.beginPath();
      ctx.moveTo(centerX, centerY);
      ctx.lineTo(centerX + Math.cos(axis.angle) * maxRadius, centerY + Math.sin(axis.angle) * maxRadius);
      ctx.strokeStyle = "#94a3b8";
      ctx.lineWidth = 1.4;
      ctx.stroke();

      const labelX = centerX + Math.cos(axis.angle) * maxRadius + (axis.offsetX || 0);
      const labelY = centerY + Math.sin(axis.angle) * maxRadius + (axis.offsetY || 0);

      ctx.fillStyle = "#090d16";
      ctx.font = "700 12.5px 'Plus Jakarta Sans', sans-serif";
      ctx.textAlign = axis.align;
      ctx.textBaseline = axis.baseline;
      ctx.fillText(axis.name, labelX, labelY);
    }});

    function renderRadarShape(vals, strokeColor, fillColor, dotColor, isDashed = false) {{
      ctx.save();
      if (isDashed) ctx.setLineDash([5, 5]);

      ctx.beginPath();
      for (let i = 0; i < axes.length; i++) {{
        const fraction = Math.max(0.08, vals[i] / 100);
        const x = centerX + Math.cos(axes[i].angle) * (maxRadius * fraction);
        const y = centerY + Math.sin(axes[i].angle) * (maxRadius * fraction);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }}
      ctx.closePath();
      ctx.fillStyle = fillColor;
      ctx.fill();
      ctx.strokeStyle = strokeColor;
      ctx.lineWidth = 2.8;
      ctx.stroke();
      ctx.restore();

      // Anchor dots
      for (let i = 0; i < axes.length; i++) {{
        const fraction = Math.max(0.08, vals[i] / 100);
        const x = centerX + Math.cos(axes[i].angle) * (maxRadius * fraction);
        const y = centerY + Math.sin(axes[i].angle) * (maxRadius * fraction);

        ctx.beginPath();
        ctx.arc(x, y, 4.5, 0, Math.PI * 2);
        ctx.fillStyle = dotColor;
        ctx.fill();
        ctx.strokeStyle = "#ffffff";
        ctx.lineWidth = 2;
        ctx.stroke();
      }}
    }}

    // Tertiary (Baseline) if present
    if (tertiaryVals) {{
      renderRadarShape(tertiaryVals, "#94a3b8", "rgba(148, 163, 184, 0.18)", "#64748b", true);
    }}

    // Secondary (Hybrid) if present
    if (secondaryVals) {{
      renderRadarShape(secondaryVals, "#818cf8", "rgba(129, 140, 248, 0.22)", "#4f46e5", false);
    }}

    // Primary
    let primaryStroke = "#059669";
    let primaryFill = "rgba(5, 150, 105, 0.28)";
    let primaryDot = "#047857";

    if (activeView === "phase1") {{
      primaryStroke = "#64748b";
      primaryFill = "rgba(100, 116, 139, 0.25)";
      primaryDot = "#475569";
    }} else if (activeView === "phase2") {{
      primaryStroke = "#4f46e5";
      primaryFill = "rgba(79, 70, 229, 0.3)";
      primaryDot = "#4338ca";
    }}

    renderRadarShape(primaryVals, primaryStroke, primaryFill, primaryDot, false);

    ctx.restore();
  }}

  // Render Modern 3-Way Grouped Bar Chart
  function renderComparativeBarChart() {{
    const container = document.getElementById("barChartContainer");
    container.innerHTML = "";

    BENCHMARK.metricsList.forEach(m => {{
      const p1H = Math.round((m.p1 / 100) * 190);
      const p2H = Math.round((m.p2 / 100) * 190);
      const p3H = Math.round((m.p3 / 100) * 190);

      const group = document.createElement("div");
      group.className = "bar-group";
      group.innerHTML = `
        <div class="bar-delta-tag">${{m.delta}}</div>
        <div class="bar-pair">
          <!-- Phase 1 Bar -->
          <div class="bar-stem baseline" style="height: ${{p1H}}px;" title="Phase 1: ${{m.p1}}%">
            <span class="bar-val-text">${{m.p1}}%</span>
          </div>
          <!-- Phase 2 Bar -->
          <div class="bar-stem hybrid" style="height: ${{p2H}}px;" title="Phase 2: ${{m.p2}}%">
            <span class="bar-val-text" style="color:#6366f1;">${{m.p2}}%</span>
          </div>
          <!-- Phase 3 Bar -->
          <div class="bar-stem rerank" style="height: ${{p3H}}px;" title="Phase 3: ${{m.p3}}%">
            <span class="bar-val-text" style="color:#059669;">${{m.p3}}%</span>
          </div>
        </div>
        <div class="bar-label-caption">${{m.label}}</div>
      `;
      container.appendChild(group);
    }});
  }}

  // Populate Diagnostics Table with 20 rows
  function renderDiagnosticTable() {{
    const tbody = document.getElementById("evalTableBody");
    tbody.innerHTML = "";

    BENCHMARK.samples.forEach(s => {{
      const p1R = (s.p1Recall * 100).toFixed(0);
      const p2R = (s.p2Recall * 100).toFixed(0);
      const p3R = (s.p3Recall * 100).toFixed(0);
      const p3P = (s.p3Precision * 100).toFixed(0);
      const p3F = (s.p3Faith * 100).toFixed(0);
      const p3Rel = (s.p3Relevance * 100).toFixed(0);

      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td><span class="qid-pill">${{s.id}}</span></td>
        <td><div class="q-title">${{s.question}}</div></td>
        <td><span class="chip chip-slate">${{p1R}}%</span></td>
        <td><span class="chip chip-blue">${{p2R}}%</span></td>
        <td><span class="chip chip-green">${{p3R}}%</span></td>
        <td><b>${{p3P}}%</b></td>
        <td><span class="chip chip-purple">${{p3F}}%</span></td>
        <td><b>${{p3Rel}}%</b></td>
        <td style="color:var(--text-muted); font-family:'JetBrains Mono', monospace;">${{s.latency}}s</td>
        <td>
          <button class="btn-inspect" onclick="openDrawer('${{s.id}}')">Inspect &rarr;</button>
        </td>
      `;
      tbody.appendChild(tr);
    }});
  }}

  function openDrawer(id) {{
    const s = BENCHMARK.samples.find(item => item.id === id);
    if (!s) return;

    document.getElementById("drawerQid").innerText = s.id;
    document.getElementById("drawerQuestion").innerText = s.question;
    document.getElementById("dmRecall").innerText = (s.p3Recall * 100).toFixed(0) + "%";
    document.getElementById("dmPrecision").innerText = (s.p3Precision * 100).toFixed(0) + "%";
    document.getElementById("dmFaith").innerText = (s.p3Faith * 100).toFixed(0) + "%";
    document.getElementById("dmRelevance").innerText = (s.p3Relevance * 100).toFixed(0) + "%";

    document.getElementById("dmGroundTruth").innerText = s.groundTruth;
    document.getElementById("dmAnswer").innerText = s.answer;
    document.getElementById("dmReasoning").innerText = s.reasoning;
    document.getElementById("dmInsight").innerText = s.insight;

    document.getElementById("drawerScrim").style.display = "flex";
  }}

  function closeDrawer(e) {{
    document.getElementById("drawerScrim").style.display = "none";
  }}

  function copyCliCommand() {{
    const cmd = "python evals/run_eval.py --provider groq --dataset amnesty_qa --mode hybrid_rerank --limit 20";
    navigator.clipboard.writeText(cmd).then(() => {{
      alert("Copied Phase 3 CLI Benchmark command to clipboard:\\n\\n" + cmd);
    }}).catch(() => {{
      prompt("Copy CLI Benchmark command:", cmd);
    }});
  }}

  // Initialize
  window.addEventListener("DOMContentLoaded", () => {{
    setDashboardView("phase3");
    renderComparativeBarChart();
    renderDiagnosticTable();
  }});
</script>

</body>
</html>
"""

    out_workspace = "eval_dashboard.html"
    with open(out_workspace, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"Generated {out_workspace} ({len(html_content)} bytes)")

    # Also update artifact directory
    artifact_dir = r"C:\Users\User\.gemini\antigravity\brain\d573e46b-a0b7-4a4b-b4a9-267d492c70cb"
    if os.path.exists(artifact_dir):
        out_artifact = os.path.join(artifact_dir, "eval_dashboard.html")
        shutil.copy(out_workspace, out_artifact)
        print(f"Synced to artifact path: {out_artifact}")

if __name__ == "__main__":
    generate_html()
