import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
json_path = ROOT / "scripts" / "benchmark_data.json"
with open(json_path, "r", encoding="utf-8") as f:
    bench_data = json.load(f)

bench_json_str = json.dumps(bench_data, indent=2)

html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>RAG Evaluation Suite | 4-Phase Architecture & Ablation Analytics</title>
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
      margin-bottom: 24px;
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
      width: 44px;
      height: 44px;
      background: linear-gradient(135deg, #4f46e5 0%, #06b6d4 100%);
      border-radius: 12px;
      display: flex;
      align-items: center;
      justify-content: center;
      color: white;
      font-size: 20px;
      box-shadow: 0 4px 14px rgba(79, 70, 229, 0.35);
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
      font-size: 12px;
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

    .metric-card.hero {{
      background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
      border-color: #312e81;
      color: white;
    }}

    .metric-card.hero .metric-label {{ color: #a5b4fc; }}
    .metric-card.hero .metric-desc {{ color: #c7d2fe; }}
    .metric-card.hero .metric-bar-bg {{ background: rgba(255, 255, 255, 0.15); }}
    .metric-card.hero .metric-bar-fill {{ background: #818cf8; }}

    .metric-card-top {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 12px;
    }}

    .metric-label {{
      font-size: 12px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: var(--text-muted);
    }}

    .metric-icon {{
      width: 32px;
      height: 32px;
      border-radius: 8px;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 15px;
    }}

    .metric-val-wrap {{
      display: flex;
      align-items: baseline;
      gap: 8px;
      margin-bottom: 12px;
    }}

    .metric-number {{
      font-size: 28px;
      font-weight: 800;
      letter-spacing: -0.03em;
      font-family: 'JetBrains Mono', monospace;
    }}

    .metric-trend {{
      font-size: 12px;
      font-weight: 700;
      padding: 2px 6px;
      border-radius: 6px;
      font-family: 'JetBrains Mono', monospace;
    }}

    .trend-pos {{ background: var(--emerald-light); color: var(--emerald); }}
    .trend-neg {{ background: #fef2f2; color: #dc2626; }}
    .trend-neutral {{ background: var(--surface-subtle); color: var(--text-muted); }}

    .metric-bar-bg {{
      height: 6px;
      background: var(--surface-subtle);
      border-radius: 4px;
      overflow: hidden;
      margin-bottom: 10px;
    }}

    .metric-bar-fill {{
      height: 100%;
      border-radius: 4px;
      transition: width 0.6s cubic-bezier(0.16, 1, 0.3, 1);
    }}

    .metric-desc {{
      font-size: 11.5px;
      color: var(--text-muted);
      line-height: 1.35;
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
      font-size: 12px;
      color: var(--text-muted);
      margin-top: 2px;
    }}

    .legend-row {{
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
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

    .radar-wrapper {{
      position: relative;
      display: flex;
      justify-content: center;
      align-items: center;
      height: 350px;
      background: radial-gradient(circle at center, rgba(79, 70, 229, 0.04) 0%, transparent 70%);
      border-radius: var(--radius-md);
    }}

    #radarCanvas {{
      width: 440px;
      height: 340px;
    }}

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
    }}

    .bar-delta-tag {{
      font-size: 11px;
      font-weight: 700;
      color: var(--emerald);
      margin-bottom: 8px;
      font-family: 'JetBrains Mono', monospace;
    }}

    .bar-quad {{
      display: flex;
      align-items: flex-end;
      gap: 4px;
      height: 210px;
    }}

    .bar-stem {{
      width: 14px;
      border-radius: 4px 4px 0 0;
      transition: height 0.6s cubic-bezier(0.16, 1, 0.3, 1);
      position: relative;
      cursor: pointer;
    }}

    .bar-stem.baseline {{ background: #cbd5e1; }}
    .bar-stem.hybrid {{ background: #818cf8; }}
    .bar-stem.rerank {{ background: #059669; }}
    .bar-stem.multi-query {{ background: #7c3aed; }}

    .bar-stem:hover {{ filter: brightness(0.92); }}

    .bar-stem .bar-val-text {{
      position: absolute;
      top: -20px;
      left: 50%;
      transform: translateX(-50%);
      font-size: 9px;
      font-weight: 700;
      font-family: 'JetBrains Mono', monospace;
      white-space: nowrap;
    }}

    .bar-label-caption {{
      font-size: 12px;
      font-weight: 700;
      color: var(--text-muted);
      margin-top: 10px;
    }}

    /* Roadmap Timeline */
    .roadmap-section {{
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--radius-lg);
      padding: 24px;
      margin-bottom: 24px;
      box-shadow: var(--shadow-card);
    }}

    .roadmap-timeline {{
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 16px;
      margin-top: 18px;
    }}

    .roadmap-node {{
      background: var(--surface-subtle);
      border: 1px solid var(--border);
      border-radius: var(--radius-md);
      padding: 16px;
      position: relative;
    }}

    .roadmap-node.done {{
      background: #f0fdf4;
      border-color: #bbf7d0;
    }}

    .roadmap-node.active {{
      background: #f5f3ff;
      border-color: #ddd6fe;
      box-shadow: 0 0 0 2px rgba(124, 58, 237, 0.2);
    }}

    .rn-tag {{
      display: inline-block;
      font-size: 10px;
      font-weight: 800;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      padding: 2px 7px;
      border-radius: 6px;
      margin-bottom: 8px;
    }}

    .rn-title {{
      font-size: 14px;
      font-weight: 800;
      color: var(--text-main);
      margin-bottom: 4px;
    }}

    .rn-desc {{
      font-size: 11.5px;
      color: var(--text-muted);
      line-height: 1.4;
    }}

    /* Table Section */
    .table-card {{
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--radius-lg);
      padding: 24px;
      box-shadow: var(--shadow-card);
      overflow-x: auto;
    }}

    .diagnostic-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 12.5px;
      text-align: left;
    }}

    .diagnostic-table th {{
      background: var(--surface-subtle);
      color: var(--text-muted);
      font-weight: 700;
      padding: 12px 14px;
      border-bottom: 1px solid var(--border);
      text-transform: uppercase;
      font-size: 11px;
      letter-spacing: 0.04em;
    }}

    .diagnostic-table td {{
      padding: 12px 14px;
      border-bottom: 1px solid var(--border-subtle);
      color: var(--text-main);
    }}

    .diagnostic-table tr:hover {{
      background: #f8fafc;
    }}

    .qid-pill {{
      font-family: 'JetBrains Mono', monospace;
      font-size: 11px;
      font-weight: 600;
      background: var(--surface-subtle);
      padding: 2px 6px;
      border-radius: 4px;
      color: var(--text-muted);
    }}

    .q-title {{
      font-weight: 600;
      color: var(--text-main);
      max-width: 380px;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }}

    .chip {{
      display: inline-block;
      padding: 2px 7px;
      border-radius: 6px;
      font-weight: 700;
      font-family: 'JetBrains Mono', monospace;
      font-size: 11.5px;
    }}

    .chip-green {{ background: #ecfdf5; color: #059669; }}
    .chip-purple {{ background: #f5f3ff; color: #7c3aed; }}
    .chip-blue {{ background: #eff6ff; color: #2563eb; }}
    .chip-slate {{ background: #f1f5f9; color: #64748b; }}

    .btn-inspect {{
      background: var(--surface);
      border: 1px solid var(--border);
      padding: 4px 10px;
      border-radius: 6px;
      font-size: 11.5px;
      font-weight: 600;
      color: var(--primary);
      cursor: pointer;
      transition: all 0.15s ease;
    }}

    .btn-inspect:hover {{
      background: var(--primary-light);
      border-color: #c7d2fe;
    }}

    /* Drawer */
    .drawer-scrim {{
      display: none;
      position: fixed;
      top: 0;
      left: 0;
      right: 0;
      bottom: 0;
      background: rgba(15, 23, 42, 0.4);
      backdrop-filter: blur(4px);
      z-index: 1000;
      justify-content: flex-end;
    }}

    .drawer-body {{
      width: 600px;
      background: white;
      height: 100%;
      padding: 32px 28px;
      overflow-y: auto;
      box-shadow: -4px 0 24px rgba(0, 0, 0, 0.15);
      animation: slideIn 0.25s cubic-bezier(0.16, 1, 0.3, 1);
    }}

    @keyframes slideIn {{
      from {{ transform: translateX(100%); }}
      to {{ transform: translateX(0); }}
    }}

    .drawer-top {{
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      margin-bottom: 20px;
      padding-bottom: 16px;
      border-bottom: 1px solid var(--border);
    }}

    .drawer-close {{
      background: none;
      border: none;
      font-size: 22px;
      cursor: pointer;
      color: var(--text-muted);
    }}

    .block-label {{
      font-size: 11px;
      font-weight: 800;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: var(--text-muted);
      margin: 16px 0 6px 0;
    }}

    .block-content {{
      background: #f8fafc;
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 12px 14px;
      font-size: 12.5px;
      line-height: 1.5;
      color: #334155;
    }}
  </style>
</head>
<body>

<div class="container">

  <!-- Top Navigation Bar -->
  <header class="topbar">
    <div class="brand-section">
      <div class="brand-logo">🧠</div>
      <div>
        <div class="brand-title-wrap">
          <h1 class="brand-title">Enterprise RAG Evaluation Suite</h1>
          <span class="pill-badge active-phase" id="activePhasePill">Phase 4 Active</span>
        </div>
        <p class="brand-meta">End-to-End Ablation Benchmark &middot; 4-Stage Architectural Progression &middot; N=20 Golden QA Pairs</p>
      </div>
    </div>

    <div class="nav-actions">
      <div class="segmented-control">
        <button class="segment-btn" onclick="setDashboardView('phase1')">P1: Dense</button>
        <button class="segment-btn" onclick="setDashboardView('phase2')">P2: Hybrid</button>
        <button class="segment-btn" onclick="setDashboardView('phase3')">P3: Rerank</button>
        <button class="segment-btn active" onclick="setDashboardView('phase4')">P4: Multi-Query</button>
      </div>
      <button class="btn-run" onclick="copyCliCommand()">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg>
        <span>Run Benchmark</span>
      </button>
    </div>
  </header>

  <!-- System Strip -->
  <section class="system-strip">
    <div class="strip-items">
      <div class="strip-item">
        <span class="strip-dot"></span>
        <span>Dataset: <b>amnesty_qa_eval</b> (20 Golden Questions)</span>
      </div>
      <div class="strip-item">
        <span>Provider Strategy: <b>Round-Robin Resilient Pool</b> (Groq, NVIDIA NIM, OpenRouter, Ollama)</span>
      </div>
      <div class="strip-item">
        <span>Reranker: <code>ms-marco-MiniLM-L-6-v2</code></span>
      </div>
      <div class="strip-item">
        <span>Transformation: <b>Multi-Query Decomposition (k=3)</b></span>
      </div>
    </div>
    <div>
      <span>Status: <b style="color:var(--emerald);">Benchmarked &amp; Verified</b></span>
    </div>
  </section>

  <!-- 6 Metric Cards Row -->
  <section class="metric-grid">
    <!-- Context Recall -->
    <div class="metric-card">
      <div class="metric-card-top">
        <span class="metric-label">Context Recall</span>
        <div class="metric-icon" style="background:#ecfdf5; color:#059669;">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path><polyline points="22 4 12 14.01 9 11.01"></polyline></svg>
        </div>
      </div>
      <div class="metric-val-wrap">
        <span class="metric-number" id="valRecall">88.0%</span>
        <span class="metric-trend trend-pos" id="trendRecall">+11.5%</span>
      </div>
      <div class="metric-bar-bg">
        <div class="metric-bar-fill" id="barRecall" style="width: 88.0%; background: #059669;"></div>
      </div>
      <p class="metric-desc">Coverage of ground truth facts in retrieved context</p>
    </div>

    <!-- Context Precision -->
    <div class="metric-card">
      <div class="metric-card-top">
        <span class="metric-label">Context Precision</span>
        <div class="metric-icon" style="background:#eff6ff; color:#2563eb;">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><circle cx="12" cy="12" r="6"></circle><circle cx="12" cy="12" r="2"></circle></svg>
        </div>
      </div>
      <div class="metric-val-wrap">
        <span class="metric-number" id="valPrecision">79.3%</span>
        <span class="metric-trend trend-pos" id="trendPrecision">+19.8%</span>
      </div>
      <div class="metric-bar-bg">
        <div class="metric-bar-fill" id="barPrecision" style="width: 79.3%; background: #2563eb;"></div>
      </div>
      <p class="metric-desc">Density of signal-bearing chunks ranked in top-K</p>
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
        <span class="metric-number" id="valFaith">88.8%</span>
        <span class="metric-trend trend-neutral" id="trendFaith">-5.2%</span>
      </div>
      <div class="metric-bar-bg">
        <div class="metric-bar-fill" id="barFaith" style="width: 88.8%; background: #7c3aed;"></div>
      </div>
      <p class="metric-desc">Factual grounding in retrieved source evidence</p>
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
        <span class="metric-number" id="valRelevance">84.9%</span>
        <span class="metric-trend trend-pos" id="trendRelevance">+11.3%</span>
      </div>
      <div class="metric-bar-bg">
        <div class="metric-bar-fill" id="barRelevance" style="width: 84.9%; background: #d97706;"></div>
      </div>
      <p class="metric-desc">Direct semantic alignment to user inquiry</p>
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
        <span class="metric-number" id="valLatency">34.85s</span>
        <span class="metric-trend trend-neutral" id="trendLatency">+23.5s</span>
      </div>
      <div class="metric-bar-bg">
        <div class="metric-bar-fill" id="barLatency" style="width: 70%; background: #64748b;"></div>
      </div>
      <p class="metric-desc">Multi-query decomposition & cross-encoder latency</p>
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
        <span class="metric-number" id="valTriad">85.2%</span>
        <span class="metric-trend" style="background:rgba(255,255,255,0.2); color:white;" id="trendTriad">+9.3%</span>
      </div>
      <div class="metric-bar-bg">
        <div class="metric-bar-fill" id="barTriad" style="width: 85.2%;"></div>
      </div>
      <p class="metric-desc">Composite 4-dimensional RAG pipeline index</p>
    </div>
  </section>

  <!-- Analytics Section: Radar + Bar Comparison -->
  <section class="analytics-grid">
    <!-- RAG Triad Health Radar -->
    <div class="chart-panel">
      <div class="panel-top">
        <div>
          <h2 class="panel-heading">RAG Triad Health Radar</h2>
          <p class="panel-sub">Multidimensional retrieval & generation footprint</p>
        </div>
        <div class="legend-row">
          <div class="legend-chip">
            <div class="legend-marker" style="background: #94a3b8;"></div>
            <span>P1: Baseline</span>
          </div>
          <div class="legend-chip">
            <div class="legend-marker" style="background: #818cf8;"></div>
            <span>P2: Hybrid</span>
          </div>
          <div class="legend-chip">
            <div class="legend-marker" style="background: #059669;"></div>
            <span>P3: Rerank</span>
          </div>
          <div class="legend-chip">
            <div class="legend-marker" style="background: #7c3aed;"></div>
            <span>P4: Multi-Query</span>
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
          <h2 class="panel-heading">Ablation Head-to-Head Progression (N=20)</h2>
          <p class="panel-sub">P1 (Dense) vs P2 (Hybrid) vs P3 (Rerank) vs P4 (Multi-Query)</p>
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
          <div class="legend-chip">
            <div class="legend-marker" style="background: #7c3aed;"></div>
            <span>P4 Transform</span>
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
        <p class="panel-sub">Systematic step-by-step engineering and empirical validation across 4 architectural phases</p>
      </div>
      <span class="pill-badge" style="background:#ecfdf5; color:#065f46; border:1px solid #a7f3d0;">All 4 Phases Fully Validated &amp; Production Ready</span>
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
        <p class="rn-desc">Reciprocal Rank Fusion of dense vectors + sparse lexical tokens. Recall +9.8%, Precision +11.5%.</p>
      </div>

      <div class="roadmap-node done">
        <span class="rn-tag" style="background:#dcfce7; color:#15803d;">&check; Phase 3 Complete</span>
        <h4 class="rn-title">Cross-Encoder Reranking</h4>
        <p class="rn-desc">Deep joint cross-attention (ms-marco-MiniLM-L-6-v2). Precision jumps to 77.0% (+17.5% net lift).</p>
      </div>

      <div class="roadmap-node active">
        <span class="rn-tag" style="background:#ddd6fe; color:#4c1d95;">&check; Phase 4 Complete</span>
        <h4 class="rn-title">Query Transformation & Routing</h4>
        <p class="rn-desc">Multi-Query sub-query decomposition, HyDE synthetic documents, and 0ms adaptive direct bypass.</p>
      </div>
    </div>
  </section>

  <!-- Diagnostics Table -->
  <section class="table-card">
    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
      <div>
        <h3 class="panel-heading" style="font-size:15px;">Sample Diagnostics & Evidence Attribution (N=20 Golden QA Pairs)</h3>
        <p class="panel-sub">Ground-truth validation across Amnesty QA benchmark samples</p>
      </div>
      <span style="font-size:12px; color:var(--text-muted);">Click <b>Inspect</b> to view ground truth, synthesized answer & judge reasoning</span>
    </div>

    <table class="diagnostic-table">
      <thead>
        <tr>
          <th>ID</th>
          <th>Evaluated Question</th>
          <th>P1 Rec</th>
          <th>P2 Rec</th>
          <th>P3 Rec</th>
          <th>P4 Rec</th>
          <th>P4 Prec</th>
          <th>P4 Faith</th>
          <th>P4 Rel</th>
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
        <div style="font-size:10.5px; color:var(--text-muted); font-weight:700;">RECALL (P4)</div>
        <div style="font-size:18px; font-weight:800; color:var(--purple); margin-top:2px;" id="dmRecall">0.85</div>
      </div>
      <div style="background:#f8fafc; border:1px solid var(--border); border-radius:8px; padding:10px; text-align:center;">
        <div style="font-size:10.5px; color:var(--text-muted); font-weight:700;">PRECISION (P4)</div>
        <div style="font-size:18px; font-weight:800; color:var(--blue); margin-top:2px;" id="dmPrecision">0.90</div>
      </div>
      <div style="background:#f8fafc; border:1px solid var(--border); border-radius:8px; padding:10px; text-align:center;">
        <div style="font-size:10.5px; color:var(--text-muted); font-weight:700;">FAITHFULNESS</div>
        <div style="font-size:18px; font-weight:800; color:var(--emerald); margin-top:2px;" id="dmFaith">1.00</div>
      </div>
      <div style="background:#f8fafc; border:1px solid var(--border); border-radius:8px; padding:10px; text-align:center;">
        <div style="font-size:10.5px; color:var(--text-muted); font-weight:700;">RELEVANCE</div>
        <div style="font-size:18px; font-weight:800; color:var(--amber); margin-top:2px;" id="dmRelevance">0.95</div>
      </div>
    </div>

    <div class="block-label">Ground Truth Target</div>
    <div class="block-content" id="dmGroundTruth"></div>

    <div class="block-label">Synthesized Response (Phase 4 Multi-Query)</div>
    <div class="block-content" style="background:#f5f3ff; border-color:#ddd6fe;" id="dmAnswer"></div>

    <div class="block-label">LLM-as-a-Judge Reasoning</div>
    <div class="block-content" style="background:#fffbeb; border-color:#fde68a;" id="dmReasoning"></div>

    <div class="block-label">Ablation Root-Cause Analysis</div>
    <div class="block-content" style="background:#eff6ff; border-color:#bfdbfe;" id="dmInsight"></div>
  </div>
</div>

<script>
  const BENCHMARK = {bench_json_str};

  let activeView = "phase4";

  function setDashboardView(phaseKey) {{
    activeView = phaseKey;
    const p = BENCHMARK[phaseKey];
    if (!p) return;

    // Update segment active states
    document.querySelectorAll(".segment-btn").forEach(btn => {{
      btn.classList.remove("active");
      if (btn.textContent.toLowerCase().includes(phaseKey.replace("phase", "p"))) {{
        btn.classList.add("active");
      }}
    }});

    // Update Top Pill
    const pill = document.getElementById("activePhasePill");
    pill.textContent = p.name + " Active";
    if (phaseKey === "phase4") {{
      pill.style.background = "#f5f3ff";
      pill.style.color = "#6b21a8";
      pill.style.borderColor = "#ddd6fe";
    }} else if (phaseKey === "phase3") {{
      pill.style.background = "#ecfdf5";
      pill.style.color = "#065f46";
      pill.style.borderColor = "#a7f3d0";
    }} else if (phaseKey === "phase2") {{
      pill.style.background = "#eef2ff";
      pill.style.color = "#3730a3";
      pill.style.borderColor = "#c7d2fe";
    }} else {{
      pill.style.background = "#f1f5f9";
      pill.style.color = "#475569";
      pill.style.borderColor = "#cbd5e1";
    }}

    // Update Metric Cards
    document.getElementById("valRecall").textContent = p.recall.toFixed(1) + "%";
    document.getElementById("trendRecall").textContent = p.delta.recall;
    document.getElementById("barRecall").style.width = p.recall + "%";

    document.getElementById("valPrecision").textContent = p.precision.toFixed(1) + "%";
    document.getElementById("trendPrecision").textContent = p.delta.precision;
    document.getElementById("barPrecision").style.width = p.precision + "%";

    document.getElementById("valFaith").textContent = p.faithfulness.toFixed(1) + "%";
    document.getElementById("trendFaith").textContent = p.delta.faith;
    document.getElementById("barFaith").style.width = p.faithfulness + "%";

    document.getElementById("valRelevance").textContent = p.relevance.toFixed(1) + "%";
    document.getElementById("trendRelevance").textContent = p.delta.relevance;
    document.getElementById("barRelevance").style.width = p.relevance + "%";

    document.getElementById("valLatency").textContent = p.latency.toFixed(2) + "s";
    document.getElementById("trendLatency").textContent = p.delta.latency;

    document.getElementById("valTriad").textContent = p.triad.toFixed(1) + "%";
    document.getElementById("trendTriad").textContent = p.delta.triad;
    document.getElementById("barTriad").style.width = p.triad + "%";

    renderRadar();
  }}

  // High-Resolution Canvas Radar Chart
  function renderRadar() {{
    const canvas = document.getElementById("radarCanvas");
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const dpr = window.devicePixelRatio || 2;

    const width = 440;
    const height = 340;
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    ctx.scale(dpr, dpr);

    ctx.clearRect(0, 0, width, height);

    const centerX = width / 2;
    const centerY = height / 2 + 6;
    const maxRadius = 115;
    const numLevels = 4;

    const axes = [
      {{ name: "Recall", angle: -Math.PI / 2, align: "center", baseline: "bottom", offsetY: -8 }},
      {{ name: "Precision", angle: 0, align: "left", baseline: "middle", offsetX: 10 }},
      {{ name: "Faithfulness", angle: Math.PI / 2, align: "center", baseline: "top", offsetY: 8 }},
      {{ name: "Relevance", angle: Math.PI, align: "right", baseline: "middle", offsetX: -10 }}
    ];

    // Grid circles
    for (let level = 1; level <= numLevels; level++) {{
      const r = (maxRadius / numLevels) * level;
      ctx.beginPath();
      ctx.arc(centerX, centerY, r, 0, Math.PI * 2);
      ctx.strokeStyle = "#e2e8f0";
      ctx.lineWidth = 1;
      ctx.setLineDash([3, 3]);
      ctx.stroke();
      ctx.setLineDash([]);
    }}

    // Axis lines & labels
    axes.forEach(axis => {{
      ctx.beginPath();
      ctx.moveTo(centerX, centerY);
      ctx.lineTo(centerX + Math.cos(axis.angle) * maxRadius, centerY + Math.sin(axis.angle) * maxRadius);
      ctx.strokeStyle = "#cbd5e1";
      ctx.lineWidth = 1.2;
      ctx.stroke();

      const labelX = centerX + Math.cos(axis.angle) * maxRadius + (axis.offsetX || 0);
      const labelY = centerY + Math.sin(axis.angle) * maxRadius + (axis.offsetY || 0);

      ctx.fillStyle = "#090d16";
      ctx.font = "700 12px 'Plus Jakarta Sans', sans-serif";
      ctx.textAlign = axis.align;
      ctx.textBaseline = axis.baseline;
      ctx.fillText(axis.name, labelX, labelY);
    }});

    function renderRadarShape(vals, strokeColor, fillColor, dotColor, isDashed = false) {{
      ctx.save();
      if (isDashed) ctx.setLineDash([4, 4]);

      ctx.beginPath();
      for (let i = 0; i < axes.length; i++) {{
        const fraction = Math.max(0.1, vals[i] / 100);
        const x = centerX + Math.cos(axes[i].angle) * (maxRadius * fraction);
        const y = centerY + Math.sin(axes[i].angle) * (maxRadius * fraction);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }}
      ctx.closePath();
      ctx.fillStyle = fillColor;
      ctx.fill();
      ctx.strokeStyle = strokeColor;
      ctx.lineWidth = 2.4;
      ctx.stroke();
      ctx.restore();

      // Anchor dots
      for (let i = 0; i < axes.length; i++) {{
        const fraction = Math.max(0.1, vals[i] / 100);
        const x = centerX + Math.cos(axes[i].angle) * (maxRadius * fraction);
        const y = centerY + Math.sin(axes[i].angle) * (maxRadius * fraction);

        ctx.beginPath();
        ctx.arc(x, y, 4, 0, Math.PI * 2);
        ctx.fillStyle = dotColor;
        ctx.fill();
        ctx.strokeStyle = "#ffffff";
        ctx.lineWidth = 2;
        ctx.stroke();
      }}
    }}

    const p1Vals = [BENCHMARK.phase1.recall, BENCHMARK.phase1.precision, BENCHMARK.phase1.faithfulness, BENCHMARK.phase1.relevance];
    const p2Vals = [BENCHMARK.phase2.recall, BENCHMARK.phase2.precision, BENCHMARK.phase2.faithfulness, BENCHMARK.phase2.relevance];
    const p3Vals = [BENCHMARK.phase3.recall, BENCHMARK.phase3.precision, BENCHMARK.phase3.faithfulness, BENCHMARK.phase3.relevance];
    const p4Vals = [BENCHMARK.phase4.recall, BENCHMARK.phase4.precision, BENCHMARK.phase4.faithfulness, BENCHMARK.phase4.relevance];

    // Overlay shapes based on active view
    renderRadarShape(p1Vals, "#94a3b8", "rgba(148, 163, 184, 0.12)", "#64748b", true);
    renderRadarShape(p2Vals, "#818cf8", "rgba(129, 140, 248, 0.18)", "#4f46e5", false);
    renderRadarShape(p3Vals, "#059669", "rgba(5, 150, 105, 0.22)", "#047857", false);
    renderRadarShape(p4Vals, "#7c3aed", "rgba(124, 58, 237, 0.28)", "#6d28d9", false);
  }}

  // Render Modern 4-Way Grouped Bar Chart
  function renderComparativeBarChart() {{
    const container = document.getElementById("barChartContainer");
    container.innerHTML = "";

    BENCHMARK.metricsList.forEach(m => {{
      const p1H = Math.round((m.p1 / 100) * 185);
      const p2H = Math.round((m.p2 / 100) * 185);
      const p3H = Math.round((m.p3 / 100) * 185);
      const p4H = Math.round((m.p4 / 100) * 185);

      const group = document.createElement("div");
      group.className = "bar-group";
      group.innerHTML = `
        <div class="bar-delta-tag">${{m.delta}}</div>
        <div class="bar-quad">
          <div class="bar-stem baseline" style="height: ${{p1H}}px;" title="Phase 1: ${{m.p1}}%">
            <span class="bar-val-text">${{m.p1}}%</span>
          </div>
          <div class="bar-stem hybrid" style="height: ${{p2H}}px;" title="Phase 2: ${{m.p2}}%">
            <span class="bar-val-text" style="color:#6366f1;">${{m.p2}}%</span>
          </div>
          <div class="bar-stem rerank" style="height: ${{p3H}}px;" title="Phase 3: ${{m.p3}}%">
            <span class="bar-val-text" style="color:#059669;">${{m.p3}}%</span>
          </div>
          <div class="bar-stem multi-query" style="height: ${{p4H}}px;" title="Phase 4: ${{m.p4}}%">
            <span class="bar-val-text" style="color:#7c3aed;">${{m.p4}}%</span>
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
      const p4R = (s.p4Recall * 100).toFixed(0);
      const p4P = (s.p4Precision * 100).toFixed(0);
      const p4F = (s.p4Faith * 100).toFixed(0);
      const p4Rel = (s.p4Relevance * 100).toFixed(0);

      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td><span class="qid-pill">${{s.id}}</span></td>
        <td><div class="q-title">${{s.question}}</div></td>
        <td><span class="chip chip-slate">${{p1R}}%</span></td>
        <td><span class="chip chip-blue">${{p2R}}%</span></td>
        <td><span class="chip chip-green">${{p3R}}%</span></td>
        <td><span class="chip chip-purple">${{p4R}}%</span></td>
        <td><b>${{p4P}}%</b></td>
        <td><span class="chip chip-purple">${{p4F}}%</span></td>
        <td><b>${{p4Rel}}%</b></td>
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
    document.getElementById("dmRecall").innerText = (s.p4Recall * 100).toFixed(0) + "%";
    document.getElementById("dmPrecision").innerText = (s.p4Precision * 100).toFixed(0) + "%";
    document.getElementById("dmFaith").innerText = (s.p4Faith * 100).toFixed(0) + "%";
    document.getElementById("dmRelevance").innerText = (s.p4Relevance * 100).toFixed(0) + "%";

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
    const cmd = "python evals/run_eval.py --provider groq --dataset amnesty_qa --mode hybrid_rerank --transform multi_query --limit 20";
    navigator.clipboard.writeText(cmd).then(() => {{
      alert("Copied Phase 4 CLI Benchmark command to clipboard:\\n\\n" + cmd);
    }}).catch(() => {{
      prompt("Copy CLI Benchmark command:", cmd);
    }});
  }}

  // Initialize
  window.addEventListener("DOMContentLoaded", () => {{
    setDashboardView("phase4");
    renderComparativeBarChart();
    renderDiagnosticTable();
  }});
</script>

</body>
</html>
"""

output_path = ROOT / "eval_dashboard.html"
with open(output_path, "w", encoding="utf-8") as f:
    f.write(html_content)

print(f"eval_dashboard.html successfully updated at {output_path}!")
