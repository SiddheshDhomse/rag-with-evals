import time
from typing import Dict, Any, List, Optional
import streamlit as st
import pandas as pd
from pathlib import Path

from src.config import settings
from src.models import get_chat_llm, get_embedding_model
from src.vectorstore import VectorStoreManager
from src.memory import ChatHistoryManager
from src.chain import ConversationalRAGChain, QA_PROMPT
from src.reranker import CrossEncoderReranker
from src.utils import load_file_to_documents, load_amnesty_qa_dataset
from langchain_core.output_parsers import StrOutputParser

# =====================================================================
# Page Configuration & Styling
# =====================================================================
st.set_page_config(
    page_title="RAG Studio | Production Chatbot",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished production UI
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap" rel="stylesheet">

<style>
    * {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    code, pre, .mono {
        font-family: 'JetBrains Mono', monospace !important;
    }
    .main .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2.5rem;
        max-width: 1200px;
    }

    /* Top Studio Hero Banner */
    .studio-header {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.5) 0%, rgba(15, 23, 42, 0.7) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 20px 24px;
        margin-bottom: 20px;
        backdrop-filter: blur(12px);
        box-shadow: 0 4px 24px rgba(0, 0, 0, 0.2);
    }
    .studio-title-wrap {
        display: flex;
        align-items: center;
        gap: 14px;
        margin-bottom: 12px;
    }
    .studio-logo-icon {
        font-size: 1.8rem;
        width: 44px;
        height: 44px;
        display: flex;
        align-items: center;
        justify-content: center;
        background: linear-gradient(135deg, #4f46e5 0%, #06b6d4 100%);
        border-radius: 12px;
        box-shadow: 0 4px 14px rgba(79, 70, 229, 0.35);
    }
    .studio-title {
        font-size: 1.45rem;
        font-weight: 800;
        letter-spacing: -0.025em;
        color: #f8fafc;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .studio-badge-pro {
        font-size: 0.65rem;
        font-weight: 800;
        letter-spacing: 0.06em;
        background: rgba(99, 102, 241, 0.2);
        color: #a5b4fc;
        border: 1px solid rgba(99, 102, 241, 0.4);
        padding: 2px 7px;
        border-radius: 6px;
    }
    .studio-subtitle {
        font-size: 0.82rem;
        color: #94a3b8;
        margin-top: 1px;
    }
    .status-ribbon {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        padding-top: 12px;
        border-top: 1px solid rgba(255, 255, 255, 0.06);
    }
    .status-chip {
        display: flex;
        align-items: center;
        gap: 6px;
        background: rgba(15, 23, 42, 0.65);
        border: 1px solid rgba(255, 255, 255, 0.08);
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.78rem;
        color: #cbd5e1;
    }
    .chip-dot {
        width: 6px;
        height: 6px;
        border-radius: 50%;
    }
    .dot-green { background: #10b981; box-shadow: 0 0 8px rgba(16, 185, 129, 0.5); }
    .dot-blue { background: #3b82f6; box-shadow: 0 0 8px rgba(59, 130, 246, 0.5); }
    .dot-indigo { background: #6366f1; box-shadow: 0 0 8px rgba(99, 102, 241, 0.5); }
    .dot-purple { background: #a855f7; box-shadow: 0 0 8px rgba(168, 85, 247, 0.5); }
    .dot-amber { background: #f59e0b; box-shadow: 0 0 8px rgba(245, 158, 11, 0.5); }

    /* Stat Cards Grid */
    .stat-card-row {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 12px;
        margin: 10px 0 16px 0;
    }
    .stat-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 14px 16px;
        transition: all 0.2s ease;
    }
    .stat-card:hover {
        background: rgba(255, 255, 255, 0.05);
        border-color: rgba(255, 255, 255, 0.15);
        transform: translateY(-1px);
    }
    .stat-card-title {
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.07em;
        text-transform: uppercase;
        color: #94a3b8;
        margin-bottom: 6px;
    }
    .stat-card-val {
        font-size: 1.6rem;
        font-weight: 800;
        line-height: 1.1;
        letter-spacing: -0.02em;
        margin-bottom: 4px;
        font-family: 'JetBrains Mono', monospace;
    }
    .stat-card-sub {
        font-size: 0.75rem;
        color: #64748b;
    }
    .text-indigo { color: #818cf8; }
    .text-emerald { color: #34d399; }
    .text-slate { color: #94a3b8; }
    .text-amber { color: #fbbf24; }
    .text-purple { color: #c084fc; }

    /* Chunk Inspection Cards */
    .chunk-card-selected {
        border-left: 3px solid #10b981;
        background: linear-gradient(90deg, rgba(16, 185, 129, 0.08) 0%, rgba(15, 23, 42, 0.3) 100%);
        border-top: 1px solid rgba(16, 185, 129, 0.2);
        border-right: 1px solid rgba(255, 255, 255, 0.04);
        border-bottom: 1px solid rgba(255, 255, 255, 0.04);
        padding: 12px 16px;
        border-radius: 0 10px 10px 0;
        margin-bottom: 12px;
    }
    .chunk-card-filtered {
        border-left: 3px solid #64748b;
        background: rgba(15, 23, 42, 0.25);
        border-top: 1px solid rgba(255, 255, 255, 0.03);
        border-right: 1px solid rgba(255, 255, 255, 0.03);
        border-bottom: 1px solid rgba(255, 255, 255, 0.03);
        padding: 12px 16px;
        border-radius: 0 10px 10px 0;
        margin-bottom: 12px;
        opacity: 0.8;
    }
    .rank-badge-up {
        background-color: rgba(16, 185, 129, 0.2);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.3);
        padding: 2px 7px;
        border-radius: 10px;
        font-weight: 700;
        font-size: 0.75rem;
        font-family: 'JetBrains Mono', monospace;
    }
    .rank-badge-down {
        background-color: rgba(239, 68, 68, 0.2);
        color: #f87171;
        border: 1px solid rgba(239, 68, 68, 0.3);
        padding: 2px 7px;
        border-radius: 10px;
        font-weight: 700;
        font-size: 0.75rem;
        font-family: 'JetBrains Mono', monospace;
    }
    .rank-badge-same {
        background-color: rgba(255, 255, 255, 0.06);
        color: #94a3b8;
        padding: 2px 7px;
        border-radius: 10px;
        font-weight: 600;
        font-size: 0.75rem;
        font-family: 'JetBrains Mono', monospace;
    }
    .status-ok { color: #10b981; font-weight: bold; }
    .status-missing { color: #f87171; font-weight: bold; }
    .metric-badge {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 8px;
        padding: 6px 12px;
        margin-bottom: 6px;
        font-size: 0.82rem;
        color: #cbd5e1;
    }
    .metric-badge b {
        color: #38bdf8;
    }
    .source-card {
        background: rgba(15, 23, 42, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-left: 3px solid #3b82f6;
        border-radius: 0 8px 8px 0;
        padding: 10px 14px;
        margin-bottom: 8px;
        font-size: 0.85rem;
        color: #cbd5e1;
    }
</style>
""", unsafe_allow_html=True)


# =====================================================================
# State & Cache Initialization
# =====================================================================
@st.cache_resource(show_spinner=False)
def init_vectorstore():
    return VectorStoreManager()

@st.cache_resource(show_spinner=False)
def init_memory():
    return ChatHistoryManager()

@st.cache_resource(show_spinner=False)
def init_reranker():
    return CrossEncoderReranker(model_name=settings.reranker_model_name)

vsm = init_vectorstore()
memory = init_memory()
reranker = init_reranker()

# Session State for UI controls
if "current_session_id" not in st.session_state:
    st.session_state.current_session_id = "default"


# =====================================================================
# Helper: Candidate Chunks & Reranker Diagnostics Renderer
# =====================================================================
def render_retrieval_and_reranking_inspection(sources_info, candidates_audit, retrieval_mode):
    if not sources_info and not candidates_audit:
        return

    num_selected = len(sources_info)
    num_candidates = len(candidates_audit) if candidates_audit else num_selected
    is_reranked = "rerank" in str(retrieval_mode).lower() or any(c.get("rank_delta") != 0 for c in (candidates_audit or []))

    title = (
        f"🔀 Retrieval & Reranker Diagnostics ({num_selected} in LLM Context | {num_candidates} Candidates Analyzed)"
        if is_reranked
        else f"🔍 Retrieved Sources ({num_selected} chunks)"
    )

    with st.expander(title, expanded=False):
        if is_reranked and candidates_audit:
            # 1. High-level Summary Metrics using Custom Frosted Glass Stat Cards
            filtered_count = max(0, num_candidates - num_selected)
            promoted_count = sum(1 for c in candidates_audit if c.get("rank_delta", 0) > 0)

            st.markdown(
                f"""
                <div class="stat-card-row">
                    <div class="stat-card">
                        <div class="stat-card-title">Stage 1 Candidates</div>
                        <div class="stat-card-val text-indigo">{num_candidates}</div>
                        <div class="stat-card-sub">Hybrid BM25+Dense RRF</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-card-title">Selected for LLM</div>
                        <div class="stat-card-val text-emerald">{num_selected}</div>
                        <div class="stat-card-sub">Cross-Encoder Top Chunks</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-card-title">Distractors Filtered</div>
                        <div class="stat-card-val text-slate">{filtered_count}</div>
                        <div class="stat-card-sub">Low Relevance Discarded</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-card-title">Rerank Shifts</div>
                        <div class="stat-card-val text-amber">{promoted_count} Promoted</div>
                        <div class="stat-card-sub">Cross-Attention Re-ordered</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            # 2. Tabs: Candidate Matrix vs Detailed Chunk View
            tab_matrix, tab_cards = st.tabs(["📊 Candidate Reranking Matrix", "📄 Detailed Chunk Inspection"])

            with tab_matrix:
                matrix_rows = []
                for cand in candidates_audit:
                    delta = cand.get("rank_delta", 0)
                    if delta > 0:
                        shift_str = f"▲ +{delta}"
                    elif delta < 0:
                        shift_str = f"▼ {delta}"
                    else:
                        shift_str = "• 0"

                    status_str = "✅ Selected (LLM Context)" if cand.get("selected") else "🚫 Filtered Out"
                    rank_val = cand.get("new_rank")
                    rank_icon = (
                        "🥇 #1" if rank_val == 1
                        else ("🥈 #2" if rank_val == 2
                        else ("🥉 #3" if rank_val == 3
                        else f"#{rank_val}"))
                    )

                    matrix_rows.append({
                        "Reranked #": rank_icon,
                        "Shift": shift_str,
                        "Initial #": f"#{cand.get('initial_rank')}",
                        "Cross-Encoder Score": f"{cand.get('rerank_score', 0.0):+.4f}",
                        "Semantic Confidence": f"{cand.get('confidence_pct', 0.0):.1f}%",
                        "Stage 1 Score": f"{cand.get('initial_score', 0.0):.4f}",
                        "Status": status_str,
                        "Source": f"{cand.get('source', 'Unknown')}" + (f" (p.{cand.get('page')})" if cand.get("page") else "")
                    })
                df_matrix = pd.DataFrame(matrix_rows)
                st.dataframe(df_matrix, use_container_width=True, hide_index=True)

            with tab_cards:
                for cand in candidates_audit:
                    is_sel = cand.get("selected", False)
                    card_cls = "chunk-card-selected" if is_sel else "chunk-card-filtered"
                    delta = cand.get("rank_delta", 0)
                    badge_cls = "rank-badge-up" if delta > 0 else ("rank-badge-down" if delta < 0 else "rank-badge-same")
                    delta_text = f"▲ +{delta}" if delta > 0 else (f"▼ {delta}" if delta < 0 else "• 0")
                    status_badge = (
                        "<span style='color: #10b981; font-weight: bold;'>✅ INCLUDED IN LLM CONTEXT</span>"
                        if is_sel
                        else "<span style='color: #94a3b8; font-weight: 500;'>🚫 FILTERED OUT (DISTRACTOR)</span>"
                    )
                    page_str = f" | Page {cand['page']}" if cand.get("page") else ""

                    st.markdown(
                        f"<div class='{card_cls}'>"
                        f"<b>Rank {cand.get('new_rank')}</b> (Initial: #{cand.get('initial_rank')} <span class='{badge_cls}'>{delta_text}</span>) &bull; "
                        f"{status_badge}<br/>"
                        f"<small style='color: #94a3b8;'>Source: <code style='color: #38bdf8;'>{cand.get('source', 'Unknown')}</code>{page_str} &bull; "
                        f"Cross-Encoder Score: <b>{cand.get('rerank_score', 0.0):+.4f}</b> &bull; "
                        f"Confidence: <b>{cand.get('confidence_pct', 0.0)}%</b> &bull; "
                        f"Stage 1 Score: <code>{cand.get('initial_score', 'N/A')}</code></small><br/>"
                        f"<div style='margin-top: 8px; font-size: 0.88rem; color: #cbd5e1; line-height: 1.5;'>"
                        f"<i>\"{cand.get('content', '')[:380]}...\"</i>"
                        f"</div>"
                        f"</div>",
                        unsafe_allow_html=True
                    )
        else:
            # Fallback for baseline dense or hybrid without reranker
            for idx, src in enumerate(sources_info):
                page_str = f" | Page {src['page']}" if src.get("page") else ""
                score_label = "RRF Score" if src.get("score_type") == "rrf_score" else "Distance"
                st.markdown(
                    f"<div class='source-card'>"
                    f"<b>Chunk {idx+1}</b> &bull; Source: <code style='color: #38bdf8;'>{src.get('source', 'Unknown')}</code>{page_str} &bull; {score_label}: <code>{src.get('score', 'N/A')}</code><br/>"
                    f"<div style='margin-top: 6px; font-size: 0.88rem; color: #cbd5e1; line-height: 1.45;'>"
                    f"<i>\"{src.get('content', '')[:300]}...\"</i>"
                    f"</div>"
                    f"</div>",
                    unsafe_allow_html=True
                )


def render_query_transformation_audit(transform_audit: Dict[str, Any]):
    """Renders Phase 4 Query Transformation and Adaptive Routing transparency audit."""
    if not transform_audit:
        return

    strategy = transform_audit.get("strategy", "standard")
    route = transform_audit.get("route", "FACT_LOOKUP")
    reasoning = transform_audit.get("reasoning", "")
    is_direct = transform_audit.get("direct_bypass", False)

    if is_direct:
        st.markdown(
            f"<div style='background: rgba(16, 185, 129, 0.12); border: 1px solid rgba(16, 185, 129, 0.35); border-radius: 10px; padding: 12px 16px; margin: 10px 0;'>"
            f"<b>⚡ Adaptive Intent Route: <span style='color: #34d399;'>DIRECT BYPASS (0ms Retrieval)</span></b><br/>"
            f"<small style='color: #94a3b8;'>{reasoning}</small>"
            f"</div>",
            unsafe_allow_html=True
        )
        return

    if strategy in ("hyde", "multi_query", "step_back") or transform_audit.get("mode") == "adaptive":
        strategy_icon = {
            "hyde": "🧠 HyDE (Hypothetical Document Embeddings)",
            "multi_query": "🔀 Multi-Query Decomposition",
            "step_back": "🔭 Step-Back Principle Retrieval",
            "standard": "⚡ Standard Direct Retrieval"
        }.get(strategy, "🔮 Query Transformation")

        title = f"{strategy_icon} &bull; Route: `{route}`"

        with st.expander(title, expanded=False):
            if reasoning:
                st.markdown(f"**Router Assessment**: *{reasoning}*")

            if strategy == "hyde" and transform_audit.get("hypothetical_doc"):
                st.markdown("**Synthetic Document Drafted (bridging vocabulary gap)**:")
                st.info(transform_audit["hypothetical_doc"])

            elif strategy == "multi_query" and transform_audit.get("sub_queries"):
                st.markdown("**Parallel Sub-Queries Deconstructed & Fused**:")
                for i, sq in enumerate(transform_audit["sub_queries"]):
                    st.markdown(f"- **Sub-Query {i+1}**: `{sq}`")

            elif strategy == "step_back" and transform_audit.get("step_back_query"):
                st.markdown("**High-Level Conceptual Step-Back Query**:")
                st.markdown(f"- 🏛️ `{transform_audit['step_back_query']}`")


# =====================================================================
# Sidebar: Model, Knowledge Base & Session Controls
# =====================================================================
with st.sidebar:
    st.title("⚙️ RAG Studio")
    st.caption("Production RAG with Multi-Provider Support & Evals")

    st.markdown("---")

    # 1. Model Provider Settings
    st.subheader("🤖 LLM Configuration")

    provider_options = ["groq", "openrouter", "nvidia", "ollama"]
    selected_provider = st.selectbox(
        "Provider",
        options=provider_options,
        index=provider_options.index(settings.default_llm_provider) if settings.default_llm_provider in provider_options else 0,
        format_func=lambda x: {
            "groq": "Groq Cloud (Ultra Fast)",
            "openrouter": "OpenRouter (Free & Multi-model)",
            "nvidia": "NVIDIA NIM",
            "ollama": "Ollama (Local)"
        }[x]
    )

    # Provider Key Status Badge
    is_valid, validation_msg = settings.validate_provider(selected_provider)
    if is_valid:
        st.markdown(f"<span class='status-ok'>✓ {validation_msg}</span>", unsafe_allow_html=True)
    else:
        st.markdown(f"<span class='status-missing'>⚠ {validation_msg}</span>", unsafe_allow_html=True)
        st.info("💡 Add your key in `.env` and reload.")

    # Model Dropdown based on Provider
    provider_models = settings.available_models.get(selected_provider, [])
    selected_model = st.selectbox("Model", options=provider_models, index=0)

    col_temp, col_k = st.columns(2)
    with col_temp:
        temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.1, step=0.05)
    with col_k:
        top_k = st.slider("Final Top K Chunks", min_value=1, max_value=10, value=4, step=1)

    # Retrieval Strategy Selector (Phases 1, 2, 3)
    retrieval_strategy_options = [
        "Hybrid + Cross-Encoder Rerank (Phase 3)",
        "Hybrid Search (BM25 + Dense RRF - Phase 2)",
        "Dense Only (Vector Baseline - Phase 1)"
    ]
    retrieval_mode_selection = st.radio(
        "Retrieval Strategy",
        options=retrieval_strategy_options,
        index=0,
        help="Phase 3 Two-Stage Reranking retrieves broad candidate chunks and re-scores with cross-attention for maximum precision."
    )

    if "Rerank" in retrieval_mode_selection:
        selected_retrieval_mode = "hybrid_rerank"
        candidates_k = st.slider(
            "Stage 1 Candidates (M)",
            min_value=max(top_k, 6),
            max_value=25,
            value=max(top_k, 15),
            step=1,
            help="High-recall candidate chunks retrieved via BM25 + Dense RRF before Cross-Encoder scoring."
        )
    elif "Hybrid" in retrieval_mode_selection:
        selected_retrieval_mode = "hybrid"
        candidates_k = top_k
    else:
        selected_retrieval_mode = "dense"
        candidates_k = top_k

    # Phase 4: Query Transformation Strategy Selector
    transform_strategy_options = [
        "Standard (Direct Query)",
        "Adaptive Auto-Router (Auto-Detect Intent & Direct Bypass)",
        "HyDE (Hypothetical Document Embeddings)",
        "Multi-Query (Sub-Query Decomposition)",
        "Step-Back (Conceptual Principle Retrieval)"
    ]
    transform_selection = st.selectbox(
        "🔮 Query Transformation (Phase 4)",
        options=transform_strategy_options,
        index=0,
        help="Advanced query transformation to bridge vocabulary gaps, deconstruct multi-hop queries, or route chitchat directly."
    )
    if "Adaptive" in transform_selection:
        selected_transform_mode = "adaptive"
    elif "HyDE" in transform_selection:
        selected_transform_mode = "hyde"
    elif "Multi-Query" in transform_selection:
        selected_transform_mode = "multi_query"
    elif "Step-Back" in transform_selection:
        selected_transform_mode = "step_back"
    else:
        selected_transform_mode = "none"

    st.markdown("---")

    # 2. Knowledge Base Management
    st.subheader("📚 Knowledge Base")
    kb_stats = vsm.get_collection_stats()

    st.markdown(
        f"<div class='metric-badge'>Total Chunks: <b>{kb_stats['total_chunks']}</b></div>"
        f"<div class='metric-badge'>Sources: <b>{kb_stats['sources_count']}</b></div>",
        unsafe_allow_html=True
    )

    if kb_stats["sources"]:
        with st.expander("Indexed Files", expanded=False):
            for s in kb_stats["sources"]:
                st.write(f"• `{s}`")

    # Upload custom documents
    uploaded_files = st.file_uploader(
        "Upload Documents",
        type=["pdf", "txt", "md", "docx"],
        accept_multiple_files=True
    )

    if uploaded_files:
        if st.button("📥 Index Uploaded Files", use_container_width=True):
            with st.spinner("Processing & indexing documents into ChromaDB..."):
                all_docs = []
                for uf in uploaded_files:
                    docs = load_file_to_documents(uf, uf.name)
                    all_docs.extend(docs)
                num_chunks = vsm.add_documents(all_docs, chunk_size=600, chunk_overlap=100)
                st.success(f"Indexed {num_chunks} chunks successfully!")
                time.sleep(1)
                st.rerun()

    # Quick Seed with Benchmark Dataset
    if st.button("⚡ Seed Amnesty QA Dataset (Benchmark)", use_container_width=True):
        with st.spinner("Downloading and indexing 'explodinggradients/amnesty_qa'..."):
            try:
                data = load_amnesty_qa_dataset(split="eval")
                docs = data["documents"][:40]
                num_chunks = vsm.add_documents(docs, chunk_size=500, chunk_overlap=80)
                st.success(f"Successfully loaded & indexed {num_chunks} chunks from Amnesty QA!")
                time.sleep(1)
                st.rerun()
            except Exception as e:
                st.error(f"Failed to load dataset: {e}")

    if st.button("🗑️ Clear Vector Database", use_container_width=True):
        vsm.clear_collection()
        st.warning("Knowledge Base cleared.")
        time.sleep(1)
        st.rerun()

    st.markdown("---")

    # 3. Session & Chat History Management
    st.subheader("💬 Chat Sessions")
    sessions = memory.list_sessions(include_internal=False)
    if st.session_state.current_session_id not in sessions:
        sessions.append(st.session_state.current_session_id)
        sessions = sorted(list(set(sessions)))

    col_sess, col_new = st.columns([3, 1])
    with col_sess:
        current_idx = sessions.index(st.session_state.current_session_id)
        selected_session = st.selectbox(
            "Select Session",
            options=sessions,
            index=current_idx,
            label_visibility="collapsed"
        )
        if selected_session != st.session_state.current_session_id:
            st.session_state.current_session_id = selected_session
            st.rerun()

    with col_new:
        if st.button("➕ New", help="Start a new chat session"):
            new_id = memory.create_new_session()
            st.session_state.current_session_id = new_id
            st.rerun()

    col_clear, col_del = st.columns(2)
    with col_clear:
        if st.button("🧹 Clear", help="Clear messages in this session", use_container_width=True):
            memory.clear_session(st.session_state.current_session_id)
            st.rerun()
    with col_del:
        is_default = (st.session_state.current_session_id == "default")
        if st.button("🗑️ Delete", help="Delete this session", use_container_width=True, disabled=is_default):
            memory.delete_session(st.session_state.current_session_id)
            st.session_state.current_session_id = "default"
            st.rerun()


# =====================================================================
# Main Chat Area
# =====================================================================
mode_label = "HYBRID + RERANK (P3)" if "Rerank" in retrieval_mode_selection else ("HYBRID (P2)" if "Hybrid" in retrieval_mode_selection else "DENSE (P1)")
transform_badge = selected_transform_mode.upper() if selected_transform_mode != "none" else "STANDARD"

st.markdown(
    f"""
    <div class="studio-header">
        <div class="studio-title-wrap">
            <div class="studio-logo-icon">🧠</div>
            <div>
                <div class="studio-title">
                    Enterprise RAG Studio
                    <span class="studio-badge-pro">PHASE 4 ACTIVE</span>
                </div>
                <div class="studio-subtitle">
                    Production Retrieval-Augmented Generation with Two-Stage Cross-Encoder Reranking & Adaptive Query Routing
                </div>
            </div>
        </div>
        <div class="status-ribbon">
            <div class="status-chip">
                <span class="chip-dot dot-blue"></span>
                <span>Provider: <b>{selected_provider.upper()}</b></span>
            </div>
            <div class="status-chip">
                <span class="chip-dot dot-indigo"></span>
                <span>Model: <code>{selected_model}</code></span>
            </div>
            <div class="status-chip">
                <span class="chip-dot dot-green"></span>
                <span>Retrieval: <b>{mode_label}</b></span>
            </div>
            <div class="status-chip">
                <span class="chip-dot dot-purple"></span>
                <span>Transform: <b>{transform_badge}</b></span>
            </div>
            <div class="status-chip">
                <span class="chip-dot dot-amber"></span>
                <span>Session: <code>{st.session_state.current_session_id}</code></span>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# Load existing messages for this session
messages = memory.get_messages(st.session_state.current_session_id)

for msg in messages:
    role = msg.get("role")
    content = msg.get("content", "")
    sources = msg.get("sources", [])
    candidates = msg.get("candidates", [])
    transform_audit = msg.get("transform_audit", {})

    with st.chat_message(role, avatar="👤" if role == "user" else "⚡"):
        st.markdown(content)
        if transform_audit:
            render_query_transformation_audit(transform_audit)
        if (sources or candidates) and not transform_audit.get("direct_bypass"):
            render_retrieval_and_reranking_inspection(sources, candidates, selected_retrieval_mode)


# =====================================================================
# User Chat Input & Execution
# =====================================================================
user_query = st.chat_input("Ask a question about your knowledge base...")

if user_query:
    # 1. Render user message in chat
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_query)

    # 2. Check prerequisites
    if not is_valid:
        with st.chat_message("assistant", avatar="⚡"):
            st.error(f"Cannot generate answer: {validation_msg}. Please configure your API key in the `.env` file.")
    elif kb_stats["total_chunks"] == 0:
        with st.chat_message("assistant", avatar="⚡"):
            st.warning("Your Knowledge Base is currently empty! Please upload a document or click **'Seed Amnesty QA Dataset'** in the sidebar to start asking questions.")
    else:
        # 3. Initialize LLM & RAG Chain
        try:
            llm = get_chat_llm(
                provider=selected_provider,
                model_name=selected_model,
                temperature=temperature,
                streaming=True
            )
            rag_chain = ConversationalRAGChain(
                llm=llm,
                vectorstore_manager=vsm,
                memory_manager=memory,
                k=top_k,
                retrieval_mode=selected_retrieval_mode,
                candidates_k=candidates_k,
                reranker=reranker,
                query_transform_mode=selected_transform_mode
            )

            # 4. Two-Stage Retrieval & Answer Streaming
            with st.chat_message("assistant", avatar="⚡"):
                spinner_text = (
                    f"Applying {selected_transform_mode.upper()} transformation & retrieving {candidates_k} candidate passages..."
                    if selected_transform_mode != "none"
                    else (
                        f"Retrieving {candidates_k} candidate passages & reranking with Cross-Encoder..."
                        if "Rerank" in retrieval_mode_selection
                        else "Retrieving relevant passages & formulating response..."
                    )
                )
                with st.spinner(spinner_text):
                    standalone_q, sources_info, context_str, candidates_audit = rag_chain.retrieve_context(
                        question=user_query,
                        session_id=st.session_state.current_session_id,
                        return_candidates=True
                    )

                # Show standalone question if query was reformulated
                if standalone_q != user_query:
                    st.caption(f"🔍 *Search query reformulated to: \"{standalone_q}\"*")

                # Stream token-by-token
                history = memory.get_langchain_messages(st.session_state.current_session_id, limit=6)

                if rag_chain.last_transform_audit.get("direct_bypass"):
                    from src.chain import DIRECT_PROMPT
                    stream_runnable = DIRECT_PROMPT | rag_chain.llm | StrOutputParser()
                    def generate_response():
                        for chunk in stream_runnable.stream({
                            "chat_history": history,
                            "question": user_query
                        }):
                            yield chunk
                else:
                    stream_runnable = QA_PROMPT | rag_chain.llm | StrOutputParser()
                    def generate_response():
                        for chunk in stream_runnable.stream({
                            "context": context_str,
                            "chat_history": history,
                            "question": user_query
                        }):
                            yield chunk

                full_answer = st.write_stream(generate_response)

                # Display query transformation audit
                render_query_transformation_audit(rag_chain.last_transform_audit)

                # Display retrieved sources & candidate reranking inspection (if not direct bypass)
                if not rag_chain.last_transform_audit.get("direct_bypass"):
                    render_retrieval_and_reranking_inspection(sources_info, candidates_audit, selected_retrieval_mode)

                # Persist turn in conversational memory with sources, candidates & transform audit
                memory.add_message(
                    st.session_state.current_session_id,
                    role="user",
                    content=user_query
                )
                memory.add_message(
                    st.session_state.current_session_id,
                    role="assistant",
                    content=full_answer,
                    sources=sources_info,
                    candidates=candidates_audit,
                    transform_audit=rag_chain.last_transform_audit
                )

        except Exception as e:
            with st.chat_message("assistant", avatar="⚡"):
                st.error(f"Error during RAG execution: {str(e)}")
