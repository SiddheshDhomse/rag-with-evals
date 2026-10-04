import time
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
<style>
    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }
    .metric-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 16px;
        font-size: 0.85rem;
        font-weight: 600;
        background-color: #f0f2f6;
        color: #31333F;
        margin-right: 8px;
    }
    .source-card {
        border-left: 3px solid #4CAF50;
        padding: 8px 12px;
        margin-bottom: 8px;
        background-color: rgba(76, 175, 80, 0.05);
        border-radius: 0 4px 4px 0;
        font-size: 0.85rem;
    }
    .chunk-card-selected {
        border-left: 4px solid #10b981;
        background-color: rgba(16, 185, 129, 0.05);
        padding: 10px 14px;
        border-radius: 0 8px 8px 0;
        margin-bottom: 10px;
    }
    .chunk-card-filtered {
        border-left: 4px solid #94a3b8;
        background-color: rgba(148, 163, 184, 0.05);
        padding: 10px 14px;
        border-radius: 0 8px 8px 0;
        margin-bottom: 10px;
        opacity: 0.88;
    }
    .rank-badge-up {
        background-color: #d1fae5;
        color: #065f46;
        padding: 2px 8px;
        border-radius: 12px;
        font-weight: 700;
        font-size: 0.8rem;
    }
    .rank-badge-down {
        background-color: #fee2e2;
        color: #991b1b;
        padding: 2px 8px;
        border-radius: 12px;
        font-weight: 700;
        font-size: 0.8rem;
    }
    .rank-badge-same {
        background-color: #f3f4f6;
        color: #4b5563;
        padding: 2px 8px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.8rem;
    }
    .status-ok { color: #2e7d32; font-weight: bold; }
    .status-missing { color: #c62828; font-weight: bold; }
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
            # 1. High-level Summary Metrics
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Stage 1 Candidates", num_candidates, help="Candidate chunks retrieved via Hybrid BM25+Dense RRF")
            c2.metric("Selected for LLM", num_selected, help="Top chunks with highest Cross-Encoder scores passed to prompt")
            filtered_count = max(0, num_candidates - num_selected)
            c3.metric("Noise Filtered", filtered_count, help="Low-relevance chunks discarded to prevent hallucination/distraction")
            promoted_count = sum(1 for c in candidates_audit if c.get("rank_delta", 0) > 0)
            c4.metric("Rerank Shifts", f"{promoted_count} Promoted", help="Chunks boosted to higher ranks by Cross-Encoder cross-attention")

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
                        "<span style='color: #059669; font-weight: bold;'>✅ INCLUDED IN LLM CONTEXT</span>"
                        if is_sel
                        else "<span style='color: #64748b;'>🚫 FILTERED OUT (DISTRACTOR)</span>"
                    )
                    page_str = f" | Page {cand['page']}" if cand.get("page") else ""

                    st.markdown(
                        f"<div class='{card_cls}'>"
                        f"<b>Rank {cand.get('new_rank')}</b> (Initial: #{cand.get('initial_rank')} <span class='{badge_cls}'>{delta_text}</span>) &bull; "
                        f"{status_badge}<br/>"
                        f"<small>Source: <code>{cand.get('source', 'Unknown')}</code>{page_str} &bull; "
                        f"Cross-Encoder Score: <b>{cand.get('rerank_score', 0.0):+.4f}</b> &bull; "
                        f"Confidence: <b>{cand.get('confidence_pct', 0.0)}%</b> &bull; "
                        f"Stage 1 Score: <code>{cand.get('initial_score', 'N/A')}</code></small><br/>"
                        f"<div style='margin-top: 6px; font-size: 0.88rem; color: #334155; line-height: 1.45;'>"
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
                    f"<b>Chunk {idx+1}</b> &bull; Source: <code>{src.get('source', 'Unknown')}</code>{page_str} &bull; {score_label}: <code>{src.get('score', 'N/A')}</code><br/>"
                    f"<i>\"{src.get('content', '')[:300]}...\"</i>"
                    f"</div>",
                    unsafe_allow_html=True
                )


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
st.header("Retrieval-Augmented Generation (RAG) Studio")

col_info1, col_info2, col_info3, col_info4 = st.columns(4)
with col_info1:
    st.caption(f"**Provider**: `{selected_provider.upper()}`")
with col_info2:
    st.caption(f"**Model**: `{selected_model}`")
with col_info3:
    mode_label = "HYBRID + RERANK (PHASE 3)" if "Rerank" in retrieval_mode_selection else ("HYBRID (PHASE 2)" if "Hybrid" in retrieval_mode_selection else "DENSE (PHASE 1)")
    st.caption(f"**Strategy**: `{mode_label}`")
with col_info4:
    st.caption(f"**Session**: `{st.session_state.current_session_id}`")

st.markdown("---")

# Load existing messages for this session
messages = memory.get_messages(st.session_state.current_session_id)

for msg in messages:
    role = msg.get("role")
    content = msg.get("content", "")
    sources = msg.get("sources", [])
    candidates = msg.get("candidates", [])

    with st.chat_message(role):
        st.markdown(content)
        if sources or candidates:
            render_retrieval_and_reranking_inspection(sources, candidates, selected_retrieval_mode)


# =====================================================================
# User Chat Input & Execution
# =====================================================================
user_query = st.chat_input("Ask a question about your knowledge base...")

if user_query:
    # 1. Render user message in chat
    with st.chat_message("user"):
        st.markdown(user_query)

    # 2. Check prerequisites
    if not is_valid:
        with st.chat_message("assistant"):
            st.error(f"Cannot generate answer: {validation_msg}. Please configure your API key in the `.env` file.")
    elif kb_stats["total_chunks"] == 0:
        with st.chat_message("assistant"):
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
                reranker=reranker
            )

            # 4. Two-Stage Retrieval & Answer Streaming
            with st.chat_message("assistant"):
                spinner_text = (
                    f"Retrieving {candidates_k} candidate passages & reranking with Cross-Encoder..."
                    if "Rerank" in retrieval_mode_selection
                    else "Retrieving relevant passages & formulating response..."
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
                stream_runnable = QA_PROMPT | rag_chain.llm | StrOutputParser()

                def generate_response():
                    for chunk in stream_runnable.stream({
                        "context": context_str,
                        "chat_history": history,
                        "question": user_query
                    }):
                        yield chunk

                full_answer = st.write_stream(generate_response)

                # Display retrieved sources & candidate reranking inspection
                render_retrieval_and_reranking_inspection(sources_info, candidates_audit, selected_retrieval_mode)

                # Persist turn in conversational memory with sources & candidates audit
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
                    candidates=candidates_audit
                )

        except Exception as e:
            with st.chat_message("assistant"):
                st.error(f"Error during RAG execution: {str(e)}")
