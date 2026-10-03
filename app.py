import time
import streamlit as st
from pathlib import Path

from src.config import settings
from src.models import get_chat_llm, get_embedding_model
from src.vectorstore import VectorStoreManager
from src.memory import ChatHistoryManager
from src.chain import ConversationalRAGChain
from src.utils import load_file_to_documents, load_amnesty_qa_dataset

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

vsm = init_vectorstore()
memory = init_memory()

# Session State for UI controls
if "current_session_id" not in st.session_state:
    st.session_state.current_session_id = "default"


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
        top_k = st.slider("Top K Chunks", min_value=1, max_value=10, value=4, step=1)

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
    sessions = memory.list_sessions()

    col_sess, col_new = st.columns([3, 1])
    with col_sess:
        current_idx = sessions.index(st.session_state.current_session_id) if st.session_state.current_session_id in sessions else 0
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
            new_id = f"chat_{len(sessions) + 1}"
            st.session_state.current_session_id = new_id
            st.rerun()

    if st.button("🧹 Clear Current History", use_container_width=True):
        memory.clear_session(st.session_state.current_session_id)
        st.rerun()


# =====================================================================
# Main Chat Area
# =====================================================================
st.header("Retrieval-Augmented Generation (RAG) Studio")

col_info1, col_info2, col_info3 = st.columns(3)
with col_info1:
    st.caption(f"**Provider**: `{selected_provider.upper()}`")
with col_info2:
    st.caption(f"**Model**: `{selected_model}`")
with col_info3:
    st.caption(f"**Session**: `{st.session_state.current_session_id}`")

st.markdown("---")

# Load existing messages for this session
messages = memory.get_messages(st.session_state.current_session_id)

for msg in messages:
    role = msg.get("role")
    content = msg.get("content", "")
    sources = msg.get("sources", [])

    with st.chat_message(role):
        st.markdown(content)
        if sources:
            with st.expander(f"🔍 Retrieved Sources ({len(sources)} chunks)", expanded=False):
                for idx, src in enumerate(sources):
                    page_str = f" | Page {src['page']}" if src.get("page") else ""
                    st.markdown(
                        f"<div class='source-card'>"
                        f"<b>Chunk {idx+1}</b> &bull; Source: <code>{src.get('source', 'Unknown')}</code>{page_str} &bull; Distance: <code>{src.get('score', 'N/A')}</code><br/>"
                        f"<i>\"{src.get('content', '')[:300]}...\"</i>"
                        f"</div>",
                        unsafe_allow_html=True
                    )


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
                k=top_k
            )

            # 4. Stream response and render sources
            with st.chat_message("assistant"):
                with st.spinner("Retrieving relevant passages & formulating response..."):
                    standalone_q, sources_info, context_str = rag_chain.retrieve_context(
                        question=user_query,
                        session_id=st.session_state.current_session_id
                    )

                # Show standalone question if query was reformulated
                if standalone_q != user_query:
                    st.caption(f"🔍 *Search query reformulated to: \"{standalone_q}\"*")

                # Stream token-by-token
                history = memory.get_langchain_messages(st.session_state.current_session_id, limit=6)
                qa_chain = rag_chain.llm

                from src.chain import QA_PROMPT
                from langchain_core.output_parsers import StrOutputParser

                stream_runnable = QA_PROMPT | qa_chain | StrOutputParser()

                def generate_response():
                    for chunk in stream_runnable.stream({
                        "context": context_str,
                        "chat_history": history,
                        "question": user_query
                    }):
                        yield chunk

                full_answer = st.write_stream(generate_response)

                # Display retrieved sources
                if sources_info:
                    with st.expander(f"🔍 Retrieved Sources ({len(sources_info)} chunks)", expanded=False):
                        for idx, src in enumerate(sources_info):
                            page_str = f" | Page {src['page']}" if src.get("page") else ""
                            st.markdown(
                                f"<div class='source-card'>"
                                f"<b>Chunk {idx+1}</b> &bull; Source: <code>{src.get('source', 'Unknown')}</code>{page_str} &bull; Distance: <code>{src.get('score', 'N/A')}</code><br/>"
                                f"<i>\"{src.get('content', '')[:300]}...\"</i>"
                                f"</div>",
                                unsafe_allow_html=True
                            )

                # Persist turn in conversational memory
                memory.add_message(st.session_state.current_session_id, role="user", content=user_query)
                memory.add_message(st.session_state.current_session_id, role="assistant", content=full_answer, sources=sources_info)

        except Exception as e:
            with st.chat_message("assistant"):
                st.error(f"Error during RAG execution: {str(e)}")
