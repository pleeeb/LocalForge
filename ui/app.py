import os
import uuid

import streamlit as st

from graph.checkpointer import SqliteDatabase, SyncCheckpointer
from graph.graph import compile_graph, create_graph
from graph.schema import State
from graph.store import SyncStoreService
from vector_store.chroma import VectorStoreProvider
from ingestion.pipeline import DocumentPipelineManager
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LocalForge",
    page_icon="⚙",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@300;400;500&display=swap');

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    background-color: #0e0f11;
    color: #c9cdd4;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.5rem 2rem 2rem 2rem; max-width: 100%; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: #13151a;
    border-right: 1px solid #1e2027;
}
[data-testid="stSidebar"] .block-container { padding: 1.5rem 1.25rem; }

/* ── Custom header bar ── */
.lf-header {
    display: flex;
    align-items: baseline;
    gap: 0.6rem;
    margin-bottom: 0.25rem;
}
.lf-wordmark {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.35rem;
    font-weight: 600;
    color: #e8eaed;
    letter-spacing: -0.02em;
}
.lf-tag {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    color: #5a9a6e;
    background: #1a2b20;
    border: 1px solid #2d4a38;
    padding: 1px 7px;
    border-radius: 3px;
    letter-spacing: 0.08em;
}
.lf-sub {
    font-size: 0.78rem;
    color: #555d6b;
    margin-bottom: 1.5rem;
    font-family: 'IBM Plex Mono', monospace;
}

/* ── Divider ── */
.lf-divider {
    border: none;
    border-top: 1px solid #1e2027;
    margin: 1rem 0;
}

/* ── Sidebar labels ── */
.lf-section-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.12em;
    color: #3d4450;
    text-transform: uppercase;
    margin: 1.25rem 0 0.5rem 0;
}

/* ── Status pill ── */
.lf-status {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    color: #5a9a6e;
    background: #1a2b20;
    border: 1px solid #2d4a38;
    padding: 3px 10px;
    border-radius: 3px;
}
.lf-status-dot {
    width: 6px; height: 6px;
    background: #5a9a6e;
    border-radius: 50%;
    display: inline-block;
    animation: pulse 2.5s ease-in-out infinite;
}
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.35} }

.lf-status-off { color: #4a4f5a; background: #16181e; border-color: #25282f; }
.lf-status-off .lf-status-dot { background: #4a4f5a; animation: none; }

/* ── Index stats card ── */
.lf-stats {
    background: #13151a;
    border: 1px solid #1e2027;
    border-radius: 6px;
    padding: 0.85rem 1rem;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    color: #555d6b;
}
.lf-stats-row {
    display: flex;
    justify-content: space-between;
    padding: 2px 0;
}
.lf-stats-val { color: #c9cdd4; }

/* ── Chat area ── */
.lf-chat-wrapper {
    background: #13151a;
    border: 1px solid #1e2027;
    border-radius: 8px;
    padding: 1rem 1.25rem;
    min-height: 480px;
    display: flex;
    flex-direction: column;
    gap: 1rem;
}

/* ── Messages ── */
.lf-msg {
    display: flex;
    gap: 0.75rem;
    align-items: flex-start;
}
.lf-avatar {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.65rem;
    color: #3d4450;
    background: #1a1c22;
    border: 1px solid #22252d;
    border-radius: 4px;
    padding: 3px 6px;
    white-space: nowrap;
    margin-top: 2px;
    flex-shrink: 0;
}
.lf-avatar-agent { color: #5a9a6e; border-color: #2d4a38; background: #161e1a; }
.lf-bubble {
    font-size: 0.88rem;
    line-height: 1.65;
    color: #c9cdd4;
    flex: 1;
}
.lf-bubble-user { color: #9da5b0; }

/* ── Source citations ── */
.lf-sources {
    margin-top: 0.6rem;
    display: flex;
    flex-wrap: wrap;
    gap: 0.4rem;
}
.lf-source-chip {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.65rem;
    color: #4a7c5e;
    background: #141e18;
    border: 1px solid #243329;
    padding: 2px 8px;
    border-radius: 3px;
}

/* ── Agent trace ── */
.lf-trace {
    background: #0e0f11;
    border: 1px solid #1e2027;
    border-radius: 5px;
    padding: 0.6rem 0.85rem;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    color: #3d4450;
    margin-top: 0.6rem;
}
.lf-trace-step { color: #4a7c5e; }
.lf-trace-tool { color: #7a6a9e; }

/* ── Tab style override ── */
.stTabs [data-baseweb="tab-list"] {
    background: transparent;
    gap: 0;
    border-bottom: 1px solid #1e2027;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    color: #3d4450;
    padding: 0.4rem 1.1rem;
    background: transparent;
    border: none;
    border-bottom: 2px solid transparent;
}
.stTabs [aria-selected="true"] {
    color: #c9cdd4 !important;
    border-bottom: 2px solid #5a9a6e !important;
    background: transparent !important;
}

/* ── Streamlit widget overrides ── */
.stSelectbox > div > div,
.stSlider > div,
.stTextInput > div > div > input {
    background-color: #16181e !important;
    border-color: #22252d !important;
    color: #c9cdd4 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.8rem !important;
}
.stButton > button {
    background: #1a2b20;
    border: 1px solid #2d4a38;
    color: #5a9a6e;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    border-radius: 4px;
    padding: 0.35rem 1rem;
    transition: all 0.15s;
}
.stButton > button:hover {
    background: #223529;
    border-color: #3d6b4a;
    color: #7dbf94;
}
.stFileUploader {
    background: #13151a !important;
    border: 1px dashed #22252d !important;
    border-radius: 6px !important;
}
[data-testid="stFileUploaderDropzone"] {
    background: transparent !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.78rem !important;
}

/* ── Chat input ── */
.stChatInput > div {
    background: #16181e !important;
    border: 1px solid #22252d !important;
    border-radius: 6px !important;
}
.stChatInput textarea {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.82rem !important;
    color: #c9cdd4 !important;
    background: transparent !important;
}
.stChatMessage { background: transparent !important; }

/* ── Metric overrides ── */
[data-testid="stMetric"] {
    background: #13151a;
    border: 1px solid #1e2027;
    border-radius: 6px;
    padding: 0.75rem 1rem !important;
}
[data-testid="stMetricLabel"] {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.65rem !important;
    color: #3d4450 !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}
[data-testid="stMetricValue"] {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 1.4rem !important;
    color: #c9cdd4 !important;
}
</style>
""", unsafe_allow_html=True)

# ─── Helper functions ───────────────────────────────────────────────────────────────
def save_files(files: list) -> None:
    save_dir = "./test_files"
    
    os.makedirs(save_dir, exist_ok=True)

    # Remove all previous files in the directory
    for existing_file in os.listdir(save_dir):
        file_path = os.path.join(save_dir, existing_file)

        if os.path.isfile(file_path):
            os.remove(file_path)

    # Save new files for indexing
    for file in files:
        file_path = os.path.join(save_dir, file.name)
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())

# ─── Session State ─────────────────────────────────────────────────────────────
if "index_ready" not in st.session_state:
    st.session_state.index_ready = False
if "doc_count" not in st.session_state:
    st.session_state.doc_count = 0
if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0
if "show_trace" not in st.session_state:
    st.session_state.show_trace = False
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

ingestion_pipeline = DocumentPipelineManager(provider=VectorStoreProvider(collection_name="test_collection"))

if not st.session_state.index_ready:
    try:
        existing_chunk_count = ingestion_pipeline.provider.collection.count()

        if existing_chunk_count > 0:
            st.session_state.index_ready = True
            st.session_state.chunk_count = existing_chunk_count
            st.session_state.doc_count = len(ingestion_pipeline.docstore.get_all_document_hashes())
    except Exception as e:
        pass  # collection might not exist yet, which is fine

sqlite_db_path = "./db_storage/checkpoints/checkpoint.db"

os.makedirs(os.path.dirname(sqlite_db_path), exist_ok=True)

sqlite_conn = SqliteDatabase(db_path=sqlite_db_path).get_connection()
checkpointer = SyncCheckpointer(connection=sqlite_conn).get_checkpointer()
store = SyncStoreService(connection=sqlite_conn).get_store()

compiled_graph = compile_graph(
    graph = create_graph(state_schema=State, context_schema=None),
    checkpointer=checkpointer,
    store=store
)

if "messages" not in st.session_state:
    config: RunnableConfig = {"configurable": {"thread_id": st.session_state.thread_id}}
    historical_state = compiled_graph.get_state(config)

    if historical_state.values and "messages" in historical_state.values:
        st.session_state.messages = historical_state.values["messages"]
        st.session_state.index_ready = True  # assume index is ready if we have historical messages
    else:
        st.session_state.messages = []

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
        <div class="lf-header">
            <span class="lf-wordmark">⚙ LocalForge</span>
            <span class="lf-tag">LOCAL</span>
        </div>
        <div class="lf-sub">agentic RAG · runs on your machine</div>
    """, unsafe_allow_html=True)

    # Model status
    ollama_ok = True  # replace with real health check
    status_cls = "lf-status" if ollama_ok else "lf-status lf-status-off"
    status_txt = "ollama · online" if ollama_ok else "ollama · offline"
    st.markdown(f'<div class="{status_cls}"><span class="lf-status-dot"></span>{status_txt}</div>', unsafe_allow_html=True)

    st.markdown('<hr class="lf-divider"/>', unsafe_allow_html=True)

    # ── Model settings ──
    st.markdown('<div class="lf-section-label">Model</div>', unsafe_allow_html=True)
    model = st.selectbox(
        "LLM",
        ["qwen2.5:7b", "llama3.2", "mistral:7b"],
        label_visibility="collapsed",
    )
    embed_model = st.selectbox(
        "Embeddings",
        ["all-MiniLM-L6-v2", "nomic-embed-text-v1.5"],
        label_visibility="collapsed",
    )

    st.markdown('<div class="lf-section-label">Retrieval</div>', unsafe_allow_html=True)
    top_k = st.slider("top_k", 1, 10, 4, label_visibility="visible")
    temperature = st.slider("temperature", 0.0, 1.0, 0.2, 0.05)

    st.markdown('<div class="lf-section-label">Agent</div>', unsafe_allow_html=True)
    agent_mode = st.toggle("LangGraph agent", value=True)
    show_trace = st.toggle("Show reasoning trace", value=st.session_state.show_trace)
    st.session_state.show_trace = show_trace

    st.markdown('<hr class="lf-divider"/>', unsafe_allow_html=True)

    # ── Document upload ──
    st.markdown('<div class="lf-section-label">Documents</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Upload files",
        type=["pdf", "txt", "docx"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded:
        save_files(uploaded)

        if st.button("⊕  Build index"):
            with st.spinner("Indexing…"):
                nodes = ingestion_pipeline.process_directory("./test_files", multi_files=True)

                st.session_state.index_ready = True
                st.session_state.doc_count = len(uploaded)
                st.session_state.chunk_count = ingestion_pipeline.provider.collection.count()
            st.success("Index ready")

    # ── Index stats ──
    if st.session_state.index_ready:
        st.markdown(f"""
        <div class="lf-stats">
            <div class="lf-stats-row"><span>documents</span><span class="lf-stats-val">{st.session_state.doc_count}</span></div>
            <div class="lf-stats-row"><span>chunks</span><span class="lf-stats-val">{st.session_state.chunk_count}</span></div>
            <div class="lf-stats-row"><span>store</span><span class="lf-stats-val">chromadb</span></div>
            <div class="lf-stats-row"><span>embed dim</span><span class="lf-stats-val">384</span></div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("✕  Clear index"):
            st.session_state.index_ready = False
            st.session_state.doc_count = 0
            st.session_state.chunk_count = 0
            st.session_state.messages = []
            st.rerun()


# ─── Main area ────────────────────────────────────────────────────────────────
tab_chat, tab_trace, tab_settings = st.tabs(["Chat", "Agent Trace", "Pipeline"])

# ── CHAT TAB ──
with tab_chat:
    if not st.session_state.index_ready:
        st.markdown("""
        <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;
                    min-height:420px;gap:0.75rem;text-align:center;">
            <div style="font-size:2rem;opacity:0.15">⊙</div>
            <div style="font-family:'IBM Plex Mono',monospace;font-size:0.8rem;color:#3d4450;">
                No index loaded
            </div>
            <div style="font-family:'IBM Plex Sans',sans-serif;font-size:0.8rem;color:#2e333c;max-width:320px;">
                Upload documents in the sidebar and click <em>Build index</em> to get started.
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Render message history
        running_trace = []
        running_sources = []
        step_counter = 1

        for msg in st.session_state.messages:
            if isinstance(msg, HumanMessage):
                with st.chat_message("user"):
                    st.markdown(msg.content)

            elif isinstance(msg, ToolMessage):
                running_trace.append({
                    "step": str(step_counter),
                    "tool": msg.name,
                    "note": f"Execution complete. Returned {len(msg.content)} chars of context."
                })
                step_counter += 1

                if msg.name == "retrieve_documents" and msg.artifact:
                    for artifact in msg.artifact:
                        if artifact not in running_sources:
                            running_sources.append(artifact)


            elif isinstance(msg, AIMessage):
                if msg.tool_calls:
                    for call in msg.tool_calls:
                        running_trace.append({
                            "step": str(step_counter),
                            "tool": "agent_node",
                            "note": f"Decided to call tool '{call['name']}' with args {call['args']}."
                        })
                        step_counter += 1

                if msg.content:
                    with st.chat_message("assistant"):
                        st.markdown(msg.content)

                        if running_sources:
                            chips = "".join(f'<span class="lf-source-chip">↗ {s}</span>' for s in running_sources)
                            st.markdown(f'<div class="lf-sources">{chips}</div>', unsafe_allow_html=True)
                    

                        if running_trace and st.session_state.show_trace:
                            with st.expander("reasoning trace", expanded=False):
                                for step in running_trace:
                                    st.markdown(
                                        f'<div class="lf-trace">'
                                        f'<span class="lf-trace-step">[{step["step"]}]</span> '
                                        f'<span class="lf-trace-tool">{step["tool"]}</span> — {step["note"]}'
                                        f'</div>',
                                        unsafe_allow_html=True,
                                    )

                        running_trace = []
                        running_sources = []
                        step_counter = 1

        # Chat input
        prompt = st.chat_input(
            "Ask a question about your documents…",
            disabled=not st.session_state.index_ready,
        )
        if prompt:
            user_msg = HumanMessage(content=prompt)
            st.session_state.messages.append(user_msg)
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner(""):
                    graph_config: RunnableConfig = {
                        "configurable": {
                            "thread_id": st.session_state.thread_id,
                            "top_k": top_k,
                            "temperature": temperature,
                            "model": model,
                            "embed_model": embed_model,
                        }
                    }

                    final_state = compiled_graph.invoke(
                        {"messages": st.session_state.messages},
                        config=graph_config
                    )

                st.session_state.messages = final_state["messages"]  # update session state with any new messages from the graph execution

                st.rerun()  # re-render to show new messages, sources, and trace

# ── AGENT TRACE TAB ──
with tab_trace:
    if not st.session_state.messages:
        st.markdown("""
        <div style="font-family:'IBM Plex Mono',monospace;font-size:0.75rem;color:#2e333c;
                    padding:2rem 0;text-align:center;">
            No agent runs yet. Ask a question in the Chat tab.
        </div>
        """, unsafe_allow_html=True)
    else:
        run_id = 1
        current_trace = []
        user_query = ""
        step_counter = 1

        # Scan the single source of truth to build the global trace view
        for msg in st.session_state.messages:
            
            # When a user speaks, it marks the beginning of a new "Run"
            if isinstance(msg, HumanMessage):
                # If we already have a trace from a previous loop, render it before resetting
                if current_trace:
                    st.markdown(
                        f'<div style="font-family:\'IBM Plex Mono\',monospace;font-size:0.7rem;'
                        f'color:#3d4450;margin-bottom:0.3rem;">run #{run_id} · query: "{user_query}"</div>',
                        unsafe_allow_html=True,
                    )
                    for step in current_trace:
                        st.markdown(
                            f'<div class="lf-trace" style="margin-bottom:0.35rem;">'
                            f'<span class="lf-trace-step">[{step["step"]}]</span>&nbsp;&nbsp;'
                            f'<span class="lf-trace-tool">{step["tool"]}</span>'
                            f'<span style="color:#2e333c"> — {step["note"]}</span>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                    st.markdown('<hr class="lf-divider"/>', unsafe_allow_html=True)
                    run_id += 1
                
                # Reset buffers for the new run
                current_trace = []
                user_query = msg.content
                step_counter = 1

            # Capture AI Decisions
            elif isinstance(msg, AIMessage) and msg.tool_calls:
                for call in msg.tool_calls:
                    current_trace.append({
                        "step": str(step_counter),
                        "tool": "agent_node",
                        "note": f"Called '{call['name']}' with args {call['args']}"
                    })
                    step_counter += 1
            
            # Capture Tool Executions
            elif isinstance(msg, ToolMessage):
                current_trace.append({
                    "step": str(step_counter),
                    "tool": msg.name,
                    "note": f"Execution complete. Returned {len(msg.artifact)} sources."
                })
                step_counter += 1

        # Render the very last run (since the loop ends before triggering a reset)
        if current_trace:
            st.markdown(
                f'<div style="font-family:\'IBM Plex Mono\',monospace;font-size:0.7rem;'
                f'color:#3d4450;margin-bottom:0.3rem;">run #{run_id} · query: "{user_query}"</div>',
                unsafe_allow_html=True,
            )
            for step in current_trace:
                st.markdown(
                    f'<div class="lf-trace" style="margin-bottom:0.35rem;">'
                    f'<span class="lf-trace-step">[{step["step"]}]</span>&nbsp;&nbsp;'
                    f'<span class="lf-trace-tool">{step["tool"]}</span>'
                    f'<span style="color:#2e333c"> — {step["note"]}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            st.markdown('<hr class="lf-divider"/>', unsafe_allow_html=True)

# ── PIPELINE TAB ──
with tab_settings:
    st.markdown("""
    <div style="font-family:'IBM Plex Mono',monospace;font-size:0.65rem;
                letter-spacing:0.1em;color:#3d4450;text-transform:uppercase;
                margin-bottom:1rem;">Architecture</div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Ingestion", "LlamaIndex")
        st.metric("Vector Store", "ChromaDB")
    with col2:
        st.metric("Embeddings", embed_model.split("/")[-1])
        st.metric("LLM", model)
    with col3:
        st.metric("Agent", "LangGraph" if agent_mode else "Direct RAG")
        st.metric("Interface", "Streamlit")

    st.markdown('<hr class="lf-divider"/>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-family:'IBM Plex Mono',monospace;font-size:0.65rem;
                letter-spacing:0.1em;color:#3d4450;text-transform:uppercase;
                margin-bottom:0.75rem;">Integration points</div>
    <div style="font-family:'IBM Plex Mono',monospace;font-size:0.75rem;color:#555d6b;
                line-height:2;background:#13151a;border:1px solid #1e2027;
                border-radius:6px;padding:1rem 1.25rem;">
        <span style="color:#4a7c5e">ingestion/</span>  →  SimpleDirectoryReader · SentenceSplitter · VectorStoreIndex<br>
        <span style="color:#4a7c5e">rag/</span>         →  RetrieverQueryEngine · similarity_top_k · reranker<br>
        <span style="color:#4a7c5e">tools/</span>       →  document_knowledge_tool · summarizer · extractor<br>
        <span style="color:#4a7c5e">graph/</span>       →  StateGraph · agent node · ToolNode · router<br>
        <span style="color:#4a7c5e">ui/</span>          →  this file
    </div>
    """, unsafe_allow_html=True)