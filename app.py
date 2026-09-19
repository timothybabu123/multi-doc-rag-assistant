import os
import streamlit as st
from dotenv import load_dotenv

from document_loader import process_uploaded_files
from rag_engine import build_vectorstore, answer_query

load_dotenv()

# Page Setup
st.set_page_config(
    page_title="DocuMind AI | Multi-Doc RAG",
    page_icon="⚡",
    layout="wide"
)

# Custom CSS for Modern UI
st.markdown("""
<style>
    /* Dark Theme Accent Adjustments */
    .stApp {
        background-color: #0E1117;
    }
    
    /* Header Gradient */
    .gradient-header {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #3B82F6, #8B5CF6, #EC4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }

    /* Metric Cards in Sidebar */
    div[data-testid="stMetricValue"] {
        font-size: 1.5rem;
        color: #3B82F6;
    }

    /* Citation Card Container */
    .citation-card {
        background-color: #1E293B;
        border-left: 4px solid #3B82F6;
        padding: 10px 14px;
        border-radius: 6px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# App Header
st.markdown('<p class="gradient-header">⚡ DocuMind AI — Multi-Doc Intelligence</p>', unsafe_allow_html=True)
st.caption("Upload multiple PDFs, cross-examine sources, and extract comparative insights instantly.")

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar - Document Ingestion & Metrics
with st.sidebar:
    st.header("📂 Document Workspace")
    uploaded_files = st.file_uploader(
        "Upload PDF Files",
        type=["pdf"],
        accept_multiple_files=True
    )

    if st.button("🚀 Process Documents", type="primary", use_container_width=True):
        if uploaded_files:
            with st.status("Ingesting Documents...", expanded=True) as status:
                st.write("📄 Reading byte streams...")
                chunks = process_uploaded_files(uploaded_files)
                st.write("🧠 Generating embeddings & indexing vectors...")
                st.session_state.vectorstore = build_vectorstore(chunks)
                st.session_state.file_names = [f.name for f in uploaded_files]
                st.session_state.chunk_count = len(chunks)
                status.update(label="Processing Complete!", state="complete", expanded=False)
            st.success(f"Ready with {len(uploaded_files)} document(s)!")
        else:
            st.warning("Please attach at least one PDF.")

    st.divider()

    # Workspace Stats & Retrieval Scope
    selected_doc = "All Documents"
    if "file_names" in st.session_state:
        st.subheader("📊 Document Insights")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Files", len(st.session_state.file_names))
        with col2:
            st.metric("Chunks", st.session_state.chunk_count)

        st.subheader("🎯 Scope Filter")
        options = ["All Documents"] + st.session_state.file_names
        selected_doc = st.selectbox("Direct retrieval to:", options)

# Main Interface
if "vectorstore" in st.session_state:
    st.markdown(f"**Active Context:** `{selected_doc}`")

    # Quick Suggestion Buttons
    st.write("💡 **Quick Actions:**")
    qcol1, qcol2, qcol3 = st.columns(3)
    quick_query = None

    if qcol1.button("📋 Executive Summary", use_container_width=True):
        quick_query = "Provide a high-level executive summary across all uploaded documents."
    if qcol2.button("⚔️ Key Differences", use_container_width=True):
        quick_query = "Compare and contrast the main differences or conflicting details in these documents."
    if qcol3.button("🔍 Action Items", use_container_width=True):
        quick_query = "List out all key takeaways, recommendations, or actionable steps mentioned."

    st.divider()

    # Render Chat History
    for message in st.session_state.messages:
        avatar = "🤖" if message["role"] == "assistant" else "👤"
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])
            if "docs" in message and message["docs"]:
                with st.expander("📄 Cited Passages & Metadata"):
                    for idx, doc in enumerate(message["docs"]):
                        source = doc.metadata.get("source_doc", "Unknown")
                        page = doc.metadata.get("page", 0) + 1
                        st.markdown(f"**Source {idx + 1}:** `{source}` | **Page {page}**")
                        st.info(doc.page_content)

    # Process New Input (Chat input or Quick Buttons)
    prompt = st.chat_input("Ask a question across your documents...") or quick_query

    if prompt:
        # Display User Input
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Generate Assistant Response
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Retrieving vector chunks and synthesizing..."):
                answer, retrieved_docs = answer_query(
                    st.session_state.vectorstore,
                    prompt,
                    selected_doc=selected_doc
                )

            st.markdown(answer)

            # Render Source Passages
            with st.expander("📄 Cited Passages & Metadata"):
                for idx, doc in enumerate(retrieved_docs):
                    source = doc.metadata.get("source_doc", "Unknown")
                    page = doc.metadata.get("page", 0) + 1
                    st.markdown(f"**Source {idx + 1}:** `{source}` | **Page {page}**")
                    st.info(doc.page_content)

        # Save to Chat History
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "docs": retrieved_docs
        })

else:
    st.info("👈 Upload 2 or more PDFs in the sidebar and click **'Process Documents'** to unlock the assistant.")