import os
import streamlit as st
from dotenv import load_dotenv

from document_loader import process_uploaded_files
from rag_engine import build_vectorstore, answer_query

load_dotenv()

st.set_page_config(
    page_title="Multi-Doc Comparison Assistant",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Multi-Document RAG & Comparison Assistant")
st.caption("Upload 2–3 PDFs, filter by specific sources, or run comparative queries across all of them.")

# Sidebar - Multi-File Upload & Document Filter
with st.sidebar:
    st.header("📂 Document Management")
    uploaded_files = st.file_uploader(
        "Upload PDFs",
        type=["pdf"],
        accept_multiple_files=True
    )

    if st.button("Process Documents", type="primary"):
        if uploaded_files:
            with st.spinner("Processing documents and attaching metadata..."):
                chunks = process_uploaded_files(uploaded_files)
                vectorstore = build_vectorstore(chunks)

                st.session_state.vectorstore = vectorstore
                st.session_state.file_names = [f.name for f in uploaded_files]
                st.session_state.chunk_count = len(chunks)

            st.success(f"Successfully processed {len(uploaded_files)} document(s)!")
        else:
            st.warning("Please upload at least one PDF file.")

    st.divider()

    # Metadata Filter Selection
    selected_doc = "All Documents"
    if "file_names" in st.session_state:
        st.subheader("🎯 Retrieval Scope Filter")
        options = ["All Documents"] + st.session_state.file_names
        selected_doc = st.selectbox("Focus retrieval on:", options)
        st.info(f"Total Chunks Ingested: {st.session_state.chunk_count}")

# Main Chat Interface
if "vectorstore" in st.session_state:
    st.subheader(f"💬 Querying Scope: `{selected_doc}`")

    question = st.chat_input("Ask a question or request a comparison between documents...")

    if question:
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Searching vector database with metadata filters..."):
                answer, retrieved_docs = answer_query(
                    st.session_state.vectorstore,
                    question,
                    selected_doc=selected_doc
                )

            st.write(answer)

            # Retrieved Passages Expander with Source Attribution
            with st.expander("📄 View Retrieved Context & Metadata Sources"):
                for idx, doc in enumerate(retrieved_docs):
                    source = doc.metadata.get("source_doc", "Unknown")
                    page = doc.metadata.get("page", 0) + 1
                    st.markdown(f"**Passage {idx + 1}** | `Source: {source}` | `Page: {page}`")
                    st.text(doc.page_content)
                    st.divider()
else:
    st.info("👈 Upload 2 or more PDFs in the sidebar and click 'Process Documents' to get started.")