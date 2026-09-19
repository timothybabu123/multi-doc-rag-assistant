import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def process_uploaded_files(uploaded_files):
    all_chunks = []
    
    # Splitter configured for standard document structures
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )

    for uploaded_file in uploaded_files:
        # Save uploaded buffer temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name

        try:
            loader = PyPDFLoader(tmp_path)
            docs = loader.load()

            # Enrich metadata for each page
            for doc in docs:
                doc.metadata["source_doc"] = uploaded_file.name
                doc.metadata["file_type"] = "pdf"
                # Keep original page number if present, defaulting to unknown
                if "page" not in doc.metadata:
                    doc.metadata["page"] = 0

            # Split enriched documents into chunks
            chunks = splitter.split_documents(docs)
            all_chunks.extend(chunks)

        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    return all_chunks