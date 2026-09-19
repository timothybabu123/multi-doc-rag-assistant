import os
import tempfile

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def process_uploaded_files(uploaded_files):

    all_chunks = []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )

    for uploaded_file in uploaded_files:

        # Save uploaded PDF temporarily
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        ) as temp_file:

            temp_file.write(uploaded_file.getvalue())
            temp_path = temp_file.name

        try:

            # Load PDF
            loader = PyPDFLoader(temp_path)
            documents = loader.load()

            # Add metadata
            for document in documents:

                document.metadata["source_doc"] = uploaded_file.name
                document.metadata["file_type"] = "pdf"

                if "page" not in document.metadata:
                    document.metadata["page"] = 0

            # Split into chunks
            chunks = splitter.split_documents(documents)

            # Remove empty chunks
            for chunk in chunks:

                if (
                    isinstance(chunk.page_content, str)
                    and chunk.page_content.strip()
                ):
                    chunk.page_content = chunk.page_content.strip()
                    all_chunks.append(chunk)

        finally:

            if os.path.exists(temp_path):
                os.remove(temp_path)

    return all_chunks