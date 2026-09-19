from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate


# --------------------------------------------------
# EMBEDDING MODEL
# --------------------------------------------------

def get_embedding_model():

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


# --------------------------------------------------
# CREATE VECTOR DATABASE
# --------------------------------------------------

def build_vectorstore(chunks):

    embeddings = get_embedding_model()

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    return vectorstore


# --------------------------------------------------
# RETRIEVE + GENERATE ANSWER
# --------------------------------------------------

def answer_query(
    vectorstore,
    question,
    selected_doc="All Documents"
):

    # Retrieval configuration
    search_kwargs = {
        "k": 5,
        "fetch_k": 15
    }

    # Metadata filtering
    if selected_doc != "All Documents":

        search_kwargs["filter"] = {
            "source_doc": selected_doc
        }

    # Create retriever
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs=search_kwargs
    )

    # Retrieve relevant chunks
    retrieved_docs = retriever.invoke(question)

    # --------------------------------------------------
    # BUILD CONTEXT
    # --------------------------------------------------

    context_parts = []

    for doc in retrieved_docs:

        source = doc.metadata.get(
            "source_doc",
            "Unknown"
        )

        page = doc.metadata.get(
            "page",
            0
        ) + 1

        context_parts.append(
            f"""
SOURCE: {source}
PAGE: {page}

{doc.page_content}
"""
        )

    context = "\n\n--------------------\n\n".join(
        context_parts
    )

    # --------------------------------------------------
    # PROMPT
    # --------------------------------------------------

    prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are an intelligent document assistant.

Your job is to help the user understand, analyze, summarize, and compare
the uploaded documents.

Use the retrieved document context as your primary source of information.
Answer naturally and directly.

When the context contains the answer:
- Give a clear and useful answer.
- Explain concepts when helpful.
- Summarize or compare information when requested.
- When comparing documents, clearly distinguish which information comes
  from which document.
- Mention the relevant document and page when it is useful.

If the retrieved context does not contain enough information to answer
the question confidently, you may use your general knowledge to provide
a helpful answer, but make it clear when you are going beyond the
uploaded documents.

Never invent document names, page numbers, quotations, or claims that
are not supported by the retrieved context.

Context:
{context}
"""
    ),
    (
        "human",
        "{question}"
    )
])

    # --------------------------------------------------
    # GEMINI
    # --------------------------------------------------

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.1
    )

    # --------------------------------------------------
    # GENERATE
    # --------------------------------------------------

    chain = prompt | llm

    response = chain.invoke({
        "context": context,
        "question": question
    })

    return response.content, retrieved_docs