from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
import streamlit as st

@st.cache_resource
def get_embedding_model():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def get_embedding_model():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def build_vectorstore(chunks):
    """Creates an in-memory Chroma vector store from document chunks."""
    embeddings = get_embedding_model()
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings
    )
    return vectorstore

def answer_query(vectorstore, question, selected_doc="All Documents"):
    """
    Retrieves context using ChromaDB metadata filters and generates an answer.
    """
    # Configure metadata filtering
    search_kwargs = {"k": 4, "fetch_k": 10}
    
    if selected_doc != "All Documents":
        # ChromaDB metadata filter syntax
        search_kwargs["filter"] = {"source_doc": selected_doc}

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs=search_kwargs
    )

    docs = retriever.invoke(question)

    # Format context with source metadata visible to the LLM
    formatted_context_list = []
    for doc in docs:
        source_name = doc.metadata.get("source_doc", "Unknown Source")
        page_num = doc.metadata.get("page", 0) + 1
        content = doc.page_content
        formatted_context_list.append(f"[Source: {source_name} | Page {page_num}]\n{content}")

    context = "\n\n---\n\n".join(formatted_context_list)

    # System prompt for comparison and multi-doc analysis
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert academic and analytical assistant. 
Your job is to answer user queries and compare information across provided documents.

Rules:
1. Rely strictly on the provided Context.
2. If comparing documents, highlight similarities and key differences clearly.
3. Cite the exact document source and page number when referencing facts.
4. If the answer cannot be found in the context, say "I could not find relevant information in the uploaded documents."

Context:
{context}"""),
        ("human", "{question}")
    ])

    llm = ChatGoogleGenerativeAI(
        model="gemini-3.6-flash",
        temperature=0.1
    )

    chain = prompt | llm
    response = chain.invoke({"context": context, "question": question})

    return response.content, docs