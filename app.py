import os
import re
import hashlib
from pathlib import Path

import streamlit as st

# -------------------------------
# Modern LangChain Ecosystem Imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# -------------------------------
# Configuration
PDF_DIR = Path("pdfs")
INDEX_DIR = Path("indexes")

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "qwen2.5-coder:7b"

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 150
TOP_K = 6
MAX_DISTANCE_THRESHOLD = 1.15

PDF_DIR.mkdir(parents=True, exist_ok=True)
INDEX_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(page_title="DocQuery", page_icon="📄")
st.title("📄 DocQuery")


# -------------------------------
# Model Loaders
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )


@st.cache_resource
def load_llm():
    return ChatOllama(
        model=LLM_MODEL,
        temperature=0.2
    )


# -------------------------------
# Query Expansion
query_prompt = ChatPromptTemplate.from_messages([
    ("system", """
Generate exactly 3 alternative search queries for finding
evidence in PDF documents that answers the user's question.

Understand informal language and use related concepts,
synonyms, and formal terminology.

Return only 3 search queries, one per line.
Do not answer the question or explain the queries.
"""),
    ("human", "{question}")
])


def expand_queries(question: str) -> list[str]:
    try:
        chain = query_prompt | load_llm() | StrOutputParser()
        output = chain.invoke({"question": question})

        alternatives = []
        for line in output.splitlines():
            clean = re.sub(r"^\s*(?:\d+[\.\)]\s*|[-*•]\s*)", "", line).strip().strip("\"'")
            if clean and not clean.lower().startswith(("alternative queries:", "search queries:", "queries:")):
                alternatives.append(clean)

        queries = []
        for query in [question.strip()] + alternatives:
            if query.lower() not in [q.lower() for q in queries]:
                queries.append(query)

        return queries[:4]
    except Exception:
        return [question.strip()]


# -------------------------------
# PDF Vectorstore & Caching
def get_vectorstore(pdf_path: Path, filename: str, embeddings: HuggingFaceEmbeddings):
    pdf_bytes = pdf_path.read_bytes()
    settings = f"{EMBEDDING_MODEL}|{CHUNK_SIZE}|{CHUNK_OVERLAP}"
    key = hashlib.sha256(filename.encode("utf-8") + pdf_bytes + settings.encode("utf-8")).hexdigest()
    index_path = INDEX_DIR / key

    if (index_path / "index.faiss").exists() and (index_path / "index.pkl").exists():
        try:
            return FAISS.load_local(str(index_path), embeddings, allow_dangerous_deserialization=True), False
        except Exception:
            pass

    documents = PyPDFLoader(str(pdf_path)).load()
    documents = [doc for doc in documents if doc.page_content.strip()]
    if not documents:
        return None, False

    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(documents)
    if not chunks:
        return None, False

    for chunk in chunks:
        chunk.metadata["source"] = filename
        chunk.metadata["page"] = int(chunk.metadata.get("page", 0)) + 1

    store = FAISS.from_documents(chunks, embeddings)
    store.save_local(str(index_path))
    return store, True


# -------------------------------
# Strict RAG Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", """
You are DocQuery, an assistant that answers questions using ONLY the provided CONTEXT.

Rules:
1. Ground every claim strictly in the facts provided in the CONTEXT.
2. If the context does not contain enough information, state:
   "The provided documents do not contain enough information to answer this question."
3. Do not speculate or use outside knowledge.
4. Reference the document name and page number for facts (e.g., [filename, page N]).

CONTEXT:
{context}
"""),
    ("human", "{question}")
])


# -------------------------------
# Sidebar
with st.sidebar:
    st.header("Upload & Select")
    uploaded_files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
    if uploaded_files:
        for file in uploaded_files:
            filename = os.path.basename(file.name.replace("\\", "/"))
            pdf_path = PDF_DIR / filename
            if not pdf_path.exists() or pdf_path.read_bytes() != file.getvalue():
                pdf_path.write_bytes(file.getvalue())
                st.success(f"Saved {filename}")

    available_pdfs = sorted(
        f for f in os.listdir(PDF_DIR) if f.lower().endswith(".pdf") and (PDF_DIR / f).is_file()
    )
    selected_pdfs = st.multiselect("Select PDFs to search", available_pdfs)


# -------------------------------
# Main QA Area
stores = []
if selected_pdfs:
    try:
        embeddings = load_embeddings()
        for filename in selected_pdfs:
            store, created = get_vectorstore(PDF_DIR / filename, filename, embeddings)
            if store:
                stores.append(store)
    except Exception as e:
        st.error(f"Error loading indexes: {e}")

if stores:
    question = st.text_input("Ask a question about your documents:")
    if st.button("Get Answer", type="primary"):
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            try:
                with st.spinner("Searching..."):
                    queries = expand_queries(question.strip())
                    matches = {}

                    for store in stores:
                        k = min(TOP_K, store.index.ntotal)
                        if k == 0:
                            continue
                        for q in queries:
                            for doc, score in store.similarity_search_with_score(q, k=k):
                                if score > MAX_DISTANCE_THRESHOLD:
                                    continue
                                key = (doc.metadata.get("source"), doc.metadata.get("page"), doc.page_content)
                                if key not in matches or score < matches[key][1]:
                                    matches[key] = (doc, score)

                    results = sorted(matches.values(), key=lambda x: x[1])[:TOP_K]

                if not results:
                    st.warning("No relevant information found in the selected documents.")
                else:
                    context = "\n\n---\n\n".join(
                        f"[{doc.metadata.get('source')}, page {doc.metadata.get('page')}]\n"
                        f"{' '.join(doc.page_content.split())}"
                        for doc, score in results
                    )

                    st.write("### Answer")
                    chain = prompt | load_llm() | StrOutputParser()
                    st.write_stream(chain.stream({"context": context, "question": question.strip()}))

                    with st.expander(f"Sources ({len(results)})"):
                        for doc, score in results:
                            st.write(f"**{doc.metadata.get('source')} — Page {doc.metadata.get('page')}**")
                            st.caption(" ".join(doc.page_content.split())[:200] + "...")

            except Exception as e:
                st.error(f"Error: {e}")
else:
    st.info("Upload and select at least one PDF in the sidebar to begin.")