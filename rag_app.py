import hashlib
import os
import tempfile
from pathlib import Path

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
load_dotenv()


st.set_page_config(page_title="Multi-PDF RAG Assistant", page_icon="📚", layout="wide")

st.title("📚 Multi-PDF RAG Assistant")
st.write(
    "Upload multiple PDFs, index them into one FAISS vector store, and ask "
    "questions across all documents. Answers include file and page references."
)

# Keep the vector store and index details across Streamlit reruns.
if "store" not in st.session_state:
    st.session_state.store = None
if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = []
if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0
if "index_signature" not in st.session_state:
    st.session_state.index_signature = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error(
        "OPENAI_API_KEY was not found. Set it as an environment variable and "
        "restart Streamlit."
    )
    st.code(
        '# macOS / Linux\nexport OPENAI_API_KEY="your-api-key"\n\n'
        '# Windows PowerShell\n$env:OPENAI_API_KEY="your-api-key"',
        language="bash",
    )
    st.stop()

uploaded_files = st.file_uploader(
    "Upload your PDF documents",
    type=["pdf"],
    accept_multiple_files=True,
    help="Select several PDF files in one upload.",
)

with st.sidebar:
    st.header("Index settings")
    chunk_size = st.number_input(
        "Chunk size", min_value=300, max_value=2500, value=1000, step=100
    )
    chunk_overlap = st.number_input(
        "Chunk overlap", min_value=0, max_value=500, value=150, step=25
    )
    top_k = st.slider("Number of chunks to retrieve", min_value=2, max_value=12, value=6)
    st.caption(
        "If answers cite many unrelated pages, try a larger chunk size or a "
        "smaller retrieval count. If answers miss details, try smaller chunks "
        "or a larger retrieval count."
    )

if chunk_overlap >= chunk_size:
    st.error("Chunk overlap must be smaller than chunk size.")
    st.stop()

def make_signature(files):
    """Create a stable signature so unchanged uploads are not re-indexed."""
    digest = hashlib.sha256()
    for uploaded in sorted(files, key=lambda item: item.name.lower()):
        content = uploaded.getvalue()
        digest.update(uploaded.name.encode("utf-8"))
        digest.update(content)
    digest.update(str(chunk_size).encode())
    digest.update(str(chunk_overlap).encode())
    return digest.hexdigest()


def build_vector_store(files):
    all_chunks = []
    indexed_names = []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=int(chunk_size),
        chunk_overlap=int(chunk_overlap),
        separators=["\n\n", "\n", ". ", " ", ""],
        add_start_index=True,
    )

    progress = st.progress(0, text="Preparing documents...")
    total_files = len(files)

    for file_number, uploaded in enumerate(files, start=1):
        # PyPDFLoader reads from a local path, so write each upload temporarily.
        safe_name = Path(uploaded.name).name
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_pdf:
            temp_pdf.write(uploaded.getvalue())
            temp_path = temp_pdf.name

        try:
            pages = PyPDFLoader(temp_path).load()
            for page_doc in pages:
                # Preserve the original upload name rather than the temporary path.
                page_doc.metadata["source"] = safe_name
                # PyPDFLoader uses zero-based page numbers. Store a reader-friendly
                # one-based page number for display.
                page_doc.metadata["page"] = int(page_doc.metadata.get("page", 0)) + 1

            file_chunks = splitter.split_documents(pages)
            all_chunks.extend(file_chunks)
            indexed_names.append(safe_name)
        finally:
            try:
                os.remove(temp_path)
            except OSError:
                pass

        progress.progress(
            file_number / total_files,
            text=f"Processed {file_number} of {total_files}: {safe_name}",
        )

    if not all_chunks:
        raise ValueError("No readable text was found in the uploaded PDFs.")

    progress.progress(1.0, text="Creating embeddings and FAISS index...")
    embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)
    vector_store = FAISS.from_documents(all_chunks, embeddings)
    progress.empty()
    return vector_store, all_chunks, indexed_names


if uploaded_files:
    st.caption("Selected: " + ", ".join(Path(f.name).name for f in uploaded_files))

    current_signature = make_signature(uploaded_files)
    needs_rebuild = (
        st.session_state.store is None
        or st.session_state.index_signature != current_signature
    )

    if st.button(
        "Build / update knowledge base",
        type="primary",
        disabled=not needs_rebuild,
    ):
        try:
            with st.spinner("Reading PDFs and building the shared vector database..."):
                store, chunks, names = build_vector_store(uploaded_files)
            st.session_state.store = store
            st.session_state.chunk_count = len(chunks)
            st.session_state.indexed_files = names
            st.session_state.index_signature = current_signature
            st.session_state.chat_history = []
            st.success(f"Indexed {len(chunks)} chunks from {len(names)} PDF(s).")
        except Exception as exc:
            st.error(f"Could not build the index: {exc}")
            st.exception(exc)

if st.session_state.store is not None:
    st.divider()
    st.subheader("Knowledge base")
    col1, col2 = st.columns(2)
    col1.metric("PDF files indexed", len(st.session_state.indexed_files))
    col2.metric("Chunks indexed", st.session_state.chunk_count)
    st.write("**Files:** " + ", ".join(st.session_state.indexed_files))

    st.subheader("Ask a question")
    question = st.chat_input("Ask a question across your uploaded PDFs...")

    if question:
        st.session_state.chat_history.append({"role": "user", "content": question})

        try:
            retrieved_docs = st.session_state.store.as_retriever(
                search_type="mmr",
                search_kwargs={"k": int(top_k), "fetch_k": max(20, int(top_k) * 3)},
            ).invoke(question)

            if not retrieved_docs:
                answer = (
                    "I couldn't find relevant information in the uploaded documents. "
                    "Try rephrasing the question."
                )
                source_docs = []
            else:
                context_parts = []
                for index, doc in enumerate(retrieved_docs, start=1):
                    filename = Path(str(doc.metadata.get("source", "Unknown file"))).name
                    page = doc.metadata.get("page", "Unknown")
                    context_parts.append(
                        f"[Source {index}: {filename}, page {page}]\n{doc.page_content}"
                    )

                context = "\n\n".join(context_parts)
                prompt = f"""You are a careful assistant answering questions from a
collection of PDF documents.

Answer the user's question using ONLY the supplied document excerpts.
Cite factual claims with the source labels exactly as provided, for example [Source 1].
If the documents do not contain the answer, say that the information was not found.
Do not invent facts, filenames, page numbers, or citations.

IMPORTANT: Compare the retrieved excerpts for disagreements. If two or more
documents give different answers, dates, figures, rules, or definitions, do NOT
silently choose one. Clearly state that the documents conflict, summarize each
position separately, and cite each relevant source. If the excerpts are compatible,
combine them into one concise answer.

DOCUMENT EXCERPTS:
{context}

USER QUESTION:
{question}

Provide a clear, concise answer with source citations."""
                llm = ChatOpenAI(
                    model="gpt-4o-mini",
                    temperature=0,
                    api_key=api_key,
                )
                response = llm.invoke(prompt)
                answer = response.content
                source_docs = retrieved_docs

            st.session_state.chat_history.append(
                {"role": "assistant", "content": answer, "sources": source_docs}
            )
        except Exception as exc:
            st.error(f"Error while answering: {exc}")
            st.exception(exc)

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and message.get("sources"):
                st.markdown("**Sources used**")
                seen = set()
                for doc in message["sources"]:
                    filename = Path(str(doc.metadata.get("source", "Unknown file"))).name
                    page = doc.metadata.get("page", "Unknown")
                    key = (filename, page)
                    if key in seen:
                        continue
                    seen.add(key)
                    with st.expander(f"{filename} — page {page}"):
                        st.write(doc.page_content)
else:
    st.info("Upload one or more PDFs, then click **Build / update knowledge base**.")

st.divider()
st.caption(
    "Note: Page references are displayed as human-readable, one-based PDF page "
    "numbers. Scanned PDFs may need OCR before their text can be retrieved."
)
