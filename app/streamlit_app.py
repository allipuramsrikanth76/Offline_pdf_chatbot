import streamlit as st
from pathlib import Path
import os
import sys

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.prepare_embeddings import pdfs_to_documents, chunk_documents, build_faiss_index
from utils.retriever import load_retriever, get_qa_chain
from config import PERSIST_PATH

st.set_page_config(page_title="Offline PDF Q&A (RAG)")
st.title("Offline PDF Q&A Chatbot")

# ---------------- Sidebar: Index PDFs ----------------
st.sidebar.header("Upload PDFs Files")
uploaded_files = st.sidebar.file_uploader(
    "Upload PDFs to index", type=["pdf"], accept_multiple_files=True
)
reindex = st.sidebar.button("Upload")

if reindex and uploaded_files:
    tmp_dir = Path("./tmp_pdfs")
    tmp_dir.mkdir(exist_ok=True)
    pdf_paths = []

    for f in uploaded_files:
        out = tmp_dir / f.name
        with open(out, "wb") as wf:
            wf.write(f.getbuffer())
        pdf_paths.append(str(out))

    docs = pdfs_to_documents(pdf_paths)
    chunks = chunk_documents(docs)
    build_faiss_index(chunks, persist_path=PERSIST_PATH)
    st.sidebar.success("Indexing complete")

# ---------------- Sidebar: Query ----------------
st.sidebar.header("Query")
question = st.sidebar.text_input("Ask a question about the uploaded PDFs:")
ask = st.sidebar.button("Ask Questions")

if ask and question.strip():
    try:
        retriever = load_retriever()
        qa = get_qa_chain(retriever)

        with st.spinner("Thinking..."):
            response = qa(question)
            answer = response["answer"]
            sources = response.get("context", [])

        st.markdown("**Answer**")
        st.write(answer)

        st.markdown("**Source Documents**")
        for doc in sources:
            st.write(doc.metadata.get("source"), "-", doc.metadata.get("chunk"))

    except Exception as e:
        st.error(f"Error running QA: {e}")

st.write("---")
st.write("**Note**: Ensure you have run the indexing step at least once before asking questions.")
