import os
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from config import PERSIST_PATH, EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP
from utils.pdf_loader import load_pdf_text

# ---------------- PDF → Documents ----------------
def pdfs_to_documents(pdf_paths):
    docs = []
    for p in pdf_paths:
        text = load_pdf_text(p)
        meta = {"source": os.path.basename(p)}
        docs.append(Document(page_content=text, metadata=meta))
    return docs

# ---------------- Chunk Documents ----------------
def chunk_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    new_docs = []
    for d in documents:
        splits = splitter.split_text(d.page_content)
        for i, s in enumerate(splits):
            new_docs.append(Document(page_content=s, metadata={**d.metadata, "chunk": i}))
    return new_docs

# ---------------- Build / Save FAISS ----------------
def build_faiss_index(documents, persist_path=PERSIST_PATH, model_name=EMBEDDING_MODEL):
    Path(persist_path).parent.mkdir(parents=True, exist_ok=True)
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    vectorstore = FAISS.from_documents(documents, embeddings)
    vectorstore.save_local(persist_path)
    print(f"Saved FAISS index to {persist_path}")
    return vectorstore

# ---------------- CLI ----------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdfs", nargs="+", help="List of PDF paths to ingest", required=True)
    parser.add_argument("--persist", default=PERSIST_PATH)
    args = parser.parse_args()
    docs = pdfs_to_documents(args.pdfs)
    print(f"Loaded {len(docs)} PDF documents")
    chunks = chunk_documents(docs)
    print(f"Created {len(chunks)} chunks")
    build_faiss_index(chunks, persist_path=args.persist)
