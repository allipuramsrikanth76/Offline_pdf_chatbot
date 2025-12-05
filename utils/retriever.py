from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from config import PERSIST_PATH, TOP_K, OLLAMA_MODEL, EMBEDDING_MODEL

# ---------------- Wrapper Class ----------------
class RetrieverWrapper:
    def __init__(self, vectorstore, k=TOP_K):
        self.vectorstore = vectorstore
        self.k = k

    # Provide the missing method
    def get_relevant_documents(self, query):
        return self.vectorstore.similarity_search(query, k=self.k)


# ---------------- Load FAISS Retriever ----------------
def load_retriever():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = FAISS.load_local(
        PERSIST_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

    # Wrap it to ensure get_relevant_documents exists
    retriever = RetrieverWrapper(vectorstore)
    return retriever


# ---------------- Build QA Chain ----------------
def get_qa_chain(retriever):
    llm = OllamaLLM(model=OLLAMA_MODEL)

    def qa_run(query: str):
        docs = retriever.get_relevant_documents(query)
        context = "\n\n".join([d.page_content for d in docs])
        prompt = f"""
Use ONLY the following context to answer the question.

Context:
{context}

Question: {query}

Answer:
"""
        # Run LLM
        try:
            answer = llm.invoke(prompt)
        except:
            answer = llm(prompt)

        return {"answer": answer, "context": docs}

    return qa_run


# ---------------- CLI Test ----------------
if __name__ == "__main__":
    retriever = load_retriever()
    qa = get_qa_chain(retriever)
    response = qa("What is the main contribution of the documents?")
    print("\nAnswer:\n", response["answer"])
    print("\nSource documents:")
    for doc in response.get("context", []):
        print(doc.metadata)
