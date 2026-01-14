import os
from dotenv import load_dotenv
from src.vectorstore import FaissVectorStore
from langchain_groq import ChatGroq

load_dotenv()

class RAGSearch:
    def __init__(self, persist_dir: str = "faiss_store", embedding_model: str = "all-MiniLM-L6-v2", llm_model: str = "llama-3.3-70b-versatile"):
        self.vectorstore = FaissVectorStore(persist_dir, embedding_model)
        # Load or build vectorstore
        faiss_path = os.path.join(persist_dir, "faiss.index")
        meta_path = os.path.join(persist_dir, "metadata.pkl")
        if not (os.path.exists(faiss_path) and os.path.exists(meta_path)):
            from data_loader import load_all_documents
            docs = load_all_documents("data")
            self.vectorstore.build_from_documents(docs)
        else:
            self.vectorstore.load()
        
        groq_api_key = os.getenv("GROQ_API_KEY", "")
        if groq_api_key:
            self.llm = ChatGroq(groq_api_key=groq_api_key, model_name=llm_model, temperature=0.7)
            print(f"[INFO] Groq LLM initialized: {llm_model}")
        else:
            self.llm = None
            print("[WARNING] GROQ_API_KEY not found. LLM summarization disabled.")
        
        print(f"[INFO] RAG Search initialized with vectorstore: {persist_dir}")

    def search_and_summarize(self, query: str, top_k: int = 5) -> str:
        results = self.vectorstore.query(query, top_k=top_k)
        texts = [r["metadata"].get("text", "") for r in results if r["metadata"]]
        context = "\n\n".join(texts)
        if not context:
            return "No relevant documents found."
        
        if not self.llm:
            return f"Top {top_k} relevant passages for query '{query}':\n\n{context}"
        
        # Use LLM to summarize
        prompt = f"""Based on the following context, provide a concise answer to the query.

Query: {query}

Context:
{context}

Answer:"""
        response = self.llm.invoke(prompt)
        return response.content

