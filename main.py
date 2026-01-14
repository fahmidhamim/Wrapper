from src.data_loader import load_all_documents
from src.vectorstore import FaissVectorStore
from src.search import RAGSearch
import os

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Example usage
if __name__ == "__main__":
    
    # Check if index already exists
    index_path = "faiss_store/faiss.index"
    
    if os.path.exists(index_path):
        print("[INFO] Loading existing FAISS index...")
        rag_search = RAGSearch()
    else:
        print("[INFO] Building new FAISS index from documents...")
        docs = load_all_documents("data")
        store = FaissVectorStore("faiss_store")
        store.build_from_documents(docs)
        rag_search = RAGSearch()
    
    # Interactive query loop
    print("\n" + "="*50)
    print("RAG Search System - Interactive Mode")
    print("="*50)
    print("Type 'exit' or 'quit' to stop")
    print("Type 'rebuild' to rebuild index from new data\n")
    
    while True:
        query = input("Ask a question: ").strip()
        
        if query.lower() in ['exit', 'quit']:
            print("Goodbye!")
            break
        
        if query.lower() == 'rebuild':
            print("[INFO] Rebuilding index from documents...")
            docs = load_all_documents("data")
            store = FaissVectorStore("faiss_store")
            store.build_from_documents(docs)
            rag_search = RAGSearch()
            print("[INFO] Index rebuilt successfully!\n")
            continue
        
        if not query:
            print("Please enter a valid question.\n")
            continue
        
        print("\nSearching...\n")
        answer = rag_search.search_and_summarize(query, top_k=3)
        print("Answer:", answer)
        print("\n" + "-"*50 + "\n")