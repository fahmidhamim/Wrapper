import os
import re
import threading
from typing import Any, Dict, List

import faiss
import numpy as np
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from sentence_transformers import SentenceTransformer


class TranscriptIndex:
    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ) -> None:
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model = SentenceTransformer(embedding_model)
        self.index = None
        self.metadata: List[Dict[str, Any]] = []
        self.text = ""

    @property
    def has_data(self) -> bool:
        return self.index is not None and bool(self.metadata)

    def build(self, text: str) -> None:
        self.text = text or ""
        self.index = None
        self.metadata = []

        if not self.text.strip():
            return

        documents = [Document(page_content=self.text, metadata={"source": "transcript"})]
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
        )
        chunks = splitter.split_documents(documents)
        texts = [chunk.page_content for chunk in chunks if chunk.page_content]
        if not texts:
            return

        embeddings = self.model.encode(texts, show_progress_bar=False)
        embeddings = np.array(embeddings).astype("float32")
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings)
        self.metadata = [{"text": text} for text in texts]

    def query(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if not self.has_data:
            return []

        query_emb = self.model.encode([query_text]).astype("float32")
        distances, indexes = self.index.search(query_emb, top_k)
        results = []
        for idx, dist in zip(indexes[0], distances[0]):
            if idx < 0:
                continue
            meta = self.metadata[idx] if idx < len(self.metadata) else None
            results.append({"index": int(idx), "distance": float(dist), "metadata": meta})
        return results


class TranscriptRAG:
    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        llm_model: str = "llama-3.3-70b-versatile",
        temperature: float = 0.3,
    ) -> None:
        self._lock = threading.Lock()
        self.index = TranscriptIndex(embedding_model=embedding_model)

        groq_api_key = os.getenv("GROQ_API_KEY", "")
        if groq_api_key:
            self.llm = ChatGroq(
                groq_api_key=groq_api_key,
                model_name=llm_model,
                temperature=temperature,
            )
        else:
            self.llm = None

    def set_transcript(self, text: str) -> None:
        with self._lock:
            self.index.build(text)

    def clear(self) -> None:
        with self._lock:
            self.index.build("")

    def has_transcript(self) -> bool:
        return self.index.has_data

    def answer(self, query: str, top_k: int = 5) -> str:
        with self._lock:
            results = self.index.query(query, top_k=top_k)

        texts = [r["metadata"].get("text", "") for r in results if r["metadata"]]
        context = "\n\n".join([text for text in texts if text])

        if not context:
            return "No relevant transcript segments found."

        if not self.llm:
            return self._best_sentence_answer(query, context)

        prompt = (
            "Answer the question using ONLY the transcript context.\n"
            "Reply with ONE short sentence. Do not include the context, "
            "do not quote the question, and do not add extra commentary.\n"
            "If the answer is not explicitly in the context, reply: "
            "'Not found in transcript.'\n\n"
            f"Question: {query}\n\n"
            f"Context:\n{context}\n\n"
            "Answer:"
        )
        response = self.llm.invoke(prompt)
        return self._sanitize_answer(response.content)

    def _sanitize_answer(self, text: str) -> str:
        if not text:
            return "Not found in transcript."

        cleaned = text.strip()
        cleaned = re.sub(r"^answer\\s*[:\\-]\\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"^response\\s*[:\\-]\\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = cleaned.splitlines()[0].strip()

        sentences = re.split(r"(?<=[.!?])\\s+", cleaned)
        if sentences and sentences[0].strip():
            return sentences[0].strip()

        return cleaned or "Not found in transcript."

    def _best_sentence_answer(self, query: str, context: str) -> str:
        sentences = re.split(r"(?<=[.!?])\\s+", context.strip())
        if not sentences:
            return "Not found in transcript."

        query_tokens = set(re.findall(r"[a-z0-9]+", query.lower()))
        if not query_tokens:
            return sentences[0].strip()

        best_sentence = ""
        best_score = 0
        for sentence in sentences:
            sentence_tokens = set(re.findall(r"[a-z0-9]+", sentence.lower()))
            if not sentence_tokens:
                continue
            score = sum(1 for token in query_tokens if token in sentence_tokens)
            if score > best_score:
                best_score = score
                best_sentence = sentence.strip()

        if best_score == 0:
            return "Not found in transcript."

        return best_sentence
