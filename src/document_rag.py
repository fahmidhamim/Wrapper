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

from src.local_llm import LocalLLM
DEFAULT_SIMILARITY_THRESHOLD = float(
    os.getenv("DOC_SIMILARITY_THRESHOLD", "0.26")
)
DEFAULT_FALLBACK_SIMILARITY_THRESHOLD = float(
    os.getenv("DOC_FALLBACK_SIMILARITY_THRESHOLD", "0.18")
)
DEFAULT_ALLOW_GENERAL_FALLBACK = os.getenv(
    "DOC_ALLOW_GENERAL_FALLBACK", "true"
).lower() in ("1", "true", "yes", "y", "on")
DEFAULT_LOCAL_LLM_ENABLED = os.getenv(
    "OLLAMA_ENABLED", "true"
).lower() in ("1", "true", "yes", "y", "on")

FOLLOWUP_TOKENS = {
    "it",
    "this",
    "that",
    "these",
    "those",
    "they",
    "them",
    "he",
    "she",
    "his",
    "her",
    "their",
    "more",
    "explain",
    "detail",
    "details",
    "clarify",
    "why",
    "how",
    "what",
    "which",
}

STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "to",
    "of",
    "in",
    "on",
    "for",
    "with",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "by",
    "at",
    "as",
    "from",
    "about",
    "into",
    "over",
    "after",
    "before",
    "between",
    "but",
    "so",
    "if",
    "then",
}


class DocumentIndex:
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

    @property
    def has_data(self) -> bool:
        return self.index is not None and bool(self.metadata)

    def build(self, documents: List[Document]) -> None:
        self.index = None
        self.metadata = []

        if not documents:
            return

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
        faiss.normalize_L2(embeddings)

        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)

        self.metadata = []
        for chunk in chunks:
            self.metadata.append(
                {
                    "text": chunk.page_content,
                    "source": chunk.metadata.get("source"),
                }
            )

    def query(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if not self.has_data:
            return []

        query_emb = self.model.encode([query_text]).astype("float32")
        faiss.normalize_L2(query_emb)
        scores, indexes = self.index.search(query_emb, top_k)

        results = []
        for idx, score in zip(indexes[0], scores[0]):
            if idx < 0:
                continue
            meta = self.metadata[idx] if idx < len(self.metadata) else None
            results.append(
                {"index": int(idx), "score": float(score), "metadata": meta}
            )
        return results


class DocumentRAG:
    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        llm_model: str = "llama-3.3-70b-versatile",
        temperature: float = 0.3,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        allow_general_fallback: bool = DEFAULT_ALLOW_GENERAL_FALLBACK,
    ) -> None:
        self._lock = threading.Lock()
        self.index = DocumentIndex(embedding_model=embedding_model)
        self.similarity_threshold = similarity_threshold
        self.allow_general_fallback = allow_general_fallback
        self.fallback_similarity_threshold = DEFAULT_FALLBACK_SIMILARITY_THRESHOLD
        self.history: List[Dict[str, str]] = []
        self.last_topic = ""
        self.local_llm = LocalLLM() if DEFAULT_LOCAL_LLM_ENABLED else None

        groq_api_key = os.getenv("GROQ_API_KEY", "")
        if groq_api_key:
            self.llm = ChatGroq(
                groq_api_key=groq_api_key,
                model_name=llm_model,
                temperature=temperature,
            )
        else:
            self.llm = None

    def set_documents(self, documents: List[Document]) -> None:
        with self._lock:
            self.index.build(documents)
            self.history = []
            self.last_topic = ""

    def clear(self) -> None:
        with self._lock:
            self.index.build([])
            self.history = []
            self.last_topic = ""

    def has_documents(self) -> bool:
        return self.index.has_data

    def reset_conversation(self) -> None:
        with self._lock:
            self.history = []
            self.last_topic = ""

    def current_topic(self) -> str:
        with self._lock:
            return self.last_topic

    def answer(self, query: str, top_k: int = 5) -> str:
        with self._lock:
            topic_hint = self.last_topic
            retrieval_query = self._augment_query(query, topic_hint)
            results = self.index.query(retrieval_query, top_k=top_k)
            history_snapshot = list(self.history[-2:])

        if not results:
            return self._general_fallback(query)

        best_score = results[0]["score"]
        if best_score < self.similarity_threshold:
            if self.allow_general_fallback and best_score >= self.fallback_similarity_threshold:
                return self._general_fallback(query)
            return "Not found in document."

        texts = [r["metadata"].get("text", "") for r in results if r["metadata"]]
        context = "\n\n".join([text for text in texts if text])

        if not context:
            return self._general_fallback(query)

        if self.llm:
            answer = self._answer_with_context(query, context, history_snapshot)
            self._record_turn(query, answer, context)
            return answer
        if self.local_llm:
            answer = self._answer_with_context_local(query, context, history_snapshot)
            self._record_turn(query, answer, context)
            return answer
        answer = self._best_sentence_answer(query, context)
        self._record_turn(query, answer, context)
        return answer

    def _answer_with_context(
        self,
        query: str,
        context: str,
        history_snapshot: List[Dict[str, str]],
    ) -> str:
        history_block = self._history_block(history_snapshot)
        prompt = (
            "Answer the question using ONLY the provided context.\n"
            "Reply with ONE short sentence. Do not include the context, "
            "do not quote the question, and do not add extra commentary.\n"
            "If the answer is not explicitly in the context, reply: "
            "'Not found in document.'\n\n"
            f"Recent conversation (for disambiguation only):\n{history_block}\n\n"
            f"Question: {query}\n\n"
            f"Context:\n{context}\n\n"
            "Answer:"
        )
        response = self.llm.invoke(prompt)
        strict_answer = self._sanitize_answer(response.content)
        if self._is_not_found(strict_answer) and self.allow_general_fallback:
            return self._general_fallback(query)
        return strict_answer

    def _answer_with_context_local(
        self,
        query: str,
        context: str,
        history_snapshot: List[Dict[str, str]],
    ) -> str:
        history_block = self._history_block(history_snapshot)
        prompt = (
            "Answer the question using ONLY the provided context.\n"
            "Reply with ONE short sentence. Do not include the context, "
            "do not quote the question, and do not add extra commentary.\n"
            "If the answer is not explicitly in the context, reply: "
            "'Not found in document.'\n\n"
            f"Recent conversation (for disambiguation only):\n{history_block}\n\n"
            f"Question: {query}\n\n"
            f"Context:\n{context}\n\n"
            "Answer:"
        )
        response = self.local_llm.generate(prompt)
        strict_answer = self._sanitize_answer(response)
        if self._is_not_found(strict_answer) and self.allow_general_fallback:
            return self._general_fallback(query)
        return strict_answer

    def _general_fallback(self, query: str) -> str:
        if not self.allow_general_fallback:
            return "Not found in document."

        prompt = (
            "Provide a concise answer in ONE sentence.\n"
            "Do not mention the document or say that you lack context.\n"
            "If you are not sure, reply: 'Not sure.'\n\n"
            f"Question: {query}\n\n"
            "Answer:"
        )

        if self.local_llm:
            response = self.local_llm.generate(prompt)
            answer = self._sanitize_answer(response)
            if answer and not self._is_not_found(answer):
                self._record_turn(query, answer, context="")
                return answer

        if self.llm:
            response = self.llm.invoke(prompt)
            answer = self._sanitize_answer(response.content)
            if answer and not self._is_not_found(answer):
                self._record_turn(query, answer, context="")
                return answer

        return "Not found in document."

    def _history_block(self, history_snapshot: List[Dict[str, str]]) -> str:
        if not history_snapshot:
            return ""
        lines = []
        for item in history_snapshot:
            lines.append(f"Q: {item['q']}")
            lines.append(f"A: {item['a']}")
        return "\n".join(lines)

    def _sanitize_answer(self, text: str) -> str:
        if not text:
            return "Not found in document."

        cleaned = text.strip()
        cleaned = re.sub(r"^answer\\s*[:\\-]\\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"^response\\s*[:\\-]\\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = cleaned.splitlines()[0].strip()

        sentences = re.split(r"(?<=[.!?])\\s+", cleaned)
        if sentences and sentences[0].strip():
            return sentences[0].strip()

        return cleaned or "Not found in document."

    def _is_not_found(self, text: str) -> bool:
        return text.strip().lower() in {
            "not found in document.",
            "not found in document",
        }

    def _best_sentence_answer(self, query: str, context: str) -> str:
        sentences = re.split(r"(?<=[.!?])\\s+", context.strip())
        if not sentences:
            return "Not found in document."

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
            return "Not found in document."

        return best_sentence

    def _augment_query(self, query: str, topic_hint: str) -> str:
        if not topic_hint:
            return query
        if self._needs_context(query):
            return f"{query} {topic_hint}"
        return query

    def _needs_context(self, query: str) -> bool:
        tokens = self._tokens(query)
        if len(tokens) < 4:
            return True
        return any(token in FOLLOWUP_TOKENS for token in tokens)

    def _tokens(self, text: str) -> List[str]:
        return re.findall(r"[a-z0-9]+", text.lower())

    def _record_turn(self, query: str, answer: str, context: str) -> None:
        if self._is_not_found(answer):
            return
        topic_hint = self._extract_topic_hint(query, context)
        with self._lock:
            if topic_hint:
                self.last_topic = topic_hint
            self.history.append({"q": query, "a": answer})
            if len(self.history) > 10:
                self.history = self.history[-10:]

    def _extract_topic_hint(self, query: str, context: str) -> str:
        tokens = [t for t in self._tokens(query) if t not in STOPWORDS]
        tokens = [t for t in tokens if len(t) > 2]
        if tokens:
            return " ".join(tokens[:6])

        sentence = context.strip().split("\n")[0]
        return sentence[:80].strip()
