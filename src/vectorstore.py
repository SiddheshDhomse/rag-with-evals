import os
import uuid
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import chromadb
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import settings
from src.models import get_embedding_model

logger = logging.getLogger(__name__)


class VectorStoreManager:
    """
    Manages persistent ChromaDB vector store, document chunking,
    dense vector indexing, BM25 sparse keyword indexing, and Hybrid
    Retrieval via Reciprocal Rank Fusion (RRF).
    """

    def __init__(
        self,
        collection_name: Optional[str] = None,
        embedding_model: Optional[Embeddings] = None,
        persist_directory: Optional[Path] = None
    ):
        self.collection_name = collection_name or settings.collection_name
        self.persist_directory = persist_directory or settings.chroma_dir
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        self.embedding_model = embedding_model or get_embedding_model()
        self.chroma_client = chromadb.PersistentClient(path=str(self.persist_directory))

        # Caching for BM25 and document corpus
        self._bm25_retriever: Optional[Any] = None
        self._all_documents_cache: Optional[List[Document]] = None

        self._init_vectorstore()

    def _init_vectorstore(self):
        """Initializes LangChain Chroma wrapper."""
        try:
            from langchain_chroma import Chroma
            self.vector_store = Chroma(
                client=self.chroma_client,
                collection_name=self.collection_name,
                embedding_function=self.embedding_model
            )
        except ImportError:
            from langchain_community.vectorstores import Chroma
            self.vector_store = Chroma(
                client=self.chroma_client,
                collection_name=self.collection_name,
                embedding_function=self.embedding_model
            )

    def _invalidate_bm25_cache(self):
        """Invalidates BM25 retriever and cached documents when corpus updates."""
        self._bm25_retriever = None
        self._all_documents_cache = None

    def _get_all_documents(self) -> List[Document]:
        """Loads all documents currently indexed in ChromaDB collection."""
        if self._all_documents_cache is not None:
            return self._all_documents_cache

        try:
            col = self.chroma_client.get_collection(self.collection_name)
            count = col.count()
            if count == 0:
                self._all_documents_cache = []
                return []

            results = col.get(include=["documents", "metadatas"])
            docs: List[Document] = []
            for text, meta in zip(results.get("documents", []), results.get("metadatas", [])):
                docs.append(Document(page_content=text, metadata=meta or {}))
            
            self._all_documents_cache = docs
            return docs
        except Exception as e:
            logger.warning(f"Error fetching collection documents for BM25: {e}")
            return []

    def get_dense_retriever(self, k: int = 4, search_type: str = "similarity"):
        """Returns standard dense vector similarity retriever."""
        return self.vector_store.as_retriever(
            search_type=search_type,
            search_kwargs={"k": k}
        )

    def get_bm25_retriever(self, k: int = 4):
        """Returns BM25 sparse keyword retriever built from the active corpus."""
        if self._bm25_retriever is None:
            docs = self._get_all_documents()
            if not docs:
                logger.warning("No documents in collection to build BM25 retriever.")
                return None
            try:
                from langchain_community.retrievers import BM25Retriever
                self._bm25_retriever = BM25Retriever.from_documents(docs)
            except Exception as e:
                logger.error(f"Error initializing BM25Retriever: {e}")
                return None

        self._bm25_retriever.k = k
        return self._bm25_retriever

    def get_hybrid_retriever(
        self,
        k: int = 4,
        dense_weight: float = 0.5,
        bm25_weight: float = 0.5
    ):
        """
        Returns a LangChain EnsembleRetriever combining BM25 and Chroma dense search
        using Reciprocal Rank Fusion (RRF).
        """
        dense_retriever = self.get_dense_retriever(k=k)
        bm25_retriever = self.get_bm25_retriever(k=k)

        if bm25_retriever is None:
            logger.warning("BM25 retriever unavailable, falling back to dense retriever.")
            return dense_retriever

        try:
            from langchain_classic.retrievers import EnsembleRetriever
            return EnsembleRetriever(
                retrievers=[bm25_retriever, dense_retriever],
                weights=[bm25_weight, dense_weight]
            )
        except Exception as e:
            logger.warning(f"Could not initialize EnsembleRetriever ({e}), using dense retriever.")
            return dense_retriever

    def get_retriever(self, k: int = 4, mode: str = "hybrid"):
        """
        General retriever factory:
        - 'hybrid': Reciprocal Rank Fusion of BM25 + Dense ChromaDB (Default)
        - 'dense': Pure Dense ChromaDB vector search
        - 'bm25': Pure BM25 sparse lexical search
        """
        mode = mode.lower()
        if mode == "dense":
            return self.get_dense_retriever(k=k)
        elif mode == "bm25":
            bm25 = self.get_bm25_retriever(k=k)
            return bm25 if bm25 is not None else self.get_dense_retriever(k=k)
        else:
            return self.get_hybrid_retriever(k=k)

    def hybrid_search_with_score(
        self,
        query: str,
        k: int = 4,
        dense_weight: float = 0.5,
        bm25_weight: float = 0.5,
        c: int = 60
    ) -> List[Tuple[Document, float]]:
        """
        Executes Hybrid Search combining Dense Vector Search and BM25 Sparse Search
        via Reciprocal Rank Fusion (RRF). Returns top-k documents with their RRF scores.
        """
        candidate_k = max(k * 3, 12)

        # 1. Dense candidates
        try:
            dense_results = self.similarity_search_with_score(query, k=candidate_k)
        except Exception as e:
            logger.error(f"Error in dense similarity search: {e}")
            dense_results = []

        # 2. BM25 candidates
        bm25_retriever = self.get_bm25_retriever(k=candidate_k)
        bm25_results: List[Document] = []
        if bm25_retriever is not None:
            try:
                bm25_results = bm25_retriever.invoke(query)
            except Exception as e:
                logger.error(f"Error in BM25 retrieval: {e}")
                bm25_results = []

        # Graceful fallbacks if one branch fails
        if not bm25_results:
            return dense_results[:k]
        if not dense_results:
            return [(doc, round(1.0 / (c + rank), 5)) for rank, doc in enumerate(bm25_results[:k], 1)]

        # 3. Reciprocal Rank Fusion
        rrf_scores: Dict[str, float] = {}
        doc_map: Dict[str, Document] = {}

        for rank, (doc, _) in enumerate(dense_results, 1):
            key = doc.page_content.strip()
            doc_map[key] = doc
            rrf_scores[key] = rrf_scores.get(key, 0.0) + (dense_weight / (c + rank))

        for rank, doc in enumerate(bm25_results, 1):
            key = doc.page_content.strip()
            doc_map[key] = doc
            rrf_scores[key] = rrf_scores.get(key, 0.0) + (bm25_weight / (c + rank))

        # Sort descending by fused RRF score
        sorted_items = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)[:k]
        return [(doc_map[key], round(score, 5)) for key, score in sorted_items]

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4
    ) -> List[Tuple[Document, float]]:
        """Dense similarity search returning documents and cosine distance scores."""
        return self.vector_store.similarity_search_with_score(query, k=k)

    def get_collection_stats(self) -> Dict[str, Any]:
        """Returns metadata stats about the current collection."""
        try:
            col = self.chroma_client.get_collection(self.collection_name)
            total_chunks = col.count()

            sources = set()
            if total_chunks > 0:
                results = col.get(limit=min(total_chunks, 1000), include=["metadatas"])
                for meta in results.get("metadatas", []):
                    if meta and "source" in meta:
                        sources.add(Path(meta["source"]).name)

            return {
                "collection_name": self.collection_name,
                "total_chunks": total_chunks,
                "sources_count": len(sources),
                "sources": sorted(list(sources))
            }
        except Exception:
            return {
                "collection_name": self.collection_name,
                "total_chunks": 0,
                "sources_count": 0,
                "sources": []
            }

    def add_documents(
        self,
        documents: List[Document],
        chunk_size: int = 600,
        chunk_overlap: int = 100
    ) -> int:
        """Splits documents into chunks and indexes them into ChromaDB."""
        if not documents:
            return 0

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_documents(documents)

        ids = [str(uuid.uuid4()) for _ in chunks]
        for i, chunk in enumerate(chunks):
            if "source" not in chunk.metadata:
                chunk.metadata["source"] = f"doc_{i}"

        self.vector_store.add_documents(documents=chunks, ids=ids)
        self._invalidate_bm25_cache()
        return len(chunks)

    def add_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        chunk_size: int = 600,
        chunk_overlap: int = 100
    ) -> int:
        """Helper to add raw text strings."""
        docs = []
        for i, text in enumerate(texts):
            meta = metadatas[i] if metadatas and i < len(metadatas) else {"source": f"text_snippet_{i+1}"}
            docs.append(Document(page_content=text, metadata=meta))
        return self.add_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    def clear_collection(self) -> bool:
        """Deletes the current collection from ChromaDB and re-initializes."""
        try:
            self.chroma_client.delete_collection(self.collection_name)
            self._init_vectorstore()
            self._invalidate_bm25_cache()
            return True
        except Exception:
            self._init_vectorstore()
            self._invalidate_bm25_cache()
            return False
