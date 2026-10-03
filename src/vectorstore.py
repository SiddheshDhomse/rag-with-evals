import os
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import chromadb
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import settings
from src.models import get_embedding_model


class VectorStoreManager:
    """
    Manages persistent ChromaDB vector store, document chunking,
    ingestion, and similarity search.
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

    def get_retriever(self, k: int = 4, search_type: str = "similarity"):
        """Returns a LangChain retriever interface for the vector store."""
        return self.vector_store.as_retriever(
            search_type=search_type,
            search_kwargs={"k": k}
        )

    def get_collection_stats(self) -> Dict[str, Any]:
        """Returns metadata stats about the current collection."""
        try:
            col = self.chroma_client.get_collection(self.collection_name)
            total_chunks = col.count()

            # Retrieve unique source names if any chunks exist
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
        """
        Splits documents into chunks and indexes them into ChromaDB.
        Returns the number of chunks added.
        """
        if not documents:
            return 0

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_documents(documents)

        # Ensure all chunks have unique ids and normalized metadata
        ids = [str(uuid.uuid4()) for _ in chunks]
        for i, chunk in enumerate(chunks):
            if "source" not in chunk.metadata:
                chunk.metadata["source"] = f"doc_{i}"

        self.vector_store.add_documents(documents=chunks, ids=ids)
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

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4
    ) -> List[Tuple[Document, float]]:
        """
        Searches for the most similar documents and returns their distance/similarity scores.
        """
        return self.vector_store.similarity_search_with_score(query, k=k)

    def clear_collection(self) -> bool:
        """Deletes the current collection from ChromaDB and re-initializes."""
        try:
            self.chroma_client.delete_collection(self.collection_name)
            self._init_vectorstore()
            return True
        except Exception:
            self._init_vectorstore()
            return False
