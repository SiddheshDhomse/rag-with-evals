import logging
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


def sigmoid(x: float) -> float:
    """Computes standard sigmoid to map unconstrained logits to [0, 1]."""
    return float(1.0 / (1.0 + np.exp(-x)))


class CrossEncoderReranker:
    """
    Two-stage retrieval Cross-Encoder reranker.
    Scores (query, passage) pairs jointly with deep cross-attention to
    maximize Context Precision and filter out false-positive distractors.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: Optional[str] = None
    ):
        self.model_name = model_name
        self.device = device
        self._model = None

    @property
    def model(self):
        """Lazy load the CrossEncoder model on first call."""
        if self._model is None:
            logger.info(f"Loading CrossEncoder model '{self.model_name}'...")
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name, device=self.device)
            logger.info(f"CrossEncoder model '{self.model_name}' loaded successfully.")
        return self._model

    def rerank(
        self,
        query: str,
        docs_with_scores: List[Tuple[Document, float]],
        top_k: int = 4
    ) -> Tuple[List[Tuple[Document, float]], List[Dict[str, Any]]]:
        """
        Reranks retrieved candidate documents using cross-attention.

        Args:
            query: The user query or reformulated search query.
            docs_with_scores: Initial candidate documents with their 1st-stage scores (e.g. RRF or cosine distance).
            top_k: Number of highest-scoring documents to return for final answer synthesis.

        Returns:
            Tuple of:
              - top_k_reranked: List of (Document, rerank_score) sorted descending
              - candidates_audit: Full list of all candidate chunks with initial vs reranked ranks,
                                  scores, deltas, and selection status (100% JSON serializable).
        """
        if not docs_with_scores:
            return [], []

        # 1. Prepare (query, doc_text) pairs for Cross-Encoder
        pairs = [(query, doc.page_content) for doc, _ in docs_with_scores]

        # 2. Predict cross-attention scores
        try:
            raw_scores = self.model.predict(pairs)
            if isinstance(raw_scores, (int, float, np.floating)):
                raw_scores = [float(raw_scores)]
        except Exception as e:
            logger.error(f"Error predicting cross-encoder scores: {e}. Falling back to 1st-stage ranks.")
            raw_scores = [0.0] * len(docs_with_scores)

        # 3. Build candidate audit entries paired with Document object
        paired_candidates: List[Tuple[Document, Dict[str, Any]]] = []
        for idx, ((doc, init_score), raw_score) in enumerate(zip(docs_with_scores, raw_scores)):
            s = float(raw_score)
            norm_s = sigmoid(s)
            audit_item = {
                "initial_rank": idx + 1,
                "source": doc.metadata.get("source", "Unknown"),
                "page": doc.metadata.get("page", None),
                "content": doc.page_content,
                "initial_score": round(float(init_score), 4),
                "rerank_score": round(s, 4),
                "normalized_score": round(norm_s, 4),
                "confidence_pct": round(norm_s * 100, 1)
            }
            paired_candidates.append((doc, audit_item))

        # 4. Sort all candidates descending by rerank_score
        paired_candidates.sort(key=lambda item: item[1]["rerank_score"], reverse=True)

        # 5. Compute new rank, rank delta (movement), and selection flag
        candidates_audit: List[Dict[str, Any]] = []
        for new_idx, (doc, cand) in enumerate(paired_candidates):
            new_rank = new_idx + 1
            cand["new_rank"] = new_rank
            cand["rank_delta"] = cand["initial_rank"] - new_rank  # positive = moved UP, negative = moved DOWN
            cand["selected"] = new_rank <= top_k
            candidates_audit.append(cand)

        # 6. Extract top-k reranked Document objects with their rerank scores
        top_k_reranked = [
            (paired_candidates[i][0], paired_candidates[i][1]["rerank_score"])
            for i in range(min(top_k, len(paired_candidates)))
        ]

        return top_k_reranked, candidates_audit
