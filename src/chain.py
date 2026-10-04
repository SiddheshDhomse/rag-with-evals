import logging
from typing import List, Dict, Any, Generator, Tuple, Optional

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel

from src.vectorstore import VectorStoreManager
from src.memory import ChatHistoryManager
from src.reranker import CrossEncoderReranker
from src.query_transform import QueryTransformer
from src.config import settings

logger = logging.getLogger(__name__)

# Prompt for turning a multi-turn conversational question into a standalone search query
CONTEXTUALIZE_Q_SYSTEM_PROMPT = """You are a search query reformulation engine.
Given the recent chat history and a user's follow-up question, your ONLY job is to rewrite the question into a concise, standalone search query that can be used to search a document index.

CRITICAL INSTRUCTIONS:
1. NEVER attempt to answer the question.
2. NEVER explain what the documents contain or what is missing.
3. NEVER output phrases like "I cannot find...", "Based on the text...", or "The documents discuss...".
4. If the question refers to prior conversation entities (e.g., "it", "they", "that decision", "the ruling"), replace those ambiguous pronouns with the specific subject/entity from chat history.
5. If the question is already self-contained, or if the chat history is about an unrelated topic, return the user's question EXACTLY AS IS.
6. Output ONLY the standalone search query string, with no quotation marks, no preamble, and no explanation.

Examples:
Chat History:
User: Tell me about the 2022 US Supreme Court ruling on abortion.
Assistant: The Supreme Court overturned Roe v. Wade in Dobbs v. Jackson Women's Health Organization...
User: How might the ruling affect funding for reproductive health organizations?
Output: How might the 2022 US Supreme Court abortion ruling affect funding for reproductive health organizations?

Chat History:
User: What is Article 207.3 of the Russian Criminal Code?
Assistant: Article 207.3 criminalizes public dissemination of knowingly false information...
User: How might the ruling affect funding for reproductive health organizations?
Output: How might the ruling affect funding for reproductive health organizations?
"""

CONTEXTUALIZE_Q_PROMPT = ChatPromptTemplate.from_messages([
    ("system", CONTEXTUALIZE_Q_SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}"),
])


def sanitize_reformulated_query(reformulated: str, original_query: str) -> str:
    """
    Validates and cleans the output from query reformulation.
    Rejects refusals, explanations, conversational filler, and multi-line paragraphs.
    """
    if not reformulated:
        return original_query

    cleaned = str(reformulated).strip().strip('"').strip("'").strip("`")

    # Strip conversational prefixes
    prefixes_to_strip = [
        "standalone question:",
        "rephrased question:",
        "search query:",
        "standalone query:",
        "output:",
        "query:",
        "here is the standalone question:",
        "reformulated question:"
    ]
    for prefix in prefixes_to_strip:
        if cleaned.lower().startswith(prefix):
            cleaned = cleaned[len(prefix):].strip().strip('"').strip("'")

    lower = cleaned.lower()

    # Guardrail 1: Refusal / explanation triggers
    refusal_triggers = [
        "cannot find",
        "can not find",
        "sufficient information",
        "provided documents",
        "knowledge base",
        "do not contain",
        "does not contain",
        "as an ai",
        "i am sorry",
        "there is no information",
        "context does not",
        "not mentioned in",
        "unable to answer",
        "i do not have"
    ]
    if any(trigger in lower for trigger in refusal_triggers):
        logger.warning(
            f"Query reformulation produced an LLM refusal instead of a search query: "
            f"'{cleaned[:70]}...'. Falling back to original user query."
        )
        return original_query

    # Guardrail 2: Multi-line / paragraph rejection
    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    if len(lines) > 2:
        logger.warning("Query reformulation produced multi-line text. Falling back to original query.")
        return original_query

    # Guardrail 3: Overly long output rejection
    words = cleaned.split()
    if len(words) > 35:
        logger.warning(
            f"Query reformulation produced overly verbose text ({len(words)} words). "
            f"Falling back to original user query."
        )
        return original_query

    return cleaned

# QA Generation Prompt with strict grounding and citation instructions
QA_SYSTEM_PROMPT = """You are a helpful, precise, and truth-focused AI assistant powered by a Retrieval-Augmented Generation (RAG) system.

Use the following retrieved context snippets to answer the user's question.
Guidelines:
1. Ground your answer ONLY in the provided context. Do NOT fabricate or assume facts not present in the context.
2. If the context does not contain enough information to answer the question, state clearly: "I cannot find sufficient information in the knowledge base to answer this question."
3. Keep the answer clear, structured, and concise.
4. Reference the document source(s) when helpful.

Context:
{context}"""

QA_PROMPT = ChatPromptTemplate.from_messages([
    ("system", QA_SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}"),
])


class ConversationalRAGChain:
    """
    Production conversational RAG chain that handles:
    1. Query reformulation with chat history context
    2. Vector store retrieval (Dense, Hybrid BM25+RRF, or Two-Stage Cross-Encoder Rerank)
    3. Grounded answer generation with streaming support
    4. Full candidate passage audit tracking for observability
    """

# Direct Conversational Prompt for bypassed chitchat queries
DIRECT_SYSTEM_PROMPT = """You are a helpful, polite, and knowledgeable AI assistant for an enterprise knowledge base.
Answer the user's conversational query clearly, concisely, and pleasantly."""

DIRECT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", DIRECT_SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}"),
])


class ConversationalRAGChain:
    """
    Production conversational RAG chain that handles:
    1. Query reformulation with chat history context
    2. Phase 4 Query Transformation (HyDE, Multi-Query, Step-Back, or Adaptive Routing)
    3. Vector store retrieval (Dense, Hybrid BM25+RRF, or Two-Stage Cross-Encoder Rerank)
    4. Grounded answer generation with streaming and direct chitchat bypass
    5. Full candidate passage and query transformation audit tracking for observability
    """

    def __init__(
        self,
        llm: BaseChatModel,
        vectorstore_manager: VectorStoreManager,
        memory_manager: ChatHistoryManager,
        k: int = 4,
        retrieval_mode: str = "hybrid_rerank",
        candidates_k: int = 15,
        reranker: Optional[CrossEncoderReranker] = None,
        query_transform_mode: str = "none",
        query_transformer: Optional[QueryTransformer] = None
    ):
        self.llm = llm
        self.vectorstore_manager = vectorstore_manager
        self.memory_manager = memory_manager
        self.k = k
        self.retrieval_mode = retrieval_mode.lower()
        self.candidates_k = max(candidates_k, k)
        self.reranker = reranker
        self.query_transform_mode = (query_transform_mode or settings.query_transform_mode or "none").lower()

        self.last_candidates_audit: List[Dict[str, Any]] = []
        self.last_transform_audit: Dict[str, Any] = {}

        if query_transformer is not None:
            self.query_transformer = query_transformer
        elif self.llm is not None:
            self.query_transformer = QueryTransformer(self.llm)
        else:
            self.query_transformer = None

        # Contextualize Chain
        if self.llm is not None:
            self.contextualize_chain = CONTEXTUALIZE_Q_PROMPT | self.llm | StrOutputParser()
        else:
            self.contextualize_chain = None

    def _format_docs(self, docs: List[Document]) -> str:
        formatted = []
        for i, doc in enumerate(docs):
            source = doc.metadata.get("source", f"Doc {i+1}")
            page = f" (Page {doc.metadata['page']})" if "page" in doc.metadata else ""
            formatted.append(f"--- Document [{i+1}] ({source}{page}) ---\n{doc.page_content}")
        return "\n\n".join(formatted)

    def _retrieve_raw_candidates(self, query: str, count: int) -> List[Tuple[Document, float]]:
        """Retrieves raw scored candidates from vectorstore using active retrieval mode."""
        if self.retrieval_mode == "dense":
            return self.vectorstore_manager.similarity_search_with_score(query=query, k=count)
        else:
            return self.vectorstore_manager.hybrid_search_with_score(query=query, k=count)

    def retrieve_context(
        self,
        question: str,
        session_id: str = "default",
        return_candidates: bool = False
    ) -> Any:
        """
        1. Reformulates query if history exists.
        2. Applies Phase 4 Query Transformation (HyDE, Multi-Query, Step-Back, or Adaptive Routing).
        3. Retrieves and fuses candidate pools across transformed queries.
        4. Applies Cross-Encoder Reranking (Phase 3) or rank-based selection.
        """
        history = self.memory_manager.get_langchain_messages(session_id, limit=6)

        standalone_question = question
        if history and self.contextualize_chain is not None:
            try:
                raw_rephrase = self.contextualize_chain.invoke({
                    "chat_history": history,
                    "question": question
                })
                standalone_question = sanitize_reformulated_query(raw_rephrase, question)
            except Exception as e:
                logger.warning(f"Error reformulating question, using original: {e}")
                standalone_question = question

        # Initialize Transformation Audit
        transform_audit = {
            "mode": self.query_transform_mode,
            "strategy": "standard",
            "route": "FACT_LOOKUP",
            "reasoning": "Standard direct retrieval execution.",
            "queries": [standalone_question],
            "hypothetical_doc": None,
            "sub_queries": [],
            "step_back_query": None,
            "direct_bypass": False
        }

        active_strategy = self.query_transform_mode

        # Step 2A: Adaptive Intent Routing
        if self.query_transform_mode == "adaptive" and self.query_transformer is not None:
            route_info = self.query_transformer.route_query(standalone_question)
            transform_audit["route"] = route_info.get("route", "FACT_LOOKUP")
            transform_audit["reasoning"] = route_info.get("reasoning", "")
            active_strategy = route_info.get("strategy", "standard")
            transform_audit["strategy"] = active_strategy

            if active_strategy == "direct":
                transform_audit["direct_bypass"] = True
                self.last_transform_audit = transform_audit
                self.last_candidates_audit = []
                if return_candidates:
                    return standalone_question, [], "", []
                return standalone_question, [], ""

        elif self.query_transform_mode in ("hyde", "multi_query", "step_back"):
            active_strategy = self.query_transform_mode
            transform_audit["strategy"] = active_strategy
            transform_audit["route"] = active_strategy.upper()

        score_label = "score"
        candidates_audit: List[Dict[str, Any]] = []
        docs_with_scores: List[Tuple[Document, float]] = []

        try:
            # Step 2B: Candidate Gathering & Fusion across Transformed Queries
            merged_candidates_dict: Dict[str, Tuple[Document, float]] = {}

            if active_strategy == "hyde" and self.query_transformer is not None:
                hypo_doc = self.query_transformer.generate_hyde(standalone_question)
                transform_audit["hypothetical_doc"] = hypo_doc
                transform_audit["queries"] = [standalone_question, "HYDE: " + hypo_doc[:60] + "..."]

                # Fetch candidates for both raw query and hypothetical document
                cands_q = self._retrieve_raw_candidates(standalone_question, self.candidates_k)
                cands_h = self._retrieve_raw_candidates(hypo_doc, self.candidates_k)

                for doc, sc in cands_q + cands_h:
                    key = doc.page_content.strip()
                    if key not in merged_candidates_dict or sc > merged_candidates_dict[key][1]:
                        merged_candidates_dict[key] = (doc, sc)

            elif active_strategy == "multi_query" and self.query_transformer is not None:
                sub_qs = self.query_transformer.generate_multi_query(
                    standalone_question,
                    count=settings.multi_query_count
                )
                transform_audit["sub_queries"] = sub_qs
                transform_audit["queries"] = sub_qs

                k_per = max(4, (self.candidates_k // max(len(sub_qs), 1)) + 2)
                for sq in sub_qs:
                    sq_cands = self._retrieve_raw_candidates(sq, k_per)
                    for doc, sc in sq_cands:
                        key = doc.page_content.strip()
                        if key not in merged_candidates_dict or sc > merged_candidates_dict[key][1]:
                            merged_candidates_dict[key] = (doc, sc)

            elif active_strategy == "step_back" and self.query_transformer is not None:
                sb_q = self.query_transformer.generate_step_back(standalone_question)
                transform_audit["step_back_query"] = sb_q
                transform_audit["queries"] = [standalone_question, sb_q]

                cands_q = self._retrieve_raw_candidates(standalone_question, self.candidates_k)
                cands_sb = self._retrieve_raw_candidates(sb_q, self.candidates_k)

                for doc, sc in cands_q + cands_sb:
                    key = doc.page_content.strip()
                    if key not in merged_candidates_dict or sc > merged_candidates_dict[key][1]:
                        merged_candidates_dict[key] = (doc, sc)

            else:
                # Standard single-query retrieval
                target_k = self.candidates_k if self.retrieval_mode == "hybrid_rerank" else self.k
                raw_cands = self._retrieve_raw_candidates(standalone_question, target_k)
                for doc, sc in raw_cands:
                    merged_candidates_dict[doc.page_content.strip()] = (doc, sc)

            merged_candidates = list(merged_candidates_dict.values())

            # Step 2C: Candidate Ranking & Stage 2 Cross-Encoder Evaluation
            if self.retrieval_mode == "hybrid_rerank":
                if self.reranker is None:
                    self.reranker = CrossEncoderReranker()

                docs_with_scores, candidates_audit = self.reranker.rerank(
                    query=standalone_question,
                    docs_with_scores=merged_candidates,
                    top_k=self.k
                )
                score_label = "cross_encoder_score"

            elif self.retrieval_mode == "dense":
                # Sort descending by cosine similarity / ascending distance
                docs_with_scores = sorted(merged_candidates, key=lambda x: x[1])[:self.k]
                score_label = "cosine_distance"
                for idx, (doc, sc) in enumerate(docs_with_scores):
                    candidates_audit.append({
                        "initial_rank": idx + 1,
                        "new_rank": idx + 1,
                        "rank_delta": 0,
                        "source": doc.metadata.get("source", "Unknown"),
                        "page": doc.metadata.get("page", None),
                        "content": doc.page_content,
                        "initial_score": round(float(sc), 4),
                        "rerank_score": round(float(sc), 4),
                        "normalized_score": round(float(sc), 4),
                        "confidence_pct": round((1.0 - min(float(sc), 1.0)) * 100, 1),
                        "selected": True
                    })

            else:
                # Hybrid RRF score descending
                docs_with_scores = sorted(merged_candidates, key=lambda x: x[1], reverse=True)[:self.k]
                score_label = "rrf_score"
                for idx, (doc, sc) in enumerate(docs_with_scores):
                    candidates_audit.append({
                        "initial_rank": idx + 1,
                        "new_rank": idx + 1,
                        "rank_delta": 0,
                        "source": doc.metadata.get("source", "Unknown"),
                        "page": doc.metadata.get("page", None),
                        "content": doc.page_content,
                        "initial_score": round(float(sc), 4),
                        "rerank_score": round(float(sc), 4),
                        "normalized_score": round(float(sc), 4),
                        "confidence_pct": round(min(float(sc) * 50, 1.0) * 100, 1),
                        "selected": True
                    })

        except Exception as e:
            logger.error(f"Error retrieving documents in mode '{self.retrieval_mode}' / transform '{active_strategy}': {e}")
            docs_with_scores = []
            candidates_audit = []

        self.last_candidates_audit = candidates_audit
        self.last_transform_audit = transform_audit

        sources_info: List[Dict[str, Any]] = []
        docs: List[Document] = []

        for doc, score in docs_with_scores:
            docs.append(doc)
            cand_info = next((c for c in candidates_audit if c.get("content") == doc.page_content), {})
            sources_info.append({
                "source": doc.metadata.get("source", "Unknown"),
                "page": doc.metadata.get("page", None),
                "content": doc.page_content,
                "score": round(float(score), 4),
                "score_type": score_label,
                "strategy": f"{self.retrieval_mode}+{active_strategy}",
                "initial_rank": cand_info.get("initial_rank", None),
                "new_rank": cand_info.get("new_rank", None),
                "rank_delta": cand_info.get("rank_delta", None),
                "initial_score": cand_info.get("initial_score", None),
                "confidence_pct": cand_info.get("confidence_pct", None)
            })

        formatted_context = self._format_docs(docs) if docs else "No relevant documents found."

        if return_candidates:
            return standalone_question, sources_info, formatted_context, candidates_audit
        return standalone_question, sources_info, formatted_context

    def stream_answer(
        self,
        question: str,
        session_id: str = "default"
    ) -> Generator[str, None, Tuple[str, List[Dict[str, Any]]]]:
        """
        Streams response tokens while yielding sources and saving candidate & transform audit at the end.
        """
        standalone_q, sources, context_str = self.retrieve_context(question, session_id)
        history = self.memory_manager.get_langchain_messages(session_id, limit=6)

        response_chunks = []

        # Direct chitchat bypass path
        if self.last_transform_audit.get("direct_bypass"):
            direct_chain = DIRECT_PROMPT | self.llm | StrOutputParser()
            for chunk in direct_chain.stream({
                "chat_history": history,
                "question": question
            }):
                response_chunks.append(chunk)
                yield chunk
        else:
            qa_chain = QA_PROMPT | self.llm | StrOutputParser()
            for chunk in qa_chain.stream({
                "context": context_str,
                "chat_history": history,
                "question": question
            }):
                response_chunks.append(chunk)
                yield chunk

        full_answer = "".join(response_chunks)

        # Save turn in conversational memory with sources, candidates, and transformation audit
        self.memory_manager.add_message(
            session_id,
            role="user",
            content=question
        )
        self.memory_manager.add_message(
            session_id,
            role="assistant",
            content=full_answer,
            sources=sources,
            candidates=self.last_candidates_audit,
            transform_audit=self.last_transform_audit
        )

        return full_answer, sources

    def invoke(
        self,
        question: str,
        session_id: str = "default"
    ) -> Dict[str, Any]:
        """Non-streaming execution of the RAG chain."""
        standalone_q, sources, context_str = self.retrieve_context(question, session_id)
        history = self.memory_manager.get_langchain_messages(session_id, limit=6)

        if self.last_transform_audit.get("direct_bypass"):
            direct_chain = DIRECT_PROMPT | self.llm | StrOutputParser()
            answer = direct_chain.invoke({
                "chat_history": history,
                "question": question
            })
        else:
            qa_chain = QA_PROMPT | self.llm | StrOutputParser()
            answer = qa_chain.invoke({
                "context": context_str,
                "chat_history": history,
                "question": question
            })

        # Save turn in conversational memory
        self.memory_manager.add_message(
            session_id,
            role="user",
            content=question
        )
        self.memory_manager.add_message(
            session_id,
            role="assistant",
            content=answer,
            sources=sources,
            candidates=self.last_candidates_audit,
            transform_audit=self.last_transform_audit
        )

        return {
            "answer": answer,
            "sources": sources,
            "standalone_question": standalone_q,
            "candidates": self.last_candidates_audit,
            "transform_audit": self.last_transform_audit
        }
