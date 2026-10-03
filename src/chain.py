import logging
from typing import List, Dict, Any, Generator, Tuple, Optional

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel

from src.vectorstore import VectorStoreManager
from src.memory import ChatHistoryManager

logger = logging.getLogger(__name__)

# Prompt for turning a multi-turn conversational question into a standalone search query
CONTEXTUALIZE_Q_SYSTEM_PROMPT = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""

CONTEXTUALIZE_Q_PROMPT = ChatPromptTemplate.from_messages([
    ("system", CONTEXTUALIZE_Q_SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}"),
])

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
    2. Vector store retrieval with relevance scores
    3. Grounded answer generation with streaming support
    """

    def __init__(
        self,
        llm: BaseChatModel,
        vectorstore_manager: VectorStoreManager,
        memory_manager: ChatHistoryManager,
        k: int = 4
    ):
        self.llm = llm
        self.vectorstore_manager = vectorstore_manager
        self.memory_manager = memory_manager
        self.k = k

        # Chains
        self.contextualize_chain = CONTEXTUALIZE_Q_PROMPT | self.llm | StrOutputParser()

    def _format_docs(self, docs: List[Document]) -> str:
        formatted = []
        for i, doc in enumerate(docs):
            source = doc.metadata.get("source", f"Doc {i+1}")
            page = f" (Page {doc.metadata['page']})" if "page" in doc.metadata else ""
            formatted.append(f"--- Document [{i+1}] ({source}{page}) ---\n{doc.page_content}")
        return "\n\n".join(formatted)

    def retrieve_context(
        self,
        question: str,
        session_id: str = "default"
    ) -> Tuple[str, List[Dict[str, Any]], str]:
        """
        1. Reformulates query if history exists.
        2. Retrieves top-k documents from ChromaDB with distance/similarity.
        Returns: (standalone_question, sources_list, formatted_context_str)
        """
        history = self.memory_manager.get_langchain_messages(session_id, limit=6)

        standalone_question = question
        if history:
            try:
                standalone_question = self.contextualize_chain.invoke({
                    "chat_history": history,
                    "question": question
                }).strip()
            except Exception as e:
                logger.warning(f"Error reformulating question, using original: {e}")
                standalone_question = question

        # Retrieve documents with distance scores
        try:
            docs_with_scores = self.vectorstore_manager.similarity_search_with_score(
                query=standalone_question,
                k=self.k
            )
        except Exception as e:
            logger.error(f"Error querying vector store: {e}")
            docs_with_scores = []

        sources_info: List[Dict[str, Any]] = []
        docs: List[Document] = []

        for doc, score in docs_with_scores:
            docs.append(doc)
            sources_info.append({
                "source": doc.metadata.get("source", "Unknown"),
                "page": doc.metadata.get("page", None),
                "content": doc.page_content,
                "score": round(float(score), 4)
            })

        formatted_context = self._format_docs(docs) if docs else "No relevant documents found."
        return standalone_question, sources_info, formatted_context

    def stream_answer(
        self,
        question: str,
        session_id: str = "default"
    ) -> Generator[str, None, Tuple[str, List[Dict[str, Any]]]]:
        """
        Streams response tokens while yielding sources at the end.
        """
        standalone_q, sources, context_str = self.retrieve_context(question, session_id)
        history = self.memory_manager.get_langchain_messages(session_id, limit=6)

        qa_chain = QA_PROMPT | self.llm | StrOutputParser()

        response_chunks = []
        for chunk in qa_chain.stream({
            "context": context_str,
            "chat_history": history,
            "question": question
        }):
            response_chunks.append(chunk)
            yield chunk

        full_answer = "".join(response_chunks)

        # Save turn in conversational memory
        self.memory_manager.add_message(session_id, role="user", content=question)
        self.memory_manager.add_message(session_id, role="assistant", content=full_answer, sources=sources)

        return full_answer, sources

    def invoke(
        self,
        question: str,
        session_id: str = "default"
    ) -> Dict[str, Any]:
        """Non-streaming execution of the RAG chain."""
        standalone_q, sources, context_str = self.retrieve_context(question, session_id)
        history = self.memory_manager.get_langchain_messages(session_id, limit=6)

        qa_chain = QA_PROMPT | self.llm | StrOutputParser()
        answer = qa_chain.invoke({
            "context": context_str,
            "chat_history": history,
            "question": question
        })

        # Save turn in conversational memory
        self.memory_manager.add_message(session_id, role="user", content=question)
        self.memory_manager.add_message(session_id, role="assistant", content=answer, sources=sources)

        return {
            "answer": answer,
            "sources": sources,
            "standalone_question": standalone_q
        }
