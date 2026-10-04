"""
Query Transformation and Adaptive Routing Module for Phase 4.

Provides:
1. HyDE (Hypothetical Document Embeddings): Generates a synthetic document passage to bridge vocabulary mismatch.
2. Multi-Query Expansion: Deconstructs compound/multi-hop questions into focused sub-queries.
3. Step-Back Prompting: Formulates broader foundational questions to capture macro context.
4. Adaptive Semantic Router: Classifies user intent and routes queries to the optimal pipeline (including Direct LLM bypass).
"""

import re
import json
import logging
from typing import List, Dict, Any, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.language_models.chat_models import BaseChatModel

logger = logging.getLogger(__name__)


HYDE_SYSTEM_PROMPT = """You are an expert document synthesis engine for a human rights and legal knowledge base.
Given a user query, write a hypothetical, highly informative passage that directly answers the question as if it were an authoritative excerpt from a formal human rights report or legal analysis.

CRITICAL INSTRUCTIONS:
1. Use formal, domain-specific terminology (e.g., specific treaty articles, legal standards, reporting organizations).
2. Write 1-2 dense, factual paragraphs (approx 100-150 words).
3. Do NOT include any introductory or concluding conversational filler (e.g., do NOT write "Here is a document...", "In this passage...").
4. Output ONLY the hypothetical document content."""

HYDE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", HYDE_SYSTEM_PROMPT),
    ("human", "{question}"),
])


MULTI_QUERY_SYSTEM_PROMPT = """You are an AI search query optimization specialist.
Your task is to decompose a complex or multi-part user question into {count} distinct, concise, and focused sub-queries from different perspectives.
These sub-queries will be executed in parallel against a document index to gather all necessary facts.

CRITICAL INSTRUCTIONS:
1. Output EXACTLY {count} sub-queries.
2. One sub-query per line.
3. Do NOT include numbers (e.g., '1.', '2.'), bullet points, quotes, or preambles.
4. Each sub-query must be self-contained and clear."""

MULTI_QUERY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", MULTI_QUERY_SYSTEM_PROMPT),
    ("human", "{question}"),
])


STEP_BACK_SYSTEM_PROMPT = """You are an expert in step-back prompting for deep reasoning.
Given a specific question, your task is to step back and generate a broader, higher-level, foundational question about the overarching concepts, principles, or historical legal context.

Examples:
Question: Which private companies in the Americas are the largest GHG emitters according to the Carbon Majors database?
Step-Back Question: What is the Carbon Majors database and how does it categorize greenhouse gas emissions from fossil fuel producers?

Question: Why did the criminalization of debunking official info in Russia cause human rights violations?
Step-Back Question: What international human rights laws and standards protect freedom of expression and independent journalism during armed conflict?

CRITICAL INSTRUCTIONS:
1. Output ONLY the single step-back question string.
2. Do NOT include any quotation marks, explanations, or conversational filler."""

STEP_BACK_PROMPT = ChatPromptTemplate.from_messages([
    ("system", STEP_BACK_SYSTEM_PROMPT),
    ("human", "{question}"),
])


ROUTER_SYSTEM_PROMPT = """You are a high-speed query routing classifier for an enterprise RAG knowledge base.
Analyze the user's question and classify it into EXACTLY ONE of the following categories:

- DIRECT: Conversational chitchat, greetings, polite remarks, or meta questions about who you are that require NO document retrieval.
  Examples: "hello", "hi there", "who are you?", "thank you", "can you help me?"

- MULTI_HOP: Compound, multi-part, or comparative questions that ask about multiple entities, companies, or events simultaneously.
  Examples: "Compare private vs state-owned emitters in the Americas and how climate disasters impacted the global South."

- CONCEPTUAL: High-level questions asking about broad principles, international legal standards, or overarching trends.
  Examples: "What international standards govern freedom of expression during wartime?"

- HYDE: Queries with high vocabulary mismatch, rare legal terminology, or questions where a hypothetical document structure helps bridge the gap.
  Examples: "How does the Rome Statute handle sudden state withdrawal?", "What are the legal implications of Article 207.3?"

- FACT_LOOKUP: Standard specific factual queries asking about an event, date, ruling, organization, or policy.
  Examples: "When did Qatar repeal migrant worker restrictions?", "What was Amnesty's response to the Ogoni 9 executions?"

You must respond in strict valid JSON format:
{{
  "route": "DIRECT" | "MULTI_HOP" | "CONCEPTUAL" | "HYDE" | "FACT_LOOKUP",
  "reasoning": "A concise 1-sentence explanation of why this route was selected."
}}"""

ROUTER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", ROUTER_SYSTEM_PROMPT),
    ("human", "{question}"),
])


class QueryTransformer:
    """
    Manages Query Transformations (HyDE, Multi-Query, Step-Back) and Adaptive Routing.
    """

    def __init__(self, llm: BaseChatModel):
        self.llm = llm
        self.hyde_chain = HYDE_PROMPT | self.llm | StrOutputParser()
        self.multi_query_chain = MULTI_QUERY_PROMPT | self.llm | StrOutputParser()
        self.step_back_chain = STEP_BACK_PROMPT | self.llm | StrOutputParser()
        self.router_chain = ROUTER_PROMPT | self.llm | StrOutputParser()

    def generate_hyde(self, query: str) -> str:
        """Generates a hypothetical document snippet answering the query."""
        try:
            raw = self.hyde_chain.invoke({"question": query})
            cleaned = raw.strip().strip('"').strip("'")
            # Strip common introductory phrases
            cleaned = re.sub(r'^(Here is a hypothetical.*?:|In this document.*?:|Hypothetical passage:)\s*', '', cleaned, flags=re.IGNORECASE).strip()
            if len(cleaned.split()) >= 10:
                return cleaned
            return query
        except Exception as e:
            logger.warning(f"HyDE generation failed ({e}), falling back to raw query.")
            return query

    def generate_multi_query(self, query: str, count: int = 3) -> List[str]:
        """Deconstructs complex queries into sub-queries."""
        try:
            raw = self.multi_query_chain.invoke({"question": query, "count": count})
            lines = [line.strip() for line in raw.splitlines() if line.strip()]
            cleaned_queries: List[str] = []

            for line in lines:
                # Strip leading numbering like '1.', '2.', '-', '*'
                q_clean = re.sub(r'^\s*(\d+[\.\)]|\-|\*)\s*', '', line).strip().strip('"').strip("'")
                if len(q_clean.split()) >= 3 and q_clean.lower() != query.lower():
                    cleaned_queries.append(q_clean)

            # Deduplicate while preserving order
            unique_queries = list(dict.fromkeys(cleaned_queries))

            # Always ensure original query is present
            if query not in unique_queries:
                unique_queries.insert(0, query)

            return unique_queries[:count + 1]
        except Exception as e:
            logger.warning(f"Multi-query generation failed ({e}), falling back to raw query.")
            return [query]

    def generate_step_back(self, query: str) -> str:
        """Generates an abstract, high-level concept question."""
        try:
            raw = self.step_back_chain.invoke({"question": query})
            cleaned = raw.strip().strip('"').strip("'")
            # Remove prefixes like 'Step-Back Question:'
            cleaned = re.sub(r'^(Step-Back Question:\s*|High-level Question:\s*)', '', cleaned, flags=re.IGNORECASE).strip()
            if len(cleaned.split()) >= 3:
                return cleaned
            return query
        except Exception as e:
            logger.warning(f"Step-back generation failed ({e}), falling back to raw query.")
            return query

    def route_query(self, query: str) -> Dict[str, Any]:
        """
        Classifies query intent and returns the optimal execution strategy.
        """
        # Fast heuristic checks for obvious chitchat to avoid LLM call
        lower = query.strip().lower()
        chitchat_set = {"hi", "hello", "hey", "good morning", "good evening", "how are you", "who are you", "thank you", "thanks"}
        if lower in chitchat_set or len(lower.split()) <= 2 and any(w in lower for w in ["hi", "hello", "hey", "thanks"]):
            return {
                "route": "DIRECT",
                "strategy": "direct",
                "reasoning": "Query is a direct greeting/chitchat requiring no external corpus retrieval.",
                "confidence": 1.0
            }

        try:
            raw = self.router_chain.invoke({"question": query})
            # Extract JSON block
            json_match = re.search(r'\{.*\}', raw, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group(0))
                route = parsed.get("route", "FACT_LOOKUP").upper()
                reasoning = parsed.get("reasoning", "Classified by semantic router.")
            else:
                route = "FACT_LOOKUP"
                reasoning = "Fallback to standard retrieval."

            # Map route to strategy
            route_map = {
                "DIRECT": "direct",
                "MULTI_HOP": "multi_query",
                "CONCEPTUAL": "step_back",
                "HYDE": "hyde",
                "FACT_LOOKUP": "standard"
            }
            strategy = route_map.get(route, "standard")

            return {
                "route": route,
                "strategy": strategy,
                "reasoning": reasoning,
                "confidence": 0.95
            }
        except Exception as e:
            logger.warning(f"Query routing failed ({e}), defaulting to standard FACT_LOOKUP.")
            return {
                "route": "FACT_LOOKUP",
                "strategy": "standard",
                "reasoning": f"Routing fallback triggered ({e}).",
                "confidence": 0.5
            }
