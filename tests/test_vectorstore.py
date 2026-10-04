import pytest
from langchain_core.documents import Document

def test_rrf_scoring_math():
    """
    Test Reciprocal Rank Fusion (RRF) formula:
    RRF = w / (c + rank)
    For c = 60, dense rank 1 has score 0.5 / 61 = 0.0081967
    """
    c = 60
    w = 0.5
    rank1_score = w / (c + 1)
    rank2_score = w / (c + 2)
    assert rank1_score > rank2_score
    assert round(rank1_score, 5) == 0.00820

def test_document_metadata_structure():
    doc = Document(
        page_content="Sample passage text for retrieval test.",
        metadata={"source": "test_report.pdf", "page": 3}
    )
    assert doc.page_content.startswith("Sample passage")
    assert doc.metadata["source"] == "test_report.pdf"
    assert doc.metadata["page"] == 3
