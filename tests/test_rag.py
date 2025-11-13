# tests/test_rag.py
import pytest
from pathlib import Path
from peds_post_discharge_agent.rag_retrieval import PediatricRAG


class TestRAGSystem:
    """Test suite for RAG (Retrieval Augmented Generation) system."""

    @pytest.fixture(scope="class")
    def rag_system(self):
        """Create a RAG system instance for testing."""
        # Use the actual peds-dataset directory
        project_root = Path(__file__).parent.parent
        data_dir = project_root / "peds-dataset" / "pediatric_agent_dataset"

        rag = PediatricRAG(data_dir=str(data_dir), persist_directory="./test_chroma_db")
        rag.initialize()
        yield rag

        # Cleanup is handled by pytest tmp directories
        # In production, you might want to clean up test_chroma_db

    def test_rag_initialization(self, rag_system):
        """Test that RAG system initializes successfully."""
        assert rag_system._initialized is True
        assert rag_system.vectorstore is not None

    def test_search_rsv_condition(self, rag_system):
        """Test searching for RSV/bronchiolitis information."""
        results = rag_system.search("cough after RSV", k=3)

        assert len(results) > 0
        # Check that we got relevant results
        assert any("RSV" in doc.page_content or "Bronchiolitis" in doc.page_content
                   for doc in results)

    def test_search_tonsillectomy(self, rag_system):
        """Test searching for tonsillectomy aftercare."""
        results = rag_system.search("after tonsillectomy care", k=3)

        assert len(results) > 0
        assert any("tonsillectomy" in doc.page_content.lower() for doc in results)

    def test_search_medication_info(self, rag_system):
        """Test searching for medication information."""
        results = rag_system.search("ibuprofen safety children", k=3)

        assert len(results) > 0
        assert any("Ibuprofen" in doc.page_content for doc in results)

    def test_search_with_scores(self, rag_system):
        """Test search with similarity scores."""
        results = rag_system.search_with_scores("ear infection", k=3)

        assert len(results) > 0
        # Check that results include both document and score
        for doc, score in results:
            assert hasattr(doc, 'page_content')
            assert isinstance(score, (int, float))

    def test_format_results(self, rag_system):
        """Test formatting results for LLM prompt."""
        results = rag_system.search("fever emergency", k=2)
        formatted = rag_system.format_results_for_prompt(results)

        assert isinstance(formatted, str)
        assert "Retrieved Information from Knowledge Base" in formatted
        assert len(formatted) > 0

    def test_empty_results_formatting(self, rag_system):
        """Test formatting when no results are found."""
        formatted = rag_system.format_results_for_prompt([])

        assert "No relevant information found" in formatted

    def test_metadata_in_results(self, rag_system):
        """Test that search results include proper metadata."""
        results = rag_system.search("pneumonia antibiotics", k=1)

        assert len(results) > 0
        doc = results[0]
        assert "source" in doc.metadata
        assert doc.metadata["source"] in ["pediatric_aftercare", "medication_guide"]

    def test_multiple_searches(self, rag_system):
        """Test that multiple searches work correctly."""
        queries = [
            "RSV cough",
            "fever treatment",
            "ear infection"
        ]

        for query in queries:
            results = rag_system.search(query, k=2)
            assert len(results) > 0
            assert all(hasattr(doc, 'page_content') for doc in results)
