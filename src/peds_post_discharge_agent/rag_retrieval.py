# src/peds_post_discharge_agent/rag_retrieval.py
"""
RAG (Retrieval Augmented Generation) system for pediatric aftercare knowledge base.
Uses ChromaDB vector database to retrieve relevant information from curated datasets.
"""
import json
import logging
import os
from pathlib import Path
from typing import List, Dict
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from chromadb.utils import embedding_functions

# Configure logging
logger = logging.getLogger(__name__)


class ChromaEmbeddingAdapter(Embeddings):
    """Adapter to make ChromaDB embedding function compatible with LangChain."""

    def __init__(self):
        self.ef = embedding_functions.DefaultEmbeddingFunction()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents."""
        return self.ef(texts)

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query."""
        return self.ef([text])[0]


class PediatricRAG:
    """RAG system for retrieving pediatric aftercare information."""

    def __init__(self, data_dir: str = None, persist_directory: str = "./chroma_db"):
        """
        Initialize the RAG system.

        Args:
            data_dir: Path to the pediatric dataset directory
            persist_directory: Where to store the vector database
        """
        if data_dir is None:
            # Default to peds-dataset folder
            project_root = Path(__file__).parent.parent.parent
            data_dir = project_root / "peds-dataset" / "pediatric_agent_dataset"

        self.data_dir = Path(data_dir)
        self.persist_directory = persist_directory
        self.vectorstore = None
        self._initialized = False

    def _load_jsonl(self, filepath: Path) -> List[Dict]:
        """Load JSONL file and return list of dictionaries."""
        documents = []
        if not filepath.exists():
            print(f"Warning: {filepath} not found")
            return documents

        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        documents.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        print(f"Error parsing line in {filepath}: {e}")
        return documents

    def _create_documents_from_aftercare(self, data: List[Dict]) -> List[Document]:
        """Convert aftercare JSONL data into LangChain Documents."""
        documents = []

        for item in data:
            # Create a comprehensive text representation
            content_parts = [
                f"Condition: {item.get('title', 'Unknown')}",
                f"Age Range: {item.get('age_range', 'Not specified')} years",
                "",
                "Normal Symptoms:",
            ]

            for symptom in item.get('normal_symptoms', []):
                content_parts.append(f"- {symptom}")

            content_parts.append("\nCare Tips:")
            for tip in item.get('care_tips', []):
                content_parts.append(f"- {tip}")

            content_parts.append("\nRed Flags (Seek Immediate Care):")
            for flag in item.get('red_flags', []):
                content_parts.append(f"- {flag}")

            content = "\n".join(content_parts)

            # Create document with metadata
            doc = Document(
                page_content=content,
                metadata={
                    "source": "pediatric_aftercare",
                    "id": item.get('id', 'unknown'),
                    "condition": item.get('condition', 'unknown'),
                    "title": item.get('title', 'Unknown'),
                    "age_range": item.get('age_range', 'Not specified')
                }
            )
            documents.append(doc)

        return documents

    def _create_documents_from_medications(self, data: List[Dict]) -> List[Document]:
        """Convert medication guides JSONL data into LangChain Documents."""
        documents = []

        for item in data:
            content_parts = [
                f"Medication: {item.get('drug', 'Unknown')}",
                f"Forms: {', '.join(item.get('forms', []))}",
                f"Use: {item.get('use', 'Not specified')}",
                f"Safety: {item.get('safety', 'Not specified')}",
                f"Storage: {item.get('storage', 'Not specified')}",
                f"Notes: {item.get('notes', '')}"
            ]

            content = "\n".join(content_parts)

            doc = Document(
                page_content=content,
                metadata={
                    "source": "medication_guide",
                    "drug": item.get('drug', 'unknown'),
                    "forms": ', '.join(item.get('forms', []))  # Convert list to string
                }
            )
            documents.append(doc)

        return documents

    def initialize(self):
        """Load data and initialize vector database."""
        if self._initialized:
            return

        try:
            logger.info("Initializing RAG system...")
            print("Initializing RAG system...")

            # Load data files
            aftercare_data = self._load_jsonl(self.data_dir / "pediatric_aftercare.jsonl")
            medication_data = self._load_jsonl(self.data_dir / "medication_guides.jsonl")

            # Convert to documents
            documents = []
            documents.extend(self._create_documents_from_aftercare(aftercare_data))
            documents.extend(self._create_documents_from_medications(medication_data))

            logger.info(f"Loaded {len(documents)} documents for RAG")
            print(f"Loaded {len(documents)} documents for RAG")

            # Create embeddings using our adapter for ChromaDB's default embedding
            embeddings = ChromaEmbeddingAdapter()

            # Create or load vector store
            self.vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=embeddings,
                persist_directory=self.persist_directory,
                collection_name="pediatric_knowledge"
            )

            self._initialized = True
            logger.info("RAG system initialized successfully!")
            print("RAG system initialized successfully!")

        except Exception as e:
            logger.error(f"Failed to initialize RAG system: {str(e)}", exc_info=True)
            raise RuntimeError(f"RAG initialization failed: {str(e)}") from e

    def search(self, query: str, k: int = 3) -> List[Document]:
        """
        Search for relevant documents.

        Args:
            query: The search query
            k: Number of top results to return

        Returns:
            List of relevant documents
        """
        if not self._initialized:
            self.initialize()

        results = self.vectorstore.similarity_search(query, k=k)
        return results

    def search_with_scores(self, query: str, k: int = 3) -> List[tuple]:
        """
        Search for relevant documents with similarity scores.

        Args:
            query: The search query
            k: Number of top results to return

        Returns:
            List of (document, score) tuples
        """
        if not self._initialized:
            self.initialize()

        results = self.vectorstore.similarity_search_with_score(query, k=k)
        return results

    def format_results_for_prompt(self, documents: List[Document]) -> str:
        """Format retrieved documents for inclusion in LLM prompt."""
        if not documents:
            return "No relevant information found in knowledge base."

        formatted = ["Retrieved Information from Knowledge Base:", ""]

        for i, doc in enumerate(documents, 1):
            formatted.append(f"[{i}] {doc.metadata.get('title', doc.metadata.get('drug', 'Document'))}")
            formatted.append(doc.page_content)
            formatted.append("")

        return "\n".join(formatted)


# Global singleton instance
_rag_instance = None


def get_rag_system() -> PediatricRAG:
    """Get the global RAG system instance."""
    global _rag_instance
    if _rag_instance is None:
        _rag_instance = PediatricRAG()
        # Lazy initialization - will initialize on first search
    return _rag_instance
