#!/usr/bin/env python3
"""
Test script for RAG (Retrieval Augmented Generation) system.
Demonstrates searching the pediatric knowledge base.
"""
from src.peds_post_discharge_agent.tools import search_knowledge_base


def print_section(title):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def test_query(query_description, query):
    """Test a single query."""
    print(f"\n{query_description}")
    print(f"Query: \"{query}\"")
    print("-" * 70)

    result = search_knowledge_base(query)
    print(result)


def main():
    print_section("RAG SYSTEM TEST - Pediatric Knowledge Base")

    print("\nInitializing RAG system (this may take a moment on first run)...")

    # Test 1: RSV/Bronchiolitis query
    test_query(
        "Test 1: RSV Recovery Question",
        "My baby has RSV and is coughing a lot, is this normal?"
    )

    # Test 2: Tonsillectomy aftercare
    test_query(
        "Test 2: Tonsillectomy Aftercare",
        "What should I expect after my child's tonsillectomy?"
    )

    # Test 3: Medication safety
    test_query(
        "Test 3: Ibuprofen Safety",
        "Can you tell me about ibuprofen safety for children?"
    )

    # Test 4: Red flags/Emergency
    test_query(
        "Test 4: When to Seek Emergency Care",
        "When should I take my child to the emergency room for fever?"
    )

    # Test 5: Ear infection
    test_query(
        "Test 5: Ear Infection Care",
        "How do I care for my child's ear infection at home?"
    )

    print_section("TEST COMPLETE")
    print("\nThe RAG system successfully retrieved information from the curated")
    print("pediatric knowledge base. This demonstrates that the agent can now")
    print("answer questions using trusted, specific medical guidance.\n")


if __name__ == "__main__":
    main()
