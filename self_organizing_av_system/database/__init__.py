"""
Atlas Persistent Memory Database Module

Provides embedded database backends for Atlas's self-organizing memory:
- VectorStore: ChromaDB-backed vector storage for episodes and embeddings
- GraphStore: SQLite-backed graph storage for concepts and relations
- NetworkStore: Persistence for self-organizing network weights
"""

from .vector_store import VectorStore
from .graph_store import GraphStore
from .network_store import NetworkStore

__all__ = ['VectorStore', 'GraphStore', 'NetworkStore']
