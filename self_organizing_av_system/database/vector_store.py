"""
Vector Store - ChromaDB-backed vector storage for Atlas

Provides persistent storage for:
- Episodic memory vectors (experiences)
- Semantic concept embeddings
- Fast approximate nearest neighbor (ANN) search via HNSW indexing
"""

import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import json
import time

logger = logging.getLogger(__name__)

# Try to import ChromaDB, fall back to in-memory if not available
try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    logger.warning("ChromaDB not installed. Using in-memory fallback. Install with: pip install chromadb")


@dataclass
class VectorSearchResult:
    """Result from a vector similarity search."""
    id: str
    embedding: np.ndarray
    metadata: Dict[str, Any]
    distance: float
    similarity: float  # 1 - normalized_distance


class VectorStore:
    """
    ChromaDB-backed vector store for Atlas memory systems.

    Collections:
    - episodes: Episodic memory vectors with timestamps and context
    - concepts: Semantic concept embeddings with names and attributes
    """

    def __init__(self, data_dir: str = "atlas_data", embedding_dim: int = 128):
        """
        Initialize the vector store.

        Args:
            data_dir: Directory for persistent storage
            embedding_dim: Dimension of embeddings (default 128)
        """
        self.data_dir = Path(data_dir)
        self.embedding_dim = embedding_dim
        self.data_dir.mkdir(parents=True, exist_ok=True)

        if CHROMADB_AVAILABLE:
            self._init_chromadb()
        else:
            self._init_fallback()

        logger.info(f"VectorStore initialized at {self.data_dir}")

    def _init_chromadb(self):
        """Initialize ChromaDB client and collections."""
        persist_dir = str(self.data_dir / "chroma")

        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )

        # Create or get collections
        self.episodes_collection = self.client.get_or_create_collection(
            name="episodes",
            metadata={"hnsw:space": "cosine"}  # Use cosine similarity
        )

        self.concepts_collection = self.client.get_or_create_collection(
            name="concepts",
            metadata={"hnsw:space": "cosine"}
        )

        self._using_chromadb = True
        logger.info(f"ChromaDB initialized with {self.episodes_collection.count()} episodes, "
                   f"{self.concepts_collection.count()} concepts")

    def _init_fallback(self):
        """Initialize in-memory fallback storage."""
        self._episodes: Dict[str, Tuple[np.ndarray, Dict[str, Any]]] = {}
        self._concepts: Dict[str, Tuple[np.ndarray, Dict[str, Any]]] = {}
        self._using_chromadb = False
        logger.warning("Using in-memory fallback - data will not persist!")

    # ==================== Episode Operations ====================

    def add_episode(self, episode_id: str, embedding: np.ndarray,
                    metadata: Dict[str, Any]) -> bool:
        """
        Add an episode to the vector store.

        Args:
            episode_id: Unique identifier for the episode
            embedding: Vector embedding of the episode state
            metadata: Episode metadata (timestamp, context, etc.)

        Returns:
            True if successful
        """
        # Ensure embedding is the right shape
        embedding = np.asarray(embedding, dtype=np.float32).flatten()
        if len(embedding) != self.embedding_dim:
            # Pad or truncate
            if len(embedding) < self.embedding_dim:
                embedding = np.pad(embedding, (0, self.embedding_dim - len(embedding)))
            else:
                embedding = embedding[:self.embedding_dim]

        # Serialize complex metadata for ChromaDB
        serialized_meta = self._serialize_metadata(metadata)

        if self._using_chromadb:
            try:
                self.episodes_collection.upsert(
                    ids=[episode_id],
                    embeddings=[embedding.tolist()],
                    metadatas=[serialized_meta]
                )
                return True
            except Exception as e:
                logger.error(f"Failed to add episode {episode_id}: {e}")
                return False
        else:
            self._episodes[episode_id] = (embedding, metadata)
            return True

    def get_episode(self, episode_id: str) -> Optional[Tuple[np.ndarray, Dict[str, Any]]]:
        """Get an episode by ID."""
        if self._using_chromadb:
            try:
                result = self.episodes_collection.get(
                    ids=[episode_id],
                    include=["embeddings", "metadatas"]
                )
                if result['ids']:
                    embedding = np.array(result['embeddings'][0])
                    metadata = self._deserialize_metadata(result['metadatas'][0])
                    return embedding, metadata
            except Exception as e:
                logger.error(f"Failed to get episode {episode_id}: {e}")
        else:
            return self._episodes.get(episode_id)
        return None

    def search_episodes(self, query_embedding: np.ndarray, n_results: int = 10,
                       where: Optional[Dict[str, Any]] = None) -> List[VectorSearchResult]:
        """
        Search for similar episodes using ANN.

        Args:
            query_embedding: Query vector
            n_results: Number of results to return
            where: Optional metadata filter

        Returns:
            List of VectorSearchResult sorted by similarity
        """
        query = np.asarray(query_embedding, dtype=np.float32).flatten()
        if len(query) != self.embedding_dim:
            if len(query) < self.embedding_dim:
                query = np.pad(query, (0, self.embedding_dim - len(query)))
            else:
                query = query[:self.embedding_dim]

        if self._using_chromadb:
            try:
                results = self.episodes_collection.query(
                    query_embeddings=[query.tolist()],
                    n_results=min(n_results, self.episodes_collection.count()),
                    where=where,
                    include=["embeddings", "metadatas", "distances"]
                )

                search_results = []
                if results['ids'] and results['ids'][0]:
                    for i, id_ in enumerate(results['ids'][0]):
                        distance = results['distances'][0][i] if results['distances'] else 0
                        search_results.append(VectorSearchResult(
                            id=id_,
                            embedding=np.array(results['embeddings'][0][i]),
                            metadata=self._deserialize_metadata(results['metadatas'][0][i]),
                            distance=distance,
                            similarity=1.0 - min(distance, 1.0)
                        ))
                return search_results
            except Exception as e:
                logger.error(f"Episode search failed: {e}")
                return []
        else:
            # Fallback: brute force search
            return self._fallback_search(self._episodes, query, n_results)

    def delete_episode(self, episode_id: str) -> bool:
        """Delete an episode by ID."""
        if self._using_chromadb:
            try:
                self.episodes_collection.delete(ids=[episode_id])
                return True
            except Exception as e:
                logger.error(f"Failed to delete episode {episode_id}: {e}")
                return False
        else:
            if episode_id in self._episodes:
                del self._episodes[episode_id]
                return True
            return False

    def count_episodes(self) -> int:
        """Get total episode count."""
        if self._using_chromadb:
            return self.episodes_collection.count()
        return len(self._episodes)

    # ==================== Concept Operations ====================

    def add_concept(self, concept_name: str, embedding: np.ndarray,
                   metadata: Dict[str, Any]) -> bool:
        """
        Add a concept to the vector store.

        Args:
            concept_name: Unique name/identifier for the concept
            embedding: Vector embedding of the concept
            metadata: Concept metadata (attributes, activation count, etc.)

        Returns:
            True if successful
        """
        embedding = np.asarray(embedding, dtype=np.float32).flatten()
        if len(embedding) != self.embedding_dim:
            if len(embedding) < self.embedding_dim:
                embedding = np.pad(embedding, (0, self.embedding_dim - len(embedding)))
            else:
                embedding = embedding[:self.embedding_dim]

        serialized_meta = self._serialize_metadata(metadata)

        if self._using_chromadb:
            try:
                self.concepts_collection.upsert(
                    ids=[concept_name],
                    embeddings=[embedding.tolist()],
                    metadatas=[serialized_meta]
                )
                return True
            except Exception as e:
                logger.error(f"Failed to add concept {concept_name}: {e}")
                return False
        else:
            self._concepts[concept_name] = (embedding, metadata)
            return True

    def get_concept(self, concept_name: str) -> Optional[Tuple[np.ndarray, Dict[str, Any]]]:
        """Get a concept by name."""
        if self._using_chromadb:
            try:
                result = self.concepts_collection.get(
                    ids=[concept_name],
                    include=["embeddings", "metadatas"]
                )
                if result['ids']:
                    embedding = np.array(result['embeddings'][0])
                    metadata = self._deserialize_metadata(result['metadatas'][0])
                    return embedding, metadata
            except Exception as e:
                logger.error(f"Failed to get concept {concept_name}: {e}")
        else:
            return self._concepts.get(concept_name)
        return None

    def search_concepts(self, query_embedding: np.ndarray, n_results: int = 10,
                       where: Optional[Dict[str, Any]] = None) -> List[VectorSearchResult]:
        """
        Search for similar concepts using ANN.

        Args:
            query_embedding: Query vector
            n_results: Number of results to return
            where: Optional metadata filter

        Returns:
            List of VectorSearchResult sorted by similarity
        """
        query = np.asarray(query_embedding, dtype=np.float32).flatten()
        if len(query) != self.embedding_dim:
            if len(query) < self.embedding_dim:
                query = np.pad(query, (0, self.embedding_dim - len(query)))
            else:
                query = query[:self.embedding_dim]

        if self._using_chromadb:
            try:
                results = self.concepts_collection.query(
                    query_embeddings=[query.tolist()],
                    n_results=min(n_results, max(1, self.concepts_collection.count())),
                    where=where,
                    include=["embeddings", "metadatas", "distances"]
                )

                search_results = []
                if results['ids'] and results['ids'][0]:
                    for i, id_ in enumerate(results['ids'][0]):
                        distance = results['distances'][0][i] if results['distances'] else 0
                        search_results.append(VectorSearchResult(
                            id=id_,
                            embedding=np.array(results['embeddings'][0][i]),
                            metadata=self._deserialize_metadata(results['metadatas'][0][i]),
                            distance=distance,
                            similarity=1.0 - min(distance, 1.0)
                        ))
                return search_results
            except Exception as e:
                logger.error(f"Concept search failed: {e}")
                return []
        else:
            return self._fallback_search(self._concepts, query, n_results)

    def delete_concept(self, concept_name: str) -> bool:
        """Delete a concept by name."""
        if self._using_chromadb:
            try:
                self.concepts_collection.delete(ids=[concept_name])
                return True
            except Exception as e:
                logger.error(f"Failed to delete concept {concept_name}: {e}")
                return False
        else:
            if concept_name in self._concepts:
                del self._concepts[concept_name]
                return True
            return False

    def count_concepts(self) -> int:
        """Get total concept count."""
        if self._using_chromadb:
            return self.concepts_collection.count()
        return len(self._concepts)

    def get_all_concept_names(self) -> List[str]:
        """Get all concept names."""
        if self._using_chromadb:
            try:
                result = self.concepts_collection.get(include=[])
                return result['ids'] if result['ids'] else []
            except Exception as e:
                logger.error(f"Failed to get concept names: {e}")
                return []
        else:
            return list(self._concepts.keys())

    # ==================== Utility Methods ====================

    def _serialize_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Serialize metadata for ChromaDB (only supports str, int, float, bool)."""
        serialized = {}
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)):
                serialized[key] = value
            # Handle numpy scalar types (np.bool_, np.int64, np.float64, etc.)
            elif isinstance(value, (np.bool_, np.integer, np.floating)):
                serialized[key] = value.item()  # Convert to Python native type
            elif isinstance(value, np.ndarray):
                serialized[f"{key}_json"] = json.dumps(value.tolist())
            elif isinstance(value, (list, dict)):
                # Recursively convert numpy types in nested structures
                serialized[f"{key}_json"] = json.dumps(value, default=self._json_default)
            elif value is None:
                serialized[key] = ""
            else:
                serialized[f"{key}_json"] = json.dumps(str(value))
        return serialized

    def _json_default(self, obj):
        """JSON encoder for numpy types."""
        if isinstance(obj, (np.bool_, np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

    def _deserialize_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Deserialize metadata from ChromaDB."""
        deserialized = {}
        for key, value in metadata.items():
            if key.endswith("_json"):
                original_key = key[:-5]
                try:
                    deserialized[original_key] = json.loads(value)
                except (json.JSONDecodeError, TypeError):
                    deserialized[original_key] = value
            else:
                deserialized[key] = value
        return deserialized

    def _fallback_search(self, storage: Dict[str, Tuple[np.ndarray, Dict[str, Any]]],
                        query: np.ndarray, n_results: int) -> List[VectorSearchResult]:
        """Brute force search for fallback mode."""
        if not storage:
            return []

        results = []
        for id_, (embedding, metadata) in storage.items():
            # Cosine similarity
            norm_q = np.linalg.norm(query)
            norm_e = np.linalg.norm(embedding)
            if norm_q > 0 and norm_e > 0:
                similarity = np.dot(query, embedding) / (norm_q * norm_e)
            else:
                similarity = 0.0

            results.append(VectorSearchResult(
                id=id_,
                embedding=embedding,
                metadata=metadata,
                distance=1.0 - similarity,
                similarity=float(similarity)
            ))

        results.sort(key=lambda x: x.similarity, reverse=True)
        return results[:n_results]

    def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        return {
            'using_chromadb': self._using_chromadb,
            'data_dir': str(self.data_dir),
            'embedding_dim': self.embedding_dim,
            'episode_count': self.count_episodes(),
            'concept_count': self.count_concepts(),
        }

    def close(self):
        """Close the vector store."""
        if self._using_chromadb and hasattr(self, 'client'):
            # ChromaDB PersistentClient auto-persists, no explicit close needed
            pass
        logger.info("VectorStore closed")
