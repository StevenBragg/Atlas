"""
Unified Knowledge Base for ATLAS

Provides a unified facade for Atlas's complete memory system, combining:
- Episodic Memory: Experiences, events, and autobiographical memories
- Semantic Memory: Concepts, relationships, and inference rules

This knowledge base is SHARED between Curriculum Learning and Free Play areas,
allowing transfer of learned knowledge between structured and exploratory learning.

Now with optional persistent storage via ChromaDB + SQLite for lifelong learning.
"""

import numpy as np
import logging
import time
import threading
from typing import List, Dict, Optional, Any, Tuple, TYPE_CHECKING
from dataclasses import dataclass, field
from collections import deque

from .backend import xp, to_cpu, to_gpu
from .episodic_memory import EpisodicMemory, Episode
from .semantic_memory import SemanticMemory, Concept, RelationType

if TYPE_CHECKING:
    from ..database.vector_store import VectorStore
    from ..database.graph_store import GraphStore

logger = logging.getLogger(__name__)


@dataclass
class KnowledgeEvent:
    """A knowledge event that can be displayed in the UI."""
    timestamp: float
    event_type: str  # "episode_stored", "concept_learned", "relation_created", "inference_made"
    description: str
    source: str  # "curriculum" or "free_play"
    consolidation_strength: float = 0.0


@dataclass
class MemoryQuery:
    """A query to the knowledge base."""
    query_state: np.ndarray
    context: Optional[str] = None
    source: Optional[str] = None  # Filter by source
    k_episodes: int = 5
    k_concepts: int = 10


@dataclass
class MemoryQueryResult:
    """Result of a knowledge base query."""
    episodes: List[Episode]
    concepts: List[Tuple[str, float]]  # (concept_name, activation)
    inferences: List[str]
    related_concepts: Dict[str, List[str]]  # concept -> related concepts


class KnowledgeBase:
    """
    Unified facade for Atlas's complete memory system.

    Combines episodic memory (experiences) and semantic memory (concepts/relations)
    into a single interface. Knowledge is shared between Curriculum and Free Play,
    enabling transfer learning.

    Features:
    - Store experiences from both learning areas
    - Extract and consolidate concepts automatically
    - Query memories by similarity or context
    - Periodic consolidation in background thread
    - Track recent learning events for UI display
    """

    def __init__(
        self,
        state_dim: int = 128,
        max_episodes: int = 50000,
        max_concepts: int = 100000,
        enable_consolidation: bool = True,
        consolidation_interval: float = 60.0,  # seconds
        vector_store: Optional['VectorStore'] = None,
        graph_store: Optional['GraphStore'] = None,
    ):
        """
        Initialize the unified knowledge base.

        Args:
            state_dim: Dimension of neural state vectors
            max_episodes: Maximum episodic memories to store
            max_concepts: Maximum semantic concepts to store
            enable_consolidation: Whether to run background consolidation
            consolidation_interval: Seconds between consolidation runs
            vector_store: Optional VectorStore for persistent vector storage
            graph_store: Optional GraphStore for persistent graph storage
        """
        self.state_dim = state_dim

        # Store references to persistent stores
        self.vector_store = vector_store
        self.graph_store = graph_store
        self._persistence_enabled = vector_store is not None or graph_store is not None

        # Initialize episodic memory with vector store
        self.episodic = EpisodicMemory(
            state_size=state_dim,
            max_episodes=max_episodes,
            enable_consolidation=True,
            enable_forgetting=True,
            forgetting_rate=0.0001,
            novelty_bonus=2.0,
            emotional_bonus=1.5,
            vector_store=vector_store,
        )

        # Initialize semantic memory with both stores
        self.semantic = SemanticMemory(
            embedding_size=state_dim,
            max_concepts=max_concepts,
            enable_inference=True,
            enable_generalization=True,
            vector_store=vector_store,
            graph_store=graph_store,
        )

        # Recent events for UI display
        self.recent_events: deque = deque(maxlen=100)

        # Statistics
        self.total_experiences_stored = 0
        self.total_concepts_extracted = 0
        self.total_queries = 0

        # Background consolidation
        self.enable_consolidation = enable_consolidation
        self.consolidation_interval = consolidation_interval
        self._consolidation_thread: Optional[threading.Thread] = None
        self._stop_consolidation = threading.Event()

        if enable_consolidation:
            self._start_consolidation_thread()

        logger.info(f"KnowledgeBase initialized: state_dim={state_dim}, "
                   f"max_episodes={max_episodes}, max_concepts={max_concepts}, "
                   f"persistence={self._persistence_enabled}")

    def _start_consolidation_thread(self):
        """Start background consolidation thread."""
        self._stop_consolidation.clear()
        self._consolidation_thread = threading.Thread(
            target=self._consolidation_loop,
            daemon=True,
            name="KnowledgeBase-Consolidation"
        )
        self._consolidation_thread.start()
        logger.info("Started background consolidation thread")

    def _consolidation_loop(self):
        """Background loop for memory consolidation."""
        while not self._stop_consolidation.is_set():
            try:
                self.consolidate()
            except Exception as e:
                logger.error(f"Error during consolidation: {e}")

            # Sleep with interruptible wait
            self._stop_consolidation.wait(self.consolidation_interval)

    def stop(self):
        """Stop the knowledge base (stops background threads)."""
        self._stop_consolidation.set()
        if self._consolidation_thread is not None:
            self._consolidation_thread.join(timeout=5.0)
        logger.info("KnowledgeBase stopped")

    def store_experience(
        self,
        state: np.ndarray,
        context: Dict[str, Any],
        sensory_data: Optional[Dict[str, np.ndarray]] = None,
        emotional_valence: float = 0.0,
        source: str = "free_play",
    ) -> None:
        """
        Store an experience from either Curriculum or Free Play.

        Args:
            state: Neural state at time of experience
            context: Contextual information (challenge name, modalities, etc.)
            sensory_data: Raw sensory inputs (vision, audio, etc.)
            emotional_valence: Emotional significance (-1 to 1)
            source: "curriculum" or "free_play"
        """
        # Add source to context
        context["source"] = source
        context["timestamp"] = time.time()

        # Compute novelty/surprise
        surprise_level = self._compute_novelty(state)

        # Store in episodic memory (pass individual arguments)
        self.episodic.store(
            state=state,
            sensory_data=sensory_data or {},
            context=context,
            emotional_valence=emotional_valence,
            surprise_level=surprise_level,
        )
        self.total_experiences_stored += 1

        # Extract concepts for semantic memory
        concepts = self._extract_concepts(state, context)
        for concept_name, embedding in concepts:
            self.semantic.add_concept(concept_name, embedding)
            self.total_concepts_extracted += 1

        # Record event for UI
        description = context.get("description", context.get("challenge_name", "Experience stored"))
        self._record_event("episode_stored", description, source, surprise_level)

        logger.debug(f"Stored experience from {source}: {description}")

    def _compute_novelty(self, state: np.ndarray) -> float:
        """Compute how novel/surprising a state is."""
        if len(self.episodic.episodes) == 0:
            return 1.0  # First experience is always novel

        # Compare to recent episodes
        recent_states = [ep.state for ep in list(self.episodic.episodes)[-100:]]
        if not recent_states:
            return 1.0

        # Compute average similarity to recent states
        similarities = []
        for recent_state in recent_states:
            sim = np.dot(state, recent_state) / (np.linalg.norm(state) * np.linalg.norm(recent_state) + 1e-8)
            similarities.append(sim)

        avg_similarity = np.mean(similarities)
        novelty = 1.0 - max(0, min(1, avg_similarity))

        return novelty

    def _extract_concepts(
        self,
        state: np.ndarray,
        context: Dict[str, Any]
    ) -> List[Tuple[str, np.ndarray]]:
        """Extract concepts from state and context for semantic memory."""
        concepts = []

        # Extract from context
        if "challenge_name" in context:
            # Create concept for challenge type
            concept_name = f"challenge:{context['challenge_name'].replace(' ', '_')}"
            concepts.append((concept_name, state.copy()))

        if "modalities" in context:
            # Create concepts for modalities
            for modality in context["modalities"]:
                mod_name = modality.name if hasattr(modality, 'name') else str(modality)
                concept_name = f"modality:{mod_name}"
                # Create a derived embedding
                mod_embedding = state * (hash(mod_name) % 100 / 100.0)
                concepts.append((concept_name, mod_embedding))

        if "accuracy" in context:
            # Create concept for performance level
            accuracy = context["accuracy"]
            if accuracy >= 0.9:
                concepts.append(("performance:excellent", state.copy()))
            elif accuracy >= 0.7:
                concepts.append(("performance:good", state.copy()))
            elif accuracy >= 0.5:
                concepts.append(("performance:moderate", state.copy()))
            else:
                concepts.append(("performance:needs_improvement", state.copy()))

        # Always add a general experience concept
        concepts.append((f"experience:{int(time.time())}", state.copy()))

        return concepts

    def _record_event(
        self,
        event_type: str,
        description: str,
        source: str,
        consolidation_strength: float = 0.0
    ):
        """Record an event for UI display."""
        event = KnowledgeEvent(
            timestamp=time.time(),
            event_type=event_type,
            description=description,
            source=source,
            consolidation_strength=consolidation_strength,
        )
        self.recent_events.append(event)

    def query(self, query: MemoryQuery) -> MemoryQueryResult:
        """
        Query both memory systems.

        Args:
            query: MemoryQuery with search parameters

        Returns:
            MemoryQueryResult with episodes, concepts, and inferences
        """
        self.total_queries += 1

        # Query episodic memory
        episodes = self.episodic.retrieve(query.query_state, k=query.k_episodes)

        # Filter by source if specified
        if query.source:
            episodes = [ep for ep in episodes if ep.context.get("source") == query.source]

        # Query semantic memory via spreading activation
        concept_activations = self.semantic.spreading_activation(query.query_state)

        # Get top concepts
        sorted_concepts = sorted(concept_activations.items(), key=lambda x: x[1], reverse=True)
        top_concepts = sorted_concepts[:query.k_concepts]

        # Get related concepts for each top concept
        related_concepts = {}
        for concept_name, _ in top_concepts:
            neighbors = self.semantic.get_neighbors(concept_name)
            related_concepts[concept_name] = neighbors

        # Attempt inferences
        inferences = []
        if self.semantic.enable_inference:
            inferences = self.semantic.infer_from_state(query.query_state)

        return MemoryQueryResult(
            episodes=episodes,
            concepts=top_concepts,
            inferences=inferences,
            related_concepts=related_concepts,
        )

    def consolidate(self) -> Dict[str, int]:
        """
        Run memory consolidation.

        This strengthens important memories and prunes weak ones.
        Should be called periodically (runs automatically in background if enabled).

        Returns:
            Statistics about consolidation
        """
        stats = {
            "episodes_consolidated": 0,
            "episodes_forgotten": 0,
            "concepts_strengthened": 0,
            "relations_strengthened": 0,
        }

        try:
            # Consolidate episodic memories
            if hasattr(self.episodic, 'consolidate_memories'):
                self.episodic.consolidate_memories()
                stats["episodes_consolidated"] = len(self.episodic.consolidated_episodes)

            # Consolidate semantic relations
            if hasattr(self.semantic, 'consolidate_relations'):
                self.semantic.consolidate_relations()
                stats["relations_strengthened"] = self.semantic.concept_graph.number_of_edges()

            logger.debug(f"Consolidation complete: {stats}")

        except Exception as e:
            logger.error(f"Error during consolidation: {e}")

        return stats

    def add_relation(
        self,
        source_concept: str,
        target_concept: str,
        relation_type: RelationType,
        strength: float = 1.0,
    ) -> None:
        """Add a relation between concepts in semantic memory."""
        self.semantic.add_relation(source_concept, target_concept, relation_type, strength)
        self._record_event(
            "relation_created",
            f"{source_concept} --{relation_type.value}--> {target_concept}",
            "system",
            strength
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get combined memory statistics."""
        episodic_stats = self.episodic.get_stats()
        semantic_stats = self.semantic.get_stats()

        return {
            # Episodic memory stats
            "total_episodes": episodic_stats.get("total_stored", 0),
            "consolidated_episodes": episodic_stats.get("consolidated_episodes", 0),
            "total_retrieved": episodic_stats.get("total_retrieved", 0),
            "total_forgotten": episodic_stats.get("total_forgotten", 0),

            # Semantic memory stats
            "total_concepts": semantic_stats.get("total_concepts", 0),
            "total_relations": semantic_stats.get("total_relations", 0),
            "total_inferences": semantic_stats.get("total_inferences_made", 0),
            "total_generalizations": semantic_stats.get("total_generalizations", 0),

            # Combined stats
            "total_experiences_stored": self.total_experiences_stored,
            "total_concepts_extracted": self.total_concepts_extracted,
            "total_queries": self.total_queries,

            # Recent events count
            "recent_events_count": len(self.recent_events),
        }

    def get_recent_events(self, n: int = 10) -> List[KnowledgeEvent]:
        """Get the n most recent knowledge events."""
        return list(self.recent_events)[-n:]

    def get_recent_event_string(self) -> str:
        """Get a formatted string of the most recent event."""
        if not self.recent_events:
            return "No recent events"

        event = self.recent_events[-1]
        return f'"{event.description}" (consolidation: {event.consolidation_strength:.0%})'

    def serialize(self) -> Dict[str, Any]:
        """Serialize the knowledge base for saving."""
        return {
            "state_dim": self.state_dim,
            "total_experiences_stored": self.total_experiences_stored,
            "total_concepts_extracted": self.total_concepts_extracted,
            "total_queries": self.total_queries,
            # Note: Full serialization of episodic/semantic memories
            # would be done through their own serialize methods
        }

    @classmethod
    def deserialize(
        cls,
        data: Dict[str, Any],
        vector_store: Optional['VectorStore'] = None,
        graph_store: Optional['GraphStore'] = None,
    ) -> 'KnowledgeBase':
        """Deserialize a knowledge base from saved data."""
        kb = cls(
            state_dim=data.get("state_dim", 128),
            vector_store=vector_store,
            graph_store=graph_store,
        )
        kb.total_experiences_stored = data.get("total_experiences_stored", 0)
        kb.total_concepts_extracted = data.get("total_concepts_extracted", 0)
        kb.total_queries = data.get("total_queries", 0)
        return kb

    # ==================== Persistence Methods ====================

    def set_stores(
        self,
        vector_store: Optional['VectorStore'] = None,
        graph_store: Optional['GraphStore'] = None,
    ) -> None:
        """
        Set persistent stores for the knowledge base.

        This enables persistence after the knowledge base has been created.
        Existing data will be synced to the stores.

        Args:
            vector_store: VectorStore for episode/concept embeddings
            graph_store: GraphStore for concept relationships
        """
        self.vector_store = vector_store
        self.graph_store = graph_store
        self._persistence_enabled = vector_store is not None or graph_store is not None

        # Update sub-memories
        if vector_store:
            self.episodic.set_vector_store(vector_store)

        if vector_store or graph_store:
            self.semantic.set_stores(vector_store, graph_store)

        logger.info(f"KnowledgeBase stores set: vector={vector_store is not None}, "
                   f"graph={graph_store is not None}")

    def sync_to_stores(self) -> Dict[str, int]:
        """
        Sync all in-memory data to persistent stores.

        Returns:
            Dictionary with counts of synced items
        """
        result = {
            'episodes_synced': 0,
            'concepts_synced': 0,
            'relations_synced': 0,
        }

        if not self._persistence_enabled:
            logger.warning("Persistence not enabled, nothing to sync")
            return result

        # Sync episodic memory
        if self.vector_store:
            result['episodes_synced'] = self.episodic.sync_consolidated_to_store()

        # Sync semantic memory
        concepts, relations = self.semantic.sync_to_stores()
        result['concepts_synced'] = concepts
        result['relations_synced'] = relations

        logger.info(f"Sync complete: {result}")
        return result

    def close(self) -> None:
        """
        Close the knowledge base and all underlying stores.

        This should be called when shutting down to ensure data is persisted.
        """
        # Stop consolidation thread
        self.stop()

        # Final sync to stores
        if self._persistence_enabled:
            self.sync_to_stores()

        # Close stores
        if self.vector_store:
            self.vector_store.close()

        if self.graph_store:
            self.graph_store.close()

        logger.info("KnowledgeBase closed")
