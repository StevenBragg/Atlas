"""
Episodic Memory System for ATLAS

Implements a hippocampal-inspired episodic memory system that stores,
consolidates, and retrieves specific experiences and events.

This is a critical component for superintelligence development, enabling:
- Long-term storage of experiences
- Context-based retrieval
- Memory consolidation and replay
- One-shot learning from significant events

Now with optional ChromaDB-backed persistent storage for lifelong learning.
"""

import numpy as np
import logging
import uuid
from typing import List, Dict, Tuple, Optional, Any, TYPE_CHECKING
from dataclasses import dataclass
from collections import deque
import time

if TYPE_CHECKING:
    from ..database.vector_store import VectorStore

logger = logging.getLogger(__name__)


@dataclass
class Episode:
    """Represents a single episodic memory"""
    timestamp: float
    state: np.ndarray  # Neural state at time of encoding
    context: Dict[str, Any]  # Contextual information
    sensory_data: Dict[str, np.ndarray]  # Multimodal sensory inputs
    emotional_valence: float  # Emotional significance (-1 to 1)
    surprise_level: float  # How surprising/novel the episode was
    replay_count: int = 0  # Number of times replayed
    consolidation_strength: float = 0.0  # Strength of consolidation
    episode_id: str = ""  # Unique identifier for persistence

    def __post_init__(self):
        if not self.episode_id:
            self.episode_id = f"ep_{uuid.uuid4().hex[:12]}"

    def __hash__(self):
        return hash(self.episode_id or self.timestamp)


class EpisodicMemory:
    """
    Hippocampal-inspired episodic memory system.

    Stores specific experiences with their contexts and enables:
    - Pattern separation for distinct storage
    - Pattern completion for retrieval from partial cues
    - Memory consolidation through replay
    - Prioritized storage based on novelty and significance
    """

    def __init__(
        self,
        state_size: int,
        max_episodes: int = 10000,
        consolidation_threshold: float = 0.7,
        replay_rate: float = 0.1,
        similarity_threshold: float = 0.85,
        enable_consolidation: bool = True,
        enable_forgetting: bool = True,
        forgetting_rate: float = 0.0001,
        novelty_bonus: float = 2.0,
        emotional_bonus: float = 1.5,
        random_seed: Optional[int] = None,
        vector_store: Optional['VectorStore'] = None,
    ):
        """
        Initialize episodic memory system.

        Args:
            state_size: Size of neural state vectors
            max_episodes: Maximum number of episodes to store
            consolidation_threshold: Minimum strength for long-term storage
            replay_rate: Probability of replaying memories
            similarity_threshold: Threshold for considering memories similar
            enable_consolidation: Whether to consolidate memories
            enable_forgetting: Whether to forget old/weak memories
            forgetting_rate: Rate of natural forgetting
            novelty_bonus: Multiplier for storing novel experiences
            emotional_bonus: Multiplier for emotionally significant experiences
            random_seed: Random seed for reproducibility
            vector_store: Optional VectorStore for persistent storage
        """
        if random_seed is not None:
            np.random.seed(random_seed)

        self.state_size = state_size
        self.max_episodes = max_episodes
        self.consolidation_threshold = consolidation_threshold
        self.replay_rate = replay_rate
        self.similarity_threshold = similarity_threshold
        self.enable_consolidation = enable_consolidation
        self.enable_forgetting = enable_forgetting
        self.forgetting_rate = forgetting_rate
        self.novelty_bonus = novelty_bonus
        self.emotional_bonus = emotional_bonus

        # Persistent storage (optional)
        self.vector_store = vector_store
        self._persistence_enabled = vector_store is not None

        # Storage for episodes (in-memory cache)
        self.episodes: List[Episode] = []
        self.episode_index: Dict[str, Episode] = {}  # Fast lookup by ID

        # Consolidation tracking
        self.consolidated_episodes: List[Episode] = []
        self.consolidation_weights = np.zeros((max_episodes, state_size))

        # Retrieval cache for efficiency
        self.retrieval_cache: Dict[int, Episode] = {}
        self.cache_hits = 0
        self.cache_misses = 0

        # Statistics
        self.total_stored = 0
        self.total_retrieved = 0
        self.total_replayed = 0
        self.total_forgotten = 0

        # Load from persistent storage if available
        if self._persistence_enabled:
            self._load_from_store()

        logger.info(f"Initialized episodic memory: state_size={state_size}, "
                   f"max_episodes={max_episodes}, persistence={self._persistence_enabled}")

    def store(
        self,
        state: np.ndarray,
        sensory_data: Dict[str, np.ndarray],
        context: Optional[Dict[str, Any]] = None,
        emotional_valence: float = 0.0,
        surprise_level: float = 0.0,
    ) -> Episode:
        """
        Store a new episodic memory.

        Args:
            state: Current neural state
            sensory_data: Multimodal sensory inputs
            context: Contextual information
            emotional_valence: Emotional significance (-1 to 1)
            surprise_level: How surprising/novel (0 to 1)

        Returns:
            The stored episode
        """
        if context is None:
            context = {}

        # Create episode
        episode = Episode(
            timestamp=time.time(),
            state=state.copy(),
            context=context,
            sensory_data={k: v.copy() for k, v in sensory_data.items()},
            emotional_valence=emotional_valence,
            surprise_level=surprise_level,
        )

        # Calculate storage priority
        priority = 1.0
        if surprise_level > 0.5:
            priority *= self.novelty_bonus
        if abs(emotional_valence) > 0.5:
            priority *= self.emotional_bonus

        # Check if similar episode exists (pattern separation)
        similar = self._find_similar_episode(state)
        if similar and priority < 1.5:
            # Don't store very similar episodes unless high priority
            logger.debug(f"Skipping storage of similar episode")
            return similar

        # Store episode
        self.episodes.append(episode)
        self.episode_index[episode.episode_id] = episode
        self.total_stored += 1

        # Check capacity and forget if needed
        if len(self.episodes) > self.max_episodes:
            self._forget_weakest()

        # Immediate consolidation for highly significant episodes
        if priority > 2.0 and self.enable_consolidation:
            episode.consolidation_strength = min(1.0, priority / 3.0)
            self.consolidated_episodes.append(episode)

        # Persist to vector store
        if self._persistence_enabled:
            self._persist_episode(episode)

        logger.debug(f"Stored episode {len(self.episodes)} with priority {priority:.2f}")
        return episode

    def retrieve(
        self,
        cue: Optional[np.ndarray] = None,
        context: Optional[Dict[str, Any]] = None,
        n_episodes: int = 1,
        similarity_threshold: Optional[float] = None,
    ) -> List[Episode]:
        """
        Retrieve episodes matching the cue and context.

        Args:
            cue: Neural state cue for retrieval (pattern completion)
            context: Contextual cue for retrieval
            n_episodes: Number of episodes to retrieve
            similarity_threshold: Minimum similarity for retrieval

        Returns:
            List of retrieved episodes
        """
        if similarity_threshold is None:
            similarity_threshold = self.similarity_threshold

        if len(self.episodes) == 0:
            return []

        # Calculate similarities to all episodes
        similarities = []

        for i, episode in enumerate(self.episodes):
            similarity = 0.0
            weight_sum = 0.0

            # State similarity
            if cue is not None:
                # Ensure compatible dimensions
                min_size = min(len(cue), len(episode.state))
                cue_norm = cue[:min_size] / (np.linalg.norm(cue[:min_size]) + 1e-10)
                state_norm = episode.state[:min_size] / (np.linalg.norm(episode.state[:min_size]) + 1e-10)

                state_sim = np.dot(cue_norm, state_norm)
                similarity += state_sim * 0.7
                weight_sum += 0.7

            # Context similarity
            if context is not None and episode.context:
                context_sim = self._context_similarity(context, episode.context)
                similarity += context_sim * 0.3
                weight_sum += 0.3

            # Normalize
            if weight_sum > 0:
                similarity /= weight_sum

            # Boost by consolidation strength
            similarity *= (1.0 + episode.consolidation_strength * 0.5)

            similarities.append((similarity, i, episode))

        # Sort by similarity
        similarities.sort(reverse=True, key=lambda x: x[0])

        # Filter by threshold and select top n
        retrieved = []
        for sim, idx, episode in similarities[:n_episodes]:
            if sim >= similarity_threshold:
                retrieved.append(episode)
                self.total_retrieved += 1

                # Update replay count
                episode.replay_count += 1

        logger.debug(f"Retrieved {len(retrieved)} episodes")
        return retrieved

    def consolidate(
        self,
        n_replay: int = 10,
        batch_size: int = 5,
    ) -> int:
        """
        Consolidate memories through replay.

        This simulates offline consolidation (e.g., during sleep) where
        memories are replayed and strengthened.

        Args:
            n_replay: Number of replay cycles
            batch_size: Number of episodes per replay cycle

        Returns:
            Number of episodes consolidated
        """
        if not self.enable_consolidation or len(self.episodes) == 0:
            return 0

        consolidated_count = 0

        for _ in range(n_replay):
            # Sample episodes for replay (prioritize recent + significant)
            candidates = self._select_replay_candidates(batch_size)

            for episode in candidates:
                # Increase consolidation strength
                episode.consolidation_strength = min(
                    1.0,
                    episode.consolidation_strength + 0.1
                )

                episode.replay_count += 1
                self.total_replayed += 1

                # Move to consolidated if threshold reached
                if (episode.consolidation_strength >= self.consolidation_threshold and
                    episode not in self.consolidated_episodes):
                    self.consolidated_episodes.append(episode)
                    consolidated_count += 1

        logger.debug(f"Consolidated {consolidated_count} episodes through replay")
        return consolidated_count

    def _find_similar_episode(
        self,
        state: np.ndarray,
        threshold: Optional[float] = None,
    ) -> Optional[Episode]:
        """Find the most similar episode to the given state."""
        if threshold is None:
            threshold = self.similarity_threshold

        if len(self.episodes) == 0:
            return None

        max_similarity = 0.0
        most_similar = None

        for episode in self.episodes[-100:]:  # Check only recent episodes for efficiency
            # Ensure compatible dimensions
            min_size = min(len(state), len(episode.state))
            state_norm = state[:min_size] / (np.linalg.norm(state[:min_size]) + 1e-10)
            episode_norm = episode.state[:min_size] / (np.linalg.norm(episode.state[:min_size]) + 1e-10)

            similarity = np.dot(state_norm, episode_norm)

            if similarity > max_similarity:
                max_similarity = similarity
                most_similar = episode

        if max_similarity >= threshold:
            return most_similar

        return None

    def _context_similarity(
        self,
        context1: Dict[str, Any],
        context2: Dict[str, Any],
    ) -> float:
        """Calculate similarity between two contexts."""
        if not context1 or not context2:
            return 0.0

        # Get common keys
        common_keys = set(context1.keys()) & set(context2.keys())
        if not common_keys:
            return 0.0

        # Calculate match rate
        matches = 0
        for key in common_keys:
            if context1[key] == context2[key]:
                matches += 1

        return matches / len(common_keys)

    def _select_replay_candidates(self, batch_size: int) -> List[Episode]:
        """Select episodes for replay prioritizing recent and significant ones."""
        if len(self.episodes) == 0:
            return []

        # Score each episode
        scores = []
        for i, episode in enumerate(self.episodes):
            # Recency score (exponential decay)
            age = len(self.episodes) - i
            recency_score = np.exp(-age / (len(self.episodes) + 1))

            # Significance score
            significance = (
                abs(episode.emotional_valence) * 0.4 +
                episode.surprise_level * 0.4 +
                (1.0 - episode.consolidation_strength) * 0.2  # Prioritize unconsolidated
            )

            # Combined score
            score = recency_score * 0.4 + significance * 0.6
            scores.append((score, episode))

        # Sort by score and select top candidates
        scores.sort(reverse=True, key=lambda x: x[0])
        return [ep for _, ep in scores[:batch_size]]

    def _forget_weakest(self) -> None:
        """Forget the weakest/oldest episode to make room."""
        if not self.enable_forgetting or len(self.episodes) == 0:
            return

        # Score episodes (inverse of consolidation + recency)
        scores = []
        for i, episode in enumerate(self.episodes):
            # Lower score = more forgettable
            age = len(self.episodes) - i
            recency = np.exp(-age / len(self.episodes))

            score = (
                episode.consolidation_strength * 0.4 +
                recency * 0.3 +
                abs(episode.emotional_valence) * 0.2 +
                episode.surprise_level * 0.1
            )

            scores.append((score, i, episode))

        # Sort and remove weakest
        scores.sort(key=lambda x: x[0])
        _, idx, episode = scores[0]

        # Don't forget highly consolidated episodes
        if episode.consolidation_strength < self.consolidation_threshold:
            self.episodes.pop(idx)
            self.total_forgotten += 1
            logger.debug(f"Forgot episode with score {scores[0][0]:.3f}")
        else:
            # Remove oldest instead
            self.episodes.pop(0)
            self.total_forgotten += 1

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the episodic memory system."""
        if len(self.episodes) > 0:
            avg_consolidation = np.mean([ep.consolidation_strength for ep in self.episodes])
            avg_replays = np.mean([ep.replay_count for ep in self.episodes])
            avg_surprise = np.mean([ep.surprise_level for ep in self.episodes])
            avg_emotion = np.mean([abs(ep.emotional_valence) for ep in self.episodes])
        else:
            avg_consolidation = 0.0
            avg_replays = 0.0
            avg_surprise = 0.0
            avg_emotion = 0.0

        return {
            'total_episodes': len(self.episodes),
            'consolidated_episodes': len(self.consolidated_episodes),
            'total_stored': self.total_stored,
            'total_retrieved': self.total_retrieved,
            'total_replayed': self.total_replayed,
            'total_forgotten': self.total_forgotten,
            'avg_consolidation_strength': float(avg_consolidation),
            'avg_replay_count': float(avg_replays),
            'avg_surprise_level': float(avg_surprise),
            'avg_emotional_valence': float(avg_emotion),
            'cache_hit_rate': self.cache_hits / max(1, self.cache_hits + self.cache_misses),
        }

    def serialize(self) -> Dict[str, Any]:
        """Serialize the episodic memory for saving."""
        return {
            'state_size': self.state_size,
            'max_episodes': self.max_episodes,
            'consolidation_threshold': self.consolidation_threshold,
            'episodes': [
                {
                    'timestamp': ep.timestamp,
                    'state': ep.state.tolist(),
                    'context': ep.context,
                    'emotional_valence': ep.emotional_valence,
                    'surprise_level': ep.surprise_level,
                    'replay_count': ep.replay_count,
                    'consolidation_strength': ep.consolidation_strength,
                }
                for ep in self.consolidated_episodes  # Only save consolidated for efficiency
            ],
            'stats': self.get_stats(),
        }

    @classmethod
    def deserialize(cls, data: Dict[str, Any], vector_store: Optional['VectorStore'] = None) -> 'EpisodicMemory':
        """Create an episodic memory instance from serialized data."""
        instance = cls(
            state_size=data['state_size'],
            max_episodes=data['max_episodes'],
            consolidation_threshold=data['consolidation_threshold'],
            vector_store=vector_store,
        )

        # Restore consolidated episodes (only if not loading from vector store)
        if not instance._persistence_enabled:
            for ep_data in data.get('episodes', []):
                episode = Episode(
                    timestamp=ep_data['timestamp'],
                    state=np.array(ep_data['state']),
                    context=ep_data['context'],
                    sensory_data={},  # Not saved for efficiency
                    emotional_valence=ep_data['emotional_valence'],
                    surprise_level=ep_data['surprise_level'],
                    replay_count=ep_data['replay_count'],
                    consolidation_strength=ep_data['consolidation_strength'],
                    episode_id=ep_data.get('episode_id', ''),
                )
                instance.consolidated_episodes.append(episode)
                instance.episodes.append(episode)
                instance.episode_index[episode.episode_id] = episode

        return instance

    # ==================== Persistence Methods ====================

    def _persist_episode(self, episode: Episode) -> bool:
        """Persist an episode to the vector store."""
        if not self._persistence_enabled or not self.vector_store:
            return False

        try:
            metadata = {
                'timestamp': episode.timestamp,
                'emotional_valence': episode.emotional_valence,
                'surprise_level': episode.surprise_level,
                'replay_count': episode.replay_count,
                'consolidation_strength': episode.consolidation_strength,
                'context': episode.context,  # Will be serialized by VectorStore
            }

            return self.vector_store.add_episode(
                episode_id=episode.episode_id,
                embedding=episode.state,
                metadata=metadata
            )
        except Exception as e:
            logger.error(f"Failed to persist episode {episode.episode_id}: {e}")
            return False

    def _load_from_store(self) -> int:
        """Load episodes from the vector store into memory."""
        if not self._persistence_enabled or not self.vector_store:
            return 0

        loaded_count = 0
        try:
            # Get count of stored episodes
            stored_count = self.vector_store.count_episodes()
            if stored_count == 0:
                logger.info("No episodes in vector store to load")
                return 0

            logger.info(f"Loading {stored_count} episodes from vector store...")

            # We don't load all episodes upfront - they'll be loaded on demand
            # Just mark that persistence is ready
            logger.info(f"Vector store has {stored_count} episodes available")
            return stored_count

        except Exception as e:
            logger.error(f"Failed to load from vector store: {e}")
            return 0

    def retrieve_from_store(
        self,
        cue: np.ndarray,
        n_episodes: int = 5,
    ) -> List[Episode]:
        """
        Retrieve episodes using the vector store's ANN search.

        This is faster than in-memory search for large episode counts.

        Args:
            cue: Neural state cue for retrieval
            n_episodes: Number of episodes to retrieve

        Returns:
            List of retrieved episodes
        """
        if not self._persistence_enabled or not self.vector_store:
            return self.retrieve(cue=cue, n_episodes=n_episodes)

        try:
            results = self.vector_store.search_episodes(cue, n_results=n_episodes)

            episodes = []
            for result in results:
                # Check if already in memory
                if result.id in self.episode_index:
                    episodes.append(self.episode_index[result.id])
                else:
                    # Reconstruct episode from stored data
                    metadata = result.metadata
                    episode = Episode(
                        timestamp=metadata.get('timestamp', time.time()),
                        state=result.embedding,
                        context=metadata.get('context', {}),
                        sensory_data={},  # Not stored
                        emotional_valence=metadata.get('emotional_valence', 0.0),
                        surprise_level=metadata.get('surprise_level', 0.0),
                        replay_count=metadata.get('replay_count', 0),
                        consolidation_strength=metadata.get('consolidation_strength', 0.0),
                        episode_id=result.id,
                    )
                    episodes.append(episode)

                    # Cache in memory
                    self.episode_index[result.id] = episode

            self.total_retrieved += len(episodes)
            return episodes

        except Exception as e:
            logger.error(f"Failed to retrieve from store: {e}")
            return self.retrieve(cue=cue, n_episodes=n_episodes)

    def sync_consolidated_to_store(self) -> int:
        """Sync all consolidated episodes to the vector store."""
        if not self._persistence_enabled or not self.vector_store:
            return 0

        synced = 0
        for episode in self.consolidated_episodes:
            if self._persist_episode(episode):
                synced += 1

        logger.info(f"Synced {synced} consolidated episodes to vector store")
        return synced

    def set_vector_store(self, vector_store: 'VectorStore') -> None:
        """Set the vector store for persistence."""
        self.vector_store = vector_store
        self._persistence_enabled = True

        # Sync existing consolidated episodes
        if self.consolidated_episodes:
            self.sync_consolidated_to_store()
