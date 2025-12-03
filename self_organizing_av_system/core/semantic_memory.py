"""
Semantic Memory Network for ATLAS

Implements a graph-based semantic memory system for storing and reasoning
about concepts, facts, and their relationships.

This is a critical component for superintelligence development, enabling:
- Abstract concept representation
- Relational reasoning
- Knowledge graph construction
- Inference and deduction
- Semantic generalization
"""

import numpy as np
import logging
from typing import List, Dict, Tuple, Optional, Any, Set, Union
from dataclasses import dataclass, field
from enum import Enum
import networkx as nx

logger = logging.getLogger(__name__)


class RelationType(Enum):
    """Types of relations between concepts"""
    IS_A = "is_a"  # Taxonomic (dog is_a animal)
    HAS_A = "has_a"  # Part-whole (car has_a wheel)
    CAUSES = "causes"  # Causal (rain causes wet)
    SIMILAR_TO = "similar_to"  # Similarity
    OPPOSITE_OF = "opposite_of"  # Antonym
    MEMBER_OF = "member_of"  # Set membership
    ATTRIBUTE = "attribute"  # Property (apple has attribute red)
    BEFORE = "before"  # Temporal ordering
    AFTER = "after"  # Temporal ordering
    ENABLES = "enables"  # Functional relationship
    PREVENTS = "prevents"  # Negative functional relationship
    USED_FOR = "used_for"  # Purpose/function
    LOCATED_AT = "located_at"  # Spatial relationship
    PART_OF = "part_of"  # Component relationship
    CREATED_BY = "created_by"  # Creation/authorship


@dataclass
class Concept:
    """Represents an abstract concept in semantic memory"""
    name: str
    embedding: np.ndarray  # Vector representation grounded in sensory patterns
    attributes: Dict[str, Any] = field(default_factory=dict)
    activation_count: int = 0
    creation_time: float = 0.0
    last_activation: float = 0.0
    consolidation_strength: float = 0.0

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if isinstance(other, Concept):
            return self.name == other.name
        return False


@dataclass
class Relation:
    """Represents a relation between two concepts"""
    source: str  # Source concept name
    target: str  # Target concept name
    relation_type: RelationType
    strength: float = 1.0  # Confidence in this relation
    bidirectional: bool = False  # Whether relation goes both ways
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __hash__(self):
        return hash((self.source, self.target, self.relation_type))


class SemanticMemory:
    """
    Graph-based semantic memory for concepts and relationships.

    Provides:
    - Concept learning and representation
    - Relational graph construction
    - Spreading activation for reasoning
    - Analogical reasoning
    - Inference and deduction
    """

    def __init__(
        self,
        embedding_size: int,
        max_concepts: int = 100000,
        activation_threshold: float = 0.3,
        spreading_decay: float = 0.7,
        learning_rate: float = 0.01,
        consolidation_rate: float = 0.001,
        similarity_threshold: float = 0.85,
        enable_inference: bool = True,
        enable_generalization: bool = True,
        random_seed: Optional[int] = None,
    ):
        """
        Initialize semantic memory system.

        Args:
            embedding_size: Size of concept embeddings
            max_concepts: Maximum number of concepts to store
            activation_threshold: Threshold for concept activation
            spreading_decay: Decay factor for spreading activation
            learning_rate: Rate of embedding updates
            consolidation_rate: Rate of strengthening relations
            similarity_threshold: Threshold for considering concepts similar
            enable_inference: Whether to perform inference
            enable_generalization: Whether to generalize from examples
            random_seed: Random seed for reproducibility
        """
        if random_seed is not None:
            np.random.seed(random_seed)

        self.embedding_size = embedding_size
        self.max_concepts = max_concepts
        self.activation_threshold = activation_threshold
        self.spreading_decay = spreading_decay
        self.learning_rate = learning_rate
        self.consolidation_rate = consolidation_rate
        self.similarity_threshold = similarity_threshold
        self.enable_inference = enable_inference
        self.enable_generalization = enable_generalization

        # Concept storage
        self.concepts: Dict[str, Concept] = {}
        self.concept_graph = nx.DiGraph()  # Directed graph of concepts

        # Current activation state
        self.activation_state: Dict[str, float] = {}

        # Inference rules learned from data
        self.inference_rules: List[Dict[str, Any]] = []

        # Statistics
        self.total_concepts_created = 0
        self.total_relations_created = 0
        self.total_inferences_made = 0
        self.total_generalizations = 0

        logger.info(f"Initialized semantic memory: embedding_size={embedding_size}")

    def add_concept(
        self,
        name: str,
        embedding: np.ndarray,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> Concept:
        """
        Add or update a concept in semantic memory.

        Args:
            name: Concept name/identifier
            embedding: Vector embedding for the concept
            attributes: Optional attributes

        Returns:
            The created or updated concept
        """
        if attributes is None:
            attributes = {}

        # Check if concept already exists
        if name in self.concepts:
            # Update existing concept
            concept = self.concepts[name]
            # Blend embeddings
            concept.embedding = (
                (1 - self.learning_rate) * concept.embedding +
                self.learning_rate * embedding
            )
            concept.attributes.update(attributes)
            concept.activation_count += 1
            logger.debug(f"Updated existing concept: {name}")
        else:
            # Create new concept
            import time
            concept = Concept(
                name=name,
                embedding=embedding.copy(),
                attributes=attributes,
                creation_time=time.time(),
            )
            self.concepts[name] = concept
            self.concept_graph.add_node(name, concept=concept)
            self.total_concepts_created += 1
            logger.debug(f"Created new concept: {name}")

        # Check capacity
        if len(self.concepts) > self.max_concepts:
            self._prune_weakest_concept()

        return concept

    def add_relation(
        self,
        source: str,
        target: str,
        relation_type: Union[str, RelationType],
        strength: float = 1.0,
        bidirectional: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Relation:
        """
        Add a relation between two concepts.

        Args:
            source: Source concept name
            target: Target concept name
            relation_type: Type of relation
            strength: Strength/confidence of relation
            bidirectional: Whether relation is bidirectional
            metadata: Optional metadata

        Returns:
            The created relation
        """
        if metadata is None:
            metadata = {}

        # Convert string to enum if needed
        if isinstance(relation_type, str):
            relation_type = RelationType(relation_type)

        # Ensure concepts exist
        if source not in self.concepts:
            logger.warning(f"Source concept {source} not found")
            return None
        if target not in self.concepts:
            logger.warning(f"Target concept {target} not found")
            return None

        # Create relation
        relation = Relation(
            source=source,
            target=target,
            relation_type=relation_type,
            strength=strength,
            bidirectional=bidirectional,
            metadata=metadata,
        )

        # Add to graph
        self.concept_graph.add_edge(
            source, target,
            relation=relation,
            type=relation_type,
            weight=strength,
        )

        # Add reverse edge if bidirectional
        if bidirectional:
            self.concept_graph.add_edge(
                target, source,
                relation=relation,
                type=relation_type,
                weight=strength,
            )

        self.total_relations_created += 1
        logger.debug(f"Added relation: {source} -{relation_type.value}-> {target}")

        # Learn inference rules if enabled
        if self.enable_inference:
            self._extract_inference_rules(source, target, relation_type)

        return relation

    def query(
        self,
        cue_embedding: Optional[np.ndarray] = None,
        cue_concept: Optional[str] = None,
        relation_type: Optional[RelationType] = None,
        n_results: int = 5,
        use_spreading_activation: bool = True,
    ) -> List[Tuple[str, float]]:
        """
        Query semantic memory for relevant concepts.

        Args:
            cue_embedding: Vector cue for similarity search
            cue_concept: Concept name to start spreading activation from
            relation_type: Filter by relation type
            n_results: Number of results to return
            use_spreading_activation: Whether to use spreading activation

        Returns:
            List of (concept_name, activation_level) tuples
        """
        if len(self.concepts) == 0:
            return []

        # Reset activation state
        self.activation_state = {}

        if cue_concept and use_spreading_activation:
            # Spreading activation from cue concept
            self._spread_activation(cue_concept, relation_type=relation_type)
            # Sort by activation
            results = sorted(
                self.activation_state.items(),
                key=lambda x: x[1],
                reverse=True
            )[:n_results]

        elif cue_embedding is not None:
            # Similarity-based retrieval
            similarities = []
            for name, concept in self.concepts.items():
                # Ensure compatible dimensions
                min_size = min(len(cue_embedding), len(concept.embedding))
                cue_norm = cue_embedding[:min_size] / (np.linalg.norm(cue_embedding[:min_size]) + 1e-10)
                concept_norm = concept.embedding[:min_size] / (np.linalg.norm(concept.embedding[:min_size]) + 1e-10)

                similarity = np.dot(cue_norm, concept_norm)
                similarities.append((name, float(similarity)))

            # Sort and return top n
            results = sorted(similarities, key=lambda x: x[1], reverse=True)[:n_results]

        else:
            # No cue provided, return empty
            return []

        logger.debug(f"Query returned {len(results)} results")
        return results

    def _spread_activation(
        self,
        source: str,
        activation: float = 1.0,
        depth: int = 0,
        max_depth: int = 3,
        relation_type: Optional[RelationType] = None,
    ) -> None:
        """
        Spread activation through the concept graph.

        Args:
            source: Starting concept
            activation: Current activation level
            depth: Current depth in spreading
            max_depth: Maximum depth to spread
            relation_type: Optional filter for relation types
        """
        # Base case: max depth reached or activation too low
        if depth >= max_depth or activation < self.activation_threshold:
            return

        # Update activation
        current_activation = self.activation_state.get(source, 0.0)
        self.activation_state[source] = max(current_activation, activation)

        # Spread to neighbors
        if source in self.concept_graph:
            for neighbor in self.concept_graph.neighbors(source):
                edge_data = self.concept_graph.edges[source, neighbor]

                # Filter by relation type if specified
                if relation_type and edge_data.get('type') != relation_type:
                    continue

                # Calculate spread activation
                edge_strength = edge_data.get('weight', 1.0)
                next_activation = activation * self.spreading_decay * edge_strength

                # Recurse
                self._spread_activation(
                    neighbor, next_activation, depth + 1, max_depth, relation_type
                )

    def infer(
        self,
        source: str,
        target: Optional[str] = None,
        max_steps: int = 5,
    ) -> List[Tuple[str, List[str], float]]:
        """
        Perform inference to find relationships.

        Args:
            source: Source concept
            target: Optional target concept (if None, find all reachable)
            max_steps: Maximum inference chain length

        Returns:
            List of (target_concept, inference_path, confidence) tuples
        """
        if not self.enable_inference:
            return []

        if source not in self.concept_graph:
            return []

        inferences = []

        if target:
            # Find paths between source and target
            try:
                paths = list(nx.all_simple_paths(
                    self.concept_graph, source, target, cutoff=max_steps
                ))

                for path in paths:
                    # Calculate path confidence
                    confidence = 1.0
                    for i in range(len(path) - 1):
                        edge_data = self.concept_graph.edges[path[i], path[i+1]]
                        confidence *= edge_data.get('weight', 1.0) * 0.9  # Decay with length

                    inferences.append((target, path, confidence))

            except nx.NetworkXNoPath:
                pass

        else:
            # Find all concepts reachable within max_steps
            reachable = {}
            self._find_reachable(source, max_steps, reachable)

            for concept, (path, confidence) in reachable.items():
                if concept != source:
                    inferences.append((concept, path, confidence))

        # Sort by confidence
        inferences.sort(key=lambda x: x[2], reverse=True)

        self.total_inferences_made += len(inferences)
        logger.debug(f"Made {len(inferences)} inferences from {source}")

        return inferences

    def _find_reachable(
        self,
        source: str,
        max_steps: int,
        reachable: Dict[str, Tuple[List[str], float]],
        current_path: Optional[List[str]] = None,
        current_confidence: float = 1.0,
    ) -> None:
        """Recursively find all reachable concepts."""
        if current_path is None:
            current_path = [source]

        if len(current_path) > max_steps:
            return

        # Mark as reachable
        if source not in reachable or current_confidence > reachable[source][1]:
            reachable[source] = (current_path.copy(), current_confidence)

        # Explore neighbors
        if source in self.concept_graph:
            for neighbor in self.concept_graph.neighbors(source):
                if neighbor not in current_path:  # Avoid cycles
                    edge_data = self.concept_graph.edges[source, neighbor]
                    edge_strength = edge_data.get('weight', 1.0)
                    new_confidence = current_confidence * edge_strength * 0.9

                    self._find_reachable(
                        neighbor,
                        max_steps,
                        reachable,
                        current_path + [neighbor],
                        new_confidence,
                    )

    def _extract_inference_rules(
        self,
        source: str,
        target: str,
        relation_type: RelationType,
    ) -> None:
        """Extract inference rules from relations (e.g., transitivity)."""
        # Example: if A is_a B and B is_a C, then A is_a C
        if relation_type == RelationType.IS_A:
            # Check for transitive chains
            if target in self.concept_graph:
                for next_target in self.concept_graph.neighbors(target):
                    edge_data = self.concept_graph.edges[target, next_target]
                    if edge_data.get('type') == RelationType.IS_A:
                        # Infer transitive relation
                        if not self.concept_graph.has_edge(source, next_target):
                            confidence = (
                                self.concept_graph.edges[source, target].get('weight', 1.0) *
                                edge_data.get('weight', 1.0) * 0.8
                            )
                            self.add_relation(
                                source, next_target,
                                RelationType.IS_A,
                                strength=confidence,
                                metadata={'inferred': True, 'rule': 'transitivity'}
                            )
                            self.total_inferences_made += 1

    def generalize(
        self,
        examples: List[str],
        min_similarity: float = 0.7,
    ) -> Optional[Concept]:
        """
        Generalize from multiple examples to create a new abstract concept.

        Args:
            examples: List of concept names to generalize from
            min_similarity: Minimum similarity required for generalization

        Returns:
            The generalized concept, or None if no generalization possible
        """
        if not self.enable_generalization or len(examples) < 2:
            return None

        # Get example concepts
        example_concepts = [self.concepts[name] for name in examples if name in self.concepts]
        if len(example_concepts) < 2:
            return None

        # Compute average embedding
        avg_embedding = np.mean([c.embedding for c in example_concepts], axis=0)

        # Check if generalization is warranted (examples are similar enough)
        similarities = []
        for i, c1 in enumerate(example_concepts):
            for c2 in example_concepts[i+1:]:
                sim = np.dot(
                    c1.embedding / (np.linalg.norm(c1.embedding) + 1e-10),
                    c2.embedding / (np.linalg.norm(c2.embedding) + 1e-10)
                )
                similarities.append(sim)

        if np.mean(similarities) < min_similarity:
            logger.debug(f"Examples not similar enough for generalization")
            return None

        # Create generalized concept
        general_name = f"GENERAL_{self.total_generalizations}"
        general_concept = self.add_concept(
            general_name,
            avg_embedding,
            attributes={'generalized_from': examples, 'type': 'generalization'}
        )

        # Add IS_A relations from examples to generalization
        for example_name in examples:
            self.add_relation(example_name, general_name, RelationType.IS_A, strength=0.9)

        self.total_generalizations += 1
        logger.debug(f"Generalized {len(examples)} concepts into {general_name}")

        return general_concept

    def _prune_weakest_concept(self) -> None:
        """Remove the weakest concept to make room."""
        if len(self.concepts) == 0:
            return

        # Score concepts
        scores = []
        for name, concept in self.concepts.items():
            # Don't prune highly activated or consolidated concepts
            if concept.consolidation_strength > 0.7:
                continue

            # Score based on activation count and consolidation
            score = concept.activation_count * 0.5 + concept.consolidation_strength * 0.5
            scores.append((score, name))

        if not scores:
            # All concepts are important, remove oldest
            oldest = min(self.concepts.items(), key=lambda x: x[1].creation_time)
            name = oldest[0]
        else:
            # Remove weakest
            scores.sort()
            name = scores[0][1]

        # Remove from graph and concepts
        if name in self.concept_graph:
            self.concept_graph.remove_node(name)
        del self.concepts[name]

        logger.debug(f"Pruned concept: {name}")

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the semantic memory system."""
        return {
            'total_concepts': len(self.concepts),
            'total_relations': self.concept_graph.number_of_edges(),
            'total_concepts_created': self.total_concepts_created,
            'total_relations_created': self.total_relations_created,
            'total_inferences_made': self.total_inferences_made,
            'total_generalizations': self.total_generalizations,
            'avg_degree': np.mean([d for n, d in self.concept_graph.degree()]) if len(self.concepts) > 0 else 0,
            'graph_density': nx.density(self.concept_graph) if len(self.concepts) > 0 else 0,
        }

    def serialize(self) -> Dict[str, Any]:
        """Serialize the semantic memory for saving."""
        return {
            'embedding_size': self.embedding_size,
            'concepts': {
                name: {
                    'embedding': concept.embedding.tolist(),
                    'attributes': concept.attributes,
                    'consolidation_strength': concept.consolidation_strength,
                }
                for name, concept in list(self.concepts.items())[:1000]  # Save top 1000 for efficiency
            },
            'relations': [
                {
                    'source': u,
                    'target': v,
                    'type': data.get('type').value if data.get('type') else 'unknown',
                    'weight': data.get('weight', 1.0),
                }
                for u, v, data in self.concept_graph.edges(data=True)
            ],
            'stats': self.get_stats(),
        }

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> 'SemanticMemory':
        """Create a semantic memory instance from serialized data."""
        instance = cls(embedding_size=data['embedding_size'])

        # Restore concepts
        for name, concept_data in data.get('concepts', {}).items():
            instance.add_concept(
                name,
                np.array(concept_data['embedding']),
                attributes=concept_data.get('attributes', {}),
            )
            instance.concepts[name].consolidation_strength = concept_data.get('consolidation_strength', 0.0)

        # Restore relations
        for rel in data.get('relations', []):
            instance.add_relation(
                rel['source'],
                rel['target'],
                rel['type'],
                strength=rel.get('weight', 1.0),
            )

        return instance
