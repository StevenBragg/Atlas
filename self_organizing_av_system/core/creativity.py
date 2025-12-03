#!/usr/bin/env python3
"""
Creativity Engine for ATLAS Superintelligence

This module implements creative cognition capabilities that enable:
1. Generative synthesis - Creating novel ideas from learned patterns
2. Conceptual blending - Combining concepts in creative ways
3. Mental simulation - Imagining hypothetical scenarios
4. Divergent thinking - Generating multiple alternative solutions
5. Novelty detection and appreciation

Core Principles:
- No backpropagation - uses local Hebbian/STDP learning
- Biologically plausible - inspired by default mode network and associative memory
- Stochastic generation - Randomness enables exploration
- Constraint relaxation - Loosening learned patterns for novelty
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Set
from enum import Enum
from collections import deque
import logging
import time

logger = logging.getLogger(__name__)


class CreativeMode(Enum):
    """Modes of creative thinking"""
    CONVERGENT = "convergent"  # Finding single best solution
    DIVERGENT = "divergent"    # Generating many possibilities
    EXPLORATORY = "exploratory"  # Random walk in concept space
    COMBINATORIAL = "combinatorial"  # Systematic combinations
    TRANSFORMATIONAL = "transformational"  # Deep structure modification


class NoveltyType(Enum):
    """Types of novelty"""
    STATISTICAL = "statistical"  # Rare in distribution
    STRUCTURAL = "structural"    # New combinations
    CONCEPTUAL = "conceptual"    # New meanings
    FUNCTIONAL = "functional"    # New uses


@dataclass
class Concept:
    """A concept that can be creatively manipulated"""
    concept_id: str
    name: str
    embedding: np.ndarray
    attributes: Dict[str, Any] = field(default_factory=dict)
    relations: List[Tuple[str, str]] = field(default_factory=list)  # (relation, target_id)
    abstraction_level: float = 0.5  # 0 = concrete, 1 = abstract
    creation_time: float = field(default_factory=time.time)


@dataclass
class Blend:
    """Result of conceptual blending"""
    blend_id: str
    source_concepts: List[str]
    blend_embedding: np.ndarray
    emergent_properties: Dict[str, Any]
    novelty_score: float
    coherence_score: float
    creation_time: float = field(default_factory=time.time)


@dataclass
class Imagination:
    """A mental simulation or imagined scenario"""
    imagination_id: str
    initial_state: np.ndarray
    trajectory: List[np.ndarray]
    constraints: Dict[str, Any]
    plausibility: float
    novelty: float
    utility: float
    creation_time: float = field(default_factory=time.time)


class CreativityEngine:
    """
    Engine for creative cognition including generative synthesis,
    conceptual blending, and mental simulation.
    """

    def __init__(
        self,
        embedding_dim: int = 64,
        max_concepts: int = 1000,
        novelty_threshold: float = 0.3,
        coherence_threshold: float = 0.4,
        temperature: float = 1.0,
        learning_rate: float = 0.05,
        random_seed: Optional[int] = None
    ):
        """
        Initialize the creativity engine.

        Args:
            embedding_dim: Dimension of concept embeddings
            max_concepts: Maximum number of concepts to store
            novelty_threshold: Minimum novelty for creative outputs
            coherence_threshold: Minimum coherence for viable outputs
            temperature: Controls randomness in generation
            learning_rate: Learning rate for adaptation
            random_seed: Random seed for reproducibility
        """
        self.embedding_dim = embedding_dim
        self.max_concepts = max_concepts
        self.novelty_threshold = novelty_threshold
        self.coherence_threshold = coherence_threshold
        self.temperature = temperature
        self.learning_rate = learning_rate

        if random_seed is not None:
            np.random.seed(random_seed)

        # Concept storage
        self.concepts: Dict[str, Concept] = {}
        self.concept_counter = 0

        # Blend storage
        self.blends: Dict[str, Blend] = {}
        self.blend_counter = 0

        # Imagination storage
        self.imaginations: Dict[str, Imagination] = {}
        self.imagination_counter = 0

        # Generative model (simple VAE-like decoder without backprop)
        self.generator_weights = np.random.randn(embedding_dim, embedding_dim) * 0.1
        self.generator_bias = np.zeros(embedding_dim)

        # Discriminator for novelty/coherence (learned critic)
        self.discriminator_weights = np.random.randn(embedding_dim, 1) * 0.1

        # Association matrix for concept relations
        self.association_matrix = np.zeros((max_concepts, max_concepts))

        # Constraint relaxation parameters
        self.relaxation_level = 0.0  # 0 = strict, 1 = fully relaxed

        # Statistics tracking
        self.exposure_history: deque = deque(maxlen=10000)  # For novelty detection
        self.total_generations = 0
        self.successful_blends = 0
        self.total_imaginations = 0

    def add_concept(
        self,
        name: str,
        embedding: np.ndarray,
        attributes: Optional[Dict[str, Any]] = None,
        relations: Optional[List[Tuple[str, str]]] = None
    ) -> str:
        """Add a concept to the creativity engine"""
        if len(self.concepts) >= self.max_concepts:
            # Remove oldest concept
            oldest = min(self.concepts.values(), key=lambda c: c.creation_time)
            del self.concepts[oldest.concept_id]

        concept_id = f"concept_{self.concept_counter}"
        self.concept_counter += 1

        embedding = np.asarray(embedding).flatten()[:self.embedding_dim]
        if len(embedding) < self.embedding_dim:
            embedding = np.pad(embedding, (0, self.embedding_dim - len(embedding)))
        embedding = embedding / (np.linalg.norm(embedding) + 1e-8)

        concept = Concept(
            concept_id=concept_id,
            name=name,
            embedding=embedding,
            attributes=attributes or {},
            relations=relations or []
        )

        self.concepts[concept_id] = concept

        # Update exposure history
        self.exposure_history.append(embedding)

        return concept_id

    def generate(
        self,
        seed: Optional[np.ndarray] = None,
        mode: CreativeMode = CreativeMode.DIVERGENT,
        constraints: Optional[Dict[str, Any]] = None,
        n_samples: int = 5
    ) -> List[Tuple[np.ndarray, float, float]]:
        """
        Generate novel concepts using the generative model.

        Args:
            seed: Optional seed embedding to start from
            mode: Creative generation mode
            constraints: Optional constraints on generation
            n_samples: Number of samples to generate

        Returns:
            List of (embedding, novelty, coherence) tuples
        """
        self.total_generations += n_samples

        results = []

        for _ in range(n_samples):
            if seed is not None:
                seed = np.asarray(seed).flatten()[:self.embedding_dim]
                if len(seed) < self.embedding_dim:
                    seed = np.pad(seed, (0, self.embedding_dim - len(seed)))
                latent = seed
            else:
                latent = np.random.randn(self.embedding_dim)

            # Apply mode-specific transformation
            if mode == CreativeMode.DIVERGENT:
                # Add noise for diversity
                noise = np.random.randn(self.embedding_dim) * self.temperature
                latent = latent + noise

            elif mode == CreativeMode.EXPLORATORY:
                # Random walk in latent space
                step = np.random.randn(self.embedding_dim)
                step /= np.linalg.norm(step) + 1e-8
                latent = latent + 0.5 * step

            elif mode == CreativeMode.TRANSFORMATIONAL:
                # Apply learned transformation with relaxation
                transformation = np.eye(self.embedding_dim) + \
                    self.relaxation_level * np.random.randn(self.embedding_dim, self.embedding_dim) * 0.1
                latent = transformation @ latent

            # Generate through decoder
            generated = self.generator_weights @ latent + self.generator_bias
            generated = np.tanh(generated)  # Non-linearity
            generated /= np.linalg.norm(generated) + 1e-8

            # Apply constraints if any
            if constraints:
                generated = self._apply_constraints(generated, constraints)

            # Compute novelty and coherence
            novelty = self._compute_novelty(generated)
            coherence = self._compute_coherence(generated)

            results.append((generated, novelty, coherence))

        # Sort by a combination of novelty and coherence
        results.sort(key=lambda x: 0.5 * x[1] + 0.5 * x[2], reverse=True)

        return results

    def blend(
        self,
        concept_ids: List[str],
        blend_mode: str = "average",
        weights: Optional[List[float]] = None
    ) -> Optional[Blend]:
        """
        Blend multiple concepts together.

        Args:
            concept_ids: IDs of concepts to blend
            blend_mode: 'average', 'max', 'interpolate', 'emergent'
            weights: Optional weights for each concept

        Returns:
            Blend object if successful, None otherwise
        """
        if not concept_ids:
            return None

        # Get concepts
        concepts = [self.concepts[cid] for cid in concept_ids if cid in self.concepts]
        if len(concepts) < 2:
            return None

        embeddings = [c.embedding for c in concepts]

        if weights is None:
            weights = [1.0 / len(concepts)] * len(concepts)

        weights = np.array(weights) / sum(weights)

        # Compute blended embedding based on mode
        if blend_mode == "average":
            blend_embedding = sum(w * e for w, e in zip(weights, embeddings))

        elif blend_mode == "max":
            blend_embedding = np.maximum.reduce(embeddings)

        elif blend_mode == "interpolate":
            # Spherical interpolation for normalized vectors
            blend_embedding = np.zeros(self.embedding_dim)
            for w, e in zip(weights, embeddings):
                blend_embedding += w * e
            blend_embedding /= np.linalg.norm(blend_embedding) + 1e-8

        elif blend_mode == "emergent":
            # Non-linear blending that can create emergent properties
            combined = np.stack(embeddings, axis=0)
            # Apply learned transformation
            blend_embedding = np.tanh(self.generator_weights @ combined.mean(axis=0))
            # Add some noise for emergence
            blend_embedding += np.random.randn(self.embedding_dim) * 0.1 * self.relaxation_level
            blend_embedding /= np.linalg.norm(blend_embedding) + 1e-8

        else:
            blend_embedding = sum(w * e for w, e in zip(weights, embeddings))

        # Compute emergent properties
        emergent = self._compute_emergent_properties(concepts, blend_embedding)

        # Compute novelty and coherence
        novelty = self._compute_novelty(blend_embedding)
        coherence = self._compute_coherence(blend_embedding)

        # Check thresholds
        if novelty < self.novelty_threshold or coherence < self.coherence_threshold:
            return None

        blend_id = f"blend_{self.blend_counter}"
        self.blend_counter += 1

        blend = Blend(
            blend_id=blend_id,
            source_concepts=[c.concept_id for c in concepts],
            blend_embedding=blend_embedding,
            emergent_properties=emergent,
            novelty_score=novelty,
            coherence_score=coherence
        )

        self.blends[blend_id] = blend
        self.successful_blends += 1

        # Learn from successful blend
        self._learn_from_blend(blend)

        return blend

    def imagine(
        self,
        initial_state: np.ndarray,
        steps: int = 10,
        constraints: Optional[Dict[str, Any]] = None,
        goal_state: Optional[np.ndarray] = None
    ) -> Imagination:
        """
        Run mental simulation from initial state.

        Args:
            initial_state: Starting state embedding
            steps: Number of simulation steps
            constraints: Constraints on the simulation
            goal_state: Optional goal state to work toward

        Returns:
            Imagination object with trajectory and evaluation
        """
        initial_state = np.asarray(initial_state).flatten()[:self.embedding_dim]
        if len(initial_state) < self.embedding_dim:
            initial_state = np.pad(initial_state, (0, self.embedding_dim - len(initial_state)))

        self.total_imaginations += 1

        trajectory = [initial_state.copy()]
        current_state = initial_state.copy()

        for step in range(steps):
            # Predict next state using generator
            next_state = self.generator_weights @ current_state + self.generator_bias
            next_state = np.tanh(next_state)

            # Add controlled noise for imagination
            noise = np.random.randn(self.embedding_dim) * 0.1 * (1 - step / steps)
            next_state += noise

            # Apply constraints
            if constraints:
                next_state = self._apply_constraints(next_state, constraints)

            # Goal-directed adjustment
            if goal_state is not None:
                goal_state = np.asarray(goal_state).flatten()[:self.embedding_dim]
                if len(goal_state) < self.embedding_dim:
                    goal_state = np.pad(goal_state, (0, self.embedding_dim - len(goal_state)))
                # Bias toward goal
                goal_direction = goal_state - current_state
                next_state += 0.1 * goal_direction

            next_state /= np.linalg.norm(next_state) + 1e-8
            trajectory.append(next_state.copy())
            current_state = next_state

        # Evaluate imagination
        plausibility = self._evaluate_plausibility(trajectory)
        novelty = self._compute_novelty(trajectory[-1])

        # Utility is higher if we reached goal
        if goal_state is not None:
            goal_distance = np.linalg.norm(trajectory[-1] - goal_state)
            utility = 1 / (1 + goal_distance)
        else:
            utility = novelty * plausibility

        imagination_id = f"imagination_{self.imagination_counter}"
        self.imagination_counter += 1

        imagination = Imagination(
            imagination_id=imagination_id,
            initial_state=initial_state,
            trajectory=trajectory,
            constraints=constraints or {},
            plausibility=plausibility,
            novelty=novelty,
            utility=utility
        )

        self.imaginations[imagination_id] = imagination

        return imagination

    def divergent_think(
        self,
        problem: np.ndarray,
        n_solutions: int = 10,
        diversity_weight: float = 0.5
    ) -> List[Tuple[np.ndarray, float]]:
        """
        Generate diverse solutions to a problem.

        Args:
            problem: Problem embedding
            n_solutions: Number of solutions to generate
            diversity_weight: Weight for diversity vs quality

        Returns:
            List of (solution, score) tuples
        """
        problem = np.asarray(problem).flatten()[:self.embedding_dim]
        if len(problem) < self.embedding_dim:
            problem = np.pad(problem, (0, self.embedding_dim - len(problem)))

        solutions = []
        solution_embeddings = []

        for i in range(n_solutions * 2):  # Generate extra for filtering
            # Start from problem and explore
            direction = np.random.randn(self.embedding_dim)
            direction /= np.linalg.norm(direction) + 1e-8

            # Different exploration distances
            distance = (i % 5 + 1) * 0.2
            solution = problem + distance * direction

            # Apply generator transformation
            solution = self.generator_weights @ solution + self.generator_bias
            solution = np.tanh(solution)
            solution /= np.linalg.norm(solution) + 1e-8

            # Compute quality (relevance to problem)
            relevance = np.dot(solution, problem)

            # Compute diversity from existing solutions
            if solution_embeddings:
                similarities = [np.dot(solution, s) for s in solution_embeddings]
                diversity = 1 - max(similarities)
            else:
                diversity = 1.0

            # Combined score
            score = (1 - diversity_weight) * relevance + diversity_weight * diversity

            solutions.append((solution, score, relevance, diversity))
            solution_embeddings.append(solution)

        # Select diverse, high-quality solutions
        solutions.sort(key=lambda x: x[1], reverse=True)
        selected = []
        selected_embeddings = []

        for solution, score, relevance, diversity in solutions:
            if len(selected) >= n_solutions:
                break

            # Check diversity from already selected
            if selected_embeddings:
                min_dist = min(1 - np.dot(solution, s) for s in selected_embeddings)
                if min_dist < 0.2:  # Too similar to existing
                    continue

            selected.append((solution, score))
            selected_embeddings.append(solution)

        return selected

    def _compute_novelty(self, embedding: np.ndarray) -> float:
        """Compute novelty of an embedding based on exposure history"""
        if len(self.exposure_history) == 0:
            return 1.0

        # Compare to recent exposures
        similarities = [np.dot(embedding, h) for h in self.exposure_history]
        max_similarity = max(similarities)

        # Novelty is inverse of similarity
        novelty = 1 - max_similarity

        # Also check against stored concepts
        if self.concepts:
            concept_sims = [np.dot(embedding, c.embedding)
                          for c in self.concepts.values()]
            max_concept_sim = max(concept_sims)
            novelty = min(novelty, 1 - max_concept_sim)

        return max(0.0, novelty)

    def _compute_coherence(self, embedding: np.ndarray) -> float:
        """Compute coherence of an embedding using learned discriminator"""
        # Use discriminator to assess coherence
        score = self.discriminator_weights.T @ embedding
        coherence = 1 / (1 + np.exp(-score[0]))  # Sigmoid

        return float(coherence)

    def _compute_emergent_properties(
        self,
        source_concepts: List[Concept],
        blend_embedding: np.ndarray
    ) -> Dict[str, Any]:
        """Compute emergent properties from a blend"""
        emergent = {}

        # Check for attribute combinations
        all_attributes = set()
        for concept in source_concepts:
            all_attributes.update(concept.attributes.keys())

        # Look for attributes that aren't in any source but emerge
        for attr in all_attributes:
            source_values = [c.attributes.get(attr) for c in source_concepts
                           if attr in c.attributes]
            if source_values:
                # Check if blend creates new value
                if all(isinstance(v, (int, float)) for v in source_values):
                    blend_value = np.mean(source_values) * np.random.uniform(0.8, 1.2)
                    if blend_value not in source_values:
                        emergent[attr] = blend_value

        # Compute abstraction level change
        source_abstractions = [c.abstraction_level for c in source_concepts]
        emergent['abstraction_level'] = np.mean(source_abstractions) + 0.1

        # Novel embedding component
        avg_source = np.mean([c.embedding for c in source_concepts], axis=0)
        emergent_component = blend_embedding - avg_source
        emergent['emergent_component_norm'] = float(np.linalg.norm(emergent_component))

        return emergent

    def _apply_constraints(
        self,
        embedding: np.ndarray,
        constraints: Dict[str, Any]
    ) -> np.ndarray:
        """Apply constraints to an embedding"""
        result = embedding.copy()

        # Similarity constraint
        if 'similar_to' in constraints:
            target = constraints['similar_to']
            target = np.asarray(target).flatten()[:self.embedding_dim]
            if len(target) < self.embedding_dim:
                target = np.pad(target, (0, self.embedding_dim - len(target)))
            min_sim = constraints.get('min_similarity', 0.5)
            current_sim = np.dot(result, target)
            if current_sim < min_sim:
                # Move toward target
                result = 0.7 * result + 0.3 * target
                result /= np.linalg.norm(result) + 1e-8

        # Dissimilarity constraint
        if 'different_from' in constraints:
            target = constraints['different_from']
            target = np.asarray(target).flatten()[:self.embedding_dim]
            if len(target) < self.embedding_dim:
                target = np.pad(target, (0, self.embedding_dim - len(target)))
            max_sim = constraints.get('max_similarity', 0.3)
            current_sim = np.dot(result, target)
            if current_sim > max_sim:
                # Move away from target
                orthogonal = result - current_sim * target
                orthogonal /= np.linalg.norm(orthogonal) + 1e-8
                result = orthogonal

        # Norm constraint (always applied)
        result /= np.linalg.norm(result) + 1e-8

        return result

    def _evaluate_plausibility(self, trajectory: List[np.ndarray]) -> float:
        """Evaluate plausibility of an imagined trajectory"""
        if len(trajectory) < 2:
            return 1.0

        # Check smoothness (sudden jumps are implausible)
        jumps = []
        for i in range(1, len(trajectory)):
            jump = np.linalg.norm(trajectory[i] - trajectory[i-1])
            jumps.append(jump)

        avg_jump = np.mean(jumps)
        max_jump = max(jumps)

        # Smoothness score
        smoothness = 1 / (1 + max_jump)

        # Check coherence of final state
        final_coherence = self._compute_coherence(trajectory[-1])

        # Combined plausibility
        plausibility = 0.5 * smoothness + 0.5 * final_coherence

        return float(plausibility)

    def _learn_from_blend(self, blend: Blend):
        """Learn from successful blend to improve future blending"""
        # Hebbian update: strengthen connections between source concepts
        source_indices = []
        for cid in blend.source_concepts:
            if cid in self.concepts:
                # Use concept counter as proxy for index
                idx = int(cid.split('_')[1]) % self.max_concepts
                source_indices.append(idx)

        for i in source_indices:
            for j in source_indices:
                if i != j:
                    self.association_matrix[i, j] += self.learning_rate * blend.novelty_score

        # Update generator to produce similar outputs
        target = blend.blend_embedding
        avg_source = np.mean([self.concepts[cid].embedding
                             for cid in blend.source_concepts
                             if cid in self.concepts], axis=0)

        prediction = self.generator_weights @ avg_source + self.generator_bias
        prediction = np.tanh(prediction)

        error = target - prediction

        # Hebbian update
        self.generator_weights += self.learning_rate * np.outer(error, avg_source)
        self.generator_bias += self.learning_rate * error

    def set_relaxation_level(self, level: float):
        """Set constraint relaxation level (0-1)"""
        self.relaxation_level = max(0.0, min(1.0, level))

    def set_temperature(self, temperature: float):
        """Set generation temperature (higher = more random)"""
        self.temperature = max(0.1, temperature)

    def get_stats(self) -> Dict[str, Any]:
        """Get creativity engine statistics"""
        return {
            'total_concepts': len(self.concepts),
            'total_blends': len(self.blends),
            'total_imaginations': len(self.imaginations),
            'total_generations': self.total_generations,
            'successful_blends': self.successful_blends,
            'relaxation_level': self.relaxation_level,
            'temperature': self.temperature,
            'novelty_threshold': self.novelty_threshold,
            'coherence_threshold': self.coherence_threshold,
            'exposure_history_size': len(self.exposure_history)
        }

    def serialize(self) -> Dict[str, Any]:
        """Serialize creativity engine to dictionary"""
        return {
            'embedding_dim': self.embedding_dim,
            'max_concepts': self.max_concepts,
            'novelty_threshold': self.novelty_threshold,
            'coherence_threshold': self.coherence_threshold,
            'temperature': self.temperature,
            'learning_rate': self.learning_rate,
            'relaxation_level': self.relaxation_level,
            'concepts': {
                cid: {
                    'concept_id': c.concept_id,
                    'name': c.name,
                    'embedding': c.embedding.tolist(),
                    'attributes': c.attributes,
                    'abstraction_level': c.abstraction_level
                }
                for cid, c in self.concepts.items()
            },
            'blends': {
                bid: {
                    'blend_id': b.blend_id,
                    'source_concepts': b.source_concepts,
                    'blend_embedding': b.blend_embedding.tolist(),
                    'novelty_score': b.novelty_score,
                    'coherence_score': b.coherence_score
                }
                for bid, b in self.blends.items()
            },
            'generator_weights': self.generator_weights.tolist(),
            'generator_bias': self.generator_bias.tolist(),
            'stats': self.get_stats()
        }

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> 'CreativityEngine':
        """Deserialize creativity engine from dictionary"""
        engine = cls(
            embedding_dim=data['embedding_dim'],
            max_concepts=data['max_concepts'],
            novelty_threshold=data['novelty_threshold'],
            coherence_threshold=data['coherence_threshold'],
            temperature=data['temperature'],
            learning_rate=data['learning_rate']
        )

        engine.relaxation_level = data.get('relaxation_level', 0.0)
        engine.generator_weights = np.array(data['generator_weights'])
        engine.generator_bias = np.array(data['generator_bias'])

        for cid, cdata in data.get('concepts', {}).items():
            concept = Concept(
                concept_id=cdata['concept_id'],
                name=cdata['name'],
                embedding=np.array(cdata['embedding']),
                attributes=cdata['attributes'],
                abstraction_level=cdata['abstraction_level']
            )
            engine.concepts[cid] = concept

        for bid, bdata in data.get('blends', {}).items():
            blend = Blend(
                blend_id=bdata['blend_id'],
                source_concepts=bdata['source_concepts'],
                blend_embedding=np.array(bdata['blend_embedding']),
                emergent_properties={},
                novelty_score=bdata['novelty_score'],
                coherence_score=bdata['coherence_score']
            )
            engine.blends[bid] = blend

        return engine
