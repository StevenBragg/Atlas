#!/usr/bin/env python3
"""
World Model System for ATLAS Superintelligence

This module implements world models that learn physics, object permanence,
and causal relationships through local plasticity rules. It includes:

1. WorldModel - Physics-based object tracking with permanence
2. CausalWorldModel - Causal reasoning with intervention support

Core Principles:
- No backpropagation - uses local Hebbian/STDP learning
- Biologically plausible - inspired by hippocampal place cells and grid cells
- Predictive coding - generates predictions and learns from errors
- Self-organizing - structure emerges from data statistics
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union, Set, Callable
from enum import Enum
from collections import deque, defaultdict
import logging
import time

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False

logger = logging.getLogger(__name__)


class ObjectState(Enum):
    """States an object can be in"""
    VISIBLE = "visible"
    OCCLUDED = "occluded"
    PREDICTED = "predicted"
    UNKNOWN = "unknown"


class PhysicsProperty(Enum):
    """Physical properties that can be learned"""
    POSITION = "position"
    VELOCITY = "velocity"
    ACCELERATION = "acceleration"
    MASS = "mass"
    SIZE = "size"
    SHAPE = "shape"
    COLOR = "color"
    TEXTURE = "texture"


class CausalRelationType(Enum):
    """Types of causal relationships"""
    CAUSES = "causes"  # X causes Y
    PREVENTS = "prevents"  # X prevents Y
    ENABLES = "enables"  # X enables Y
    REQUIRES = "requires"  # X requires Y
    CORRELATES = "correlates"  # X correlates with Y (not necessarily causal)


@dataclass
class WorldObject:
    """Representation of an object in the world model"""
    object_id: str
    embedding: np.ndarray  # Learned representation
    position: np.ndarray  # 3D position estimate
    velocity: np.ndarray  # 3D velocity estimate
    properties: Dict[str, Any] = field(default_factory=dict)
    state: ObjectState = ObjectState.VISIBLE
    confidence: float = 1.0
    last_seen: float = 0.0
    permanence_strength: float = 0.0  # How strongly we believe object persists
    creation_time: float = field(default_factory=time.time)
    update_count: int = 0

    def update(self, new_embedding: np.ndarray, new_position: np.ndarray,
               learning_rate: float = 0.1):
        """Update object representation with new observation"""
        # Hebbian update of embedding
        self.embedding = (1 - learning_rate) * self.embedding + learning_rate * new_embedding
        self.embedding /= np.linalg.norm(self.embedding) + 1e-8

        # Update velocity from position change
        if self.update_count > 0:
            dt = 1.0  # Assume unit time step
            self.velocity = 0.8 * self.velocity + 0.2 * (new_position - self.position) / dt

        self.position = new_position
        self.last_seen = time.time()
        self.state = ObjectState.VISIBLE
        self.confidence = min(1.0, self.confidence + 0.1)
        self.permanence_strength = min(1.0, self.permanence_strength + 0.05)
        self.update_count += 1


@dataclass
class PhysicsModel:
    """Simple physics model learned from observations"""
    gravity: np.ndarray = field(default_factory=lambda: np.array([0.0, -9.8, 0.0]))
    friction: float = 0.1
    collision_elasticity: float = 0.7
    learned_forces: Dict[str, np.ndarray] = field(default_factory=dict)
    transition_matrix: Optional[np.ndarray] = None  # Learned state transition

    def predict_next_state(self, position: np.ndarray, velocity: np.ndarray,
                           dt: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """Predict next position and velocity using learned physics"""
        # Simple physics integration
        acceleration = self.gravity - self.friction * velocity

        new_velocity = velocity + acceleration * dt
        new_position = position + velocity * dt + 0.5 * acceleration * dt * dt

        return new_position, new_velocity


@dataclass
class StateTransition:
    """Record of a state transition for learning"""
    state_before: np.ndarray
    action: Optional[np.ndarray]
    state_after: np.ndarray
    timestamp: float
    prediction_error: float = 0.0


@dataclass
class Variable:
    """Represents a variable in the causal world model"""
    name: str
    value: Any
    observable: bool = True  # Whether directly observable
    continuous: bool = False  # Continuous vs discrete
    prior_distribution: Optional[Dict[Any, float]] = None


@dataclass
class CausalEdge:
    """Represents a causal relationship between variables"""
    cause: str
    effect: str
    relation_type: CausalRelationType
    strength: float = 1.0  # Causal strength
    mechanism: Optional[Callable] = None  # Function mapping cause to effect
    learned_from_intervention: bool = False


@dataclass
class CausalObject:
    """Represents a persistent object in the causal world model"""
    object_id: str
    object_type: str
    position: np.ndarray  # Spatial location
    velocity: np.ndarray  # Movement vector
    properties: Dict[str, Any]
    last_observed: float
    existence_confidence: float = 1.0
    permanence: bool = True  # Object continues to exist when not observed


class WorldModel:
    """
    Comprehensive world model that learns object permanence, physics,
    and state dynamics through local plasticity rules.
    """

    def __init__(
        self,
        state_dim: int = 64,
        max_objects: int = 100,
        permanence_decay: float = 0.01,
        prediction_horizon: int = 10,
        learning_rate: float = 0.05,
        random_seed: Optional[int] = None
    ):
        """
        Initialize the world model.

        Args:
            state_dim: Dimension of state embeddings
            max_objects: Maximum number of objects to track
            permanence_decay: Rate at which occluded object confidence decays
            prediction_horizon: How many steps to predict ahead
            learning_rate: Learning rate for plasticity updates
            random_seed: Random seed for reproducibility
        """
        self.state_dim = state_dim
        self.max_objects = max_objects
        self.permanence_decay = permanence_decay
        self.prediction_horizon = prediction_horizon
        self.learning_rate = learning_rate

        if random_seed is not None:
            np.random.seed(random_seed)

        # Object tracking
        self.objects: Dict[str, WorldObject] = {}
        self.object_counter = 0

        # Physics model
        self.physics = PhysicsModel()

        # State transition learning
        self.state_transitions: deque = deque(maxlen=1000)

        # Transition model: W such that state_next ≈ W @ state_current
        self.transition_weights = np.random.randn(state_dim, state_dim) * 0.01
        # Make it stable (spectral radius < 1)
        self.transition_weights /= (np.linalg.norm(self.transition_weights) + 0.1)

        # Action-conditioned transition (for interventions)
        self.action_dim = 16
        self.action_weights = np.random.randn(state_dim, self.action_dim) * 0.01

        # Prediction error tracking
        self.prediction_errors: deque = deque(maxlen=1000)
        self.total_predictions = 0
        self.total_surprises = 0

        # Hidden state inference (simple Kalman-like)
        self.state_covariance = np.eye(state_dim) * 0.1
        self.observation_noise = 0.1
        self.process_noise = 0.05

        # Statistics
        self.total_observations = 0
        self.objects_created = 0
        self.objects_removed = 0

    def observe(
        self,
        observations: List[Dict[str, Any]],
        global_state: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Process observations and update world model.

        Args:
            observations: List of observed objects with embeddings and positions
            global_state: Optional global scene embedding

        Returns:
            Dictionary with updates, predictions, and surprises
        """
        self.total_observations += 1
        current_time = time.time()

        updates = []
        new_objects = []
        matched_ids = set()

        for obs in observations:
            embedding = obs.get('embedding', np.random.randn(self.state_dim))
            position = obs.get('position', np.zeros(3))
            properties = obs.get('properties', {})

            # Ensure proper shapes
            embedding = np.asarray(embedding).flatten()[:self.state_dim]
            if len(embedding) < self.state_dim:
                embedding = np.pad(embedding, (0, self.state_dim - len(embedding)))
            embedding /= np.linalg.norm(embedding) + 1e-8
            position = np.asarray(position).flatten()[:3]
            if len(position) < 3:
                position = np.pad(position, (0, 3 - len(position)))

            # Try to match with existing object
            best_match = None
            best_score = 0.5  # Threshold for matching

            for obj_id, obj in self.objects.items():
                if obj_id in matched_ids:
                    continue

                # Compute similarity (Hebbian-inspired matching)
                embedding_sim = np.dot(embedding, obj.embedding)
                position_dist = np.linalg.norm(position - obj.position)

                # Predict where object should be
                if obj.state == ObjectState.OCCLUDED:
                    predicted_pos, _ = self.physics.predict_next_state(
                        obj.position, obj.velocity,
                        dt=current_time - obj.last_seen
                    )
                    position_dist = min(position_dist,
                                       np.linalg.norm(position - predicted_pos))

                # Combined score
                score = embedding_sim * np.exp(-position_dist / 10.0)

                if score > best_score:
                    best_score = score
                    best_match = obj_id

            if best_match:
                # Update existing object
                obj = self.objects[best_match]
                obj.update(embedding, position, self.learning_rate)
                obj.properties.update(properties)
                matched_ids.add(best_match)
                updates.append({'id': best_match, 'type': 'update'})
            else:
                # Create new object
                new_id = self._create_object(embedding, position, properties)
                new_objects.append(new_id)
                updates.append({'id': new_id, 'type': 'new'})

        # Handle objects that weren't observed (occlusion)
        for obj_id, obj in list(self.objects.items()):
            if obj_id not in matched_ids and obj_id not in new_objects:
                if obj.state == ObjectState.VISIBLE:
                    # Object became occluded
                    obj.state = ObjectState.OCCLUDED

                # Decay confidence
                time_occluded = current_time - obj.last_seen
                obj.confidence *= np.exp(-self.permanence_decay * time_occluded)

                # Predict position while occluded
                obj.position, obj.velocity = self.physics.predict_next_state(
                    obj.position, obj.velocity, dt=1.0
                )

                # Remove if confidence too low and not strongly permanent
                if obj.confidence < 0.1 and obj.permanence_strength < 0.5:
                    del self.objects[obj_id]
                    self.objects_removed += 1

        # Learn state transitions if we have global state
        if global_state is not None:
            self._learn_transition(global_state)

        # Generate predictions
        predictions = self._generate_predictions()

        return {
            'updates': updates,
            'new_objects': new_objects,
            'occluded': [oid for oid, o in self.objects.items()
                        if o.state == ObjectState.OCCLUDED],
            'predictions': predictions,
            'total_objects': len(self.objects)
        }

    def _create_object(
        self,
        embedding: np.ndarray,
        position: np.ndarray,
        properties: Dict[str, Any]
    ) -> str:
        """Create a new object in the world model"""
        if len(self.objects) >= self.max_objects:
            # Remove least confident object
            min_conf = min(self.objects.values(), key=lambda x: x.confidence)
            del self.objects[min_conf.object_id]
            self.objects_removed += 1

        obj_id = f"obj_{self.object_counter}"
        self.object_counter += 1

        obj = WorldObject(
            object_id=obj_id,
            embedding=embedding.copy(),
            position=position.copy(),
            velocity=np.zeros(3),
            properties=properties,
            state=ObjectState.VISIBLE,
            confidence=0.5,  # Start with moderate confidence
            last_seen=time.time()
        )

        self.objects[obj_id] = obj
        self.objects_created += 1

        return obj_id

    def _learn_transition(self, current_state: np.ndarray):
        """Learn state transitions using local plasticity"""
        current_state = np.asarray(current_state).flatten()[:self.state_dim]
        if len(current_state) < self.state_dim:
            current_state = np.pad(current_state, (0, self.state_dim - len(current_state)))

        if len(self.state_transitions) > 0:
            last_transition = self.state_transitions[-1]
            prev_state = last_transition.state_after

            # Predict current state from previous
            predicted_state = self.transition_weights @ prev_state

            # Compute prediction error
            error = current_state - predicted_state
            prediction_error = np.mean(error ** 2)

            self.prediction_errors.append(prediction_error)
            self.total_predictions += 1

            if prediction_error > 0.5:  # Surprise threshold
                self.total_surprises += 1

            # Hebbian update of transition weights
            # Δw = η * error * prev_state^T (delta rule with local signals)
            delta_w = self.learning_rate * np.outer(error, prev_state)
            self.transition_weights += delta_w

            # Normalize to maintain stability
            norm = np.linalg.norm(self.transition_weights)
            if norm > 1.0:
                self.transition_weights /= norm

        # Store transition
        transition = StateTransition(
            state_before=current_state.copy() if len(self.state_transitions) == 0
                        else self.state_transitions[-1].state_after.copy(),
            action=None,
            state_after=current_state.copy(),
            timestamp=time.time()
        )
        self.state_transitions.append(transition)

    def _generate_predictions(self) -> List[Dict[str, Any]]:
        """Generate predictions about future states"""
        predictions = []

        # Predict object positions
        for obj_id, obj in self.objects.items():
            future_positions = []
            pos, vel = obj.position.copy(), obj.velocity.copy()

            for t in range(1, self.prediction_horizon + 1):
                pos, vel = self.physics.predict_next_state(pos, vel, dt=1.0)
                future_positions.append({
                    'timestep': t,
                    'position': pos.copy(),
                    'velocity': vel.copy(),
                    'confidence': obj.confidence * (0.9 ** t)
                })

            predictions.append({
                'object_id': obj_id,
                'current_state': obj.state.value,
                'future_positions': future_positions
            })

        # Predict global state if we have enough history
        if len(self.state_transitions) > 0:
            current_state = self.state_transitions[-1].state_after
            predicted_states = []
            state = current_state.copy()

            for t in range(1, min(5, self.prediction_horizon + 1)):
                state = self.transition_weights @ state
                predicted_states.append({
                    'timestep': t,
                    'state': state.copy(),
                    'confidence': 0.9 ** t
                })

            predictions.append({
                'type': 'global_state',
                'predicted_states': predicted_states
            })

        return predictions

    def predict_with_intervention(
        self,
        action: np.ndarray,
        steps: int = 5
    ) -> List[np.ndarray]:
        """
        Predict future states given an intervention/action.
        Implements do-calculus for world modeling.
        """
        action = np.asarray(action).flatten()[:self.action_dim]
        if len(action) < self.action_dim:
            action = np.pad(action, (0, self.action_dim - len(action)))

        if len(self.state_transitions) == 0:
            return []

        current_state = self.state_transitions[-1].state_after.copy()
        predictions = []

        for t in range(steps):
            # State transition with action influence
            action_effect = self.action_weights @ action
            next_state = self.transition_weights @ current_state + action_effect

            # Normalize
            next_state /= np.linalg.norm(next_state) + 1e-8

            predictions.append(next_state.copy())
            current_state = next_state

        return predictions

    def query_object_permanence(self, object_id: str) -> Dict[str, Any]:
        """Query the permanence status of an object"""
        if object_id not in self.objects:
            return {
                'exists': False,
                'object_id': object_id,
                'message': 'Object not in world model'
            }

        obj = self.objects[object_id]

        return {
            'exists': True,
            'object_id': object_id,
            'state': obj.state.value,
            'confidence': obj.confidence,
            'permanence_strength': obj.permanence_strength,
            'position': obj.position.tolist(),
            'velocity': obj.velocity.tolist(),
            'time_since_seen': time.time() - obj.last_seen,
            'predicted_position': self.physics.predict_next_state(
                obj.position, obj.velocity,
                dt=time.time() - obj.last_seen
            )[0].tolist() if obj.state == ObjectState.OCCLUDED else obj.position.tolist()
        }

    def infer_hidden_state(
        self,
        observations: np.ndarray,
        prior_state: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Infer hidden state from noisy observations using Bayesian update.
        Returns estimated state and uncertainty.
        """
        observations = np.asarray(observations).flatten()[:self.state_dim]
        if len(observations) < self.state_dim:
            observations = np.pad(observations, (0, self.state_dim - len(observations)))

        if prior_state is None:
            if len(self.state_transitions) > 0:
                prior_state = self.state_transitions[-1].state_after
            else:
                prior_state = np.zeros(self.state_dim)

        prior_state = np.asarray(prior_state).flatten()[:self.state_dim]
        if len(prior_state) < self.state_dim:
            prior_state = np.pad(prior_state, (0, self.state_dim - len(prior_state)))

        # Simple Kalman-like update
        # Predict step
        predicted_state = self.transition_weights @ prior_state
        predicted_cov = (self.transition_weights @ self.state_covariance @
                        self.transition_weights.T + np.eye(self.state_dim) * self.process_noise)

        # Update step
        kalman_gain = predicted_cov / (predicted_cov +
                                       np.eye(self.state_dim) * self.observation_noise)

        innovation = observations - predicted_state
        estimated_state = predicted_state + kalman_gain @ innovation

        # Update covariance
        self.state_covariance = (np.eye(self.state_dim) - kalman_gain) @ predicted_cov

        # Uncertainty estimate (diagonal of covariance)
        uncertainty = np.diag(self.state_covariance)

        return estimated_state, uncertainty

    def learn_physics(self, trajectory: List[Dict[str, np.ndarray]]):
        """
        Learn physics parameters from observed trajectories.
        Uses local correlation-based learning.
        """
        if len(trajectory) < 3:
            return

        accelerations = []
        velocities = []

        for i in range(1, len(trajectory) - 1):
            pos_prev = np.asarray(trajectory[i-1].get('position', np.zeros(3)))
            pos_curr = np.asarray(trajectory[i].get('position', np.zeros(3)))
            pos_next = np.asarray(trajectory[i+1].get('position', np.zeros(3)))

            vel_prev = pos_curr - pos_prev
            vel_next = pos_next - pos_curr

            acc = vel_next - vel_prev

            velocities.append(vel_prev)
            accelerations.append(acc)

        if len(accelerations) > 0:
            # Estimate gravity (constant acceleration component)
            mean_acc = np.mean(accelerations, axis=0)

            # Hebbian-like update
            self.physics.gravity = (0.9 * self.physics.gravity +
                                   0.1 * mean_acc)

            # Estimate friction (velocity-dependent deceleration)
            velocities_arr = np.array(velocities)
            accelerations_arr = np.array(accelerations)

            # Simple correlation to estimate friction
            vel_norms = np.linalg.norm(velocities_arr, axis=1, keepdims=True) + 1e-8
            acc_in_vel_dir = np.sum(accelerations_arr * velocities_arr, axis=1) / vel_norms.flatten()

            mean_decel = -np.mean(acc_in_vel_dir)
            if mean_decel > 0:
                self.physics.friction = 0.9 * self.physics.friction + 0.1 * mean_decel

    def get_world_state_embedding(self) -> np.ndarray:
        """Get a unified embedding representing the entire world state"""
        if len(self.objects) == 0:
            return np.zeros(self.state_dim)

        # Aggregate object embeddings weighted by confidence
        total_weight = 0.0
        weighted_sum = np.zeros(self.state_dim)

        for obj in self.objects.values():
            weight = obj.confidence * obj.permanence_strength
            weighted_sum += weight * obj.embedding
            total_weight += weight

        if total_weight > 0:
            world_embedding = weighted_sum / total_weight
        else:
            world_embedding = np.zeros(self.state_dim)

        # Combine with transition state if available
        if len(self.state_transitions) > 0:
            transition_state = self.state_transitions[-1].state_after
            world_embedding = 0.5 * world_embedding + 0.5 * transition_state

        world_embedding /= np.linalg.norm(world_embedding) + 1e-8

        return world_embedding

    def get_stats(self) -> Dict[str, Any]:
        """Get world model statistics"""
        return {
            'total_objects': len(self.objects),
            'visible_objects': sum(1 for o in self.objects.values()
                                  if o.state == ObjectState.VISIBLE),
            'occluded_objects': sum(1 for o in self.objects.values()
                                   if o.state == ObjectState.OCCLUDED),
            'total_observations': self.total_observations,
            'objects_created': self.objects_created,
            'objects_removed': self.objects_removed,
            'total_predictions': self.total_predictions,
            'total_surprises': self.total_surprises,
            'surprise_rate': self.total_surprises / max(1, self.total_predictions),
            'mean_prediction_error': np.mean(self.prediction_errors) if self.prediction_errors else 0.0,
            'physics_gravity': self.physics.gravity.tolist(),
            'physics_friction': self.physics.friction,
            'transition_matrix_norm': np.linalg.norm(self.transition_weights)
        }

    def serialize(self) -> Dict[str, Any]:
        """Serialize world model to dictionary"""
        return {
            'state_dim': self.state_dim,
            'max_objects': self.max_objects,
            'permanence_decay': self.permanence_decay,
            'prediction_horizon': self.prediction_horizon,
            'learning_rate': self.learning_rate,
            'objects': {
                oid: {
                    'object_id': obj.object_id,
                    'embedding': obj.embedding.tolist(),
                    'position': obj.position.tolist(),
                    'velocity': obj.velocity.tolist(),
                    'properties': obj.properties,
                    'state': obj.state.value,
                    'confidence': obj.confidence,
                    'permanence_strength': obj.permanence_strength
                }
                for oid, obj in self.objects.items()
            },
            'transition_weights': self.transition_weights.tolist(),
            'action_weights': self.action_weights.tolist(),
            'physics': {
                'gravity': self.physics.gravity.tolist(),
                'friction': self.physics.friction,
                'collision_elasticity': self.physics.collision_elasticity
            },
            'stats': self.get_stats()
        }

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> 'WorldModel':
        """Deserialize world model from dictionary"""
        model = cls(
            state_dim=data['state_dim'],
            max_objects=data['max_objects'],
            permanence_decay=data['permanence_decay'],
            prediction_horizon=data['prediction_horizon'],
            learning_rate=data['learning_rate']
        )

        model.transition_weights = np.array(data['transition_weights'])
        model.action_weights = np.array(data['action_weights'])

        if 'physics' in data:
            model.physics.gravity = np.array(data['physics']['gravity'])
            model.physics.friction = data['physics']['friction']
            model.physics.collision_elasticity = data['physics']['collision_elasticity']

        for oid, obj_data in data.get('objects', {}).items():
            obj = WorldObject(
                object_id=obj_data['object_id'],
                embedding=np.array(obj_data['embedding']),
                position=np.array(obj_data['position']),
                velocity=np.array(obj_data['velocity']),
                properties=obj_data['properties'],
                state=ObjectState(obj_data['state']),
                confidence=obj_data['confidence'],
                permanence_strength=obj_data['permanence_strength']
            )
            model.objects[oid] = obj

        return model


class CausalWorldModel:
    """
    Predictive world model with causal reasoning capabilities.

    Capabilities:
    - Causal graph learning from observations and interventions
    - Forward prediction using learned causal mechanisms
    - Counterfactual reasoning (what would happen if...)
    - Object tracking with permanence
    - Physics simulation for basic dynamics
    - Hidden state inference
    """

    def __init__(
        self,
        state_dim: int,
        max_objects: int = 100,
        enable_physics: bool = True,
        enable_counterfactuals: bool = True,
        causal_threshold: float = 0.3,
        intervention_learning_rate: float = 0.1,
        prediction_horizon: int = 10,
        object_persistence_threshold: float = 0.5,
        random_seed: Optional[int] = None,
    ):
        """
        Initialize causal world model.

        Args:
            state_dim: Dimension of world state vectors
            max_objects: Maximum number of tracked objects
            enable_physics: Whether to use physics simulation
            enable_counterfactuals: Whether to support counterfactual reasoning
            causal_threshold: Minimum strength for causal relationships
            intervention_learning_rate: Rate of learning from interventions
            prediction_horizon: Maximum forward prediction steps
            object_persistence_threshold: Confidence threshold for object persistence
            random_seed: Random seed for reproducibility
        """
        if not HAS_NETWORKX:
            raise ImportError("CausalWorldModel requires networkx. Install with: pip install networkx")

        if random_seed is not None:
            np.random.seed(random_seed)

        self.state_dim = state_dim
        self.max_objects = max_objects
        self.enable_physics = enable_physics
        self.enable_counterfactuals = enable_counterfactuals
        self.causal_threshold = causal_threshold
        self.intervention_learning_rate = intervention_learning_rate
        self.prediction_horizon = prediction_horizon
        self.object_persistence_threshold = object_persistence_threshold

        # Causal graph
        self.causal_graph = nx.DiGraph()
        self.variables: Dict[str, Variable] = {}

        # Object tracking
        self.objects: Dict[str, CausalObject] = {}
        self.object_id_counter = 0

        # State history for learning
        self.state_history: List[Dict[str, Any]] = []
        self.intervention_history: List[Dict[str, Any]] = []

        # Learned causal mechanisms
        self.causal_mechanisms: Dict[Tuple[str, str], Callable] = {}

        # Physics parameters
        self.gravity = np.array([0.0, -9.8, 0.0]) if enable_physics else None
        self.friction = 0.1
        self.elasticity = 0.8

        # Hidden state inference
        self.hidden_states: Dict[str, np.ndarray] = {}

        # Statistics
        self.total_predictions = 0
        self.prediction_errors: List[float] = []
        self.total_interventions = 0
        self.causal_discoveries = 0

        logger.info(f"Initialized causal world model: state_dim={state_dim}")

    def add_variable(
        self,
        name: str,
        initial_value: Any = None,
        observable: bool = True,
        continuous: bool = False,
    ) -> Variable:
        """Add a variable to the world model."""
        variable = Variable(
            name=name,
            value=initial_value,
            observable=observable,
            continuous=continuous,
        )

        self.variables[name] = variable
        self.causal_graph.add_node(name, variable=variable)

        logger.debug(f"Added variable: {name}")
        return variable

    def add_causal_relation(
        self,
        cause: str,
        effect: str,
        relation_type: CausalRelationType,
        strength: float = 1.0,
        mechanism: Optional[Callable] = None,
    ) -> Optional[CausalEdge]:
        """Add a causal relationship between variables."""
        if cause not in self.variables or effect not in self.variables:
            logger.warning(f"Variables not found: {cause}, {effect}")
            return None

        edge = CausalEdge(
            cause=cause,
            effect=effect,
            relation_type=relation_type,
            strength=strength,
            mechanism=mechanism,
        )

        self.causal_graph.add_edge(
            cause, effect,
            edge=edge,
            type=relation_type,
            weight=strength,
        )

        if mechanism:
            self.causal_mechanisms[(cause, effect)] = mechanism

        self.causal_discoveries += 1
        logger.debug(f"Added causal relation: {cause} -{relation_type.value}-> {effect}")

        return edge

    def observe(
        self,
        observations: Dict[str, Any],
        timestamp: Optional[float] = None,
    ) -> None:
        """
        Update world model with new observations.

        Args:
            observations: Dictionary of observed variable values
            timestamp: Time of observation
        """
        if timestamp is None:
            timestamp = time.time()

        # Update variable values
        for var_name, value in observations.items():
            if var_name in self.variables:
                self.variables[var_name].value = value
            else:
                # Auto-create observable variables
                self.add_variable(var_name, initial_value=value, observable=True)

        # Store in history
        obs_record = observations.copy()
        obs_record['timestamp'] = timestamp
        self.state_history.append(obs_record)

        # Limit history size
        if len(self.state_history) > 1000:
            self.state_history.pop(0)

        # Learn causal relationships from temporal patterns
        if len(self.state_history) >= 2:
            self._learn_causal_relationships()

    def intervene(
        self,
        variable: str,
        value: Any,
    ) -> Dict[str, Any]:
        """
        Perform an intervention (do-operation) and observe effects.

        This is critical for learning true causal relationships
        vs mere correlations.

        Args:
            variable: Variable to intervene on
            value: Value to set

        Returns:
            Resulting state after intervention
        """
        if variable not in self.variables:
            logger.warning(f"Variable {variable} not found for intervention")
            return {}

        # Record intervention
        intervention = {
            'variable': variable,
            'value': value,
            'timestamp': time.time(),
            'pre_state': {v: self.variables[v].value for v in self.variables},
        }

        # Set variable value (breaking incoming causal links)
        self.variables[variable].value = value

        # Propagate causal effects forward
        affected_state = self._propagate_causation(variable, value)

        # Record post-intervention state
        intervention['post_state'] = affected_state
        self.intervention_history.append(intervention)

        # Learn from intervention
        self._learn_from_intervention(intervention)

        self.total_interventions += 1
        logger.debug(f"Intervention on {variable} = {value}")

        return affected_state

    def predict(
        self,
        horizon: int = 1,
        initial_state: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Predict future states using the causal model.

        Args:
            horizon: Number of steps to predict forward
            initial_state: Starting state (None for current)

        Returns:
            List of predicted states
        """
        if initial_state is None:
            # Use current state
            current_state = {name: var.value for name, var in self.variables.items()}
        else:
            current_state = initial_state.copy()

        predictions = []
        state = current_state.copy()

        for step in range(min(horizon, self.prediction_horizon)):
            # Apply causal mechanisms to predict next state
            next_state = state.copy()

            # Topological sort to respect causal order
            if len(self.causal_graph.nodes()) > 0:
                try:
                    causal_order = list(nx.topological_sort(self.causal_graph))

                    for var_name in causal_order:
                        # Get all causes of this variable
                        causes = list(self.causal_graph.predecessors(var_name))

                        if causes:
                            # Apply causal mechanism
                            predicted_value = self._apply_causal_mechanism(
                                var_name, causes, state
                            )
                            next_state[var_name] = predicted_value

                except nx.NetworkXError:
                    # Graph has cycles, use approximate method
                    logger.warning("Causal graph has cycles, using approximate prediction")

            # Apply physics if enabled
            if self.enable_physics:
                next_state = self._apply_physics(next_state, step)

            predictions.append(next_state)
            state = next_state

        self.total_predictions += len(predictions)
        return predictions

    def counterfactual(
        self,
        variable: str,
        counterfactual_value: Any,
        actual_observations: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Perform counterfactual reasoning: "What would have happened if..."

        Args:
            variable: Variable to change in counterfactual
            counterfactual_value: Alternative value
            actual_observations: What actually happened

        Returns:
            Predicted counterfactual outcome
        """
        if not self.enable_counterfactuals:
            logger.warning("Counterfactual reasoning not enabled")
            return {}

        # Step 1: Abduction - infer hidden/unobserved variables from actual observations
        hidden = self._infer_hidden_states(actual_observations)

        # Step 2: Action - set the counterfactual variable value
        counterfactual_state = actual_observations.copy()
        counterfactual_state[variable] = counterfactual_value

        # Step 3: Prediction - forward simulate with new value
        # Propagate through causal graph, keeping hidden variables fixed
        result = self._propagate_causation(variable, counterfactual_value, hidden)

        logger.debug(f"Counterfactual: {variable}={counterfactual_value}")
        return result

    def track_object(
        self,
        position: np.ndarray,
        properties: Dict[str, Any],
        object_type: str = "unknown",
    ) -> CausalObject:
        """
        Track an object in the world with persistence.

        Args:
            position: Spatial location
            properties: Object properties
            object_type: Type classification

        Returns:
            Tracked object
        """
        # Check if this is an existing object
        existing = self._find_nearest_object(position, object_type)

        if existing and np.linalg.norm(existing.position - position) < 0.5:
            # Update existing object
            existing.position = position
            existing.properties.update(properties)
            existing.last_observed = time.time()
            existing.existence_confidence = min(1.0, existing.existence_confidence + 0.1)
            return existing
        else:
            # Create new object
            obj = CausalObject(
                object_id=f"obj_{self.object_id_counter}",
                object_type=object_type,
                position=position.copy(),
                velocity=np.zeros(len(position)),
                properties=properties,
                last_observed=time.time(),
            )

            self.objects[obj.object_id] = obj
            self.object_id_counter += 1

            # Limit number of objects
            if len(self.objects) > self.max_objects:
                self._prune_objects()

            logger.debug(f"Tracking new object: {obj.object_id}")
            return obj

    def _learn_causal_relationships(self) -> None:
        """Learn causal relationships from temporal patterns in observations."""
        if len(self.state_history) < 2:
            return

        # Compare consecutive states to find temporal dependencies
        recent_states = self.state_history[-10:]

        for i in range(len(recent_states) - 1):
            prev_state = recent_states[i]
            next_state = recent_states[i + 1]

            # Find variables that changed
            for var_name in prev_state:
                if var_name == 'timestamp':
                    continue

                if var_name not in next_state:
                    continue

                # Check if other variables predicted this change
                for potential_cause in prev_state:
                    if potential_cause == 'timestamp' or potential_cause == var_name:
                        continue

                    # Simple heuristic: if cause changed and effect changed, might be causal
                    if potential_cause in next_state:
                        # Calculate correlation strength
                        strength = self._estimate_causal_strength(
                            potential_cause, var_name, recent_states
                        )

                        if strength > self.causal_threshold:
                            # Add or strengthen causal edge
                            if not self.causal_graph.has_edge(potential_cause, var_name):
                                self.add_causal_relation(
                                    potential_cause,
                                    var_name,
                                    CausalRelationType.CAUSES,
                                    strength=strength,
                                )

    def _learn_from_intervention(self, intervention: Dict[str, Any]) -> None:
        """Learn causal relationships from interventions."""
        variable = intervention['variable']
        pre_state = intervention['pre_state']
        post_state = intervention['post_state']

        # Compare pre and post to identify causal effects
        for var_name in post_state:
            if var_name != variable:
                # Check if this variable changed
                pre_val = pre_state.get(var_name)
                post_val = post_state.get(var_name)

                if pre_val != post_val:
                    # This variable was affected by the intervention
                    # Add or strengthen causal edge
                    if self.causal_graph.has_edge(variable, var_name):
                        # Strengthen existing edge
                        edge_data = self.causal_graph.edges[variable, var_name]
                        edge_data['weight'] = min(
                            1.0,
                            edge_data['weight'] + self.intervention_learning_rate
                        )
                        edge_data['edge'].learned_from_intervention = True
                    else:
                        # Create new edge with high confidence (from intervention)
                        self.add_causal_relation(
                            variable,
                            var_name,
                            CausalRelationType.CAUSES,
                            strength=0.8,
                        )
                        self.causal_graph.edges[variable, var_name]['edge'].learned_from_intervention = True

    def _propagate_causation(
        self,
        changed_variable: str,
        new_value: Any,
        fixed_variables: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Propagate causal effects through the graph."""
        if fixed_variables is None:
            fixed_variables = {}

        # Start with current state
        state = {name: var.value for name, var in self.variables.items()}
        state[changed_variable] = new_value

        # Get all variables causally downstream
        if changed_variable in self.causal_graph:
            affected = nx.descendants(self.causal_graph, changed_variable)

            # Topological sort for correct order
            subgraph = self.causal_graph.subgraph([changed_variable] + list(affected))
            try:
                causal_order = list(nx.topological_sort(subgraph))

                for var_name in causal_order:
                    if var_name in fixed_variables:
                        # Keep fixed for counterfactuals
                        state[var_name] = fixed_variables[var_name]
                    else:
                        # Apply causal mechanism
                        causes = list(self.causal_graph.predecessors(var_name))
                        if causes:
                            state[var_name] = self._apply_causal_mechanism(
                                var_name, causes, state
                            )
            except nx.NetworkXError:
                logger.warning("Causal propagation failed due to cycles")

        return state

    def _apply_causal_mechanism(
        self,
        effect: str,
        causes: List[str],
        state: Dict[str, Any],
    ) -> Any:
        """Apply learned causal mechanism to predict effect."""
        # Check if we have a learned mechanism
        for cause in causes:
            if (cause, effect) in self.causal_mechanisms:
                mechanism = self.causal_mechanisms[(cause, effect)]
                try:
                    return mechanism(state[cause])
                except Exception:
                    pass

        # Default: weighted sum of causes (for continuous variables)
        if self.variables[effect].continuous:
            value = 0.0
            total_weight = 0.0

            for cause in causes:
                if self.causal_graph.has_edge(cause, effect):
                    weight = self.causal_graph.edges[cause, effect]['weight']
                    cause_val = state.get(cause, 0)

                    if isinstance(cause_val, (int, float)):
                        value += weight * cause_val
                        total_weight += weight

            if total_weight > 0:
                return value / total_weight

        # Return current value as fallback
        return self.variables[effect].value

    def _apply_physics(
        self,
        state: Dict[str, Any],
        timestep: int,
    ) -> Dict[str, Any]:
        """Apply physics simulation to object positions."""
        dt = 0.1  # Time step

        # Update object positions based on velocities
        for obj_id, obj in self.objects.items():
            if obj.permanence:
                # Apply gravity
                if self.gravity is not None:
                    obj.velocity += self.gravity * dt

                # Apply friction
                obj.velocity *= (1.0 - self.friction)

                # Update position
                obj.position += obj.velocity * dt

                # Store in state
                state[f"{obj_id}_position"] = obj.position.tolist()
                state[f"{obj_id}_velocity"] = obj.velocity.tolist()

        return state

    def _infer_hidden_states(
        self,
        observations: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Infer values of hidden/unobserved variables from observations."""
        hidden = {}

        # For each hidden variable, use causal graph to infer
        for var_name, var in self.variables.items():
            if not var.observable and var_name not in observations:
                # Find observed descendants
                if var_name in self.causal_graph:
                    descendants = nx.descendants(self.causal_graph, var_name)
                    observed_descendants = [d for d in descendants if d in observations]

                    if observed_descendants:
                        # Simple inference: average of inverse mechanisms
                        # (This is a placeholder for more sophisticated inference)
                        inferred_value = np.mean([observations[d] for d in observed_descendants])
                        hidden[var_name] = inferred_value

        return hidden

    def _find_nearest_object(
        self,
        position: np.ndarray,
        object_type: str,
    ) -> Optional[CausalObject]:
        """Find nearest object of given type."""
        min_dist = float('inf')
        nearest = None

        for obj in self.objects.values():
            if obj.object_type == object_type:
                dist = np.linalg.norm(obj.position - position)
                if dist < min_dist:
                    min_dist = dist
                    nearest = obj

        return nearest

    def _prune_objects(self) -> None:
        """Remove objects with low confidence or not observed recently."""
        current_time = time.time()
        to_remove = []

        for obj_id, obj in self.objects.items():
            # Remove if not observed recently and low confidence
            time_since_observation = current_time - obj.last_observed

            if (time_since_observation > 10.0 and
                obj.existence_confidence < self.object_persistence_threshold):
                to_remove.append(obj_id)

        for obj_id in to_remove:
            del self.objects[obj_id]
            logger.debug(f"Pruned object: {obj_id}")

    def _estimate_causal_strength(
        self,
        cause: str,
        effect: str,
        states: List[Dict[str, Any]],
    ) -> float:
        """Estimate causal strength from temporal correlations."""
        if len(states) < 2:
            return 0.0

        # Calculate temporal correlation
        cause_changes = []
        effect_changes = []

        for i in range(len(states) - 1):
            if cause in states[i] and cause in states[i+1]:
                if effect in states[i] and effect in states[i+1]:
                    cause_change = states[i+1][cause] != states[i][cause]
                    effect_change = states[i+1][effect] != states[i][effect]

                    cause_changes.append(int(cause_change))
                    effect_changes.append(int(effect_change))

        if not cause_changes:
            return 0.0

        # Simple correlation
        correlation = np.corrcoef(cause_changes, effect_changes)[0, 1]
        return abs(correlation) if not np.isnan(correlation) else 0.0

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the causal world model."""
        return {
            'num_variables': len(self.variables),
            'num_causal_relations': self.causal_graph.number_of_edges(),
            'num_objects': len(self.objects),
            'causal_discoveries': self.causal_discoveries,
            'total_predictions': self.total_predictions,
            'total_interventions': self.total_interventions,
            'avg_prediction_error': np.mean(self.prediction_errors) if self.prediction_errors else 0.0,
            'state_history_size': len(self.state_history),
            'intervention_history_size': len(self.intervention_history),
        }

    def serialize(self) -> Dict[str, Any]:
        """Serialize the causal world model."""
        return {
            'state_dim': self.state_dim,
            'variables': {
                name: {
                    'value': var.value,
                    'observable': var.observable,
                    'continuous': var.continuous,
                }
                for name, var in list(self.variables.items())[:100]  # Limit size
            },
            'causal_relations': [
                {
                    'cause': u,
                    'effect': v,
                    'type': data.get('type').value if data.get('type') else 'unknown',
                    'weight': data.get('weight', 1.0),
                }
                for u, v, data in self.causal_graph.edges(data=True)
            ],
            'objects': len(self.objects),
            'stats': self.get_stats(),
        }

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> 'CausalWorldModel':
        """Create a causal world model from serialized data."""
        instance = cls(state_dim=data['state_dim'])

        # Restore variables
        for name, var_data in data.get('variables', {}).items():
            instance.add_variable(
                name,
                initial_value=var_data['value'],
                observable=var_data['observable'],
                continuous=var_data['continuous'],
            )

        # Restore causal relations
        for rel in data.get('causal_relations', []):
            instance.add_causal_relation(
                rel['cause'],
                rel['effect'],
                CausalRelationType(rel['type']),
                strength=rel['weight'],
            )

        return instance
