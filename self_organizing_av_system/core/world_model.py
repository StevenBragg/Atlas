#!/usr/bin/env python3
"""
World Model System for ATLAS Superintelligence

This module implements a world model that learns physics, object permanence,
and causal relationships through local plasticity rules. It enables:
1. Object-centric representations with permanence tracking
2. Physics simulation for forward state prediction
3. Hidden state inference using Bayesian principles
4. State transitions learned through local plasticity

Core Principles:
- No backpropagation - uses local Hebbian/STDP learning
- Biologically plausible - inspired by hippocampal place cells and grid cells
- Predictive coding - generates predictions and learns from errors
- Self-organizing - structure emerges from data statistics
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union
from enum import Enum
from collections import deque
import logging
import time

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
