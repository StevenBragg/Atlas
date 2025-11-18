"""
World Model and Causal Reasoning for ATLAS

Implements a predictive world model with causal reasoning capabilities,
enabling the system to understand causation, perform counterfactual
reasoning, and make interventions.

This is critical for superintelligence as it enables:
- Understanding cause and effect relationships
- Predicting consequences of actions
- Counterfactual reasoning (what if scenarios)
- Planning with intervention modeling
- Physics understanding and object permanence
"""

import numpy as np
import logging
from typing import List, Dict, Tuple, Optional, Any, Set, Callable
from dataclasses import dataclass, field
from enum import Enum
import networkx as nx
from collections import defaultdict
import time

logger = logging.getLogger(__name__)


class CausalRelationType(Enum):
    """Types of causal relationships"""
    CAUSES = "causes"  # X causes Y
    PREVENTS = "prevents"  # X prevents Y
    ENABLES = "enables"  # X enables Y
    REQUIRES = "requires"  # X requires Y
    CORRELATES = "correlates"  # X correlates with Y (not necessarily causal)


@dataclass
class Variable:
    """Represents a variable in the world model"""
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
class Object:
    """Represents a persistent object in the world"""
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
        Initialize world model.

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
        self.objects: Dict[str, Object] = {}
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

        logger.info(f"Initialized world model: state_dim={state_dim}")

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
    ) -> CausalEdge:
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
    ) -> Object:
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
            obj = Object(
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
                except:
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
    ) -> Optional[Object]:
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
        """Get statistics about the world model."""
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
        """Serialize the world model."""
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
    def deserialize(cls, data: Dict[str, Any]) -> 'WorldModel':
        """Create a world model from serialized data."""
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
