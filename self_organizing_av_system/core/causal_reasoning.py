"""
Causal Reasoning Engine for ATLAS

Implements causal inference and reasoning capabilities:
- Learning causal structure from observations
- Interventional reasoning (do-calculus)
- Counterfactual reasoning ("what if")
- Causal explanation generation
- World model construction

Based on Pearl's causal hierarchy:
1. Association (seeing): P(Y|X) - observational
2. Intervention (doing): P(Y|do(X)) - experimental
3. Counterfactual (imagining): P(Y_x|X',Y') - hypothetical

This enables ATLAS to reason about cause-and-effect rather than
just correlations, which is essential for planning and understanding.
"""

import numpy as np
import logging
from typing import List, Dict, Tuple, Optional, Any, Set, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import time

logger = logging.getLogger(__name__)


class CausalRelationType(Enum):
    """Types of causal relationships"""
    CAUSES = "causes"           # X causes Y
    PREVENTS = "prevents"       # X prevents Y
    ENABLES = "enables"         # X enables Y (necessary but not sufficient)
    CORRELATES = "correlates"   # X correlates with Y (no direction)
    CONFOUNDED = "confounded"   # X and Y share common cause


@dataclass
class CausalLink:
    """A causal link between two variables"""
    cause: str
    effect: str
    relation_type: CausalRelationType
    strength: float  # Causal strength [0, 1]
    confidence: float  # Confidence in this link
    observations: int  # Number of supporting observations
    mechanism: Optional[str] = None  # Description of mechanism
    conditions: List[str] = field(default_factory=list)  # Necessary conditions


@dataclass
class CausalVariable:
    """A variable in the causal graph"""
    name: str
    current_value: Optional[float] = None
    is_observable: bool = True
    is_manipulable: bool = True
    domain: Tuple[float, float] = (0.0, 1.0)
    parents: List[str] = field(default_factory=list)
    children: List[str] = field(default_factory=list)


@dataclass
class Intervention:
    """An intervention (do-operation)"""
    variable: str
    value: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class CounterfactualQuery:
    """A counterfactual query"""
    # Factual: What actually happened
    factual_evidence: Dict[str, float]
    # Hypothetical: What if this had been different?
    hypothetical_intervention: Dict[str, float]
    # Query: What would have happened to this variable?
    query_variable: str


@dataclass
class CausalExplanation:
    """An explanation for why something happened"""
    effect: str
    effect_value: float
    causes: List[Tuple[str, float, float]]  # (cause, value, contribution)
    counterfactual: Optional[str] = None
    confidence: float = 0.5


class CausalGraph:
    """
    Directed Acyclic Graph (DAG) representing causal structure.

    Nodes are variables, edges are causal relationships.
    """

    def __init__(self):
        """Initialize empty causal graph."""
        self.variables: Dict[str, CausalVariable] = {}
        self.links: Dict[Tuple[str, str], CausalLink] = {}

        # Adjacency lists
        self.parents: Dict[str, Set[str]] = defaultdict(set)
        self.children: Dict[str, Set[str]] = defaultdict(set)

        # Cached topological order
        self._topo_order: Optional[List[str]] = None
        self._order_valid = False

    def add_variable(
        self,
        name: str,
        is_observable: bool = True,
        is_manipulable: bool = True,
        domain: Tuple[float, float] = (0.0, 1.0),
    ) -> CausalVariable:
        """Add a variable to the graph."""
        if name not in self.variables:
            var = CausalVariable(
                name=name,
                is_observable=is_observable,
                is_manipulable=is_manipulable,
                domain=domain,
            )
            self.variables[name] = var
            self._order_valid = False
        return self.variables[name]

    def add_link(
        self,
        cause: str,
        effect: str,
        relation_type: CausalRelationType = CausalRelationType.CAUSES,
        strength: float = 0.5,
        confidence: float = 0.5,
    ) -> CausalLink:
        """Add a causal link to the graph."""
        # Ensure variables exist
        if cause not in self.variables:
            self.add_variable(cause)
        if effect not in self.variables:
            self.add_variable(effect)

        # Check for cycles
        if self._would_create_cycle(cause, effect):
            logger.warning(f"Cannot add link {cause}->{effect}: would create cycle")
            return None

        link = CausalLink(
            cause=cause,
            effect=effect,
            relation_type=relation_type,
            strength=strength,
            confidence=confidence,
            observations=1,
        )

        self.links[(cause, effect)] = link
        self.parents[effect].add(cause)
        self.children[cause].add(effect)

        # Update variable parent/child lists
        self.variables[cause].children.append(effect)
        self.variables[effect].parents.append(cause)

        self._order_valid = False

        return link

    def _would_create_cycle(self, cause: str, effect: str) -> bool:
        """Check if adding cause->effect would create a cycle."""
        # Would create cycle if effect is an ancestor of cause
        visited = set()
        stack = [effect]

        while stack:
            node = stack.pop()
            if node == cause:
                return True
            if node in visited:
                continue
            visited.add(node)
            stack.extend(self.children.get(node, set()))

        return False

    def get_topological_order(self) -> List[str]:
        """Get topological ordering of variables."""
        if self._order_valid and self._topo_order is not None:
            return self._topo_order

        # Kahn's algorithm
        in_degree = {name: 0 for name in self.variables}
        for name in self.variables:
            in_degree[name] = len(self.parents.get(name, set()))

        queue = [name for name, deg in in_degree.items() if deg == 0]
        order = []

        while queue:
            node = queue.pop(0)
            order.append(node)

            for child in self.children.get(node, set()):
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)

        self._topo_order = order
        self._order_valid = True

        return order

    def get_ancestors(self, variable: str) -> Set[str]:
        """Get all ancestors of a variable."""
        ancestors = set()
        stack = list(self.parents.get(variable, set()))

        while stack:
            node = stack.pop()
            if node not in ancestors:
                ancestors.add(node)
                stack.extend(self.parents.get(node, set()))

        return ancestors

    def get_descendants(self, variable: str) -> Set[str]:
        """Get all descendants of a variable."""
        descendants = set()
        stack = list(self.children.get(variable, set()))

        while stack:
            node = stack.pop()
            if node not in descendants:
                descendants.add(node)
                stack.extend(self.children.get(node, set()))

        return descendants

    def get_markov_blanket(self, variable: str) -> Set[str]:
        """Get the Markov blanket of a variable."""
        # Parents + Children + Parents of children
        blanket = set()

        # Parents
        blanket.update(self.parents.get(variable, set()))

        # Children
        blanket.update(self.children.get(variable, set()))

        # Parents of children
        for child in self.children.get(variable, set()):
            blanket.update(self.parents.get(child, set()))

        # Remove self
        blanket.discard(variable)

        return blanket

    def get_link(self, cause: str, effect: str) -> Optional[CausalLink]:
        """Get the link between cause and effect."""
        return self.links.get((cause, effect))

    def remove_link(self, cause: str, effect: str) -> bool:
        """Remove a causal link."""
        if (cause, effect) in self.links:
            del self.links[(cause, effect)]
            self.parents[effect].discard(cause)
            self.children[cause].discard(effect)
            self._order_valid = False
            return True
        return False

    def serialize(self) -> Dict[str, Any]:
        """Serialize the causal graph."""
        return {
            'variables': {
                name: {
                    'is_observable': var.is_observable,
                    'is_manipulable': var.is_manipulable,
                    'domain': var.domain,
                    'current_value': var.current_value,
                }
                for name, var in self.variables.items()
            },
            'links': [
                {
                    'cause': link.cause,
                    'effect': link.effect,
                    'relation_type': link.relation_type.value,
                    'strength': link.strength,
                    'confidence': link.confidence,
                    'observations': link.observations,
                }
                for link in self.links.values()
            ],
        }

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> 'CausalGraph':
        """Deserialize a causal graph."""
        graph = cls()

        for name, var_data in data.get('variables', {}).items():
            var = graph.add_variable(
                name=name,
                is_observable=var_data.get('is_observable', True),
                is_manipulable=var_data.get('is_manipulable', True),
                domain=tuple(var_data.get('domain', (0.0, 1.0))),
            )
            var.current_value = var_data.get('current_value')

        for link_data in data.get('links', []):
            graph.add_link(
                cause=link_data['cause'],
                effect=link_data['effect'],
                relation_type=CausalRelationType(link_data['relation_type']),
                strength=link_data['strength'],
                confidence=link_data['confidence'],
            )

        return graph


class CausalModel:
    """
    A structural causal model (SCM) that can be used for inference.

    Combines the causal graph with functional relationships
    that determine how causes produce effects.
    """

    def __init__(self, graph: Optional[CausalGraph] = None):
        """
        Initialize causal model.

        Args:
            graph: Optional pre-existing causal graph
        """
        self.graph = graph if graph is not None else CausalGraph()

        # Structural equations: effect = f(parents, noise)
        self.equations: Dict[str, Callable] = {}

        # Learned parameter weights
        self.weights: Dict[Tuple[str, str], float] = {}

        # Noise distributions (for counterfactuals)
        self.noise_terms: Dict[str, float] = {}

        # Observation history for learning
        self.observations: List[Dict[str, float]] = []

    def add_equation(
        self,
        variable: str,
        equation: Callable[[Dict[str, float], float], float],
    ) -> None:
        """
        Add a structural equation for a variable.

        The equation takes (parent_values, noise) and returns the variable value.
        """
        self.equations[variable] = equation

    def set_default_linear_equations(self) -> None:
        """Set default linear equations based on graph structure."""
        for var_name, var in self.graph.variables.items():
            parents = list(self.graph.parents.get(var_name, set()))

            if len(parents) == 0:
                # Root node: just noise
                self.equations[var_name] = lambda parents, noise: noise
            else:
                # Linear combination of parents + noise
                def make_equation(parent_list):
                    def eq(parent_vals, noise):
                        total = noise
                        for p in parent_list:
                            weight = self.weights.get((p, var_name), 0.5)
                            total += weight * parent_vals.get(p, 0.0)
                        return np.clip(total, 0, 1)
                    return eq

                self.equations[var_name] = make_equation(parents)

    def observe(self, observation: Dict[str, float]) -> None:
        """
        Record an observation of variable values.

        Used for learning causal structure and parameters.
        """
        self.observations.append(observation.copy())

        # Update variable current values
        for name, value in observation.items():
            if name in self.graph.variables:
                self.graph.variables[name].current_value = value

    def predict(
        self,
        evidence: Dict[str, float],
        query: str,
    ) -> float:
        """
        Predict the value of a query variable given evidence.

        This is observational prediction P(query | evidence).
        """
        # Get topological order
        order = self.graph.get_topological_order()

        # Initialize values with evidence
        values = evidence.copy()

        # Forward propagation in topological order
        for var_name in order:
            if var_name in values:
                continue  # Already observed

            if var_name not in self.equations:
                values[var_name] = 0.5  # Default
                continue

            # Get parent values
            parent_vals = {}
            for parent in self.graph.parents.get(var_name, set()):
                parent_vals[parent] = values.get(parent, 0.5)

            # Get noise
            noise = self.noise_terms.get(var_name, 0.0)

            # Apply equation
            values[var_name] = self.equations[var_name](parent_vals, noise)

        return values.get(query, 0.5)

    def intervene(
        self,
        intervention: Dict[str, float],
        query: str,
    ) -> float:
        """
        Predict the effect of an intervention.

        This computes P(query | do(intervention)).
        The do-operator breaks incoming causal links to intervened variables.
        """
        # Get topological order
        order = self.graph.get_topological_order()

        # Initialize values
        values = {}

        # Forward propagation with intervention
        for var_name in order:
            if var_name in intervention:
                # Intervention: set value directly (break incoming links)
                values[var_name] = intervention[var_name]
                continue

            if var_name not in self.equations:
                values[var_name] = 0.5
                continue

            # Get parent values
            parent_vals = {}
            for parent in self.graph.parents.get(var_name, set()):
                parent_vals[parent] = values.get(parent, 0.5)

            # Get noise
            noise = self.noise_terms.get(var_name, 0.0)

            # Apply equation
            values[var_name] = self.equations[var_name](parent_vals, noise)

        return values.get(query, 0.5)

    def counterfactual(
        self,
        query: CounterfactualQuery,
    ) -> float:
        """
        Answer a counterfactual query.

        Three steps (Pearl's approach):
        1. Abduction: Infer noise terms from factual evidence
        2. Action: Apply hypothetical intervention
        3. Prediction: Predict query under modified model
        """
        # Step 1: Abduction - infer noise terms from evidence
        self._abduct_noise(query.factual_evidence)

        # Step 2 & 3: Action + Prediction - intervene and predict
        result = self.intervene(query.hypothetical_intervention, query.query_variable)

        return result

    def _abduct_noise(self, evidence: Dict[str, float]) -> None:
        """Infer noise terms from observed evidence."""
        # Simple approach: work backwards from evidence
        order = self.graph.get_topological_order()

        values = evidence.copy()

        # Fill in missing values with predictions
        for var_name in order:
            if var_name in values:
                continue
            values[var_name] = 0.5  # Default for unobserved

        # Compute noise that would produce observed values
        for var_name in order:
            if var_name not in evidence:
                continue

            parents = self.graph.parents.get(var_name, set())
            if len(parents) == 0:
                # Root node: noise = value
                self.noise_terms[var_name] = evidence[var_name]
            else:
                # Compute what parent contribution would be
                parent_contribution = 0.0
                for parent in parents:
                    weight = self.weights.get((parent, var_name), 0.5)
                    parent_contribution += weight * values.get(parent, 0.5)

                # Noise = observed - parent_contribution
                self.noise_terms[var_name] = evidence[var_name] - parent_contribution

    def learn_structure(
        self,
        threshold: float = 0.1,
        min_observations: int = 10,
    ) -> int:
        """
        Learn causal structure from observations using correlations.

        This is a simplified approach. Full causal discovery would use
        PC algorithm, FCI, or similar.

        Returns number of links learned.
        """
        if len(self.observations) < min_observations:
            return 0

        # Get all variables that appear in observations
        all_vars = set()
        for obs in self.observations:
            all_vars.update(obs.keys())

        # Ensure all variables exist in graph
        for var in all_vars:
            if var not in self.graph.variables:
                self.graph.add_variable(var)

        # Compute correlations
        var_list = list(all_vars)
        n_vars = len(var_list)
        n_obs = len(self.observations)

        # Build data matrix
        data = np.zeros((n_obs, n_vars))
        for i, obs in enumerate(self.observations):
            for j, var in enumerate(var_list):
                data[i, j] = obs.get(var, 0.5)

        # Compute correlation matrix
        correlations = np.corrcoef(data.T)

        # Add links for strong correlations
        links_added = 0
        for i in range(n_vars):
            for j in range(i + 1, n_vars):
                corr = correlations[i, j]
                if np.isnan(corr):
                    continue

                if abs(corr) > threshold:
                    # Determine direction heuristically (temporal order, etc.)
                    # For simplicity, use alphabetical order
                    if var_list[i] < var_list[j]:
                        cause, effect = var_list[i], var_list[j]
                    else:
                        cause, effect = var_list[j], var_list[i]

                    # Add link if doesn't exist
                    if (cause, effect) not in self.graph.links:
                        link = self.graph.add_link(
                            cause=cause,
                            effect=effect,
                            strength=abs(corr),
                            confidence=min(1.0, len(self.observations) / 100),
                        )
                        if link is not None:
                            links_added += 1

                            # Learn weight
                            self.weights[(cause, effect)] = corr

        logger.info(f"Learned {links_added} causal links from {len(self.observations)} observations")

        # Set default equations
        self.set_default_linear_equations()

        return links_added

    def get_causal_effect(
        self,
        cause: str,
        effect: str,
        cause_values: Tuple[float, float] = (0.0, 1.0),
    ) -> float:
        """
        Estimate the causal effect of cause on effect.

        Computes: E[effect | do(cause=1)] - E[effect | do(cause=0)]
        """
        effect_low = self.intervene({cause: cause_values[0]}, effect)
        effect_high = self.intervene({cause: cause_values[1]}, effect)

        return effect_high - effect_low

    def explain(
        self,
        effect: str,
        effect_value: float,
        evidence: Dict[str, float],
    ) -> CausalExplanation:
        """
        Generate a causal explanation for an effect value.

        Returns the causes and their contributions.
        """
        causes = []

        # Get direct parents
        parents = self.graph.parents.get(effect, set())

        for parent in parents:
            parent_value = evidence.get(parent, 0.5)

            # Compute contribution using causal effect
            weight = self.weights.get((parent, effect), 0.5)
            contribution = weight * parent_value

            causes.append((parent, parent_value, contribution))

        # Sort by contribution
        causes.sort(key=lambda x: abs(x[2]), reverse=True)

        # Generate counterfactual explanation
        counterfactual_text = None
        if len(causes) > 0:
            top_cause = causes[0][0]
            top_value = causes[0][1]

            # What if top cause had been different?
            alt_value = 1.0 - top_value
            cf_result = self.counterfactual(CounterfactualQuery(
                factual_evidence=evidence,
                hypothetical_intervention={top_cause: alt_value},
                query_variable=effect,
            ))

            counterfactual_text = (
                f"If {top_cause} had been {alt_value:.2f} instead of {top_value:.2f}, "
                f"{effect} would have been {cf_result:.2f} instead of {effect_value:.2f}"
            )

        return CausalExplanation(
            effect=effect,
            effect_value=effect_value,
            causes=causes,
            counterfactual=counterfactual_text,
            confidence=sum(link.confidence for link in self.graph.links.values()
                          if link.effect == effect) / max(1, len(parents)),
        )


class CausalReasoner:
    """
    High-level causal reasoning system that integrates with ATLAS.

    Provides:
    - Causal learning from experience
    - Intervention planning
    - Counterfactual reasoning
    - Causal explanation
    """

    def __init__(
        self,
        model: Optional[CausalModel] = None,
        learning_rate: float = 0.01,
        min_observations_for_learning: int = 20,
    ):
        """
        Initialize causal reasoner.

        Args:
            model: Optional pre-existing causal model
            learning_rate: Learning rate for weight updates
            min_observations_for_learning: Minimum observations before structure learning
        """
        self.model = model if model is not None else CausalModel()
        self.learning_rate = learning_rate
        self.min_observations_for_learning = min_observations_for_learning

        # Intervention history
        self.intervention_history: List[Tuple[Intervention, Dict[str, float]]] = []

        # Statistics
        self.total_observations = 0
        self.total_interventions = 0
        self.total_counterfactuals = 0
        self.structure_learning_runs = 0

    def observe(
        self,
        observation: Dict[str, float],
        learn_structure: bool = True,
    ) -> None:
        """
        Observe variable values.

        Args:
            observation: Variable name -> value mapping
            learn_structure: Whether to trigger structure learning
        """
        self.model.observe(observation)
        self.total_observations += 1

        # Periodically learn structure
        if (learn_structure and
            self.total_observations >= self.min_observations_for_learning and
            self.total_observations % 50 == 0):
            self.model.learn_structure()
            self.structure_learning_runs += 1

    def do(
        self,
        interventions: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Perform interventions and predict all effects.

        Args:
            interventions: Variable name -> intervention value

        Returns:
            Predicted values for all variables
        """
        self.total_interventions += 1

        # Predict all variables
        predictions = {}
        for var_name in self.model.graph.variables:
            predictions[var_name] = self.model.intervene(interventions, var_name)

        # Record intervention
        for var, val in interventions.items():
            intervention = Intervention(variable=var, value=val)
            self.intervention_history.append((intervention, predictions))

        return predictions

    def what_if(
        self,
        factual: Dict[str, float],
        hypothetical: Dict[str, float],
        query: str,
    ) -> Tuple[float, str]:
        """
        Answer a "what if" counterfactual question.

        Args:
            factual: What actually happened
            hypothetical: What if this had been different
            query: What would have happened to this variable

        Returns:
            (predicted_value, explanation)
        """
        self.total_counterfactuals += 1

        cf_query = CounterfactualQuery(
            factual_evidence=factual,
            hypothetical_intervention=hypothetical,
            query_variable=query,
        )

        result = self.model.counterfactual(cf_query)

        # Generate explanation
        actual_value = factual.get(query, self.model.predict(factual, query))
        hyp_vars = ", ".join(f"{k}={v:.2f}" for k, v in hypothetical.items())

        explanation = (
            f"If {hyp_vars} (instead of current values), "
            f"then {query} would be {result:.2f} (instead of {actual_value:.2f})"
        )

        return result, explanation

    def why(
        self,
        effect: str,
        evidence: Dict[str, float],
    ) -> CausalExplanation:
        """
        Explain why an effect has a certain value.

        Args:
            effect: Variable to explain
            evidence: Observed values

        Returns:
            Causal explanation
        """
        effect_value = evidence.get(effect, self.model.predict(evidence, effect))
        return self.model.explain(effect, effect_value, evidence)

    def find_causes(self, effect: str) -> List[Tuple[str, float]]:
        """Find the direct causes of an effect and their strengths."""
        causes = []
        parents = self.model.graph.parents.get(effect, set())

        for parent in parents:
            link = self.model.graph.get_link(parent, effect)
            if link:
                causes.append((parent, link.strength))

        causes.sort(key=lambda x: x[1], reverse=True)
        return causes

    def find_effects(self, cause: str) -> List[Tuple[str, float]]:
        """Find the direct effects of a cause and their strengths."""
        effects = []
        children = self.model.graph.children.get(cause, set())

        for child in children:
            link = self.model.graph.get_link(cause, child)
            if link:
                effects.append((child, link.strength))

        effects.sort(key=lambda x: x[1], reverse=True)
        return effects

    def plan_intervention(
        self,
        goal: Dict[str, float],
        available_interventions: List[str],
        current_state: Dict[str, float],
    ) -> List[Tuple[str, float]]:
        """
        Plan interventions to achieve a goal.

        Args:
            goal: Desired variable values
            available_interventions: Variables that can be manipulated
            current_state: Current variable values

        Returns:
            List of (variable, value) interventions to perform
        """
        best_plan = []
        best_distance = float('inf')

        # Simple greedy search over single interventions
        for var in available_interventions:
            if var not in self.model.graph.variables:
                continue

            # Try different intervention values
            for value in [0.0, 0.25, 0.5, 0.75, 1.0]:
                predictions = self.do({var: value})

                # Compute distance to goal
                distance = 0.0
                for goal_var, goal_val in goal.items():
                    pred_val = predictions.get(goal_var, 0.5)
                    distance += (pred_val - goal_val) ** 2

                distance = np.sqrt(distance)

                if distance < best_distance:
                    best_distance = distance
                    best_plan = [(var, value)]

        return best_plan

    def get_stats(self) -> Dict[str, Any]:
        """Get reasoner statistics."""
        return {
            'total_observations': self.total_observations,
            'total_interventions': self.total_interventions,
            'total_counterfactuals': self.total_counterfactuals,
            'structure_learning_runs': self.structure_learning_runs,
            'num_variables': len(self.model.graph.variables),
            'num_links': len(self.model.graph.links),
        }

    def serialize(self) -> Dict[str, Any]:
        """Serialize the reasoner state."""
        return {
            'graph': self.model.graph.serialize(),
            'weights': {f"{k[0]}_{k[1]}": v for k, v in self.model.weights.items()},
            'stats': self.get_stats(),
        }

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> 'CausalReasoner':
        """Deserialize a reasoner from saved state."""
        graph = CausalGraph.deserialize(data['graph'])
        model = CausalModel(graph)

        # Restore weights
        for key, value in data.get('weights', {}).items():
            parts = key.split('_')
            if len(parts) >= 2:
                model.weights[(parts[0], parts[1])] = value

        model.set_default_linear_equations()

        return cls(model=model)


class WorldModel:
    """
    World model that uses causal reasoning to simulate the environment.

    Enables:
    - Forward simulation (prediction)
    - Backward inference (explanation)
    - Planning through mental simulation
    """

    def __init__(
        self,
        causal_reasoner: Optional[CausalReasoner] = None,
        state_dim: int = 64,
    ):
        """
        Initialize world model.

        Args:
            causal_reasoner: Causal reasoning system to use
            state_dim: Dimension of state vectors
        """
        self.reasoner = causal_reasoner if causal_reasoner else CausalReasoner()
        self.state_dim = state_dim

        # Current world state
        self.current_state: Dict[str, float] = {}

        # State history
        self.state_history: List[Dict[str, float]] = []

        # Action effects cache
        self.action_effects: Dict[str, Dict[str, float]] = {}

    def update_state(self, observation: Dict[str, float]) -> None:
        """Update world state from observation."""
        self.current_state.update(observation)
        self.state_history.append(self.current_state.copy())
        self.reasoner.observe(observation)

    def simulate_action(
        self,
        action: str,
        action_value: float = 1.0,
    ) -> Dict[str, float]:
        """
        Simulate the effect of an action.

        Returns predicted next state.
        """
        intervention = {action: action_value}
        predicted_state = self.reasoner.do(intervention)

        return predicted_state

    def plan_actions(
        self,
        goal_state: Dict[str, float],
        available_actions: List[str],
        max_steps: int = 5,
    ) -> List[Tuple[str, float]]:
        """
        Plan a sequence of actions to reach a goal state.

        Simple forward search planning.
        """
        plan = []
        simulated_state = self.current_state.copy()

        for step in range(max_steps):
            # Find best single action
            best_action = None
            best_value = None
            best_distance = float('inf')

            for action in available_actions:
                for value in [0.0, 0.5, 1.0]:
                    # Simulate
                    test_state = self.reasoner.do({action: value})

                    # Compute distance to goal
                    distance = sum(
                        (test_state.get(k, 0) - v) ** 2
                        for k, v in goal_state.items()
                    )

                    if distance < best_distance:
                        best_distance = distance
                        best_action = action
                        best_value = value

            if best_action is None:
                break

            plan.append((best_action, best_value))
            simulated_state = self.reasoner.do({best_action: best_value})

            # Check if goal reached
            goal_distance = sum(
                (simulated_state.get(k, 0) - v) ** 2
                for k, v in goal_state.items()
            )
            if goal_distance < 0.01:
                break

        return plan

    def get_state_summary(self) -> Dict[str, Any]:
        """Get summary of world model state."""
        return {
            'num_state_variables': len(self.current_state),
            'history_length': len(self.state_history),
            'causal_links': len(self.reasoner.model.graph.links),
            'current_state': self.current_state.copy(),
        }
