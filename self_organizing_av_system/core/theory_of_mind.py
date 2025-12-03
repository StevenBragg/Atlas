#!/usr/bin/env python3
"""
Theory of Mind Module for ATLAS Superintelligence

This module implements social cognition capabilities that enable:
1. Agent modeling - Representing other agents' internal states
2. Belief tracking - Modeling what others believe (including false beliefs)
3. Intention inference - Understanding goals and motivations
4. Cooperative/competitive reasoning - Strategic social interaction
5. Social learning - Learning from observing others

Core Principles:
- No backpropagation - uses local Hebbian/STDP learning
- Biologically plausible - inspired by mirror neurons and TPJ
- Simulation-based - Uses self-model to simulate others
- Bayesian inference - Updates beliefs based on observations
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Set
from enum import Enum
from collections import deque
import logging
import time

logger = logging.getLogger(__name__)


class AgentType(Enum):
    """Types of agents that can be modeled"""
    SELF = "self"
    HUMAN = "human"
    AI = "ai"
    ANIMAL = "animal"
    GENERIC = "generic"


class MentalStateType(Enum):
    """Types of mental states"""
    BELIEF = "belief"
    DESIRE = "desire"
    INTENTION = "intention"
    EMOTION = "emotion"
    ATTENTION = "attention"
    KNOWLEDGE = "knowledge"


class SocialRelation(Enum):
    """Types of social relationships"""
    COOPERATIVE = "cooperative"
    COMPETITIVE = "competitive"
    NEUTRAL = "neutral"
    HIERARCHICAL = "hierarchical"
    RECIPROCAL = "reciprocal"


@dataclass
class MentalState:
    """A mental state attribution"""
    state_type: MentalStateType
    content: np.ndarray  # Embedding of the content
    confidence: float
    source: str  # How was this inferred
    timestamp: float = field(default_factory=time.time)

    def decay(self, rate: float = 0.01):
        """Decay confidence over time"""
        self.confidence *= (1 - rate)


@dataclass
class AgentModel:
    """Model of another agent's mind"""
    agent_id: str
    agent_type: AgentType
    mental_states: Dict[str, MentalState] = field(default_factory=dict)
    beliefs: Dict[str, np.ndarray] = field(default_factory=dict)  # What they believe
    desires: List[np.ndarray] = field(default_factory=list)  # What they want
    intentions: List[np.ndarray] = field(default_factory=list)  # What they intend to do
    knowledge_state: Dict[str, float] = field(default_factory=dict)  # What they know
    personality_embedding: Optional[np.ndarray] = None
    relationship: SocialRelation = SocialRelation.NEUTRAL
    trust_level: float = 0.5
    predictability: float = 0.5  # How well we can predict this agent
    interaction_history: deque = field(default_factory=lambda: deque(maxlen=100))
    creation_time: float = field(default_factory=time.time)
    last_interaction: float = field(default_factory=time.time)


@dataclass
class SocialContext:
    """Context for social reasoning"""
    agents_present: List[str]
    shared_attention: Optional[np.ndarray]
    social_norms: Dict[str, Any]
    power_dynamics: Dict[str, float]  # agent_id -> power level
    common_ground: Dict[str, np.ndarray]  # Shared beliefs/knowledge


@dataclass
class Observation:
    """An observation of another agent's behavior"""
    agent_id: str
    action: np.ndarray
    context: np.ndarray
    outcome: Optional[np.ndarray]
    timestamp: float = field(default_factory=time.time)


class TheoryOfMind:
    """
    Theory of Mind module for modeling and reasoning about other agents.
    """

    def __init__(
        self,
        embedding_dim: int = 64,
        max_agents: int = 50,
        belief_update_rate: float = 0.1,
        simulation_steps: int = 5,
        learning_rate: float = 0.05,
        random_seed: Optional[int] = None
    ):
        """
        Initialize the Theory of Mind module.

        Args:
            embedding_dim: Dimension of mental state embeddings
            max_agents: Maximum number of agent models to maintain
            belief_update_rate: Rate of Bayesian belief updates
            simulation_steps: Steps for mental simulation
            learning_rate: Learning rate for adaptation
            random_seed: Random seed for reproducibility
        """
        self.embedding_dim = embedding_dim
        self.max_agents = max_agents
        self.belief_update_rate = belief_update_rate
        self.simulation_steps = simulation_steps
        self.learning_rate = learning_rate

        if random_seed is not None:
            np.random.seed(random_seed)

        # Agent models
        self.agents: Dict[str, AgentModel] = {}
        self.agent_counter = 0

        # Self model (used for simulation-based inference)
        self.self_model: Optional[AgentModel] = None
        self._initialize_self_model()

        # Observation history
        self.observations: deque = deque(maxlen=1000)

        # Inference model (maps observations to mental states)
        self.inference_weights = np.random.randn(embedding_dim, embedding_dim) * 0.1
        self.inference_bias = np.zeros(embedding_dim)

        # Prediction model (maps mental states to expected actions)
        self.prediction_weights = np.random.randn(embedding_dim, embedding_dim) * 0.1
        self.prediction_bias = np.zeros(embedding_dim)

        # Social context
        self.current_context: Optional[SocialContext] = None

        # Statistics
        self.total_inferences = 0
        self.successful_predictions = 0
        self.total_predictions = 0
        self.interaction_count = 0

    def _initialize_self_model(self):
        """Initialize model of self"""
        self.self_model = AgentModel(
            agent_id="self",
            agent_type=AgentType.SELF,
            personality_embedding=np.random.randn(self.embedding_dim) * 0.1,
            relationship=SocialRelation.NEUTRAL,
            trust_level=1.0,
            predictability=1.0
        )

    def add_agent(
        self,
        agent_type: AgentType = AgentType.GENERIC,
        personality: Optional[np.ndarray] = None,
        relationship: SocialRelation = SocialRelation.NEUTRAL
    ) -> str:
        """Add a new agent to track"""
        if len(self.agents) >= self.max_agents:
            # Remove least recently interacted agent
            oldest = min(self.agents.values(), key=lambda a: a.last_interaction)
            del self.agents[oldest.agent_id]

        agent_id = f"agent_{self.agent_counter}"
        self.agent_counter += 1

        if personality is None:
            personality = np.random.randn(self.embedding_dim)
            personality /= np.linalg.norm(personality) + 1e-8

        agent = AgentModel(
            agent_id=agent_id,
            agent_type=agent_type,
            personality_embedding=personality,
            relationship=relationship
        )

        self.agents[agent_id] = agent

        return agent_id

    def observe(
        self,
        agent_id: str,
        action: np.ndarray,
        context: np.ndarray,
        outcome: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Observe an agent's action and update mental model.

        Args:
            agent_id: ID of observed agent
            action: Embedding of the action taken
            context: Embedding of the context
            outcome: Optional embedding of the outcome

        Returns:
            Dictionary with inferred mental states
        """
        if agent_id not in self.agents:
            agent_id = self.add_agent()

        action = np.asarray(action).flatten()[:self.embedding_dim]
        if len(action) < self.embedding_dim:
            action = np.pad(action, (0, self.embedding_dim - len(action)))

        context = np.asarray(context).flatten()[:self.embedding_dim]
        if len(context) < self.embedding_dim:
            context = np.pad(context, (0, self.embedding_dim - len(context)))

        observation = Observation(
            agent_id=agent_id,
            action=action,
            context=context,
            outcome=outcome
        )

        self.observations.append(observation)

        agent = self.agents[agent_id]
        agent.last_interaction = time.time()
        agent.interaction_history.append(observation)

        self.interaction_count += 1

        # Infer mental states from observation
        inferences = self._infer_mental_states(agent, observation)

        # Update prediction accuracy
        self._update_prediction_accuracy(agent, action, context)

        return inferences

    def _infer_mental_states(
        self,
        agent: AgentModel,
        observation: Observation
    ) -> Dict[str, Any]:
        """Infer mental states from observation"""
        self.total_inferences += 1

        inferences = {}

        # Combined input for inference
        combined = np.concatenate([observation.action, observation.context])
        combined = combined[:self.embedding_dim]
        if len(combined) < self.embedding_dim:
            combined = np.pad(combined, (0, self.embedding_dim - len(combined)))

        # Infer intention
        intention_embedding = self.inference_weights @ combined + self.inference_bias
        intention_embedding = np.tanh(intention_embedding)

        intention = MentalState(
            state_type=MentalStateType.INTENTION,
            content=intention_embedding,
            confidence=0.7,
            source="action_inference"
        )
        agent.mental_states['current_intention'] = intention
        agent.intentions.append(intention_embedding)
        if len(agent.intentions) > 10:
            agent.intentions.pop(0)

        inferences['intention'] = {
            'embedding': intention_embedding.tolist(),
            'confidence': intention.confidence
        }

        # Infer beliefs about context
        # What the agent believes about the situation
        belief_embedding = observation.context + 0.1 * np.random.randn(self.embedding_dim)
        belief_embedding /= np.linalg.norm(belief_embedding) + 1e-8

        belief = MentalState(
            state_type=MentalStateType.BELIEF,
            content=belief_embedding,
            confidence=0.6,
            source="context_inference"
        )
        agent.mental_states['current_belief'] = belief
        agent.beliefs['context'] = belief_embedding

        inferences['belief'] = {
            'embedding': belief_embedding.tolist(),
            'confidence': belief.confidence
        }

        # Infer desires from repeated actions
        if len(agent.interaction_history) >= 3:
            recent_actions = [obs.action for obs in list(agent.interaction_history)[-5:]]
            avg_action = np.mean(recent_actions, axis=0)

            desire_embedding = self.inference_weights @ avg_action
            desire_embedding = np.tanh(desire_embedding)

            desire = MentalState(
                state_type=MentalStateType.DESIRE,
                content=desire_embedding,
                confidence=0.5,
                source="pattern_inference"
            )
            agent.mental_states['inferred_desire'] = desire
            agent.desires.append(desire_embedding)
            if len(agent.desires) > 5:
                agent.desires.pop(0)

            inferences['desire'] = {
                'embedding': desire_embedding.tolist(),
                'confidence': desire.confidence
            }

        return inferences

    def _update_prediction_accuracy(
        self,
        agent: AgentModel,
        actual_action: np.ndarray,
        context: np.ndarray
    ):
        """Update how accurately we predicted this action"""
        if 'predicted_action' in agent.mental_states:
            predicted = agent.mental_states['predicted_action'].content
            similarity = np.dot(predicted, actual_action) / (
                np.linalg.norm(predicted) * np.linalg.norm(actual_action) + 1e-8
            )

            # Update predictability
            agent.predictability = 0.9 * agent.predictability + 0.1 * similarity

            if similarity > 0.7:
                self.successful_predictions += 1

            self.total_predictions += 1

            # Hebbian update of prediction weights
            error = actual_action - predicted
            self.prediction_weights += self.learning_rate * np.outer(error, context)

    def predict_action(
        self,
        agent_id: str,
        context: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """
        Predict what action an agent will take.

        Args:
            agent_id: ID of agent to predict
            context: Current context embedding

        Returns:
            Tuple of (predicted_action, confidence)
        """
        if agent_id not in self.agents:
            return np.zeros(self.embedding_dim), 0.0

        agent = self.agents[agent_id]

        context = np.asarray(context).flatten()[:self.embedding_dim]
        if len(context) < self.embedding_dim:
            context = np.pad(context, (0, self.embedding_dim - len(context)))

        # Combine context with agent's mental states
        combined = context.copy()

        if 'current_intention' in agent.mental_states:
            combined += 0.3 * agent.mental_states['current_intention'].content

        if agent.desires:
            combined += 0.2 * agent.desires[-1]

        # Predict action
        predicted = self.prediction_weights @ combined + self.prediction_bias
        predicted = np.tanh(predicted)
        predicted /= np.linalg.norm(predicted) + 1e-8

        # Store prediction for later accuracy check
        agent.mental_states['predicted_action'] = MentalState(
            state_type=MentalStateType.INTENTION,
            content=predicted,
            confidence=agent.predictability,
            source="prediction"
        )

        return predicted, agent.predictability

    def infer_false_belief(
        self,
        agent_id: str,
        true_state: np.ndarray,
        agent_observation: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Infer if an agent holds a false belief (classic false belief task).

        Args:
            agent_id: ID of agent
            true_state: True state of the world
            agent_observation: What the agent actually observed (may differ)

        Returns:
            Dictionary with false belief analysis
        """
        if agent_id not in self.agents:
            return {'error': 'Agent not found'}

        agent = self.agents[agent_id]

        true_state = np.asarray(true_state).flatten()[:self.embedding_dim]
        if len(true_state) < self.embedding_dim:
            true_state = np.pad(true_state, (0, self.embedding_dim - len(true_state)))

        # If agent didn't observe, their belief may be outdated
        if agent_observation is not None:
            agent_observation = np.asarray(agent_observation).flatten()[:self.embedding_dim]
            if len(agent_observation) < self.embedding_dim:
                agent_observation = np.pad(agent_observation, (0, self.embedding_dim - len(agent_observation)))
            agent_belief = agent_observation
        elif 'context' in agent.beliefs:
            agent_belief = agent.beliefs['context']
        else:
            agent_belief = np.zeros(self.embedding_dim)

        # Compute divergence between true state and agent's belief
        divergence = np.linalg.norm(true_state - agent_belief)
        similarity = np.dot(true_state, agent_belief) / (
            np.linalg.norm(true_state) * np.linalg.norm(agent_belief) + 1e-8
        )

        # Determine if this constitutes a false belief
        has_false_belief = divergence > 0.5 or similarity < 0.7

        return {
            'agent_id': agent_id,
            'has_false_belief': has_false_belief,
            'divergence': float(divergence),
            'similarity': float(similarity),
            'agent_belief': agent_belief.tolist(),
            'true_state': true_state.tolist(),
            'confidence': agent.predictability
        }

    def simulate_perspective(
        self,
        agent_id: str,
        scenario: np.ndarray,
        steps: int = None
    ) -> List[np.ndarray]:
        """
        Simulate what an agent would think/do in a scenario.
        Uses simulation theory - running self-model with agent's parameters.
        """
        if agent_id not in self.agents:
            return []

        agent = self.agents[agent_id]
        steps = steps or self.simulation_steps

        scenario = np.asarray(scenario).flatten()[:self.embedding_dim]
        if len(scenario) < self.embedding_dim:
            scenario = np.pad(scenario, (0, self.embedding_dim - len(scenario)))

        trajectory = []
        state = scenario.copy()

        for _ in range(steps):
            # Transform using inference model but biased by agent's personality
            if agent.personality_embedding is not None:
                state = 0.8 * state + 0.2 * agent.personality_embedding

            # Predict next thought/action
            next_state = self.inference_weights @ state + self.inference_bias
            next_state = np.tanh(next_state)

            # Add agent-specific bias from desires
            if agent.desires:
                next_state += 0.1 * agent.desires[-1]

            next_state /= np.linalg.norm(next_state) + 1e-8
            trajectory.append(next_state)
            state = next_state

        return trajectory

    def reason_cooperatively(
        self,
        agent_ids: List[str],
        shared_goal: np.ndarray,
        context: np.ndarray
    ) -> Dict[str, Any]:
        """
        Reason about cooperative action with other agents.

        Args:
            agent_ids: IDs of cooperating agents
            shared_goal: Shared goal embedding
            context: Current context

        Returns:
            Cooperative reasoning results
        """
        shared_goal = np.asarray(shared_goal).flatten()[:self.embedding_dim]
        if len(shared_goal) < self.embedding_dim:
            shared_goal = np.pad(shared_goal, (0, self.embedding_dim - len(shared_goal)))

        context = np.asarray(context).flatten()[:self.embedding_dim]
        if len(context) < self.embedding_dim:
            context = np.pad(context, (0, self.embedding_dim - len(context)))

        results = {
            'agents': [],
            'coordination_quality': 0.0,
            'trust_levels': {},
            'recommended_strategy': None
        }

        agent_predictions = []

        for agent_id in agent_ids:
            if agent_id not in self.agents:
                continue

            agent = self.agents[agent_id]

            # Predict agent's action toward goal
            predicted_action, confidence = self.predict_action(agent_id, context)

            # Check alignment with shared goal
            goal_alignment = np.dot(predicted_action, shared_goal)

            agent_predictions.append({
                'agent_id': agent_id,
                'predicted_action': predicted_action,
                'confidence': confidence,
                'goal_alignment': goal_alignment,
                'trust': agent.trust_level
            })

            results['trust_levels'][agent_id] = agent.trust_level

        if agent_predictions:
            # Compute coordination quality
            alignments = [p['goal_alignment'] for p in agent_predictions]
            trusts = [p['trust'] for p in agent_predictions]

            results['coordination_quality'] = np.mean(alignments) * np.mean(trusts)
            results['agents'] = agent_predictions

            # Recommend strategy based on trust and alignment
            if results['coordination_quality'] > 0.7:
                results['recommended_strategy'] = 'full_cooperation'
            elif results['coordination_quality'] > 0.4:
                results['recommended_strategy'] = 'cautious_cooperation'
            else:
                results['recommended_strategy'] = 'independent_action'

        return results

    def reason_competitively(
        self,
        opponent_id: str,
        own_goal: np.ndarray,
        opponent_goal: Optional[np.ndarray] = None,
        context: np.ndarray = None
    ) -> Dict[str, Any]:
        """
        Reason about competitive interaction with another agent.
        """
        if opponent_id not in self.agents:
            return {'error': 'Opponent not found'}

        opponent = self.agents[opponent_id]

        own_goal = np.asarray(own_goal).flatten()[:self.embedding_dim]
        if len(own_goal) < self.embedding_dim:
            own_goal = np.pad(own_goal, (0, self.embedding_dim - len(own_goal)))

        if context is None:
            context = np.zeros(self.embedding_dim)
        else:
            context = np.asarray(context).flatten()[:self.embedding_dim]
            if len(context) < self.embedding_dim:
                context = np.pad(context, (0, self.embedding_dim - len(context)))

        # Infer opponent's goal if not provided
        if opponent_goal is None and opponent.desires:
            opponent_goal = opponent.desires[-1]
        elif opponent_goal is None:
            opponent_goal = np.zeros(self.embedding_dim)
        else:
            opponent_goal = np.asarray(opponent_goal).flatten()[:self.embedding_dim]
            if len(opponent_goal) < self.embedding_dim:
                opponent_goal = np.pad(opponent_goal, (0, self.embedding_dim - len(opponent_goal)))

        # Predict opponent's action
        opponent_action, opponent_confidence = self.predict_action(opponent_id, context)

        # Goal conflict analysis
        goal_conflict = -np.dot(own_goal, opponent_goal)  # Negative = opposing

        # Strategic analysis
        if goal_conflict > 0.5:
            # Strong opposition
            # Counter-strategy: opposite of predicted opponent action
            counter_action = -opponent_action
            counter_action /= np.linalg.norm(counter_action) + 1e-8
            strategy = 'counter'
        elif goal_conflict > 0:
            # Mild opposition - be unpredictable
            noise = np.random.randn(self.embedding_dim)
            counter_action = own_goal + 0.3 * noise
            counter_action /= np.linalg.norm(counter_action) + 1e-8
            strategy = 'unpredictable'
        else:
            # Goals align - can potentially cooperate
            counter_action = own_goal
            strategy = 'potential_cooperation'

        return {
            'opponent_id': opponent_id,
            'predicted_opponent_action': opponent_action.tolist(),
            'opponent_confidence': opponent_confidence,
            'goal_conflict': float(goal_conflict),
            'recommended_action': counter_action.tolist(),
            'strategy': strategy,
            'opponent_predictability': opponent.predictability
        }

    def learn_from_observation(
        self,
        demonstrator_id: str,
        demonstration: List[Tuple[np.ndarray, np.ndarray]]
    ) -> Dict[str, Any]:
        """
        Social learning: learn from observing another agent's behavior.

        Args:
            demonstrator_id: ID of agent being observed
            demonstration: List of (context, action) tuples

        Returns:
            Learning results
        """
        if demonstrator_id not in self.agents:
            demonstrator_id = self.add_agent()

        agent = self.agents[demonstrator_id]

        learned_patterns = 0

        for context, action in demonstration:
            context = np.asarray(context).flatten()[:self.embedding_dim]
            if len(context) < self.embedding_dim:
                context = np.pad(context, (0, self.embedding_dim - len(context)))

            action = np.asarray(action).flatten()[:self.embedding_dim]
            if len(action) < self.embedding_dim:
                action = np.pad(action, (0, self.embedding_dim - len(action)))

            # Observe the demonstration
            self.observe(demonstrator_id, action, context)

            # Update prediction model (imitation learning via Hebbian rule)
            # Strengthen context -> action mapping
            delta = self.learning_rate * np.outer(action, context)
            self.prediction_weights += delta
            self.prediction_weights /= np.linalg.norm(self.prediction_weights) + 1e-8

            learned_patterns += 1

        # Increase trust in demonstrator (they're teaching us)
        agent.trust_level = min(1.0, agent.trust_level + 0.1)

        return {
            'demonstrator_id': demonstrator_id,
            'patterns_learned': learned_patterns,
            'updated_trust': agent.trust_level
        }

    def update_trust(
        self,
        agent_id: str,
        interaction_outcome: float
    ):
        """Update trust level based on interaction outcome"""
        if agent_id not in self.agents:
            return

        agent = self.agents[agent_id]

        # Bayesian-like trust update
        if interaction_outcome > 0:
            agent.trust_level = min(1.0, agent.trust_level + 0.1 * interaction_outcome)
        else:
            agent.trust_level = max(0.0, agent.trust_level + 0.1 * interaction_outcome)

    def set_relationship(self, agent_id: str, relationship: SocialRelation):
        """Set relationship type with an agent"""
        if agent_id in self.agents:
            self.agents[agent_id].relationship = relationship

    def get_stats(self) -> Dict[str, Any]:
        """Get Theory of Mind statistics"""
        return {
            'total_agents': len(self.agents),
            'total_inferences': self.total_inferences,
            'total_predictions': self.total_predictions,
            'successful_predictions': self.successful_predictions,
            'prediction_accuracy': (self.successful_predictions /
                                   max(1, self.total_predictions)),
            'interaction_count': self.interaction_count,
            'observation_history_size': len(self.observations),
            'avg_trust': np.mean([a.trust_level for a in self.agents.values()])
                        if self.agents else 0.0,
            'avg_predictability': np.mean([a.predictability for a in self.agents.values()])
                                 if self.agents else 0.0
        }

    def serialize(self) -> Dict[str, Any]:
        """Serialize Theory of Mind module to dictionary"""
        return {
            'embedding_dim': self.embedding_dim,
            'max_agents': self.max_agents,
            'belief_update_rate': self.belief_update_rate,
            'simulation_steps': self.simulation_steps,
            'learning_rate': self.learning_rate,
            'agents': {
                aid: {
                    'agent_id': a.agent_id,
                    'agent_type': a.agent_type.value,
                    'personality_embedding': a.personality_embedding.tolist()
                        if a.personality_embedding is not None else None,
                    'relationship': a.relationship.value,
                    'trust_level': a.trust_level,
                    'predictability': a.predictability,
                    'beliefs': {k: v.tolist() for k, v in a.beliefs.items()},
                    'desires': [d.tolist() for d in a.desires]
                }
                for aid, a in self.agents.items()
            },
            'inference_weights': self.inference_weights.tolist(),
            'inference_bias': self.inference_bias.tolist(),
            'prediction_weights': self.prediction_weights.tolist(),
            'prediction_bias': self.prediction_bias.tolist(),
            'stats': self.get_stats()
        }

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> 'TheoryOfMind':
        """Deserialize Theory of Mind module from dictionary"""
        tom = cls(
            embedding_dim=data['embedding_dim'],
            max_agents=data['max_agents'],
            belief_update_rate=data['belief_update_rate'],
            simulation_steps=data['simulation_steps'],
            learning_rate=data['learning_rate']
        )

        tom.inference_weights = np.array(data['inference_weights'])
        tom.inference_bias = np.array(data['inference_bias'])
        tom.prediction_weights = np.array(data['prediction_weights'])
        tom.prediction_bias = np.array(data['prediction_bias'])

        for aid, adata in data.get('agents', {}).items():
            agent = AgentModel(
                agent_id=adata['agent_id'],
                agent_type=AgentType(adata['agent_type']),
                personality_embedding=np.array(adata['personality_embedding'])
                    if adata['personality_embedding'] else None,
                relationship=SocialRelation(adata['relationship']),
                trust_level=adata['trust_level'],
                predictability=adata['predictability']
            )
            agent.beliefs = {k: np.array(v) for k, v in adata['beliefs'].items()}
            agent.desires = [np.array(d) for d in adata['desires']]
            tom.agents[aid] = agent

        return tom
