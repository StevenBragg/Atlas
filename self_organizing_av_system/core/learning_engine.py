"""
Learning Engine for Challenge-Based Learning

This module executes the learning process using biology-inspired
local plasticity rules. NO backpropagation is used.

Learning Strategies (all use local rules):
- Hebbian: "Neurons that fire together wire together"
- STDP: Spike-timing dependent plasticity
- Oja: Normalized Hebbian with decay
- BCM: Bienenstock-Cooper-Munro sliding threshold
- Anti-Hebbian: Decorrelation learning
- Competitive: Winner-take-all
- Cooperative: Ensemble coordination
"""

import time
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging

from .challenge import (
    Challenge,
    ChallengeType,
    ChallengeStatus,
    TrainingData,
    LearningResult,
    LearnedCapability,
)
from .meta_learning import MetaLearner, LearningStrategy
from .goal_planning import GoalPlanningSystem, GoalType, Goal
from .episodic_memory import EpisodicMemory, Episode
from .progress_tracker import ProgressTracker

# Optional import - semantic memory requires networkx
try:
    from .semantic_memory import SemanticMemory, RelationType
    HAS_SEMANTIC_MEMORY = True
except ImportError:
    HAS_SEMANTIC_MEMORY = False
    SemanticMemory = None

logger = logging.getLogger(__name__)


@dataclass
class LearningState:
    """Current state of a learning session."""
    challenge: Challenge
    strategy: LearningStrategy
    hyperparameters: Dict[str, float]
    weights: np.ndarray
    iteration: int = 0
    best_accuracy: float = 0.0
    plateau_count: int = 0
    start_time: float = field(default_factory=time.time)


class LearningEngine:
    """
    Executes biology-inspired learning for challenges.

    Uses local plasticity rules (Hebbian, STDP, BCM, etc.) to learn
    from data. Integrates with MetaLearner for strategy selection
    and adaptation.
    """

    def __init__(
        self,
        state_dim: int = 128,
        meta_learner: Optional[MetaLearner] = None,
        episodic_memory: Optional[EpisodicMemory] = None,
        semantic_memory: Optional[SemanticMemory] = None,
        progress_tracker: Optional[ProgressTracker] = None,
        learning_rate: float = 0.01,
        batch_size: int = 32,
        max_plateau_count: int = 5,
    ):
        """
        Initialize the learning engine.

        Args:
            state_dim: Dimension of internal state representations
            meta_learner: MetaLearner for strategy selection
            episodic_memory: EpisodicMemory for experience storage
            semantic_memory: SemanticMemory for knowledge extraction
            progress_tracker: ProgressTracker for metrics
            learning_rate: Base learning rate for plasticity
            batch_size: Batch size for training
            max_plateau_count: Max plateaus before strategy switch
        """
        self.state_dim = state_dim
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_plateau_count = max_plateau_count

        # Initialize or use provided systems
        self.meta_learner = meta_learner or MetaLearner(
            num_strategies=7,
        )
        self.episodic_memory = episodic_memory or EpisodicMemory(
            state_size=state_dim,
        )
        # Semantic memory is optional (requires networkx)
        if HAS_SEMANTIC_MEMORY:
            self.semantic_memory = semantic_memory or SemanticMemory(
                embedding_dim=state_dim,
            )
        else:
            self.semantic_memory = None
            logger.warning("SemanticMemory not available (install networkx)")
        self.progress_tracker = progress_tracker or ProgressTracker()

        # Active learning sessions
        self.active_sessions: Dict[str, LearningState] = {}

        # Learned capabilities
        self.capabilities: Dict[str, LearnedCapability] = {}

        logger.info(f"LearningEngine initialized with state_dim={state_dim}")

    def create_learning_goal(self, challenge: Challenge) -> Goal:
        """
        Create a learning goal from a challenge.

        Args:
            challenge: The challenge to learn

        Returns:
            Goal object for the goal planning system
        """
        goal_planner = GoalPlanningSystem(state_dim=self.state_dim)

        goal = goal_planner.generate_goal(
            goal_type=GoalType.LEARNING,
            context={
                'challenge_id': challenge.id,
                'challenge_name': challenge.name,
                'challenge_type': challenge.challenge_type.name,
                'difficulty': challenge.difficulty,
                'target_accuracy': challenge.success_criteria.accuracy,
            }
        )

        return goal

    def select_strategy(
        self,
        challenge: Challenge,
    ) -> Tuple[LearningStrategy, Dict[str, float]]:
        """
        Select the best learning strategy for a challenge.

        Args:
            challenge: The challenge to learn

        Returns:
            Tuple of (strategy, hyperparameters)
        """
        task_characteristics = challenge.to_task_characteristics()
        strategy, hyperparameters = self.meta_learner.select_strategy(
            task_characteristics
        )

        logger.info(
            f"Selected strategy {strategy.name} for challenge {challenge.name}"
        )

        return strategy, hyperparameters

    def execute_learning_loop(
        self,
        challenge: Challenge,
        strategy: Optional[LearningStrategy] = None,
        callback: Optional[Callable[[int, float], None]] = None,
    ) -> LearningResult:
        """
        Execute the main learning loop for a challenge.

        Uses local plasticity rules (no backpropagation).

        Args:
            challenge: The challenge to learn
            strategy: Optional specific strategy (auto-select if None)
            callback: Optional callback(iteration, accuracy) for progress

        Returns:
            LearningResult with metrics and learned capability
        """
        # Start tracking
        self.progress_tracker.start_challenge(challenge)
        challenge.start()

        # Select strategy if not provided
        if strategy is None:
            strategy, hyperparameters = self.select_strategy(challenge)
        else:
            _, hyperparameters = self.meta_learner.select_strategy(
                challenge.to_task_characteristics()
            )

        # Initialize weights
        weights = self._initialize_weights(challenge)

        # Create learning state
        state = LearningState(
            challenge=challenge,
            strategy=strategy,
            hyperparameters=hyperparameters,
            weights=weights,
        )
        self.active_sessions[challenge.id] = state

        # Learning curve tracking
        learning_curve = []
        start_time = time.time()

        try:
            challenge.status = ChallengeStatus.LEARNING

            while True:
                # Get batch of training data
                batch_x, batch_y = self._get_training_batch(challenge)

                # Apply local plasticity rule
                accuracy, weights = self._apply_plasticity(
                    state, batch_x, batch_y
                )

                state.weights = weights
                state.iteration += 1

                # Track progress
                learning_curve.append(accuracy)

                progress_info = self.progress_tracker.update_progress(
                    challenge.id,
                    accuracy=accuracy,
                    strategy=strategy.name,
                )

                # Store experience in episodic memory
                self._store_experience(challenge, accuracy, strategy)

                # Callback for progress updates
                if callback:
                    callback(state.iteration, accuracy)

                # Check for plateau and adapt strategy
                if progress_info['should_adapt']:
                    state.plateau_count += 1
                    if state.plateau_count >= self.max_plateau_count:
                        # Switch strategy
                        new_strategy, new_hyperparams = self._adapt_strategy(
                            challenge, accuracy
                        )
                        state.strategy = new_strategy
                        state.hyperparameters = new_hyperparams
                        state.plateau_count = 0
                        challenge.status = ChallengeStatus.ADAPTING
                        logger.info(
                            f"Adapted to strategy {new_strategy.name} "
                            f"at iteration {state.iteration}"
                        )
                        challenge.status = ChallengeStatus.LEARNING

                # Update best accuracy
                if accuracy > state.best_accuracy:
                    state.best_accuracy = accuracy

                # Check completion criteria
                is_complete, reason = self.progress_tracker.check_completion(
                    challenge.id,
                    challenge.success_criteria,
                )

                if is_complete:
                    logger.info(f"Challenge complete: {reason}")
                    break

            # Determine success
            success = state.best_accuracy >= challenge.success_criteria.accuracy
            challenge.complete(success)

            # Consolidate learning
            challenge.status = ChallengeStatus.CONSOLIDATING
            capability = self._consolidate_learning(challenge, state, success)

            # Update meta-learner with results
            self._update_meta_learner(challenge, state, success)

            # Update progress tracker
            self.progress_tracker.complete_challenge(
                challenge.id,
                success=success,
                final_accuracy=state.best_accuracy,
            )

            duration = time.time() - start_time

            return LearningResult(
                challenge_id=challenge.id,
                challenge_name=challenge.name,
                success=success,
                accuracy=state.best_accuracy,
                iterations=state.iteration,
                duration_seconds=duration,
                strategy_used=state.strategy.name,
                learning_curve=learning_curve,
                final_metrics={
                    'best_accuracy': state.best_accuracy,
                    'final_accuracy': accuracy,
                    'plateau_count': state.plateau_count,
                },
                capability_id=capability.id if capability else None,
            )

        except Exception as e:
            logger.error(f"Learning failed: {e}")
            challenge.status = ChallengeStatus.FAILED
            duration = time.time() - start_time

            return LearningResult(
                challenge_id=challenge.id,
                challenge_name=challenge.name,
                success=False,
                accuracy=state.best_accuracy,
                iterations=state.iteration,
                duration_seconds=duration,
                strategy_used=state.strategy.name,
                learning_curve=learning_curve,
                error_message=str(e),
            )

        finally:
            # Clean up
            if challenge.id in self.active_sessions:
                del self.active_sessions[challenge.id]

    def _initialize_weights(self, challenge: Challenge) -> np.ndarray:
        """Initialize weights for learning."""
        if challenge.training_data and challenge.training_data.feature_dim:
            input_dim = challenge.training_data.feature_dim
        else:
            input_dim = self.state_dim

        if challenge.training_data and challenge.training_data.num_classes:
            output_dim = challenge.training_data.num_classes
        else:
            output_dim = self.state_dim

        # Xavier-like initialization scaled for Hebbian
        scale = np.sqrt(2.0 / (input_dim + output_dim))
        weights = np.random.randn(output_dim, input_dim) * scale

        return weights

    def _get_training_batch(
        self,
        challenge: Challenge,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Get a batch of training data."""
        if challenge.training_data:
            samples, labels = challenge.training_data.get_batch(self.batch_size)

            # Convert to numpy arrays
            batch_x = np.array(samples)
            batch_y = np.array(labels) if labels else None

            return batch_x, batch_y
        else:
            # Generate synthetic data for demonstration
            batch_x = np.random.randn(self.batch_size, self.state_dim)
            batch_y = np.random.randint(0, 10, self.batch_size)
            return batch_x, batch_y

    def _apply_plasticity(
        self,
        state: LearningState,
        batch_x: np.ndarray,
        batch_y: Optional[np.ndarray],
    ) -> Tuple[float, np.ndarray]:
        """
        Apply local plasticity rule based on selected strategy.

        All rules use LOCAL updates only - no backpropagation.

        Args:
            state: Current learning state
            batch_x: Input batch
            batch_y: Label batch (optional)

        Returns:
            Tuple of (accuracy, updated_weights)
        """
        strategy = state.strategy
        weights = state.weights.copy()
        lr = self.learning_rate * state.hyperparameters.get('learning_rate', 1.0)

        # Normalize input
        if batch_x.ndim == 1:
            batch_x = batch_x.reshape(1, -1)

        # Ensure weights match input dimension
        if weights.shape[1] != batch_x.shape[1]:
            # Resize weights
            new_weights = np.random.randn(weights.shape[0], batch_x.shape[1]) * 0.01
            weights = new_weights

        # Compute activations
        activations = batch_x @ weights.T  # (batch, output_dim)

        # Apply learning rule based on strategy
        if strategy == LearningStrategy.HEBBIAN:
            weights = self._hebbian_update(weights, batch_x, activations, lr)

        elif strategy == LearningStrategy.OJA:
            weights = self._oja_update(weights, batch_x, activations, lr)

        elif strategy == LearningStrategy.STDP:
            weights = self._stdp_update(weights, batch_x, activations, lr)

        elif strategy == LearningStrategy.BCM:
            theta = state.hyperparameters.get('bcm_threshold', 0.5)
            weights = self._bcm_update(weights, batch_x, activations, lr, theta)

        elif strategy == LearningStrategy.ANTI_HEBBIAN:
            weights = self._anti_hebbian_update(weights, batch_x, activations, lr)

        elif strategy == LearningStrategy.COMPETITIVE:
            weights = self._competitive_update(weights, batch_x, activations, lr)

        elif strategy == LearningStrategy.COOPERATIVE:
            weights = self._cooperative_update(weights, batch_x, activations, lr)

        # Calculate accuracy
        accuracy = self._calculate_accuracy(weights, batch_x, batch_y)

        return accuracy, weights

    def _hebbian_update(
        self,
        weights: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        lr: float,
    ) -> np.ndarray:
        """
        Hebbian learning: "Neurons that fire together wire together"

        ΔW = lr * y.T @ x
        """
        dW = lr * (y.T @ x) / x.shape[0]
        weights += dW

        # Normalize to prevent explosion
        norms = np.linalg.norm(weights, axis=1, keepdims=True)
        weights = weights / (norms + 1e-8)

        return weights

    def _oja_update(
        self,
        weights: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        lr: float,
    ) -> np.ndarray:
        """
        Oja's rule: Normalized Hebbian with decay

        ΔW = lr * y * (x - y * W)
        """
        for i in range(x.shape[0]):
            xi = x[i:i+1]  # (1, input_dim)
            yi = y[i:i+1]  # (1, output_dim)

            # Oja update
            dW = lr * yi.T @ (xi - yi @ weights)
            weights += dW

        return weights

    def _stdp_update(
        self,
        weights: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        lr: float,
    ) -> np.ndarray:
        """
        Spike-timing dependent plasticity (simplified)

        Potentiation when pre fires before post, depression otherwise.
        """
        # Simulate spike timing with sign of correlation
        pre_post = y.T @ x  # Post-pre correlation
        post_pre = x.T @ y  # Pre-post correlation

        # STDP: potentiate positive correlation, depress negative
        dW = lr * (pre_post - 0.5 * post_pre.T) / x.shape[0]
        weights += dW

        # Clip weights
        weights = np.clip(weights, -2, 2)

        return weights

    def _bcm_update(
        self,
        weights: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        lr: float,
        theta: float,
    ) -> np.ndarray:
        """
        BCM (Bienenstock-Cooper-Munro) rule with sliding threshold

        ΔW = lr * y * (y - θ) * x
        """
        y_mean = np.mean(y, axis=0, keepdims=True)

        for i in range(x.shape[0]):
            xi = x[i:i+1]
            yi = y[i:i+1]

            # BCM: modulate by (y - threshold)
            modulation = yi * (yi - theta)
            dW = lr * modulation.T @ xi
            weights += dW

        # Update sliding threshold toward mean activity
        theta = 0.9 * theta + 0.1 * np.mean(y_mean ** 2)

        # Normalize
        norms = np.linalg.norm(weights, axis=1, keepdims=True)
        weights = weights / (norms + 1e-8)

        return weights

    def _anti_hebbian_update(
        self,
        weights: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        lr: float,
    ) -> np.ndarray:
        """
        Anti-Hebbian: Decorrelation learning

        ΔW = -lr * y.T @ x (opposite of Hebbian)
        """
        dW = -lr * (y.T @ x) / x.shape[0]
        weights += dW

        # Keep weights bounded
        weights = np.clip(weights, -2, 2)

        return weights

    def _competitive_update(
        self,
        weights: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        lr: float,
    ) -> np.ndarray:
        """
        Competitive learning: Winner-take-all

        Only the most active unit learns.
        """
        for i in range(x.shape[0]):
            xi = x[i:i+1]
            yi = y[i]

            # Find winner (most active unit)
            winner = np.argmax(yi)

            # Only winner learns
            dW = lr * (xi - weights[winner:winner+1])
            weights[winner:winner+1] += dW

        return weights

    def _cooperative_update(
        self,
        weights: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        lr: float,
    ) -> np.ndarray:
        """
        Cooperative learning: Ensemble coordination

        Units near the winner also learn (neighborhood function).
        """
        for i in range(x.shape[0]):
            xi = x[i:i+1]
            yi = y[i]

            # Find winner
            winner = np.argmax(yi)

            # Create neighborhood function (Gaussian)
            indices = np.arange(weights.shape[0])
            neighborhood = np.exp(-0.5 * ((indices - winner) / 2) ** 2)
            neighborhood = neighborhood.reshape(-1, 1)

            # All units learn proportionally to neighborhood
            dW = lr * neighborhood * (xi - weights)
            weights += dW

        return weights

    def _calculate_accuracy(
        self,
        weights: np.ndarray,
        x: np.ndarray,
        y: Optional[np.ndarray],
    ) -> float:
        """Calculate accuracy of current weights."""
        if y is None:
            # Unsupervised: use reconstruction error as proxy
            activations = x @ weights.T
            reconstruction = activations @ weights
            error = np.mean((x - reconstruction) ** 2)
            return max(0, 1 - error)

        # Supervised: classification accuracy
        activations = x @ weights.T
        predictions = np.argmax(activations, axis=1)

        if y.ndim > 1:
            targets = np.argmax(y, axis=1)
        else:
            targets = y

        # Handle case where targets exceed weight dimensions
        valid_mask = targets < weights.shape[0]
        if not np.any(valid_mask):
            return 0.0

        accuracy = np.mean(predictions[valid_mask] == targets[valid_mask])
        return float(accuracy)

    def _adapt_strategy(
        self,
        challenge: Challenge,
        current_accuracy: float,
    ) -> Tuple[LearningStrategy, Dict[str, float]]:
        """Adapt learning strategy when plateaued."""
        # Get task characteristics
        task_chars = challenge.to_task_characteristics()

        # Update meta-learner with poor performance to encourage exploration
        self.meta_learner.update(
            strategy=self.active_sessions[challenge.id].strategy,
            hyperparameters=self.active_sessions[challenge.id].hyperparameters,
            task_characteristics=task_chars,
            performance_metrics={'accuracy': current_accuracy * 0.5},  # Penalize
        )

        # Select new strategy
        return self.meta_learner.select_strategy(task_chars)

    def _store_experience(
        self,
        challenge: Challenge,
        accuracy: float,
        strategy: LearningStrategy,
    ) -> None:
        """Store learning experience in episodic memory."""
        state = np.random.randn(self.state_dim)  # Abstract state representation

        # Sensory data should be numpy arrays for episodic memory
        sensory_data = {
            'challenge_embedding': np.random.randn(self.state_dim),  # Abstract embedding
            'performance': np.array([accuracy]),
        }
        context = {
            'challenge_id': challenge.id,
            'challenge_type': challenge.challenge_type.name,
            'strategy': strategy.name,
            'accuracy': accuracy,
        }

        # Higher surprise for unexpected performance
        expected_accuracy = challenge.success_criteria.accuracy * 0.5
        surprise = abs(accuracy - expected_accuracy)

        self.episodic_memory.store(
            state=state,
            sensory_data=sensory_data,
            context=context,
            emotional_valence=accuracy - 0.5,  # Positive if good
            surprise_level=surprise,
        )

    def _consolidate_learning(
        self,
        challenge: Challenge,
        state: LearningState,
        success: bool,
    ) -> Optional[LearnedCapability]:
        """Consolidate learned knowledge into a capability."""
        if not success:
            return None

        # Create capability
        capability = LearnedCapability(
            name=f"capability_{challenge.name}",
            description=f"Learned from challenge: {challenge.description}",
            challenge_id=challenge.id,
            challenge_type=challenge.challenge_type,
            modalities=challenge.modalities,
            proficiency=state.best_accuracy,
            weights=state.weights.copy(),
            metadata={
                'strategy': state.strategy.name,
                'iterations': state.iteration,
            },
        )

        self.capabilities[capability.id] = capability

        # Add to semantic memory (if available)
        if self.semantic_memory is not None:
            self.semantic_memory.add_concept(
                name=capability.name,
                embedding=np.mean(state.weights, axis=0),
                attributes={
                    'type': 'learned_capability',
                    'challenge_type': challenge.challenge_type.name,
                    'proficiency': capability.proficiency,
                },
            )

        # Consolidate episodic memories
        self.episodic_memory.consolidate(n_replay=10)

        logger.info(f"Consolidated capability: {capability.name}")

        return capability

    def _update_meta_learner(
        self,
        challenge: Challenge,
        state: LearningState,
        success: bool,
    ) -> None:
        """Update meta-learner with learning results."""
        task_chars = challenge.to_task_characteristics()

        self.meta_learner.update(
            strategy=state.strategy,
            hyperparameters=state.hyperparameters,
            task_characteristics=task_chars,
            performance_metrics={
                'accuracy': state.best_accuracy,
                'prediction_error': 1 - state.best_accuracy,
            },
        )

    def get_capability(self, capability_id: str) -> Optional[LearnedCapability]:
        """Get a learned capability by ID."""
        return self.capabilities.get(capability_id)

    def list_capabilities(self) -> List[LearnedCapability]:
        """List all learned capabilities."""
        return list(self.capabilities.values())
