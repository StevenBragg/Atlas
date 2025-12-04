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

from .backend import xp, to_cpu, to_gpu, HAS_GPU

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
from .self_organizing_network import SelfOrganizingNetwork, StructuralEventType

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
    structural_changes: List[Dict[str, Any]] = field(default_factory=list)


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
        learning_rate: float = 0.1,  # Increased from 0.01 for faster convergence
        batch_size: int = 32,
        max_plateau_count: int = 15,  # Increased from 5 to avoid rapid strategy switching
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
                embedding_size=state_dim,
            )
        else:
            self.semantic_memory = None
            logger.warning("SemanticMemory not available (install networkx)")
        self.progress_tracker = progress_tracker or ProgressTracker()

        # Active learning sessions
        self.active_sessions: Dict[str, LearningState] = {}

        # Learned capabilities
        self.capabilities: Dict[str, LearnedCapability] = {}

        # Self-organizing neural network with dynamic topology
        self.network = SelfOrganizingNetwork(
            input_dim=state_dim,
            initial_layer_sizes=[64, 32],  # Start small, will grow
            output_dim=state_dim,
            learning_rate=learning_rate,
            enable_structural_plasticity=True,
        )

        # Callbacks for structural changes
        self._structural_change_callbacks: List[Callable[[Dict[str, Any]], None]] = []

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
            logger.info(f"Starting learning loop for {challenge.name}")

            while True:
                # Get batch of training data
                batch_x, batch_y = self._get_training_batch(challenge)

                if state.iteration == 0:
                    logger.info(f"First batch shape: x={batch_x.shape}, y={batch_y.shape if batch_y is not None else None}")

                # Apply local plasticity rule
                accuracy, weights = self._apply_plasticity(
                    state, batch_x, batch_y
                )

                state.weights = weights
                state.iteration += 1

                # Track progress
                learning_curve.append(accuracy)

                # Log progress periodically
                if state.iteration % 50 == 0:
                    logger.info(f"Iteration {state.iteration}: accuracy={accuracy:.3f}")

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

                # Yield to other threads EVERY iteration to keep GUI responsive
                # This is critical for PyQt responsiveness
                time.sleep(0.01)  # 10ms yield every iteration

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

    def _initialize_weights(self, challenge: Challenge):
        """Initialize weights for learning (on GPU if available)."""
        if challenge.training_data and challenge.training_data.feature_dim:
            input_dim = challenge.training_data.feature_dim
        else:
            input_dim = self.state_dim

        if challenge.training_data and challenge.training_data.num_classes:
            output_dim = challenge.training_data.num_classes
        else:
            output_dim = self.state_dim

        # Xavier-like initialization scaled for Hebbian (on GPU)
        scale = xp.sqrt(2.0 / (input_dim + output_dim))
        weights = xp.random.randn(output_dim, input_dim) * scale

        return weights

    def _get_training_batch(
        self,
        challenge: Challenge,
    ) -> Tuple[Any, Optional[Any]]:
        """Get a batch of training data (on GPU if available)."""
        if challenge.training_data:
            samples, labels = challenge.training_data.get_batch(self.batch_size)

            # Convert to arrays and move to GPU
            batch_x = to_gpu(np.array(samples))

            # Check for None explicitly to avoid numpy array boolean ambiguity
            if labels is None:
                batch_y = None
            else:
                batch_y = to_gpu(np.array(labels))

            return batch_x, batch_y
        else:
            # Generate synthetic data for demonstration (on GPU)
            batch_x = xp.random.randn(self.batch_size, self.state_dim)
            batch_y = xp.random.randint(0, 10, self.batch_size)
            return batch_x, batch_y

    def _apply_plasticity(
        self,
        state: LearningState,
        batch_x,
        batch_y,
    ) -> Tuple[float, Any]:
        """
        Apply local plasticity rule based on selected strategy.

        All rules use LOCAL updates only - no backpropagation.
        Computations run on GPU if available.

        For classification tasks with labels, adds a supervised signal
        to guide local learning toward correct predictions.

        Args:
            state: Current learning state
            batch_x: Input batch (on GPU)
            batch_y: Label batch (optional, on GPU)

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
            # Resize weights (on GPU)
            new_weights = xp.random.randn(weights.shape[0], batch_x.shape[1]) * 0.01
            weights = new_weights

        # Compute activations
        activations = batch_x @ weights.T  # (batch, output_dim)

        # SUPERVISED COMPONENT: If we have labels, add supervised learning signal
        # This is critical for classification tasks - pure unsupervised Hebbian
        # won't learn to classify without guidance toward correct outputs
        if batch_y is not None and batch_y.ndim == 1:
            # Create target activations from labels (one-hot encoding)
            num_classes = weights.shape[0]
            target_activations = xp.zeros((batch_x.shape[0], num_classes), dtype=xp.float32)

            # Only set targets for valid labels
            valid_mask = (batch_y >= 0) & (batch_y < num_classes)
            for i in range(batch_x.shape[0]):
                if valid_mask[i]:
                    label = int(batch_y[i])
                    target_activations[i, label] = 1.0

            # Error-driven learning: strengthen correct, weaken incorrect
            # This is still local (per-synapse) but guided by supervision
            error = target_activations - xp.tanh(activations)  # Bounded activations

            # Supervised weight update (Delta rule - local Hebbian with error)
            # Use higher weight for supervised signal for faster convergence
            supervised_lr = lr * 1.0  # Equal weight to unsupervised
            dW_supervised = supervised_lr * (error.T @ batch_x) / batch_x.shape[0]

            # Clip to prevent explosion
            dW_supervised = xp.clip(dW_supervised, -0.5, 0.5)
            weights += dW_supervised

            # For classification tasks with labels, the supervised signal is dominant
            # Only apply compatible unsupervised rules (Hebbian-family) at reduced rate
            # Skip incompatible rules (COMPETITIVE, COOPERATIVE) that fight supervised learning
            has_supervision = True
            unsupervised_lr = lr * 0.1  # Much smaller than supervised
        else:
            has_supervision = False
            unsupervised_lr = lr  # Full LR for unsupervised tasks

        # Apply learning rule based on strategy
        # Skip COMPETITIVE/COOPERATIVE when supervised - they fight against labels
        if strategy == LearningStrategy.HEBBIAN:
            weights = self._hebbian_update(weights, batch_x, activations, unsupervised_lr)

        elif strategy == LearningStrategy.OJA:
            weights = self._oja_update(weights, batch_x, activations, unsupervised_lr)

        elif strategy == LearningStrategy.STDP:
            weights = self._stdp_update(weights, batch_x, activations, unsupervised_lr)

        elif strategy == LearningStrategy.BCM:
            theta = state.hyperparameters.get('bcm_threshold', 0.5)
            weights = self._bcm_update(weights, batch_x, activations, unsupervised_lr, theta)

        elif strategy == LearningStrategy.ANTI_HEBBIAN:
            weights = self._anti_hebbian_update(weights, batch_x, activations, unsupervised_lr)

        elif strategy == LearningStrategy.COMPETITIVE:
            # Only apply if unsupervised - competitive fights supervised signal
            if not has_supervision:
                weights = self._competitive_update(weights, batch_x, activations, unsupervised_lr)

        elif strategy == LearningStrategy.COOPERATIVE:
            # Only apply if unsupervised - cooperative fights supervised signal
            if not has_supervision:
                weights = self._cooperative_update(weights, batch_x, activations, unsupervised_lr)

        # Calculate accuracy FIRST (needed for network learning reward)
        accuracy = self._calculate_accuracy(weights, batch_x, batch_y)

        # Also learn through the self-organizing network for structural plasticity
        # Only run every 10 iterations to reduce overhead and keep UI responsive
        if state.iteration % 10 == 0:
            try:
                self._learn_through_network(state, batch_x, batch_y, accuracy)
            except Exception as e:
                # Network learning is supplementary - don't fail the main learning
                logger.debug(f"Network learning skipped: {e}")

        return accuracy, weights

    def _hebbian_update(
        self,
        weights,
        x,
        y,
        lr: float,
    ):
        """
        Hebbian learning: "Neurons that fire together wire together"

        ΔW = lr * y.T @ x
        """
        dW = lr * (y.T @ x) / x.shape[0]
        weights += dW

        # Normalize to prevent explosion
        norms = xp.linalg.norm(weights, axis=1, keepdims=True)
        weights = weights / (norms + 1e-8)

        return weights

    def _oja_update(
        self,
        weights,
        x,
        y,
        lr: float,
    ):
        """
        Oja's rule: Normalized Hebbian with decay

        ΔW = lr * y * (x - y * W)
        """
        # Clip activations to prevent overflow
        y = xp.clip(y, -10, 10)

        for i in range(x.shape[0]):
            xi = x[i:i+1]  # (1, input_dim)
            yi = y[i:i+1]  # (1, output_dim)

            # Oja update with numerical stability
            reconstruction = yi @ weights
            dW = lr * yi.T @ (xi - reconstruction)

            # Clip updates to prevent explosion
            dW = xp.clip(dW, -1, 1)
            weights += dW

        # Normalize weights to prevent explosion
        norms = xp.linalg.norm(weights, axis=1, keepdims=True)
        weights = weights / (norms + 1e-8)

        return weights

    def _stdp_update(
        self,
        weights,
        x,
        y,
        lr: float,
    ):
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
        weights = xp.clip(weights, -2, 2)

        return weights

    def _bcm_update(
        self,
        weights,
        x,
        y,
        lr: float,
        theta: float,
    ):
        """
        BCM (Bienenstock-Cooper-Munro) rule with sliding threshold

        ΔW = lr * y * (y - θ) * x
        """
        y_mean = xp.mean(y, axis=0, keepdims=True)

        for i in range(x.shape[0]):
            xi = x[i:i+1]
            yi = y[i:i+1]

            # BCM: modulate by (y - threshold)
            modulation = yi * (yi - theta)
            dW = lr * modulation.T @ xi
            weights += dW

        # Update sliding threshold toward mean activity
        theta = 0.9 * theta + 0.1 * float(xp.mean(y_mean ** 2))

        # Normalize
        norms = xp.linalg.norm(weights, axis=1, keepdims=True)
        weights = weights / (norms + 1e-8)

        return weights

    def _anti_hebbian_update(
        self,
        weights,
        x,
        y,
        lr: float,
    ):
        """
        Anti-Hebbian: Decorrelation learning

        ΔW = -lr * y.T @ x (opposite of Hebbian)
        """
        dW = -lr * (y.T @ x) / x.shape[0]
        weights += dW

        # Keep weights bounded
        weights = xp.clip(weights, -2, 2)

        return weights

    def _competitive_update(
        self,
        weights,
        x,
        y,
        lr: float,
    ):
        """
        Competitive learning: Winner-take-all

        Only the most active unit learns.
        """
        for i in range(x.shape[0]):
            xi = x[i:i+1]
            yi = y[i]

            # Find winner (most active unit)
            winner = int(xp.argmax(yi))

            # Only winner learns
            dW = lr * (xi - weights[winner:winner+1])
            weights[winner:winner+1] += dW

        return weights

    def _cooperative_update(
        self,
        weights,
        x,
        y,
        lr: float,
    ):
        """
        Cooperative learning: Ensemble coordination

        Units near the winner also learn (neighborhood function).
        """
        for i in range(x.shape[0]):
            xi = x[i:i+1]
            yi = y[i]

            # Find winner
            winner = int(xp.argmax(yi))

            # Create neighborhood function (Gaussian)
            indices = xp.arange(weights.shape[0])
            neighborhood = xp.exp(-0.5 * ((indices - winner) / 2) ** 2)
            neighborhood = neighborhood.reshape(-1, 1)

            # All units learn proportionally to neighborhood
            dW = lr * neighborhood * (xi - weights)
            weights += dW

        return weights

    def _calculate_accuracy(
        self,
        weights,
        x,
        y,
    ) -> float:
        """Calculate accuracy of current weights."""
        if y is None:
            # Unsupervised: use reconstruction error as proxy
            activations = x @ weights.T
            reconstruction = activations @ weights
            error = float(xp.mean((x - reconstruction) ** 2))
            return max(0, 1 - error)

        # Supervised: classification accuracy
        activations = x @ weights.T
        predictions = xp.argmax(activations, axis=1)

        if y.ndim > 1:
            targets = xp.argmax(y, axis=1)
        else:
            targets = y

        # Handle case where targets exceed weight dimensions
        valid_mask = targets < weights.shape[0]
        if not xp.any(valid_mask):
            return 0.0

        accuracy = xp.mean(predictions[valid_mask] == targets[valid_mask])
        return float(accuracy)

    def _learn_through_network(
        self,
        state: LearningState,
        batch_x,
        batch_y,
        accuracy: float
    ) -> None:
        """
        Apply learning through the self-organizing network.

        Projects inputs to the expected dimension and applies structural plasticity.
        """
        learning_rule_map = {
            LearningStrategy.HEBBIAN: 'hebbian',
            LearningStrategy.OJA: 'oja',
            LearningStrategy.STDP: 'stdp',
            LearningStrategy.BCM: 'oja',
            LearningStrategy.ANTI_HEBBIAN: 'hebbian',
            LearningStrategy.COMPETITIVE: 'oja',
            LearningStrategy.COOPERATIVE: 'oja',
        }

        # Project input to network's expected dimension if needed
        input_dim = batch_x.shape[1] if batch_x.ndim > 1 else batch_x.shape[0]
        network_input_dim = self.network.input_dim

        if input_dim != network_input_dim:
            # Create/update projection matrix (random projection)
            if not hasattr(self, '_input_projection') or self._input_projection.shape != (input_dim, network_input_dim):
                self._input_projection = xp.random.randn(input_dim, network_input_dim).astype(xp.float32)
                self._input_projection /= xp.sqrt(input_dim)  # Normalize

            # Project input
            if batch_x.ndim > 1:
                projected_x = batch_x @ self._input_projection
            else:
                projected_x = batch_x @ self._input_projection
        else:
            projected_x = batch_x

        # Prepare target for network learning
        network_target = None
        if batch_y is not None and batch_y.ndim == 1:
            max_label = int(xp.max(batch_y))
            if max_label < self.network.output_dim:
                network_target = xp.zeros((batch_y.shape[0], self.network.output_dim), dtype=xp.float32)
                for i in range(batch_y.shape[0]):
                    label_idx = int(batch_y[i])
                    if label_idx < self.network.output_dim:
                        network_target[i, label_idx] = 1.0

        # Learn through the network (triggers structural plasticity)
        num_samples = min(projected_x.shape[0] if projected_x.ndim > 1 else 1, 3)

        for i in range(num_samples):
            if projected_x.ndim > 1:
                sample = projected_x[i]
            else:
                sample = projected_x

            target = None
            if network_target is not None and i < network_target.shape[0]:
                target = network_target[i]

            # Use accuracy as reward signal
            reward = accuracy if accuracy > 0 else 0.0

            network_result = self.network.learn(
                sample,
                target=target,
                reward=reward,
                learning_rule=learning_rule_map.get(state.strategy, 'oja')
            )

            # Track structural changes
            if network_result.get('neurons_added') or network_result.get('neurons_pruned'):
                state.structural_changes.append(network_result)

                # Notify callbacks
                for callback in self._structural_change_callbacks:
                    try:
                        callback(network_result)
                    except Exception as e:
                        logger.debug(f"Structural change callback error: {e}")

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

    def get_network_structure(self) -> Dict[str, Any]:
        """
        Get the current structure of the self-organizing network.

        This returns the complete network topology including:
        - Layer sizes and neuron states
        - Connection weights and statistics
        - Recent structural changes (neurons added/pruned)

        Returns:
            Dictionary with complete network structure for visualization
        """
        return self.network.get_structure()

    def get_recent_structural_changes(self, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Get recent structural changes for animation.

        Args:
            limit: Maximum number of events to return

        Returns:
            List of recent structural change events
        """
        return self.network.get_recent_structural_changes(limit)

    def register_structural_change_callback(
        self,
        callback: Callable[[Dict[str, Any]], None]
    ) -> None:
        """
        Register a callback to be notified of structural changes.

        Args:
            callback: Function to call when structure changes
        """
        self._structural_change_callbacks.append(callback)

    def get_network_stats(self) -> Dict[str, Any]:
        """Get network statistics."""
        return self.network.get_stats()

    def get_layer_activations(self) -> List[np.ndarray]:
        """Get activations from the last forward pass through the network."""
        return self.network.get_layer_activations()

    def get_internal_state(self) -> np.ndarray:
        """
        Get the current internal state representation.

        This is useful for the creative canvas controller.

        Returns:
            numpy array representing current internal state
        """
        # Combine layer activations into a single state vector
        activations = self.network.get_layer_activations()
        if not activations:
            return np.zeros(self.state_dim)

        # Flatten and pad/truncate to state_dim
        combined = np.concatenate([a.flatten() for a in activations])

        if len(combined) >= self.state_dim:
            return combined[:self.state_dim].astype(np.float32)
        else:
            result = np.zeros(self.state_dim, dtype=np.float32)
            result[:len(combined)] = combined
            return result
