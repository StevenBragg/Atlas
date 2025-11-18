"""
Meta-Learning System for ATLAS

Implements learning-to-learn capabilities allowing the system to:
- Adapt learning strategies based on experience
- Discover effective learning algorithms
- Optimize hyperparameters dynamically
- Transfer knowledge across tasks
- Improve learning efficiency over time

This is critical for superintelligence as it enables continuous improvement
of the learning process itself.
"""

import numpy as np
import logging
from typing import List, Dict, Tuple, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum
import time

logger = logging.getLogger(__name__)


class LearningStrategy(Enum):
    """Available learning strategies"""
    HEBBIAN = "hebbian"
    OJA = "oja"
    STDP = "stdp"
    BCM = "bcm"
    ANTI_HEBBIAN = "anti_hebbian"
    COMPETITIVE = "competitive"
    COOPERATIVE = "cooperative"


@dataclass
class LearningExperience:
    """Records a learning experience for meta-learning"""
    strategy: LearningStrategy
    hyperparameters: Dict[str, float]
    task_characteristics: Dict[str, float]
    performance_metrics: Dict[str, float]
    timestamp: float
    success_score: float  # Overall success measure


class MetaLearner:
    """
    Meta-learning system that learns to optimize the learning process.

    Capabilities:
    - Strategy selection based on task characteristics
    - Hyperparameter optimization
    - Learning curriculum generation
    - Transfer learning
    - Algorithm discovery
    """

    def __init__(
        self,
        num_strategies: int = 7,
        num_hyperparameters: int = 10,
        exploration_rate: float = 0.2,
        learning_rate: float = 0.01,
        memory_size: int = 1000,
        enable_algorithm_discovery: bool = True,
        enable_curriculum: bool = True,
        random_seed: Optional[int] = None,
    ):
        """
        Initialize meta-learner.

        Args:
            num_strategies: Number of learning strategies to consider
            num_hyperparameters: Dimension of hyperparameter space
            exploration_rate: Rate of exploring new strategies
            learning_rate: Rate of updating meta-knowledge
            memory_size: Size of experience replay buffer
            enable_algorithm_discovery: Whether to discover new algorithms
            enable_curriculum: Whether to generate learning curricula
            random_seed: Random seed for reproducibility
        """
        if random_seed is not None:
            np.random.seed(random_seed)

        self.num_strategies = num_strategies
        self.num_hyperparameters = num_hyperparameters
        self.exploration_rate = exploration_rate
        self.learning_rate = learning_rate
        self.memory_size = memory_size
        self.enable_algorithm_discovery = enable_algorithm_discovery
        self.enable_curriculum = enable_curriculum

        # Strategy performance tracking
        self.strategy_performance: Dict[LearningStrategy, List[float]] = {
            strategy: [] for strategy in LearningStrategy
        }

        # Q-values for strategy selection (state -> strategy -> value)
        self.strategy_values: Dict[str, Dict[LearningStrategy, float]] = {}

        # Hyperparameter optimization
        self.optimal_hyperparameters: Dict[LearningStrategy, Dict[str, float]] = {
            strategy: self._initialize_hyperparameters(strategy)
            for strategy in LearningStrategy
        }

        # Experience replay buffer
        self.experience_buffer: List[LearningExperience] = []

        # Curriculum learning
        self.difficulty_progression = 0.0  # Current difficulty level
        self.mastery_threshold = 0.8  # Performance needed to advance

        # Algorithm discovery
        self.discovered_algorithms: List[Dict[str, Any]] = []

        # Statistics
        self.total_selections = 0
        self.total_updates = 0
        self.strategy_usage = {strategy: 0 for strategy in LearningStrategy}

        logger.info("Initialized meta-learner")

    def select_strategy(
        self,
        task_characteristics: Dict[str, float],
        explore: Optional[bool] = None,
    ) -> Tuple[LearningStrategy, Dict[str, float]]:
        """
        Select the best learning strategy for the current task.

        Args:
            task_characteristics: Features describing the current task
            explore: Whether to explore (None for epsilon-greedy)

        Returns:
            (selected_strategy, hyperparameters)
        """
        # Encode task state
        state_key = self._encode_state(task_characteristics)

        # Initialize Q-values for this state if new
        if state_key not in self.strategy_values:
            self.strategy_values[state_key] = {
                strategy: 0.0 for strategy in LearningStrategy
            }

        # Epsilon-greedy exploration
        if explore is None:
            explore = np.random.random() < self.exploration_rate

        if explore:
            # Random strategy
            strategy = np.random.choice(list(LearningStrategy))
            logger.debug(f"Exploring with strategy: {strategy.value}")
        else:
            # Greedy selection
            q_values = self.strategy_values[state_key]
            strategy = max(q_values, key=q_values.get)
            logger.debug(f"Exploiting with strategy: {strategy.value}")

        # Get optimal hyperparameters for this strategy
        hyperparameters = self.optimal_hyperparameters[strategy].copy()

        # Update statistics
        self.total_selections += 1
        self.strategy_usage[strategy] += 1

        return strategy, hyperparameters

    def update(
        self,
        strategy: LearningStrategy,
        hyperparameters: Dict[str, float],
        task_characteristics: Dict[str, float],
        performance_metrics: Dict[str, float],
    ) -> None:
        """
        Update meta-knowledge based on learning experience.

        Args:
            strategy: Strategy that was used
            hyperparameters: Hyperparameters that were used
            task_characteristics: Task features
            performance_metrics: Resulting performance
        """
        # Calculate success score from performance metrics
        success_score = self._calculate_success_score(performance_metrics)

        # Create experience
        experience = LearningExperience(
            strategy=strategy,
            hyperparameters=hyperparameters,
            task_characteristics=task_characteristics,
            performance_metrics=performance_metrics,
            timestamp=time.time(),
            success_score=success_score,
        )

        # Store experience
        self.experience_buffer.append(experience)
        if len(self.experience_buffer) > self.memory_size:
            self.experience_buffer.pop(0)

        # Update strategy performance
        self.strategy_performance[strategy].append(success_score)
        if len(self.strategy_performance[strategy]) > 100:
            self.strategy_performance[strategy].pop(0)

        # Update Q-values
        state_key = self._encode_state(task_characteristics)
        if state_key in self.strategy_values:
            current_q = self.strategy_values[state_key][strategy]
            # TD update
            new_q = current_q + self.learning_rate * (success_score - current_q)
            self.strategy_values[state_key][strategy] = new_q

        # Update hyperparameters using gradient-free optimization
        self._optimize_hyperparameters(strategy, success_score, hyperparameters)

        # Update curriculum if enabled
        if self.enable_curriculum:
            self._update_curriculum(success_score)

        self.total_updates += 1

        logger.debug(f"Updated meta-learner: strategy={strategy.value}, score={success_score:.3f}")

    def _initialize_hyperparameters(self, strategy: LearningStrategy) -> Dict[str, float]:
        """Initialize default hyperparameters for a strategy."""
        defaults = {
            'learning_rate': 0.01,
            'momentum': 0.5,
            'decay': 0.001,
            'threshold': 0.5,
            'sparsity': 0.1,
            'k_winners': 0.1,
            'lateral_inhibition': 0.2,
            'temporal_window': 5,
            'normalization': 1.0,
            'regularization': 0.001,
        }
        return defaults

    def _optimize_hyperparameters(
        self,
        strategy: LearningStrategy,
        score: float,
        current_params: Dict[str, float],
    ) -> None:
        """
        Optimize hyperparameters using evolutionary strategy.

        Uses a simple evolutionary approach: if new parameters worked better,
        move the optimal parameters in that direction.
        """
        optimal = self.optimal_hyperparameters[strategy]

        # Get recent average performance
        recent_scores = self.strategy_performance[strategy][-10:]
        if len(recent_scores) < 2:
            return

        baseline = np.mean(recent_scores[:-1])

        # If current score is better, update optimal parameters
        if score > baseline:
            for param, value in current_params.items():
                if param in optimal:
                    # Move optimal toward current
                    optimal[param] = (
                        (1 - self.learning_rate) * optimal[param] +
                        self.learning_rate * value
                    )

    def _calculate_success_score(self, metrics: Dict[str, float]) -> float:
        """Calculate an overall success score from performance metrics."""
        # Weight different metrics
        score = 0.0

        if 'accuracy' in metrics:
            score += metrics['accuracy'] * 0.4
        if 'prediction_error' in metrics:
            score += (1.0 - min(1.0, metrics['prediction_error'])) * 0.3
        if 'reconstruction_error' in metrics:
            score += (1.0 - min(1.0, metrics['reconstruction_error'])) * 0.2
        if 'sparsity' in metrics:
            # Prefer moderate sparsity
            target_sparsity = 0.1
            sparsity_score = 1.0 - abs(metrics['sparsity'] - target_sparsity)
            score += sparsity_score * 0.1

        # Normalize to 0-1
        return max(0.0, min(1.0, score))

    def _encode_state(self, characteristics: Dict[str, float]) -> str:
        """
        Encode task characteristics into a discrete state.

        Args:
            characteristics: Continuous task features

        Returns:
            String key representing the state
        """
        # Discretize characteristics into bins
        state_components = []
        for key in sorted(characteristics.keys()):
            value = characteristics[key]
            # Bin into 5 categories
            bin_idx = int(np.clip(value * 5, 0, 4))
            state_components.append(f"{key}:{bin_idx}")

        return "_".join(state_components)

    def _update_curriculum(self, score: float) -> None:
        """Update the curriculum difficulty based on recent performance."""
        # If consistently performing well, increase difficulty
        recent_scores = []
        for strategy_scores in self.strategy_performance.values():
            recent_scores.extend(strategy_scores[-5:])

        if len(recent_scores) >= 5:
            avg_score = np.mean(recent_scores)

            if avg_score > self.mastery_threshold:
                # Increase difficulty
                self.difficulty_progression = min(1.0, self.difficulty_progression + 0.1)
                logger.info(f"Curriculum difficulty increased to {self.difficulty_progression:.2f}")
            elif avg_score < 0.5:
                # Decrease difficulty
                self.difficulty_progression = max(0.0, self.difficulty_progression - 0.05)
                logger.info(f"Curriculum difficulty decreased to {self.difficulty_progression:.2f}")

    def discover_algorithm(self) -> Optional[Dict[str, Any]]:
        """
        Attempt to discover a new learning algorithm by combining
        successful strategies.

        Returns:
            New algorithm specification, or None
        """
        if not self.enable_algorithm_discovery:
            return None

        # Need sufficient experience
        if len(self.experience_buffer) < 50:
            return None

        # Find top-performing experiences
        sorted_experiences = sorted(
            self.experience_buffer,
            key=lambda x: x.success_score,
            reverse=True
        )[:10]

        # Extract common patterns
        common_strategies = {}
        for exp in sorted_experiences:
            strategy = exp.strategy
            common_strategies[strategy] = common_strategies.get(strategy, 0) + 1

        # Check if a combination of strategies might work
        if len(common_strategies) >= 2:
            # Create hybrid algorithm
            top_strategies = sorted(
                common_strategies.items(),
                key=lambda x: x[1],
                reverse=True
            )[:2]

            # Combine hyperparameters
            combined_params = {}
            for exp in sorted_experiences:
                if exp.strategy in [s for s, _ in top_strategies]:
                    for param, value in exp.hyperparameters.items():
                        if param not in combined_params:
                            combined_params[param] = []
                        combined_params[param].append(value)

            # Average parameters
            avg_params = {
                param: np.mean(values)
                for param, values in combined_params.items()
            }

            # Create new algorithm
            algorithm = {
                'strategies': [s.value for s, _ in top_strategies],
                'hyperparameters': avg_params,
                'discovery_time': time.time(),
                'expected_performance': np.mean([exp.success_score for exp in sorted_experiences]),
            }

            self.discovered_algorithms.append(algorithm)
            logger.info(f"Discovered new algorithm combining {[s.value for s, _ in top_strategies]}")

            return algorithm

        return None

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about meta-learning."""
        # Calculate average performance per strategy
        avg_performance = {}
        for strategy, scores in self.strategy_performance.items():
            if scores:
                avg_performance[strategy.value] = float(np.mean(scores))
            else:
                avg_performance[strategy.value] = 0.0

        # Calculate usage distribution
        total_usage = sum(self.strategy_usage.values())
        usage_distribution = {
            strategy.value: count / max(1, total_usage)
            for strategy, count in self.strategy_usage.items()
        }

        return {
            'total_selections': self.total_selections,
            'total_updates': self.total_updates,
            'exploration_rate': self.exploration_rate,
            'avg_performance_by_strategy': avg_performance,
            'usage_distribution': usage_distribution,
            'experience_buffer_size': len(self.experience_buffer),
            'num_states_explored': len(self.strategy_values),
            'curriculum_difficulty': float(self.difficulty_progression),
            'discovered_algorithms': len(self.discovered_algorithms),
        }

    def serialize(self) -> Dict[str, Any]:
        """Serialize the meta-learner state."""
        return {
            'strategy_performance': {
                s.value: scores for s, scores in self.strategy_performance.items()
            },
            'optimal_hyperparameters': {
                s.value: params for s, params in self.optimal_hyperparameters.items()
            },
            'difficulty_progression': self.difficulty_progression,
            'discovered_algorithms': self.discovered_algorithms,
            'stats': self.get_stats(),
        }

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> 'MetaLearner':
        """Create a meta-learner from serialized data."""
        instance = cls()

        # Restore performance history
        for strategy_name, scores in data.get('strategy_performance', {}).items():
            strategy = LearningStrategy(strategy_name)
            instance.strategy_performance[strategy] = scores

        # Restore hyperparameters
        for strategy_name, params in data.get('optimal_hyperparameters', {}).items():
            strategy = LearningStrategy(strategy_name)
            instance.optimal_hyperparameters[strategy] = params

        # Restore other state
        instance.difficulty_progression = data.get('difficulty_progression', 0.0)
        instance.discovered_algorithms = data.get('discovered_algorithms', [])

        return instance
