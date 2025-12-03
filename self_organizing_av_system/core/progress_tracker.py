"""
Progress Tracking for Challenge-Based Learning

This module tracks learning progress, manages curriculum difficulty,
and provides metrics for challenge completion.

All learning uses biology-inspired local plasticity rules - this module
only tracks and monitors the learning process.
"""

import time
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from collections import deque
import logging

from .challenge import (
    Challenge,
    ChallengeStatus,
    ProgressReport,
    SuccessCriteria,
)

logger = logging.getLogger(__name__)


@dataclass
class ChallengeMetrics:
    """Metrics for a single challenge."""
    challenge_id: str
    accuracy_history: List[float] = field(default_factory=list)
    loss_history: List[float] = field(default_factory=list)
    iteration_times: List[float] = field(default_factory=list)
    strategy_history: List[str] = field(default_factory=list)
    custom_metrics: Dict[str, List[float]] = field(default_factory=dict)
    start_time: float = field(default_factory=time.time)
    last_update: float = field(default_factory=time.time)

    @property
    def current_accuracy(self) -> float:
        """Get the most recent accuracy."""
        return self.accuracy_history[-1] if self.accuracy_history else 0.0

    @property
    def best_accuracy(self) -> float:
        """Get the best accuracy achieved."""
        return max(self.accuracy_history) if self.accuracy_history else 0.0

    @property
    def iterations(self) -> int:
        """Get the number of iterations."""
        return len(self.accuracy_history)

    @property
    def elapsed_time(self) -> float:
        """Get elapsed time in seconds."""
        return time.time() - self.start_time

    def get_learning_rate(self, window: int = 10) -> float:
        """Calculate the rate of improvement over recent iterations."""
        if len(self.accuracy_history) < 2:
            return 0.0
        recent = self.accuracy_history[-window:]
        if len(recent) < 2:
            return 0.0
        return (recent[-1] - recent[0]) / len(recent)

    def is_plateaued(self, window: int = 10, threshold: float = 0.01) -> bool:
        """Check if learning has plateaued."""
        if len(self.accuracy_history) < window:
            return False
        recent = self.accuracy_history[-window:]
        return (max(recent) - min(recent)) < threshold


class ProgressTracker:
    """
    Tracks progress across multiple learning challenges.

    Responsibilities:
    - Track metrics for each challenge
    - Manage curriculum difficulty progression
    - Detect plateaus and suggest strategy changes
    - Estimate completion time
    """

    def __init__(
        self,
        mastery_threshold: float = 0.8,
        plateau_window: int = 20,
        plateau_threshold: float = 0.01,
        curriculum_adjustment_rate: float = 0.1,
    ):
        """
        Initialize progress tracker.

        Args:
            mastery_threshold: Accuracy threshold for mastery
            plateau_window: Window size for plateau detection
            plateau_threshold: Minimum improvement to avoid plateau
            curriculum_adjustment_rate: Rate of difficulty adjustment
        """
        self.mastery_threshold = mastery_threshold
        self.plateau_window = plateau_window
        self.plateau_threshold = plateau_threshold
        self.curriculum_adjustment_rate = curriculum_adjustment_rate

        # Track metrics per challenge
        self.challenges: Dict[str, ChallengeMetrics] = {}

        # Global curriculum state
        self.curriculum_difficulty: float = 0.3  # Start easy
        self.completed_challenges: List[str] = []
        self.failed_challenges: List[str] = []

        # Historical performance for curriculum adjustment
        self.performance_history: deque = deque(maxlen=100)

        logger.info("ProgressTracker initialized")

    def start_challenge(self, challenge: Challenge) -> str:
        """
        Start tracking a new challenge.

        Args:
            challenge: The challenge to track

        Returns:
            Challenge ID
        """
        metrics = ChallengeMetrics(challenge_id=challenge.id)
        self.challenges[challenge.id] = metrics
        logger.info(f"Started tracking challenge: {challenge.name} ({challenge.id})")
        return challenge.id

    def update_progress(
        self,
        challenge_id: str,
        accuracy: float,
        strategy: str,
        custom_metrics: Optional[Dict[str, float]] = None,
        loss: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Update progress for a challenge.

        Args:
            challenge_id: ID of the challenge
            accuracy: Current accuracy
            strategy: Current learning strategy
            custom_metrics: Optional custom metrics
            loss: Optional loss value

        Returns:
            Status information including plateau detection
        """
        if challenge_id not in self.challenges:
            raise ValueError(f"Unknown challenge ID: {challenge_id}")

        metrics = self.challenges[challenge_id]
        metrics.accuracy_history.append(accuracy)
        metrics.strategy_history.append(strategy)
        metrics.iteration_times.append(time.time() - metrics.last_update)
        metrics.last_update = time.time()

        if loss is not None:
            metrics.loss_history.append(loss)

        if custom_metrics:
            for name, value in custom_metrics.items():
                if name not in metrics.custom_metrics:
                    metrics.custom_metrics[name] = []
                metrics.custom_metrics[name].append(value)

        # Check for plateau
        is_plateau = metrics.is_plateaued(
            window=self.plateau_window,
            threshold=self.plateau_threshold
        )

        # Calculate learning rate
        learning_rate = metrics.get_learning_rate(window=10)

        return {
            'iterations': metrics.iterations,
            'accuracy': accuracy,
            'best_accuracy': metrics.best_accuracy,
            'is_plateau': is_plateau,
            'learning_rate': learning_rate,
            'elapsed_time': metrics.elapsed_time,
            'should_adapt': is_plateau and accuracy < self.mastery_threshold,
        }

    def check_completion(
        self,
        challenge_id: str,
        criteria: SuccessCriteria,
    ) -> Tuple[bool, str]:
        """
        Check if a challenge meets completion criteria.

        Args:
            challenge_id: ID of the challenge
            criteria: Success criteria to check against

        Returns:
            Tuple of (is_complete, reason)
        """
        if challenge_id not in self.challenges:
            return False, "Unknown challenge"

        metrics = self.challenges[challenge_id]

        # Check minimum iterations
        if metrics.iterations < criteria.min_samples:
            return False, f"Need {criteria.min_samples - metrics.iterations} more iterations"

        # Check maximum iterations
        if metrics.iterations >= criteria.max_iterations:
            if metrics.current_accuracy >= criteria.accuracy:
                return True, "Reached max iterations with sufficient accuracy"
            return True, "Reached max iterations (timeout)"

        # Check time limit
        if criteria.time_limit_seconds and metrics.elapsed_time >= criteria.time_limit_seconds:
            return True, "Time limit reached"

        # Check accuracy threshold
        if metrics.current_accuracy >= criteria.accuracy:
            # Check convergence
            if metrics.iterations >= criteria.convergence_window:
                recent_improvement = metrics.get_learning_rate(criteria.convergence_window)
                if abs(recent_improvement) < criteria.convergence_threshold:
                    return True, "Converged at target accuracy"

        # Check custom metrics
        for metric_name, threshold in criteria.custom_metrics.items():
            if metric_name in metrics.custom_metrics:
                current_value = metrics.custom_metrics[metric_name][-1]
                if current_value < threshold:
                    return False, f"{metric_name} ({current_value:.3f}) below threshold ({threshold})"

        return False, "Still learning"

    def get_progress_report(
        self,
        challenge_id: str,
        challenge: Challenge,
    ) -> ProgressReport:
        """
        Generate a progress report for a challenge.

        Args:
            challenge_id: ID of the challenge
            challenge: The challenge object

        Returns:
            ProgressReport with current status
        """
        if challenge_id not in self.challenges:
            raise ValueError(f"Unknown challenge ID: {challenge_id}")

        metrics = self.challenges[challenge_id]

        # Calculate progress percentage
        progress = self._calculate_progress(metrics, challenge.success_criteria)

        # Estimate completion time
        estimated_remaining = self._estimate_completion_time(
            metrics, challenge.success_criteria
        )

        # Count strategy switches
        strategy_switches = self._count_strategy_switches(metrics.strategy_history)

        return ProgressReport(
            challenge_id=challenge_id,
            challenge_name=challenge.name,
            status=challenge.status,
            progress_percent=progress,
            current_accuracy=metrics.current_accuracy,
            iterations_completed=metrics.iterations,
            time_elapsed_seconds=metrics.elapsed_time,
            current_strategy=metrics.strategy_history[-1] if metrics.strategy_history else "none",
            learning_curve=metrics.accuracy_history.copy(),
            strategy_switches=strategy_switches,
            estimated_completion=estimated_remaining,
        )

    def get_learning_curve(self, challenge_id: str) -> List[float]:
        """Get the learning curve for a challenge."""
        if challenge_id not in self.challenges:
            return []
        return self.challenges[challenge_id].accuracy_history.copy()

    def complete_challenge(
        self,
        challenge_id: str,
        success: bool,
        final_accuracy: float,
    ) -> None:
        """
        Mark a challenge as completed and update curriculum.

        Args:
            challenge_id: ID of the challenge
            success: Whether the challenge was successful
            final_accuracy: Final accuracy achieved
        """
        if success:
            self.completed_challenges.append(challenge_id)
        else:
            self.failed_challenges.append(challenge_id)

        # Update performance history
        self.performance_history.append({
            'challenge_id': challenge_id,
            'success': success,
            'accuracy': final_accuracy,
            'difficulty': self.curriculum_difficulty,
        })

        # Adjust curriculum difficulty
        self._adjust_curriculum(success, final_accuracy)

        logger.info(
            f"Challenge {challenge_id} {'completed' if success else 'failed'} "
            f"with accuracy {final_accuracy:.2%}. "
            f"Curriculum difficulty: {self.curriculum_difficulty:.2f}"
        )

    def adjust_curriculum(self, challenge_id: str) -> float:
        """
        Get recommended difficulty for next challenge.

        Args:
            challenge_id: ID of the current challenge

        Returns:
            Recommended difficulty level (0.0 - 1.0)
        """
        return self.curriculum_difficulty

    def get_stats(self) -> Dict[str, Any]:
        """Get overall statistics."""
        total = len(self.completed_challenges) + len(self.failed_challenges)
        success_rate = (
            len(self.completed_challenges) / total if total > 0 else 0.0
        )

        return {
            'total_challenges': total,
            'completed': len(self.completed_challenges),
            'failed': len(self.failed_challenges),
            'success_rate': success_rate,
            'current_difficulty': self.curriculum_difficulty,
            'active_challenges': len(self.challenges) - total,
        }

    def _calculate_progress(
        self,
        metrics: ChallengeMetrics,
        criteria: SuccessCriteria,
    ) -> float:
        """Calculate progress percentage."""
        # Weight different factors
        iteration_progress = min(metrics.iterations / criteria.min_samples, 1.0) * 0.3
        accuracy_progress = min(metrics.current_accuracy / criteria.accuracy, 1.0) * 0.5
        convergence_progress = 0.0

        if metrics.iterations >= criteria.convergence_window:
            improvement = abs(metrics.get_learning_rate(criteria.convergence_window))
            if improvement < criteria.convergence_threshold:
                convergence_progress = 0.2
            else:
                convergence_progress = 0.1 * (1 - min(improvement / criteria.convergence_threshold, 1.0))

        return min(iteration_progress + accuracy_progress + convergence_progress, 1.0)

    def _estimate_completion_time(
        self,
        metrics: ChallengeMetrics,
        criteria: SuccessCriteria,
    ) -> Optional[float]:
        """Estimate remaining time to completion."""
        if metrics.iterations < 5:
            return None

        # Average time per iteration
        avg_iteration_time = np.mean(metrics.iteration_times[-20:])

        # Current improvement rate
        improvement_rate = metrics.get_learning_rate(10)

        if improvement_rate <= 0:
            # Not improving, can't estimate
            return None

        # Estimate iterations needed to reach accuracy
        accuracy_gap = criteria.accuracy - metrics.current_accuracy
        if accuracy_gap <= 0:
            return avg_iteration_time * criteria.convergence_window

        estimated_iterations = accuracy_gap / improvement_rate
        return estimated_iterations * avg_iteration_time

    def _count_strategy_switches(self, strategy_history: List[str]) -> int:
        """Count the number of strategy switches."""
        if len(strategy_history) < 2:
            return 0
        switches = 0
        for i in range(1, len(strategy_history)):
            if strategy_history[i] != strategy_history[i-1]:
                switches += 1
        return switches

    def _adjust_curriculum(self, success: bool, accuracy: float) -> None:
        """Adjust curriculum difficulty based on performance."""
        if success and accuracy >= self.mastery_threshold:
            # Increase difficulty
            self.curriculum_difficulty = min(
                1.0,
                self.curriculum_difficulty + self.curriculum_adjustment_rate
            )
        elif not success or accuracy < self.mastery_threshold * 0.5:
            # Decrease difficulty
            self.curriculum_difficulty = max(
                0.1,
                self.curriculum_difficulty - self.curriculum_adjustment_rate
            )
        # Otherwise, keep current difficulty
