"""
Challenge Learner - Main Interface for Challenge-Based Learning

This is the primary entry point for teaching ATLAS new tasks through challenges.
Supports both natural language descriptions and structured data input.

All learning uses biology-inspired local plasticity rules (Hebbian, STDP, BCM, etc.)
with NO backpropagation.

Example Usage:
    ```python
    from self_organizing_av_system.core import ChallengeLearner

    # Create learner
    learner = ChallengeLearner()

    # Learn from natural language
    result = learner.learn("Learn to classify handwritten digits")

    # Learn from structured data
    result = learner.learn_from_data(
        data=images,
        labels=labels,
        success_criteria={"accuracy": 0.95}
    )

    # Check learned capabilities
    for cap in learner.get_capabilities():
        print(f"{cap.name}: {cap.proficiency:.0%}")
    ```
"""

import numpy as np
from typing import Dict, List, Optional, Any, Union, Callable
import logging
import time

from .challenge import (
    Challenge,
    ChallengeType,
    ChallengeStatus,
    Modality,
    TrainingData,
    SuccessCriteria,
    LearningResult,
    LearnedCapability,
    ProgressReport,
)
from .challenge_parser import ChallengeParser
from .learning_engine import LearningEngine, HAS_SEMANTIC_MEMORY
from .progress_tracker import ProgressTracker
from .meta_learning import MetaLearner
from .episodic_memory import EpisodicMemory

# Optional import
if HAS_SEMANTIC_MEMORY:
    from .semantic_memory import SemanticMemory
else:
    SemanticMemory = None

logger = logging.getLogger(__name__)


class ChallengeLearner:
    """
    Main interface for challenge-based learning in ATLAS.

    Enables learning new tasks from:
    - Natural language descriptions ("Learn to recognize cats")
    - Structured data with labels and success criteria

    All learning uses biology-inspired local plasticity rules:
    - Hebbian: "Fire together, wire together"
    - STDP: Spike-timing dependent plasticity
    - BCM: Bienenstock-Cooper-Munro sliding threshold
    - Oja: Normalized Hebbian
    - Competitive: Winner-take-all
    - Cooperative: Ensemble learning
    - Anti-Hebbian: Decorrelation

    NO backpropagation is used.
    """

    def __init__(
        self,
        state_dim: int = 128,
        learning_rate: float = 0.01,
        batch_size: int = 32,
        verbose: bool = True,
    ):
        """
        Initialize the ChallengeLearner.

        Args:
            state_dim: Dimension of internal state representations
            learning_rate: Base learning rate for plasticity rules
            batch_size: Batch size for training
            verbose: Whether to print progress messages
        """
        self.state_dim = state_dim
        self.verbose = verbose

        # Initialize components
        self.parser = ChallengeParser()
        self.progress_tracker = ProgressTracker()

        self.meta_learner = MetaLearner(
            num_strategies=7,
        )

        self.episodic_memory = EpisodicMemory(
            state_size=state_dim,
            max_episodes=10000,
        )

        # Semantic memory is optional (requires networkx)
        if HAS_SEMANTIC_MEMORY and SemanticMemory is not None:
            self.semantic_memory = SemanticMemory(
                embedding_dim=state_dim,
            )
        else:
            self.semantic_memory = None

        self.learning_engine = LearningEngine(
            state_dim=state_dim,
            meta_learner=self.meta_learner,
            episodic_memory=self.episodic_memory,
            semantic_memory=self.semantic_memory,
            progress_tracker=self.progress_tracker,
            learning_rate=learning_rate,
            batch_size=batch_size,
        )

        # Track all challenges
        self.challenges: Dict[str, Challenge] = {}
        self.results: Dict[str, LearningResult] = {}

        logger.info("ChallengeLearner initialized")
        if self.verbose:
            print("ChallengeLearner ready - using biology-inspired learning (no backprop)")

    def learn(
        self,
        challenge: Union[str, Challenge, Dict[str, Any]],
        callback: Optional[Callable[[int, float], None]] = None,
    ) -> LearningResult:
        """
        Learn from a challenge.

        This is the main entry point. Accepts:
        - Natural language string: "Learn to classify images of cats vs dogs"
        - Challenge object: Pre-constructed Challenge
        - Dict: Structured challenge specification

        Args:
            challenge: Challenge specification (string, Challenge, or dict)
            callback: Optional callback(iteration, accuracy) for progress updates

        Returns:
            LearningResult with metrics and capability info
        """
        # Parse challenge if needed
        if isinstance(challenge, str):
            parsed_challenge = self.parser.parse_natural_language(challenge)
        elif isinstance(challenge, dict):
            parsed_challenge = self.parser.parse_structured(challenge)
        elif isinstance(challenge, Challenge):
            parsed_challenge = challenge
        else:
            raise ValueError(f"Unsupported challenge type: {type(challenge)}")

        # Store challenge
        self.challenges[parsed_challenge.id] = parsed_challenge

        if self.verbose:
            self._print_challenge_info(parsed_challenge)

        # Execute learning
        result = self.learning_engine.execute_learning_loop(
            parsed_challenge,
            callback=callback or (self._progress_callback if self.verbose else None),
        )

        # Store result
        self.results[parsed_challenge.id] = result

        if self.verbose:
            self._print_result(result)

        return result

    def learn_from_description(
        self,
        description: str,
        callback: Optional[Callable[[int, float], None]] = None,
    ) -> LearningResult:
        """
        Learn from a natural language description.

        Args:
            description: Natural language description of the task
            callback: Optional progress callback

        Returns:
            LearningResult
        """
        return self.learn(description, callback=callback)

    def learn_from_data(
        self,
        data: Union[List[Any], np.ndarray],
        labels: Optional[Union[List[Any], np.ndarray]] = None,
        success_criteria: Optional[Dict[str, float]] = None,
        name: str = "data_challenge",
        description: str = "",
        modality: Optional[Modality] = None,
        callback: Optional[Callable[[int, float], None]] = None,
    ) -> LearningResult:
        """
        Learn from structured data.

        Args:
            data: Training samples (list or numpy array)
            labels: Optional labels for supervised learning
            success_criteria: Dict with criteria like {"accuracy": 0.95}
            name: Name for the challenge
            description: Optional description
            modality: Data modality (auto-detected if None)
            callback: Optional progress callback

        Returns:
            LearningResult
        """
        # Convert to lists
        samples = list(data) if isinstance(data, np.ndarray) else data
        labels_list = list(labels) if labels is not None else None

        # Auto-detect modality
        if modality is None:
            modality = self.parser._infer_modality_from_data(samples)

        # Create training data
        training_data = TrainingData(
            samples=samples,
            labels=labels_list,
            modality=modality,
        )

        # Create success criteria
        criteria = SuccessCriteria(
            accuracy=success_criteria.get('accuracy', 0.8) if success_criteria else 0.8,
            min_samples=success_criteria.get('min_samples', 10) if success_criteria else 10,
            max_iterations=success_criteria.get('max_iterations', 1000) if success_criteria else 1000,
            custom_metrics={
                k: v for k, v in (success_criteria or {}).items()
                if k not in ['accuracy', 'min_samples', 'max_iterations']
            },
        )

        # Determine challenge type
        challenge_type = ChallengeType.PATTERN_RECOGNITION
        if labels_list is None:
            challenge_type = ChallengeType.CONCEPT_FORMATION

        # Create challenge
        challenge = Challenge(
            name=name,
            description=description or f"Learn from {len(samples)} samples",
            challenge_type=challenge_type,
            modalities=[modality],
            training_data=training_data,
            success_criteria=criteria,
            difficulty=0.5,
        )

        return self.learn(challenge, callback=callback)

    def get_progress(self, challenge_id: str) -> ProgressReport:
        """
        Get progress report for a challenge.

        Args:
            challenge_id: ID of the challenge

        Returns:
            ProgressReport with current status
        """
        if challenge_id not in self.challenges:
            raise ValueError(f"Unknown challenge: {challenge_id}")

        return self.progress_tracker.get_progress_report(
            challenge_id,
            self.challenges[challenge_id],
        )

    def get_capabilities(self) -> List[LearnedCapability]:
        """
        Get all learned capabilities.

        Returns:
            List of LearnedCapability objects
        """
        return self.learning_engine.list_capabilities()

    def get_capability(self, capability_id: str) -> Optional[LearnedCapability]:
        """
        Get a specific capability by ID.

        Args:
            capability_id: ID of the capability

        Returns:
            LearnedCapability or None
        """
        return self.learning_engine.get_capability(capability_id)

    def apply_capability(
        self,
        capability_id: str,
        data: Union[np.ndarray, List[Any]],
    ) -> np.ndarray:
        """
        Apply a learned capability to new data.

        Args:
            capability_id: ID of the capability to apply
            data: Input data

        Returns:
            Output predictions/activations
        """
        capability = self.get_capability(capability_id)
        if capability is None:
            raise ValueError(f"Unknown capability: {capability_id}")

        if capability.weights is None:
            raise ValueError(f"Capability {capability_id} has no learned weights")

        # Convert input
        x = np.array(data)
        if x.ndim == 1:
            x = x.reshape(1, -1)

        # Apply learned weights
        output = x @ capability.weights.T

        # Update usage
        capability.use()

        return output

    def get_stats(self) -> Dict[str, Any]:
        """
        Get overall statistics.

        Returns:
            Dict with learning statistics
        """
        tracker_stats = self.progress_tracker.get_stats()
        meta_stats = self.meta_learner.get_stats()

        return {
            **tracker_stats,
            'total_challenges_attempted': len(self.challenges),
            'capabilities_learned': len(self.get_capabilities()),
            'meta_learning_selections': meta_stats.get('total_selections', 0),
            'curriculum_difficulty': tracker_stats.get('current_difficulty', 0.3),
            'episodic_memories': self.episodic_memory.get_statistics().get('total_stored', 0),
            'semantic_concepts': self.semantic_memory.get_statistics().get('total_concepts', 0) if self.semantic_memory else 0,
        }

    def _progress_callback(self, iteration: int, accuracy: float) -> None:
        """Default progress callback for verbose mode."""
        if iteration % 10 == 0:
            print(f"  Iteration {iteration}: accuracy = {accuracy:.2%}")

    def _print_challenge_info(self, challenge: Challenge) -> None:
        """Print challenge information."""
        print("\n" + "=" * 60)
        print(f"LEARNING CHALLENGE: {challenge.name}")
        print("=" * 60)
        print(f"  Type: {challenge.challenge_type.name}")
        print(f"  Modalities: {[m.name for m in challenge.modalities]}")
        print(f"  Difficulty: {challenge.difficulty:.1%}")
        print(f"  Target accuracy: {challenge.success_criteria.accuracy:.1%}")
        if challenge.training_data:
            print(f"  Training samples: {len(challenge.training_data)}")
        print("-" * 60)

    def _print_result(self, result: LearningResult) -> None:
        """Print learning result."""
        print("-" * 60)
        status = "SUCCESS" if result.success else "FAILED"
        print(f"RESULT: {status}")
        print(f"  Final accuracy: {result.accuracy:.2%}")
        print(f"  Iterations: {result.iterations}")
        print(f"  Duration: {result.duration_seconds:.2f}s")
        print(f"  Strategy: {result.strategy_used}")
        if result.capability_id:
            print(f"  Capability ID: {result.capability_id}")
        print("=" * 60 + "\n")


# Convenience function for quick learning
def learn_challenge(
    challenge: Union[str, Dict[str, Any]],
    verbose: bool = True,
) -> LearningResult:
    """
    Quick function to learn from a challenge.

    Args:
        challenge: Natural language string or structured dict
        verbose: Print progress

    Returns:
        LearningResult
    """
    learner = ChallengeLearner(verbose=verbose)
    return learner.learn(challenge)
