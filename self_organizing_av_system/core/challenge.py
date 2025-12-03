"""
Challenge-Based Learning System for ATLAS

This module defines the core data structures for challenge-based learning,
enabling ATLAS to learn new tasks from natural language descriptions or
structured data across all modalities.

The learning uses biology-inspired local plasticity rules (Hebbian, STDP, etc.)
with NO backpropagation - all learning is through local synaptic updates.
"""

import uuid
import time
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union, Callable
from enum import Enum, auto
import logging

logger = logging.getLogger(__name__)


class ChallengeType(Enum):
    """Types of learning challenges."""
    PATTERN_RECOGNITION = auto()  # Classify patterns, detect features
    PREDICTION = auto()           # Predict sequences, outcomes, future states
    PROBLEM_SOLVING = auto()      # Logic puzzles, optimization, planning
    ASSOCIATION = auto()          # Learn associations between stimuli
    SEQUENCE_LEARNING = auto()    # Learn temporal sequences
    CONCEPT_FORMATION = auto()    # Form abstract concepts from examples
    ANOMALY_DETECTION = auto()    # Detect unusual patterns
    GENERATION = auto()           # Generate new patterns/sequences


class Modality(Enum):
    """Data modalities supported for learning."""
    VISION = auto()       # Images, video frames
    AUDIO = auto()        # Sound, speech, music
    TEXT = auto()         # Natural language text
    SENSOR = auto()       # Generic sensor data (temperature, pressure, etc.)
    TIME_SERIES = auto()  # Sequential numerical data
    MULTIMODAL = auto()   # Combined modalities
    EMBEDDING = auto()    # Pre-computed embeddings/features
    SYMBOLIC = auto()     # Symbolic/logical data


class ChallengeStatus(Enum):
    """Status of a learning challenge."""
    PENDING = auto()      # Not yet started
    ANALYZING = auto()    # Parsing and understanding the challenge
    LEARNING = auto()     # Active learning in progress
    ADAPTING = auto()     # Adapting strategy due to plateau
    CONSOLIDATING = auto() # Consolidating learned knowledge
    COMPLETED = auto()    # Successfully completed
    FAILED = auto()       # Failed to meet success criteria
    PAUSED = auto()       # Temporarily paused


class DifficultyLevel(Enum):
    """Difficulty levels for curriculum learning."""
    TRIVIAL = 0.1
    EASY = 0.3
    MEDIUM = 0.5
    HARD = 0.7
    VERY_HARD = 0.9
    EXPERT = 1.0


@dataclass
class SuccessCriteria:
    """Criteria for determining challenge completion."""
    accuracy: float = 0.8           # Minimum accuracy required
    min_samples: int = 10           # Minimum samples to process
    max_iterations: int = 1000      # Maximum learning iterations
    convergence_threshold: float = 0.01  # Performance improvement threshold
    convergence_window: int = 10    # Window for checking convergence
    time_limit_seconds: Optional[float] = None  # Optional time limit
    custom_metrics: Dict[str, float] = field(default_factory=dict)

    def is_met(self, metrics: Dict[str, float], iterations: int) -> bool:
        """Check if success criteria are met."""
        if iterations < self.min_samples:
            return False
        if metrics.get('accuracy', 0) < self.accuracy:
            return False
        for metric_name, threshold in self.custom_metrics.items():
            if metrics.get(metric_name, 0) < threshold:
                return False
        return True


@dataclass
class TrainingData:
    """Container for training data across modalities."""
    samples: List[Any] = field(default_factory=list)
    labels: Optional[List[Any]] = None
    modality: Modality = Modality.EMBEDDING
    feature_dim: Optional[int] = None
    num_classes: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.samples and self.feature_dim is None:
            # Try to infer feature dimension
            sample = self.samples[0]
            if isinstance(sample, np.ndarray):
                self.feature_dim = sample.shape[-1] if sample.ndim > 0 else 1
            elif isinstance(sample, (list, tuple)):
                self.feature_dim = len(sample)

        if self.labels is not None and self.num_classes is None:
            # Try to infer number of classes
            try:
                # Only works for hashable labels
                if self.labels and not isinstance(self.labels[0], (np.ndarray, list)):
                    unique_labels = set(self.labels)
                    self.num_classes = len(unique_labels)
            except (TypeError, IndexError):
                pass  # Can't infer num_classes from these labels

    def __len__(self) -> int:
        return len(self.samples)

    def get_batch(self, batch_size: int, shuffle: bool = True) -> tuple:
        """Get a batch of training samples."""
        indices = np.arange(len(self.samples))
        if shuffle:
            np.random.shuffle(indices)
        indices = indices[:batch_size]

        batch_samples = [self.samples[i] for i in indices]
        batch_labels = [self.labels[i] for i in indices] if self.labels else None

        return batch_samples, batch_labels


@dataclass
class Challenge:
    """
    Represents a learning challenge for ATLAS.

    A challenge can be created from natural language description or
    structured data. ATLAS will use its biology-inspired learning
    systems to master the challenge.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = ""
    description: str = ""
    challenge_type: ChallengeType = ChallengeType.PATTERN_RECOGNITION
    modalities: List[Modality] = field(default_factory=lambda: [Modality.EMBEDDING])
    training_data: Optional[TrainingData] = None
    success_criteria: SuccessCriteria = field(default_factory=SuccessCriteria)
    difficulty: float = 0.5  # 0.0 - 1.0
    status: ChallengeStatus = ChallengeStatus.PENDING
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.name and self.description:
            # Generate name from description
            words = self.description.split()[:5]
            self.name = "_".join(words).lower().replace(".", "")

    def start(self) -> None:
        """Mark the challenge as started."""
        self.status = ChallengeStatus.ANALYZING
        self.started_at = time.time()
        logger.info(f"Challenge '{self.name}' started")

    def complete(self, success: bool = True) -> None:
        """Mark the challenge as completed."""
        self.status = ChallengeStatus.COMPLETED if success else ChallengeStatus.FAILED
        self.completed_at = time.time()
        duration = self.completed_at - (self.started_at or self.created_at)
        logger.info(f"Challenge '{self.name}' {'completed' if success else 'failed'} in {duration:.2f}s")

    @property
    def duration(self) -> Optional[float]:
        """Get the duration of the challenge in seconds."""
        if self.started_at is None:
            return None
        end_time = self.completed_at or time.time()
        return end_time - self.started_at

    def to_task_characteristics(self) -> Dict[str, float]:
        """
        Convert challenge to task characteristics for MetaLearner.

        These characteristics help the meta-learner select the best
        learning strategy (Hebbian, STDP, BCM, etc.).
        """
        characteristics = {
            'difficulty': self.difficulty,
            'temporal': 1.0 if self.challenge_type in [
                ChallengeType.SEQUENCE_LEARNING,
                ChallengeType.PREDICTION
            ] else 0.3,
            'spatial': 1.0 if Modality.VISION in self.modalities else 0.3,
            'complexity': self.difficulty * 0.8 + 0.2,
            'noise_level': self.metadata.get('noise_level', 0.1),
            'sparsity': self.metadata.get('sparsity', 0.5),
        }

        # Add data-specific characteristics if available
        if self.training_data:
            if self.training_data.feature_dim:
                # Normalize to 0-1 range (assuming max dim ~1000)
                characteristics['dimensionality'] = min(
                    self.training_data.feature_dim / 1000, 1.0
                )
            if self.training_data.num_classes:
                # More classes = higher complexity
                characteristics['num_classes'] = min(
                    self.training_data.num_classes / 100, 1.0
                )

        return characteristics


@dataclass
class LearningResult:
    """Result of a challenge learning attempt."""
    challenge_id: str
    challenge_name: str
    success: bool
    accuracy: float
    iterations: int
    duration_seconds: float
    strategy_used: str
    learning_curve: List[float] = field(default_factory=list)
    final_metrics: Dict[str, float] = field(default_factory=dict)
    capability_id: Optional[str] = None
    error_message: Optional[str] = None

    def __str__(self) -> str:
        status = "SUCCESS" if self.success else "FAILED"
        return (
            f"LearningResult({status}): {self.challenge_name}\n"
            f"  Accuracy: {self.accuracy:.2%}\n"
            f"  Iterations: {self.iterations}\n"
            f"  Duration: {self.duration_seconds:.2f}s\n"
            f"  Strategy: {self.strategy_used}"
        )


@dataclass
class LearnedCapability:
    """Represents a capability learned from a challenge."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = ""
    description: str = ""
    challenge_id: str = ""
    challenge_type: ChallengeType = ChallengeType.PATTERN_RECOGNITION
    modalities: List[Modality] = field(default_factory=list)
    proficiency: float = 0.0  # 0.0 - 1.0
    learned_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)
    use_count: int = 0
    weights: Optional[np.ndarray] = None  # Learned weights for this capability
    metadata: Dict[str, Any] = field(default_factory=dict)

    def use(self) -> None:
        """Record usage of this capability."""
        self.use_count += 1
        self.last_used = time.time()

    def update_proficiency(self, performance: float, decay: float = 0.1) -> None:
        """Update proficiency based on performance."""
        # Exponential moving average
        self.proficiency = (1 - decay) * self.proficiency + decay * performance


@dataclass
class ProgressReport:
    """Progress report for a learning challenge."""
    challenge_id: str
    challenge_name: str
    status: ChallengeStatus
    progress_percent: float
    current_accuracy: float
    iterations_completed: int
    time_elapsed_seconds: float
    current_strategy: str
    learning_curve: List[float]
    strategy_switches: int = 0
    estimated_completion: Optional[float] = None  # Estimated remaining seconds

    def __str__(self) -> str:
        return (
            f"Progress: {self.challenge_name}\n"
            f"  Status: {self.status.name}\n"
            f"  Progress: {self.progress_percent:.1%}\n"
            f"  Accuracy: {self.current_accuracy:.2%}\n"
            f"  Iterations: {self.iterations_completed}\n"
            f"  Time: {self.time_elapsed_seconds:.1f}s\n"
            f"  Strategy: {self.current_strategy}"
        )
