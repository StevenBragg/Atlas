"""
Challenge Parser for ATLAS

Parses natural language descriptions and structured data into
Challenge objects that can be processed by the learning system.

Uses keyword matching and pattern detection to infer:
- Challenge type (pattern recognition, prediction, etc.)
- Data modality (vision, audio, text, etc.)
- Success criteria
- Difficulty level
"""

import re
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass
import logging

from .challenge import (
    Challenge,
    ChallengeType,
    Modality,
    TrainingData,
    SuccessCriteria,
    DifficultyLevel,
)

logger = logging.getLogger(__name__)


# Keyword mappings for challenge type inference
CHALLENGE_TYPE_KEYWORDS = {
    ChallengeType.PATTERN_RECOGNITION: [
        'classify', 'recognize', 'detect', 'identify', 'categorize',
        'classification', 'recognition', 'detection', 'distinguish',
        'tell apart', 'discriminate', 'label', 'tag',
    ],
    ChallengeType.PREDICTION: [
        'predict', 'forecast', 'anticipate', 'estimate', 'project',
        'prediction', 'forecasting', 'next', 'future', 'will be',
        'going to', 'expect', 'extrapolate',
    ],
    ChallengeType.PROBLEM_SOLVING: [
        'solve', 'optimize', 'find', 'search', 'plan', 'navigate',
        'puzzle', 'problem', 'solution', 'optimal', 'best',
        'maximize', 'minimize', 'strategy', 'decision',
    ],
    ChallengeType.ASSOCIATION: [
        'associate', 'connect', 'link', 'relate', 'map',
        'association', 'correlation', 'relationship', 'pair',
        'match', 'bind', 'couple',
    ],
    ChallengeType.SEQUENCE_LEARNING: [
        'sequence', 'series', 'order', 'temporal', 'sequential',
        'pattern in time', 'time series', 'trajectory', 'path',
        'steps', 'progression', 'chain',
    ],
    ChallengeType.CONCEPT_FORMATION: [
        'concept', 'abstract', 'generalize', 'category', 'group',
        'cluster', 'similarity', 'prototype', 'exemplar',
        'form concept', 'learn concept', 'understand',
    ],
    ChallengeType.ANOMALY_DETECTION: [
        'anomaly', 'outlier', 'unusual', 'abnormal', 'rare',
        'deviation', 'exception', 'novelty', 'unexpected',
        'strange', 'irregular', 'fault',
    ],
    ChallengeType.GENERATION: [
        'generate', 'create', 'produce', 'synthesize', 'make',
        'generation', 'creative', 'compose', 'design',
        'imagination', 'dream', 'hallucinate',
    ],
}

# Keyword mappings for modality inference
MODALITY_KEYWORDS = {
    Modality.VISION: [
        'image', 'picture', 'photo', 'visual', 'video', 'frame',
        'pixel', 'camera', 'see', 'look', 'watch', 'view',
        'color', 'shape', 'face', 'object', 'scene', 'handwritten',
        'digit', 'mnist', 'cifar', 'imagenet',
    ],
    Modality.AUDIO: [
        'audio', 'sound', 'music', 'speech', 'voice', 'acoustic',
        'hear', 'listen', 'frequency', 'waveform', 'spectrogram',
        'song', 'noise', 'spoken', 'mel',
    ],
    Modality.TEXT: [
        'text', 'word', 'sentence', 'document', 'language', 'nlp',
        'read', 'write', 'parse', 'semantic', 'syntax', 'grammar',
        'sentiment', 'topic', 'entity', 'named', 'translation',
    ],
    Modality.SENSOR: [
        'sensor', 'temperature', 'pressure', 'humidity', 'motion',
        'accelerometer', 'gyroscope', 'magnetometer', 'lidar',
        'radar', 'sonar', 'gps', 'imu', 'measurement',
    ],
    Modality.TIME_SERIES: [
        'time series', 'temporal', 'stock', 'price', 'trend',
        'historical', 'sequential data', 'signal', 'ecg', 'eeg',
        'weather', 'traffic', 'sales', 'demand',
    ],
}

# Difficulty keywords
DIFFICULTY_KEYWORDS = {
    DifficultyLevel.TRIVIAL: ['trivial', 'very easy', 'simple', 'basic'],
    DifficultyLevel.EASY: ['easy', 'beginner', 'introductory', 'straightforward'],
    DifficultyLevel.MEDIUM: ['medium', 'moderate', 'intermediate', 'average'],
    DifficultyLevel.HARD: ['hard', 'difficult', 'challenging', 'advanced'],
    DifficultyLevel.VERY_HARD: ['very hard', 'very difficult', 'expert level'],
    DifficultyLevel.EXPERT: ['expert', 'master', 'extreme', 'impossible'],
}

# Accuracy keywords and values
ACCURACY_PATTERNS = [
    (r'(\d+(?:\.\d+)?)\s*%\s*accura', lambda m: float(m.group(1)) / 100),
    (r'accura\w*\s*(?:of\s*)?(\d+(?:\.\d+)?)\s*%', lambda m: float(m.group(1)) / 100),
    (r'(\d+(?:\.\d+)?)\s*%\s*correct', lambda m: float(m.group(1)) / 100),
    (r'perfect', lambda m: 1.0),
    (r'high accura', lambda m: 0.95),
    (r'good accura', lambda m: 0.85),
    (r'reasonable', lambda m: 0.75),
]


class ChallengeParser:
    """
    Parses challenges from natural language or structured data.

    Infers challenge type, modality, and success criteria from
    the input to create a Challenge object.
    """

    def __init__(self):
        """Initialize the challenge parser."""
        self._compile_patterns()
        logger.info("ChallengeParser initialized")

    def _compile_patterns(self) -> None:
        """Pre-compile regex patterns for efficiency."""
        self._accuracy_patterns = [
            (re.compile(pattern, re.IGNORECASE), extractor)
            for pattern, extractor in ACCURACY_PATTERNS
        ]

    def parse(self, input_data: Union[str, Dict[str, Any]]) -> Challenge:
        """
        Parse input into a Challenge object.

        Args:
            input_data: Natural language string or structured dict

        Returns:
            Challenge object ready for learning
        """
        if isinstance(input_data, str):
            return self.parse_natural_language(input_data)
        elif isinstance(input_data, dict):
            return self.parse_structured(input_data)
        else:
            raise ValueError(f"Unsupported input type: {type(input_data)}")

    def parse_natural_language(self, description: str) -> Challenge:
        """
        Parse a natural language description into a Challenge.

        Args:
            description: Natural language description of the challenge

        Returns:
            Challenge object
        """
        logger.info(f"Parsing NL challenge: {description[:50]}...")

        # Infer challenge type
        challenge_type = self.infer_challenge_type(description)

        # Infer modalities
        modalities = self.infer_modalities(description)

        # Extract success criteria
        success_criteria = self.extract_success_criteria(description)

        # Infer difficulty
        difficulty = self.infer_difficulty(description)

        # Generate name from description
        name = self._generate_name(description)

        challenge = Challenge(
            name=name,
            description=description,
            challenge_type=challenge_type,
            modalities=modalities,
            success_criteria=success_criteria,
            difficulty=difficulty,
            metadata={'source': 'natural_language'},
        )

        logger.info(
            f"Parsed challenge: type={challenge_type.name}, "
            f"modalities={[m.name for m in modalities]}, "
            f"difficulty={difficulty:.2f}"
        )

        return challenge

    def parse_structured(self, data: Dict[str, Any]) -> Challenge:
        """
        Parse structured data into a Challenge.

        Args:
            data: Dictionary with challenge specification

        Returns:
            Challenge object
        """
        logger.info("Parsing structured challenge")

        # Extract or create training data
        training_data = None
        if 'samples' in data or 'data' in data:
            samples = data.get('samples', data.get('data', []))
            labels = data.get('labels', data.get('targets'))
            modality = self._infer_modality_from_data(samples)

            training_data = TrainingData(
                samples=list(samples),
                labels=list(labels) if labels is not None else None,
                modality=modality,
                metadata=data.get('data_metadata', {}),
            )

        # Get challenge type
        challenge_type = ChallengeType.PATTERN_RECOGNITION
        if 'challenge_type' in data:
            if isinstance(data['challenge_type'], ChallengeType):
                challenge_type = data['challenge_type']
            elif isinstance(data['challenge_type'], str):
                challenge_type = ChallengeType[data['challenge_type'].upper()]
        elif training_data and training_data.labels is None:
            # No labels suggests unsupervised task
            challenge_type = ChallengeType.CONCEPT_FORMATION

        # Get modalities
        modalities = [Modality.EMBEDDING]
        if 'modality' in data or 'modalities' in data:
            mod_spec = data.get('modalities', data.get('modality'))
            if isinstance(mod_spec, list):
                modalities = [
                    Modality[m.upper()] if isinstance(m, str) else m
                    for m in mod_spec
                ]
            elif isinstance(mod_spec, str):
                modalities = [Modality[mod_spec.upper()]]
            elif isinstance(mod_spec, Modality):
                modalities = [mod_spec]
        elif training_data:
            modalities = [training_data.modality]

        # Get success criteria
        criteria_dict = data.get('success_criteria', data.get('criteria', {}))
        success_criteria = SuccessCriteria(
            accuracy=criteria_dict.get('accuracy', 0.8),
            min_samples=criteria_dict.get('min_samples', 10),
            max_iterations=criteria_dict.get('max_iterations', 1000),
            convergence_threshold=criteria_dict.get('convergence_threshold', 0.01),
            custom_metrics=criteria_dict.get('custom_metrics', {}),
        )

        # Get difficulty
        difficulty = data.get('difficulty', 0.5)
        if isinstance(difficulty, str):
            difficulty = DifficultyLevel[difficulty.upper()].value

        challenge = Challenge(
            name=data.get('name', 'structured_challenge'),
            description=data.get('description', ''),
            challenge_type=challenge_type,
            modalities=modalities,
            training_data=training_data,
            success_criteria=success_criteria,
            difficulty=difficulty,
            metadata=data.get('metadata', {'source': 'structured'}),
        )

        logger.info(
            f"Parsed structured challenge: type={challenge_type.name}, "
            f"samples={len(training_data) if training_data else 0}"
        )

        return challenge

    def infer_challenge_type(self, description: str) -> ChallengeType:
        """
        Infer the challenge type from description.

        Args:
            description: Natural language description

        Returns:
            Most likely ChallengeType
        """
        description_lower = description.lower()
        scores = {}

        for challenge_type, keywords in CHALLENGE_TYPE_KEYWORDS.items():
            score = sum(
                1 for keyword in keywords
                if keyword in description_lower
            )
            if score > 0:
                scores[challenge_type] = score

        if not scores:
            # Default to pattern recognition
            return ChallengeType.PATTERN_RECOGNITION

        return max(scores, key=scores.get)

    def infer_modalities(self, description: str) -> List[Modality]:
        """
        Infer data modalities from description.

        Args:
            description: Natural language description

        Returns:
            List of detected Modality values
        """
        description_lower = description.lower()
        detected = []

        for modality, keywords in MODALITY_KEYWORDS.items():
            for keyword in keywords:
                if keyword in description_lower:
                    if modality not in detected:
                        detected.append(modality)
                    break

        if not detected:
            # Default to embedding (generic)
            return [Modality.EMBEDDING]

        if len(detected) > 1:
            detected.append(Modality.MULTIMODAL)

        return detected

    def infer_difficulty(self, description: str) -> float:
        """
        Infer difficulty level from description.

        Args:
            description: Natural language description

        Returns:
            Difficulty value (0.0 - 1.0)
        """
        description_lower = description.lower()

        for level, keywords in DIFFICULTY_KEYWORDS.items():
            for keyword in keywords:
                if keyword in description_lower:
                    return level.value

        # Default medium difficulty
        return DifficultyLevel.MEDIUM.value

    def extract_success_criteria(self, description: str) -> SuccessCriteria:
        """
        Extract success criteria from description.

        Args:
            description: Natural language description

        Returns:
            SuccessCriteria object
        """
        accuracy = 0.8  # Default

        # Try to extract accuracy from description
        for pattern, extractor in self._accuracy_patterns:
            match = pattern.search(description)
            if match:
                try:
                    accuracy = extractor(match)
                    break
                except (ValueError, IndexError):
                    continue

        # Ensure valid range
        accuracy = max(0.1, min(1.0, accuracy))

        return SuccessCriteria(
            accuracy=accuracy,
            min_samples=10,
            max_iterations=1000,
        )

    def _infer_modality_from_data(self, samples: List[Any]) -> Modality:
        """Infer modality from data samples."""
        if not samples:
            return Modality.EMBEDDING

        sample = samples[0]

        # Check if numpy array
        if isinstance(sample, np.ndarray):
            if sample.ndim == 1:
                return Modality.TIME_SERIES if len(sample) > 100 else Modality.EMBEDDING
            elif sample.ndim == 2:
                # Could be image (grayscale) or spectrogram
                if sample.shape[0] == sample.shape[1]:
                    return Modality.VISION
                return Modality.AUDIO  # Likely spectrogram
            elif sample.ndim == 3:
                return Modality.VISION  # Color image
            elif sample.ndim == 4:
                return Modality.VISION  # Video or batch of images

        # Check if string
        if isinstance(sample, str):
            return Modality.TEXT

        # Check if list/tuple of numbers
        if isinstance(sample, (list, tuple)):
            if all(isinstance(x, (int, float)) for x in sample):
                return Modality.EMBEDDING

        return Modality.EMBEDDING

    def _generate_name(self, description: str) -> str:
        """Generate a short name from description."""
        # Remove common words
        stop_words = {
            'learn', 'to', 'the', 'a', 'an', 'from', 'with', 'using',
            'how', 'can', 'you', 'please', 'i', 'want', 'need',
        }

        words = description.lower().split()
        words = [w for w in words if w.isalnum() and w not in stop_words]

        # Take first 4 meaningful words
        name_words = words[:4]
        name = '_'.join(name_words) if name_words else 'challenge'

        return name[:50]  # Limit length
