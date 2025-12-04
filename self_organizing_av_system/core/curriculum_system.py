"""
Curriculum System for ATLAS

Implements a school-like curriculum with predefined levels and multimodal challenges.
Each level builds on previous knowledge, progressing from basic patterns to expert-level
creative problem solving.

All learning uses biology-inspired local rules (Hebbian, STDP, BCM) - NO backpropagation!
"""

import numpy as np
import logging
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum, auto
import time

from .backend import xp, to_cpu
from .challenge import Challenge, ChallengeType, Modality, SuccessCriteria, TrainingData

logger = logging.getLogger(__name__)


class CurriculumLevel(Enum):
    """Curriculum levels from beginner to expert."""
    LEVEL_1_BASIC = 1
    LEVEL_2_ASSOCIATION = 2
    LEVEL_3_INTERMEDIATE = 3
    LEVEL_4_ADVANCED = 4
    LEVEL_5_EXPERT = 5


@dataclass
class LevelInfo:
    """Information about a curriculum level."""
    level: CurriculumLevel
    name: str
    description: str
    challenges: List[Dict[str, Any]]
    unlock_threshold: float = 0.80  # Need this average accuracy to unlock next level


@dataclass
class ChallengeResult:
    """Result of a curriculum challenge attempt."""
    challenge_name: str
    level: CurriculumLevel
    accuracy: float
    passed: bool
    iterations: int
    strategy_used: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class LevelProgress:
    """Track progress within a level."""
    level: CurriculumLevel
    challenges_completed: int = 0
    challenges_total: int = 0
    accuracies: List[float] = field(default_factory=list)
    current_challenge_index: int = 0
    unlocked: bool = False
    completed: bool = False

    @property
    def average_accuracy(self) -> float:
        if not self.accuracies:
            return 0.0
        return sum(self.accuracies) / len(self.accuracies)

    @property
    def progress_percent(self) -> float:
        if self.challenges_total == 0:
            return 0.0
        return self.challenges_completed / self.challenges_total


class CurriculumSystem:
    """
    School-like curriculum with predefined levels and multimodal challenges.

    Levels:
    1. Basic Patterns - Fundamental pattern recognition (single modality)
    2. Simple Association - Cross-modal associations
    3. Intermediate - Complex multimodal reasoning
    4. Advanced - Abstract reasoning and inference
    5. Expert - Creative problem solving and transfer learning
    """

    # Predefined curriculum levels with challenges
    CURRICULUM = {
        CurriculumLevel.LEVEL_1_BASIC: LevelInfo(
            level=CurriculumLevel.LEVEL_1_BASIC,
            name="Learn to Draw",
            description="Learn to generate images on the 512x512 canvas by matching target patterns",
            unlock_threshold=0.70,
            challenges=[
                {
                    "name": "Draw solid red",
                    "description": "Learn to fill the canvas with solid red color",
                    "type": ChallengeType.GENERATION,
                    "modalities": [Modality.CANVAS],
                    "target_accuracy": 0.80,  # Raised from 0.60
                    "difficulty": 0.1,
                    "data_generator": "canvas_solid_color",
                    "target_color": [255, 0, 0],  # Red
                },
                {
                    "name": "Draw solid green",
                    "description": "Learn to fill the canvas with solid green color",
                    "type": ChallengeType.GENERATION,
                    "modalities": [Modality.CANVAS],
                    "target_accuracy": 0.80,  # Raised from 0.60
                    "difficulty": 0.1,
                    "data_generator": "canvas_solid_color",
                    "target_color": [0, 255, 0],  # Green
                },
                {
                    "name": "Draw solid blue",
                    "description": "Learn to fill the canvas with solid blue color",
                    "type": ChallengeType.GENERATION,
                    "modalities": [Modality.CANVAS],
                    "target_accuracy": 0.80,  # Raised from 0.60
                    "difficulty": 0.1,
                    "data_generator": "canvas_solid_color",
                    "target_color": [0, 0, 255],  # Blue
                },
                {
                    "name": "Draw horizontal gradient",
                    "description": "Learn to draw a left-to-right color gradient",
                    "type": ChallengeType.GENERATION,
                    "modalities": [Modality.CANVAS],
                    "target_accuracy": 0.75,  # Raised from 0.55
                    "difficulty": 0.2,
                    "data_generator": "canvas_gradient",
                    "gradient_type": "horizontal",
                },
                {
                    "name": "Draw vertical gradient",
                    "description": "Learn to draw a top-to-bottom color gradient",
                    "type": ChallengeType.GENERATION,
                    "modalities": [Modality.CANVAS],
                    "target_accuracy": 0.75,  # Raised from 0.55
                    "difficulty": 0.2,
                    "data_generator": "canvas_gradient",
                    "gradient_type": "vertical",
                },
                {
                    "name": "Draw centered circle",
                    "description": "Learn to draw a red circle in the center of the canvas",
                    "type": ChallengeType.GENERATION,
                    "modalities": [Modality.CANVAS],
                    "target_accuracy": 0.70,  # Raised from 0.50
                    "difficulty": 0.3,
                    "data_generator": "canvas_shape",
                    "shape": "circle",
                },
                {
                    "name": "Draw centered square",
                    "description": "Learn to draw a blue square in the center of the canvas",
                    "type": ChallengeType.GENERATION,
                    "modalities": [Modality.CANVAS],
                    "target_accuracy": 0.70,  # Raised from 0.50
                    "difficulty": 0.3,
                    "data_generator": "canvas_shape",
                    "shape": "square",
                },
                {
                    "name": "Draw checkerboard pattern",
                    "description": "Learn to draw a simple 4x4 checkerboard pattern",
                    "type": ChallengeType.GENERATION,
                    "modalities": [Modality.CANVAS],
                    "target_accuracy": 0.65,  # Raised from 0.45
                    "difficulty": 0.4,
                    "data_generator": "canvas_pattern",
                    "pattern": "checkerboard",
                },
            ]
        ),

        CurriculumLevel.LEVEL_2_ASSOCIATION: LevelInfo(
            level=CurriculumLevel.LEVEL_2_ASSOCIATION,
            name="Simple Association",
            description="Learn cross-modal associations between vision and audio",
            unlock_threshold=0.80,
            challenges=[
                {
                    "name": "Match shapes to sounds",
                    "description": "Learn that circles go with high tones, squares with low tones",
                    "type": ChallengeType.ASSOCIATION,
                    "modalities": [Modality.VISION, Modality.AUDIO],
                    "target_accuracy": 0.70,
                    "difficulty": 0.35,
                    "data_generator": "shape_sound_pairs",
                },
                {
                    "name": "Associate colors with tones",
                    "description": "Learn color-to-frequency mappings",
                    "type": ChallengeType.ASSOCIATION,
                    "modalities": [Modality.VISION, Modality.AUDIO],
                    "target_accuracy": 0.75,
                    "difficulty": 0.4,
                    "data_generator": "color_tone_pairs",
                },
                {
                    "name": "Predict next shape in sequence",
                    "description": "Learn simple visual sequences (A-B-A-B pattern)",
                    "type": ChallengeType.SEQUENCE_LEARNING,
                    "modalities": [Modality.VISION],
                    "target_accuracy": 0.70,
                    "difficulty": 0.35,
                    "data_generator": "shape_sequences",
                },
            ]
        ),

        CurriculumLevel.LEVEL_3_INTERMEDIATE: LevelInfo(
            level=CurriculumLevel.LEVEL_3_INTERMEDIATE,
            name="Intermediate",
            description="Complex multimodal reasoning with multiple inputs",
            unlock_threshold=0.80,
            challenges=[
                {
                    "name": "Identify objects with audio cues",
                    "description": "Recognize visual objects while processing related audio descriptions",
                    "type": ChallengeType.PATTERN_RECOGNITION,
                    "modalities": [Modality.VISION, Modality.AUDIO, Modality.TEXT],
                    "target_accuracy": 0.75,
                    "difficulty": 0.5,
                    "data_generator": "object_audio_text",
                },
                {
                    "name": "Match visual scenes to audio environments",
                    "description": "Associate visual scenes with corresponding ambient sounds",
                    "type": ChallengeType.ASSOCIATION,
                    "modalities": [Modality.VISION, Modality.AUDIO],
                    "target_accuracy": 0.75,
                    "difficulty": 0.55,
                    "data_generator": "scene_audio_pairs",
                },
                {
                    "name": "Predict audio from visual motion",
                    "description": "Learn that fast motion predicts high-frequency sounds",
                    "type": ChallengeType.PREDICTION,
                    "modalities": [Modality.VISION, Modality.AUDIO],
                    "target_accuracy": 0.70,
                    "difficulty": 0.5,
                    "data_generator": "motion_audio",
                },
            ]
        ),

        CurriculumLevel.LEVEL_4_ADVANCED: LevelInfo(
            level=CurriculumLevel.LEVEL_4_ADVANCED,
            name="Advanced",
            description="Abstract reasoning, analogies, and inference",
            unlock_threshold=0.85,
            challenges=[
                {
                    "name": "Visual analogy solving",
                    "description": "Solve visual analogies: A is to B as C is to ?",
                    "type": ChallengeType.PROBLEM_SOLVING,
                    "modalities": [Modality.VISION],
                    "target_accuracy": 0.70,
                    "difficulty": 0.65,
                    "data_generator": "visual_analogies",
                },
                {
                    "name": "Audio-visual synchronization",
                    "description": "Detect when audio and visual events are synchronized vs offset",
                    "type": ChallengeType.PATTERN_RECOGNITION,
                    "modalities": [Modality.VISION, Modality.AUDIO, Modality.TIME_SERIES],
                    "target_accuracy": 0.75,
                    "difficulty": 0.7,
                    "data_generator": "av_sync",
                },
                {
                    "name": "Causal relationship inference",
                    "description": "Learn that visual event A causes audio event B",
                    "type": ChallengeType.PROBLEM_SOLVING,
                    "modalities": [Modality.VISION, Modality.AUDIO],
                    "target_accuracy": 0.70,
                    "difficulty": 0.7,
                    "data_generator": "causal_pairs",
                },
            ]
        ),

        CurriculumLevel.LEVEL_5_EXPERT: LevelInfo(
            level=CurriculumLevel.LEVEL_5_EXPERT,
            name="Expert",
            description="Creative problem solving and transfer learning",
            unlock_threshold=0.90,
            challenges=[
                {
                    "name": "Generate novel patterns",
                    "description": "Create new visual-audio patterns from learned examples",
                    "type": ChallengeType.GENERATION,
                    "modalities": [Modality.VISION, Modality.AUDIO],
                    "target_accuracy": 0.65,
                    "difficulty": 0.8,
                    "data_generator": "generative_patterns",
                },
                {
                    "name": "Explain concept in symbols",
                    "description": "Express learned visual concepts using symbolic representations",
                    "type": ChallengeType.CONCEPT_FORMATION,
                    "modalities": [Modality.TEXT, Modality.SYMBOLIC],
                    "target_accuracy": 0.70,
                    "difficulty": 0.85,
                    "data_generator": "concept_symbols",
                },
                {
                    "name": "Transfer to new domain",
                    "description": "Apply learned patterns to completely new modality combinations",
                    "type": ChallengeType.PATTERN_RECOGNITION,
                    "modalities": [Modality.MULTIMODAL],
                    "target_accuracy": 0.65,
                    "difficulty": 0.9,
                    "data_generator": "transfer_domain",
                },
            ]
        ),
    }

    def __init__(self, state_dim: int = 128):
        """
        Initialize the curriculum system.

        Args:
            state_dim: Dimension of neural state vectors
        """
        self.state_dim = state_dim

        # Initialize progress for all levels
        self.progress: Dict[CurriculumLevel, LevelProgress] = {}
        for level in CurriculumLevel:
            level_info = self.CURRICULUM[level]
            self.progress[level] = LevelProgress(
                level=level,
                challenges_total=len(level_info.challenges),
                unlocked=(level == CurriculumLevel.LEVEL_1_BASIC),  # First level unlocked
            )

        # Track all results
        self.results: List[ChallengeResult] = []

        # Current state
        self.current_level = CurriculumLevel.LEVEL_1_BASIC

        logger.info("CurriculumSystem initialized with 5 levels")

    def reset_progress(self) -> None:
        """Reset all curriculum progress to start fresh from level 1."""
        logger.info("Resetting curriculum progress...")

        # Reset progress for all levels
        for level in CurriculumLevel:
            level_info = self.CURRICULUM[level]
            self.progress[level] = LevelProgress(
                level=level,
                challenges_total=len(level_info.challenges),
                unlocked=(level == CurriculumLevel.LEVEL_1_BASIC),
            )

        # Clear results history
        self.results = []

        # Reset to level 1
        self.current_level = CurriculumLevel.LEVEL_1_BASIC

        logger.info("Curriculum progress reset - back to Level 1!")

    def get_level_info(self, level: CurriculumLevel) -> LevelInfo:
        """Get information about a specific level."""
        return self.CURRICULUM[level]

    def get_current_challenge(self) -> Optional[Dict[str, Any]]:
        """Get the current challenge to attempt."""
        level_progress = self.progress[self.current_level]
        level_info = self.CURRICULUM[self.current_level]

        if level_progress.current_challenge_index >= len(level_info.challenges):
            return None

        return level_info.challenges[level_progress.current_challenge_index]

    def create_challenge_object(self, challenge_dict: Dict[str, Any]) -> Challenge:
        """Create a Challenge object from the challenge dictionary."""
        training_data = self._generate_training_data(challenge_dict)

        return Challenge(
            name=challenge_dict["name"],
            description=challenge_dict["description"],
            challenge_type=challenge_dict["type"],
            modalities=challenge_dict["modalities"],
            training_data=training_data,
            success_criteria=SuccessCriteria(
                accuracy=challenge_dict["target_accuracy"],
                max_iterations=1000,
            ),
            difficulty=challenge_dict["difficulty"],
        )

    def _generate_training_data(self, challenge_dict: Dict[str, Any]) -> TrainingData:
        """Generate synthetic training data for a challenge."""
        generator = challenge_dict.get("data_generator", "default")
        modalities = challenge_dict["modalities"]
        num_samples = 200

        samples = []
        labels = []

        if generator == "shapes_binary":
            # Circles (0) vs Squares (1)
            for i in range(num_samples):
                label = i % 2
                img = self._generate_shape_image(label)
                samples.append(img.flatten())
                labels.append(label)

        elif generator == "colors_rgb":
            # Red (0), Green (1), Blue (2)
            for i in range(num_samples):
                label = i % 3
                img = self._generate_color_image(label)
                samples.append(img.flatten())
                labels.append(label)

        elif generator == "tones_binary":
            # High tone (0) vs Low tone (1)
            for i in range(num_samples):
                label = i % 2
                audio = self._generate_tone(label)
                samples.append(audio)
                labels.append(label)

        elif generator == "rhythms":
            # Fast (0) vs Slow (1) rhythm
            for i in range(num_samples):
                label = i % 2
                audio = self._generate_rhythm(label)
                samples.append(audio)
                labels.append(label)

        elif generator == "shape_sound_pairs":
            # Circle+HighTone (0) vs Square+LowTone (1)
            for i in range(num_samples):
                label = i % 2
                img = self._generate_shape_image(label)
                audio = self._generate_tone(label)
                combined = np.concatenate([img.flatten(), audio])
                samples.append(combined)
                labels.append(label)

        elif generator == "color_tone_pairs":
            # RGB colors paired with frequency tones
            for i in range(num_samples):
                label = i % 3
                img = self._generate_color_image(label)
                audio = self._generate_tone_frequency(label)
                combined = np.concatenate([img.flatten(), audio])
                samples.append(combined)
                labels.append(label)

        elif generator == "shape_sequences":
            # Predict next in A-B-A-B sequence
            for i in range(num_samples):
                seq_pos = i % 4
                # Sequence: Circle, Square, Circle, Square
                current = seq_pos % 2
                next_shape = (seq_pos + 1) % 2
                img = self._generate_shape_image(current)
                samples.append(img.flatten())
                labels.append(next_shape)

        elif generator == "canvas_solid_color":
            # Generate target image: solid color fill
            target_color = challenge_dict.get("target_color", [255, 0, 0])
            target_image = self._generate_canvas_solid(target_color)
            # For generation tasks, samples are the target images
            # Labels can be the flattened target for loss computation
            for i in range(num_samples):
                samples.append(target_image.copy())
                labels.append(target_image.flatten() / 255.0)  # Normalized target

        elif generator == "canvas_gradient":
            # Generate target image: color gradient
            gradient_type = challenge_dict.get("gradient_type", "horizontal")
            target_image = self._generate_canvas_gradient(gradient_type)
            for i in range(num_samples):
                samples.append(target_image.copy())
                labels.append(target_image.flatten() / 255.0)

        elif generator == "canvas_shape":
            # Generate target image: centered shape
            shape = challenge_dict.get("shape", "circle")
            target_image = self._generate_canvas_shape(shape)
            for i in range(num_samples):
                samples.append(target_image.copy())
                labels.append(target_image.flatten() / 255.0)

        elif generator == "canvas_pattern":
            # Generate target image: pattern
            pattern = challenge_dict.get("pattern", "checkerboard")
            target_image = self._generate_canvas_pattern(pattern)
            for i in range(num_samples):
                samples.append(target_image.copy())
                labels.append(target_image.flatten() / 255.0)

        else:
            # Default: random embeddings
            for i in range(num_samples):
                label = i % 2
                embedding = np.random.randn(self.state_dim)
                if label == 1:
                    embedding += 0.5  # Shift class 1
                samples.append(embedding)
                labels.append(label)

        # Handle canvas generation differently
        if generator.startswith("canvas_"):
            return TrainingData(
                samples=np.array(samples),
                labels=np.array(labels),
                modality=Modality.CANVAS,
                metadata={"challenge_dict": challenge_dict},
            )

        return TrainingData(
            samples=np.array(samples),
            labels=np.array(labels),
            modality=modalities[0] if len(modalities) == 1 else Modality.MULTIMODAL,
        )

    def _generate_shape_image(self, shape_type: int, size: int = 32) -> np.ndarray:
        """Generate a simple shape image (circle=0, square=1)."""
        img = np.zeros((size, size, 3), dtype=np.float32)
        center = size // 2

        if shape_type == 0:  # Circle
            y, x = np.ogrid[:size, :size]
            mask = (x - center)**2 + (y - center)**2 <= (size//4)**2
            img[mask] = [1.0, 0.0, 0.0]  # Red circle
        else:  # Square
            quarter = size // 4
            img[center-quarter:center+quarter, center-quarter:center+quarter] = [0.0, 0.0, 1.0]  # Blue square

        return img

    def _generate_color_image(self, color_type: int, size: int = 32) -> np.ndarray:
        """Generate a solid color image (R=0, G=1, B=2)."""
        img = np.zeros((size, size, 3), dtype=np.float32)

        if color_type == 0:
            img[:, :, 0] = 1.0  # Red
        elif color_type == 1:
            img[:, :, 1] = 1.0  # Green
        else:
            img[:, :, 2] = 1.0  # Blue

        return img

    def _generate_tone(self, tone_type: int, length: int = 64) -> np.ndarray:
        """Generate a simple tone (high=0, low=1)."""
        t = np.linspace(0, 1, length)
        freq = 10.0 if tone_type == 0 else 2.0  # High vs low frequency
        return np.sin(2 * np.pi * freq * t).astype(np.float32)

    def _generate_tone_frequency(self, freq_type: int, length: int = 64) -> np.ndarray:
        """Generate tone with specific frequency (0=low, 1=mid, 2=high)."""
        t = np.linspace(0, 1, length)
        freqs = [2.0, 5.0, 10.0]
        return np.sin(2 * np.pi * freqs[freq_type] * t).astype(np.float32)

    def _generate_rhythm(self, rhythm_type: int, length: int = 128) -> np.ndarray:
        """Generate rhythm pattern (fast=0, slow=1)."""
        pattern = np.zeros(length, dtype=np.float32)

        if rhythm_type == 0:  # Fast - every 8 samples
            for i in range(0, length, 8):
                pattern[i:min(i+4, length)] = 1.0
        else:  # Slow - every 32 samples
            for i in range(0, length, 32):
                pattern[i:min(i+16, length)] = 1.0

        return pattern

    # ==================== CANVAS TARGET GENERATORS ====================

    CANVAS_SIZE = 512  # 512x512 canvas

    def _generate_canvas_solid(self, color: List[int]) -> np.ndarray:
        """Generate a 512x512 canvas with solid color fill."""
        canvas = np.zeros((self.CANVAS_SIZE, self.CANVAS_SIZE, 3), dtype=np.uint8)
        canvas[:, :, 0] = color[0]
        canvas[:, :, 1] = color[1]
        canvas[:, :, 2] = color[2]
        return canvas

    def _generate_canvas_gradient(self, gradient_type: str = "horizontal") -> np.ndarray:
        """Generate a 512x512 canvas with color gradient."""
        canvas = np.zeros((self.CANVAS_SIZE, self.CANVAS_SIZE, 3), dtype=np.uint8)

        if gradient_type == "horizontal":
            # Left to right: black to white
            for x in range(self.CANVAS_SIZE):
                value = int(255 * x / (self.CANVAS_SIZE - 1))
                canvas[:, x, :] = value
        elif gradient_type == "vertical":
            # Top to bottom: black to white
            for y in range(self.CANVAS_SIZE):
                value = int(255 * y / (self.CANVAS_SIZE - 1))
                canvas[y, :, :] = value
        elif gradient_type == "diagonal":
            # Diagonal gradient
            for y in range(self.CANVAS_SIZE):
                for x in range(self.CANVAS_SIZE):
                    value = int(255 * (x + y) / (2 * self.CANVAS_SIZE - 2))
                    canvas[y, x, :] = value
        elif gradient_type == "radial":
            # Radial gradient from center
            center = self.CANVAS_SIZE // 2
            max_dist = np.sqrt(2) * center
            for y in range(self.CANVAS_SIZE):
                for x in range(self.CANVAS_SIZE):
                    dist = np.sqrt((x - center)**2 + (y - center)**2)
                    value = int(255 * (1 - dist / max_dist))
                    canvas[y, x, :] = max(0, value)

        return canvas

    def _generate_canvas_shape(self, shape: str = "circle") -> np.ndarray:
        """Generate a 512x512 canvas with a centered shape."""
        canvas = np.zeros((self.CANVAS_SIZE, self.CANVAS_SIZE, 3), dtype=np.uint8)
        center = self.CANVAS_SIZE // 2
        radius = self.CANVAS_SIZE // 4  # Shape takes up half the canvas

        if shape == "circle":
            # Draw a red filled circle
            y, x = np.ogrid[:self.CANVAS_SIZE, :self.CANVAS_SIZE]
            mask = (x - center)**2 + (y - center)**2 <= radius**2
            canvas[mask] = [255, 0, 0]  # Red circle

        elif shape == "square":
            # Draw a blue filled square
            half = radius
            canvas[center-half:center+half, center-half:center+half] = [0, 0, 255]

        elif shape == "triangle":
            # Draw a green filled triangle (pointing up)
            for y in range(center - radius, center + radius):
                # Width increases as we go down
                progress = (y - (center - radius)) / (2 * radius)
                half_width = int(radius * progress)
                x_start = max(0, center - half_width)
                x_end = min(self.CANVAS_SIZE, center + half_width)
                canvas[y, x_start:x_end] = [0, 255, 0]

        elif shape == "ring":
            # Draw a yellow ring
            y, x = np.ogrid[:self.CANVAS_SIZE, :self.CANVAS_SIZE]
            dist_sq = (x - center)**2 + (y - center)**2
            inner_radius = radius * 0.6
            mask = (dist_sq <= radius**2) & (dist_sq >= inner_radius**2)
            canvas[mask] = [255, 255, 0]  # Yellow ring

        return canvas

    def _generate_canvas_pattern(self, pattern: str = "checkerboard") -> np.ndarray:
        """Generate a 512x512 canvas with a pattern."""
        canvas = np.zeros((self.CANVAS_SIZE, self.CANVAS_SIZE, 3), dtype=np.uint8)

        if pattern == "checkerboard":
            # 4x4 checkerboard pattern
            cell_size = self.CANVAS_SIZE // 4
            for row in range(4):
                for col in range(4):
                    if (row + col) % 2 == 0:
                        y_start = row * cell_size
                        x_start = col * cell_size
                        canvas[y_start:y_start+cell_size, x_start:x_start+cell_size] = [255, 255, 255]

        elif pattern == "stripes_h":
            # Horizontal stripes
            stripe_height = self.CANVAS_SIZE // 8
            for i in range(8):
                if i % 2 == 0:
                    canvas[i*stripe_height:(i+1)*stripe_height, :] = [255, 255, 255]

        elif pattern == "stripes_v":
            # Vertical stripes
            stripe_width = self.CANVAS_SIZE // 8
            for i in range(8):
                if i % 2 == 0:
                    canvas[:, i*stripe_width:(i+1)*stripe_width] = [255, 255, 255]

        elif pattern == "grid":
            # Grid pattern
            line_width = 4
            cell_size = self.CANVAS_SIZE // 8
            for i in range(9):
                pos = i * cell_size
                # Horizontal lines
                canvas[max(0, pos-line_width//2):min(self.CANVAS_SIZE, pos+line_width//2), :] = [255, 255, 255]
                # Vertical lines
                canvas[:, max(0, pos-line_width//2):min(self.CANVAS_SIZE, pos+line_width//2)] = [255, 255, 255]

        elif pattern == "dots":
            # Grid of dots
            dot_radius = 20
            spacing = self.CANVAS_SIZE // 4
            for row in range(4):
                for col in range(4):
                    cy = spacing // 2 + row * spacing
                    cx = spacing // 2 + col * spacing
                    y, x = np.ogrid[:self.CANVAS_SIZE, :self.CANVAS_SIZE]
                    mask = (x - cx)**2 + (y - cy)**2 <= dot_radius**2
                    canvas[mask] = [255, 255, 255]

        return canvas

    def get_target_image(self, challenge_dict: Dict[str, Any]) -> np.ndarray:
        """Get the target image for a canvas generation challenge."""
        generator = challenge_dict.get("data_generator", "")

        if generator == "canvas_solid_color":
            return self._generate_canvas_solid(challenge_dict.get("target_color", [255, 0, 0]))
        elif generator == "canvas_gradient":
            return self._generate_canvas_gradient(challenge_dict.get("gradient_type", "horizontal"))
        elif generator == "canvas_shape":
            return self._generate_canvas_shape(challenge_dict.get("shape", "circle"))
        elif generator == "canvas_pattern":
            return self._generate_canvas_pattern(challenge_dict.get("pattern", "checkerboard"))
        else:
            # Default: black canvas
            return np.zeros((self.CANVAS_SIZE, self.CANVAS_SIZE, 3), dtype=np.uint8)

    def record_result(self, result: ChallengeResult) -> None:
        """Record a challenge result and update progress."""
        self.results.append(result)

        level_progress = self.progress[result.level]
        level_progress.accuracies.append(result.accuracy)

        if result.passed:
            level_progress.challenges_completed += 1
            level_progress.current_challenge_index += 1

            # Check if level is completed
            level_info = self.CURRICULUM[result.level]
            if level_progress.challenges_completed >= len(level_info.challenges):
                level_progress.completed = True
                self._check_unlock_next_level(result.level)

        logger.info(f"Recorded result for {result.challenge_name}: {result.accuracy:.2%} {'PASSED' if result.passed else 'FAILED'}")

    def _check_unlock_next_level(self, completed_level: CurriculumLevel) -> None:
        """Check if next level should be unlocked."""
        level_info = self.CURRICULUM[completed_level]
        level_progress = self.progress[completed_level]

        if level_progress.average_accuracy >= level_info.unlock_threshold:
            # Unlock next level
            next_level_value = completed_level.value + 1
            if next_level_value <= 5:
                next_level = CurriculumLevel(next_level_value)
                self.progress[next_level].unlocked = True
                logger.info(f"Unlocked level {next_level.value}: {self.CURRICULUM[next_level].name}")

    def advance_to_next_level(self) -> bool:
        """Advance to the next level if possible."""
        current_value = self.current_level.value
        if current_value >= 5:
            return False

        next_level = CurriculumLevel(current_value + 1)
        if self.progress[next_level].unlocked:
            self.current_level = next_level
            logger.info(f"Advanced to level {next_level.value}: {self.CURRICULUM[next_level].name}")
            return True
        return False

    def skip_challenge(self) -> bool:
        """Skip the current challenge (counts as 0% accuracy)."""
        current_challenge = self.get_current_challenge()
        if current_challenge is None:
            return False

        result = ChallengeResult(
            challenge_name=current_challenge["name"],
            level=self.current_level,
            accuracy=0.0,
            passed=False,
            iterations=0,
            strategy_used="skipped",
        )
        self.record_result(result)

        # Move to next challenge anyway
        level_progress = self.progress[self.current_level]
        level_progress.current_challenge_index += 1

        return True

    def set_level(self, level: CurriculumLevel) -> None:
        """Set the current curriculum level."""
        if self.progress[level].unlocked:
            self.current_level = level
            logger.info(f"Set level to {level.value}: {self.CURRICULUM[level].name}")
        else:
            logger.warning(f"Cannot set level {level.value} - not unlocked")

    def set_challenge_index(self, index: int) -> None:
        """Set the current challenge index within the current level."""
        level_info = self.CURRICULUM[self.current_level]
        if 0 <= index < len(level_info.challenges):
            self.progress[self.current_level].current_challenge_index = index
            logger.info(f"Set challenge index to {index}")
        else:
            logger.warning(f"Invalid challenge index: {index}")

    def is_level_unlocked(self, level: CurriculumLevel) -> bool:
        """Check if a level is unlocked."""
        return self.progress[level].unlocked

    def get_level_progress(self, level: CurriculumLevel) -> float:
        """Get progress percentage for a level (0.0 to 1.0)."""
        return self.progress[level].progress_percent

    def get_challenge_target(self, level: CurriculumLevel, challenge_idx: int) -> float:
        """Get the target accuracy for a specific challenge."""
        level_info = self.CURRICULUM.get(level)
        if level_info and 0 <= challenge_idx < len(level_info.challenges):
            return level_info.challenges[challenge_idx].get("target_accuracy", 0.7)
        return 0.7

    def update_progress(self, level: CurriculumLevel, challenge_idx: int, accuracy: float) -> None:
        """Update progress for a specific challenge."""
        level_progress = self.progress[level]
        level_info = self.CURRICULUM[level]

        if 0 <= challenge_idx < len(level_info.challenges):
            challenge = level_info.challenges[challenge_idx]
            target = challenge.get("target_accuracy", 0.7)

            level_progress.accuracies.append(accuracy)

            if accuracy >= target:
                level_progress.challenges_completed = max(
                    level_progress.challenges_completed,
                    challenge_idx + 1
                )

    def generate_challenge_data(
        self,
        level: CurriculumLevel,
        challenge_idx: int
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Generate training data for a specific challenge."""
        level_info = self.CURRICULUM.get(level)
        if not level_info or challenge_idx >= len(level_info.challenges):
            return None, None

        challenge_dict = level_info.challenges[challenge_idx]
        training_data = self._generate_training_data(challenge_dict)

        return training_data.samples, training_data.labels

    def get_stats(self) -> Dict[str, Any]:
        """Get curriculum progress statistics."""
        total_challenges = sum(p.challenges_total for p in self.progress.values())
        completed_challenges = sum(p.challenges_completed for p in self.progress.values())

        level_stats = {}
        for level, progress in self.progress.items():
            level_info = self.CURRICULUM[level]
            level_stats[level.value] = {
                "name": level_info.name,
                "unlocked": progress.unlocked,
                "completed": progress.completed,
                "challenges_completed": progress.challenges_completed,
                "challenges_total": progress.challenges_total,
                "average_accuracy": progress.average_accuracy,
                "progress_percent": progress.progress_percent,
            }

        return {
            "current_level": self.current_level.value,
            "current_level_name": self.CURRICULUM[self.current_level].name,
            "total_challenges": total_challenges,
            "completed_challenges": completed_challenges,
            "overall_progress": completed_challenges / total_challenges if total_challenges > 0 else 0,
            "levels": level_stats,
            "total_attempts": len(self.results),
        }
