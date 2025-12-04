"""
Creative Canvas Controller for ATLAS

Gives Atlas COMPLETE pixel-level control over a 512x512 RGB canvas.
Every single pixel is determined by learned weights - NO hardcoded patterns.

The canvas emerges purely from:
    canvas = sigmoid(internal_state @ learned_weights)

Patterns, colors, and structure all emerge from learning, not from
sin/cos functions or procedural generation.
"""

import numpy as np
import logging
import time
from typing import Optional, Dict, Any
from dataclasses import dataclass

from .backend import xp, to_cpu, to_gpu, HAS_GPU, sync

logger = logging.getLogger(__name__)


@dataclass
class CanvasState:
    """Current state of the creative canvas."""
    last_update: float
    frame_count: int
    internal_state_norm: float
    creativity_level: float
    weight_updates: int


class CreativeCanvasController:
    """
    Pure learned pixel control for Atlas's 512x512 RGB canvas.

    Every pixel value is determined by:
        pixel[i,j,c] = sigmoid(internal_state @ weights[:, pixel_index])

    There are NO hardcoded patterns, NO sin/cos functions, NO procedural
    generation. Everything emerges from learned weights.

    The weights are updated via Hebbian learning when Atlas receives
    reward signals from successful learning.
    """

    CANVAS_SIZE = 512

    def __init__(
        self,
        state_dim: int = 128,
        creativity_rate: float = 0.3,
    ):
        """
        Initialize the creative canvas controller.

        Args:
            state_dim: Dimension of internal state vectors
            creativity_rate: Amount of exploration noise (0.0 to 1.0)
        """
        self.state_dim = state_dim
        self.creativity_rate = creativity_rate

        # Output dimensions: 512 x 512 x 3 = 786,432 pixels
        self.output_size = self.CANVAS_SIZE * self.CANVAS_SIZE * 3

        # THE CORE: Learned weights mapping state -> every pixel
        # Shape: (state_dim, 786432) - each column controls one pixel value
        # Initialize with small random values - patterns will EMERGE from learning
        self.canvas_weights = xp.random.randn(
            self.state_dim, self.output_size
        ).astype(xp.float32) * 0.1

        # Current internal state from Atlas
        self.internal_state = xp.zeros(state_dim, dtype=xp.float32)

        # Creativity/exploration noise level
        self.creativity_level = 0.5

        # Learning is happening flag
        self.learning_active = False

        # Animation state
        self.frame_count = 0

        # Learning parameters - HIGHER rate for visible changes
        self.canvas_learning_rate = 0.05  # Higher than before
        self.hebbian_decay = 0.9995  # Slower decay
        self.last_reward = 0.0
        self.reward_history = []
        self.learning_enabled = True

        # Memory of successful states
        self.state_visual_memory = []
        self.max_memory_size = 500

        # Statistics
        self.total_frames_generated = 0
        self.weight_updates = 0

        logger.info(
            f"CreativeCanvasController initialized: "
            f"state_dim={state_dim}, canvas={self.CANVAS_SIZE}x{self.CANVAS_SIZE}, "
            f"weights={self.state_dim}x{self.output_size}"
        )

    def reset_weights(self) -> None:
        """Reset all canvas weights to random initialization (start fresh)."""
        logger.info("Resetting canvas weights to random initialization...")

        # Re-randomize weights
        self.canvas_weights = xp.random.randn(
            self.state_dim, self.output_size
        ).astype(xp.float32) * 0.1

        # Reset internal state
        self.internal_state = xp.zeros(self.state_dim, dtype=xp.float32)

        # Clear memory
        self.state_visual_memory = []
        self.reward_history = []

        # Reset stats
        self.total_frames_generated = 0
        self.weight_updates = 0
        self.last_reward = 0.0

        logger.info("Canvas weights reset complete")

    def update_internal_state(self, state: np.ndarray) -> None:
        """Update the internal state from Atlas's neural activity."""
        self.internal_state = to_gpu(state.astype(np.float32))

    def generate_canvas(self) -> np.ndarray:
        """
        Generate the 512x512 RGB canvas from PURELY LEARNED weights.

        Every pixel = sigmoid(state @ weights)

        No modes, no patterns, no sin/cos - just learned transformation.

        Returns:
            numpy array of shape (512, 512, 3) with dtype uint8
        """
        self.frame_count += 1

        # Add exploration noise for creativity
        if self.creativity_rate > 0:
            noise = xp.random.randn(self.state_dim).astype(xp.float32) * self.creativity_rate
            state = self.internal_state + noise * self.creativity_level
        else:
            state = self.internal_state

        # THE CORE TRANSFORMATION: state -> pixels via learned weights
        # This is where ALL the magic happens - pure matrix multiplication
        raw_output = state @ self.canvas_weights  # (state_dim,) @ (state_dim, 786432) -> (786432,)

        # Sigmoid activation to get values in [0, 1]
        pixels = 1.0 / (1.0 + xp.exp(-raw_output))

        # Reshape to image: (786432,) -> (512, 512, 3)
        canvas = pixels.reshape(self.CANVAS_SIZE, self.CANVAS_SIZE, 3)

        # Scale to [0, 255]
        canvas = canvas * 255.0

        # Clip and convert to uint8
        canvas = xp.clip(canvas, 0, 255).astype(xp.uint8)

        self.total_frames_generated += 1

        # Sync GPU before CPU transfer to prevent blocking main thread
        if HAS_GPU:
            sync()

        return to_cpu(canvas)

    def set_learning_active(self, active: bool) -> None:
        """Set whether learning is currently active."""
        self.learning_active = active

    def provide_reward(self, reward: float) -> None:
        """
        Provide reward signal to update canvas weights.

        When Atlas succeeds at learning, this strengthens the connection
        between the current internal state and the visual output.

        Args:
            reward: Reward value (-1 to 1, positive = success)
        """
        self.last_reward = reward
        self.reward_history.append(reward)
        if len(self.reward_history) > 100:
            self.reward_history.pop(0)

        if self.learning_enabled and abs(reward) > 0.01:
            self._update_canvas_weights(reward)

    def _update_canvas_weights(self, reward: float) -> None:
        """
        Update canvas weights using Hebbian learning.

        ΔW = learning_rate * reward * outer(state, output_gradient)

        This teaches Atlas which pixel patterns correlate with success.
        """
        state = self.internal_state
        state_norm = float(xp.linalg.norm(state))

        if state_norm < 0.01:
            return

        # Normalize state
        state_normalized = state / (state_norm + 1e-8)

        # Compute current output (what pixels we're generating)
        raw_output = state_normalized @ self.canvas_weights
        output = xp.tanh(raw_output)  # Squash to [-1, 1]

        # Hebbian update: strengthen connections that produced this output
        # when reward is positive, weaken when negative
        dW = self.canvas_learning_rate * reward * xp.outer(state_normalized, output)

        # Apply update
        self.canvas_weights += dW

        # Weight decay to prevent explosion and encourage sparsity
        self.canvas_weights *= self.hebbian_decay

        # Store successful state-reward pairs
        if reward > 0:
            self.state_visual_memory.append((to_cpu(state.copy()), reward))
            if len(self.state_visual_memory) > self.max_memory_size:
                self.state_visual_memory.pop(0)

        self.weight_updates += 1

        if self.weight_updates % 100 == 0:
            weight_norm = float(xp.linalg.norm(self.canvas_weights))
            logger.debug(f"Canvas weight update #{self.weight_updates}, norm={weight_norm:.2f}")

    def set_creativity_level(self, level: float) -> None:
        """Set creativity level (0.0 to 1.0)."""
        self.creativity_level = max(0.0, min(1.0, level))

    def get_state(self) -> CanvasState:
        """Get current canvas state."""
        return CanvasState(
            last_update=time.time(),
            frame_count=self.frame_count,
            internal_state_norm=float(to_cpu(xp.linalg.norm(self.internal_state))),
            creativity_level=self.creativity_level,
            weight_updates=self.weight_updates,
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get canvas statistics."""
        avg_reward = sum(self.reward_history) / len(self.reward_history) if self.reward_history else 0.0
        weight_norm = float(to_cpu(xp.linalg.norm(self.canvas_weights)))

        return {
            "total_frames": self.total_frames_generated,
            "creativity_level": self.creativity_level,
            "learning_active": self.learning_active,
            "state_dim": self.state_dim,
            "canvas_size": self.CANVAS_SIZE,
            "output_size": self.output_size,
            # Learning stats
            "weight_updates": self.weight_updates,
            "weight_norm": weight_norm,
            "avg_reward": avg_reward,
            "memory_size": len(self.state_visual_memory),
        }

    def recall_successful_pattern(self) -> Optional[np.ndarray]:
        """
        Recall an internal state that led to high reward.

        Returns:
            State vector that led to success, or None
        """
        if not self.state_visual_memory:
            return None

        # Find highest reward memory
        best_idx = max(
            range(len(self.state_visual_memory)),
            key=lambda i: self.state_visual_memory[i][1]
        )

        return self.state_visual_memory[best_idx][0]
