"""
Text Response Learner for ATLAS

Implements learnable text generation using biology-inspired local learning rules.
Atlas learns to generate responses through Hebbian learning, STDP for sequencing,
and competitive learning for word selection - NO backpropagation!

This enables Atlas to learn to communicate through interaction, not templates.
"""

import numpy as np
import logging
import time
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict

from .backend import xp, to_cpu, to_gpu

logger = logging.getLogger(__name__)


@dataclass
class ResponseContext:
    """Context for generating a response."""
    challenge_text: Optional[str] = None
    accuracy: float = 0.0
    iterations: int = 0
    strategy: str = "unknown"
    success: bool = False
    modalities: List[str] = field(default_factory=list)
    source: str = "free_play"  # "curriculum" or "free_play"


@dataclass
class GeneratedResponse:
    """A generated text response."""
    text: str
    confidence: float
    token_activations: List[float]
    generation_time: float


class TextResponseLearner:
    """
    Atlas learns to generate text responses using biology-inspired rules.

    Learning mechanisms:
    - Hebbian: Strengthen connections between context patterns and successful words
    - STDP: Time-dependent learning for sequential word generation
    - Competitive: Winner-take-all selection among candidate words
    - BCM: Sliding threshold to prevent saturation

    The system improves over time based on:
    - Successful challenge completions (positive reinforcement)
    - User interactions in Free Play mode
    - Context-appropriate word selection
    """

    # Base vocabulary organized by category
    VOCABULARY = {
        # Acknowledgment words
        "acknowledge": ["I", "am", "learning", "processing", "understanding", "analyzing"],

        # Progress words
        "progress": ["Progress", "improving", "advancing", "developing", "growing", "evolving"],

        # Challenge words
        "challenge": ["challenge", "task", "problem", "pattern", "data", "input"],

        # Result words
        "result": ["Successfully", "completed", "achieved", "reached", "attained", "mastered"],

        # Struggle words
        "struggle": ["Still", "working", "trying", "attempting", "difficult", "challenging"],

        # Strategy words
        "strategy": ["Using", "strategy", "Hebbian", "STDP", "BCM", "Oja", "competitive", "cooperative"],

        # Metric words
        "metrics": ["accuracy", "iterations", "performance", "score", "rate", "level"],

        # Modality words
        "modality": ["visual", "audio", "text", "sensor", "multimodal", "pattern"],

        # Emotion words
        "emotion": ["excited", "curious", "focused", "determined", "satisfied", "intrigued"],

        # Connectors
        "connector": ["and", "with", "for", "on", "the", "this", "my", "is", "was", "at"],

        # Punctuation
        "punctuation": [".", ",", "!", "?", ":", "-"],
    }

    def __init__(
        self,
        state_dim: int = 128,
        learning_rate: float = 0.01,
        stdp_window: float = 20.0,  # ms equivalent
        bcm_threshold: float = 0.5,
        response_length: int = 15,
    ):
        """
        Initialize the text response learner.

        Args:
            state_dim: Dimension of context embeddings
            learning_rate: Learning rate for weight updates
            stdp_window: Time window for STDP learning
            bcm_threshold: Initial BCM sliding threshold
            response_length: Target number of tokens in responses
        """
        self.state_dim = state_dim
        self.learning_rate = learning_rate
        self.stdp_window = stdp_window
        self.bcm_threshold = bcm_threshold
        self.response_length = response_length

        # Build flat vocabulary
        self.vocab: List[str] = []
        self.vocab_categories: Dict[str, List[int]] = {}  # category -> indices
        self._build_vocabulary()

        self.vocab_size = len(self.vocab)

        # Initialize weights (context -> word)
        # Using Xavier initialization scaled for biology
        scale = np.sqrt(2.0 / (state_dim + self.vocab_size))
        self.context_weights = xp.random.randn(state_dim, self.vocab_size).astype(xp.float32) * scale

        # Sequence weights (word -> next word) for STDP
        self.sequence_weights = xp.random.randn(self.vocab_size, self.vocab_size).astype(xp.float32) * 0.01

        # Category bias weights (for appropriate category selection based on context)
        self.category_weights = xp.random.randn(state_dim, len(self.VOCABULARY)).astype(xp.float32) * 0.1

        # BCM sliding thresholds per word
        self.bcm_thresholds = xp.ones(self.vocab_size, dtype=xp.float32) * bcm_threshold

        # Statistics
        self.total_responses_generated = 0
        self.total_feedback_received = 0
        self.avg_confidence = 0.0

        logger.info(f"TextResponseLearner initialized: vocab_size={self.vocab_size}, state_dim={state_dim}")

    def _build_vocabulary(self):
        """Build flat vocabulary from categories."""
        idx = 0
        for category, words in self.VOCABULARY.items():
            self.vocab_categories[category] = []
            for word in words:
                self.vocab.append(word)
                self.vocab_categories[category].append(idx)
                idx += 1

    def _encode_context(self, context: ResponseContext) -> 'xp.ndarray':
        """Encode response context into a state vector."""
        encoding = xp.zeros(self.state_dim, dtype=xp.float32)

        # Encode challenge text (simple character-level)
        if context.challenge_text:
            chars = list(context.challenge_text.lower())[:self.state_dim // 4]
            for i, c in enumerate(chars):
                encoding[i] = ord(c) / 256.0

        # Encode metrics
        base_idx = self.state_dim // 4
        encoding[base_idx] = context.accuracy
        encoding[base_idx + 1] = min(context.iterations / 1000.0, 1.0)
        encoding[base_idx + 2] = 1.0 if context.success else 0.0

        # Encode strategy (hash-based)
        strategy_hash = hash(context.strategy) % (self.state_dim // 4)
        encoding[base_idx + 3 + strategy_hash % 10] = 1.0

        # Encode modalities
        for i, mod in enumerate(context.modalities[:5]):
            mod_hash = hash(mod) % (self.state_dim // 8)
            encoding[self.state_dim // 2 + mod_hash] = 1.0

        # Encode source
        if context.source == "curriculum":
            encoding[self.state_dim - 2] = 1.0
        else:
            encoding[self.state_dim - 1] = 1.0

        return encoding

    def generate_response(self, context: ResponseContext) -> GeneratedResponse:
        """
        Generate a text response through learned weights.

        Uses Hebbian-selected words based on context activation,
        then STDP-influenced sequencing for coherent output.
        """
        start_time = time.time()

        # Encode context
        context_encoding = self._encode_context(context)

        # Compute word activations (Hebbian: context * weights)
        word_activations = context_encoding @ self.context_weights

        # Apply BCM modulation
        word_activations = word_activations * (word_activations > self.bcm_thresholds).astype(xp.float32)

        # Compute category preferences
        category_names = list(self.VOCABULARY.keys())
        category_activations = context_encoding @ self.category_weights

        # Select words using competitive learning
        selected_tokens = []
        token_activations = []

        # Determine which categories to emphasize based on context
        if context.success:
            priority_categories = ["result", "progress", "emotion", "metrics"]
        else:
            priority_categories = ["struggle", "progress", "challenge", "strategy"]

        # Add strategy mention
        priority_categories.append("strategy")

        # Generate tokens
        prev_token_idx = -1
        for i in range(self.response_length):
            # Combine word activations with sequence predictions
            if prev_token_idx >= 0:
                sequence_boost = self.sequence_weights[prev_token_idx, :]
                combined_activations = word_activations + sequence_boost * 0.3
            else:
                combined_activations = word_activations

            # Boost priority categories
            for cat in priority_categories:
                if cat in self.vocab_categories:
                    cat_idx = list(self.VOCABULARY.keys()).index(cat)
                    cat_boost = float(to_cpu(category_activations[cat_idx])) * 0.5
                    for word_idx in self.vocab_categories[cat]:
                        combined_activations[word_idx] += cat_boost

            # Softmax for selection probabilities
            combined_cpu = to_cpu(combined_activations)
            exp_activations = np.exp(combined_cpu - np.max(combined_cpu))
            probs = exp_activations / (exp_activations.sum() + 1e-8)

            # Sample or select top (with some randomness)
            if np.random.random() < 0.7:
                # Select top token
                token_idx = int(np.argmax(probs))
            else:
                # Sample from distribution
                token_idx = int(np.random.choice(len(self.vocab), p=probs))

            # Avoid immediate repetition
            if token_idx == prev_token_idx and len(self.vocab) > 1:
                probs[token_idx] = 0
                probs = probs / (probs.sum() + 1e-8)
                token_idx = int(np.argmax(probs))

            selected_tokens.append(self.vocab[token_idx])
            token_activations.append(float(probs[token_idx]))

            prev_token_idx = token_idx

        # Post-process response
        response_text = self._post_process_response(selected_tokens, context)

        # Compute confidence
        confidence = np.mean(token_activations)

        self.total_responses_generated += 1
        self.avg_confidence = (self.avg_confidence * (self.total_responses_generated - 1) + confidence) / self.total_responses_generated

        generation_time = time.time() - start_time

        return GeneratedResponse(
            text=response_text,
            confidence=confidence,
            token_activations=token_activations,
            generation_time=generation_time,
        )

    def _post_process_response(self, tokens: List[str], context: ResponseContext) -> str:
        """Post-process tokens into coherent response."""
        # Filter duplicates while preserving order
        seen = set()
        filtered = []
        for token in tokens:
            if token not in seen or token in [".", ",", "and", "the"]:
                filtered.append(token)
                seen.add(token)

        # Build response with structure
        response_parts = []

        # Opening based on success/failure
        if context.success:
            response_parts.append("Successfully learned")
        else:
            response_parts.append("Still learning")

        # Add relevant tokens
        content_tokens = [t for t in filtered if t not in [".", ",", "!", "?", ":", "-"]]
        response_parts.extend(content_tokens[:6])

        # Add metrics
        response_parts.append(f"Accuracy: {context.accuracy:.1%}")

        # Add strategy info
        if context.strategy and context.strategy != "unknown":
            response_parts.append(f"Strategy: {context.strategy}")

        # Join with appropriate spacing
        text = " ".join(response_parts)

        # Clean up spacing around punctuation
        text = text.replace(" .", ".").replace(" ,", ",")
        text = text.replace("  ", " ")

        return text

    def learn_from_feedback(
        self,
        response: GeneratedResponse,
        feedback_signal: float,
        context: ResponseContext,
    ) -> None:
        """
        Update weights based on feedback (reinforcement-like learning).

        Uses Hebbian learning to strengthen context->word connections
        for successful responses, and BCM to adjust thresholds.

        Args:
            response: The generated response
            feedback_signal: Positive (good response) or negative (poor response)
            context: The context that generated this response
        """
        self.total_feedback_received += 1

        context_encoding = self._encode_context(context)

        # Get word indices used in response
        used_words = response.text.split()
        used_indices = []
        for word in used_words:
            if word in self.vocab:
                used_indices.append(self.vocab.index(word))

        if not used_indices:
            return

        # Hebbian update: strengthen context->used_words if positive feedback
        for word_idx in used_indices:
            # Outer product for Hebbian learning
            delta = self.learning_rate * feedback_signal * xp.outer(
                context_encoding,
                xp.eye(self.vocab_size, dtype=xp.float32)[word_idx]
            )
            self.context_weights += delta[:, word_idx:word_idx+1].reshape(-1, 1) @ xp.ones((1, 1))

        # BCM threshold update
        for word_idx in used_indices:
            activation = float(to_cpu(context_encoding @ self.context_weights[:, word_idx]))
            # Sliding threshold moves toward average activation
            self.bcm_thresholds[word_idx] = (
                0.99 * self.bcm_thresholds[word_idx] +
                0.01 * activation
            )

        # STDP update for word sequences
        for i in range(len(used_indices) - 1):
            pre_idx = used_indices[i]
            post_idx = used_indices[i + 1]

            # Pre before post -> potentiate
            dt = 1.0  # Positive time delta
            stdp_delta = self.learning_rate * feedback_signal * np.exp(-dt / self.stdp_window)
            self.sequence_weights[pre_idx, post_idx] += stdp_delta

        logger.debug(f"Learned from feedback: signal={feedback_signal:.2f}, words={len(used_indices)}")

    def get_stats(self) -> Dict[str, Any]:
        """Get learner statistics."""
        return {
            "vocab_size": self.vocab_size,
            "total_responses": self.total_responses_generated,
            "total_feedback": self.total_feedback_received,
            "avg_confidence": self.avg_confidence,
            "learning_rate": self.learning_rate,
            "bcm_threshold_mean": float(to_cpu(xp.mean(self.bcm_thresholds))),
        }
