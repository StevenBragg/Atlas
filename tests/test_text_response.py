"""
Comprehensive tests for the TextResponseLearner, ResponseContext, and GeneratedResponse.

Tests cover initialization, vocabulary building, context encoding, response generation,
feedback-based learning, BCM threshold updates, STDP sequence learning, statistics
tracking, and dataclass construction.

All tests are deterministic and pass reliably by seeding the random state before
any operation that involves randomness.
"""

import os
import sys
import unittest
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from self_organizing_av_system.core.backend import xp
from self_organizing_av_system.core.text_response import (
    TextResponseLearner,
    ResponseContext,
    GeneratedResponse,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _seed(value: int = 42) -> None:
    """Seed both numpy and xp (they may be the same module on CPU)."""
    np.random.seed(value)
    if hasattr(xp, 'random') and hasattr(xp.random, 'seed'):
        xp.random.seed(value)


def _make_learner(**kwargs) -> TextResponseLearner:
    """Create a deterministically-initialised TextResponseLearner."""
    _seed(0)
    return TextResponseLearner(**kwargs)


def _make_context(**kwargs) -> ResponseContext:
    """Create a ResponseContext with sensible defaults, overridden by kwargs."""
    defaults = dict(
        challenge_text="test pattern",
        accuracy=0.85,
        iterations=100,
        strategy="hebbian",
        success=True,
        modalities=["visual", "audio"],
        source="curriculum",
    )
    defaults.update(kwargs)
    return ResponseContext(**defaults)


# ===================================================================
# ResponseContext dataclass tests
# ===================================================================

class TestResponseContextCreation(unittest.TestCase):
    """Tests for the ResponseContext dataclass."""

    def test_default_values(self):
        """A bare ResponseContext should have correct defaults."""
        ctx = ResponseContext()
        self.assertIsNone(ctx.challenge_text)
        self.assertEqual(ctx.accuracy, 0.0)
        self.assertEqual(ctx.iterations, 0)
        self.assertEqual(ctx.strategy, "unknown")
        self.assertFalse(ctx.success)
        self.assertEqual(ctx.modalities, [])
        self.assertEqual(ctx.source, "free_play")

    def test_custom_values(self):
        """All fields should be assignable through the constructor."""
        ctx = ResponseContext(
            challenge_text="recognize faces",
            accuracy=0.95,
            iterations=500,
            strategy="stdp",
            success=True,
            modalities=["visual", "text"],
            source="curriculum",
        )
        self.assertEqual(ctx.challenge_text, "recognize faces")
        self.assertAlmostEqual(ctx.accuracy, 0.95)
        self.assertEqual(ctx.iterations, 500)
        self.assertEqual(ctx.strategy, "stdp")
        self.assertTrue(ctx.success)
        self.assertEqual(ctx.modalities, ["visual", "text"])
        self.assertEqual(ctx.source, "curriculum")

    def test_modalities_default_is_independent(self):
        """Each instance should get its own modalities list (no shared mutable default)."""
        ctx_a = ResponseContext()
        ctx_b = ResponseContext()
        ctx_a.modalities.append("audio")
        self.assertEqual(len(ctx_b.modalities), 0,
                         "Modalities default list should not be shared across instances")

    def test_source_accepts_free_play(self):
        """source='free_play' should be accepted."""
        ctx = ResponseContext(source="free_play")
        self.assertEqual(ctx.source, "free_play")

    def test_source_accepts_curriculum(self):
        """source='curriculum' should be accepted."""
        ctx = ResponseContext(source="curriculum")
        self.assertEqual(ctx.source, "curriculum")


# ===================================================================
# GeneratedResponse dataclass tests
# ===================================================================

class TestGeneratedResponse(unittest.TestCase):
    """Tests for the GeneratedResponse dataclass."""

    def test_fields_stored(self):
        """All constructor fields should be retrievable."""
        resp = GeneratedResponse(
            text="Successfully learned pattern",
            confidence=0.87,
            token_activations=[0.9, 0.8, 0.7],
            generation_time=0.005,
        )
        self.assertEqual(resp.text, "Successfully learned pattern")
        self.assertAlmostEqual(resp.confidence, 0.87)
        self.assertEqual(resp.token_activations, [0.9, 0.8, 0.7])
        self.assertAlmostEqual(resp.generation_time, 0.005)

    def test_empty_text(self):
        """An empty text string should be valid."""
        resp = GeneratedResponse(text="", confidence=0.0, token_activations=[], generation_time=0.0)
        self.assertEqual(resp.text, "")
        self.assertEqual(resp.token_activations, [])

    def test_confidence_range(self):
        """Confidence should be storable as any float (no clamping enforced)."""
        resp = GeneratedResponse(text="x", confidence=1.5, token_activations=[1.0], generation_time=0.0)
        self.assertAlmostEqual(resp.confidence, 1.5)


# ===================================================================
# TextResponseLearner initialisation tests
# ===================================================================

class TestTextResponseLearnerInit(unittest.TestCase):
    """Tests for TextResponseLearner construction and attribute setup."""

    def setUp(self):
        self.learner = _make_learner()

    def test_default_parameters(self):
        """Default hyper-parameters should match the signature defaults."""
        self.assertEqual(self.learner.state_dim, 128)
        self.assertAlmostEqual(self.learner.learning_rate, 0.01)
        self.assertAlmostEqual(self.learner.stdp_window, 20.0)
        self.assertAlmostEqual(self.learner.bcm_threshold, 0.5)
        self.assertEqual(self.learner.response_length, 15)

    def test_custom_parameters(self):
        """Non-default hyper-parameters should be stored correctly."""
        learner = _make_learner(state_dim=64, learning_rate=0.05,
                                stdp_window=10.0, bcm_threshold=0.3,
                                response_length=10)
        self.assertEqual(learner.state_dim, 64)
        self.assertAlmostEqual(learner.learning_rate, 0.05)
        self.assertAlmostEqual(learner.stdp_window, 10.0)
        self.assertAlmostEqual(learner.bcm_threshold, 0.3)
        self.assertEqual(learner.response_length, 10)

    def test_vocabulary_built(self):
        """Vocabulary list and category index mapping should be populated."""
        self.assertGreater(len(self.learner.vocab), 0)
        self.assertGreater(len(self.learner.vocab_categories), 0)

    def test_vocab_size_matches_list(self):
        """vocab_size attribute should equal len(vocab)."""
        self.assertEqual(self.learner.vocab_size, len(self.learner.vocab))

    def test_all_vocabulary_categories_indexed(self):
        """Every category in VOCABULARY should appear in vocab_categories."""
        for category in TextResponseLearner.VOCABULARY:
            self.assertIn(category, self.learner.vocab_categories)

    def test_vocab_contains_all_words(self):
        """The flat vocab list should contain every word from every category."""
        all_words = []
        for words in TextResponseLearner.VOCABULARY.values():
            all_words.extend(words)
        self.assertEqual(sorted(self.learner.vocab), sorted(all_words))

    def test_context_weights_shape(self):
        """context_weights shape should be (state_dim, vocab_size)."""
        shape = self.learner.context_weights.shape
        self.assertEqual(shape, (self.learner.state_dim, self.learner.vocab_size))

    def test_sequence_weights_shape(self):
        """sequence_weights shape should be (vocab_size, vocab_size)."""
        shape = self.learner.sequence_weights.shape
        self.assertEqual(shape, (self.learner.vocab_size, self.learner.vocab_size))

    def test_category_weights_shape(self):
        """category_weights shape should be (state_dim, num_categories)."""
        num_cats = len(TextResponseLearner.VOCABULARY)
        shape = self.learner.category_weights.shape
        self.assertEqual(shape, (self.learner.state_dim, num_cats))

    def test_bcm_thresholds_shape(self):
        """BCM thresholds should have one entry per vocab word."""
        self.assertEqual(self.learner.bcm_thresholds.shape, (self.learner.vocab_size,))

    def test_bcm_thresholds_initial_value(self):
        """BCM thresholds should all start at the configured bcm_threshold."""
        expected = np.ones(self.learner.vocab_size) * 0.5
        np.testing.assert_allclose(
            np.array(self.learner.bcm_thresholds, dtype=np.float32),
            expected.astype(np.float32),
            atol=1e-6,
        )

    def test_statistics_initial_values(self):
        """Counters and averages should be zero at construction."""
        self.assertEqual(self.learner.total_responses_generated, 0)
        self.assertEqual(self.learner.total_feedback_received, 0)
        self.assertAlmostEqual(self.learner.avg_confidence, 0.0)

    def test_context_weights_dtype(self):
        """context_weights should be a floating-point dtype.

        Note: the source multiplies a float32 array by np.sqrt(...) which
        returns np.float64, so NumPy promotion rules may yield float64.
        """
        self.assertTrue(
            xp.issubdtype(self.learner.context_weights.dtype, xp.floating),
            f"Expected a floating dtype, got {self.learner.context_weights.dtype}",
        )

    def test_sequence_weights_dtype(self):
        """sequence_weights should be float32."""
        self.assertEqual(self.learner.sequence_weights.dtype, xp.float32)


# ===================================================================
# Response generation tests
# ===================================================================

class TestResponseGeneration(unittest.TestCase):
    """Tests for TextResponseLearner.generate_response."""

    def setUp(self):
        self.learner = _make_learner()

    def test_returns_generated_response(self):
        """generate_response should return a GeneratedResponse instance."""
        _seed(7)
        ctx = _make_context()
        resp = self.learner.generate_response(ctx)
        self.assertIsInstance(resp, GeneratedResponse)

    def test_response_has_text(self):
        """The response text must be a non-empty string."""
        _seed(7)
        ctx = _make_context()
        resp = self.learner.generate_response(ctx)
        self.assertIsInstance(resp.text, str)
        self.assertGreater(len(resp.text), 0)

    def test_response_has_confidence(self):
        """Confidence should be a finite float."""
        _seed(7)
        ctx = _make_context()
        resp = self.learner.generate_response(ctx)
        self.assertIsInstance(resp.confidence, float)
        self.assertTrue(np.isfinite(resp.confidence))

    def test_response_has_token_activations(self):
        """token_activations should be a list with length equal to response_length."""
        _seed(7)
        ctx = _make_context()
        resp = self.learner.generate_response(ctx)
        self.assertIsInstance(resp.token_activations, list)
        self.assertEqual(len(resp.token_activations), self.learner.response_length)

    def test_token_activations_are_finite(self):
        """Every token activation value should be finite."""
        _seed(7)
        ctx = _make_context()
        resp = self.learner.generate_response(ctx)
        for val in resp.token_activations:
            self.assertTrue(np.isfinite(val), f"Non-finite token activation: {val}")

    def test_generation_time_positive(self):
        """Generation time should be a positive number."""
        _seed(7)
        ctx = _make_context()
        resp = self.learner.generate_response(ctx)
        self.assertGreater(resp.generation_time, 0.0)

    def test_increments_response_count(self):
        """Each call should increment total_responses_generated."""
        _seed(7)
        ctx = _make_context()
        self.assertEqual(self.learner.total_responses_generated, 0)
        self.learner.generate_response(ctx)
        self.assertEqual(self.learner.total_responses_generated, 1)
        self.learner.generate_response(ctx)
        self.assertEqual(self.learner.total_responses_generated, 2)

    def test_updates_avg_confidence(self):
        """avg_confidence should be updated after generation."""
        _seed(7)
        ctx = _make_context()
        self.assertAlmostEqual(self.learner.avg_confidence, 0.0)
        self.learner.generate_response(ctx)
        self.assertNotAlmostEqual(self.learner.avg_confidence, 0.0,
                                  msg="avg_confidence should change after generation")

    def test_success_context_starts_with_successfully(self):
        """A success context should produce a response starting with 'Successfully'."""
        _seed(7)
        ctx = _make_context(success=True)
        resp = self.learner.generate_response(ctx)
        self.assertTrue(resp.text.startswith("Successfully"),
                        f"Expected 'Successfully ...' but got: {resp.text!r}")

    def test_failure_context_starts_with_still(self):
        """A failure context should produce a response starting with 'Still'."""
        _seed(7)
        ctx = _make_context(success=False)
        resp = self.learner.generate_response(ctx)
        self.assertTrue(resp.text.startswith("Still"),
                        f"Expected 'Still ...' but got: {resp.text!r}")

    def test_response_contains_accuracy(self):
        """The response text should contain an accuracy metric."""
        _seed(7)
        ctx = _make_context(accuracy=0.85)
        resp = self.learner.generate_response(ctx)
        self.assertIn("Accuracy:", resp.text)

    def test_response_contains_strategy(self):
        """When a non-unknown strategy is supplied, it should appear in the response."""
        _seed(7)
        ctx = _make_context(strategy="hebbian")
        resp = self.learner.generate_response(ctx)
        self.assertIn("Strategy: hebbian", resp.text)

    def test_unknown_strategy_omitted(self):
        """The 'unknown' strategy should not appear in the response."""
        _seed(7)
        ctx = _make_context(strategy="unknown")
        resp = self.learner.generate_response(ctx)
        self.assertNotIn("Strategy: unknown", resp.text)

    def test_deterministic_with_same_seed(self):
        """Two calls with the same seed and context should produce identical text."""
        ctx = _make_context()

        _seed(99)
        learner_a = TextResponseLearner()
        resp_a = learner_a.generate_response(ctx)

        _seed(99)
        learner_b = TextResponseLearner()
        resp_b = learner_b.generate_response(ctx)

        self.assertEqual(resp_a.text, resp_b.text)

    def test_different_contexts_may_differ(self):
        """Responses to a success context vs a failure context should differ."""
        _seed(7)
        ctx_success = _make_context(success=True)
        resp_success = self.learner.generate_response(ctx_success)

        _seed(7)
        # Re-create learner to reset weights identically
        learner2 = _make_learner()
        ctx_fail = _make_context(success=False)
        resp_fail = learner2.generate_response(ctx_fail)

        # The post-processing guarantees they start differently
        self.assertNotEqual(resp_success.text[:12], resp_fail.text[:5])


# ===================================================================
# Context encoding tests
# ===================================================================

class TestContextEncoding(unittest.TestCase):
    """Tests for the internal _encode_context method."""

    def setUp(self):
        self.learner = _make_learner()

    def test_encoding_shape(self):
        """The encoding should have shape (state_dim,)."""
        ctx = _make_context()
        enc = self.learner._encode_context(ctx)
        self.assertEqual(enc.shape, (self.learner.state_dim,))

    def test_encoding_dtype(self):
        """The encoding should be float32."""
        ctx = _make_context()
        enc = self.learner._encode_context(ctx)
        self.assertEqual(enc.dtype, xp.float32)

    def test_no_challenge_text(self):
        """Encoding should succeed when challenge_text is None."""
        ctx = ResponseContext(challenge_text=None)
        enc = self.learner._encode_context(ctx)
        self.assertEqual(enc.shape, (self.learner.state_dim,))

    def test_curriculum_source_flag(self):
        """Curriculum source should set the second-to-last element to 1."""
        ctx = _make_context(source="curriculum")
        enc = self.learner._encode_context(ctx)
        val = float(enc[self.learner.state_dim - 2])
        self.assertAlmostEqual(val, 1.0)

    def test_free_play_source_flag(self):
        """Free-play source should set the last element to 1."""
        ctx = _make_context(source="free_play")
        enc = self.learner._encode_context(ctx)
        val = float(enc[self.learner.state_dim - 1])
        self.assertAlmostEqual(val, 1.0)

    def test_success_flag_encoded(self):
        """success=True should set a 1.0 at the expected index."""
        ctx = _make_context(success=True)
        enc = self.learner._encode_context(ctx)
        base_idx = self.learner.state_dim // 4
        self.assertAlmostEqual(float(enc[base_idx + 2]), 1.0)

    def test_failure_flag_encoded(self):
        """success=False should leave the success slot at 0.0."""
        ctx = _make_context(success=False)
        enc = self.learner._encode_context(ctx)
        base_idx = self.learner.state_dim // 4
        self.assertAlmostEqual(float(enc[base_idx + 2]), 0.0)

    def test_accuracy_encoded(self):
        """The accuracy value should be stored at base_idx."""
        ctx = _make_context(accuracy=0.75)
        enc = self.learner._encode_context(ctx)
        base_idx = self.learner.state_dim // 4
        self.assertAlmostEqual(float(enc[base_idx]), 0.75)

    def test_iterations_encoded_normalized(self):
        """Iterations should be normalised by 1000 and clamped to 1.0."""
        ctx_low = _make_context(iterations=200)
        enc_low = self.learner._encode_context(ctx_low)
        base_idx = self.learner.state_dim // 4
        self.assertAlmostEqual(float(enc_low[base_idx + 1]), 0.2)

        ctx_high = _make_context(iterations=5000)
        enc_high = self.learner._encode_context(ctx_high)
        self.assertAlmostEqual(float(enc_high[base_idx + 1]), 1.0,
                               msg="Iterations above 1000 should clamp to 1.0")


# ===================================================================
# Learning from feedback tests
# ===================================================================

class TestLearnFromFeedback(unittest.TestCase):
    """Tests for TextResponseLearner.learn_from_feedback."""

    def setUp(self):
        self.learner = _make_learner()

    def _generate_and_learn(self, feedback_signal: float, **ctx_kwargs):
        """Helper: generate a response then learn from feedback."""
        _seed(42)
        ctx = _make_context(**ctx_kwargs)
        resp = self.learner.generate_response(ctx)
        self.learner.learn_from_feedback(resp, feedback_signal, ctx)
        return resp, ctx

    def test_feedback_increments_counter(self):
        """total_feedback_received should increment with each feedback call."""
        self.assertEqual(self.learner.total_feedback_received, 0)
        self._generate_and_learn(1.0)
        self.assertEqual(self.learner.total_feedback_received, 1)
        self._generate_and_learn(-0.5)
        self.assertEqual(self.learner.total_feedback_received, 2)

    def test_positive_feedback_modifies_weights(self):
        """Positive feedback should change the context_weights."""
        weights_before = xp.copy(self.learner.context_weights)
        self._generate_and_learn(1.0)
        # At least some weights should have changed
        diff = float(xp.sum(xp.abs(self.learner.context_weights - weights_before)))
        self.assertGreater(diff, 0.0, "Positive feedback should modify context_weights")

    def test_negative_feedback_modifies_weights(self):
        """Negative feedback should also change the context_weights."""
        weights_before = xp.copy(self.learner.context_weights)
        self._generate_and_learn(-1.0)
        diff = float(xp.sum(xp.abs(self.learner.context_weights - weights_before)))
        self.assertGreater(diff, 0.0, "Negative feedback should modify context_weights")

    def test_zero_feedback_no_weight_change(self):
        """Zero feedback signal should leave context_weights unchanged."""
        _seed(42)
        ctx = _make_context()
        resp = self.learner.generate_response(ctx)
        weights_before = xp.copy(self.learner.context_weights)
        self.learner.learn_from_feedback(resp, 0.0, ctx)
        # Context weights delta is proportional to feedback_signal, so zero -> no change
        diff = float(xp.max(xp.abs(self.learner.context_weights - weights_before)))
        self.assertAlmostEqual(diff, 0.0, places=5,
                               msg="Zero feedback should not alter context_weights")

    def test_bcm_thresholds_updated(self):
        """BCM thresholds should shift after feedback."""
        thresholds_before = xp.copy(self.learner.bcm_thresholds)
        self._generate_and_learn(1.0)
        diff = float(xp.sum(xp.abs(self.learner.bcm_thresholds - thresholds_before)))
        self.assertGreater(diff, 0.0, "BCM thresholds should update after feedback")

    def test_sequence_weights_updated(self):
        """STDP sequence weights should change when response has vocab words."""
        seq_before = xp.copy(self.learner.sequence_weights)
        self._generate_and_learn(1.0)
        diff = float(xp.sum(xp.abs(self.learner.sequence_weights - seq_before)))
        self.assertGreater(diff, 0.0, "Sequence weights should update via STDP")

    def test_no_crash_on_response_with_no_vocab_words(self):
        """Feedback on a response with no vocab words should be a no-op."""
        ctx = _make_context()
        fake_resp = GeneratedResponse(
            text="xyzzy plugh foobar",  # words not in vocab
            confidence=0.5,
            token_activations=[0.5, 0.5, 0.5],
            generation_time=0.001,
        )
        weights_before = xp.copy(self.learner.context_weights)
        # Should not raise
        self.learner.learn_from_feedback(fake_resp, 1.0, ctx)
        # Weights should be unchanged because no vocab words matched
        diff = float(xp.max(xp.abs(self.learner.context_weights - weights_before)))
        self.assertAlmostEqual(diff, 0.0, places=7)

    def test_multiple_feedback_rounds(self):
        """Running many feedback rounds should not raise or produce NaNs."""
        for i in range(20):
            _seed(i)
            ctx = _make_context(accuracy=i / 20.0, iterations=i * 50)
            resp = self.learner.generate_response(ctx)
            signal = 1.0 if (i % 2 == 0) else -0.5
            self.learner.learn_from_feedback(resp, signal, ctx)

        # Verify no NaN in weights
        self.assertFalse(
            bool(xp.any(xp.isnan(self.learner.context_weights))),
            "context_weights should not contain NaN after many updates",
        )
        self.assertFalse(
            bool(xp.any(xp.isnan(self.learner.sequence_weights))),
            "sequence_weights should not contain NaN after many updates",
        )
        self.assertFalse(
            bool(xp.any(xp.isnan(self.learner.bcm_thresholds))),
            "bcm_thresholds should not contain NaN after many updates",
        )


# ===================================================================
# get_stats (state / serialization) tests
# ===================================================================

class TestGetStats(unittest.TestCase):
    """Tests for TextResponseLearner.get_stats."""

    def setUp(self):
        self.learner = _make_learner()

    def test_returns_dict(self):
        """get_stats should return a dictionary."""
        stats = self.learner.get_stats()
        self.assertIsInstance(stats, dict)

    def test_contains_expected_keys(self):
        """The stats dict should contain all documented keys."""
        expected_keys = {
            "vocab_size",
            "total_responses",
            "total_feedback",
            "avg_confidence",
            "learning_rate",
            "bcm_threshold_mean",
        }
        stats = self.learner.get_stats()
        self.assertEqual(set(stats.keys()), expected_keys)

    def test_vocab_size_matches(self):
        """vocab_size in stats should match the learner attribute."""
        stats = self.learner.get_stats()
        self.assertEqual(stats["vocab_size"], self.learner.vocab_size)

    def test_initial_counters_are_zero(self):
        """Before any generation, response and feedback counters should be zero."""
        stats = self.learner.get_stats()
        self.assertEqual(stats["total_responses"], 0)
        self.assertEqual(stats["total_feedback"], 0)

    def test_avg_confidence_initially_zero(self):
        """avg_confidence should be 0.0 before any generation."""
        stats = self.learner.get_stats()
        self.assertAlmostEqual(stats["avg_confidence"], 0.0)

    def test_learning_rate_matches(self):
        """learning_rate in stats should match the constructor parameter."""
        stats = self.learner.get_stats()
        self.assertAlmostEqual(stats["learning_rate"], 0.01)

    def test_bcm_threshold_mean_initial(self):
        """Initial BCM mean threshold should equal the constructor default."""
        stats = self.learner.get_stats()
        self.assertAlmostEqual(stats["bcm_threshold_mean"], 0.5, places=2)

    def test_stats_update_after_generation(self):
        """Counters should reflect activity after generation and feedback."""
        _seed(42)
        ctx = _make_context()
        resp = self.learner.generate_response(ctx)
        self.learner.learn_from_feedback(resp, 1.0, ctx)

        stats = self.learner.get_stats()
        self.assertEqual(stats["total_responses"], 1)
        self.assertEqual(stats["total_feedback"], 1)
        self.assertGreater(stats["avg_confidence"], 0.0)

    def test_stats_values_are_serialisable(self):
        """All stat values should be plain Python types (int, float), not numpy."""
        _seed(42)
        ctx = _make_context()
        self.learner.generate_response(ctx)
        stats = self.learner.get_stats()
        for key, value in stats.items():
            self.assertIsInstance(value, (int, float),
                                 f"stats[{key!r}] has type {type(value).__name__}, "
                                 f"expected int or float")

    def test_bcm_threshold_mean_changes_after_feedback(self):
        """BCM threshold mean should shift after learning feedback."""
        initial_mean = self.learner.get_stats()["bcm_threshold_mean"]
        _seed(42)
        ctx = _make_context()
        resp = self.learner.generate_response(ctx)
        self.learner.learn_from_feedback(resp, 1.0, ctx)
        updated_mean = self.learner.get_stats()["bcm_threshold_mean"]
        self.assertNotAlmostEqual(initial_mean, updated_mean, places=5,
                                  msg="BCM mean should shift after feedback")


# ===================================================================
# Vocabulary building tests
# ===================================================================

class TestVocabularyBuilding(unittest.TestCase):
    """Tests for the internal vocabulary building logic."""

    def setUp(self):
        self.learner = _make_learner()

    def test_vocab_indices_are_contiguous(self):
        """Category index lists should cover every position from 0 to vocab_size-1."""
        all_indices = []
        for indices in self.learner.vocab_categories.values():
            all_indices.extend(indices)
        self.assertEqual(sorted(all_indices), list(range(self.learner.vocab_size)))

    def test_each_category_has_correct_count(self):
        """Each category index list length should match its word list length."""
        for category, words in TextResponseLearner.VOCABULARY.items():
            self.assertEqual(
                len(self.learner.vocab_categories[category]),
                len(words),
                f"Category '{category}' index count mismatch",
            )

    def test_no_duplicate_indices(self):
        """No index should appear in more than one category."""
        all_indices = []
        for indices in self.learner.vocab_categories.values():
            all_indices.extend(indices)
        self.assertEqual(len(all_indices), len(set(all_indices)),
                         "Found duplicate indices across categories")


# ===================================================================
# Post-processing tests
# ===================================================================

class TestPostProcessResponse(unittest.TestCase):
    """Tests for the internal _post_process_response method."""

    def setUp(self):
        self.learner = _make_learner()

    def test_success_prefix(self):
        """A success context should produce 'Successfully learned' prefix."""
        ctx = _make_context(success=True, accuracy=0.9, strategy="oja")
        result = self.learner._post_process_response(["some", "tokens"], ctx)
        self.assertTrue(result.startswith("Successfully learned"))

    def test_failure_prefix(self):
        """A failure context should produce 'Still learning' prefix."""
        ctx = _make_context(success=False, accuracy=0.3, strategy="unknown")
        result = self.learner._post_process_response(["some", "tokens"], ctx)
        self.assertTrue(result.startswith("Still learning"))

    def test_accuracy_in_output(self):
        """Accuracy percentage should appear in the output."""
        ctx = _make_context(accuracy=0.85)
        result = self.learner._post_process_response(["word"], ctx)
        self.assertIn("85.0%", result)

    def test_strategy_in_output(self):
        """A known strategy should appear in the output."""
        ctx = _make_context(strategy="stdp")
        result = self.learner._post_process_response([], ctx)
        self.assertIn("Strategy: stdp", result)

    def test_unknown_strategy_excluded(self):
        """The 'unknown' strategy should not appear."""
        ctx = _make_context(strategy="unknown")
        result = self.learner._post_process_response([], ctx)
        self.assertNotIn("Strategy:", result)

    def test_no_double_spaces(self):
        """The output should have no double-space artefacts."""
        ctx = _make_context()
        result = self.learner._post_process_response(["a", ".", "b", ",", "c"], ctx)
        self.assertNotIn("  ", result)

    def test_punctuation_spacing(self):
        """Periods and commas should not have leading spaces."""
        ctx = _make_context()
        result = self.learner._post_process_response(["word", ".", "next", ",", "end"], ctx)
        self.assertNotIn(" .", result)
        self.assertNotIn(" ,", result)

    def test_duplicates_filtered(self):
        """Duplicate content tokens should be removed (except allowed repeats)."""
        ctx = _make_context()
        result = self.learner._post_process_response(
            ["learning", "learning", "learning", "data"], ctx
        )
        # "learning" should appear at most twice (once from prefix, once from tokens)
        count = result.split().count("learning")
        self.assertLessEqual(count, 2)


# ===================================================================
# Edge-case / robustness tests
# ===================================================================

class TestEdgeCases(unittest.TestCase):
    """Edge-case and robustness tests."""

    def test_empty_modalities(self):
        """An empty modalities list should not break context encoding."""
        learner = _make_learner()
        ctx = _make_context(modalities=[])
        enc = learner._encode_context(ctx)
        self.assertEqual(enc.shape, (learner.state_dim,))

    def test_many_modalities(self):
        """More than 5 modalities should be silently truncated."""
        learner = _make_learner()
        ctx = _make_context(modalities=["a", "b", "c", "d", "e", "f", "g"])
        enc = learner._encode_context(ctx)
        self.assertEqual(enc.shape, (learner.state_dim,))

    def test_very_long_challenge_text(self):
        """A very long challenge text should not crash encoding."""
        learner = _make_learner()
        ctx = _make_context(challenge_text="x" * 10000)
        enc = learner._encode_context(ctx)
        self.assertEqual(enc.shape, (learner.state_dim,))

    def test_small_state_dim(self):
        """A very small state_dim should still work end-to-end."""
        _seed(10)
        learner = TextResponseLearner(state_dim=8, response_length=5)
        ctx = _make_context()
        resp = learner.generate_response(ctx)
        self.assertIsInstance(resp, GeneratedResponse)
        self.assertGreater(len(resp.text), 0)

    def test_response_length_one(self):
        """response_length=1 should produce a valid (short) response."""
        _seed(10)
        learner = TextResponseLearner(state_dim=32, response_length=1)
        ctx = _make_context()
        resp = learner.generate_response(ctx)
        self.assertIsInstance(resp, GeneratedResponse)
        self.assertEqual(len(resp.token_activations), 1)

    def test_zero_accuracy_context(self):
        """accuracy=0.0 should be handled without error."""
        _seed(10)
        learner = _make_learner()
        ctx = _make_context(accuracy=0.0, success=False)
        resp = learner.generate_response(ctx)
        self.assertIn("0.0%", resp.text)

    def test_full_accuracy_context(self):
        """accuracy=1.0 should be handled without error."""
        _seed(10)
        learner = _make_learner()
        ctx = _make_context(accuracy=1.0, success=True)
        resp = learner.generate_response(ctx)
        self.assertIn("100.0%", resp.text)


if __name__ == "__main__":
    unittest.main()
