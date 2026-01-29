"""
Comprehensive tests for the challenge-based learning system.

Tests cover:
- ChallengeType enum values
- ChallengeStatus enum values
- Modality enum values
- SuccessCriteria dataclass and is_met() logic
- TrainingData dataclass and batch retrieval
- Challenge dataclass, lifecycle methods, and task characteristics
- LearningResult dataclass
- LearnedCapability dataclass and usage tracking
- ProgressReport dataclass
- ChallengeLearner initialization
- ChallengeLearner.learn() with string, Challenge, and dict inputs
- ChallengeLearner.learn_from_description()
- ChallengeLearner.learn_from_data()
- ChallengeLearner.get_progress()
- ChallengeLearner.apply_capability()
- ChallengeLearner.get_capabilities()

All tests are deterministic and pass reliably. Heavy learning loops are mocked
to avoid non-determinism and long execution times.
"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from self_organizing_av_system.core.backend import xp

from self_organizing_av_system.core.challenge import (
    Challenge,
    ChallengeType,
    ChallengeStatus,
    Modality,
    DifficultyLevel,
    TrainingData,
    SuccessCriteria,
    LearningResult,
    LearnedCapability,
    ProgressReport,
)

from self_organizing_av_system.core.challenge_learner import (
    ChallengeLearner,
    learn_challenge,
)


# ---------------------------------------------------------------------------
# Helper to build a mock LearningResult for tests that exercise routing logic
# without running the full learning loop.
# ---------------------------------------------------------------------------

def _make_mock_result(challenge_id="test-id", challenge_name="test",
                      success=True, accuracy=0.95, iterations=50,
                      capability_id="cap-001"):
    """Return a deterministic LearningResult for mocking."""
    return LearningResult(
        challenge_id=challenge_id,
        challenge_name=challenge_name,
        success=success,
        accuracy=accuracy,
        iterations=iterations,
        duration_seconds=1.0,
        strategy_used="HEBBIAN",
        learning_curve=[0.1 * i for i in range(1, 11)],
        final_metrics={"best_accuracy": accuracy, "final_accuracy": accuracy},
        capability_id=capability_id,
    )


# ===================================================================
# ChallengeType enum
# ===================================================================

class TestChallengeType(unittest.TestCase):
    """Tests for the ChallengeType enum."""

    def test_all_expected_members_exist(self):
        """ChallengeType must define the eight documented challenge types."""
        expected = [
            "PATTERN_RECOGNITION",
            "PREDICTION",
            "PROBLEM_SOLVING",
            "ASSOCIATION",
            "SEQUENCE_LEARNING",
            "CONCEPT_FORMATION",
            "ANOMALY_DETECTION",
            "GENERATION",
        ]
        for name in expected:
            self.assertTrue(
                hasattr(ChallengeType, name),
                f"ChallengeType missing member '{name}'",
            )

    def test_member_count(self):
        """ChallengeType should have exactly 8 members."""
        self.assertEqual(len(ChallengeType), 8)

    def test_members_are_unique(self):
        """Each ChallengeType value should be unique."""
        values = [m.value for m in ChallengeType]
        self.assertEqual(len(values), len(set(values)))

    def test_access_by_name(self):
        """ChallengeType members must be accessible by name."""
        self.assertIs(ChallengeType["PATTERN_RECOGNITION"],
                      ChallengeType.PATTERN_RECOGNITION)
        self.assertIs(ChallengeType["GENERATION"],
                      ChallengeType.GENERATION)

    def test_name_property(self):
        """The .name property should return the string name of the member."""
        self.assertEqual(ChallengeType.PREDICTION.name, "PREDICTION")


# ===================================================================
# ChallengeStatus enum
# ===================================================================

class TestChallengeStatus(unittest.TestCase):
    """Tests for the ChallengeStatus enum."""

    def test_all_expected_members_exist(self):
        """ChallengeStatus must define the documented statuses."""
        expected = [
            "PENDING", "ANALYZING", "LEARNING", "ADAPTING",
            "CONSOLIDATING", "COMPLETED", "FAILED", "PAUSED",
        ]
        for name in expected:
            self.assertTrue(
                hasattr(ChallengeStatus, name),
                f"ChallengeStatus missing member '{name}'",
            )

    def test_member_count(self):
        """ChallengeStatus should have exactly 8 members."""
        self.assertEqual(len(ChallengeStatus), 8)


# ===================================================================
# Modality enum
# ===================================================================

class TestModality(unittest.TestCase):
    """Tests for the Modality enum."""

    def test_all_expected_members_exist(self):
        """Modality must define the documented modality types."""
        expected = [
            "VISION", "AUDIO", "TEXT", "SENSOR",
            "TIME_SERIES", "MULTIMODAL", "EMBEDDING",
            "SYMBOLIC", "CANVAS",
        ]
        for name in expected:
            self.assertTrue(
                hasattr(Modality, name),
                f"Modality missing member '{name}'",
            )

    def test_member_count(self):
        """Modality should have exactly 9 members."""
        self.assertEqual(len(Modality), 9)


# ===================================================================
# DifficultyLevel enum
# ===================================================================

class TestDifficultyLevel(unittest.TestCase):
    """Tests for the DifficultyLevel enum."""

    def test_values_are_floats(self):
        """Each DifficultyLevel value should be a float."""
        for member in DifficultyLevel:
            self.assertIsInstance(member.value, float)

    def test_ordering(self):
        """Difficulty levels should be ordered from lowest to highest."""
        self.assertLess(DifficultyLevel.TRIVIAL.value,
                        DifficultyLevel.EASY.value)
        self.assertLess(DifficultyLevel.EASY.value,
                        DifficultyLevel.MEDIUM.value)
        self.assertLess(DifficultyLevel.MEDIUM.value,
                        DifficultyLevel.HARD.value)
        self.assertLess(DifficultyLevel.HARD.value,
                        DifficultyLevel.VERY_HARD.value)
        self.assertLessEqual(DifficultyLevel.VERY_HARD.value,
                             DifficultyLevel.EXPERT.value)

    def test_known_values(self):
        """Check a few known difficulty values."""
        self.assertAlmostEqual(DifficultyLevel.TRIVIAL.value, 0.1)
        self.assertAlmostEqual(DifficultyLevel.MEDIUM.value, 0.5)
        self.assertAlmostEqual(DifficultyLevel.EXPERT.value, 1.0)


# ===================================================================
# SuccessCriteria dataclass
# ===================================================================

class TestSuccessCriteria(unittest.TestCase):
    """Tests for the SuccessCriteria dataclass."""

    def test_default_values(self):
        """SuccessCriteria with defaults should have sensible values."""
        sc = SuccessCriteria()
        self.assertAlmostEqual(sc.accuracy, 0.8)
        self.assertEqual(sc.min_samples, 10)
        self.assertEqual(sc.max_iterations, 1000)
        self.assertAlmostEqual(sc.convergence_threshold, 0.01)
        self.assertEqual(sc.convergence_window, 10)
        self.assertIsNone(sc.time_limit_seconds)
        self.assertEqual(sc.custom_metrics, {})

    def test_custom_values(self):
        """SuccessCriteria should accept custom values."""
        sc = SuccessCriteria(accuracy=0.95, min_samples=5, max_iterations=500)
        self.assertAlmostEqual(sc.accuracy, 0.95)
        self.assertEqual(sc.min_samples, 5)
        self.assertEqual(sc.max_iterations, 500)

    def test_is_met_insufficient_iterations(self):
        """is_met should return False when iterations < min_samples."""
        sc = SuccessCriteria(accuracy=0.8, min_samples=10)
        metrics = {"accuracy": 0.99}
        self.assertFalse(sc.is_met(metrics, iterations=5))

    def test_is_met_insufficient_accuracy(self):
        """is_met should return False when accuracy is below threshold."""
        sc = SuccessCriteria(accuracy=0.8, min_samples=10)
        metrics = {"accuracy": 0.5}
        self.assertFalse(sc.is_met(metrics, iterations=20))

    def test_is_met_success(self):
        """is_met should return True when all criteria are met."""
        sc = SuccessCriteria(accuracy=0.8, min_samples=10)
        metrics = {"accuracy": 0.85}
        self.assertTrue(sc.is_met(metrics, iterations=15))

    def test_is_met_with_custom_metrics_failing(self):
        """is_met should fail if a custom metric is below its threshold."""
        sc = SuccessCriteria(
            accuracy=0.8,
            min_samples=5,
            custom_metrics={"f1_score": 0.9},
        )
        metrics = {"accuracy": 0.9, "f1_score": 0.7}
        self.assertFalse(sc.is_met(metrics, iterations=10))

    def test_is_met_with_custom_metrics_passing(self):
        """is_met should pass when all custom metrics meet thresholds."""
        sc = SuccessCriteria(
            accuracy=0.8,
            min_samples=5,
            custom_metrics={"f1_score": 0.9},
        )
        metrics = {"accuracy": 0.9, "f1_score": 0.95}
        self.assertTrue(sc.is_met(metrics, iterations=10))


# ===================================================================
# TrainingData dataclass
# ===================================================================

class TestTrainingData(unittest.TestCase):
    """Tests for the TrainingData dataclass."""

    def test_default_construction(self):
        """TrainingData with defaults should have empty samples."""
        td = TrainingData()
        self.assertEqual(len(td), 0)
        self.assertIsNone(td.labels)

    def test_len(self):
        """len() on TrainingData should return the number of samples."""
        samples = [np.array([1, 2]), np.array([3, 4]), np.array([5, 6])]
        td = TrainingData(samples=samples)
        self.assertEqual(len(td), 3)

    def test_feature_dim_inference(self):
        """feature_dim should be inferred from numpy array samples."""
        samples = [np.array([1.0, 2.0, 3.0])]
        td = TrainingData(samples=samples)
        self.assertEqual(td.feature_dim, 3)

    def test_feature_dim_inference_from_list(self):
        """feature_dim should be inferred from list samples."""
        samples = [[1.0, 2.0, 3.0, 4.0]]
        td = TrainingData(samples=samples)
        self.assertEqual(td.feature_dim, 4)

    def test_num_classes_inference(self):
        """num_classes should be inferred from labels."""
        samples = [np.array([1]), np.array([2]), np.array([3])]
        labels = [0, 1, 2, 1]
        td = TrainingData(samples=samples, labels=labels)
        self.assertEqual(td.num_classes, 3)

    def test_get_batch_returns_correct_size(self):
        """get_batch should return at most batch_size samples."""
        np.random.seed(42)
        samples = [np.array([i]) for i in range(20)]
        labels = list(range(20))
        td = TrainingData(samples=samples, labels=labels)

        batch_samples, batch_labels = td.get_batch(batch_size=5, shuffle=False)
        self.assertEqual(len(batch_samples), 5)
        self.assertEqual(len(batch_labels), 5)

    def test_get_batch_no_labels(self):
        """get_batch with no labels should return None for labels."""
        np.random.seed(42)
        samples = [np.array([i]) for i in range(10)]
        td = TrainingData(samples=samples)

        batch_samples, batch_labels = td.get_batch(batch_size=3)
        self.assertEqual(len(batch_samples), 3)
        self.assertIsNone(batch_labels)

    def test_modality_default(self):
        """Default modality should be EMBEDDING."""
        td = TrainingData()
        self.assertEqual(td.modality, Modality.EMBEDDING)


# ===================================================================
# Challenge dataclass
# ===================================================================

class TestChallenge(unittest.TestCase):
    """Tests for the Challenge dataclass."""

    def test_default_construction(self):
        """Challenge with defaults should have sensible values."""
        c = Challenge()
        self.assertIsNotNone(c.id)
        self.assertEqual(len(c.id), 8)
        self.assertEqual(c.challenge_type, ChallengeType.PATTERN_RECOGNITION)
        self.assertEqual(c.status, ChallengeStatus.PENDING)
        self.assertAlmostEqual(c.difficulty, 0.5)
        self.assertIsNone(c.training_data)

    def test_custom_construction(self):
        """Challenge should accept custom parameters."""
        c = Challenge(
            name="test_challenge",
            description="A test challenge",
            challenge_type=ChallengeType.PREDICTION,
            difficulty=0.7,
        )
        self.assertEqual(c.name, "test_challenge")
        self.assertEqual(c.description, "A test challenge")
        self.assertEqual(c.challenge_type, ChallengeType.PREDICTION)
        self.assertAlmostEqual(c.difficulty, 0.7)

    def test_auto_name_from_description(self):
        """Challenge should auto-generate name from description if name is empty."""
        c = Challenge(description="learn to classify images quickly")
        self.assertIn("learn", c.name)

    def test_unique_ids(self):
        """Each Challenge instance should get a unique id."""
        c1 = Challenge()
        c2 = Challenge()
        self.assertNotEqual(c1.id, c2.id)

    def test_start_sets_status_and_timestamp(self):
        """start() should set status to ANALYZING and record started_at."""
        c = Challenge(name="test")
        self.assertEqual(c.status, ChallengeStatus.PENDING)
        self.assertIsNone(c.started_at)

        c.start()
        self.assertEqual(c.status, ChallengeStatus.ANALYZING)
        self.assertIsNotNone(c.started_at)

    def test_complete_success(self):
        """complete(success=True) should set status to COMPLETED."""
        c = Challenge(name="test")
        c.start()
        c.complete(success=True)
        self.assertEqual(c.status, ChallengeStatus.COMPLETED)
        self.assertIsNotNone(c.completed_at)

    def test_complete_failure(self):
        """complete(success=False) should set status to FAILED."""
        c = Challenge(name="test")
        c.start()
        c.complete(success=False)
        self.assertEqual(c.status, ChallengeStatus.FAILED)

    def test_duration_before_start(self):
        """duration should be None before start()."""
        c = Challenge(name="test")
        self.assertIsNone(c.duration)

    def test_duration_after_start(self):
        """duration should be a positive float after start()."""
        c = Challenge(name="test")
        c.start()
        # duration is computed dynamically
        d = c.duration
        self.assertIsNotNone(d)
        self.assertGreaterEqual(d, 0.0)

    def test_to_task_characteristics(self):
        """to_task_characteristics should return a dict with expected keys."""
        c = Challenge(
            name="test",
            challenge_type=ChallengeType.SEQUENCE_LEARNING,
            modalities=[Modality.VISION],
            difficulty=0.8,
        )
        chars = c.to_task_characteristics()
        self.assertIsInstance(chars, dict)
        self.assertIn("difficulty", chars)
        self.assertIn("temporal", chars)
        self.assertIn("spatial", chars)
        self.assertIn("complexity", chars)
        self.assertAlmostEqual(chars["difficulty"], 0.8)

    def test_to_task_characteristics_temporal(self):
        """Temporal challenge types should have temporal=1.0."""
        c = Challenge(
            name="test",
            challenge_type=ChallengeType.SEQUENCE_LEARNING,
        )
        chars = c.to_task_characteristics()
        self.assertAlmostEqual(chars["temporal"], 1.0)

    def test_to_task_characteristics_non_temporal(self):
        """Non-temporal challenge types should have temporal=0.3."""
        c = Challenge(
            name="test",
            challenge_type=ChallengeType.PATTERN_RECOGNITION,
        )
        chars = c.to_task_characteristics()
        self.assertAlmostEqual(chars["temporal"], 0.3)

    def test_to_task_characteristics_vision(self):
        """Vision modality should have spatial=1.0."""
        c = Challenge(
            name="test",
            modalities=[Modality.VISION],
        )
        chars = c.to_task_characteristics()
        self.assertAlmostEqual(chars["spatial"], 1.0)

    def test_to_task_characteristics_with_training_data(self):
        """Task characteristics should include data info when training data is present."""
        samples = [np.zeros(50) for _ in range(10)]
        labels = list(range(10))
        td = TrainingData(samples=samples, labels=labels)
        c = Challenge(name="test", training_data=td)
        chars = c.to_task_characteristics()
        self.assertIn("dimensionality", chars)
        self.assertIn("num_classes", chars)

    def test_default_modalities(self):
        """Default modalities should be [Modality.EMBEDDING]."""
        c = Challenge()
        self.assertEqual(c.modalities, [Modality.EMBEDDING])

    def test_default_success_criteria(self):
        """Default success criteria should be a SuccessCriteria instance."""
        c = Challenge()
        self.assertIsInstance(c.success_criteria, SuccessCriteria)
        self.assertAlmostEqual(c.success_criteria.accuracy, 0.8)


# ===================================================================
# LearningResult dataclass
# ===================================================================

class TestLearningResult(unittest.TestCase):
    """Tests for the LearningResult dataclass."""

    def test_construction(self):
        """LearningResult should store all provided fields."""
        lr = LearningResult(
            challenge_id="c1",
            challenge_name="test",
            success=True,
            accuracy=0.92,
            iterations=100,
            duration_seconds=5.0,
            strategy_used="HEBBIAN",
        )
        self.assertEqual(lr.challenge_id, "c1")
        self.assertTrue(lr.success)
        self.assertAlmostEqual(lr.accuracy, 0.92)
        self.assertEqual(lr.iterations, 100)
        self.assertAlmostEqual(lr.duration_seconds, 5.0)
        self.assertEqual(lr.strategy_used, "HEBBIAN")

    def test_default_optional_fields(self):
        """Optional fields should have proper defaults."""
        lr = LearningResult(
            challenge_id="c1",
            challenge_name="test",
            success=False,
            accuracy=0.5,
            iterations=10,
            duration_seconds=1.0,
            strategy_used="OJA",
        )
        self.assertEqual(lr.learning_curve, [])
        self.assertEqual(lr.final_metrics, {})
        self.assertIsNone(lr.capability_id)
        self.assertIsNone(lr.error_message)

    def test_str_representation_success(self):
        """str() on a successful result should contain 'SUCCESS'."""
        lr = LearningResult(
            challenge_id="c1",
            challenge_name="test",
            success=True,
            accuracy=0.9,
            iterations=50,
            duration_seconds=2.0,
            strategy_used="BCM",
        )
        s = str(lr)
        self.assertIn("SUCCESS", s)
        self.assertIn("test", s)

    def test_str_representation_failure(self):
        """str() on a failed result should contain 'FAILED'."""
        lr = LearningResult(
            challenge_id="c1",
            challenge_name="test",
            success=False,
            accuracy=0.3,
            iterations=100,
            duration_seconds=10.0,
            strategy_used="STDP",
        )
        s = str(lr)
        self.assertIn("FAILED", s)


# ===================================================================
# LearnedCapability dataclass
# ===================================================================

class TestLearnedCapability(unittest.TestCase):
    """Tests for the LearnedCapability dataclass."""

    def test_default_construction(self):
        """LearnedCapability with defaults should have expected values."""
        cap = LearnedCapability()
        self.assertIsNotNone(cap.id)
        self.assertEqual(len(cap.id), 8)
        self.assertEqual(cap.use_count, 0)
        self.assertAlmostEqual(cap.proficiency, 0.0)
        self.assertIsNone(cap.weights)

    def test_custom_construction(self):
        """LearnedCapability should accept custom parameters."""
        w = np.eye(4)
        cap = LearnedCapability(
            name="digit_classifier",
            proficiency=0.95,
            weights=w,
        )
        self.assertEqual(cap.name, "digit_classifier")
        self.assertAlmostEqual(cap.proficiency, 0.95)
        np.testing.assert_array_equal(cap.weights, np.eye(4))

    def test_use_increments_count(self):
        """use() should increment use_count."""
        cap = LearnedCapability()
        self.assertEqual(cap.use_count, 0)
        cap.use()
        self.assertEqual(cap.use_count, 1)
        cap.use()
        self.assertEqual(cap.use_count, 2)

    def test_use_updates_last_used(self):
        """use() should update last_used timestamp."""
        cap = LearnedCapability()
        old_time = cap.last_used
        # Ensure time advances slightly
        time.sleep(0.01)
        cap.use()
        self.assertGreaterEqual(cap.last_used, old_time)

    def test_update_proficiency(self):
        """update_proficiency should apply exponential moving average."""
        cap = LearnedCapability(proficiency=0.5)
        cap.update_proficiency(1.0, decay=0.1)
        # New proficiency = 0.9 * 0.5 + 0.1 * 1.0 = 0.55
        self.assertAlmostEqual(cap.proficiency, 0.55)

    def test_update_proficiency_repeated(self):
        """Repeated updates should approach the target performance."""
        cap = LearnedCapability(proficiency=0.0)
        for _ in range(100):
            cap.update_proficiency(1.0, decay=0.1)
        # After many updates approaching 1.0, proficiency should be close
        self.assertGreater(cap.proficiency, 0.9)

    def test_unique_ids(self):
        """Each LearnedCapability should get a unique id."""
        c1 = LearnedCapability()
        c2 = LearnedCapability()
        self.assertNotEqual(c1.id, c2.id)


# ===================================================================
# ProgressReport dataclass
# ===================================================================

class TestProgressReport(unittest.TestCase):
    """Tests for the ProgressReport dataclass."""

    def test_construction(self):
        """ProgressReport should store all provided fields."""
        pr = ProgressReport(
            challenge_id="c1",
            challenge_name="test",
            status=ChallengeStatus.LEARNING,
            progress_percent=0.5,
            current_accuracy=0.6,
            iterations_completed=25,
            time_elapsed_seconds=3.0,
            current_strategy="HEBBIAN",
            learning_curve=[0.1, 0.3, 0.5, 0.6],
        )
        self.assertEqual(pr.challenge_id, "c1")
        self.assertEqual(pr.status, ChallengeStatus.LEARNING)
        self.assertAlmostEqual(pr.progress_percent, 0.5)
        self.assertAlmostEqual(pr.current_accuracy, 0.6)
        self.assertEqual(pr.iterations_completed, 25)
        self.assertEqual(len(pr.learning_curve), 4)

    def test_str_representation(self):
        """str() on ProgressReport should include key information."""
        pr = ProgressReport(
            challenge_id="c1",
            challenge_name="my_challenge",
            status=ChallengeStatus.LEARNING,
            progress_percent=0.75,
            current_accuracy=0.85,
            iterations_completed=100,
            time_elapsed_seconds=10.0,
            current_strategy="BCM",
            learning_curve=[],
        )
        s = str(pr)
        self.assertIn("my_challenge", s)
        self.assertIn("LEARNING", s)

    def test_default_optional_fields(self):
        """Optional fields should have proper defaults."""
        pr = ProgressReport(
            challenge_id="c1",
            challenge_name="test",
            status=ChallengeStatus.PENDING,
            progress_percent=0.0,
            current_accuracy=0.0,
            iterations_completed=0,
            time_elapsed_seconds=0.0,
            current_strategy="none",
            learning_curve=[],
        )
        self.assertEqual(pr.strategy_switches, 0)
        self.assertIsNone(pr.estimated_completion)


# ===================================================================
# ChallengeLearner initialization
# ===================================================================

class TestChallengeLearnerInit(unittest.TestCase):
    """Tests for ChallengeLearner initialization."""

    @classmethod
    def setUpClass(cls):
        """Create a learner once for all init tests (suppressing verbose output)."""
        cls.learner = ChallengeLearner(
            state_dim=64,
            learning_rate=0.01,
            batch_size=16,
            verbose=False,
        )

    def test_state_dim_stored(self):
        """state_dim should be stored on the learner."""
        self.assertEqual(self.learner.state_dim, 64)

    def test_verbose_stored(self):
        """verbose flag should be stored on the learner."""
        self.assertFalse(self.learner.verbose)

    def test_parser_initialized(self):
        """parser should be a ChallengeParser instance."""
        from self_organizing_av_system.core.challenge_parser import ChallengeParser
        self.assertIsInstance(self.learner.parser, ChallengeParser)

    def test_progress_tracker_initialized(self):
        """progress_tracker should be a ProgressTracker instance."""
        from self_organizing_av_system.core.progress_tracker import ProgressTracker
        self.assertIsInstance(self.learner.progress_tracker, ProgressTracker)

    def test_meta_learner_initialized(self):
        """meta_learner should be a MetaLearner instance."""
        from self_organizing_av_system.core.meta_learning import MetaLearner
        self.assertIsInstance(self.learner.meta_learner, MetaLearner)

    def test_episodic_memory_initialized(self):
        """episodic_memory should be an EpisodicMemory instance."""
        from self_organizing_av_system.core.episodic_memory import EpisodicMemory
        self.assertIsInstance(self.learner.episodic_memory, EpisodicMemory)

    def test_learning_engine_initialized(self):
        """learning_engine should be a LearningEngine instance."""
        from self_organizing_av_system.core.learning_engine import LearningEngine
        self.assertIsInstance(self.learner.learning_engine, LearningEngine)

    def test_challenges_dict_empty(self):
        """challenges dict should start empty."""
        learner = ChallengeLearner(verbose=False)
        self.assertEqual(len(learner.challenges), 0)

    def test_results_dict_empty(self):
        """results dict should start empty."""
        learner = ChallengeLearner(verbose=False)
        self.assertEqual(len(learner.results), 0)

    def test_default_state_dim(self):
        """Default state_dim should be 128."""
        learner = ChallengeLearner(verbose=False)
        self.assertEqual(learner.state_dim, 128)


# ===================================================================
# ChallengeLearner.learn()
# ===================================================================

class TestChallengeLearnerLearn(unittest.TestCase):
    """Tests for ChallengeLearner.learn() method."""

    def setUp(self):
        """Create a fresh learner with mocked learning engine."""
        self.learner = ChallengeLearner(
            state_dim=64,
            verbose=False,
        )

    def test_learn_from_string(self):
        """learn() with a string should parse and execute learning."""
        mock_result = _make_mock_result()
        with patch.object(
            self.learner.learning_engine,
            'execute_learning_loop',
            return_value=mock_result,
        ):
            result = self.learner.learn("Learn to classify handwritten digits")

        self.assertIsInstance(result, LearningResult)
        self.assertTrue(result.success)
        self.assertAlmostEqual(result.accuracy, 0.95)

    def test_learn_from_string_stores_challenge(self):
        """learn() should store the parsed challenge in the challenges dict."""
        mock_result = _make_mock_result()
        with patch.object(
            self.learner.learning_engine,
            'execute_learning_loop',
            return_value=mock_result,
        ):
            self.learner.learn("Learn to detect anomalies")

        self.assertEqual(len(self.learner.challenges), 1)

    def test_learn_from_challenge_object(self):
        """learn() with a Challenge object should use it directly."""
        challenge = Challenge(
            name="test_direct",
            description="Direct challenge",
            challenge_type=ChallengeType.ASSOCIATION,
        )
        mock_result = _make_mock_result(
            challenge_id=challenge.id,
            challenge_name=challenge.name,
        )
        with patch.object(
            self.learner.learning_engine,
            'execute_learning_loop',
            return_value=mock_result,
        ):
            result = self.learner.learn(challenge)

        self.assertIsInstance(result, LearningResult)
        self.assertIn(challenge.id, self.learner.challenges)

    def test_learn_from_dict(self):
        """learn() with a dict should parse it as a structured challenge."""
        challenge_dict = {
            "name": "dict_challenge",
            "description": "From a dictionary",
            "challenge_type": "PREDICTION",
            "difficulty": 0.6,
        }
        mock_result = _make_mock_result()
        with patch.object(
            self.learner.learning_engine,
            'execute_learning_loop',
            return_value=mock_result,
        ):
            result = self.learner.learn(challenge_dict)

        self.assertIsInstance(result, LearningResult)
        self.assertEqual(len(self.learner.challenges), 1)

    def test_learn_stores_result(self):
        """learn() should store the result in the results dict."""
        mock_result = _make_mock_result()
        with patch.object(
            self.learner.learning_engine,
            'execute_learning_loop',
            return_value=mock_result,
        ):
            self.learner.learn("Learn something")

        self.assertEqual(len(self.learner.results), 1)

    def test_learn_invalid_input_raises(self):
        """learn() with an unsupported type should raise ValueError."""
        with self.assertRaises(ValueError):
            self.learner.learn(12345)

    def test_learn_with_callback(self):
        """learn() should pass a callback to the learning engine."""
        mock_result = _make_mock_result()
        callback_calls = []

        def my_callback(iteration, accuracy):
            callback_calls.append((iteration, accuracy))

        with patch.object(
            self.learner.learning_engine,
            'execute_learning_loop',
            return_value=mock_result,
        ) as mock_exec:
            self.learner.learn("test", callback=my_callback)
            # Verify callback was passed through
            call_kwargs = mock_exec.call_args
            self.assertEqual(call_kwargs.kwargs.get('callback'), my_callback)


# ===================================================================
# ChallengeLearner.learn_from_description()
# ===================================================================

class TestChallengeLearnerLearnFromDescription(unittest.TestCase):
    """Tests for ChallengeLearner.learn_from_description() method."""

    def setUp(self):
        """Create a fresh learner with mocked learning engine."""
        self.learner = ChallengeLearner(
            state_dim=64,
            verbose=False,
        )

    def test_learn_from_description_returns_result(self):
        """learn_from_description should return a LearningResult."""
        mock_result = _make_mock_result()
        with patch.object(
            self.learner.learning_engine,
            'execute_learning_loop',
            return_value=mock_result,
        ):
            result = self.learner.learn_from_description(
                "Learn to recognize faces in images"
            )

        self.assertIsInstance(result, LearningResult)

    def test_learn_from_description_delegates_to_learn(self):
        """learn_from_description should delegate to learn()."""
        mock_result = _make_mock_result()
        with patch.object(
            self.learner, 'learn', return_value=mock_result,
        ) as mock_learn:
            self.learner.learn_from_description("test description")
            mock_learn.assert_called_once()
            args, kwargs = mock_learn.call_args
            self.assertEqual(args[0], "test description")

    def test_learn_from_description_with_callback(self):
        """learn_from_description should pass callback through."""
        mock_result = _make_mock_result()
        callback = lambda i, a: None

        with patch.object(
            self.learner, 'learn', return_value=mock_result,
        ) as mock_learn:
            self.learner.learn_from_description("test", callback=callback)
            _, kwargs = mock_learn.call_args
            self.assertEqual(kwargs.get('callback'), callback)


# ===================================================================
# ChallengeLearner.learn_from_data()
# ===================================================================

class TestChallengeLearnerLearnFromData(unittest.TestCase):
    """Tests for ChallengeLearner.learn_from_data() method."""

    def setUp(self):
        """Create a fresh learner with mocked learning engine."""
        self.learner = ChallengeLearner(
            state_dim=64,
            verbose=False,
        )

    def test_learn_from_data_with_labels(self):
        """learn_from_data should handle data with labels."""
        np.random.seed(42)
        data = np.random.randn(20, 10)
        labels = np.random.randint(0, 3, size=20)
        mock_result = _make_mock_result()

        with patch.object(
            self.learner.learning_engine,
            'execute_learning_loop',
            return_value=mock_result,
        ):
            result = self.learner.learn_from_data(data=data, labels=labels)

        self.assertIsInstance(result, LearningResult)

    def test_learn_from_data_without_labels(self):
        """learn_from_data without labels should use CONCEPT_FORMATION type."""
        np.random.seed(42)
        data = np.random.randn(20, 10)
        mock_result = _make_mock_result()

        with patch.object(
            self.learner.learning_engine,
            'execute_learning_loop',
            return_value=mock_result,
        ) as mock_exec:
            result = self.learner.learn_from_data(data=data)

        self.assertIsInstance(result, LearningResult)
        # The challenge passed to execute_learning_loop should be CONCEPT_FORMATION
        challenge_arg = mock_exec.call_args[0][0]
        self.assertEqual(
            challenge_arg.challenge_type,
            ChallengeType.CONCEPT_FORMATION,
        )

    def test_learn_from_data_with_success_criteria(self):
        """learn_from_data should use provided success criteria."""
        np.random.seed(42)
        data = np.random.randn(20, 10)
        labels = np.random.randint(0, 3, size=20)
        mock_result = _make_mock_result()

        with patch.object(
            self.learner.learning_engine,
            'execute_learning_loop',
            return_value=mock_result,
        ) as mock_exec:
            result = self.learner.learn_from_data(
                data=data,
                labels=labels,
                success_criteria={"accuracy": 0.95},
            )

        challenge_arg = mock_exec.call_args[0][0]
        self.assertAlmostEqual(challenge_arg.success_criteria.accuracy, 0.95)

    def test_learn_from_data_with_custom_name(self):
        """learn_from_data should use provided name."""
        np.random.seed(42)
        data = np.random.randn(10, 5)
        mock_result = _make_mock_result()

        with patch.object(
            self.learner.learning_engine,
            'execute_learning_loop',
            return_value=mock_result,
        ) as mock_exec:
            self.learner.learn_from_data(data=data, name="my_task")

        challenge_arg = mock_exec.call_args[0][0]
        self.assertEqual(challenge_arg.name, "my_task")

    def test_learn_from_data_with_explicit_modality(self):
        """learn_from_data should use the provided modality."""
        np.random.seed(42)
        data = np.random.randn(10, 5)
        mock_result = _make_mock_result()

        with patch.object(
            self.learner.learning_engine,
            'execute_learning_loop',
            return_value=mock_result,
        ) as mock_exec:
            self.learner.learn_from_data(
                data=data,
                modality=Modality.VISION,
            )

        challenge_arg = mock_exec.call_args[0][0]
        self.assertIn(Modality.VISION, challenge_arg.modalities)

    def test_learn_from_data_creates_training_data(self):
        """learn_from_data should create TrainingData in the challenge."""
        np.random.seed(42)
        data = np.random.randn(15, 8)
        labels = np.random.randint(0, 2, size=15)
        mock_result = _make_mock_result()

        with patch.object(
            self.learner.learning_engine,
            'execute_learning_loop',
            return_value=mock_result,
        ) as mock_exec:
            self.learner.learn_from_data(data=data, labels=labels)

        challenge_arg = mock_exec.call_args[0][0]
        self.assertIsNotNone(challenge_arg.training_data)
        self.assertEqual(len(challenge_arg.training_data), 15)

    def test_learn_from_data_list_input(self):
        """learn_from_data should accept plain lists as data."""
        data = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        labels = [0, 1, 0]
        mock_result = _make_mock_result()

        with patch.object(
            self.learner.learning_engine,
            'execute_learning_loop',
            return_value=mock_result,
        ):
            result = self.learner.learn_from_data(data=data, labels=labels)

        self.assertIsInstance(result, LearningResult)


# ===================================================================
# ChallengeLearner.get_progress()
# ===================================================================

class TestChallengeLearnerGetProgress(unittest.TestCase):
    """Tests for ChallengeLearner.get_progress() method."""

    def setUp(self):
        """Create a fresh learner."""
        self.learner = ChallengeLearner(
            state_dim=64,
            verbose=False,
        )

    def test_get_progress_unknown_challenge_raises(self):
        """get_progress with unknown challenge_id should raise ValueError."""
        with self.assertRaises(ValueError):
            self.learner.get_progress("nonexistent-id")

    def test_get_progress_after_learn(self):
        """get_progress should work for a challenge that has been tracked."""
        mock_result = _make_mock_result()

        def mock_execute(challenge, **kwargs):
            # Simulate what execute_learning_loop does for the tracker
            self.learner.progress_tracker.start_challenge(challenge)
            self.learner.progress_tracker.update_progress(
                challenge.id, accuracy=0.5, strategy="HEBBIAN",
            )
            return mock_result

        with patch.object(
            self.learner.learning_engine,
            'execute_learning_loop',
            side_effect=mock_execute,
        ):
            result = self.learner.learn("Learn patterns")

        # Get the actual challenge_id that was stored
        challenge_id = list(self.learner.challenges.keys())[0]
        progress = self.learner.get_progress(challenge_id)
        self.assertIsInstance(progress, ProgressReport)
        self.assertEqual(progress.challenge_id, challenge_id)

    def test_get_progress_returns_correct_type(self):
        """get_progress should return a ProgressReport instance."""
        # Manually register a challenge and start tracking it
        challenge = Challenge(name="manual_test")
        self.learner.challenges[challenge.id] = challenge
        self.learner.progress_tracker.start_challenge(challenge)

        progress = self.learner.get_progress(challenge.id)
        self.assertIsInstance(progress, ProgressReport)
        self.assertEqual(progress.challenge_name, "manual_test")


# ===================================================================
# ChallengeLearner.get_capabilities()
# ===================================================================

class TestChallengeLearnerGetCapabilities(unittest.TestCase):
    """Tests for ChallengeLearner.get_capabilities() method."""

    def setUp(self):
        """Create a fresh learner."""
        self.learner = ChallengeLearner(
            state_dim=64,
            verbose=False,
        )

    def test_get_capabilities_initially_empty(self):
        """get_capabilities should return an empty list initially."""
        caps = self.learner.get_capabilities()
        self.assertIsInstance(caps, list)
        self.assertEqual(len(caps), 0)

    def test_get_capabilities_after_adding(self):
        """get_capabilities should return capabilities registered on the engine."""
        # Directly add a capability to the engine for testing
        cap = LearnedCapability(
            name="test_cap",
            proficiency=0.9,
            weights=np.eye(4),
        )
        self.learner.learning_engine.capabilities[cap.id] = cap

        caps = self.learner.get_capabilities()
        self.assertEqual(len(caps), 1)
        self.assertEqual(caps[0].name, "test_cap")

    def test_get_capabilities_returns_list_of_correct_type(self):
        """Each element from get_capabilities should be a LearnedCapability."""
        cap = LearnedCapability(name="cap1")
        self.learner.learning_engine.capabilities[cap.id] = cap

        caps = self.learner.get_capabilities()
        for c in caps:
            self.assertIsInstance(c, LearnedCapability)


# ===================================================================
# ChallengeLearner.apply_capability()
# ===================================================================

class TestChallengeLearnerApplyCapability(unittest.TestCase):
    """Tests for ChallengeLearner.apply_capability() method."""

    def setUp(self):
        """Create a learner with a pre-registered capability."""
        self.learner = ChallengeLearner(
            state_dim=64,
            verbose=False,
        )
        # Register a capability with known weights
        self.weights = np.eye(4, dtype=np.float64)
        self.cap = LearnedCapability(
            name="test_apply",
            proficiency=0.9,
            weights=self.weights,
        )
        self.learner.learning_engine.capabilities[self.cap.id] = self.cap

    def test_apply_capability_returns_array(self):
        """apply_capability should return a numpy array."""
        data = np.array([[1.0, 0.0, 0.0, 0.0]])
        result = self.learner.apply_capability(self.cap.id, data)
        self.assertIsInstance(result, np.ndarray)

    def test_apply_capability_identity_weights(self):
        """With identity weights, output should match input."""
        data = np.array([[1.0, 2.0, 3.0, 4.0]])
        result = self.learner.apply_capability(self.cap.id, data)
        np.testing.assert_array_almost_equal(result, data)

    def test_apply_capability_1d_input(self):
        """apply_capability should handle 1D input by reshaping."""
        data = np.array([1.0, 0.0, 0.0, 0.0])
        result = self.learner.apply_capability(self.cap.id, data)
        self.assertEqual(result.shape, (1, 4))

    def test_apply_capability_updates_usage(self):
        """apply_capability should increment the capability's use_count."""
        data = np.array([[1.0, 0.0, 0.0, 0.0]])
        self.assertEqual(self.cap.use_count, 0)
        self.learner.apply_capability(self.cap.id, data)
        self.assertEqual(self.cap.use_count, 1)

    def test_apply_capability_unknown_id_raises(self):
        """apply_capability with unknown capability_id should raise ValueError."""
        data = np.array([[1.0, 0.0, 0.0, 0.0]])
        with self.assertRaises(ValueError):
            self.learner.apply_capability("nonexistent", data)

    def test_apply_capability_no_weights_raises(self):
        """apply_capability on a capability with no weights should raise ValueError."""
        cap_no_weights = LearnedCapability(name="no_weights")
        self.learner.learning_engine.capabilities[cap_no_weights.id] = cap_no_weights

        data = np.array([[1.0, 0.0]])
        with self.assertRaises(ValueError):
            self.learner.apply_capability(cap_no_weights.id, data)

    def test_apply_capability_batch_input(self):
        """apply_capability should handle batched 2D input."""
        data = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ])
        result = self.learner.apply_capability(self.cap.id, data)
        self.assertEqual(result.shape, (3, 4))
        np.testing.assert_array_almost_equal(result, data)


# ===================================================================
# ChallengeLearner.get_capability()
# ===================================================================

class TestChallengeLearnerGetCapability(unittest.TestCase):
    """Tests for ChallengeLearner.get_capability() method."""

    def setUp(self):
        """Create a learner with a registered capability."""
        self.learner = ChallengeLearner(state_dim=64, verbose=False)
        self.cap = LearnedCapability(name="findme")
        self.learner.learning_engine.capabilities[self.cap.id] = self.cap

    def test_get_existing_capability(self):
        """get_capability should return the correct capability."""
        result = self.learner.get_capability(self.cap.id)
        self.assertIsNotNone(result)
        self.assertEqual(result.name, "findme")

    def test_get_nonexistent_capability(self):
        """get_capability should return None for unknown id."""
        result = self.learner.get_capability("does-not-exist")
        self.assertIsNone(result)


# ===================================================================
# ChallengeLearner.get_stats()
# ===================================================================

class TestChallengeLearnerGetStats(unittest.TestCase):
    """Tests for ChallengeLearner.get_stats() method."""

    def test_get_stats_returns_dict(self):
        """get_stats should return a dictionary."""
        learner = ChallengeLearner(state_dim=64, verbose=False)
        stats = learner.get_stats()
        self.assertIsInstance(stats, dict)

    def test_get_stats_has_expected_keys(self):
        """get_stats should include expected keys."""
        learner = ChallengeLearner(state_dim=64, verbose=False)
        stats = learner.get_stats()
        self.assertIn("total_challenges_attempted", stats)
        self.assertIn("capabilities_learned", stats)
        self.assertIn("meta_learning_selections", stats)

    def test_get_stats_initial_values(self):
        """Initial stats should show zero challenges and capabilities."""
        learner = ChallengeLearner(state_dim=64, verbose=False)
        stats = learner.get_stats()
        self.assertEqual(stats["total_challenges_attempted"], 0)
        self.assertEqual(stats["capabilities_learned"], 0)


# ===================================================================
# learn_challenge convenience function
# ===================================================================

class TestLearnChallengeFunction(unittest.TestCase):
    """Tests for the module-level learn_challenge convenience function."""

    def test_learn_challenge_returns_learning_result(self):
        """learn_challenge should return a LearningResult."""
        mock_result = _make_mock_result()
        with patch.object(
            ChallengeLearner,
            'learn',
            return_value=mock_result,
        ):
            result = learn_challenge("test challenge", verbose=False)

        self.assertIsInstance(result, LearningResult)


if __name__ == "__main__":
    unittest.main()
