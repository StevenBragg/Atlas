"""
Comprehensive tests for the ProgressTracker and ChallengeMetrics modules.

Tests cover:
- ChallengeMetrics creation and properties
- ChallengeMetrics learning rate and plateau detection
- ProgressTracker initialization and defaults
- Metric recording and tracking (accuracy, convergence, efficiency)
- Progress reporting
- Challenge completion and curriculum adjustment
- Serialization / get_state via get_stats()

All tests are deterministic and pass reliably in both CPU-only and GPU environments.
"""

import os
import sys
import time
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from self_organizing_av_system.core.backend import xp
from self_organizing_av_system.core.progress_tracker import (
    ProgressTracker,
    ChallengeMetrics,
)
from self_organizing_av_system.core.challenge import (
    Challenge,
    ChallengeStatus,
    ChallengeType,
    Modality,
    ProgressReport,
    SuccessCriteria,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_challenge(
    name="test_challenge",
    accuracy=0.8,
    min_samples=10,
    max_iterations=1000,
    convergence_threshold=0.01,
    convergence_window=10,
    time_limit_seconds=None,
    custom_metrics=None,
    difficulty=0.5,
):
    """Create a Challenge with configurable success criteria."""
    criteria = SuccessCriteria(
        accuracy=accuracy,
        min_samples=min_samples,
        max_iterations=max_iterations,
        convergence_threshold=convergence_threshold,
        convergence_window=convergence_window,
        time_limit_seconds=time_limit_seconds,
        custom_metrics=custom_metrics or {},
    )
    return Challenge(
        name=name,
        description=f"Test challenge: {name}",
        challenge_type=ChallengeType.PATTERN_RECOGNITION,
        modalities=[Modality.EMBEDDING],
        success_criteria=criteria,
        difficulty=difficulty,
    )


# ---------------------------------------------------------------------------
# ChallengeMetrics tests
# ---------------------------------------------------------------------------

class TestChallengeMetricsCreation(unittest.TestCase):
    """Tests for ChallengeMetrics dataclass creation."""

    def test_basic_creation(self):
        """ChallengeMetrics can be created with just a challenge_id."""
        metrics = ChallengeMetrics(challenge_id="abc123")
        self.assertEqual(metrics.challenge_id, "abc123")

    def test_default_lists_are_empty(self):
        """All history lists should default to empty."""
        metrics = ChallengeMetrics(challenge_id="test")
        self.assertEqual(metrics.accuracy_history, [])
        self.assertEqual(metrics.loss_history, [])
        self.assertEqual(metrics.iteration_times, [])
        self.assertEqual(metrics.strategy_history, [])
        self.assertEqual(metrics.custom_metrics, {})

    def test_default_lists_are_independent(self):
        """Two instances must not share mutable defaults."""
        m1 = ChallengeMetrics(challenge_id="a")
        m2 = ChallengeMetrics(challenge_id="b")
        m1.accuracy_history.append(0.5)
        self.assertEqual(len(m2.accuracy_history), 0)

    def test_timestamps_are_set(self):
        """start_time and last_update should be set automatically."""
        before = time.time()
        metrics = ChallengeMetrics(challenge_id="ts")
        after = time.time()
        self.assertGreaterEqual(metrics.start_time, before)
        self.assertLessEqual(metrics.start_time, after)
        self.assertGreaterEqual(metrics.last_update, before)
        self.assertLessEqual(metrics.last_update, after)


class TestChallengeMetricsProperties(unittest.TestCase):
    """Tests for ChallengeMetrics computed properties."""

    def test_current_accuracy_empty(self):
        """current_accuracy should return 0.0 when no data recorded."""
        metrics = ChallengeMetrics(challenge_id="empty")
        self.assertEqual(metrics.current_accuracy, 0.0)

    def test_current_accuracy_with_data(self):
        """current_accuracy should return the last recorded value."""
        metrics = ChallengeMetrics(challenge_id="data")
        metrics.accuracy_history = [0.1, 0.3, 0.7, 0.85]
        self.assertAlmostEqual(metrics.current_accuracy, 0.85)

    def test_best_accuracy_empty(self):
        """best_accuracy should return 0.0 when no data recorded."""
        metrics = ChallengeMetrics(challenge_id="empty")
        self.assertEqual(metrics.best_accuracy, 0.0)

    def test_best_accuracy_with_data(self):
        """best_accuracy should return the maximum recorded value."""
        metrics = ChallengeMetrics(challenge_id="data")
        metrics.accuracy_history = [0.1, 0.9, 0.5, 0.7]
        self.assertAlmostEqual(metrics.best_accuracy, 0.9)

    def test_iterations_empty(self):
        """iterations should return 0 when no data recorded."""
        metrics = ChallengeMetrics(challenge_id="empty")
        self.assertEqual(metrics.iterations, 0)

    def test_iterations_with_data(self):
        """iterations should return len(accuracy_history)."""
        metrics = ChallengeMetrics(challenge_id="data")
        metrics.accuracy_history = [0.1, 0.2, 0.3]
        self.assertEqual(metrics.iterations, 3)

    def test_elapsed_time_is_positive(self):
        """elapsed_time should be a non-negative float."""
        metrics = ChallengeMetrics(challenge_id="time")
        self.assertGreaterEqual(metrics.elapsed_time, 0.0)


class TestChallengeMetricsLearningRate(unittest.TestCase):
    """Tests for ChallengeMetrics.get_learning_rate()."""

    def test_learning_rate_empty(self):
        """Learning rate with no data should be 0.0."""
        metrics = ChallengeMetrics(challenge_id="lr")
        self.assertEqual(metrics.get_learning_rate(), 0.0)

    def test_learning_rate_single_entry(self):
        """Learning rate with one entry should be 0.0."""
        metrics = ChallengeMetrics(challenge_id="lr")
        metrics.accuracy_history = [0.5]
        self.assertEqual(metrics.get_learning_rate(), 0.0)

    def test_learning_rate_positive_improvement(self):
        """Positive learning rate when accuracy is increasing."""
        metrics = ChallengeMetrics(challenge_id="lr")
        # Linearly increasing: 0.0, 0.1, 0.2, 0.3, 0.4
        metrics.accuracy_history = [0.0, 0.1, 0.2, 0.3, 0.4]
        rate = metrics.get_learning_rate(window=5)
        # rate = (0.4 - 0.0) / 5 = 0.08
        self.assertAlmostEqual(rate, 0.08)

    def test_learning_rate_negative(self):
        """Negative learning rate when accuracy is decreasing."""
        metrics = ChallengeMetrics(challenge_id="lr")
        metrics.accuracy_history = [0.9, 0.7, 0.5]
        rate = metrics.get_learning_rate(window=10)
        # window covers all 3: rate = (0.5 - 0.9) / 3 = -0.1333...
        self.assertAlmostEqual(rate, -0.4 / 3)

    def test_learning_rate_window_limits(self):
        """Learning rate should only consider the last `window` entries."""
        metrics = ChallengeMetrics(challenge_id="lr")
        # First 5 are low, last 3 jump up
        metrics.accuracy_history = [0.1, 0.1, 0.1, 0.1, 0.1, 0.4, 0.7, 1.0]
        rate = metrics.get_learning_rate(window=3)
        # last 3: [0.4, 0.7, 1.0]; rate = (1.0 - 0.4) / 3 = 0.2
        self.assertAlmostEqual(rate, 0.2)


class TestChallengeMetricsPlateau(unittest.TestCase):
    """Tests for ChallengeMetrics.is_plateaued()."""

    def test_not_plateaued_insufficient_data(self):
        """Should not be considered plateaued with fewer entries than window."""
        metrics = ChallengeMetrics(challenge_id="p")
        metrics.accuracy_history = [0.5, 0.5, 0.5]
        self.assertFalse(metrics.is_plateaued(window=10))

    def test_plateaued_flat(self):
        """Should detect plateau when all recent values are the same."""
        metrics = ChallengeMetrics(challenge_id="p")
        metrics.accuracy_history = [0.5] * 20
        self.assertTrue(metrics.is_plateaued(window=10, threshold=0.01))

    def test_not_plateaued_improving(self):
        """Should not be considered plateaued when actively improving."""
        metrics = ChallengeMetrics(challenge_id="p")
        metrics.accuracy_history = [i * 0.05 for i in range(20)]
        self.assertFalse(metrics.is_plateaued(window=10, threshold=0.01))

    def test_plateaued_with_tiny_variance(self):
        """Should detect plateau with variance below threshold."""
        metrics = ChallengeMetrics(challenge_id="p")
        # Values within 0.005 of each other (below 0.01 threshold)
        metrics.accuracy_history = [0.50, 0.502, 0.501, 0.503, 0.505,
                                    0.504, 0.501, 0.503, 0.502, 0.504]
        self.assertTrue(metrics.is_plateaued(window=10, threshold=0.01))

    def test_not_plateaued_with_large_threshold(self):
        """Larger threshold makes it harder to trigger plateau."""
        metrics = ChallengeMetrics(challenge_id="p")
        # Oscillating values with range 0.1
        metrics.accuracy_history = [0.5, 0.55, 0.5, 0.55, 0.5,
                                    0.55, 0.5, 0.55, 0.5, 0.55]
        self.assertTrue(metrics.is_plateaued(window=10, threshold=0.1))
        self.assertFalse(metrics.is_plateaued(window=10, threshold=0.01))


# ---------------------------------------------------------------------------
# ProgressTracker initialization tests
# ---------------------------------------------------------------------------

class TestProgressTrackerInit(unittest.TestCase):
    """Tests for ProgressTracker initialization."""

    def test_default_parameters(self):
        """Defaults should match documented values."""
        tracker = ProgressTracker()
        self.assertAlmostEqual(tracker.mastery_threshold, 0.8)
        self.assertEqual(tracker.plateau_window, 50)
        self.assertAlmostEqual(tracker.plateau_threshold, 0.02)
        self.assertAlmostEqual(tracker.curriculum_adjustment_rate, 0.1)

    def test_custom_parameters(self):
        """Custom parameters should override defaults."""
        tracker = ProgressTracker(
            mastery_threshold=0.95,
            plateau_window=30,
            plateau_threshold=0.05,
            curriculum_adjustment_rate=0.2,
        )
        self.assertAlmostEqual(tracker.mastery_threshold, 0.95)
        self.assertEqual(tracker.plateau_window, 30)
        self.assertAlmostEqual(tracker.plateau_threshold, 0.05)
        self.assertAlmostEqual(tracker.curriculum_adjustment_rate, 0.2)

    def test_initial_state_empty(self):
        """Initially, all tracking structures should be empty."""
        tracker = ProgressTracker()
        self.assertEqual(len(tracker.challenges), 0)
        self.assertEqual(len(tracker.completed_challenges), 0)
        self.assertEqual(len(tracker.failed_challenges), 0)
        self.assertEqual(len(tracker.performance_history), 0)

    def test_initial_curriculum_difficulty(self):
        """Starting difficulty should be 0.3 (easy)."""
        tracker = ProgressTracker()
        self.assertAlmostEqual(tracker.curriculum_difficulty, 0.3)


# ---------------------------------------------------------------------------
# ProgressTracker.start_challenge tests
# ---------------------------------------------------------------------------

class TestStartChallenge(unittest.TestCase):
    """Tests for ProgressTracker.start_challenge()."""

    def test_start_challenge_returns_id(self):
        """start_challenge should return the challenge ID."""
        tracker = ProgressTracker()
        challenge = _make_challenge(name="first")
        cid = tracker.start_challenge(challenge)
        self.assertEqual(cid, challenge.id)

    def test_start_challenge_registers_metrics(self):
        """After starting, the challenge should appear in tracker.challenges."""
        tracker = ProgressTracker()
        challenge = _make_challenge()
        cid = tracker.start_challenge(challenge)
        self.assertIn(cid, tracker.challenges)
        self.assertIsInstance(tracker.challenges[cid], ChallengeMetrics)

    def test_start_multiple_challenges(self):
        """Tracker should support multiple concurrent challenges."""
        tracker = ProgressTracker()
        c1 = _make_challenge(name="c1")
        c2 = _make_challenge(name="c2")
        id1 = tracker.start_challenge(c1)
        id2 = tracker.start_challenge(c2)
        self.assertIn(id1, tracker.challenges)
        self.assertIn(id2, tracker.challenges)
        self.assertNotEqual(id1, id2)


# ---------------------------------------------------------------------------
# ProgressTracker.update_progress (metric recording) tests
# ---------------------------------------------------------------------------

class TestUpdateProgress(unittest.TestCase):
    """Tests for metric recording via update_progress()."""

    def setUp(self):
        self.tracker = ProgressTracker()
        self.challenge = _make_challenge()
        self.cid = self.tracker.start_challenge(self.challenge)

    def test_basic_update(self):
        """update_progress should record accuracy and strategy."""
        result = self.tracker.update_progress(self.cid, accuracy=0.5, strategy="hebbian")
        self.assertEqual(result['accuracy'], 0.5)
        self.assertEqual(result['iterations'], 1)

    def test_accuracy_tracking(self):
        """Accuracy values should accumulate in accuracy_history."""
        accuracies = [0.1, 0.3, 0.5, 0.7, 0.9]
        for acc in accuracies:
            self.tracker.update_progress(self.cid, accuracy=acc, strategy="hebbian")

        metrics = self.tracker.challenges[self.cid]
        self.assertEqual(metrics.accuracy_history, accuracies)

    def test_best_accuracy_tracking(self):
        """update_progress should report the best accuracy so far."""
        self.tracker.update_progress(self.cid, accuracy=0.5, strategy="hebbian")
        self.tracker.update_progress(self.cid, accuracy=0.9, strategy="hebbian")
        result = self.tracker.update_progress(self.cid, accuracy=0.7, strategy="hebbian")
        self.assertAlmostEqual(result['best_accuracy'], 0.9)

    def test_loss_tracking(self):
        """Loss values should be recorded when provided."""
        self.tracker.update_progress(self.cid, accuracy=0.5, strategy="hebbian", loss=0.8)
        self.tracker.update_progress(self.cid, accuracy=0.6, strategy="hebbian", loss=0.6)
        metrics = self.tracker.challenges[self.cid]
        self.assertEqual(metrics.loss_history, [0.8, 0.6])

    def test_loss_not_recorded_when_none(self):
        """Loss history should not grow when loss is None."""
        self.tracker.update_progress(self.cid, accuracy=0.5, strategy="hebbian")
        metrics = self.tracker.challenges[self.cid]
        self.assertEqual(len(metrics.loss_history), 0)

    def test_custom_metrics_tracking(self):
        """Custom metrics should be recorded in the correct buckets."""
        self.tracker.update_progress(
            self.cid, accuracy=0.5, strategy="hebbian",
            custom_metrics={"f1_score": 0.6, "recall": 0.7},
        )
        self.tracker.update_progress(
            self.cid, accuracy=0.6, strategy="hebbian",
            custom_metrics={"f1_score": 0.8, "recall": 0.9},
        )
        metrics = self.tracker.challenges[self.cid]
        self.assertEqual(metrics.custom_metrics["f1_score"], [0.6, 0.8])
        self.assertEqual(metrics.custom_metrics["recall"], [0.7, 0.9])

    def test_strategy_tracking(self):
        """Strategy names should be recorded in strategy_history."""
        self.tracker.update_progress(self.cid, accuracy=0.3, strategy="hebbian")
        self.tracker.update_progress(self.cid, accuracy=0.5, strategy="stdp")
        metrics = self.tracker.challenges[self.cid]
        self.assertEqual(metrics.strategy_history, ["hebbian", "stdp"])

    def test_iteration_count(self):
        """Iteration count should increment with each update."""
        for i in range(5):
            result = self.tracker.update_progress(
                self.cid, accuracy=0.1 * (i + 1), strategy="hebbian"
            )
        self.assertEqual(result['iterations'], 5)

    def test_unknown_challenge_raises(self):
        """Updating an unknown challenge_id should raise ValueError."""
        with self.assertRaises(ValueError):
            self.tracker.update_progress("nonexistent", accuracy=0.5, strategy="x")

    def test_elapsed_time_is_reported(self):
        """elapsed_time should appear in the result dict."""
        result = self.tracker.update_progress(self.cid, accuracy=0.5, strategy="hebbian")
        self.assertIn('elapsed_time', result)
        self.assertGreaterEqual(result['elapsed_time'], 0.0)

    def test_plateau_detection_no_plateau(self):
        """is_plateau should be False when there is insufficient data."""
        result = self.tracker.update_progress(self.cid, accuracy=0.5, strategy="hebbian")
        self.assertFalse(result['is_plateau'])

    def test_should_adapt_flag(self):
        """should_adapt should be True when plateaued below mastery."""
        tracker = ProgressTracker(
            mastery_threshold=0.9,
            plateau_window=5,
            plateau_threshold=0.01,
        )
        challenge = _make_challenge()
        cid = tracker.start_challenge(challenge)
        # Record identical values to force plateau
        for _ in range(10):
            result = tracker.update_progress(cid, accuracy=0.5, strategy="hebbian")
        self.assertTrue(result['is_plateau'])
        self.assertTrue(result['should_adapt'])

    def test_should_adapt_false_when_mastered(self):
        """should_adapt should be False when accuracy exceeds mastery."""
        tracker = ProgressTracker(
            mastery_threshold=0.8,
            plateau_window=5,
            plateau_threshold=0.01,
        )
        challenge = _make_challenge()
        cid = tracker.start_challenge(challenge)
        for _ in range(10):
            result = tracker.update_progress(cid, accuracy=0.95, strategy="hebbian")
        # Plateaued but above mastery -> should_adapt = False
        self.assertTrue(result['is_plateau'])
        self.assertFalse(result['should_adapt'])


# ---------------------------------------------------------------------------
# ProgressTracker.check_completion tests
# ---------------------------------------------------------------------------

class TestCheckCompletion(unittest.TestCase):
    """Tests for challenge completion checking."""

    def setUp(self):
        self.tracker = ProgressTracker()

    def test_unknown_challenge(self):
        """Unknown challenge_id should return (False, 'Unknown challenge')."""
        criteria = SuccessCriteria()
        is_complete, reason = self.tracker.check_completion("nope", criteria)
        self.assertFalse(is_complete)
        self.assertIn("Unknown", reason)

    def test_not_enough_iterations(self):
        """Should not complete before min_samples iterations."""
        challenge = _make_challenge(min_samples=10)
        cid = self.tracker.start_challenge(challenge)
        for i in range(5):
            self.tracker.update_progress(cid, accuracy=0.9, strategy="hebbian")
        is_complete, reason = self.tracker.check_completion(
            cid, challenge.success_criteria
        )
        self.assertFalse(is_complete)
        self.assertIn("more iterations", reason)

    def test_max_iterations_with_accuracy(self):
        """Should complete at max_iterations if accuracy is sufficient."""
        challenge = _make_challenge(max_iterations=20, accuracy=0.7, min_samples=5)
        cid = self.tracker.start_challenge(challenge)
        for _ in range(20):
            self.tracker.update_progress(cid, accuracy=0.8, strategy="hebbian")
        is_complete, reason = self.tracker.check_completion(
            cid, challenge.success_criteria
        )
        self.assertTrue(is_complete)
        self.assertIn("max iterations", reason.lower())

    def test_max_iterations_timeout(self):
        """Should complete (timeout) at max_iterations even with low accuracy."""
        challenge = _make_challenge(max_iterations=10, accuracy=0.9, min_samples=5)
        cid = self.tracker.start_challenge(challenge)
        for _ in range(10):
            self.tracker.update_progress(cid, accuracy=0.3, strategy="hebbian")
        is_complete, reason = self.tracker.check_completion(
            cid, challenge.success_criteria
        )
        self.assertTrue(is_complete)
        self.assertIn("timeout", reason.lower())

    def test_convergence_at_target_accuracy(self):
        """Should detect convergence when accuracy is stable at target."""
        challenge = _make_challenge(
            accuracy=0.8,
            min_samples=5,
            convergence_window=5,
            convergence_threshold=0.01,
            max_iterations=1000,
        )
        cid = self.tracker.start_challenge(challenge)
        # Feed stable high accuracy for enough iterations
        for _ in range(15):
            self.tracker.update_progress(cid, accuracy=0.85, strategy="hebbian")
        is_complete, reason = self.tracker.check_completion(
            cid, challenge.success_criteria
        )
        self.assertTrue(is_complete)
        self.assertIn("Converged", reason)

    def test_still_learning(self):
        """Should report 'Still learning' when actively improving below target."""
        challenge = _make_challenge(
            accuracy=0.9,
            min_samples=5,
            max_iterations=1000,
            convergence_window=5,
        )
        cid = self.tracker.start_challenge(challenge)
        # Provide enough samples but below accuracy
        for i in range(10):
            self.tracker.update_progress(
                cid, accuracy=0.3 + i * 0.02, strategy="hebbian"
            )
        is_complete, reason = self.tracker.check_completion(
            cid, challenge.success_criteria
        )
        self.assertFalse(is_complete)
        self.assertIn("Still learning", reason)

    def test_custom_metrics_below_threshold(self):
        """Failing custom metric should prevent completion."""
        challenge = _make_challenge(
            accuracy=0.9,
            min_samples=5,
            max_iterations=1000,
            custom_metrics={"f1_score": 0.8},
        )
        cid = self.tracker.start_challenge(challenge)
        # Accuracy below criteria.accuracy (0.9) so convergence block is skipped,
        # and the custom metrics check at the end is reached.
        for _ in range(10):
            self.tracker.update_progress(
                cid, accuracy=0.6, strategy="hebbian",
                custom_metrics={"f1_score": 0.3},
            )
        is_complete, reason = self.tracker.check_completion(
            cid, challenge.success_criteria
        )
        self.assertFalse(is_complete)
        self.assertIn("f1_score", reason)


# ---------------------------------------------------------------------------
# ProgressTracker.get_progress_report tests
# ---------------------------------------------------------------------------

class TestGetProgressReport(unittest.TestCase):
    """Tests for progress report generation."""

    def setUp(self):
        self.tracker = ProgressTracker()
        self.challenge = _make_challenge(
            accuracy=0.8,
            min_samples=10,
            convergence_window=5,
        )
        self.cid = self.tracker.start_challenge(self.challenge)

    def test_report_type(self):
        """get_progress_report should return a ProgressReport."""
        self.tracker.update_progress(self.cid, accuracy=0.5, strategy="hebbian")
        report = self.tracker.get_progress_report(self.cid, self.challenge)
        self.assertIsInstance(report, ProgressReport)

    def test_report_challenge_id(self):
        """Report should contain the correct challenge ID."""
        self.tracker.update_progress(self.cid, accuracy=0.5, strategy="hebbian")
        report = self.tracker.get_progress_report(self.cid, self.challenge)
        self.assertEqual(report.challenge_id, self.cid)

    def test_report_challenge_name(self):
        """Report should contain the correct challenge name."""
        self.tracker.update_progress(self.cid, accuracy=0.5, strategy="hebbian")
        report = self.tracker.get_progress_report(self.cid, self.challenge)
        self.assertEqual(report.challenge_name, self.challenge.name)

    def test_report_accuracy(self):
        """Report should reflect the current accuracy."""
        self.tracker.update_progress(self.cid, accuracy=0.65, strategy="hebbian")
        report = self.tracker.get_progress_report(self.cid, self.challenge)
        self.assertAlmostEqual(report.current_accuracy, 0.65)

    def test_report_iterations(self):
        """Report should count completed iterations."""
        for i in range(7):
            self.tracker.update_progress(
                self.cid, accuracy=0.1 * (i + 1), strategy="hebbian"
            )
        report = self.tracker.get_progress_report(self.cid, self.challenge)
        self.assertEqual(report.iterations_completed, 7)

    def test_report_learning_curve(self):
        """Report should include the full learning curve."""
        values = [0.1, 0.3, 0.5]
        for v in values:
            self.tracker.update_progress(self.cid, accuracy=v, strategy="hebbian")
        report = self.tracker.get_progress_report(self.cid, self.challenge)
        self.assertEqual(report.learning_curve, values)

    def test_report_current_strategy(self):
        """Report should show the most recent strategy."""
        self.tracker.update_progress(self.cid, accuracy=0.3, strategy="hebbian")
        self.tracker.update_progress(self.cid, accuracy=0.5, strategy="stdp")
        report = self.tracker.get_progress_report(self.cid, self.challenge)
        self.assertEqual(report.current_strategy, "stdp")

    def test_report_strategy_switches(self):
        """Report should count how many times the strategy changed."""
        strategies = ["hebbian", "hebbian", "stdp", "stdp", "bcm"]
        for i, s in enumerate(strategies):
            self.tracker.update_progress(
                self.cid, accuracy=0.1 * (i + 1), strategy=s
            )
        report = self.tracker.get_progress_report(self.cid, self.challenge)
        # hebbian->stdp = 1, stdp->bcm = 2
        self.assertEqual(report.strategy_switches, 2)

    def test_report_progress_percent_range(self):
        """Progress percentage should be in [0.0, 1.0]."""
        self.tracker.update_progress(self.cid, accuracy=0.5, strategy="hebbian")
        report = self.tracker.get_progress_report(self.cid, self.challenge)
        self.assertGreaterEqual(report.progress_percent, 0.0)
        self.assertLessEqual(report.progress_percent, 1.0)

    def test_report_time_elapsed(self):
        """Time elapsed should be non-negative."""
        self.tracker.update_progress(self.cid, accuracy=0.5, strategy="hebbian")
        report = self.tracker.get_progress_report(self.cid, self.challenge)
        self.assertGreaterEqual(report.time_elapsed_seconds, 0.0)

    def test_report_str_does_not_error(self):
        """ProgressReport.__str__ should not raise."""
        self.tracker.update_progress(self.cid, accuracy=0.5, strategy="hebbian")
        report = self.tracker.get_progress_report(self.cid, self.challenge)
        text = str(report)
        self.assertIsInstance(text, str)
        self.assertIn(self.challenge.name, text)

    def test_report_unknown_challenge_raises(self):
        """get_progress_report should raise ValueError for unknown ID."""
        with self.assertRaises(ValueError):
            self.tracker.get_progress_report("bogus", self.challenge)


# ---------------------------------------------------------------------------
# ProgressTracker.complete_challenge and curriculum tests
# ---------------------------------------------------------------------------

class TestCompleteChallenge(unittest.TestCase):
    """Tests for challenge completion and curriculum adjustment."""

    def test_successful_completion_records(self):
        """Successful completion should add to completed_challenges."""
        tracker = ProgressTracker()
        challenge = _make_challenge()
        cid = tracker.start_challenge(challenge)
        tracker.update_progress(cid, accuracy=0.9, strategy="hebbian")
        tracker.complete_challenge(cid, success=True, final_accuracy=0.9)
        self.assertIn(cid, tracker.completed_challenges)
        self.assertNotIn(cid, tracker.failed_challenges)

    def test_failed_completion_records(self):
        """Failed completion should add to failed_challenges."""
        tracker = ProgressTracker()
        challenge = _make_challenge()
        cid = tracker.start_challenge(challenge)
        tracker.update_progress(cid, accuracy=0.2, strategy="hebbian")
        tracker.complete_challenge(cid, success=False, final_accuracy=0.2)
        self.assertIn(cid, tracker.failed_challenges)
        self.assertNotIn(cid, tracker.completed_challenges)

    def test_performance_history_updated(self):
        """Completion should add an entry to performance_history."""
        tracker = ProgressTracker()
        challenge = _make_challenge()
        cid = tracker.start_challenge(challenge)
        tracker.complete_challenge(cid, success=True, final_accuracy=0.85)
        self.assertEqual(len(tracker.performance_history), 1)
        entry = tracker.performance_history[0]
        self.assertEqual(entry['challenge_id'], cid)
        self.assertTrue(entry['success'])
        self.assertAlmostEqual(entry['accuracy'], 0.85)

    def test_curriculum_difficulty_increases_on_success(self):
        """Difficulty should increase after a successful high-accuracy result."""
        tracker = ProgressTracker(mastery_threshold=0.8, curriculum_adjustment_rate=0.1)
        initial = tracker.curriculum_difficulty
        challenge = _make_challenge()
        cid = tracker.start_challenge(challenge)
        tracker.complete_challenge(cid, success=True, final_accuracy=0.9)
        self.assertAlmostEqual(tracker.curriculum_difficulty, initial + 0.1)

    def test_curriculum_difficulty_decreases_on_failure(self):
        """Difficulty should decrease after failure."""
        tracker = ProgressTracker(mastery_threshold=0.8, curriculum_adjustment_rate=0.1)
        initial = tracker.curriculum_difficulty
        challenge = _make_challenge()
        cid = tracker.start_challenge(challenge)
        tracker.complete_challenge(cid, success=False, final_accuracy=0.2)
        self.assertAlmostEqual(tracker.curriculum_difficulty, initial - 0.1)

    def test_curriculum_difficulty_clamped_at_max(self):
        """Difficulty should not exceed 1.0."""
        tracker = ProgressTracker(curriculum_adjustment_rate=0.5)
        tracker.curriculum_difficulty = 0.9
        challenge = _make_challenge()
        cid = tracker.start_challenge(challenge)
        tracker.complete_challenge(cid, success=True, final_accuracy=0.95)
        self.assertLessEqual(tracker.curriculum_difficulty, 1.0)

    def test_curriculum_difficulty_clamped_at_min(self):
        """Difficulty should not go below 0.1."""
        tracker = ProgressTracker(curriculum_adjustment_rate=0.5)
        tracker.curriculum_difficulty = 0.2
        challenge = _make_challenge()
        cid = tracker.start_challenge(challenge)
        tracker.complete_challenge(cid, success=False, final_accuracy=0.1)
        self.assertGreaterEqual(tracker.curriculum_difficulty, 0.1)

    def test_curriculum_unchanged_moderate_performance(self):
        """Difficulty unchanged when not successful but accuracy not very low."""
        tracker = ProgressTracker(mastery_threshold=0.8, curriculum_adjustment_rate=0.1)
        initial = tracker.curriculum_difficulty
        challenge = _make_challenge()
        cid = tracker.start_challenge(challenge)
        # success=True but accuracy below mastery -> no increase
        # accuracy >= mastery * 0.5 = 0.4 -> no decrease either
        tracker.complete_challenge(cid, success=True, final_accuracy=0.5)
        self.assertAlmostEqual(tracker.curriculum_difficulty, initial)


# ---------------------------------------------------------------------------
# ProgressTracker.get_stats / serialization tests
# ---------------------------------------------------------------------------

class TestGetStats(unittest.TestCase):
    """Tests for get_stats() which provides a serializable state snapshot."""

    def test_initial_stats(self):
        """Stats on a fresh tracker should show zeros."""
        tracker = ProgressTracker()
        stats = tracker.get_stats()
        self.assertEqual(stats['total_challenges'], 0)
        self.assertEqual(stats['completed'], 0)
        self.assertEqual(stats['failed'], 0)
        self.assertAlmostEqual(stats['success_rate'], 0.0)
        self.assertAlmostEqual(stats['current_difficulty'], 0.3)

    def test_stats_after_completions(self):
        """Stats should reflect completed and failed counts."""
        tracker = ProgressTracker()
        for i in range(3):
            c = _make_challenge(name=f"success_{i}")
            cid = tracker.start_challenge(c)
            tracker.complete_challenge(cid, success=True, final_accuracy=0.9)
        for i in range(2):
            c = _make_challenge(name=f"fail_{i}")
            cid = tracker.start_challenge(c)
            tracker.complete_challenge(cid, success=False, final_accuracy=0.2)

        stats = tracker.get_stats()
        self.assertEqual(stats['total_challenges'], 5)
        self.assertEqual(stats['completed'], 3)
        self.assertEqual(stats['failed'], 2)
        self.assertAlmostEqual(stats['success_rate'], 3.0 / 5.0)

    def test_stats_active_challenges(self):
        """Active challenges = total tracked - completed - failed."""
        tracker = ProgressTracker()
        c1 = _make_challenge(name="active")
        c2 = _make_challenge(name="done")
        tracker.start_challenge(c1)
        cid2 = tracker.start_challenge(c2)
        tracker.complete_challenge(cid2, success=True, final_accuracy=0.9)

        stats = tracker.get_stats()
        # 2 challenges tracked, 1 completed + 0 failed = 1 total finished
        self.assertEqual(stats['active_challenges'], 2 - 1)

    def test_stats_returns_dict(self):
        """get_stats should always return a plain dictionary."""
        tracker = ProgressTracker()
        stats = tracker.get_stats()
        self.assertIsInstance(stats, dict)

    def test_stats_keys(self):
        """get_stats dict should contain all expected keys."""
        tracker = ProgressTracker()
        stats = tracker.get_stats()
        expected_keys = {
            'total_challenges', 'completed', 'failed',
            'success_rate', 'current_difficulty', 'active_challenges',
        }
        self.assertEqual(set(stats.keys()), expected_keys)

    def test_stats_current_difficulty_tracks_curriculum(self):
        """current_difficulty in stats should match curriculum_difficulty."""
        tracker = ProgressTracker()
        tracker.curriculum_difficulty = 0.75
        stats = tracker.get_stats()
        self.assertAlmostEqual(stats['current_difficulty'], 0.75)


# ---------------------------------------------------------------------------
# ProgressTracker.get_learning_curve tests
# ---------------------------------------------------------------------------

class TestGetLearningCurve(unittest.TestCase):
    """Tests for the get_learning_curve convenience method."""

    def test_unknown_challenge_returns_empty(self):
        """get_learning_curve for unknown ID should return []."""
        tracker = ProgressTracker()
        self.assertEqual(tracker.get_learning_curve("nope"), [])

    def test_returns_copy(self):
        """The returned list should be a copy, not the original."""
        tracker = ProgressTracker()
        challenge = _make_challenge()
        cid = tracker.start_challenge(challenge)
        tracker.update_progress(cid, accuracy=0.5, strategy="hebbian")
        curve = tracker.get_learning_curve(cid)
        curve.append(999.0)  # mutate the copy
        self.assertEqual(len(tracker.challenges[cid].accuracy_history), 1)

    def test_returns_correct_values(self):
        """The curve should match the recorded accuracy history."""
        tracker = ProgressTracker()
        challenge = _make_challenge()
        cid = tracker.start_challenge(challenge)
        values = [0.1, 0.4, 0.6, 0.8]
        for v in values:
            tracker.update_progress(cid, accuracy=v, strategy="hebbian")
        self.assertEqual(tracker.get_learning_curve(cid), values)


# ---------------------------------------------------------------------------
# ProgressTracker.adjust_curriculum tests
# ---------------------------------------------------------------------------

class TestAdjustCurriculum(unittest.TestCase):
    """Tests for the adjust_curriculum method."""

    def test_returns_current_difficulty(self):
        """adjust_curriculum should return the current curriculum_difficulty."""
        tracker = ProgressTracker()
        challenge = _make_challenge()
        cid = tracker.start_challenge(challenge)
        result = tracker.adjust_curriculum(cid)
        self.assertAlmostEqual(result, tracker.curriculum_difficulty)

    def test_reflects_changes(self):
        """After completing challenges, adjust_curriculum should reflect new difficulty."""
        tracker = ProgressTracker(curriculum_adjustment_rate=0.1)
        c1 = _make_challenge(name="c1")
        cid1 = tracker.start_challenge(c1)
        tracker.complete_challenge(cid1, success=True, final_accuracy=0.95)
        c2 = _make_challenge(name="c2")
        cid2 = tracker.start_challenge(c2)
        result = tracker.adjust_curriculum(cid2)
        self.assertAlmostEqual(result, 0.4)  # 0.3 + 0.1


# ---------------------------------------------------------------------------
# Strategy switch counting (internal helper)
# ---------------------------------------------------------------------------

class TestStrategySwitch(unittest.TestCase):
    """Tests for the internal _count_strategy_switches helper."""

    def test_empty_history(self):
        """No strategy history should yield 0 switches."""
        tracker = ProgressTracker()
        self.assertEqual(tracker._count_strategy_switches([]), 0)

    def test_single_entry(self):
        """Single-entry history should yield 0 switches."""
        tracker = ProgressTracker()
        self.assertEqual(tracker._count_strategy_switches(["hebbian"]), 0)

    def test_no_switches(self):
        """Same strategy repeated should yield 0 switches."""
        tracker = ProgressTracker()
        self.assertEqual(
            tracker._count_strategy_switches(["hebbian"] * 5), 0
        )

    def test_every_switch(self):
        """Alternating strategies should count every transition."""
        tracker = ProgressTracker()
        history = ["hebbian", "stdp", "hebbian", "stdp"]
        self.assertEqual(tracker._count_strategy_switches(history), 3)

    def test_multiple_switches(self):
        """Mixed history should count correctly."""
        tracker = ProgressTracker()
        history = ["a", "a", "b", "b", "b", "c", "a"]
        # a->b = 1, b->c = 2, c->a = 3
        self.assertEqual(tracker._count_strategy_switches(history), 3)


# ---------------------------------------------------------------------------
# Integration: full lifecycle test
# ---------------------------------------------------------------------------

class TestFullLifecycle(unittest.TestCase):
    """Integration test covering a full challenge lifecycle."""

    def test_complete_lifecycle(self):
        """Start, update, check, report, and complete a challenge."""
        tracker = ProgressTracker(
            mastery_threshold=0.8,
            plateau_window=5,
            plateau_threshold=0.01,
        )
        challenge = _make_challenge(
            accuracy=0.7,
            min_samples=5,
            max_iterations=100,
            convergence_window=3,
            convergence_threshold=0.01,
        )

        # Step 1: Start
        cid = tracker.start_challenge(challenge)
        self.assertIn(cid, tracker.challenges)

        # Step 2: Update with improving accuracy
        accuracies = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.75, 0.75]
        for acc in accuracies:
            result = tracker.update_progress(cid, accuracy=acc, strategy="hebbian")

        self.assertEqual(result['iterations'], 10)
        self.assertAlmostEqual(result['best_accuracy'], 0.75)

        # Step 3: Check completion - should converge at 0.75 (>= 0.7 target)
        is_complete, reason = tracker.check_completion(cid, challenge.success_criteria)
        self.assertTrue(is_complete)
        self.assertIn("Converged", reason)

        # Step 4: Get report
        report = tracker.get_progress_report(cid, challenge)
        self.assertIsInstance(report, ProgressReport)
        self.assertEqual(report.iterations_completed, 10)
        self.assertAlmostEqual(report.current_accuracy, 0.75)

        # Step 5: Complete
        tracker.complete_challenge(cid, success=True, final_accuracy=0.75)
        self.assertIn(cid, tracker.completed_challenges)

        # Step 6: Verify stats
        stats = tracker.get_stats()
        self.assertEqual(stats['completed'], 1)
        self.assertEqual(stats['failed'], 0)
        self.assertAlmostEqual(stats['success_rate'], 1.0)


if __name__ == "__main__":
    unittest.main()
