"""
Comprehensive tests for the CurriculumSystem module.

Tests cover initialization, curriculum level progression, challenge result
recording, level progress tracking, difficulty scaling, and state
serialization (get_stats).

All tests are deterministic and pass reliably in both CPU-only and GPU
environments.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from self_organizing_av_system.core.backend import xp
from self_organizing_av_system.core.curriculum_system import (
    CurriculumLevel,
    CurriculumSystem,
    ChallengeResult,
    LevelInfo,
    LevelProgress,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_passing_result(level, name="test_challenge", accuracy=0.95,
                         iterations=100, strategy="hebbian"):
    """Create a ChallengeResult that counts as passed."""
    return ChallengeResult(
        challenge_name=name,
        level=level,
        accuracy=accuracy,
        passed=True,
        iterations=iterations,
        strategy_used=strategy,
        timestamp=0.0,
    )


def _make_failing_result(level, name="test_challenge", accuracy=0.30,
                         iterations=100, strategy="hebbian"):
    """Create a ChallengeResult that counts as failed."""
    return ChallengeResult(
        challenge_name=name,
        level=level,
        accuracy=accuracy,
        passed=False,
        iterations=iterations,
        strategy_used=strategy,
        timestamp=0.0,
    )


def _complete_level(cs, level, accuracy=0.95):
    """Record enough passing results to complete a level."""
    level_info = cs.CURRICULUM[level]
    for challenge_dict in level_info.challenges:
        result = _make_passing_result(
            level=level,
            name=challenge_dict["name"],
            accuracy=accuracy,
        )
        cs.record_result(result)


# =========================================================================
# CurriculumLevel enum
# =========================================================================

class TestCurriculumLevel(unittest.TestCase):
    """Tests for the CurriculumLevel enum."""

    def test_has_five_levels(self):
        """There should be exactly five curriculum levels."""
        self.assertEqual(len(CurriculumLevel), 5)

    def test_level_values_are_sequential(self):
        """Level values should be 1 through 5."""
        expected = [1, 2, 3, 4, 5]
        actual = sorted(level.value for level in CurriculumLevel)
        self.assertEqual(actual, expected)

    def test_level_names(self):
        """Each level enum member should have the expected name."""
        self.assertEqual(CurriculumLevel.LEVEL_1_BASIC.value, 1)
        self.assertEqual(CurriculumLevel.LEVEL_2_ASSOCIATION.value, 2)
        self.assertEqual(CurriculumLevel.LEVEL_3_INTERMEDIATE.value, 3)
        self.assertEqual(CurriculumLevel.LEVEL_4_ADVANCED.value, 4)
        self.assertEqual(CurriculumLevel.LEVEL_5_EXPERT.value, 5)

    def test_construct_from_value(self):
        """CurriculumLevel(n) should return the correct member."""
        for val in range(1, 6):
            level = CurriculumLevel(val)
            self.assertEqual(level.value, val)

    def test_invalid_value_raises(self):
        """CurriculumLevel with an out-of-range value should raise ValueError."""
        with self.assertRaises(ValueError):
            CurriculumLevel(0)
        with self.assertRaises(ValueError):
            CurriculumLevel(6)


# =========================================================================
# LevelProgress dataclass
# =========================================================================

class TestLevelProgress(unittest.TestCase):
    """Tests for the LevelProgress dataclass and its computed properties."""

    def test_default_values(self):
        """A fresh LevelProgress should have zero completions."""
        lp = LevelProgress(level=CurriculumLevel.LEVEL_1_BASIC)
        self.assertEqual(lp.challenges_completed, 0)
        self.assertEqual(lp.challenges_total, 0)
        self.assertEqual(lp.current_challenge_index, 0)
        self.assertFalse(lp.unlocked)
        self.assertFalse(lp.completed)
        self.assertEqual(lp.accuracies, [])

    def test_average_accuracy_empty(self):
        """average_accuracy with no data should return 0.0."""
        lp = LevelProgress(level=CurriculumLevel.LEVEL_1_BASIC)
        self.assertAlmostEqual(lp.average_accuracy, 0.0)

    def test_average_accuracy_single(self):
        """average_accuracy with one entry should return that entry."""
        lp = LevelProgress(
            level=CurriculumLevel.LEVEL_1_BASIC,
            accuracies=[0.85],
        )
        self.assertAlmostEqual(lp.average_accuracy, 0.85)

    def test_average_accuracy_multiple(self):
        """average_accuracy should be the arithmetic mean of all entries."""
        lp = LevelProgress(
            level=CurriculumLevel.LEVEL_1_BASIC,
            accuracies=[0.70, 0.80, 0.90],
        )
        self.assertAlmostEqual(lp.average_accuracy, 0.80)

    def test_progress_percent_zero_total(self):
        """progress_percent with zero total challenges should return 0.0."""
        lp = LevelProgress(
            level=CurriculumLevel.LEVEL_1_BASIC,
            challenges_total=0,
        )
        self.assertAlmostEqual(lp.progress_percent, 0.0)

    def test_progress_percent_partial(self):
        """progress_percent should be completed / total."""
        lp = LevelProgress(
            level=CurriculumLevel.LEVEL_1_BASIC,
            challenges_completed=3,
            challenges_total=8,
        )
        self.assertAlmostEqual(lp.progress_percent, 3.0 / 8.0)

    def test_progress_percent_full(self):
        """progress_percent at full completion should be 1.0."""
        lp = LevelProgress(
            level=CurriculumLevel.LEVEL_1_BASIC,
            challenges_completed=5,
            challenges_total=5,
        )
        self.assertAlmostEqual(lp.progress_percent, 1.0)


# =========================================================================
# ChallengeResult dataclass
# =========================================================================

class TestChallengeResult(unittest.TestCase):
    """Tests for the ChallengeResult dataclass."""

    def test_fields(self):
        """ChallengeResult should store all provided fields."""
        cr = ChallengeResult(
            challenge_name="Draw solid red",
            level=CurriculumLevel.LEVEL_1_BASIC,
            accuracy=0.92,
            passed=True,
            iterations=150,
            strategy_used="hebbian",
            timestamp=1000.0,
        )
        self.assertEqual(cr.challenge_name, "Draw solid red")
        self.assertEqual(cr.level, CurriculumLevel.LEVEL_1_BASIC)
        self.assertAlmostEqual(cr.accuracy, 0.92)
        self.assertTrue(cr.passed)
        self.assertEqual(cr.iterations, 150)
        self.assertEqual(cr.strategy_used, "hebbian")
        self.assertAlmostEqual(cr.timestamp, 1000.0)

    def test_default_timestamp(self):
        """timestamp should default to a float (time.time) when omitted."""
        cr = ChallengeResult(
            challenge_name="test",
            level=CurriculumLevel.LEVEL_1_BASIC,
            accuracy=0.5,
            passed=False,
            iterations=10,
            strategy_used="stdp",
        )
        self.assertIsInstance(cr.timestamp, float)
        self.assertGreater(cr.timestamp, 0.0)


# =========================================================================
# LevelInfo dataclass
# =========================================================================

class TestLevelInfo(unittest.TestCase):
    """Tests for the LevelInfo dataclass."""

    def test_fields(self):
        """LevelInfo should store all provided fields."""
        li = LevelInfo(
            level=CurriculumLevel.LEVEL_1_BASIC,
            name="Test Level",
            description="A test level",
            challenges=[{"name": "c1"}],
            unlock_threshold=0.75,
        )
        self.assertEqual(li.level, CurriculumLevel.LEVEL_1_BASIC)
        self.assertEqual(li.name, "Test Level")
        self.assertEqual(li.description, "A test level")
        self.assertEqual(len(li.challenges), 1)
        self.assertAlmostEqual(li.unlock_threshold, 0.75)

    def test_default_unlock_threshold(self):
        """Default unlock_threshold should be 0.80."""
        li = LevelInfo(
            level=CurriculumLevel.LEVEL_1_BASIC,
            name="test",
            description="test",
            challenges=[],
        )
        self.assertAlmostEqual(li.unlock_threshold, 0.80)


# =========================================================================
# CurriculumSystem initialization
# =========================================================================

class TestCurriculumSystemInitialization(unittest.TestCase):
    """Tests for CurriculumSystem construction and initial state."""

    def setUp(self):
        self.cs = CurriculumSystem(state_dim=64)

    def test_state_dim_stored(self):
        """state_dim should be stored on the instance."""
        self.assertEqual(self.cs.state_dim, 64)

    def test_default_state_dim(self):
        """Default state_dim should be 128."""
        cs = CurriculumSystem()
        self.assertEqual(cs.state_dim, 128)

    def test_current_level_is_basic(self):
        """Initial current_level should be LEVEL_1_BASIC."""
        self.assertEqual(self.cs.current_level, CurriculumLevel.LEVEL_1_BASIC)

    def test_results_initially_empty(self):
        """Results list should start empty."""
        self.assertEqual(len(self.cs.results), 0)

    def test_progress_for_all_levels(self):
        """Progress should be initialized for all five levels."""
        self.assertEqual(len(self.cs.progress), 5)
        for level in CurriculumLevel:
            self.assertIn(level, self.cs.progress)

    def test_first_level_unlocked(self):
        """Only the first level should be unlocked at start."""
        self.assertTrue(self.cs.progress[CurriculumLevel.LEVEL_1_BASIC].unlocked)

    def test_higher_levels_locked(self):
        """Levels 2-5 should be locked at start."""
        for level in [CurriculumLevel.LEVEL_2_ASSOCIATION,
                      CurriculumLevel.LEVEL_3_INTERMEDIATE,
                      CurriculumLevel.LEVEL_4_ADVANCED,
                      CurriculumLevel.LEVEL_5_EXPERT]:
            self.assertFalse(self.cs.progress[level].unlocked)

    def test_challenges_total_matches_curriculum(self):
        """challenges_total for each level should match the CURRICULUM data."""
        for level in CurriculumLevel:
            expected = len(self.cs.CURRICULUM[level].challenges)
            actual = self.cs.progress[level].challenges_total
            self.assertEqual(actual, expected,
                             f"Level {level.name}: expected {expected}, got {actual}")

    def test_no_challenges_completed_initially(self):
        """All levels should have zero challenges_completed at start."""
        for level in CurriculumLevel:
            self.assertEqual(self.cs.progress[level].challenges_completed, 0)

    def test_curriculum_has_all_levels(self):
        """The CURRICULUM class attribute should map every CurriculumLevel."""
        for level in CurriculumLevel:
            self.assertIn(level, CurriculumSystem.CURRICULUM)
            self.assertIsInstance(CurriculumSystem.CURRICULUM[level], LevelInfo)


# =========================================================================
# Curriculum level progression
# =========================================================================

class TestCurriculumLevelProgression(unittest.TestCase):
    """Tests for advancing and setting curriculum levels."""

    def setUp(self):
        self.cs = CurriculumSystem(state_dim=64)

    def test_cannot_advance_when_next_locked(self):
        """advance_to_next_level should return False when next level is locked."""
        result = self.cs.advance_to_next_level()
        self.assertFalse(result)
        self.assertEqual(self.cs.current_level, CurriculumLevel.LEVEL_1_BASIC)

    def test_advance_after_unlocking(self):
        """advance_to_next_level should succeed after the next level is unlocked."""
        self.cs.progress[CurriculumLevel.LEVEL_2_ASSOCIATION].unlocked = True
        result = self.cs.advance_to_next_level()
        self.assertTrue(result)
        self.assertEqual(self.cs.current_level, CurriculumLevel.LEVEL_2_ASSOCIATION)

    def test_advance_at_max_level(self):
        """advance_to_next_level should return False at level 5."""
        self.cs.current_level = CurriculumLevel.LEVEL_5_EXPERT
        result = self.cs.advance_to_next_level()
        self.assertFalse(result)
        self.assertEqual(self.cs.current_level, CurriculumLevel.LEVEL_5_EXPERT)

    def test_sequential_advancement(self):
        """Unlocking and advancing through levels 1 through 5."""
        for current_val in range(1, 5):
            next_level = CurriculumLevel(current_val + 1)
            self.cs.progress[next_level].unlocked = True
            self.assertTrue(self.cs.advance_to_next_level())
            self.assertEqual(self.cs.current_level, next_level)

    def test_set_level_unlocked(self):
        """set_level should work for an unlocked level."""
        self.cs.progress[CurriculumLevel.LEVEL_3_INTERMEDIATE].unlocked = True
        self.cs.set_level(CurriculumLevel.LEVEL_3_INTERMEDIATE)
        self.assertEqual(self.cs.current_level, CurriculumLevel.LEVEL_3_INTERMEDIATE)

    def test_set_level_locked_no_change(self):
        """set_level should not change current_level for a locked level."""
        original = self.cs.current_level
        self.cs.set_level(CurriculumLevel.LEVEL_4_ADVANCED)
        self.assertEqual(self.cs.current_level, original)

    def test_is_level_unlocked(self):
        """is_level_unlocked should reflect the unlocked flag."""
        self.assertTrue(self.cs.is_level_unlocked(CurriculumLevel.LEVEL_1_BASIC))
        self.assertFalse(self.cs.is_level_unlocked(CurriculumLevel.LEVEL_2_ASSOCIATION))

    def test_completing_level_unlocks_next(self):
        """Completing all challenges of a level with high accuracy should unlock next."""
        _complete_level(self.cs, CurriculumLevel.LEVEL_1_BASIC, accuracy=0.95)
        self.assertTrue(self.cs.progress[CurriculumLevel.LEVEL_2_ASSOCIATION].unlocked)

    def test_completing_level_low_accuracy_does_not_unlock(self):
        """Completing a level below the threshold should NOT unlock next level."""
        cs = CurriculumSystem(state_dim=64)
        level = CurriculumLevel.LEVEL_1_BASIC
        level_info = cs.CURRICULUM[level]
        # Record results that pass individually but have a very low accuracy
        # that is below the unlock_threshold of 0.70 for level 1
        for challenge_dict in level_info.challenges:
            result = ChallengeResult(
                challenge_name=challenge_dict["name"],
                level=level,
                accuracy=0.50,
                passed=True,
                iterations=100,
                strategy_used="hebbian",
                timestamp=0.0,
            )
            cs.record_result(result)
        self.assertFalse(cs.progress[CurriculumLevel.LEVEL_2_ASSOCIATION].unlocked)


# =========================================================================
# Challenge result recording
# =========================================================================

class TestChallengeResultRecording(unittest.TestCase):
    """Tests for recording challenge results."""

    def setUp(self):
        self.cs = CurriculumSystem(state_dim=64)

    def test_record_appends_to_results(self):
        """Each call to record_result should append to the results list."""
        result = _make_passing_result(CurriculumLevel.LEVEL_1_BASIC)
        self.cs.record_result(result)
        self.assertEqual(len(self.cs.results), 1)
        self.assertIs(self.cs.results[0], result)

    def test_record_multiple_results(self):
        """Multiple recorded results should be stored in order."""
        for i in range(5):
            r = _make_passing_result(
                CurriculumLevel.LEVEL_1_BASIC,
                name=f"challenge_{i}",
            )
            self.cs.record_result(r)
        self.assertEqual(len(self.cs.results), 5)
        for i in range(5):
            self.assertEqual(self.cs.results[i].challenge_name, f"challenge_{i}")

    def test_passed_result_increments_completed(self):
        """A passed result should increment challenges_completed."""
        before = self.cs.progress[CurriculumLevel.LEVEL_1_BASIC].challenges_completed
        result = _make_passing_result(CurriculumLevel.LEVEL_1_BASIC)
        self.cs.record_result(result)
        after = self.cs.progress[CurriculumLevel.LEVEL_1_BASIC].challenges_completed
        self.assertEqual(after, before + 1)

    def test_failed_result_does_not_increment_completed(self):
        """A failed result should NOT increment challenges_completed."""
        before = self.cs.progress[CurriculumLevel.LEVEL_1_BASIC].challenges_completed
        result = _make_failing_result(CurriculumLevel.LEVEL_1_BASIC)
        self.cs.record_result(result)
        after = self.cs.progress[CurriculumLevel.LEVEL_1_BASIC].challenges_completed
        self.assertEqual(after, before)

    def test_accuracy_recorded_regardless_of_pass(self):
        """Both passed and failed results should append accuracy."""
        self.cs.record_result(_make_passing_result(
            CurriculumLevel.LEVEL_1_BASIC, accuracy=0.90))
        self.cs.record_result(_make_failing_result(
            CurriculumLevel.LEVEL_1_BASIC, accuracy=0.40))
        accs = self.cs.progress[CurriculumLevel.LEVEL_1_BASIC].accuracies
        self.assertEqual(len(accs), 2)
        self.assertAlmostEqual(accs[0], 0.90)
        self.assertAlmostEqual(accs[1], 0.40)

    def test_passed_result_advances_challenge_index(self):
        """A passed result should increment current_challenge_index."""
        before = self.cs.progress[CurriculumLevel.LEVEL_1_BASIC].current_challenge_index
        result = _make_passing_result(CurriculumLevel.LEVEL_1_BASIC)
        self.cs.record_result(result)
        after = self.cs.progress[CurriculumLevel.LEVEL_1_BASIC].current_challenge_index
        self.assertEqual(after, before + 1)

    def test_failed_result_does_not_advance_challenge_index(self):
        """A failed result should NOT increment current_challenge_index."""
        before = self.cs.progress[CurriculumLevel.LEVEL_1_BASIC].current_challenge_index
        result = _make_failing_result(CurriculumLevel.LEVEL_1_BASIC)
        self.cs.record_result(result)
        after = self.cs.progress[CurriculumLevel.LEVEL_1_BASIC].current_challenge_index
        self.assertEqual(after, before)

    def test_level_marked_completed_when_all_challenges_done(self):
        """Level should be marked completed when all challenges pass."""
        _complete_level(self.cs, CurriculumLevel.LEVEL_1_BASIC)
        self.assertTrue(self.cs.progress[CurriculumLevel.LEVEL_1_BASIC].completed)


# =========================================================================
# Level progress tracking
# =========================================================================

class TestLevelProgressTracking(unittest.TestCase):
    """Tests for level progress queries and updates."""

    def setUp(self):
        self.cs = CurriculumSystem(state_dim=64)

    def test_get_level_progress_initially_zero(self):
        """get_level_progress should be 0.0 for a fresh level."""
        for level in CurriculumLevel:
            self.assertAlmostEqual(self.cs.get_level_progress(level), 0.0)

    def test_get_level_progress_after_some_completions(self):
        """get_level_progress should reflect challenges_completed / total."""
        level = CurriculumLevel.LEVEL_1_BASIC
        total = self.cs.progress[level].challenges_total
        # Complete two challenges
        for i in range(2):
            r = _make_passing_result(level)
            self.cs.record_result(r)
        expected = 2.0 / total
        self.assertAlmostEqual(self.cs.get_level_progress(level), expected)

    def test_get_level_info(self):
        """get_level_info should return the LevelInfo for the given level."""
        for level in CurriculumLevel:
            info = self.cs.get_level_info(level)
            self.assertIsInstance(info, LevelInfo)
            self.assertEqual(info.level, level)

    def test_get_current_challenge_at_start(self):
        """get_current_challenge should return the first challenge of level 1."""
        ch = self.cs.get_current_challenge()
        self.assertIsNotNone(ch)
        expected_first = self.cs.CURRICULUM[CurriculumLevel.LEVEL_1_BASIC].challenges[0]
        self.assertEqual(ch["name"], expected_first["name"])

    def test_get_current_challenge_returns_none_past_end(self):
        """get_current_challenge should return None when all challenges exhausted."""
        level = CurriculumLevel.LEVEL_1_BASIC
        total = self.cs.progress[level].challenges_total
        self.cs.progress[level].current_challenge_index = total
        self.assertIsNone(self.cs.get_current_challenge())

    def test_set_challenge_index_valid(self):
        """set_challenge_index should update the index within bounds."""
        self.cs.set_challenge_index(3)
        self.assertEqual(
            self.cs.progress[CurriculumLevel.LEVEL_1_BASIC].current_challenge_index, 3)

    def test_set_challenge_index_out_of_range(self):
        """set_challenge_index with invalid index should not change it."""
        self.cs.set_challenge_index(0)
        self.cs.set_challenge_index(999)
        # Should remain at 0 (the last valid setting)
        self.assertEqual(
            self.cs.progress[CurriculumLevel.LEVEL_1_BASIC].current_challenge_index, 0)

    def test_set_challenge_index_negative(self):
        """set_challenge_index with negative index should not change it."""
        self.cs.set_challenge_index(2)
        self.cs.set_challenge_index(-1)
        self.assertEqual(
            self.cs.progress[CurriculumLevel.LEVEL_1_BASIC].current_challenge_index, 2)

    def test_update_progress_records_accuracy(self):
        """update_progress should append accuracy to the level's accuracies."""
        self.cs.update_progress(CurriculumLevel.LEVEL_1_BASIC, 0, 0.85)
        accs = self.cs.progress[CurriculumLevel.LEVEL_1_BASIC].accuracies
        self.assertEqual(len(accs), 1)
        self.assertAlmostEqual(accs[0], 0.85)

    def test_update_progress_increments_completed_on_target_met(self):
        """update_progress should increment challenges_completed when target is met."""
        target = self.cs.CURRICULUM[CurriculumLevel.LEVEL_1_BASIC].challenges[0]["target_accuracy"]
        self.cs.update_progress(CurriculumLevel.LEVEL_1_BASIC, 0, target + 0.01)
        self.assertGreaterEqual(
            self.cs.progress[CurriculumLevel.LEVEL_1_BASIC].challenges_completed, 1)

    def test_update_progress_does_not_increment_below_target(self):
        """update_progress should NOT increment completed below target accuracy."""
        self.cs.update_progress(CurriculumLevel.LEVEL_1_BASIC, 0, 0.01)
        self.assertEqual(
            self.cs.progress[CurriculumLevel.LEVEL_1_BASIC].challenges_completed, 0)

    def test_get_challenge_target_valid(self):
        """get_challenge_target should return the challenge's target_accuracy."""
        level = CurriculumLevel.LEVEL_1_BASIC
        expected = self.cs.CURRICULUM[level].challenges[0]["target_accuracy"]
        actual = self.cs.get_challenge_target(level, 0)
        self.assertAlmostEqual(actual, expected)

    def test_get_challenge_target_out_of_range(self):
        """get_challenge_target with invalid index should return 0.7 default."""
        actual = self.cs.get_challenge_target(CurriculumLevel.LEVEL_1_BASIC, 999)
        self.assertAlmostEqual(actual, 0.7)

    def test_skip_challenge_records_zero_accuracy(self):
        """skip_challenge should record 0% accuracy and advance the index."""
        old_index = self.cs.progress[CurriculumLevel.LEVEL_1_BASIC].current_challenge_index
        result = self.cs.skip_challenge()
        self.assertTrue(result)
        self.assertEqual(len(self.cs.results), 1)
        self.assertAlmostEqual(self.cs.results[0].accuracy, 0.0)
        self.assertFalse(self.cs.results[0].passed)
        # Index should advance (skip records a fail which does NOT increment,
        # but skip_challenge itself does increment afterwards)
        new_index = self.cs.progress[CurriculumLevel.LEVEL_1_BASIC].current_challenge_index
        self.assertEqual(new_index, old_index + 1)

    def test_skip_challenge_returns_false_when_no_challenge(self):
        """skip_challenge should return False when there are no more challenges."""
        total = self.cs.progress[CurriculumLevel.LEVEL_1_BASIC].challenges_total
        self.cs.progress[CurriculumLevel.LEVEL_1_BASIC].current_challenge_index = total
        result = self.cs.skip_challenge()
        self.assertFalse(result)

    def test_generate_challenge_data_returns_arrays(self):
        """generate_challenge_data should return numpy arrays."""
        samples, labels = self.cs.generate_challenge_data(
            CurriculumLevel.LEVEL_1_BASIC, 0)
        self.assertIsNotNone(samples)
        self.assertIsNotNone(labels)
        self.assertGreater(len(samples), 0)
        self.assertGreater(len(labels), 0)

    def test_generate_challenge_data_invalid_index(self):
        """generate_challenge_data with invalid index should return (None, None)."""
        samples, labels = self.cs.generate_challenge_data(
            CurriculumLevel.LEVEL_1_BASIC, 999)
        self.assertIsNone(samples)
        self.assertIsNone(labels)


# =========================================================================
# Difficulty scaling
# =========================================================================

class TestDifficultyScaling(unittest.TestCase):
    """Tests for difficulty values across the curriculum."""

    def test_level_1_challenges_have_low_difficulty(self):
        """Level 1 challenges should have difficulty <= 0.5."""
        level_info = CurriculumSystem.CURRICULUM[CurriculumLevel.LEVEL_1_BASIC]
        for ch in level_info.challenges:
            self.assertLessEqual(ch["difficulty"], 0.5,
                                 f"Challenge '{ch['name']}' difficulty too high for level 1")

    def test_level_5_challenges_have_high_difficulty(self):
        """Level 5 challenges should have difficulty >= 0.7."""
        level_info = CurriculumSystem.CURRICULUM[CurriculumLevel.LEVEL_5_EXPERT]
        for ch in level_info.challenges:
            self.assertGreaterEqual(ch["difficulty"], 0.7,
                                    f"Challenge '{ch['name']}' difficulty too low for level 5")

    def test_difficulty_generally_increases_across_levels(self):
        """Average difficulty per level should be non-decreasing."""
        avg_difficulties = []
        for level in CurriculumLevel:
            level_info = CurriculumSystem.CURRICULUM[level]
            difficulties = [ch["difficulty"] for ch in level_info.challenges]
            avg_difficulties.append(sum(difficulties) / len(difficulties))

        for i in range(1, len(avg_difficulties)):
            self.assertGreaterEqual(
                avg_difficulties[i], avg_difficulties[i - 1],
                f"Average difficulty decreased from level {i} to level {i + 1}")

    def test_all_difficulties_in_valid_range(self):
        """Every challenge difficulty should be in [0.0, 1.0]."""
        for level in CurriculumLevel:
            level_info = CurriculumSystem.CURRICULUM[level]
            for ch in level_info.challenges:
                self.assertGreaterEqual(ch["difficulty"], 0.0)
                self.assertLessEqual(ch["difficulty"], 1.0)

    def test_all_target_accuracies_in_valid_range(self):
        """Every challenge target_accuracy should be in (0.0, 1.0]."""
        for level in CurriculumLevel:
            level_info = CurriculumSystem.CURRICULUM[level]
            for ch in level_info.challenges:
                target = ch["target_accuracy"]
                self.assertGreater(target, 0.0)
                self.assertLessEqual(target, 1.0)

    def test_unlock_thresholds_increase_for_higher_levels(self):
        """Unlock thresholds should be non-decreasing across levels."""
        thresholds = []
        for level in CurriculumLevel:
            thresholds.append(CurriculumSystem.CURRICULUM[level].unlock_threshold)

        for i in range(1, len(thresholds)):
            self.assertGreaterEqual(
                thresholds[i], thresholds[i - 1],
                f"Unlock threshold decreased from level {i} to level {i + 1}")

    def test_each_level_has_at_least_one_challenge(self):
        """Every level must have at least one challenge."""
        for level in CurriculumLevel:
            level_info = CurriculumSystem.CURRICULUM[level]
            self.assertGreater(len(level_info.challenges), 0,
                               f"Level {level.name} has no challenges")

    def test_challenge_names_unique_within_level(self):
        """Challenge names within each level should be unique."""
        for level in CurriculumLevel:
            level_info = CurriculumSystem.CURRICULUM[level]
            names = [ch["name"] for ch in level_info.challenges]
            self.assertEqual(len(names), len(set(names)),
                             f"Duplicate challenge names in {level.name}")


# =========================================================================
# get_stats (serialization / state)
# =========================================================================

class TestGetStats(unittest.TestCase):
    """Tests for the get_stats method (serialization)."""

    def setUp(self):
        self.cs = CurriculumSystem(state_dim=64)

    def test_returns_dict(self):
        """get_stats should return a dictionary."""
        stats = self.cs.get_stats()
        self.assertIsInstance(stats, dict)

    def test_has_required_top_level_keys(self):
        """get_stats should have all expected top-level keys."""
        stats = self.cs.get_stats()
        expected_keys = [
            "current_level",
            "current_level_name",
            "total_challenges",
            "completed_challenges",
            "overall_progress",
            "levels",
            "total_attempts",
        ]
        for key in expected_keys:
            self.assertIn(key, stats, f"Missing key: {key}")

    def test_initial_stats_values(self):
        """get_stats on a fresh system should reflect initial state."""
        stats = self.cs.get_stats()
        self.assertEqual(stats["current_level"], 1)
        self.assertEqual(stats["current_level_name"], "Learn to Draw")
        self.assertEqual(stats["completed_challenges"], 0)
        self.assertAlmostEqual(stats["overall_progress"], 0.0)
        self.assertEqual(stats["total_attempts"], 0)

    def test_total_challenges_matches_curriculum(self):
        """total_challenges should equal sum of all levels' challenge counts."""
        stats = self.cs.get_stats()
        expected = sum(len(info.challenges)
                       for info in CurriculumSystem.CURRICULUM.values())
        self.assertEqual(stats["total_challenges"], expected)

    def test_levels_dict_has_all_levels(self):
        """The 'levels' dict should contain entries for all 5 levels."""
        stats = self.cs.get_stats()
        self.assertEqual(len(stats["levels"]), 5)
        for val in range(1, 6):
            self.assertIn(val, stats["levels"])

    def test_level_stats_keys(self):
        """Each level entry should have the expected sub-keys."""
        stats = self.cs.get_stats()
        expected_sub_keys = [
            "name", "unlocked", "completed", "challenges_completed",
            "challenges_total", "average_accuracy", "progress_percent",
        ]
        for val in range(1, 6):
            level_stat = stats["levels"][val]
            for key in expected_sub_keys:
                self.assertIn(key, level_stat,
                              f"Level {val} missing key: {key}")

    def test_stats_after_recording_results(self):
        """get_stats should reflect recorded results."""
        self.cs.record_result(_make_passing_result(
            CurriculumLevel.LEVEL_1_BASIC, accuracy=0.90))
        self.cs.record_result(_make_failing_result(
            CurriculumLevel.LEVEL_1_BASIC, accuracy=0.40))

        stats = self.cs.get_stats()
        self.assertEqual(stats["total_attempts"], 2)
        self.assertEqual(stats["completed_challenges"], 1)
        self.assertGreater(stats["overall_progress"], 0.0)

        level_1_stats = stats["levels"][1]
        self.assertEqual(level_1_stats["challenges_completed"], 1)
        self.assertAlmostEqual(level_1_stats["average_accuracy"], 0.65)

    def test_stats_after_full_level_completion(self):
        """get_stats should show a completed level correctly."""
        _complete_level(self.cs, CurriculumLevel.LEVEL_1_BASIC)
        stats = self.cs.get_stats()

        level_1_stats = stats["levels"][1]
        self.assertTrue(level_1_stats["completed"])
        self.assertAlmostEqual(level_1_stats["progress_percent"], 1.0)
        self.assertTrue(level_1_stats["unlocked"])

        # Level 2 should now be unlocked
        level_2_stats = stats["levels"][2]
        self.assertTrue(level_2_stats["unlocked"])

    def test_overall_progress_formula(self):
        """overall_progress should be completed_challenges / total_challenges."""
        self.cs.record_result(_make_passing_result(CurriculumLevel.LEVEL_1_BASIC))
        stats = self.cs.get_stats()
        expected = stats["completed_challenges"] / stats["total_challenges"]
        self.assertAlmostEqual(stats["overall_progress"], expected)


# =========================================================================
# Reset progress
# =========================================================================

class TestResetProgress(unittest.TestCase):
    """Tests for the reset_progress method."""

    def test_reset_clears_results(self):
        """reset_progress should clear the results list."""
        cs = CurriculumSystem(state_dim=64)
        cs.record_result(_make_passing_result(CurriculumLevel.LEVEL_1_BASIC))
        cs.reset_progress()
        self.assertEqual(len(cs.results), 0)

    def test_reset_returns_to_level_1(self):
        """reset_progress should set current_level back to LEVEL_1_BASIC."""
        cs = CurriculumSystem(state_dim=64)
        cs.progress[CurriculumLevel.LEVEL_2_ASSOCIATION].unlocked = True
        cs.advance_to_next_level()
        cs.reset_progress()
        self.assertEqual(cs.current_level, CurriculumLevel.LEVEL_1_BASIC)

    def test_reset_locks_higher_levels(self):
        """reset_progress should lock all levels above 1."""
        cs = CurriculumSystem(state_dim=64)
        _complete_level(cs, CurriculumLevel.LEVEL_1_BASIC)
        cs.reset_progress()
        self.assertTrue(cs.progress[CurriculumLevel.LEVEL_1_BASIC].unlocked)
        for level in [CurriculumLevel.LEVEL_2_ASSOCIATION,
                      CurriculumLevel.LEVEL_3_INTERMEDIATE,
                      CurriculumLevel.LEVEL_4_ADVANCED,
                      CurriculumLevel.LEVEL_5_EXPERT]:
            self.assertFalse(cs.progress[level].unlocked)

    def test_reset_zeroes_progress(self):
        """reset_progress should zero out all challenges_completed."""
        cs = CurriculumSystem(state_dim=64)
        _complete_level(cs, CurriculumLevel.LEVEL_1_BASIC)
        cs.reset_progress()
        for level in CurriculumLevel:
            self.assertEqual(cs.progress[level].challenges_completed, 0)
            self.assertEqual(cs.progress[level].current_challenge_index, 0)
            self.assertEqual(cs.progress[level].accuracies, [])
            self.assertFalse(cs.progress[level].completed)


# =========================================================================
# Canvas target image generation
# =========================================================================

class TestGetTargetImage(unittest.TestCase):
    """Tests for get_target_image returning canvas images."""

    def setUp(self):
        self.cs = CurriculumSystem(state_dim=64)

    def test_solid_color_returns_correct_shape(self):
        """Solid color target should be a 512x512x3 array."""
        ch = {"data_generator": "canvas_solid_color", "target_color": [255, 0, 0]}
        img = self.cs.get_target_image(ch)
        self.assertEqual(img.shape, (512, 512, 3))

    def test_solid_color_fills_correctly(self):
        """Solid red target should have R=255, G=0, B=0 everywhere."""
        import numpy as np
        ch = {"data_generator": "canvas_solid_color", "target_color": [255, 0, 0]}
        img = self.cs.get_target_image(ch)
        self.assertTrue(np.all(img[:, :, 0] == 255))
        self.assertTrue(np.all(img[:, :, 1] == 0))
        self.assertTrue(np.all(img[:, :, 2] == 0))

    def test_gradient_returns_correct_shape(self):
        """Gradient target should be a 512x512x3 array."""
        ch = {"data_generator": "canvas_gradient", "gradient_type": "horizontal"}
        img = self.cs.get_target_image(ch)
        self.assertEqual(img.shape, (512, 512, 3))

    def test_shape_returns_correct_shape(self):
        """Shape target should be a 512x512x3 array."""
        ch = {"data_generator": "canvas_shape", "shape": "circle"}
        img = self.cs.get_target_image(ch)
        self.assertEqual(img.shape, (512, 512, 3))

    def test_pattern_returns_correct_shape(self):
        """Pattern target should be a 512x512x3 array."""
        ch = {"data_generator": "canvas_pattern", "pattern": "checkerboard"}
        img = self.cs.get_target_image(ch)
        self.assertEqual(img.shape, (512, 512, 3))

    def test_unknown_generator_returns_black(self):
        """Unknown generator should return an all-black canvas."""
        import numpy as np
        ch = {"data_generator": "unknown_type"}
        img = self.cs.get_target_image(ch)
        self.assertEqual(img.shape, (512, 512, 3))
        self.assertTrue(np.all(img == 0))

    def test_missing_generator_returns_black(self):
        """Missing data_generator key should return an all-black canvas."""
        import numpy as np
        ch = {}
        img = self.cs.get_target_image(ch)
        self.assertEqual(img.shape, (512, 512, 3))
        self.assertTrue(np.all(img == 0))


# =========================================================================
# End-to-end progression scenario
# =========================================================================

class TestEndToEndProgression(unittest.TestCase):
    """Integration test: progress through multiple levels."""

    def test_progress_through_first_two_levels(self):
        """Complete level 1, unlock level 2, advance, and verify state."""
        cs = CurriculumSystem(state_dim=64)

        # Level 1 should be current and unlocked
        self.assertEqual(cs.current_level, CurriculumLevel.LEVEL_1_BASIC)
        self.assertTrue(cs.is_level_unlocked(CurriculumLevel.LEVEL_1_BASIC))
        self.assertFalse(cs.is_level_unlocked(CurriculumLevel.LEVEL_2_ASSOCIATION))

        # Complete level 1 with high accuracy
        _complete_level(cs, CurriculumLevel.LEVEL_1_BASIC, accuracy=0.95)

        # Level 2 should now be unlocked
        self.assertTrue(cs.is_level_unlocked(CurriculumLevel.LEVEL_2_ASSOCIATION))
        self.assertTrue(cs.progress[CurriculumLevel.LEVEL_1_BASIC].completed)

        # Advance to level 2
        advanced = cs.advance_to_next_level()
        self.assertTrue(advanced)
        self.assertEqual(cs.current_level, CurriculumLevel.LEVEL_2_ASSOCIATION)

        # Level 2 current challenge should be the first of level 2
        ch = cs.get_current_challenge()
        self.assertIsNotNone(ch)
        first_l2_challenge = cs.CURRICULUM[CurriculumLevel.LEVEL_2_ASSOCIATION].challenges[0]
        self.assertEqual(ch["name"], first_l2_challenge["name"])

        # Stats should reflect the progress
        stats = cs.get_stats()
        self.assertEqual(stats["current_level"], 2)
        level_1_stats = stats["levels"][1]
        self.assertTrue(level_1_stats["completed"])
        self.assertAlmostEqual(level_1_stats["progress_percent"], 1.0)


if __name__ == "__main__":
    unittest.main()
