"""
Comprehensive tests for the MetaCognitiveMonitor.

Tests cover learning episode tracking, confusion detection, stuck detection,
strategy adaptation, and meta-cognitive assessment.
"""

import os
import sys
import unittest
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.meta_cognition import (
    MetaCognitiveMonitor,
    CognitiveState,
    LearningStrategy,
    UncertaintyType,
    LearningEpisode,
    StrategyEffectiveness,
    ConfusionEvent,
    MetaCognitiveAssessment,
)


class TestMetaCognitiveMonitorInitialization(unittest.TestCase):
    """Tests for MetaCognitiveMonitor initialization."""

    def setUp(self):
        self.mcm = MetaCognitiveMonitor(random_seed=42)

    def test_default_initialization(self):
        """Test that default construction sets expected values."""
        self.assertAlmostEqual(self.mcm.confusion_threshold, 0.7)
        self.assertAlmostEqual(self.mcm.stuck_threshold, 0.1)
        self.assertAlmostEqual(self.mcm.adaptation_rate, 0.1)
        self.assertEqual(self.mcm.window_size, 100)
        self.assertTrue(self.mcm.enable_auto_adaptation)

    def test_custom_initialization(self):
        """Test construction with custom parameters."""
        mcm = MetaCognitiveMonitor(
            confusion_threshold=0.8,
            stuck_threshold=0.05,
            adaptation_rate=0.2,
            window_size=50,
            enable_auto_adaptation=False,
            random_seed=99
        )
        self.assertAlmostEqual(mcm.confusion_threshold, 0.8)
        self.assertAlmostEqual(mcm.stuck_threshold, 0.05)
        self.assertFalse(mcm.enable_auto_adaptation)

    def test_initial_state(self):
        """Test initial cognitive state."""
        self.assertEqual(self.mcm.current_state, CognitiveState.FOCUSED)

    def test_initial_histories_empty(self):
        """Test that history buffers start empty."""
        self.assertEqual(len(self.mcm.performance_history), 0)
        self.assertEqual(len(self.mcm.confidence_history), 0)
        self.assertEqual(len(self.mcm.error_history), 0)

    def test_initial_statistics_zero(self):
        """Test that statistics start at zero."""
        self.assertEqual(self.mcm.total_episodes, 0)
        self.assertEqual(self.mcm.total_confusion_events, 0)
        self.assertEqual(self.mcm.total_stuck_events, 0)
        self.assertEqual(self.mcm.strategy_adaptations, 0)


class TestLearningEpisodeManagement(unittest.TestCase):
    """Tests for learning episode management."""

    def setUp(self):
        self.mcm = MetaCognitiveMonitor(random_seed=42)

    def test_start_learning_episode(self):
        """Test starting a learning episode."""
        episode_id = self.mcm.start_learning_episode(
            task_id="task_1",
            strategy=LearningStrategy.DEEP_PRACTICE,
            initial_performance=0.2,
            initial_confidence=0.3
        )
        
        self.assertIsNotNone(episode_id)
        self.assertIn(episode_id, self.mcm.learning_episodes)
        self.assertEqual(self.mcm.active_episode, episode_id)
        
        episode = self.mcm.learning_episodes[episode_id]
        self.assertEqual(episode.task_id, "task_1")
        self.assertEqual(episode.strategy, LearningStrategy.DEEP_PRACTICE)
        self.assertAlmostEqual(episode.initial_performance, 0.2)

    def test_update_episode_progress(self):
        """Test updating episode progress."""
        episode_id = self.mcm.start_learning_episode(
            task_id="task_1",
            strategy=LearningStrategy.DEEP_PRACTICE
        )
        
        self.mcm.update_episode_progress(
            performance=0.4,
            confidence=0.5,
            error=0.2
        )
        
        episode = self.mcm.learning_episodes[episode_id]
        self.assertEqual(episode.attempts, 1)
        self.assertAlmostEqual(episode.final_performance, 0.4)
        
        # Check history updated
        self.assertEqual(len(self.mcm.performance_history), 1)
        self.assertEqual(len(self.mcm.confidence_history), 1)
        self.assertEqual(len(self.mcm.error_history), 1)

    def test_finish_learning_episode(self):
        """Test finishing a learning episode."""
        episode_id = self.mcm.start_learning_episode(
            task_id="task_1",
            strategy=LearningStrategy.DEEP_PRACTICE,
            initial_performance=0.2
        )
        
        self.mcm.update_episode_progress(performance=0.6, confidence=0.7)
        result = self.mcm.finish_learning_episode(success=True)
        
        self.assertTrue(result['success'])
        self.assertAlmostEqual(result['improvement'], 0.4, places=5)
        
        episode = self.mcm.learning_episodes[episode_id]
        self.assertTrue(episode.completed)
        self.assertTrue(episode.success)

    def test_strategy_effectiveness_tracking(self):
        """Test that strategy effectiveness is tracked."""
        episode_id = self.mcm.start_learning_episode(
            task_id="task_1",
            strategy=LearningStrategy.DEEP_PRACTICE,
            initial_performance=0.2
        )
        
        self.mcm.update_episode_progress(performance=0.6, confidence=0.7)
        self.mcm.finish_learning_episode(success=True)
        
        effectiveness = self.mcm.strategy_effectiveness[LearningStrategy.DEEP_PRACTICE]
        self.assertEqual(effectiveness.usage_count, 1)
        self.assertEqual(effectiveness.success_count, 1)


class TestConfusionDetection(unittest.TestCase):
    """Tests for confusion detection."""

    def setUp(self):
        # Use lower threshold for testing
        self.mcm = MetaCognitiveMonitor(confusion_threshold=0.5, random_seed=42)
        self.mcm.start_learning_episode(
            task_id="task_1",
            strategy=LearningStrategy.DEEP_PRACTICE
        )

    def test_confusion_detected_low_confidence(self):
        """Test confusion detection with low confidence."""
        # Update with very low confidence and high error to trigger confusion
        for _ in range(3):  # Multiple updates to build up confusion score
            self.mcm.update_episode_progress(
                performance=0.3,
                confidence=0.1,  # Very low confidence
                error=0.8  # High error
            )
        
        # Should have detected confusion
        self.assertIsNotNone(self.mcm.current_confusion)
        self.assertEqual(self.mcm.current_state, CognitiveState.CONFUSED)

    def test_confusion_not_detected_high_confidence(self):
        """Test that high confidence doesn't trigger confusion."""
        self.mcm.update_episode_progress(
            performance=0.8,
            confidence=0.9,  # High confidence
            error=0.1
        )
        
        self.assertIsNone(self.mcm.current_confusion)

    def test_resolve_confusion(self):
        """Test resolving confusion."""
        # First trigger confusion with multiple updates
        for _ in range(3):
            self.mcm.update_episode_progress(
                performance=0.3,
                confidence=0.1,
                error=0.8
            )
        
        self.assertIsNotNone(self.mcm.current_confusion)
        
        # Resolve it
        self.mcm.resolve_confusion("Reviewed fundamentals")
        
        self.assertIsNone(self.mcm.current_confusion)
        self.assertEqual(self.mcm.current_state, CognitiveState.ADAPTING)

    def test_confusion_event_recorded(self):
        """Test that confusion events are recorded."""
        # Trigger confusion with multiple updates
        for _ in range(3):
            self.mcm.update_episode_progress(
                performance=0.3,
                confidence=0.1,
                error=0.8
            )
        
        self.assertEqual(len(self.mcm.confusion_events), 1)
        self.assertEqual(self.mcm.total_confusion_events, 1)


class TestStuckDetection(unittest.TestCase):
    """Tests for stuck detection."""

    def setUp(self):
        self.mcm = MetaCognitiveMonitor(random_seed=42)
        self.mcm.start_learning_episode(
            task_id="task_1",
            strategy=LearningStrategy.DEEP_PRACTICE
        )

    def test_stuck_detected_no_improvement(self):
        """Test stuck detection with no improvement."""
        # Add many updates with no improvement
        for _ in range(15):
            self.mcm.update_episode_progress(
                performance=0.3,  # No improvement
                confidence=0.5
            )
        
        # Should detect stuck
        self.assertTrue(self.mcm.is_stuck())

    def test_not_stuck_with_improvement(self):
        """Test that improvement prevents stuck detection."""
        # Add updates with strong improvement
        for i in range(15):
            self.mcm.update_episode_progress(
                performance=0.3 + i * 0.2,  # Very strong improvement
                confidence=0.5 + i * 0.03
            )
        
        self.assertFalse(self.mcm.is_stuck())

    def test_stuck_detected_oscillation(self):
        """Test stuck detection with oscillating performance."""
        # Add oscillating updates
        for i in range(15):
            perf = 0.5 + 0.1 * ((-1) ** i)  # Oscillate
            self.mcm.update_episode_progress(performance=perf, confidence=0.5)
        
        self.assertTrue(self.mcm.is_stuck())


class TestStrategyAdaptation(unittest.TestCase):
    """Tests for strategy adaptation."""

    def setUp(self):
        self.mcm = MetaCognitiveMonitor(random_seed=42)

    def test_recommend_strategy_default(self):
        """Test strategy recommendation."""
        strategy = self.mcm.recommend_strategy()
        
        self.assertIsInstance(strategy, LearningStrategy)

    def test_recommend_strategy_when_confused(self):
        """Test that simplification is recommended when confused."""
        # Use monitor with lower confusion threshold
        mcm = MetaCognitiveMonitor(confusion_threshold=0.5, random_seed=42)
        mcm.start_learning_episode(
            task_id="task_1",
            strategy=LearningStrategy.DEEP_PRACTICE
        )
        
        # Trigger confusion with multiple updates
        for _ in range(3):
            mcm.update_episode_progress(
                performance=0.3,
                confidence=0.1,
                error=0.9
            )
        
        strategy = mcm.recommend_strategy()
        self.assertEqual(strategy, LearningStrategy.SIMPLIFICATION)

    def test_recommend_strategy_when_stuck(self):
        """Test that analogy is recommended when stuck."""
        self.mcm.start_learning_episode(
            task_id="task_1",
            strategy=LearningStrategy.DEEP_PRACTICE
        )
        
        # Trigger stuck
        for _ in range(15):
            self.mcm.update_episode_progress(performance=0.3, confidence=0.5)
        
        strategy = self.mcm.recommend_strategy()
        self.assertEqual(strategy, LearningStrategy.ANALOGY)

    def test_adapt_strategy(self):
        """Test strategy adaptation."""
        self.mcm.start_learning_episode(
            task_id="task_1",
            strategy=LearningStrategy.DEEP_PRACTICE
        )
        
        # Trigger stuck
        for _ in range(15):
            self.mcm.update_episode_progress(performance=0.3, confidence=0.5)
        
        new_strategy = self.mcm.adapt_strategy()
        
        self.assertIsNotNone(new_strategy)
        self.assertEqual(self.mcm.current_state, CognitiveState.ADAPTING)
        self.assertEqual(self.mcm.strategy_adaptations, 1)

    def test_adapt_strategy_forced(self):
        """Test forced strategy adaptation."""
        new_strategy = self.mcm.adapt_strategy(forced=True)
        
        self.assertIsNotNone(new_strategy)

    def test_no_adaptation_when_not_needed(self):
        """Test that adaptation doesn't happen when not needed."""
        self.mcm.enable_auto_adaptation = True
        
        # Don't trigger any issues
        result = self.mcm.adapt_strategy()
        
        self.assertIsNone(result)


class TestMetaCognitiveAssessment(unittest.TestCase):
    """Tests for meta-cognitive assessment."""

    def setUp(self):
        self.mcm = MetaCognitiveMonitor(random_seed=42)

    def test_assess_metacognition(self):
        """Test comprehensive meta-cognitive assessment."""
        assessment = self.mcm.assess_metacognition()
        
        self.assertIsInstance(assessment, MetaCognitiveAssessment)
        self.assertIn(assessment.cognitive_state, CognitiveState)
        self.assertIsNotNone(assessment.recommended_strategy)
        self.assertIsNotNone(assessment.suggested_action)

    def test_assessment_includes_confidence(self):
        """Test that assessment includes confidence."""
        # Add some confidence history
        self.mcm.start_learning_episode(task_id="task_1", strategy=LearningStrategy.DEEP_PRACTICE)
        for _ in range(5):
            self.mcm.update_episode_progress(performance=0.5, confidence=0.7)
        
        assessment = self.mcm.assess_metacognition()
        
        self.assertGreater(assessment.overall_confidence, 0.0)
        self.assertLessEqual(assessment.overall_confidence, 1.0)

    def test_assessment_detects_needs_intervention(self):
        """Test that assessment detects when intervention is needed."""
        # Use monitor with lower confusion threshold
        mcm = MetaCognitiveMonitor(confusion_threshold=0.5, random_seed=42)
        mcm.start_learning_episode(task_id="task_1", strategy=LearningStrategy.DEEP_PRACTICE)
        
        # Trigger confusion with multiple updates
        for _ in range(5):
            mcm.update_episode_progress(
                performance=0.2,
                confidence=0.05,  # Very low confidence
                error=0.95  # Very high error
            )
        
        assessment = mcm.assess_metacognition()
        
        self.assertTrue(assessment.needs_intervention)
        self.assertTrue(assessment.is_confused)

    def test_calculate_learning_rate(self):
        """Test learning rate calculation."""
        self.mcm.start_learning_episode(task_id="task_1", strategy=LearningStrategy.DEEP_PRACTICE)
        
        # Add improving performance
        for i in range(10):
            self.mcm.update_episode_progress(
                performance=0.2 + i * 0.05,
                confidence=0.5
            )
        
        rate = self.mcm._calculate_learning_rate()
        self.assertGreater(rate, 0.0)  # Should show improvement

    def test_calculate_efficiency(self):
        """Test efficiency calculation."""
        # Start and complete an episode
        self.mcm.start_learning_episode(
            task_id="task_1",
            strategy=LearningStrategy.DEEP_PRACTICE,
            initial_performance=0.2
        )
        self.mcm.update_episode_progress(performance=0.6, confidence=0.7)
        self.mcm.finish_learning_episode(success=True)
        
        efficiency = self.mcm._calculate_efficiency()
        self.assertGreaterEqual(efficiency, 0.0)
        self.assertLessEqual(efficiency, 1.0)


class TestStrategyEffectiveness(unittest.TestCase):
    """Tests for StrategyEffectiveness tracking."""

    def setUp(self):
        self.se = StrategyEffectiveness(strategy=LearningStrategy.DEEP_PRACTICE)

    def test_record_outcome(self):
        """Test recording strategy outcome."""
        self.se.record_outcome(
            success=True,
            improvement=0.3,
            time_spent=10.0,
            confused=False
        )
        
        self.assertEqual(self.se.usage_count, 1)
        self.assertEqual(self.se.success_count, 1)
        self.assertAlmostEqual(self.se.success_rate, 1.0)

    def test_success_rate_calculation(self):
        """Test success rate calculation."""
        self.se.record_outcome(success=True, improvement=0.2, time_spent=5.0, confused=False)
        self.se.record_outcome(success=True, improvement=0.3, time_spent=5.0, confused=False)
        self.se.record_outcome(success=False, improvement=0.0, time_spent=5.0, confused=True)
        
        self.assertAlmostEqual(self.se.success_rate, 2/3)

    def test_efficiency_calculation(self):
        """Test efficiency calculation."""
        self.se.record_outcome(success=True, improvement=0.3, time_spent=10.0, confused=False)
        
        efficiency = self.se.efficiency
        self.assertGreater(efficiency, 0.0)


class TestUtilityMethods(unittest.TestCase):
    """Tests for utility methods."""

    def setUp(self):
        self.mcm = MetaCognitiveMonitor(random_seed=42)

    def test_get_strategy_rankings(self):
        """Test getting strategy rankings."""
        # Record some outcomes
        self.mcm.start_learning_episode(task_id="task_1", strategy=LearningStrategy.DEEP_PRACTICE)
        self.mcm.update_episode_progress(performance=0.8, confidence=0.8)
        self.mcm.finish_learning_episode(success=True)
        
        rankings = self.mcm.get_strategy_rankings()
        
        self.assertIsInstance(rankings, list)
        self.assertGreater(len(rankings), 0)
        
        # Check sorted by efficiency
        for i in range(len(rankings) - 1):
            self.assertGreaterEqual(rankings[i][1], rankings[i+1][1])

    def test_get_learning_summary(self):
        """Test getting learning summary."""
        # Create some episodes
        for i in range(3):
            self.mcm.start_learning_episode(
                task_id=f"task_{i}",
                strategy=LearningStrategy.DEEP_PRACTICE,
                initial_performance=0.2
            )
            self.mcm.update_episode_progress(performance=0.6, confidence=0.7)
            self.mcm.finish_learning_episode(success=True)
        
        summary = self.mcm.get_learning_summary()
        
        self.assertIn('total_episodes', summary)
        self.assertIn('success_rate', summary)
        self.assertIn('current_state', summary)
        self.assertEqual(summary['total_episodes'], 3)

    def test_get_stats(self):
        """Test getting statistics."""
        stats = self.mcm.get_stats()
        
        self.assertIn('total_episodes', stats)
        self.assertIn('current_state', stats)
        self.assertIn('strategy_rankings', stats)
        self.assertIn('learning_summary', stats)


class TestSerialization(unittest.TestCase):
    """Tests for serialization/deserialization."""

    def setUp(self):
        self.mcm = MetaCognitiveMonitor(random_seed=42)

    def test_serialize(self):
        """Test serialization."""
        # Add some data
        self.mcm.start_learning_episode(task_id="task_1", strategy=LearningStrategy.DEEP_PRACTICE)
        self.mcm.update_episode_progress(performance=0.6, confidence=0.7)
        self.mcm.finish_learning_episode(success=True)
        
        data = self.mcm.serialize()
        
        self.assertIn('confusion_threshold', data)
        self.assertIn('strategy_effectiveness', data)
        self.assertIn('stats', data)

    def test_deserialize(self):
        """Test deserialization."""
        # Create and populate
        self.mcm.start_learning_episode(task_id="task_1", strategy=LearningStrategy.DEEP_PRACTICE)
        self.mcm.finish_learning_episode(success=True)
        
        data = self.mcm.serialize()
        restored = MetaCognitiveMonitor.deserialize(data)
        
        self.assertEqual(restored.confusion_threshold, self.mcm.confusion_threshold)
        self.assertEqual(restored.enable_auto_adaptation, self.mcm.enable_auto_adaptation)

    def test_round_trip_preserves_strategy_effectiveness(self):
        """Test that round trip preserves strategy effectiveness data."""
        # Record outcomes
        se = self.mcm.strategy_effectiveness[LearningStrategy.DEEP_PRACTICE]
        se.usage_count = 5
        se.success_count = 4
        se.avg_improvement = 0.3
        
        data = self.mcm.serialize()
        restored = MetaCognitiveMonitor.deserialize(data)
        
        restored_se = restored.strategy_effectiveness[LearningStrategy.DEEP_PRACTICE]
        self.assertEqual(restored_se.usage_count, 5)
        self.assertEqual(restored_se.success_count, 4)
        self.assertAlmostEqual(restored_se.avg_improvement, 0.3)


if __name__ == '__main__':
    unittest.main()
