"""
Comprehensive tests for the AutonomousGoalSystem.

Tests cover goal generation, intrinsic motivation, curiosity-driven exploration,
goal management, and value learning.
"""

import os
import sys
import unittest
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.autonomous_goals import (
    AutonomousGoalSystem,
    GoalType,
    GoalStatus,
    IntrinsicDrive,
    Goal,
    DriveState,
    ExplorationFrontier,
)


class TestAutonomousGoalSystemInitialization(unittest.TestCase):
    """Tests for AutonomousGoalSystem initialization."""

    def setUp(self):
        self.ags = AutonomousGoalSystem(random_seed=42)

    def test_default_initialization(self):
        """Test that default construction sets expected values."""
        self.assertEqual(self.ags.max_active_goals, 10)
        self.assertEqual(self.ags.max_goal_queue, 50)
        self.assertAlmostEqual(self.ags.curiosity_threshold, 0.3)
        self.assertTrue(self.ags.enable_meta_goals)
        self.assertTrue(self.ags.enable_social_goals)

    def test_custom_initialization(self):
        """Test construction with custom parameters."""
        ags = AutonomousGoalSystem(
            max_active_goals=5,
            max_goal_queue=20,
            curiosity_threshold=0.5,
            enable_meta_goals=False,
            enable_social_goals=False,
            random_seed=99
        )
        self.assertEqual(ags.max_active_goals, 5)
        self.assertEqual(ags.max_goal_queue, 20)
        self.assertFalse(ags.enable_meta_goals)
        self.assertFalse(ags.enable_social_goals)

    def test_meta_goals_initialized(self):
        """Test that meta-goals are initialized by default."""
        # Should have meta-goals
        meta_goals = [g for g in self.ags.goals.values() if g.goal_type == GoalType.META]
        self.assertGreater(len(meta_goals), 0)

    def test_no_meta_goals_when_disabled(self):
        """Test that meta-goals are not created when disabled."""
        ags = AutonomousGoalSystem(enable_meta_goals=False, random_seed=42)
        meta_goals = [g for g in ags.goals.values() if g.goal_type == GoalType.META]
        self.assertEqual(len(meta_goals), 0)

    def test_initial_drives(self):
        """Test initial drive states."""
        self.assertAlmostEqual(self.ags.drives.curiosity, 1.0)
        self.assertAlmostEqual(self.ags.drives.competence, 0.5)
        self.assertAlmostEqual(self.ags.drives.autonomy, 0.5)


class TestGoalGeneration(unittest.TestCase):
    """Tests for goal generation."""

    def setUp(self):
        self.ags = AutonomousGoalSystem(enable_meta_goals=False, random_seed=42)

    def test_generate_learning_goal(self):
        """Test generating a learning goal."""
        goal_id = self.ags.generate_goal(goal_type=GoalType.LEARNING)
        
        self.assertIsNotNone(goal_id)
        self.assertIn(goal_id, self.ags.goals)
        
        goal = self.ags.goals[goal_id]
        self.assertEqual(goal.goal_type, GoalType.LEARNING)
        self.assertEqual(goal.status, GoalStatus.PROPOSED)

    def test_generate_goal_with_name(self):
        """Test generating a goal with specific name."""
        goal_id = self.ags.generate_goal(
            goal_type=GoalType.EXPLORATION,
            name="my_exploration_goal"
        )
        
        goal = self.ags.goals[goal_id]
        self.assertEqual(goal.name, "my_exploration_goal")

    def test_generate_goal_with_description(self):
        """Test generating a goal with description."""
        goal_id = self.ags.generate_goal(
            goal_type=GoalType.CREATION,
            description="Create something amazing"
        )
        
        goal = self.ags.goals[goal_id]
        self.assertEqual(goal.description, "Create something amazing")

    def test_generate_goal_with_parent(self):
        """Test generating a subgoal."""
        parent_id = self.ags.generate_goal(goal_type=GoalType.META)
        child_id = self.ags.generate_goal(
            goal_type=GoalType.LEARNING,
            parent_goal_id=parent_id
        )
        
        parent = self.ags.goals[parent_id]
        self.assertIn(child_id, parent.subgoals)
        
        child = self.ags.goals[child_id]
        self.assertEqual(child.parent_goal, parent_id)

    def test_generate_goal_queue_full(self):
        """Test that goal generation fails when queue is full."""
        ags = AutonomousGoalSystem(
            max_goal_queue=2,
            enable_meta_goals=False,
            random_seed=42
        )
        
        ags.generate_goal(goal_type=GoalType.LEARNING)
        ags.generate_goal(goal_type=GoalType.LEARNING)
        
        # Queue should be full now
        result = ags.generate_goal(goal_type=GoalType.LEARNING)
        self.assertIsNone(result)

    def test_goal_priority_computation(self):
        """Test goal priority computation."""
        goal_id = self.ags.generate_goal(
            goal_type=GoalType.LEARNING,
            intrinsic_value=0.8
        )
        
        goal = self.ags.goals[goal_id]
        priority = goal.compute_priority()
        
        self.assertGreater(priority, 0.0)
        self.assertLessEqual(priority, 1.0)


class TestDriveDynamics(unittest.TestCase):
    """Tests for intrinsic drive dynamics."""

    def setUp(self):
        self.ags = AutonomousGoalSystem(enable_meta_goals=False, random_seed=42)

    def test_drive_update_success(self):
        """Test drive update after successful goal."""
        initial_competence = self.ags.drives.competence
        
        self.ags.drives.update(goal_achieved=True, novelty_encountered=0.5)
        
        self.assertGreater(self.ags.drives.competence, initial_competence)

    def test_drive_update_failure(self):
        """Test drive update after failed goal."""
        initial_competence = self.ags.drives.competence
        
        self.ags.drives.update(goal_achieved=False, novelty_encountered=0.0)
        
        self.assertLess(self.ags.drives.competence, initial_competence)

    def test_curiosity_decay(self):
        """Test that curiosity decays over time."""
        initial_curiosity = self.ags.drives.curiosity
        
        # Update with no novelty
        self.ags.drives.update(goal_achieved=True, novelty_encountered=0.0)
        
        self.assertLess(self.ags.drives.curiosity, initial_curiosity)

    def test_curiosity_recovery(self):
        """Test that curiosity recovers with novelty."""
        # First let it decay
        for _ in range(5):
            self.ags.drives.update(goal_achieved=True, novelty_encountered=0.0)
        
        decayed_curiosity = self.ags.drives.curiosity
        
        # Now recover
        self.ags.drives.update(goal_achieved=True, novelty_encountered=1.0)
        
        self.assertGreater(self.ags.drives.curiosity, decayed_curiosity)

    def test_get_dominant_drive(self):
        """Test getting dominant drive."""
        # Set curiosity high
        self.ags.drives.curiosity = 1.0
        self.ags.drives.competence = 0.3
        
        dominant = self.ags.drives.get_dominant_drive()
        self.assertEqual(dominant, IntrinsicDrive.CURIOSITY)


class TestExploration(unittest.TestCase):
    """Tests for curiosity-driven exploration."""

    def setUp(self):
        self.ags = AutonomousGoalSystem(enable_meta_goals=False, random_seed=42)

    def test_add_exploration_frontier(self):
        """Test adding an exploration frontier."""
        frontier_id = self.ags.add_exploration_frontier(
            domain="test_domain",
            novelty_score=0.8,
            predicted_difficulty=0.5,
            estimated_reward=0.7
        )
        
        self.assertIsNotNone(frontier_id)
        self.assertIn(frontier_id, self.ags.exploration_frontiers)

    def test_add_frontier_generates_goal(self):
        """Test that high novelty frontier generates curiosity goal."""
        initial_goals = len(self.ags.goals)
        
        self.ags.add_exploration_frontier(
            domain="novel_domain",
            novelty_score=0.9,  # Above threshold
            predicted_difficulty=0.5,
            estimated_reward=0.7
        )
        
        # Should have generated a curiosity goal
        curiosity_goals = [g for g in self.ags.goals.values() if g.goal_type == GoalType.CURIOSITY]
        self.assertGreater(len(curiosity_goals), 0)

    def test_select_exploration_target(self):
        """Test selecting exploration target."""
        # Add frontiers
        self.ags.add_exploration_frontier(
            domain="domain1",
            novelty_score=0.5,
            predicted_difficulty=0.5,
            estimated_reward=0.5
        )
        self.ags.add_exploration_frontier(
            domain="domain2",
            novelty_score=0.9,
            predicted_difficulty=0.5,
            estimated_reward=0.5
        )
        
        target = self.ags.select_exploration_target()
        self.assertIsNotNone(target)
        self.assertIn(target, self.ags.exploration_frontiers)

    def test_record_exploration(self):
        """Test recording exploration results."""
        frontier_id = self.ags.add_exploration_frontier(
            domain="test",
            novelty_score=0.5,
            predicted_difficulty=0.5,
            estimated_reward=0.5
        )
        
        self.ags.record_exploration(
            frontier_id=frontier_id,
            novelty_found=0.8,
            success=True
        )
        
        frontier = self.ags.exploration_frontiers[frontier_id]
        self.assertEqual(frontier.exploration_count, 1)
        self.assertIsNotNone(frontier.last_explored)


class TestGoalManagement(unittest.TestCase):
    """Tests for goal management."""

    def setUp(self):
        self.ags = AutonomousGoalSystem(enable_meta_goals=False, random_seed=42)

    def test_activate_next_goal(self):
        """Test activating the next goal."""
        # Generate a goal
        goal_id = self.ags.generate_goal(goal_type=GoalType.LEARNING)
        
        # Activate it
        activated = self.ags.activate_next_goal()
        
        self.assertEqual(activated, goal_id)
        self.assertIn(goal_id, self.ags.active_goals)
        self.assertEqual(self.ags.goals[goal_id].status, GoalStatus.ACTIVE)

    def test_activate_respects_capacity(self):
        """Test that activation respects max_active_goals."""
        ags = AutonomousGoalSystem(
            max_active_goals=1,
            enable_meta_goals=False,
            random_seed=42
        )
        
        # Generate two goals
        ags.generate_goal(goal_type=GoalType.LEARNING)
        ags.generate_goal(goal_type=GoalType.LEARNING)
        
        # Activate first
        ags.activate_next_goal()
        
        # Second should not activate (capacity reached)
        result = ags.activate_next_goal()
        self.assertIsNone(result)

    def test_update_goal_progress(self):
        """Test updating goal progress."""
        goal_id = self.ags.generate_goal(goal_type=GoalType.LEARNING)
        self.ags.activate_next_goal()
        
        self.ags.update_goal_progress(goal_id, progress=0.5)
        
        goal = self.ags.goals[goal_id]
        self.assertAlmostEqual(goal.progress, 0.5)

    def test_complete_goal_success(self):
        """Test completing a goal successfully."""
        goal_id = self.ags.generate_goal(goal_type=GoalType.LEARNING)
        self.ags.activate_next_goal()
        
        self.ags.complete_goal(goal_id, success=True)
        
        goal = self.ags.goals[goal_id]
        self.assertEqual(goal.status, GoalStatus.COMPLETED)
        self.assertIsNotNone(goal.completed_at)
        self.assertNotIn(goal_id, self.ags.active_goals)
        self.assertIn(goal_id, self.ags.completed_goals)

    def test_complete_goal_failure(self):
        """Test failing a goal."""
        goal_id = self.ags.generate_goal(goal_type=GoalType.LEARNING)
        self.ags.activate_next_goal()
        
        self.ags.complete_goal(goal_id, success=False)
        
        goal = self.ags.goals[goal_id]
        self.assertEqual(goal.status, GoalStatus.FAILED)
        self.assertIn(goal_id, self.ags.failed_goals)

    def test_abandon_goal(self):
        """Test abandoning a goal."""
        goal_id = self.ags.generate_goal(goal_type=GoalType.LEARNING)
        self.ags.activate_next_goal()
        
        self.ags.abandon_goal(goal_id, reason="No longer relevant")
        
        goal = self.ags.goals[goal_id]
        self.assertEqual(goal.status, GoalStatus.ABANDONED)
        self.assertNotIn(goal_id, self.ags.active_goals)


class TestGoalHierarchy(unittest.TestCase):
    """Tests for goal hierarchy."""

    def setUp(self):
        self.ags = AutonomousGoalSystem(enable_meta_goals=False, random_seed=42)

    def test_get_goal_hierarchy(self):
        """Test getting goal hierarchy."""
        # Create hierarchy
        parent_id = self.ags.generate_goal(goal_type=GoalType.META, name="parent")
        child1_id = self.ags.generate_goal(
            goal_type=GoalType.LEARNING,
            name="child1",
            parent_goal_id=parent_id
        )
        child2_id = self.ags.generate_goal(
            goal_type=GoalType.LEARNING,
            name="child2",
            parent_goal_id=parent_id
        )
        
        hierarchy = self.ags.get_goal_hierarchy(root_goal_id=parent_id)
        
        self.assertEqual(hierarchy['goal_id'], parent_id)
        self.assertEqual(len(hierarchy['subgoals']), 2)

    def test_parent_completion_check(self):
        """Test that parent completes when subgoals complete."""
        parent_id = self.ags.generate_goal(goal_type=GoalType.META)
        child_id = self.ags.generate_goal(
            goal_type=GoalType.LEARNING,
            parent_goal_id=parent_id
        )
        
        self.ags.activate_next_goal()  # Activate parent
        self.ags.activate_next_goal()  # Activate child
        
        # Complete child
        self.ags.complete_goal(child_id, success=True)
        
        # Check if parent completes
        parent = self.ags.goals[parent_id]
        # Should complete since all subgoals are done
        self.assertEqual(parent.status, GoalStatus.COMPLETED)


class TestAutonomousCycle(unittest.TestCase):
    """Tests for autonomous operation cycle."""

    def setUp(self):
        self.ags = AutonomousGoalSystem(enable_meta_goals=False, random_seed=42)

    def test_run_autonomous_cycle(self):
        """Test running one autonomous cycle."""
        # Set high curiosity to trigger goal generation
        self.ags.drives.curiosity = 0.9
        
        result = self.ags.run_autonomous_cycle()
        
        self.assertIn('goals_generated', result)
        self.assertIn('goals_activated', result)
        self.assertIn('drive_states', result)

    def test_cycle_generates_goals_based_on_drives(self):
        """Test that cycle generates goals based on drive states."""
        # Low competence should generate mastery goals
        self.ags.drives.competence = 0.3
        
        initial_goals = len(self.ags.goals)
        self.ags.run_autonomous_cycle()
        
        self.assertGreater(len(self.ags.goals), initial_goals)


class TestSerialization(unittest.TestCase):
    """Tests for serialization/deserialization."""

    def setUp(self):
        self.ags = AutonomousGoalSystem(enable_meta_goals=False, random_seed=42)

    def test_serialize(self):
        """Test serialization."""
        # Add some goals
        self.ags.generate_goal(goal_type=GoalType.LEARNING)
        self.ags.add_exploration_frontier("test", 0.5, 0.5, 0.5)
        
        data = self.ags.serialize()
        
        self.assertIn('max_active_goals', data)
        self.assertIn('goals', data)
        self.assertIn('exploration_frontiers', data)
        self.assertIn('stats', data)

    def test_deserialize(self):
        """Test deserialization."""
        # Create and populate
        self.ags.generate_goal(goal_type=GoalType.LEARNING, name="test_goal")
        
        data = self.ags.serialize()
        restored = AutonomousGoalSystem.deserialize(data)
        
        self.assertEqual(restored.max_active_goals, self.ags.max_active_goals)
        self.assertEqual(len(restored.goals), len(self.ags.goals))

    def test_round_trip_preserves_goals(self):
        """Test that round trip preserves goal data."""
        goal_id = self.ags.generate_goal(
            goal_type=GoalType.LEARNING,
            name="preserved_goal",
            intrinsic_value=0.8
        )
        
        data = self.ags.serialize()
        restored = AutonomousGoalSystem.deserialize(data)
        
        self.assertIn(goal_id, restored.goals)
        restored_goal = restored.goals[goal_id]
        self.assertEqual(restored_goal.name, "preserved_goal")
        self.assertAlmostEqual(restored_goal.intrinsic_value, 0.8)


class TestStats(unittest.TestCase):
    """Tests for statistics."""

    def setUp(self):
        self.ags = AutonomousGoalSystem(enable_meta_goals=False, random_seed=42)

    def test_get_stats(self):
        """Test getting statistics."""
        # Generate some activity
        self.ags.generate_goal(goal_type=GoalType.LEARNING)
        self.ags.activate_next_goal()
        
        stats = self.ags.get_stats()
        
        self.assertIn('total_goals_generated', stats)
        self.assertIn('active_goals_count', stats)
        self.assertIn('drive_states', stats)
        self.assertIn('goal_success_rates', stats)
        
        self.assertEqual(stats['total_goals_generated'], 1)
        self.assertEqual(stats['active_goals_count'], 1)


if __name__ == '__main__':
    unittest.main()
