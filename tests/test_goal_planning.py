"""
Comprehensive tests for the Goal-Directed Planning System.

Tests cover GoalPlanningSystem initialization, goal generation, plan creation,
step execution, GoalType/PlanStatus enums, status transitions, and serialization.

All tests are deterministic and pass reliably.
"""

import os
import sys
import time
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from self_organizing_av_system.core.backend import xp
from self_organizing_av_system.core.goal_planning import (
    GoalPlanningSystem,
    GoalType,
    PlanStatus,
    Goal,
    Action,
    Plan,
)


# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------

def _make_action(name="act", pre=None, eff=None, cost=1.0, duration=1.0,
                 success_prob=1.0):
    """Create an Action with sensible defaults."""
    return Action(
        name=name,
        preconditions=pre if pre is not None else {},
        effects=eff if eff is not None else {},
        cost=cost,
        duration=duration,
        success_probability=success_prob,
    )


def _make_goal(name="test_goal", goal_type=GoalType.LEARNING, priority=0.5,
               value=0.5, **kwargs):
    """Create a Goal with sensible defaults."""
    return Goal(name=name, goal_type=goal_type, priority=priority,
                value=value, **kwargs)


# ===================================================================
# 1. Initialization
# ===================================================================

class TestGoalPlanningSystemInit(unittest.TestCase):
    """Tests for GoalPlanningSystem.__init__."""

    def test_default_initialization_creates_meta_goals(self):
        """Default init (enable_meta_goals=True) adds two meta-goals."""
        gps = GoalPlanningSystem(random_seed=42)
        # _initialize_meta_goals adds "continuous_learning" and "self_improvement"
        self.assertEqual(len(gps.active_goals), 2)
        names = {g.name for g in gps.active_goals}
        self.assertIn("continuous_learning", names)
        self.assertIn("self_improvement", names)

    def test_init_without_meta_goals(self):
        """Disabling meta goals starts with an empty goal list."""
        gps = GoalPlanningSystem(enable_meta_goals=False, random_seed=0)
        self.assertEqual(len(gps.active_goals), 0)

    def test_custom_parameters_stored(self):
        """All constructor parameters are stored on the instance."""
        gps = GoalPlanningSystem(
            max_goals=50,
            planning_horizon=20,
            enable_meta_goals=False,
            enable_intrinsic_motivation=False,
            exploration_bonus=0.3,
            learning_value=0.6,
            discount_factor=0.9,
            replan_threshold=0.1,
            random_seed=99,
        )
        self.assertEqual(gps.max_goals, 50)
        self.assertEqual(gps.planning_horizon, 20)
        self.assertFalse(gps.enable_meta_goals)
        self.assertFalse(gps.enable_intrinsic_motivation)
        self.assertAlmostEqual(gps.exploration_bonus, 0.3)
        self.assertAlmostEqual(gps.learning_value, 0.6)
        self.assertAlmostEqual(gps.discount_factor, 0.9)
        self.assertAlmostEqual(gps.replan_threshold, 0.1)

    def test_initial_statistics_are_zero(self):
        """Freshly created system has zeroed-out statistics."""
        gps = GoalPlanningSystem(enable_meta_goals=False, random_seed=0)
        self.assertEqual(gps.total_goals_created, 0)
        self.assertEqual(gps.total_plans_executed, 0)
        self.assertEqual(gps.total_replans, 0)
        self.assertAlmostEqual(gps.goal_success_rate, 0.0)

    def test_initial_drive_levels(self):
        """Default intrinsic drive levels are set correctly."""
        gps = GoalPlanningSystem(enable_meta_goals=False, random_seed=0)
        self.assertAlmostEqual(gps.curiosity_level, 1.0)
        self.assertAlmostEqual(gps.competence_level, 0.0)
        self.assertAlmostEqual(gps.autonomy_level, 0.5)

    def test_empty_collections_on_init(self):
        """Plans, history, value tables start empty."""
        gps = GoalPlanningSystem(enable_meta_goals=False, random_seed=0)
        self.assertEqual(len(gps.completed_goals), 0)
        self.assertEqual(len(gps.failed_goals), 0)
        self.assertEqual(len(gps.active_plans), 0)
        self.assertEqual(len(gps.plan_history), 0)
        self.assertEqual(len(gps.state_values), 0)
        self.assertEqual(len(gps.action_values), 0)
        self.assertEqual(len(gps.action_library), 0)


# ===================================================================
# 2. Goal generation
# ===================================================================

class TestGenerateGoal(unittest.TestCase):
    """Tests for GoalPlanningSystem.generate_goal."""

    def setUp(self):
        self.gps = GoalPlanningSystem(enable_meta_goals=False, random_seed=7)

    def test_generate_learning_goal(self):
        goal = self.gps.generate_goal(goal_type=GoalType.LEARNING)
        self.assertIsNotNone(goal)
        self.assertEqual(goal.goal_type, GoalType.LEARNING)
        self.assertTrue(goal.name.startswith("learn_pattern_"))
        self.assertAlmostEqual(goal.priority, 0.7)
        self.assertAlmostEqual(goal.value, self.gps.learning_value)

    def test_generate_exploration_goal(self):
        goal = self.gps.generate_goal(goal_type=GoalType.EXPLORATION)
        self.assertIsNotNone(goal)
        self.assertEqual(goal.goal_type, GoalType.EXPLORATION)
        self.assertTrue(goal.name.startswith("explore_state_"))
        self.assertAlmostEqual(goal.priority, 0.6)
        self.assertAlmostEqual(goal.value, self.gps.exploration_bonus)

    def test_generate_optimization_goal(self):
        goal = self.gps.generate_goal(goal_type=GoalType.OPTIMIZATION)
        self.assertIsNotNone(goal)
        self.assertEqual(goal.goal_type, GoalType.OPTIMIZATION)
        self.assertTrue(goal.name.startswith("optimize_performance_"))
        self.assertAlmostEqual(goal.priority, 0.8)
        self.assertAlmostEqual(goal.value, 0.9)

    def test_generate_meta_goal(self):
        goal = self.gps.generate_goal(goal_type=GoalType.META)
        self.assertIsNotNone(goal)
        self.assertEqual(goal.goal_type, GoalType.META)
        self.assertTrue(goal.name.startswith("improve_capability_"))
        self.assertAlmostEqual(goal.priority, 0.9)
        self.assertAlmostEqual(goal.value, 1.0)
        self.assertTrue(goal.metadata.get("meta"))

    def test_generate_default_goal_for_other_types(self):
        """Types without dedicated generators (SURVIVAL, CREATION, SOCIAL)
        produce a fallback goal."""
        for gtype in (GoalType.SURVIVAL, GoalType.CREATION, GoalType.SOCIAL):
            gps = GoalPlanningSystem(enable_meta_goals=False, random_seed=0)
            goal = gps.generate_goal(goal_type=gtype)
            self.assertIsNotNone(goal)
            self.assertEqual(goal.goal_type, gtype)
            self.assertIn(gtype.value, goal.name)
            self.assertAlmostEqual(goal.priority, 0.5)
            self.assertAlmostEqual(goal.value, 0.5)

    def test_goal_added_to_active_goals(self):
        goal = self.gps.generate_goal(goal_type=GoalType.LEARNING)
        self.assertIn(goal, self.gps.active_goals)

    def test_total_goals_created_incremented(self):
        self.assertEqual(self.gps.total_goals_created, 0)
        self.gps.generate_goal(goal_type=GoalType.LEARNING)
        self.assertEqual(self.gps.total_goals_created, 1)
        self.gps.generate_goal(goal_type=GoalType.EXPLORATION)
        self.assertEqual(self.gps.total_goals_created, 2)

    def test_goal_name_uses_counter(self):
        """Sequential goals embed the running counter in their name."""
        g1 = self.gps.generate_goal(goal_type=GoalType.LEARNING)
        g2 = self.gps.generate_goal(goal_type=GoalType.LEARNING)
        self.assertIn("_0", g1.name)
        self.assertIn("_1", g2.name)

    def test_max_goals_limit_returns_none(self):
        gps = GoalPlanningSystem(max_goals=2, enable_meta_goals=False,
                                 random_seed=0)
        gps.generate_goal(goal_type=GoalType.LEARNING)
        gps.generate_goal(goal_type=GoalType.LEARNING)
        result = gps.generate_goal(goal_type=GoalType.LEARNING)
        self.assertIsNone(result)
        self.assertEqual(len(gps.active_goals), 2)

    def test_context_passed_to_metadata(self):
        ctx = {"key": "value", "level": 3}
        goal = self.gps.generate_goal(goal_type=GoalType.LEARNING,
                                      context=ctx)
        self.assertEqual(goal.metadata.get("context"), ctx)

    def test_goal_default_status_is_pending(self):
        goal = self.gps.generate_goal(goal_type=GoalType.LEARNING)
        self.assertEqual(goal.status, PlanStatus.PENDING)


# ===================================================================
# 3. Plan creation
# ===================================================================

class TestCreatePlan(unittest.TestCase):
    """Tests for GoalPlanningSystem.create_plan."""

    def setUp(self):
        self.gps = GoalPlanningSystem(enable_meta_goals=False, random_seed=0)

    def test_create_plan_no_actions_returns_none(self):
        """When the action library is empty, no plan can be found."""
        goal = _make_goal(name="unreachable")
        plan = self.gps.create_plan(goal, {})
        self.assertIsNone(plan)

    def test_create_plan_goal_already_satisfied(self):
        """If the initial state satisfies the goal, plan has zero actions."""
        goal = _make_goal(name="done")
        state = {"done": True}
        # No actions needed; the goal is satisfied in the start state.
        plan = self.gps.create_plan(goal, state)
        self.assertIsNotNone(plan)
        self.assertEqual(len(plan.actions), 0)
        self.assertEqual(plan.goal, goal)
        self.assertAlmostEqual(plan.expected_cost, 0.0)

    def test_create_plan_single_action(self):
        """One action transforms state to satisfy the goal."""
        action = _make_action(
            name="do_it",
            pre={},
            eff={"my_goal": True},
            cost=2.0,
        )
        self.gps.action_library = [action]
        goal = _make_goal(name="my_goal", value=5.0)
        plan = self.gps.create_plan(goal, {})
        self.assertIsNotNone(plan)
        self.assertEqual(len(plan.actions), 1)
        self.assertEqual(plan.actions[0].name, "do_it")
        self.assertAlmostEqual(plan.expected_cost, 2.0)
        self.assertAlmostEqual(plan.expected_value, 5.0)

    def test_create_plan_multi_step(self):
        """Two sequential actions are needed to satisfy the goal."""
        a1 = _make_action(name="step1", pre={}, eff={"halfway": True},
                          cost=1.0)
        a2 = _make_action(name="step2", pre={"halfway": True},
                          eff={"target": True}, cost=3.0)
        self.gps.action_library = [a1, a2]
        goal = _make_goal(name="target", value=10.0)
        plan = self.gps.create_plan(goal, {})
        self.assertIsNotNone(plan)
        self.assertEqual(len(plan.actions), 2)
        action_names = [a.name for a in plan.actions]
        self.assertEqual(action_names, ["step1", "step2"])
        self.assertAlmostEqual(plan.expected_cost, 4.0)

    def test_create_plan_preconditions_block_action(self):
        """Action whose preconditions are never met cannot be used."""
        blocked = _make_action(name="blocked", pre={"impossible": True},
                               eff={"goal_x": True}, cost=1.0)
        self.gps.action_library = [blocked]
        goal = _make_goal(name="goal_x")
        plan = self.gps.create_plan(goal, {})
        self.assertIsNone(plan)

    def test_plan_stored_in_active_plans(self):
        goal = _make_goal(name="stored")
        plan = self.gps.create_plan(goal, {"stored": True})
        self.assertIsNotNone(plan)
        self.assertIn(goal, self.gps.active_plans)
        self.assertIs(self.gps.active_plans[goal], plan)

    def test_plan_status_defaults_to_pending(self):
        goal = _make_goal(name="pend")
        plan = self.gps.create_plan(goal, {"pend": True})
        self.assertIsNotNone(plan)
        self.assertEqual(plan.status, PlanStatus.PENDING)

    def test_create_plan_with_completion_criteria(self):
        """Goal with custom completion_criteria is respected."""
        goal = _make_goal(
            name="custom",
            completion_criteria=lambda s: s.get("x", 0) > 5,
        )
        # State already satisfies criteria
        plan = self.gps.create_plan(goal, {"x": 10})
        self.assertIsNotNone(plan)
        self.assertEqual(len(plan.actions), 0)

        # State does not satisfy, and no actions available
        plan2 = self.gps.create_plan(goal, {"x": 1})
        self.assertIsNone(plan2)

    def test_hierarchical_plan_with_subgoals(self):
        """Goal with subgoals produces a combined plan."""
        sub1 = _make_goal(name="sub1", priority=0.5, value=1.0)
        sub2 = _make_goal(name="sub2", priority=0.5, value=2.0)
        parent = _make_goal(name="parent", priority=0.8, value=5.0,
                            subgoals=[sub1, sub2])
        # Satisfy both subgoals immediately in state
        state = {"sub1": True, "sub2": True}
        plan = self.gps.create_plan(parent, state)
        self.assertIsNotNone(plan)
        self.assertEqual(plan.goal, parent)
        # Both sub-plans have 0 actions, so combined has 0
        self.assertEqual(len(plan.actions), 0)

    def test_hierarchical_plan_fails_if_subgoals_unplannable(self):
        """If no subgoal can be planned, the parent plan is None."""
        sub = _make_goal(name="unreachable_sub")
        parent = _make_goal(name="parent", subgoals=[sub])
        plan = self.gps.create_plan(parent, {})
        self.assertIsNone(plan)

    def test_plan_confidence_decreases_with_length(self):
        """Longer plans have lower confidence (1 / (len+1))."""
        a1 = _make_action(name="s1", pre={}, eff={"mid": True}, cost=1.0)
        a2 = _make_action(name="s2", pre={"mid": True},
                          eff={"end": True}, cost=1.0)
        self.gps.action_library = [a1, a2]

        short_goal = _make_goal(name="mid")
        long_goal = _make_goal(name="end")

        p_short = self.gps.create_plan(short_goal, {})
        # Reset active_plans so the next call is fresh
        self.gps.active_plans.clear()
        p_long = self.gps.create_plan(long_goal, {})

        self.assertIsNotNone(p_short)
        self.assertIsNotNone(p_long)
        # 1-step => confidence 1/2, 2-step => confidence 1/3
        self.assertAlmostEqual(p_short.confidence, 0.5)
        self.assertAlmostEqual(p_long.confidence, 1.0 / 3.0)
        self.assertGreater(p_short.confidence, p_long.confidence)


# ===================================================================
# 4. Step execution
# ===================================================================

class TestExecuteStep(unittest.TestCase):
    """Tests for GoalPlanningSystem.execute_step."""

    def setUp(self):
        self.gps = GoalPlanningSystem(enable_meta_goals=False, random_seed=0)

    def test_execute_step_no_goals(self):
        """Returns None when there are no active goals."""
        result = self.gps.execute_step({})
        self.assertIsNone(result)

    def test_execute_step_no_plan(self):
        """Returns None when the active goal has no plan."""
        goal = _make_goal(name="no_plan", priority=0.9)
        self.gps.active_goals.append(goal)
        result = self.gps.execute_step({})
        self.assertIsNone(result)

    def test_execute_step_returns_action(self):
        """Pops the first action from the plan and returns it."""
        goal = _make_goal(name="exec_goal", priority=0.9)
        action = _make_action(name="step_action")
        plan = Plan(
            goal=goal,
            actions=[action],
            expected_value=1.0,
            expected_cost=1.0,
            confidence=0.8,
        )
        self.gps.active_goals.append(goal)
        self.gps.active_plans[goal] = plan

        result = self.gps.execute_step({})
        self.assertIsNotNone(result)
        self.assertEqual(result.name, "step_action")
        self.assertEqual(plan.status, PlanStatus.IN_PROGRESS)

    def test_execute_step_pops_actions_sequentially(self):
        """Successive calls pop actions in order."""
        goal = _make_goal(name="seq_goal", priority=0.9)
        a1 = _make_action(name="first")
        a2 = _make_action(name="second")
        a3 = _make_action(name="third")
        plan = Plan(
            goal=goal,
            actions=[a1, a2, a3],
            expected_value=3.0,
            expected_cost=3.0,
            confidence=0.8,
        )
        self.gps.active_goals.append(goal)
        self.gps.active_plans[goal] = plan

        r1 = self.gps.execute_step({})
        self.assertEqual(r1.name, "first")
        r2 = self.gps.execute_step({})
        self.assertEqual(r2.name, "second")
        r3 = self.gps.execute_step({})
        self.assertEqual(r3.name, "third")

    def test_execute_step_completes_goal_when_plan_empty(self):
        """When the plan's action list is exhausted, the goal is completed."""
        goal = _make_goal(name="finish_goal", priority=0.9)
        plan = Plan(
            goal=goal,
            actions=[],
            expected_value=1.0,
            expected_cost=0.0,
            confidence=0.8,
        )
        self.gps.active_goals.append(goal)
        self.gps.active_plans[goal] = plan

        result = self.gps.execute_step({})
        self.assertIsNone(result)
        self.assertEqual(plan.status, PlanStatus.COMPLETED)
        self.assertEqual(goal.status, PlanStatus.COMPLETED)
        self.assertNotIn(goal, self.gps.active_goals)
        self.assertIn(goal, self.gps.completed_goals)
        self.assertIsNotNone(goal.completion_time)

    def test_execute_step_updates_success_rate(self):
        """Completing a goal updates goal_success_rate."""
        goal = _make_goal(name="rate_goal", priority=0.9)
        plan = Plan(
            goal=goal,
            actions=[],
            expected_value=1.0,
            expected_cost=0.0,
            confidence=0.8,
        )
        self.gps.active_goals.append(goal)
        self.gps.active_plans[goal] = plan

        self.assertAlmostEqual(self.gps.goal_success_rate, 0.0)
        self.gps.execute_step({})
        self.assertAlmostEqual(self.gps.goal_success_rate, 1.0)

    def test_execute_step_selects_highest_priority(self):
        """When multiple goals have plans, the highest-priority one is chosen."""
        low = _make_goal(name="low_p", priority=0.2)
        high = _make_goal(name="high_p", priority=0.95)

        a_low = _make_action(name="low_action")
        a_high = _make_action(name="high_action")

        self.gps.active_goals.extend([low, high])
        self.gps.active_plans[low] = Plan(
            goal=low, actions=[a_low], expected_value=1.0,
            expected_cost=1.0, confidence=0.8)
        self.gps.active_plans[high] = Plan(
            goal=high, actions=[a_high], expected_value=1.0,
            expected_cost=1.0, confidence=0.8)

        result = self.gps.execute_step({})
        self.assertEqual(result.name, "high_action")

    def test_execute_step_replans_on_low_confidence(self):
        """When plan confidence is below replan_threshold, replanning occurs."""
        gps = GoalPlanningSystem(
            enable_meta_goals=False,
            random_seed=0,
            replan_threshold=0.5,
        )
        goal = _make_goal(name="replan_goal", priority=0.9, value=2.0)
        action = _make_action(name="old_action", eff={"replan_goal": True})
        gps.action_library = [action]

        # Set up plan with confidence below threshold
        low_conf_plan = Plan(
            goal=goal,
            actions=[_make_action(name="stale_action")],
            expected_value=1.0,
            expected_cost=1.0,
            confidence=0.1,  # below 0.5 threshold
        )
        gps.active_goals.append(goal)
        gps.active_plans[goal] = low_conf_plan

        result = gps.execute_step({})
        # After replanning the new plan's first action should be the library action
        self.assertIsNotNone(result)
        self.assertEqual(result.name, "old_action")
        self.assertEqual(gps.total_replans, 1)


# ===================================================================
# 5. GoalType enum values
# ===================================================================

class TestGoalTypeEnum(unittest.TestCase):
    """Tests for GoalType enum."""

    def test_all_expected_members_exist(self):
        expected = {"SURVIVAL", "LEARNING", "EXPLORATION", "OPTIMIZATION",
                    "CREATION", "SOCIAL", "META"}
        actual = {m.name for m in GoalType}
        self.assertEqual(actual, expected)

    def test_string_values(self):
        self.assertEqual(GoalType.SURVIVAL.value, "survival")
        self.assertEqual(GoalType.LEARNING.value, "learning")
        self.assertEqual(GoalType.EXPLORATION.value, "exploration")
        self.assertEqual(GoalType.OPTIMIZATION.value, "optimization")
        self.assertEqual(GoalType.CREATION.value, "creation")
        self.assertEqual(GoalType.SOCIAL.value, "social")
        self.assertEqual(GoalType.META.value, "meta")

    def test_enum_identity(self):
        self.assertIs(GoalType("learning"), GoalType.LEARNING)
        self.assertIs(GoalType("meta"), GoalType.META)

    def test_enum_member_count(self):
        self.assertEqual(len(GoalType), 7)


# ===================================================================
# 6. PlanStatus transitions
# ===================================================================

class TestPlanStatusTransitions(unittest.TestCase):
    """Tests for PlanStatus enum and status transitions on Goal/Plan."""

    def test_all_expected_statuses_exist(self):
        expected = {"PENDING", "IN_PROGRESS", "COMPLETED", "FAILED",
                    "SUSPENDED"}
        actual = {m.name for m in PlanStatus}
        self.assertEqual(actual, expected)

    def test_string_values(self):
        self.assertEqual(PlanStatus.PENDING.value, "pending")
        self.assertEqual(PlanStatus.IN_PROGRESS.value, "in_progress")
        self.assertEqual(PlanStatus.COMPLETED.value, "completed")
        self.assertEqual(PlanStatus.FAILED.value, "failed")
        self.assertEqual(PlanStatus.SUSPENDED.value, "suspended")

    def test_goal_default_status(self):
        goal = _make_goal()
        self.assertEqual(goal.status, PlanStatus.PENDING)

    def test_goal_status_can_transition(self):
        goal = _make_goal()
        goal.status = PlanStatus.IN_PROGRESS
        self.assertEqual(goal.status, PlanStatus.IN_PROGRESS)
        goal.status = PlanStatus.COMPLETED
        self.assertEqual(goal.status, PlanStatus.COMPLETED)

    def test_plan_default_status(self):
        goal = _make_goal()
        plan = Plan(goal=goal, actions=[], expected_value=0.0,
                    expected_cost=0.0, confidence=1.0)
        self.assertEqual(plan.status, PlanStatus.PENDING)

    def test_plan_transitions_to_in_progress(self):
        """execute_step sets the plan status to IN_PROGRESS when an action
        is popped."""
        gps = GoalPlanningSystem(enable_meta_goals=False, random_seed=0)
        goal = _make_goal(name="trans", priority=0.9)
        action = _make_action(name="go")
        plan = Plan(goal=goal, actions=[action], expected_value=1.0,
                    expected_cost=1.0, confidence=0.8)
        gps.active_goals.append(goal)
        gps.active_plans[goal] = plan

        self.assertEqual(plan.status, PlanStatus.PENDING)
        gps.execute_step({})
        self.assertEqual(plan.status, PlanStatus.IN_PROGRESS)

    def test_plan_transitions_to_completed(self):
        """Plan and goal move to COMPLETED when action list is exhausted."""
        gps = GoalPlanningSystem(enable_meta_goals=False, random_seed=0)
        goal = _make_goal(name="comp", priority=0.9)
        plan = Plan(goal=goal, actions=[], expected_value=1.0,
                    expected_cost=0.0, confidence=0.8)
        gps.active_goals.append(goal)
        gps.active_plans[goal] = plan

        gps.execute_step({})
        self.assertEqual(plan.status, PlanStatus.COMPLETED)
        self.assertEqual(goal.status, PlanStatus.COMPLETED)

    def test_plan_status_set_to_failed_manually(self):
        """PlanStatus.FAILED can be assigned (e.g. by monitoring logic)."""
        goal = _make_goal()
        plan = Plan(goal=goal, actions=[], expected_value=0.0,
                    expected_cost=0.0, confidence=0.0)
        plan.status = PlanStatus.FAILED
        self.assertEqual(plan.status, PlanStatus.FAILED)

    def test_plan_status_set_to_suspended_manually(self):
        goal = _make_goal()
        plan = Plan(goal=goal, actions=[], expected_value=0.0,
                    expected_cost=0.0, confidence=0.0)
        plan.status = PlanStatus.SUSPENDED
        self.assertEqual(plan.status, PlanStatus.SUSPENDED)


# ===================================================================
# 7. Serialization / get_state
# ===================================================================

class TestSerialization(unittest.TestCase):
    """Tests for get_stats(), serialize(), and deserialize()."""

    def test_get_stats_keys(self):
        gps = GoalPlanningSystem(enable_meta_goals=False, random_seed=0)
        stats = gps.get_stats()
        expected_keys = {
            'active_goals', 'completed_goals', 'failed_goals',
            'active_plans', 'total_goals_created', 'total_plans_executed',
            'total_replans', 'goal_success_rate', 'curiosity_level',
            'competence_level', 'autonomy_level',
        }
        self.assertEqual(set(stats.keys()), expected_keys)

    def test_get_stats_values_no_activity(self):
        gps = GoalPlanningSystem(enable_meta_goals=False, random_seed=0)
        stats = gps.get_stats()
        self.assertEqual(stats['active_goals'], 0)
        self.assertEqual(stats['completed_goals'], 0)
        self.assertEqual(stats['failed_goals'], 0)
        self.assertEqual(stats['active_plans'], 0)
        self.assertEqual(stats['total_goals_created'], 0)
        self.assertAlmostEqual(stats['goal_success_rate'], 0.0)

    def test_get_stats_after_generating_goals(self):
        gps = GoalPlanningSystem(enable_meta_goals=False, random_seed=0)
        gps.generate_goal(goal_type=GoalType.LEARNING)
        gps.generate_goal(goal_type=GoalType.EXPLORATION)
        stats = gps.get_stats()
        self.assertEqual(stats['active_goals'], 2)
        self.assertEqual(stats['total_goals_created'], 2)

    def test_serialize_structure(self):
        gps = GoalPlanningSystem(enable_meta_goals=False, random_seed=0)
        gps.generate_goal(goal_type=GoalType.LEARNING)
        data = gps.serialize()
        self.assertIn('active_goals', data)
        self.assertIn('completed_goals', data)
        self.assertIn('failed_goals', data)
        self.assertIn('drives', data)
        self.assertIn('stats', data)

    def test_serialize_active_goals_format(self):
        gps = GoalPlanningSystem(enable_meta_goals=False, random_seed=0)
        gps.generate_goal(goal_type=GoalType.OPTIMIZATION)
        data = gps.serialize()
        self.assertEqual(len(data['active_goals']), 1)
        entry = data['active_goals'][0]
        self.assertIn('name', entry)
        self.assertIn('type', entry)
        self.assertIn('priority', entry)
        self.assertEqual(entry['type'], 'optimization')

    def test_serialize_drives(self):
        gps = GoalPlanningSystem(enable_meta_goals=False, random_seed=0)
        gps.curiosity_level = 0.7
        gps.competence_level = 0.3
        gps.autonomy_level = 0.9
        data = gps.serialize()
        drives = data['drives']
        self.assertAlmostEqual(drives['curiosity'], 0.7)
        self.assertAlmostEqual(drives['competence'], 0.3)
        self.assertAlmostEqual(drives['autonomy'], 0.9)

    def test_serialize_completed_and_failed_counts(self):
        gps = GoalPlanningSystem(enable_meta_goals=False, random_seed=0)
        # Manually add completed/failed goals
        gps.completed_goals.append(_make_goal(name="c1"))
        gps.completed_goals.append(_make_goal(name="c2"))
        gps.failed_goals.append(_make_goal(name="f1"))
        data = gps.serialize()
        self.assertEqual(data['completed_goals'], 2)
        self.assertEqual(data['failed_goals'], 1)

    def test_deserialize_restores_drives(self):
        data = {
            'drives': {
                'curiosity': 0.4,
                'competence': 0.6,
                'autonomy': 0.2,
            },
        }
        gps = GoalPlanningSystem.deserialize(data)
        self.assertIsInstance(gps, GoalPlanningSystem)
        self.assertAlmostEqual(gps.curiosity_level, 0.4)
        self.assertAlmostEqual(gps.competence_level, 0.6)
        self.assertAlmostEqual(gps.autonomy_level, 0.2)

    def test_deserialize_uses_defaults_without_drives(self):
        gps = GoalPlanningSystem.deserialize({})
        self.assertAlmostEqual(gps.curiosity_level, 1.0)
        self.assertAlmostEqual(gps.competence_level, 0.0)
        self.assertAlmostEqual(gps.autonomy_level, 0.5)

    def test_serialize_roundtrip_drives(self):
        """serialize -> deserialize preserves drive levels."""
        original = GoalPlanningSystem(enable_meta_goals=False, random_seed=0)
        original.curiosity_level = 0.55
        original.competence_level = 0.33
        original.autonomy_level = 0.77
        data = original.serialize()
        restored = GoalPlanningSystem.deserialize(data)
        self.assertAlmostEqual(restored.curiosity_level, 0.55)
        self.assertAlmostEqual(restored.competence_level, 0.33)
        self.assertAlmostEqual(restored.autonomy_level, 0.77)

    def test_serialize_stats_embedded(self):
        """The 'stats' key in serialize() matches get_stats()."""
        gps = GoalPlanningSystem(enable_meta_goals=False, random_seed=0)
        gps.generate_goal(goal_type=GoalType.LEARNING)
        data = gps.serialize()
        stats_direct = gps.get_stats()
        self.assertEqual(data['stats'], stats_direct)


# ===================================================================
# 8. Dataclass and auxiliary tests
# ===================================================================

class TestGoalDataclass(unittest.TestCase):
    """Tests for the Goal dataclass."""

    def test_goal_hash_uses_name_and_creation_time(self):
        g1 = _make_goal(name="a")
        g2 = _make_goal(name="b")
        # Different names => different hashes (almost certainly)
        self.assertNotEqual(hash(g1), hash(g2))

    def test_goal_can_be_dict_key(self):
        g = _make_goal()
        d = {g: "value"}
        self.assertEqual(d[g], "value")

    def test_goal_subgoals_default_empty(self):
        g = _make_goal()
        self.assertEqual(g.subgoals, [])

    def test_goal_parent_goal_default_none(self):
        g = _make_goal()
        self.assertIsNone(g.parent_goal)

    def test_goal_creation_time_set(self):
        before = time.time()
        g = _make_goal()
        after = time.time()
        self.assertGreaterEqual(g.creation_time, before)
        self.assertLessEqual(g.creation_time, after)


class TestActionDataclass(unittest.TestCase):
    """Tests for the Action dataclass."""

    def test_action_fields(self):
        a = _make_action(name="x", pre={"a": 1}, eff={"b": 2},
                         cost=3.5, duration=2.0, success_prob=0.9)
        self.assertEqual(a.name, "x")
        self.assertEqual(a.preconditions, {"a": 1})
        self.assertEqual(a.effects, {"b": 2})
        self.assertAlmostEqual(a.cost, 3.5)
        self.assertAlmostEqual(a.duration, 2.0)
        self.assertAlmostEqual(a.success_probability, 0.9)

    def test_action_default_success_probability(self):
        a = Action(name="d", preconditions={}, effects={},
                   cost=1.0, duration=1.0)
        self.assertAlmostEqual(a.success_probability, 1.0)


class TestPlanDataclass(unittest.TestCase):
    """Tests for the Plan dataclass."""

    def test_plan_fields(self):
        goal = _make_goal()
        actions = [_make_action(name="a1"), _make_action(name="a2")]
        plan = Plan(goal=goal, actions=actions, expected_value=5.0,
                    expected_cost=2.0, confidence=0.9)
        self.assertIs(plan.goal, goal)
        self.assertEqual(len(plan.actions), 2)
        self.assertAlmostEqual(plan.expected_value, 5.0)
        self.assertAlmostEqual(plan.expected_cost, 2.0)
        self.assertAlmostEqual(plan.confidence, 0.9)
        self.assertEqual(plan.status, PlanStatus.PENDING)


# ===================================================================
# 9. Internal helpers (white-box tests)
# ===================================================================

class TestInternalHelpers(unittest.TestCase):
    """White-box tests for private helper methods."""

    def setUp(self):
        self.gps = GoalPlanningSystem(enable_meta_goals=False, random_seed=0)

    def test_state_to_key_deterministic(self):
        state = {"b": 2, "a": 1}
        key1 = self.gps._state_to_key(state)
        key2 = self.gps._state_to_key(state)
        self.assertEqual(key1, key2)

    def test_state_to_key_order_independent(self):
        s1 = {"a": 1, "b": 2}
        s2 = {"b": 2, "a": 1}
        self.assertEqual(self.gps._state_to_key(s1),
                         self.gps._state_to_key(s2))

    def test_check_preconditions_empty(self):
        action = _make_action(pre={})
        self.assertTrue(self.gps._check_preconditions(action, {}))

    def test_check_preconditions_met(self):
        action = _make_action(pre={"x": True})
        self.assertTrue(self.gps._check_preconditions(action, {"x": True}))

    def test_check_preconditions_not_met(self):
        action = _make_action(pre={"x": True})
        self.assertFalse(self.gps._check_preconditions(action, {"x": False}))
        self.assertFalse(self.gps._check_preconditions(action, {}))

    def test_apply_action_creates_new_state(self):
        action = _make_action(eff={"y": 42})
        original = {"x": 1}
        new = self.gps._apply_action(action, original)
        self.assertEqual(new, {"x": 1, "y": 42})
        # Original unchanged
        self.assertNotIn("y", original)

    def test_is_goal_satisfied_by_state(self):
        goal = _make_goal(name="sat")
        self.assertTrue(self.gps._is_goal_satisfied(goal, {"sat": True}))
        self.assertFalse(self.gps._is_goal_satisfied(goal, {"sat": False}))
        self.assertFalse(self.gps._is_goal_satisfied(goal, {}))

    def test_is_goal_satisfied_with_criteria(self):
        goal = _make_goal(
            name="crit",
            completion_criteria=lambda s: s.get("val", 0) >= 10,
        )
        self.assertTrue(self.gps._is_goal_satisfied(goal, {"val": 10}))
        self.assertFalse(self.gps._is_goal_satisfied(goal, {"val": 5}))

    def test_heuristic_satisfied_goal(self):
        goal = _make_goal(name="h")
        h = self.gps._heuristic({"h": True}, goal)
        self.assertAlmostEqual(h, 0.0)

    def test_heuristic_unsatisfied_goal(self):
        goal = _make_goal(name="h")
        h = self.gps._heuristic({}, goal)
        self.assertAlmostEqual(h, 1.0)

    def test_heuristic_with_criteria(self):
        goal = _make_goal(
            name="hc",
            completion_criteria=lambda s: s.get("ok", False),
        )
        self.assertAlmostEqual(self.gps._heuristic({"ok": True}, goal), 0.0)
        self.assertAlmostEqual(self.gps._heuristic({"ok": False}, goal), 1.0)

    def test_should_replan_low_confidence(self):
        goal = _make_goal()
        plan = Plan(goal=goal, actions=[], expected_value=0.0,
                    expected_cost=0.0, confidence=0.1)
        self.assertTrue(self.gps._should_replan(plan, {}))

    def test_should_replan_high_confidence(self):
        goal = _make_goal()
        plan = Plan(goal=goal, actions=[], expected_value=0.0,
                    expected_cost=0.0, confidence=0.9)
        self.assertFalse(self.gps._should_replan(plan, {}))

    def test_should_replan_preconditions_not_met(self):
        goal = _make_goal()
        action = _make_action(pre={"needed": True})
        plan = Plan(goal=goal, actions=[action], expected_value=0.0,
                    expected_cost=0.0, confidence=0.9)
        # Preconditions not met in state
        self.assertTrue(self.gps._should_replan(plan, {}))
        # Preconditions met
        self.assertFalse(self.gps._should_replan(plan, {"needed": True}))


if __name__ == '__main__':
    unittest.main()
