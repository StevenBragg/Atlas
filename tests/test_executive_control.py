#!/usr/bin/env python3
"""
Comprehensive tests for the ExecutiveController and related classes
from self_organizing_av_system.core.executive_control.
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from self_organizing_av_system.core.backend import xp
from self_organizing_av_system.core.executive_control import (
    ExecutiveController,
    ControlSignal,
    CognitiveState,
    UncertaintyType,
    Task,
    InhibitionUnit,
    MetacognitiveAssessment,
)


class TestControlSignalEnum(unittest.TestCase):
    """Tests for the ControlSignal enum."""

    def test_enum_values(self):
        self.assertEqual(ControlSignal.INHIBIT.value, "inhibit")
        self.assertEqual(ControlSignal.ACTIVATE.value, "activate")
        self.assertEqual(ControlSignal.SWITCH.value, "switch")
        self.assertEqual(ControlSignal.MAINTAIN.value, "maintain")
        self.assertEqual(ControlSignal.UPDATE.value, "update")
        self.assertEqual(ControlSignal.GATE.value, "gate")

    def test_enum_member_count(self):
        self.assertEqual(len(ControlSignal), 6)

    def test_enum_from_value(self):
        self.assertIs(ControlSignal("inhibit"), ControlSignal.INHIBIT)
        self.assertIs(ControlSignal("gate"), ControlSignal.GATE)


class TestCognitiveStateEnum(unittest.TestCase):
    """Tests for the CognitiveState enum."""

    def test_enum_values(self):
        self.assertEqual(CognitiveState.FOCUSED.value, "focused")
        self.assertEqual(CognitiveState.SWITCHING.value, "switching")
        self.assertEqual(CognitiveState.IDLE.value, "idle")
        self.assertEqual(CognitiveState.OVERLOADED.value, "overloaded")
        self.assertEqual(CognitiveState.RECOVERY.value, "recovery")

    def test_enum_member_count(self):
        self.assertEqual(len(CognitiveState), 5)

    def test_enum_from_value(self):
        self.assertIs(CognitiveState("idle"), CognitiveState.IDLE)
        self.assertIs(CognitiveState("recovery"), CognitiveState.RECOVERY)


class TestUncertaintyTypeEnum(unittest.TestCase):
    """Tests for the UncertaintyType enum."""

    def test_enum_values(self):
        self.assertEqual(UncertaintyType.EPISTEMIC.value, "epistemic")
        self.assertEqual(UncertaintyType.ALEATORIC.value, "aleatoric")
        self.assertEqual(UncertaintyType.COMPUTATIONAL.value, "computational")

    def test_enum_member_count(self):
        self.assertEqual(len(UncertaintyType), 3)

    def test_enum_from_value(self):
        self.assertIs(UncertaintyType("epistemic"), UncertaintyType.EPISTEMIC)
        self.assertIs(UncertaintyType("computational"), UncertaintyType.COMPUTATIONAL)


class TestTask(unittest.TestCase):
    """Tests for the Task dataclass."""

    def test_task_creation(self):
        ctx = np.zeros(64)
        task = Task(
            task_id="t0",
            name="test_task",
            priority=0.5,
            context=ctx,
        )
        self.assertEqual(task.task_id, "t0")
        self.assertEqual(task.name, "test_task")
        self.assertAlmostEqual(task.priority, 0.5)
        self.assertEqual(task.completion_progress, 0.0)
        self.assertEqual(task.estimated_effort, 1.0)
        self.assertEqual(task.actual_effort, 0.0)
        self.assertIsInstance(task.requirements, dict)
        self.assertIsInstance(task.state, dict)

    def test_update_progress(self):
        ctx = np.zeros(64)
        task = Task(task_id="t1", name="progress_test", priority=1.0, context=ctx)
        task.update_progress(0.3)
        self.assertAlmostEqual(task.completion_progress, 0.3)
        self.assertAlmostEqual(task.actual_effort, 0.3)

    def test_update_progress_clamped_to_one(self):
        ctx = np.zeros(64)
        task = Task(task_id="t2", name="clamp_test", priority=1.0, context=ctx)
        task.update_progress(0.6)
        task.update_progress(0.6)
        self.assertAlmostEqual(task.completion_progress, 1.0)
        # actual_effort should accumulate beyond 1.0
        self.assertAlmostEqual(task.actual_effort, 1.2)

    def test_update_progress_default_delta(self):
        ctx = np.zeros(64)
        task = Task(task_id="t3", name="default_delta", priority=1.0, context=ctx)
        task.update_progress()
        self.assertAlmostEqual(task.completion_progress, 0.1)
        self.assertAlmostEqual(task.actual_effort, 0.1)


class TestInhibitionUnit(unittest.TestCase):
    """Tests for the InhibitionUnit dataclass."""

    def test_creation(self):
        unit = InhibitionUnit(target_id="target_1", inhibition_strength=0.8)
        self.assertEqual(unit.target_id, "target_1")
        self.assertAlmostEqual(unit.inhibition_strength, 0.8)
        self.assertAlmostEqual(unit.decay_rate, 0.1)
        self.assertTrue(unit.active)

    def test_decay_reduces_strength(self):
        unit = InhibitionUnit(target_id="x", inhibition_strength=1.0, decay_rate=0.1)
        unit.decay()
        self.assertAlmostEqual(unit.inhibition_strength, 0.9)
        self.assertTrue(unit.active)

    def test_decay_deactivates_when_low(self):
        unit = InhibitionUnit(target_id="x", inhibition_strength=0.005, decay_rate=0.1)
        unit.decay()
        # 0.005 * 0.9 = 0.0045 < 0.01
        self.assertFalse(unit.active)

    def test_decay_at_boundary(self):
        # Exactly at deactivation boundary: strength * (1 - rate) = 0.01
        # strength = 0.01 / 0.9 ~ 0.01111
        unit = InhibitionUnit(target_id="x", inhibition_strength=0.02, decay_rate=0.1)
        unit.decay()
        # 0.02 * 0.9 = 0.018 >= 0.01 -> still active
        self.assertTrue(unit.active)


class TestMetacognitiveAssessment(unittest.TestCase):
    """Tests for the MetacognitiveAssessment dataclass."""

    def test_creation(self):
        assessment = MetacognitiveAssessment(
            confidence=0.8,
            uncertainty_type=UncertaintyType.EPISTEMIC,
            knowledge_state={"math": 0.9},
            processing_load=0.3,
            time_pressure=0.2,
            recommendations=["Study more"],
        )
        self.assertAlmostEqual(assessment.confidence, 0.8)
        self.assertIs(assessment.uncertainty_type, UncertaintyType.EPISTEMIC)
        self.assertEqual(assessment.knowledge_state, {"math": 0.9})
        self.assertAlmostEqual(assessment.processing_load, 0.3)
        self.assertAlmostEqual(assessment.time_pressure, 0.2)
        self.assertEqual(len(assessment.recommendations), 1)


class TestExecutiveControllerInit(unittest.TestCase):
    """Tests for ExecutiveController initialization."""

    def test_default_init(self):
        ec = ExecutiveController(random_seed=42)
        self.assertEqual(ec.control_dim, 64)
        self.assertEqual(ec.max_tasks, 10)
        self.assertAlmostEqual(ec.inhibition_threshold, 0.5)
        self.assertAlmostEqual(ec.switch_cost, 0.2)
        self.assertAlmostEqual(ec.learning_rate, 0.05)
        self.assertEqual(len(ec.tasks), 0)
        self.assertIsNone(ec.current_task)
        self.assertEqual(ec.task_stack, [])
        self.assertEqual(ec.task_counter, 0)
        self.assertIs(ec.cognitive_state, CognitiveState.IDLE)
        self.assertAlmostEqual(ec.processing_load, 0.0)
        self.assertAlmostEqual(ec.fatigue, 0.0)
        self.assertEqual(len(ec.inhibitions), 0)
        self.assertAlmostEqual(ec.global_inhibition, 0.0)
        self.assertEqual(ec.total_decisions, 0)
        self.assertEqual(ec.successful_inhibitions, 0)
        self.assertEqual(ec.task_switches, 0)
        self.assertAlmostEqual(ec.total_processing_time, 0.0)

    def test_custom_init(self):
        ec = ExecutiveController(
            control_dim=32,
            max_tasks=5,
            inhibition_threshold=0.3,
            switch_cost=0.5,
            learning_rate=0.1,
            random_seed=99,
        )
        self.assertEqual(ec.control_dim, 32)
        self.assertEqual(ec.max_tasks, 5)
        self.assertAlmostEqual(ec.inhibition_threshold, 0.3)
        self.assertAlmostEqual(ec.switch_cost, 0.5)
        self.assertAlmostEqual(ec.learning_rate, 0.1)
        self.assertEqual(ec.control_weights.shape, (32, 32))
        self.assertEqual(ec.control_bias.shape, (32,))
        self.assertEqual(ec.context_buffer.shape, (32,))

    def test_control_weights_shape(self):
        ec = ExecutiveController(control_dim=16, random_seed=0)
        self.assertEqual(ec.control_weights.shape, (16, 16))
        self.assertEqual(ec.control_bias.shape, (16,))
        # control_bias should be all zeros initially
        np.testing.assert_array_equal(ec.control_bias, np.zeros(16))


class TestExecutiveControllerTaskManagement(unittest.TestCase):
    """Tests for task add / complete / switch operations."""

    def setUp(self):
        self.ec = ExecutiveController(control_dim=16, random_seed=42)

    def test_add_task_returns_id(self):
        tid = self.ec.add_task("task_a", priority=1.0)
        self.assertEqual(tid, "task_0")

    def test_add_task_increments_counter(self):
        self.ec.add_task("a", priority=1.0)
        self.ec.add_task("b", priority=2.0)
        self.assertEqual(self.ec.task_counter, 2)

    def test_first_task_becomes_current(self):
        tid = self.ec.add_task("first", priority=1.0)
        self.assertEqual(self.ec.current_task, tid)
        self.assertIs(self.ec.cognitive_state, CognitiveState.FOCUSED)

    def test_second_task_does_not_replace_current(self):
        t0 = self.ec.add_task("first", priority=1.0)
        self.ec.add_task("second", priority=2.0)
        self.assertEqual(self.ec.current_task, t0)

    def test_add_task_with_explicit_context(self):
        ctx = np.ones(16)
        tid = self.ec.add_task("explicit_ctx", priority=0.5, context=ctx)
        np.testing.assert_array_almost_equal(self.ec.tasks[tid].context, ctx)

    def test_add_task_with_requirements(self):
        tid = self.ec.add_task("req_task", priority=1.0, requirements={"cpu": 0.5})
        self.assertEqual(self.ec.tasks[tid].requirements, {"cpu": 0.5})

    def test_add_task_evicts_completed_lowest_priority_when_full(self):
        ec = ExecutiveController(control_dim=8, max_tasks=3, random_seed=0)
        t0 = ec.add_task("a", priority=1.0)
        ec.add_task("b", priority=2.0)
        ec.add_task("c", priority=3.0)
        # Mark the lowest-priority task as completed
        ec.tasks[t0].completion_progress = 1.0
        # This should evict the completed lowest-priority task
        t3 = ec.add_task("d", priority=4.0)
        self.assertNotIn(t0, ec.tasks)
        self.assertIn(t3, ec.tasks)

    def test_add_task_evicts_lowest_priority_incomplete_when_full(self):
        ec = ExecutiveController(control_dim=8, max_tasks=3, random_seed=0)
        t0 = ec.add_task("a", priority=1.0)
        ec.add_task("b", priority=2.0)
        ec.add_task("c", priority=3.0)
        # None are completed, so the lowest-priority incomplete task should be evicted
        t3 = ec.add_task("d", priority=4.0)
        self.assertNotIn(t0, ec.tasks)
        self.assertIn(t3, ec.tasks)

    def test_complete_current_task_goes_idle(self):
        tid = self.ec.add_task("solo", priority=1.0)
        result = self.ec.complete_task(tid)
        self.assertTrue(result['success'])
        self.assertAlmostEqual(self.ec.tasks[tid].completion_progress, 1.0)
        self.assertIsNone(self.ec.current_task)
        self.assertIs(self.ec.cognitive_state, CognitiveState.IDLE)

    def test_complete_task_not_found(self):
        result = self.ec.complete_task("nonexistent")
        self.assertFalse(result['success'])
        self.assertEqual(result['reason'], 'Task not found')

    def test_complete_task_defaults_to_current(self):
        tid = self.ec.add_task("current", priority=1.0)
        result = self.ec.complete_task()
        self.assertTrue(result['success'])
        self.assertEqual(result['task_id'], tid)

    def test_complete_task_resumes_previous(self):
        t0 = self.ec.add_task("base", priority=1.0)
        t1 = self.ec.add_task("overlay", priority=2.0)
        # Switch to t1 to push t0 onto the stack
        self.ec.switch_task(t1)
        # Now complete t1, which should resume t0
        result = self.ec.complete_task(t1)
        self.assertTrue(result['success'])
        self.assertEqual(self.ec.current_task, t0)
        self.assertIn('resumed_task', result)
        self.assertEqual(result['resumed_task'], t0)


class TestTaskSwitching(unittest.TestCase):
    """Tests for task switching behavior."""

    def setUp(self):
        self.ec = ExecutiveController(control_dim=16, switch_cost=0.2, random_seed=42)
        self.t0 = self.ec.add_task("task_a", priority=1.0, context=np.ones(16))
        self.t1 = self.ec.add_task("task_b", priority=2.0, context=np.full(16, 2.0))

    def test_switch_task_success(self):
        result = self.ec.switch_task(self.t1)
        self.assertTrue(result['success'])
        self.assertEqual(result['previous_task'], self.t0)
        self.assertEqual(result['new_task'], self.t1)
        self.assertAlmostEqual(result['switch_cost'], 0.2)
        self.assertEqual(self.ec.current_task, self.t1)

    def test_switch_task_increments_counter(self):
        self.ec.switch_task(self.t1)
        self.assertEqual(self.ec.task_switches, 1)

    def test_switch_task_not_found(self):
        result = self.ec.switch_task("nonexistent")
        self.assertFalse(result['success'])
        self.assertEqual(result['reason'], 'Task not found')

    def test_switch_preserves_context(self):
        self.ec.switch_task(self.t1, preserve_context=True)
        # Previous task should be on the stack
        self.assertIn(self.t0, self.ec.task_stack)
        self.assertEqual(len(self.ec.context_history), 1)

    def test_switch_without_preserving_context(self):
        initial_stack_len = len(self.ec.task_stack)
        self.ec.switch_task(self.t1, preserve_context=False)
        # Stack should not grow
        self.assertEqual(len(self.ec.task_stack), initial_stack_len)

    def test_switch_task_updates_context_buffer(self):
        self.ec.switch_task(self.t1)
        np.testing.assert_array_almost_equal(
            self.ec.context_buffer, np.full(16, 2.0)
        )

    def test_switch_cost_increases_processing_load(self):
        load_before = self.ec.processing_load
        self.ec.switch_task(self.t1)
        self.assertAlmostEqual(self.ec.processing_load, load_before + 0.2)

    def test_switch_task_cognitive_state_returns_to_focused(self):
        self.ec.switch_task(self.t1)
        self.assertIs(self.ec.cognitive_state, CognitiveState.FOCUSED)


class TestDecisionMaking(unittest.TestCase):
    """Tests for decide_under_uncertainty."""

    def setUp(self):
        self.ec = ExecutiveController(control_dim=16, random_seed=42)

    def test_empty_options(self):
        chosen, conf, info = self.ec.decide_under_uncertainty([])
        self.assertEqual(chosen, -1)
        self.assertAlmostEqual(conf, 0.0)
        self.assertIn('error', info)

    def test_single_option(self):
        opts = [{'value': 1.0, 'risk': 0.0}]
        chosen, conf, info = self.ec.decide_under_uncertainty(opts)
        self.assertEqual(chosen, 0)
        self.assertGreater(conf, 0.0)
        self.assertEqual(len(info['utilities']), 1)
        self.assertEqual(len(info['uncertainties']), 1)

    def test_high_value_preferred(self):
        opts = [
            {'value': 0.1, 'risk': 0.0},
            {'value': 10.0, 'risk': 0.0},
        ]
        chosen, conf, info = self.ec.decide_under_uncertainty(opts)
        self.assertEqual(chosen, 1)

    def test_high_risk_penalized(self):
        opts = [
            {'value': 5.0, 'risk': 0.0},
            {'value': 5.0, 'risk': 100.0},
        ]
        chosen, conf, info = self.ec.decide_under_uncertainty(opts)
        self.assertEqual(chosen, 0)

    def test_total_decisions_incremented(self):
        self.ec.decide_under_uncertainty([{'value': 1.0, 'risk': 0.0}])
        self.assertEqual(self.ec.total_decisions, 1)
        self.ec.decide_under_uncertainty([{'value': 1.0, 'risk': 0.0}])
        self.assertEqual(self.ec.total_decisions, 2)

    def test_decision_info_structure(self):
        opts = [{'value': 1.0, 'risk': 0.1}, {'value': 2.0, 'risk': 0.2}]
        _, _, info = self.ec.decide_under_uncertainty(opts)
        self.assertIn('utilities', info)
        self.assertIn('uncertainties', info)
        self.assertIn('probabilities', info)
        self.assertIn('temperature', info)
        self.assertIn('epistemic_uncertainty', info)
        self.assertIn('fatigue_level', info)
        self.assertEqual(len(info['probabilities']), 2)

    def test_decision_with_context(self):
        ctx = np.ones(16)
        opts = [
            {'value': 1.0, 'risk': 0.0, 'embedding': np.ones(16)},
            {'value': 1.0, 'risk': 0.0, 'embedding': -np.ones(16)},
        ]
        chosen, conf, info = self.ec.decide_under_uncertainty(opts, context=ctx)
        # With context aligned to option 0's embedding, option 0 should be preferred
        self.assertEqual(chosen, 0)

    def test_fatigue_increases_risk_aversion(self):
        opts = [
            {'value': 5.0, 'risk': 3.0},
            {'value': 6.0, 'risk': 4.0},
        ]
        # Decide with no fatigue
        self.ec.fatigue = 0.0
        chosen_fresh, _, _ = self.ec.decide_under_uncertainty(opts)

        # Now with high fatigue - risk penalty is amplified
        self.ec.fatigue = 10.0
        chosen_tired, _, info_tired = self.ec.decide_under_uncertainty(opts)
        self.assertAlmostEqual(info_tired['fatigue_level'], 10.0)
        # High fatigue makes the riskier option even less attractive
        # Both options have risk, but the second has more; with extreme fatigue
        # the first should be chosen
        self.assertEqual(chosen_tired, 0)

    def test_empty_options_does_not_increment_decisions(self):
        self.ec.decide_under_uncertainty([])
        self.assertEqual(self.ec.total_decisions, 0)


class TestAttentionControl(unittest.TestCase):
    """Tests for generate_control_signal (attention / control modulation)."""

    def setUp(self):
        self.ec = ExecutiveController(control_dim=16, random_seed=42)

    def test_inhibit_signal_all_nonpositive(self):
        inp = np.random.RandomState(0).randn(16)
        signal = self.ec.generate_control_signal(inp, ControlSignal.INHIBIT)
        self.assertTrue(np.all(signal <= 0.0))

    def test_activate_signal_all_nonnegative(self):
        inp = np.random.RandomState(0).randn(16)
        signal = self.ec.generate_control_signal(inp, ControlSignal.ACTIVATE)
        self.assertTrue(np.all(signal >= 0.0))

    def test_gate_signal_between_zero_and_one(self):
        inp = np.random.RandomState(0).randn(16)
        signal = self.ec.generate_control_signal(inp, ControlSignal.GATE)
        self.assertTrue(np.all(signal >= 0.0))
        self.assertTrue(np.all(signal <= 1.0))

    def test_maintain_signal_mixes_with_context(self):
        self.ec.context_buffer = np.ones(16)
        inp = np.zeros(16)
        signal = self.ec.generate_control_signal(inp, ControlSignal.MAINTAIN)
        # With zero input the control component is just bias (zeros), so
        # result should be 0.5 * 0 + 0.5 * context = 0.5 * ones
        np.testing.assert_array_almost_equal(signal, 0.5 * np.ones(16))

    def test_global_inhibition_scales_signal(self):
        self.ec.global_inhibition = 0.5
        inp = np.random.RandomState(0).randn(16)
        signal = self.ec.generate_control_signal(inp, ControlSignal.ACTIVATE)
        # Compare with zero global inhibition
        self.ec.global_inhibition = 0.0
        signal_full = self.ec.generate_control_signal(inp, ControlSignal.ACTIVATE)
        np.testing.assert_array_almost_equal(signal, 0.5 * signal_full)

    def test_signal_output_shape(self):
        inp = np.random.RandomState(0).randn(16)
        for sig_type in ControlSignal:
            signal = self.ec.generate_control_signal(inp, sig_type)
            self.assertEqual(signal.shape, (16,), msg=f"Failed for {sig_type}")

    def test_short_input_is_padded(self):
        inp = np.array([1.0, 2.0, 3.0])
        signal = self.ec.generate_control_signal(inp, ControlSignal.ACTIVATE)
        self.assertEqual(signal.shape, (16,))

    def test_long_input_is_truncated(self):
        inp = np.random.RandomState(0).randn(100)
        signal = self.ec.generate_control_signal(inp, ControlSignal.ACTIVATE)
        self.assertEqual(signal.shape, (16,))


class TestResponseInhibition(unittest.TestCase):
    """Tests for inhibit / release_inhibition / get_inhibition_level."""

    def setUp(self):
        self.ec = ExecutiveController(
            control_dim=16, inhibition_threshold=0.5, random_seed=42
        )

    def test_inhibit_above_threshold(self):
        result = self.ec.inhibit("target_a", strength=0.8)
        self.assertTrue(result)
        self.assertIn("target_a", self.ec.inhibitions)
        self.assertEqual(self.ec.successful_inhibitions, 1)

    def test_inhibit_below_threshold_rejected(self):
        result = self.ec.inhibit("target_b", strength=0.3)
        self.assertFalse(result)
        self.assertNotIn("target_b", self.ec.inhibitions)
        self.assertEqual(self.ec.successful_inhibitions, 0)

    def test_inhibit_at_threshold(self):
        result = self.ec.inhibit("target_c", strength=0.5)
        self.assertTrue(result)
        self.assertIn("target_c", self.ec.inhibitions)

    def test_inhibit_strength_clamped_to_one(self):
        self.ec.inhibit("target_d", strength=5.0)
        self.assertAlmostEqual(self.ec.inhibitions["target_d"].inhibition_strength, 1.0)

    def test_get_inhibition_level(self):
        self.ec.inhibit("target_e", strength=0.7)
        level = self.ec.get_inhibition_level("target_e")
        self.assertAlmostEqual(level, 0.7)

    def test_get_inhibition_level_missing_target(self):
        level = self.ec.get_inhibition_level("nonexistent")
        self.assertAlmostEqual(level, 0.0)

    def test_release_inhibition(self):
        self.ec.inhibit("target_f", strength=0.9)
        released = self.ec.release_inhibition("target_f")
        self.assertTrue(released)
        self.assertNotIn("target_f", self.ec.inhibitions)

    def test_release_inhibition_not_found(self):
        released = self.ec.release_inhibition("nonexistent")
        self.assertFalse(released)

    def test_inhibit_with_no_decay(self):
        self.ec.inhibit("no_decay", strength=0.8, duration=10.0)
        unit = self.ec.inhibitions["no_decay"]
        # duration != None => decay_rate = 0.0
        self.assertAlmostEqual(unit.decay_rate, 0.0)
        initial = unit.inhibition_strength
        unit.decay()
        self.assertAlmostEqual(unit.inhibition_strength, initial)
        self.assertTrue(unit.active)


class TestConflictMonitoring(unittest.TestCase):
    """Tests for step() behaviour covering cognitive state transitions and decay."""

    def setUp(self):
        self.ec = ExecutiveController(control_dim=16, random_seed=42)

    def test_step_increases_processing_time(self):
        self.ec.step(dt=1.0)
        self.assertAlmostEqual(self.ec.total_processing_time, 1.0)
        self.ec.step(dt=2.0)
        self.assertAlmostEqual(self.ec.total_processing_time, 3.0)

    def test_step_decays_global_inhibition(self):
        self.ec.global_inhibition = 1.0
        self.ec.step()
        self.assertAlmostEqual(self.ec.global_inhibition, 0.95)

    def test_step_decays_inhibition_units(self):
        self.ec.inhibit("x", strength=0.6)
        initial_strength = self.ec.inhibitions["x"].inhibition_strength
        self.ec.step()
        self.assertLess(
            self.ec.inhibitions["x"].inhibition_strength, initial_strength
        )

    def test_step_removes_dead_inhibition_units(self):
        self.ec.inhibit("y", strength=0.5)
        self.ec.inhibitions["y"].inhibition_strength = 0.005
        self.ec.step()
        self.assertNotIn("y", self.ec.inhibitions)

    def test_step_focused_increases_fatigue(self):
        self.ec.add_task("focus_task", priority=1.0)
        self.ec.cognitive_state = CognitiveState.FOCUSED
        self.ec.fatigue = 0.0
        self.ec.step(dt=1.0)
        self.assertGreater(self.ec.fatigue, 0.0)

    def test_step_idle_decreases_fatigue(self):
        self.ec.cognitive_state = CognitiveState.IDLE
        self.ec.fatigue = 0.5
        self.ec.step(dt=1.0)
        self.assertLess(self.ec.fatigue, 0.5)

    def test_step_fatigue_clamped_min_zero(self):
        self.ec.cognitive_state = CognitiveState.IDLE
        self.ec.fatigue = 0.01
        self.ec.step(dt=1.0)
        self.assertGreaterEqual(self.ec.fatigue, 0.0)

    def test_step_overloaded_state(self):
        self.ec.add_task("load_task", priority=1.0)
        self.ec.cognitive_state = CognitiveState.FOCUSED
        self.ec.processing_load = 0.95
        self.ec.step()
        self.assertIs(self.ec.cognitive_state, CognitiveState.OVERLOADED)

    def test_step_recovery_state(self):
        self.ec.add_task("tired_task", priority=1.0)
        self.ec.cognitive_state = CognitiveState.FOCUSED
        self.ec.fatigue = 0.89
        # Running step with dt=2.0 should push fatigue above 0.9
        self.ec.step(dt=2.0)
        self.assertIs(self.ec.cognitive_state, CognitiveState.RECOVERY)

    def test_step_updates_task_progress(self):
        tid = self.ec.add_task("progressing", priority=1.0)
        self.ec.step(dt=1.0)
        self.assertGreater(self.ec.tasks[tid].completion_progress, 0.0)


class TestMetacognition(unittest.TestCase):
    """Tests for assess_metacognition and update_knowledge_state."""

    def setUp(self):
        self.ec = ExecutiveController(control_dim=16, random_seed=42)

    def test_default_assessment(self):
        assessment = self.ec.assess_metacognition()
        self.assertIsInstance(assessment, MetacognitiveAssessment)
        self.assertAlmostEqual(assessment.confidence, 0.5)
        self.assertIs(assessment.uncertainty_type, UncertaintyType.EPISTEMIC)

    def test_assessment_with_performance_history(self):
        # Inject some performance history with small errors
        for i in range(5):
            self.ec.performance_history.append({
                'decision_id': i,
                'prediction_error': 0.1,
                'timestamp': 0.0,
            })
        assessment = self.ec.assess_metacognition()
        # confidence = 1 - mean(|0.1|) = 0.9
        self.assertAlmostEqual(assessment.confidence, 0.9)

    def test_assessment_high_load_recommendation(self):
        self.ec.processing_load = 0.85
        assessment = self.ec.assess_metacognition()
        self.assertTrue(
            any("Reduce cognitive load" in r for r in assessment.recommendations)
        )

    def test_assessment_high_fatigue_recommendation(self):
        self.ec.fatigue = 0.8
        assessment = self.ec.assess_metacognition()
        self.assertTrue(
            any("Rest recommended" in r for r in assessment.recommendations)
        )

    def test_assessment_low_confidence_recommendation(self):
        for i in range(5):
            self.ec.performance_history.append({
                'decision_id': i,
                'prediction_error': 0.8,
                'timestamp': 0.0,
            })
        assessment = self.ec.assess_metacognition()
        # confidence = 1 - 0.8 = 0.2 < 0.4
        self.assertTrue(
            any("Gather more information" in r for r in assessment.recommendations)
        )

    def test_assessment_many_tasks_recommendation(self):
        for i in range(9):
            self.ec.add_task(f"task_{i}", priority=float(i))
        assessment = self.ec.assess_metacognition()
        # 9/10 = 0.9 > 0.8
        self.assertTrue(
            any("Prioritize" in r for r in assessment.recommendations)
        )

    def test_assessment_computational_uncertainty(self):
        self.ec.knowledge_states['domain_a'] = 0.9
        self.ec.processing_load = 0.8
        assessment = self.ec.assess_metacognition()
        self.assertIs(assessment.uncertainty_type, UncertaintyType.COMPUTATIONAL)

    def test_assessment_aleatoric_uncertainty(self):
        self.ec.knowledge_states['domain_a'] = 0.9
        self.ec.processing_load = 0.3
        assessment = self.ec.assess_metacognition()
        self.assertIs(assessment.uncertainty_type, UncertaintyType.ALEATORIC)

    def test_update_knowledge_state_new_domain(self):
        self.ec.update_knowledge_state("physics", 1.0)
        # initial default is 0.5 -> 0.9 * 0.5 + 0.1 * 1.0 = 0.55
        self.assertAlmostEqual(self.ec.knowledge_states["physics"], 0.55)

    def test_update_knowledge_state_existing_domain(self):
        self.ec.knowledge_states["math"] = 0.8
        self.ec.update_knowledge_state("math", 1.0)
        # 0.9 * 0.8 + 0.1 * 1.0 = 0.82
        self.assertAlmostEqual(self.ec.knowledge_states["math"], 0.82)

    def test_time_pressure_reflects_task_count(self):
        for i in range(5):
            self.ec.add_task(f"task_{i}", priority=float(i))
        assessment = self.ec.assess_metacognition()
        self.assertAlmostEqual(assessment.time_pressure, 5 / 10)


class TestFeedback(unittest.TestCase):
    """Tests for update_from_feedback."""

    def setUp(self):
        self.ec = ExecutiveController(control_dim=8, random_seed=42)

    def test_large_error_decreases_calibration(self):
        initial = self.ec.confidence_calibration
        self.ec.update_from_feedback(decision_id=0, outcome=1.0, expected=0.0)
        self.assertLess(self.ec.confidence_calibration, initial)

    def test_small_error_increases_calibration(self):
        self.ec.confidence_calibration = 0.5
        self.ec.update_from_feedback(decision_id=0, outcome=0.5, expected=0.6)
        self.assertGreater(self.ec.confidence_calibration, 0.5)

    def test_calibration_capped_at_one(self):
        self.ec.confidence_calibration = 1.0
        self.ec.update_from_feedback(decision_id=0, outcome=0.5, expected=0.6)
        self.assertLessEqual(self.ec.confidence_calibration, 1.0)

    def test_prediction_accuracy_updated(self):
        initial = self.ec.prediction_accuracy
        self.ec.update_from_feedback(decision_id=0, outcome=0.5, expected=0.5)
        # error = 0.0 < 0.3 -> accuracy_update = 1
        # new = 0.99 * 0.5 + 0.01 * 1 = 0.505
        self.assertAlmostEqual(self.ec.prediction_accuracy, 0.99 * initial + 0.01 * 1)

    def test_performance_history_appended(self):
        self.ec.update_from_feedback(decision_id=7, outcome=1.0, expected=0.5)
        self.assertEqual(len(self.ec.performance_history), 1)
        entry = self.ec.performance_history[0]
        self.assertEqual(entry['decision_id'], 7)
        self.assertAlmostEqual(entry['prediction_error'], 0.5)

    def test_hebbian_update_with_context(self):
        self.ec.context_history.append(np.ones(8))
        weights_before = self.ec.control_weights.copy()
        self.ec.update_from_feedback(decision_id=0, outcome=1.0, expected=0.0)
        # Weights should have been updated
        self.assertFalse(np.allclose(self.ec.control_weights, weights_before))


class TestGetStatsAndSerialization(unittest.TestCase):
    """Tests for get_stats() and serialize() / deserialize()."""

    def setUp(self):
        self.ec = ExecutiveController(control_dim=16, random_seed=42)

    def test_get_stats_structure(self):
        stats = self.ec.get_stats()
        expected_keys = {
            'total_tasks', 'current_task', 'cognitive_state',
            'processing_load', 'fatigue', 'active_inhibitions',
            'global_inhibition', 'total_decisions', 'successful_inhibitions',
            'task_switches', 'prediction_accuracy', 'confidence_calibration',
            'knowledge_domains', 'total_processing_time',
        }
        self.assertEqual(set(stats.keys()), expected_keys)

    def test_get_stats_values(self):
        stats = self.ec.get_stats()
        self.assertEqual(stats['total_tasks'], 0)
        self.assertIsNone(stats['current_task'])
        self.assertEqual(stats['cognitive_state'], 'idle')

    def test_serialize_structure(self):
        self.ec.add_task("ser_task", priority=1.0, context=np.ones(16))
        data = self.ec.serialize()
        self.assertIn('control_dim', data)
        self.assertIn('max_tasks', data)
        self.assertIn('tasks', data)
        self.assertIn('control_weights', data)
        self.assertIn('control_bias', data)
        self.assertIn('cognitive_state', data)
        self.assertIn('stats', data)

    def test_serialize_roundtrip(self):
        ctx = np.random.RandomState(1).randn(16)
        self.ec.add_task("roundtrip", priority=0.7, context=ctx)
        self.ec.fatigue = 0.3
        self.ec.processing_load = 0.2
        self.ec.knowledge_states['math'] = 0.9

        data = self.ec.serialize()
        restored = ExecutiveController.deserialize(data)

        self.assertEqual(restored.control_dim, self.ec.control_dim)
        self.assertEqual(restored.max_tasks, self.ec.max_tasks)
        self.assertAlmostEqual(restored.inhibition_threshold, self.ec.inhibition_threshold)
        self.assertAlmostEqual(restored.switch_cost, self.ec.switch_cost)
        self.assertAlmostEqual(restored.learning_rate, self.ec.learning_rate)
        self.assertAlmostEqual(restored.fatigue, self.ec.fatigue)
        self.assertAlmostEqual(restored.processing_load, self.ec.processing_load)
        self.assertIs(restored.cognitive_state, self.ec.cognitive_state)
        self.assertEqual(restored.current_task, self.ec.current_task)
        np.testing.assert_array_almost_equal(
            restored.control_weights, self.ec.control_weights
        )
        np.testing.assert_array_almost_equal(
            restored.control_bias, self.ec.control_bias
        )
        self.assertEqual(restored.knowledge_states, self.ec.knowledge_states)

    def test_serialize_task_data(self):
        ctx = np.ones(16)
        tid = self.ec.add_task("s_task", priority=0.5, context=ctx, requirements={"cpu": 0.2})
        data = self.ec.serialize()
        task_data = data['tasks'][tid]
        self.assertEqual(task_data['name'], "s_task")
        self.assertAlmostEqual(task_data['priority'], 0.5)
        self.assertEqual(task_data['requirements'], {"cpu": 0.2})
        self.assertEqual(len(task_data['context']), 16)

    def test_deserialize_restores_tasks(self):
        ctx = np.ones(16) * 0.5
        tid = self.ec.add_task("deser_task", priority=1.0, context=ctx)
        data = self.ec.serialize()
        restored = ExecutiveController.deserialize(data)
        self.assertIn(tid, restored.tasks)
        np.testing.assert_array_almost_equal(
            restored.tasks[tid].context, ctx
        )
        self.assertEqual(restored.tasks[tid].name, "deser_task")


class TestEdgeCases(unittest.TestCase):
    """Edge case and integration tests."""

    def test_multiple_switches_and_completions(self):
        ec = ExecutiveController(control_dim=8, random_seed=0)
        t0 = ec.add_task("base", priority=1.0, context=np.zeros(8))
        t1 = ec.add_task("mid", priority=2.0, context=np.ones(8))
        t2 = ec.add_task("top", priority=3.0, context=np.full(8, 2.0))

        ec.switch_task(t1)
        ec.switch_task(t2)
        self.assertEqual(ec.current_task, t2)
        self.assertEqual(ec.task_switches, 2)

        # Complete t2, should resume t1
        result = ec.complete_task(t2)
        self.assertTrue(result['success'])
        self.assertEqual(ec.current_task, t1)

    def test_inhibition_decay_over_multiple_steps(self):
        ec = ExecutiveController(control_dim=8, inhibition_threshold=0.5, random_seed=0)
        ec.inhibit("target", strength=1.0)
        # Decay over many steps until removed
        for _ in range(100):
            ec.step()
        self.assertNotIn("target", ec.inhibitions)

    def test_decision_with_missing_optional_fields(self):
        ec = ExecutiveController(control_dim=8, random_seed=0)
        opts = [{'value': 1.0}, {'value': 2.0}]
        chosen, conf, info = ec.decide_under_uncertainty(opts)
        self.assertIn(chosen, [0, 1])
        self.assertGreater(conf, 0.0)

    def test_step_without_current_task(self):
        ec = ExecutiveController(control_dim=8, random_seed=0)
        ec.cognitive_state = CognitiveState.IDLE
        ec.fatigue = 0.5
        ec.step(dt=1.0)
        # Should decrease fatigue when idle
        self.assertLess(ec.fatigue, 0.5)
        self.assertAlmostEqual(ec.total_processing_time, 1.0)

    def test_xp_backend_is_numpy_compatible(self):
        # Verify that xp from the backend module provides standard array ops
        arr = xp.array([1.0, 2.0, 3.0])
        self.assertEqual(arr.shape, (3,))
        self.assertAlmostEqual(float(xp.sum(arr)), 6.0)


if __name__ == '__main__':
    unittest.main()
