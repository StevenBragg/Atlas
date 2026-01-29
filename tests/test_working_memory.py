"""
Comprehensive tests for the WorkingMemory, AttentionController,
and CognitiveController classes.
"""

import os
import sys
import unittest
import numpy as np
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from self_organizing_av_system.core.backend import xp
from self_organizing_av_system.core.working_memory import (
    WorkingMemory,
    AttentionController,
    CognitiveController,
    WorkspaceItem,
    WorkspaceSlotType,
    AttentionType,
)


def _make_content(dim=128, seed=0):
    """Helper: create a deterministic content vector."""
    rng = np.random.RandomState(seed)
    vec = rng.randn(dim).astype(np.float64)
    return vec


def _make_workspace_item(item_id="test_item", dim=128, seed=0,
                         slot_type=WorkspaceSlotType.SENSORY,
                         source="test", salience=0.5, relevance=0.5,
                         activation=1.0):
    """Helper: create a WorkspaceItem with deterministic content."""
    content = _make_content(dim, seed)
    content = content / (np.linalg.norm(content) + 1e-8)
    return WorkspaceItem(
        item_id=item_id,
        content=content,
        slot_type=slot_type,
        source=source,
        salience=salience,
        relevance=relevance,
        activation=activation,
        timestamp=time.time(),
    )


# ---------- WorkingMemory Tests ----------

class TestWorkingMemoryInit(unittest.TestCase):
    """Tests for WorkingMemory initialization."""

    def test_default_initialization(self):
        wm = WorkingMemory()
        self.assertEqual(wm.capacity, 7)
        self.assertEqual(wm.content_dim, 128)
        self.assertAlmostEqual(wm.decay_rate, 0.05)
        self.assertAlmostEqual(wm.refresh_threshold, 0.3)
        self.assertAlmostEqual(wm.broadcast_threshold, 0.5)
        self.assertEqual(len(wm.slots), 0)
        self.assertIsInstance(wm.attention, AttentionController)
        self.assertEqual(wm.current_goals, [])
        self.assertIsNone(wm.current_task)
        self.assertEqual(wm.task_stack, [])
        self.assertEqual(wm.total_items_added, 0)
        self.assertEqual(wm.total_items_evicted, 0)
        self.assertEqual(wm.total_broadcasts, 0)
        self.assertEqual(wm.total_refreshes, 0)

    def test_custom_initialization(self):
        wm = WorkingMemory(capacity=10, content_dim=64, decay_rate=0.1,
                           refresh_threshold=0.2, broadcast_threshold=0.6)
        self.assertEqual(wm.capacity, 10)
        self.assertEqual(wm.content_dim, 64)
        self.assertAlmostEqual(wm.decay_rate, 0.1)
        self.assertAlmostEqual(wm.refresh_threshold, 0.2)
        self.assertAlmostEqual(wm.broadcast_threshold, 0.6)

    def test_empty_state_after_init(self):
        wm = WorkingMemory()
        self.assertEqual(len(wm.slots), 0)
        self.assertEqual(len(wm.broadcast_listeners), 0)
        self.assertEqual(len(wm.broadcast_history), 0)


class TestWorkingMemoryAdd(unittest.TestCase):
    """Tests for the add() method."""

    def setUp(self):
        self.wm = WorkingMemory(capacity=7, content_dim=128)

    def test_add_single_item(self):
        content = _make_content(128, seed=1)
        item_id = self.wm.add(
            content=content,
            slot_type=WorkspaceSlotType.SENSORY,
            source="test_source",
        )
        self.assertIsInstance(item_id, str)
        self.assertEqual(len(self.wm.slots), 1)
        self.assertIn(item_id, self.wm.slots)
        self.assertEqual(self.wm.total_items_added, 1)

    def test_add_item_with_explicit_id(self):
        content = _make_content(128, seed=2)
        item_id = self.wm.add(
            content=content,
            slot_type=WorkspaceSlotType.SEMANTIC,
            source="test_source",
            item_id="my_custom_id",
        )
        self.assertEqual(item_id, "my_custom_id")
        self.assertIn("my_custom_id", self.wm.slots)

    def test_add_item_content_is_normalized(self):
        content = np.ones(128) * 5.0
        item_id = self.wm.add(
            content=content,
            slot_type=WorkspaceSlotType.SENSORY,
            source="test",
        )
        stored = self.wm.slots[item_id].content
        norm = np.linalg.norm(stored)
        self.assertAlmostEqual(norm, 1.0, places=5)

    def test_add_item_with_smaller_content_pads(self):
        content = np.ones(64)
        item_id = self.wm.add(
            content=content,
            slot_type=WorkspaceSlotType.SENSORY,
            source="test",
        )
        stored = self.wm.slots[item_id].content
        self.assertEqual(len(stored), 128)

    def test_add_item_with_larger_content_truncates(self):
        content = np.ones(256)
        item_id = self.wm.add(
            content=content,
            slot_type=WorkspaceSlotType.SENSORY,
            source="test",
        )
        stored = self.wm.slots[item_id].content
        self.assertEqual(len(stored), 128)

    def test_add_item_initial_activation_is_one(self):
        content = _make_content(128, seed=3)
        item_id = self.wm.add(
            content=content,
            slot_type=WorkspaceSlotType.GOAL,
            source="test",
        )
        self.assertAlmostEqual(self.wm.slots[item_id].activation, 1.0)

    def test_add_item_preserves_slot_type(self):
        for stype in WorkspaceSlotType:
            content = _make_content(128, seed=hash(stype.value) % 10000)
            item_id = self.wm.add(
                content=content,
                slot_type=stype,
                source="test",
            )
            self.assertEqual(self.wm.slots[item_id].slot_type, stype)
        # Should have one per type
        self.assertEqual(len(self.wm.slots), len(WorkspaceSlotType))

    def test_add_item_preserves_metadata(self):
        content = _make_content(128, seed=10)
        meta = {"key": "value", "number": 42}
        item_id = self.wm.add(
            content=content,
            slot_type=WorkspaceSlotType.SENSORY,
            source="test",
            metadata=meta,
        )
        self.assertEqual(self.wm.slots[item_id].metadata, meta)

    def test_add_updates_attention_weights(self):
        content = _make_content(128, seed=4)
        item_id = self.wm.add(
            content=content,
            slot_type=WorkspaceSlotType.SENSORY,
            source="test",
        )
        # After adding, attention should have been updated
        self.assertIn(item_id, self.wm.attention.attention_weights)

    def test_add_multiple_items(self):
        ids = []
        for i in range(5):
            content = _make_content(128, seed=100 + i)
            item_id = self.wm.add(
                content=content,
                slot_type=WorkspaceSlotType.SENSORY,
                source="test",
            )
            ids.append(item_id)
        self.assertEqual(len(self.wm.slots), 5)
        self.assertEqual(self.wm.total_items_added, 5)
        for iid in ids:
            self.assertIn(iid, self.wm.slots)


class TestWorkingMemoryCapacity(unittest.TestCase):
    """Tests for capacity limits and eviction behaviour."""

    def test_capacity_limit_evicts_at_boundary(self):
        """Adding beyond capacity evicts the lowest priority item."""
        wm = WorkingMemory(capacity=5, content_dim=32)
        ids = []
        for i in range(5):
            content = _make_content(32, seed=200 + i)
            item_id = wm.add(
                content=content,
                slot_type=WorkspaceSlotType.SENSORY,
                source="test",
                item_id=f"item_{i}",
                salience=0.5,
                relevance=0.5,
            )
            ids.append(item_id)
        self.assertEqual(len(wm.slots), 5)

        # Adding a 6th item should cause eviction
        content = _make_content(32, seed=300)
        new_id = wm.add(
            content=content,
            slot_type=WorkspaceSlotType.SENSORY,
            source="test",
            item_id="item_overflow",
            salience=0.9,
            relevance=0.9,
        )
        self.assertEqual(len(wm.slots), 5)
        self.assertIn(new_id, wm.slots)
        self.assertGreater(wm.total_items_evicted, 0)

    def test_capacity_8_items(self):
        """Test with a capacity of 8."""
        wm = WorkingMemory(capacity=8, content_dim=32)
        for i in range(8):
            content = _make_content(32, seed=400 + i)
            wm.add(content=content, slot_type=WorkspaceSlotType.SENSORY,
                    source="test", item_id=f"cap8_{i}")
        self.assertEqual(len(wm.slots), 8)

        # 9th item triggers eviction
        content = _make_content(32, seed=500)
        wm.add(content=content, slot_type=WorkspaceSlotType.SENSORY,
                source="test", item_id="cap8_overflow")
        self.assertEqual(len(wm.slots), 8)

    def test_capacity_10_items(self):
        """Test with a capacity of 10."""
        wm = WorkingMemory(capacity=10, content_dim=32)
        for i in range(10):
            content = _make_content(32, seed=600 + i)
            wm.add(content=content, slot_type=WorkspaceSlotType.SENSORY,
                    source="test", item_id=f"cap10_{i}")
        self.assertEqual(len(wm.slots), 10)

        # 11th item triggers eviction
        content = _make_content(32, seed=700)
        wm.add(content=content, slot_type=WorkspaceSlotType.SENSORY,
                source="test", item_id="cap10_overflow")
        self.assertEqual(len(wm.slots), 10)

    def test_never_exceeds_capacity(self):
        """Adding many items never exceeds capacity."""
        wm = WorkingMemory(capacity=5, content_dim=32)
        for i in range(20):
            content = _make_content(32, seed=800 + i)
            wm.add(content=content, slot_type=WorkspaceSlotType.SENSORY,
                    source="test")
        self.assertLessEqual(len(wm.slots), 5)

    def test_eviction_counter_increments(self):
        wm = WorkingMemory(capacity=3, content_dim=32)
        for i in range(3):
            content = _make_content(32, seed=900 + i)
            wm.add(content=content, slot_type=WorkspaceSlotType.SENSORY,
                    source="test")
        self.assertEqual(wm.total_items_evicted, 0)

        content = _make_content(32, seed=999)
        wm.add(content=content, slot_type=WorkspaceSlotType.SENSORY,
                source="test")
        self.assertEqual(wm.total_items_evicted, 1)

    def test_replace_existing_item_no_eviction(self):
        """Adding an item with an existing ID replaces, no eviction."""
        wm = WorkingMemory(capacity=3, content_dim=32)
        for i in range(3):
            content = _make_content(32, seed=1000 + i)
            wm.add(content=content, slot_type=WorkspaceSlotType.SENSORY,
                    source="test", item_id=f"fixed_{i}")
        self.assertEqual(len(wm.slots), 3)

        # Replace item with same id -- should not evict
        content = _make_content(32, seed=1050)
        wm.add(content=content, slot_type=WorkspaceSlotType.SENSORY,
                source="test", item_id="fixed_0")
        self.assertEqual(len(wm.slots), 3)
        self.assertEqual(wm.total_items_evicted, 0)


class TestWorkingMemoryFocusOn(unittest.TestCase):
    """Tests for the focus_on() interaction via attention controller."""

    def setUp(self):
        self.wm = WorkingMemory(capacity=7, content_dim=64)
        self.item_ids = []
        for i in range(4):
            content = _make_content(64, seed=1100 + i)
            iid = self.wm.add(
                content=content,
                slot_type=WorkspaceSlotType.SENSORY,
                source="test",
                item_id=f"focus_{i}",
            )
            self.item_ids.append(iid)

    def test_focus_on_creates_attention_focus(self):
        self.wm.attention.focus_on(["focus_0"])
        self.assertGreaterEqual(len(self.wm.attention.current_foci), 1)
        latest_focus = self.wm.attention.current_foci[-1]
        self.assertEqual(latest_focus.target_ids, ["focus_0"])

    def test_focus_on_boosts_attention_weight(self):
        # Capture weight before
        weight_before = self.wm.attention.attention_weights.get("focus_0", 0.0)
        self.wm.attention.focus_on(["focus_0"])
        weight_after = self.wm.attention.attention_weights.get("focus_0", 0.0)
        self.assertGreaterEqual(weight_after, weight_before)

    def test_focus_on_with_top_down_attention(self):
        self.wm.attention.focus_on(
            ["focus_1"],
            attention_type=AttentionType.TOP_DOWN,
        )
        latest = self.wm.attention.current_foci[-1]
        self.assertEqual(latest.attention_type, AttentionType.TOP_DOWN)

    def test_focus_on_with_bottom_up_attention(self):
        self.wm.attention.focus_on(
            ["focus_2"],
            attention_type=AttentionType.BOTTOM_UP,
        )
        latest = self.wm.attention.current_foci[-1]
        self.assertEqual(latest.attention_type, AttentionType.BOTTOM_UP)

    def test_focus_on_with_executive_attention(self):
        self.wm.attention.focus_on(
            ["focus_3"],
            attention_type=AttentionType.EXECUTIVE,
        )
        latest = self.wm.attention.current_foci[-1]
        self.assertEqual(latest.attention_type, AttentionType.EXECUTIVE)

    def test_focus_on_tracks_switches(self):
        switches_before = self.wm.attention.total_switches
        self.wm.attention.focus_on(["focus_0"])
        self.wm.attention.focus_on(["focus_1"])
        switches_after = self.wm.attention.total_switches
        self.assertGreater(switches_after, switches_before)

    def test_focus_on_respects_head_limit(self):
        ac = AttentionController(num_attention_heads=2)
        ac.attention_weights = {"a": 0.5, "b": 0.3, "c": 0.2}
        ac.focus_on(["a"])
        ac.focus_on(["b"])
        ac.focus_on(["c"])
        # Should have at most 2 foci
        self.assertLessEqual(len(ac.current_foci), 2)

    def test_focus_on_multiple_items(self):
        self.wm.attention.focus_on(["focus_0", "focus_1"])
        latest = self.wm.attention.current_foci[-1]
        self.assertEqual(set(latest.target_ids), {"focus_0", "focus_1"})


class TestWorkingMemoryQueryByType(unittest.TestCase):
    """Tests for query_by_type()."""

    def setUp(self):
        self.wm = WorkingMemory(capacity=10, content_dim=64)
        # Add items of different types
        types_and_counts = [
            (WorkspaceSlotType.SENSORY, 3),
            (WorkspaceSlotType.SEMANTIC, 2),
            (WorkspaceSlotType.GOAL, 1),
        ]
        seed_idx = 0
        for stype, count in types_and_counts:
            for _ in range(count):
                content = _make_content(64, seed=1200 + seed_idx)
                self.wm.add(content=content, slot_type=stype, source="test")
                seed_idx += 1

    def test_query_returns_correct_type(self):
        sensory = self.wm.query_by_type(WorkspaceSlotType.SENSORY)
        for item in sensory:
            self.assertEqual(item.slot_type, WorkspaceSlotType.SENSORY)

    def test_query_returns_correct_count(self):
        sensory = self.wm.query_by_type(WorkspaceSlotType.SENSORY)
        self.assertEqual(len(sensory), 3)
        semantic = self.wm.query_by_type(WorkspaceSlotType.SEMANTIC)
        self.assertEqual(len(semantic), 2)
        goal = self.wm.query_by_type(WorkspaceSlotType.GOAL)
        self.assertEqual(len(goal), 1)

    def test_query_empty_type_returns_empty(self):
        action = self.wm.query_by_type(WorkspaceSlotType.ACTION)
        self.assertEqual(len(action), 0)

    def test_query_returns_workspace_items(self):
        items = self.wm.query_by_type(WorkspaceSlotType.SENSORY)
        for item in items:
            self.assertIsInstance(item, WorkspaceItem)


class TestWorkingMemoryGetMostActive(unittest.TestCase):
    """Tests for get_most_active()."""

    def test_returns_items_sorted_by_activation(self):
        wm = WorkingMemory(capacity=10, content_dim=32)
        # Add items and manually set different activations
        for i in range(5):
            content = _make_content(32, seed=1300 + i)
            iid = wm.add(content=content, slot_type=WorkspaceSlotType.SENSORY,
                         source="test", item_id=f"active_{i}")
            wm.slots[iid].activation = (i + 1) * 0.2  # 0.2, 0.4, 0.6, 0.8, 1.0

        most_active = wm.get_most_active(top_k=3)
        self.assertEqual(len(most_active), 3)
        # Highest activation first
        self.assertGreaterEqual(most_active[0].activation, most_active[1].activation)
        self.assertGreaterEqual(most_active[1].activation, most_active[2].activation)

    def test_top_k_limits_results(self):
        wm = WorkingMemory(capacity=10, content_dim=32)
        for i in range(5):
            content = _make_content(32, seed=1400 + i)
            wm.add(content=content, slot_type=WorkspaceSlotType.SENSORY,
                    source="test")
        result = wm.get_most_active(top_k=2)
        self.assertEqual(len(result), 2)

    def test_top_k_larger_than_size(self):
        wm = WorkingMemory(capacity=10, content_dim=32)
        for i in range(3):
            content = _make_content(32, seed=1500 + i)
            wm.add(content=content, slot_type=WorkspaceSlotType.SENSORY,
                    source="test")
        result = wm.get_most_active(top_k=10)
        self.assertEqual(len(result), 3)

    def test_most_active_on_empty_wm(self):
        wm = WorkingMemory()
        result = wm.get_most_active()
        self.assertEqual(result, [])

    def test_most_active_returns_highest_first(self):
        wm = WorkingMemory(capacity=10, content_dim=32)
        content_low = _make_content(32, seed=1600)
        content_high = _make_content(32, seed=1601)
        id_low = wm.add(content=content_low, slot_type=WorkspaceSlotType.SENSORY,
                        source="test", item_id="low")
        id_high = wm.add(content=content_high, slot_type=WorkspaceSlotType.SENSORY,
                         source="test", item_id="high")
        wm.slots["low"].activation = 0.1
        wm.slots["high"].activation = 0.9
        result = wm.get_most_active(top_k=2)
        self.assertEqual(result[0].item_id, "high")
        self.assertEqual(result[1].item_id, "low")


class TestWorkingMemoryGetState(unittest.TestCase):
    """Tests for get_state_summary() and get_stats()."""

    def test_get_state_summary_keys(self):
        wm = WorkingMemory(capacity=5, content_dim=32)
        state = wm.get_state_summary()
        expected_keys = {
            'capacity', 'current_size', 'utilization', 'items_by_type',
            'current_task', 'task_stack_depth', 'num_goals',
            'most_active_ids', 'most_attended_ids', 'avg_activation',
        }
        self.assertEqual(set(state.keys()), expected_keys)

    def test_get_state_summary_values_empty(self):
        wm = WorkingMemory(capacity=5, content_dim=32)
        state = wm.get_state_summary()
        self.assertEqual(state['capacity'], 5)
        self.assertEqual(state['current_size'], 0)
        self.assertAlmostEqual(state['utilization'], 0.0)
        self.assertEqual(state['items_by_type'], {})
        self.assertIsNone(state['current_task'])
        self.assertEqual(state['task_stack_depth'], 0)
        self.assertEqual(state['num_goals'], 0)
        self.assertEqual(state['avg_activation'], 0)

    def test_get_state_summary_with_items(self):
        wm = WorkingMemory(capacity=5, content_dim=32)
        for i in range(3):
            content = _make_content(32, seed=1700 + i)
            wm.add(content=content, slot_type=WorkspaceSlotType.SENSORY,
                    source="test")
        state = wm.get_state_summary()
        self.assertEqual(state['current_size'], 3)
        self.assertAlmostEqual(state['utilization'], 3 / 5)
        self.assertIn('sensory', state['items_by_type'])
        self.assertEqual(state['items_by_type']['sensory'], 3)

    def test_get_stats_keys(self):
        wm = WorkingMemory()
        stats = wm.get_stats()
        expected_keys = {
            'capacity', 'content_dim', 'total_items_added',
            'total_items_evicted', 'total_broadcasts', 'total_refreshes',
            'current_size', 'attention_stats',
        }
        self.assertEqual(set(stats.keys()), expected_keys)

    def test_get_stats_values(self):
        wm = WorkingMemory(capacity=5, content_dim=32)
        content = _make_content(32, seed=1800)
        wm.add(content=content, slot_type=WorkspaceSlotType.SENSORY,
                source="test")
        stats = wm.get_stats()
        self.assertEqual(stats['capacity'], 5)
        self.assertEqual(stats['content_dim'], 32)
        self.assertEqual(stats['total_items_added'], 1)
        self.assertEqual(stats['current_size'], 1)
        self.assertIsInstance(stats['attention_stats'], dict)

    def test_get_state_summary_with_task(self):
        wm = WorkingMemory()
        wm.push_task("my_task")
        state = wm.get_state_summary()
        self.assertEqual(state['current_task'], "my_task")

    def test_get_state_summary_with_goals(self):
        wm = WorkingMemory(content_dim=32)
        goal = _make_content(32, seed=1900)
        wm.set_goal(goal)
        state = wm.get_state_summary()
        self.assertEqual(state['num_goals'], 1)


# ---------- AttentionController Tests ----------

class TestAttentionControllerInit(unittest.TestCase):
    """Tests for AttentionController initialization."""

    def test_default_init(self):
        ac = AttentionController()
        self.assertEqual(ac.num_attention_heads, 4)
        self.assertAlmostEqual(ac.attention_capacity, 1.0)
        self.assertAlmostEqual(ac.switch_cost, 0.1)
        self.assertEqual(ac.current_foci, [])
        self.assertEqual(ac.attention_weights, {})
        self.assertEqual(ac.goal_biases, {})
        self.assertEqual(ac.salience_map, {})
        self.assertEqual(ac.inhibited_items, {})
        self.assertEqual(ac.total_switches, 0)
        self.assertEqual(ac.total_updates, 0)

    def test_custom_init(self):
        ac = AttentionController(
            num_attention_heads=8,
            attention_capacity=2.0,
            switch_cost=0.3,
        )
        self.assertEqual(ac.num_attention_heads, 8)
        self.assertAlmostEqual(ac.attention_capacity, 2.0)
        self.assertAlmostEqual(ac.switch_cost, 0.3)


class TestAttentionControllerAttentionTypes(unittest.TestCase):
    """Tests for different attention types and focus_on."""

    def setUp(self):
        self.ac = AttentionController()
        # Pre-populate attention weights so focus_on can boost them
        self.ac.attention_weights = {
            "item_a": 0.3,
            "item_b": 0.3,
            "item_c": 0.3,
        }

    def test_bottom_up_focus(self):
        self.ac.focus_on(["item_a"], attention_type=AttentionType.BOTTOM_UP)
        focus = self.ac.current_foci[-1]
        self.assertEqual(focus.attention_type, AttentionType.BOTTOM_UP)
        self.assertAlmostEqual(focus.intensity, 1.0)

    def test_top_down_focus(self):
        self.ac.focus_on(["item_b"], attention_type=AttentionType.TOP_DOWN)
        focus = self.ac.current_foci[-1]
        self.assertEqual(focus.attention_type, AttentionType.TOP_DOWN)

    def test_executive_focus(self):
        self.ac.focus_on(["item_c"], attention_type=AttentionType.EXECUTIVE)
        focus = self.ac.current_foci[-1]
        self.assertEqual(focus.attention_type, AttentionType.EXECUTIVE)

    def test_focus_boosts_weight(self):
        before = self.ac.attention_weights["item_a"]
        self.ac.focus_on(["item_a"])
        after = self.ac.attention_weights["item_a"]
        self.assertGreater(after, before)

    def test_focus_caps_at_one(self):
        self.ac.attention_weights["item_a"] = 0.9
        self.ac.focus_on(["item_a"])
        self.assertLessEqual(self.ac.attention_weights["item_a"], 1.0)


class TestAttentionControllerComputeSalience(unittest.TestCase):
    """Tests for salience computation."""

    def test_salience_single_item_no_context(self):
        ac = AttentionController()
        item = _make_workspace_item("s1", dim=64, seed=2000)
        sal = ac.compute_salience(item, [])
        self.assertGreaterEqual(sal, 0.0)
        self.assertLessEqual(sal, 1.0)

    def test_salience_with_context(self):
        ac = AttentionController()
        item = _make_workspace_item("s1", dim=64, seed=2100)
        ctx = [_make_workspace_item(f"ctx_{i}", dim=64, seed=2200 + i)
               for i in range(3)]
        sal = ac.compute_salience(item, ctx)
        self.assertGreaterEqual(sal, 0.0)
        self.assertLessEqual(sal, 1.0)

    def test_salience_identical_items_lower_novelty(self):
        ac = AttentionController()
        # Same seed so identical content
        item = _make_workspace_item("s1", dim=64, seed=2300, salience=0.5)
        ctx = [_make_workspace_item("s2", dim=64, seed=2300, salience=0.5)]
        sal_identical = ac.compute_salience(item, ctx)

        # Different seed so novel content
        ctx_novel = [_make_workspace_item("s3", dim=64, seed=2400, salience=0.5)]
        sal_novel = ac.compute_salience(item, ctx_novel)

        # Identical context should give lower novelty => lower salience
        self.assertLessEqual(sal_identical, sal_novel)


class TestAttentionControllerComputeRelevance(unittest.TestCase):
    """Tests for relevance computation."""

    def test_no_goals_returns_neutral(self):
        ac = AttentionController()
        item = _make_workspace_item("r1", dim=64, seed=2500)
        rel = ac.compute_relevance(item, [])
        self.assertAlmostEqual(rel, 0.5)

    def test_with_aligned_goal(self):
        ac = AttentionController()
        content = _make_content(64, seed=2600)
        content_norm = content / (np.linalg.norm(content) + 1e-8)
        item = WorkspaceItem(
            item_id="r2", content=content_norm,
            slot_type=WorkspaceSlotType.SENSORY, source="test",
            salience=0.5, relevance=0.5, activation=1.0,
            timestamp=time.time(),
        )
        # Goal is same direction -- high relevance
        rel = ac.compute_relevance(item, [content_norm])
        self.assertGreater(rel, 0.5)

    def test_relevance_clipped_to_0_1(self):
        ac = AttentionController()
        item = _make_workspace_item("r3", dim=64, seed=2700)
        goal = _make_content(64, seed=2800)
        goal = goal / (np.linalg.norm(goal) + 1e-8)
        rel = ac.compute_relevance(item, [goal])
        self.assertGreaterEqual(rel, 0.0)
        self.assertLessEqual(rel, 1.0)

    def test_goal_bias_applied(self):
        ac = AttentionController()
        item = _make_workspace_item("r4", dim=64, seed=2900)
        goal = _make_content(64, seed=3000)
        goal = goal / (np.linalg.norm(goal) + 1e-8)
        rel_no_bias = ac.compute_relevance(item, [goal])
        ac.set_goal_bias("r4", 1.0)
        rel_with_bias = ac.compute_relevance(item, [goal])
        # With a high bias, relevance should be at least as high
        self.assertGreaterEqual(rel_with_bias, rel_no_bias - 0.01)


class TestAttentionControllerUpdateAttention(unittest.TestCase):
    """Tests for update_attention()."""

    def test_empty_items_returns_empty(self):
        ac = AttentionController()
        result = ac.update_attention([], [])
        self.assertEqual(result, {})

    def test_returns_weights_for_all_items(self):
        ac = AttentionController()
        items = [_make_workspace_item(f"u{i}", dim=64, seed=3100 + i) for i in range(3)]
        weights = ac.update_attention(items, [])
        self.assertEqual(len(weights), 3)
        for item in items:
            self.assertIn(item.item_id, weights)

    def test_weights_sum_approximately_one(self):
        ac = AttentionController()
        items = [_make_workspace_item(f"u{i}", dim=64, seed=3200 + i) for i in range(4)]
        weights = ac.update_attention(items, [])
        total = sum(weights.values())
        self.assertAlmostEqual(total, 1.0, places=3)

    def test_all_weights_non_negative(self):
        ac = AttentionController()
        items = [_make_workspace_item(f"u{i}", dim=64, seed=3300 + i) for i in range(5)]
        weights = ac.update_attention(items, [])
        for w in weights.values():
            self.assertGreaterEqual(w, 0.0)

    def test_update_increments_counter(self):
        ac = AttentionController()
        items = [_make_workspace_item("u0", dim=64, seed=3400)]
        self.assertEqual(ac.total_updates, 0)
        ac.update_attention(items, [])
        self.assertEqual(ac.total_updates, 1)
        ac.update_attention(items, [])
        self.assertEqual(ac.total_updates, 2)


class TestAttentionControllerGetMostAttended(unittest.TestCase):
    """Tests for get_most_attended()."""

    def test_returns_correct_count(self):
        ac = AttentionController()
        ac.attention_weights = {"a": 0.5, "b": 0.3, "c": 0.1, "d": 0.1}
        result = ac.get_most_attended(top_k=2)
        self.assertEqual(len(result), 2)

    def test_returns_highest_first(self):
        ac = AttentionController()
        ac.attention_weights = {"a": 0.1, "b": 0.9, "c": 0.5}
        result = ac.get_most_attended(top_k=3)
        self.assertEqual(result[0], "b")

    def test_get_stats(self):
        ac = AttentionController()
        stats = ac.get_stats()
        self.assertIn('num_attention_heads', stats)
        self.assertIn('current_foci_count', stats)
        self.assertIn('total_switches', stats)
        self.assertIn('total_updates', stats)
        self.assertIn('items_with_weights', stats)
        self.assertIn('items_inhibited', stats)


class TestAttentionControllerInhibition(unittest.TestCase):
    """Tests for inhibition of return."""

    def test_inhibit_item(self):
        ac = AttentionController()
        ac.inhibit("item_x", strength=0.8)
        self.assertIn("item_x", ac.inhibited_items)
        self.assertAlmostEqual(ac.inhibited_items["item_x"], 0.8)

    def test_inhibit_caps_at_one(self):
        ac = AttentionController()
        ac.inhibit("item_y", strength=1.5)
        self.assertLessEqual(ac.inhibited_items["item_y"], 1.0)

    def test_inhibition_reduces_priority(self):
        ac = AttentionController()
        items = [_make_workspace_item(f"inh_{i}", dim=64, seed=3500 + i)
                 for i in range(3)]
        # Get baseline weights
        weights_before = ac.update_attention(items, [])
        w_before = weights_before.get("inh_0", 0.0)

        # Inhibit first item heavily
        ac.inhibit("inh_0", strength=0.9)
        weights_after = ac.update_attention(items, [])
        w_after = weights_after.get("inh_0", 0.0)

        # Inhibited item should have lower or equal weight
        self.assertLessEqual(w_after, w_before + 0.01)


# ---------- CognitiveController Tests ----------

class TestCognitiveControllerInit(unittest.TestCase):
    """Tests for CognitiveController initialization."""

    def test_default_init(self):
        wm = WorkingMemory(capacity=5, content_dim=32)
        cc = CognitiveController(wm)
        self.assertIs(cc.wm, wm)
        self.assertAlmostEqual(cc.switch_cost_ms, 200)
        self.assertAlmostEqual(cc.inhibition_strength, 0.5)
        self.assertEqual(cc.inhibitions, {})
        self.assertEqual(cc.task_history, [])
        self.assertAlmostEqual(cc.conflict_level, 0.0)
        self.assertAlmostEqual(cc.cognitive_load, 0.0)

    def test_custom_init(self):
        wm = WorkingMemory(capacity=5, content_dim=32)
        cc = CognitiveController(wm, switch_cost_ms=100, inhibition_strength=0.8)
        self.assertAlmostEqual(cc.switch_cost_ms, 100)
        self.assertAlmostEqual(cc.inhibition_strength, 0.8)


class TestCognitiveControllerPushTask(unittest.TestCase):
    """Tests for push_task and pop_task on the working memory task stack."""

    def setUp(self):
        self.wm = WorkingMemory(capacity=5, content_dim=32)

    def test_push_single_task(self):
        self.wm.push_task("task_a")
        self.assertEqual(self.wm.current_task, "task_a")
        self.assertEqual(self.wm.task_stack, [])

    def test_push_multiple_tasks(self):
        self.wm.push_task("task_a")
        self.wm.push_task("task_b")
        self.assertEqual(self.wm.current_task, "task_b")
        self.assertEqual(self.wm.task_stack, ["task_a"])

    def test_push_three_tasks(self):
        self.wm.push_task("task_a")
        self.wm.push_task("task_b")
        self.wm.push_task("task_c")
        self.assertEqual(self.wm.current_task, "task_c")
        self.assertEqual(self.wm.task_stack, ["task_a", "task_b"])

    def test_pop_restores_previous(self):
        self.wm.push_task("task_a")
        self.wm.push_task("task_b")
        completed = self.wm.pop_task()
        self.assertEqual(completed, "task_b")
        self.assertEqual(self.wm.current_task, "task_a")

    def test_pop_empty_stack(self):
        self.wm.push_task("task_a")
        completed = self.wm.pop_task()
        self.assertEqual(completed, "task_a")
        self.assertIsNone(self.wm.current_task)

    def test_pop_from_no_task(self):
        completed = self.wm.pop_task()
        self.assertIsNone(completed)
        self.assertIsNone(self.wm.current_task)

    def test_push_pop_push(self):
        self.wm.push_task("first")
        self.wm.pop_task()
        self.wm.push_task("second")
        self.assertEqual(self.wm.current_task, "second")


class TestCognitiveControllerSwitchTask(unittest.TestCase):
    """Tests for task switching via CognitiveController."""

    def setUp(self):
        self.wm = WorkingMemory(capacity=5, content_dim=32)
        self.cc = CognitiveController(self.wm)

    def test_switch_task_returns_cost(self):
        cost = self.cc.switch_task("new_task")
        self.assertEqual(cost, 200)

    def test_switch_task_updates_current(self):
        self.cc.switch_task("alpha")
        self.assertEqual(self.wm.current_task, "alpha")

    def test_switch_task_increases_conflict(self):
        conflict_before = self.cc.conflict_level
        self.cc.switch_task("beta")
        self.assertGreater(self.cc.conflict_level, conflict_before)

    def test_switch_task_records_history(self):
        self.cc.switch_task("first")
        self.cc.switch_task("second")
        # First task should be in history
        self.assertEqual(len(self.cc.task_history), 1)
        self.assertEqual(self.cc.task_history[0]['task'], "first")

    def test_switch_task_filters_irrelevant(self):
        """After a switch, non-goal/non-action items have reduced activation."""
        for i in range(3):
            content = _make_content(32, seed=3600 + i)
            self.wm.add(content=content, slot_type=WorkspaceSlotType.SENSORY,
                        source="test", item_id=f"sw_{i}")
        # Add a goal item
        content = _make_content(32, seed=3650)
        self.wm.add(content=content, slot_type=WorkspaceSlotType.GOAL,
                     source="test", item_id="sw_goal")

        # Store activations before switch
        sens_act_before = self.wm.slots["sw_0"].activation
        goal_act_before = self.wm.slots["sw_goal"].activation

        self.cc.switch_task("different_task")

        # Sensory items should have reduced activation
        self.assertLess(self.wm.slots["sw_0"].activation, sens_act_before)
        # Goal item activation should not be reduced by the filter
        # (it only reduces non-GOAL, non-ACTION items)


class TestCognitiveControllerInhibit(unittest.TestCase):
    """Tests for inhibitory control."""

    def setUp(self):
        self.wm = WorkingMemory(capacity=5, content_dim=32)
        self.cc = CognitiveController(self.wm)
        content = _make_content(32, seed=3700)
        self.wm.add(content=content, slot_type=WorkspaceSlotType.ACTION,
                     source="test", item_id="inhibit_target")

    def test_inhibit_existing_item(self):
        result = self.cc.inhibit_response("inhibit_target")
        self.assertTrue(result)
        self.assertIn("inhibit_target", self.cc.inhibitions)

    def test_inhibit_nonexistent_item(self):
        result = self.cc.inhibit_response("nonexistent")
        self.assertFalse(result)

    def test_release_inhibition(self):
        self.cc.inhibit_response("inhibit_target")
        self.cc.release_inhibition("inhibit_target")
        self.assertNotIn("inhibit_target", self.cc.inhibitions)

    def test_release_nonexistent_inhibition(self):
        # Should not raise
        self.cc.release_inhibition("nonexistent")


class TestCognitiveControllerCognitiveLoad(unittest.TestCase):
    """Tests for cognitive load estimation."""

    def test_empty_wm_low_load(self):
        wm = WorkingMemory(capacity=5, content_dim=32)
        cc = CognitiveController(wm)
        load = cc.estimate_cognitive_load()
        self.assertAlmostEqual(load, 0.0, places=1)

    def test_full_wm_higher_load(self):
        wm = WorkingMemory(capacity=5, content_dim=32)
        cc = CognitiveController(wm)
        for i in range(5):
            content = _make_content(32, seed=3800 + i)
            wm.add(content=content, slot_type=WorkspaceSlotType.SENSORY,
                    source="test")
        load = cc.estimate_cognitive_load()
        # wm_load = 5/5 = 1.0, so at minimum 0.4 * 1.0 = 0.4
        self.assertGreaterEqual(load, 0.4)

    def test_goals_increase_load(self):
        wm = WorkingMemory(capacity=5, content_dim=32)
        cc = CognitiveController(wm)
        load_no_goals = cc.estimate_cognitive_load()
        for i in range(3):
            wm.add_goal(_make_content(32, seed=3900 + i))
        load_with_goals = cc.estimate_cognitive_load()
        self.assertGreater(load_with_goals, load_no_goals)

    def test_task_stack_increases_load(self):
        wm = WorkingMemory(capacity=5, content_dim=32)
        cc = CognitiveController(wm)
        load_before = cc.estimate_cognitive_load()
        wm.push_task("t1")
        wm.push_task("t2")
        wm.push_task("t3")
        load_after = cc.estimate_cognitive_load()
        self.assertGreater(load_after, load_before)


class TestCognitiveControllerConflictMonitoring(unittest.TestCase):
    """Tests for conflict monitoring."""

    def test_conflict_decays_with_few_items(self):
        wm = WorkingMemory(capacity=5, content_dim=32)
        cc = CognitiveController(wm)
        cc.conflict_level = 0.5
        cc.monitor_conflict()
        self.assertLess(cc.conflict_level, 0.5)

    def test_conflict_returns_float(self):
        wm = WorkingMemory(capacity=5, content_dim=32)
        cc = CognitiveController(wm)
        conflict = cc.monitor_conflict()
        self.assertIsInstance(conflict, float)
        self.assertGreaterEqual(conflict, 0.0)
        self.assertLessEqual(conflict, 1.0)


class TestCognitiveControllerShouldSwitch(unittest.TestCase):
    """Tests for should_switch decision."""

    def test_high_priority_triggers_switch(self):
        wm = WorkingMemory(capacity=5, content_dim=32)
        cc = CognitiveController(wm)
        # With zero load, threshold = 0.6
        result = cc.should_switch(0.9)
        self.assertTrue(result)

    def test_low_priority_no_switch(self):
        wm = WorkingMemory(capacity=5, content_dim=32)
        cc = CognitiveController(wm)
        result = cc.should_switch(0.1)
        self.assertFalse(result)


class TestCognitiveControllerStep(unittest.TestCase):
    """Tests for the step method."""

    def test_step_returns_metrics(self):
        wm = WorkingMemory(capacity=5, content_dim=32)
        cc = CognitiveController(wm)
        metrics = cc.step()
        self.assertIn('conflict_level', metrics)
        self.assertIn('cognitive_load', metrics)
        self.assertIn('inhibition_count', metrics)

    def test_step_decays_inhibitions(self):
        wm = WorkingMemory(capacity=5, content_dim=32)
        cc = CognitiveController(wm)
        content = _make_content(32, seed=4000)
        wm.add(content=content, slot_type=WorkspaceSlotType.ACTION,
                source="test", item_id="decay_test")
        cc.inhibit_response("decay_test")
        initial_inh = cc.inhibitions["decay_test"]
        cc.step()
        # Inhibition should have decayed
        if "decay_test" in cc.inhibitions:
            self.assertLess(cc.inhibitions["decay_test"], initial_inh)


class TestCognitiveControllerGetStats(unittest.TestCase):
    """Tests for CognitiveController.get_stats()."""

    def test_get_stats_keys(self):
        wm = WorkingMemory(capacity=5, content_dim=32)
        cc = CognitiveController(wm)
        stats = cc.get_stats()
        expected_keys = {
            'conflict_level', 'cognitive_load', 'active_inhibitions',
            'tasks_completed', 'current_task', 'wm_stats',
        }
        self.assertEqual(set(stats.keys()), expected_keys)

    def test_get_stats_after_operations(self):
        wm = WorkingMemory(capacity=5, content_dim=32)
        cc = CognitiveController(wm)
        cc.switch_task("task1")
        cc.switch_task("task2")
        stats = cc.get_stats()
        self.assertEqual(stats['tasks_completed'], 1)
        self.assertEqual(stats['current_task'], "task2")
        self.assertIsInstance(stats['wm_stats'], dict)


# ---------- WorkspaceItem Tests ----------

class TestWorkspaceItem(unittest.TestCase):
    """Tests for the WorkspaceItem dataclass."""

    def test_creation(self):
        item = _make_workspace_item("wi_1", seed=4100)
        self.assertEqual(item.item_id, "wi_1")
        self.assertEqual(item.slot_type, WorkspaceSlotType.SENSORY)
        self.assertAlmostEqual(item.activation, 1.0)

    def test_compute_priority_range(self):
        item = _make_workspace_item("wi_2", seed=4200)
        priority = item.compute_priority()
        self.assertIsInstance(priority, float)
        # Priority should be reasonable (not negative, bounded)
        self.assertGreaterEqual(priority, 0.0)
        self.assertLessEqual(priority, 2.0)

    def test_higher_salience_higher_priority(self):
        low = _make_workspace_item("lo", seed=4300, salience=0.1, relevance=0.5)
        high = _make_workspace_item("hi", seed=4300, salience=0.9, relevance=0.5)
        self.assertGreater(high.compute_priority(), low.compute_priority())

    def test_higher_relevance_higher_priority(self):
        low = _make_workspace_item("lo", seed=4400, salience=0.5, relevance=0.1)
        high = _make_workspace_item("hi", seed=4400, salience=0.5, relevance=0.9)
        self.assertGreater(high.compute_priority(), low.compute_priority())

    def test_default_metadata_empty(self):
        item = _make_workspace_item("wi_3", seed=4500)
        self.assertEqual(item.metadata, {})

    def test_default_decay_rate(self):
        item = _make_workspace_item("wi_4", seed=4600)
        self.assertAlmostEqual(item.decay_rate, 0.1)

    def test_default_consolidation_count(self):
        item = _make_workspace_item("wi_5", seed=4700)
        self.assertEqual(item.consolidation_count, 0)


# ---------- Enum Tests ----------

class TestWorkspaceSlotType(unittest.TestCase):
    """Tests for WorkspaceSlotType enum."""

    def test_all_values(self):
        expected = {"sensory", "semantic", "episodic", "goal", "action",
                    "language", "reasoning"}
        actual = {e.value for e in WorkspaceSlotType}
        self.assertEqual(actual, expected)

    def test_enum_identity(self):
        self.assertEqual(WorkspaceSlotType.SENSORY.value, "sensory")
        self.assertEqual(WorkspaceSlotType.SEMANTIC.value, "semantic")
        self.assertEqual(WorkspaceSlotType.EPISODIC.value, "episodic")
        self.assertEqual(WorkspaceSlotType.GOAL.value, "goal")
        self.assertEqual(WorkspaceSlotType.ACTION.value, "action")
        self.assertEqual(WorkspaceSlotType.LANGUAGE.value, "language")
        self.assertEqual(WorkspaceSlotType.REASONING.value, "reasoning")


class TestAttentionTypeEnum(unittest.TestCase):
    """Tests for AttentionType enum."""

    def test_all_values(self):
        expected = {"bottom_up", "top_down", "executive"}
        actual = {e.value for e in AttentionType}
        self.assertEqual(actual, expected)

    def test_enum_identity(self):
        self.assertEqual(AttentionType.BOTTOM_UP.value, "bottom_up")
        self.assertEqual(AttentionType.TOP_DOWN.value, "top_down")
        self.assertEqual(AttentionType.EXECUTIVE.value, "executive")


# ---------- Additional Integration-style Tests ----------

class TestWorkingMemoryRefreshAndRemove(unittest.TestCase):
    """Tests for refresh and remove operations."""

    def setUp(self):
        self.wm = WorkingMemory(capacity=5, content_dim=32)
        content = _make_content(32, seed=4800)
        self.item_id = self.wm.add(
            content=content, slot_type=WorkspaceSlotType.SENSORY,
            source="test", item_id="refresh_me",
        )

    def test_refresh_existing(self):
        self.wm.slots[self.item_id].activation = 0.5
        result = self.wm.refresh(self.item_id)
        self.assertTrue(result)
        self.assertAlmostEqual(self.wm.slots[self.item_id].activation, 1.0)
        self.assertEqual(self.wm.total_refreshes, 1)

    def test_refresh_increments_consolidation(self):
        self.wm.refresh(self.item_id)
        self.assertEqual(self.wm.slots[self.item_id].consolidation_count, 1)
        self.wm.refresh(self.item_id)
        self.assertEqual(self.wm.slots[self.item_id].consolidation_count, 2)

    def test_refresh_nonexistent(self):
        result = self.wm.refresh("does_not_exist")
        self.assertFalse(result)

    def test_remove_existing(self):
        result = self.wm.remove(self.item_id)
        self.assertTrue(result)
        self.assertNotIn(self.item_id, self.wm.slots)

    def test_remove_nonexistent(self):
        result = self.wm.remove("nope")
        self.assertFalse(result)

    def test_clear(self):
        self.wm.clear()
        self.assertEqual(len(self.wm.slots), 0)


class TestWorkingMemoryGoals(unittest.TestCase):
    """Tests for goal management."""

    def setUp(self):
        self.wm = WorkingMemory(capacity=5, content_dim=32)

    def test_set_goal(self):
        goal = _make_content(32, seed=4900)
        self.wm.set_goal(goal)
        self.assertEqual(len(self.wm.current_goals), 1)

    def test_set_goal_replaces(self):
        self.wm.set_goal(_make_content(32, seed=5000))
        self.wm.set_goal(_make_content(32, seed=5001))
        self.assertEqual(len(self.wm.current_goals), 1)

    def test_add_goal_appends(self):
        self.wm.set_goal(_make_content(32, seed=5100))
        self.wm.add_goal(_make_content(32, seed=5101))
        self.assertEqual(len(self.wm.current_goals), 2)

    def test_clear_goals(self):
        self.wm.set_goal(_make_content(32, seed=5200))
        self.wm.add_goal(_make_content(32, seed=5201))
        self.wm.clear_goals()
        self.assertEqual(len(self.wm.current_goals), 0)

    def test_goal_normalization(self):
        goal = np.ones(32) * 10.0
        self.wm.set_goal(goal)
        stored = self.wm.current_goals[0]
        norm = np.linalg.norm(stored)
        self.assertAlmostEqual(norm, 1.0, places=5)

    def test_goal_padding(self):
        goal = np.ones(16)
        self.wm.set_goal(goal)
        stored = self.wm.current_goals[0]
        self.assertEqual(len(stored), 32)

    def test_goal_truncation(self):
        goal = np.ones(64)
        self.wm.set_goal(goal)
        stored = self.wm.current_goals[0]
        self.assertEqual(len(stored), 32)


class TestWorkingMemoryBroadcast(unittest.TestCase):
    """Tests for the broadcast mechanism."""

    def test_broadcast_listener_called(self):
        wm = WorkingMemory(capacity=5, content_dim=32, broadcast_threshold=0.0)
        received = []
        wm.register_broadcast_listener(lambda item: received.append(item.item_id))
        content = _make_content(32, seed=5300)
        item_id = wm.add(content=content, slot_type=WorkspaceSlotType.SENSORY,
                         source="test", salience=0.8, relevance=0.8)
        # With broadcast_threshold=0.0, item should be broadcast
        self.assertGreater(len(received), 0)

    def test_broadcast_history_tracked(self):
        wm = WorkingMemory(capacity=5, content_dim=32, broadcast_threshold=0.0)
        content = _make_content(32, seed=5400)
        wm.add(content=content, slot_type=WorkspaceSlotType.SENSORY,
                source="test", salience=0.9, relevance=0.9)
        self.assertGreater(len(wm.broadcast_history), 0)


class TestWorkingMemorySerialization(unittest.TestCase):
    """Tests for serialize/deserialize round-trip."""

    def test_serialize_empty(self):
        wm = WorkingMemory(capacity=5, content_dim=32)
        data = wm.serialize()
        self.assertEqual(data['capacity'], 5)
        self.assertEqual(data['content_dim'], 32)
        self.assertEqual(data['slots'], {})

    def test_serialize_deserialize_roundtrip(self):
        wm = WorkingMemory(capacity=5, content_dim=32)
        content = _make_content(32, seed=5500)
        wm.add(content=content, slot_type=WorkspaceSlotType.SEMANTIC,
                source="test", item_id="ser_item")
        wm.push_task("ser_task")
        wm.set_goal(_make_content(32, seed=5501))

        data = wm.serialize()
        wm2 = WorkingMemory.deserialize(data)

        self.assertEqual(wm2.capacity, 5)
        self.assertEqual(wm2.content_dim, 32)
        self.assertIn("ser_item", wm2.slots)
        self.assertEqual(wm2.current_task, "ser_task")
        self.assertEqual(len(wm2.current_goals), 1)


class TestWorkingMemoryQueryBySimilarity(unittest.TestCase):
    """Tests for query_by_similarity()."""

    def test_query_returns_correct_count(self):
        wm = WorkingMemory(capacity=10, content_dim=32)
        for i in range(5):
            content = _make_content(32, seed=5600 + i)
            wm.add(content=content, slot_type=WorkspaceSlotType.SENSORY,
                    source="test")
        query = _make_content(32, seed=5600)  # Same seed as first item
        results = wm.query_by_similarity(query, top_k=3)
        self.assertEqual(len(results), 3)

    def test_query_most_similar_first(self):
        wm = WorkingMemory(capacity=10, content_dim=32)
        # Add item with known content
        seed_target = 5700
        content_target = _make_content(32, seed=seed_target)
        wm.add(content=content_target, slot_type=WorkspaceSlotType.SENSORY,
                source="test", item_id="target")
        # Add other items
        for i in range(3):
            content = _make_content(32, seed=5800 + i)
            wm.add(content=content, slot_type=WorkspaceSlotType.SENSORY,
                    source="test")
        # Query with same content as target
        results = wm.query_by_similarity(content_target, top_k=4)
        self.assertEqual(results[0][0].item_id, "target")
        self.assertGreater(results[0][1], results[-1][1])


class TestWorkingMemoryStep(unittest.TestCase):
    """Tests for the step() dynamics."""

    def test_step_decays_activation(self):
        wm = WorkingMemory(capacity=5, content_dim=32, decay_rate=0.5)
        content = _make_content(32, seed=5900)
        item_id = wm.add(content=content, slot_type=WorkspaceSlotType.SENSORY,
                         source="test", item_id="decay_item")
        initial = wm.slots[item_id].activation
        wm.step()
        if item_id in wm.slots:
            self.assertLess(wm.slots[item_id].activation, initial)

    def test_step_returns_broadcast_list(self):
        wm = WorkingMemory(capacity=5, content_dim=32)
        result = wm.step()
        self.assertIsInstance(result, list)


if __name__ == '__main__':
    unittest.main()
