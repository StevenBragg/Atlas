"""
Comprehensive tests for the World Model system.

Tests cover WorldModel initialization, object tracking and permanence,
physics simulation, state transitions, and serialization/deserialization.

All tests are deterministic and pass reliably by using fixed random seeds
and avoiding timing-sensitive assertions.
"""

import os
import sys
import unittest
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from self_organizing_av_system.core.backend import xp
from self_organizing_av_system.core.world_model import (
    WorldModel,
    WorldObject,
    PhysicsModel,
    StateTransition,
    Variable,
    CausalEdge,
    CausalObject,
    ObjectState,
    PhysicsProperty,
    CausalRelationType,
)


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _make_embedding(state_dim, seed=0):
    """Create a deterministic unit-norm embedding vector."""
    rng = np.random.RandomState(seed)
    vec = rng.randn(state_dim)
    vec /= np.linalg.norm(vec) + 1e-8
    return vec


def _make_observation(state_dim, position=None, seed=0, properties=None):
    """Build an observation dict suitable for WorldModel.observe()."""
    embedding = _make_embedding(state_dim, seed=seed)
    if position is None:
        position = np.zeros(3)
    obs = {
        'embedding': embedding,
        'position': np.asarray(position, dtype=float),
    }
    if properties is not None:
        obs['properties'] = properties
    return obs


# ===================================================================
# Enum and dataclass tests
# ===================================================================

class TestObjectStateEnum(unittest.TestCase):
    """Tests for the ObjectState enum."""

    def test_visible_value(self):
        self.assertEqual(ObjectState.VISIBLE.value, "visible")

    def test_occluded_value(self):
        self.assertEqual(ObjectState.OCCLUDED.value, "occluded")

    def test_predicted_value(self):
        self.assertEqual(ObjectState.PREDICTED.value, "predicted")

    def test_unknown_value(self):
        self.assertEqual(ObjectState.UNKNOWN.value, "unknown")

    def test_all_members(self):
        names = {m.name for m in ObjectState}
        self.assertEqual(names, {"VISIBLE", "OCCLUDED", "PREDICTED", "UNKNOWN"})


class TestPhysicsPropertyEnum(unittest.TestCase):
    """Tests for the PhysicsProperty enum."""

    def test_position_value(self):
        self.assertEqual(PhysicsProperty.POSITION.value, "position")

    def test_velocity_value(self):
        self.assertEqual(PhysicsProperty.VELOCITY.value, "velocity")

    def test_member_count(self):
        self.assertEqual(len(PhysicsProperty), 8)


class TestCausalRelationTypeEnum(unittest.TestCase):
    """Tests for the CausalRelationType enum."""

    def test_causes_value(self):
        self.assertEqual(CausalRelationType.CAUSES.value, "causes")

    def test_prevents_value(self):
        self.assertEqual(CausalRelationType.PREVENTS.value, "prevents")

    def test_correlates_value(self):
        self.assertEqual(CausalRelationType.CORRELATES.value, "correlates")

    def test_all_members(self):
        names = {m.name for m in CausalRelationType}
        expected = {"CAUSES", "PREVENTS", "ENABLES", "REQUIRES", "CORRELATES"}
        self.assertEqual(names, expected)


# ===================================================================
# Dataclass tests
# ===================================================================

class TestWorldObjectDataclass(unittest.TestCase):
    """Tests for the WorldObject dataclass."""

    def _make_obj(self):
        return WorldObject(
            object_id="test_obj",
            embedding=np.array([1.0, 0.0, 0.0]),
            position=np.array([0.0, 0.0, 0.0]),
            velocity=np.array([0.0, 0.0, 0.0]),
        )

    def test_default_state_is_visible(self):
        obj = self._make_obj()
        self.assertEqual(obj.state, ObjectState.VISIBLE)

    def test_default_confidence(self):
        obj = self._make_obj()
        self.assertEqual(obj.confidence, 1.0)

    def test_default_permanence_strength(self):
        obj = self._make_obj()
        self.assertEqual(obj.permanence_strength, 0.0)

    def test_default_update_count(self):
        obj = self._make_obj()
        self.assertEqual(obj.update_count, 0)

    def test_default_properties_empty(self):
        obj = self._make_obj()
        self.assertEqual(obj.properties, {})

    def test_update_increments_count(self):
        obj = self._make_obj()
        obj.update(np.array([0.0, 1.0, 0.0]), np.array([1.0, 0.0, 0.0]))
        self.assertEqual(obj.update_count, 1)

    def test_update_changes_position(self):
        obj = self._make_obj()
        new_pos = np.array([5.0, 3.0, 1.0])
        obj.update(np.array([1.0, 0.0, 0.0]), new_pos)
        np.testing.assert_array_equal(obj.position, new_pos)

    def test_update_sets_state_visible(self):
        obj = self._make_obj()
        obj.state = ObjectState.OCCLUDED
        obj.update(np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]))
        self.assertEqual(obj.state, ObjectState.VISIBLE)

    def test_update_increases_permanence(self):
        obj = self._make_obj()
        initial = obj.permanence_strength
        obj.update(np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]))
        self.assertGreater(obj.permanence_strength, initial)

    def test_update_embedding_normalization(self):
        """After update, embedding should remain approximately unit norm."""
        obj = self._make_obj()
        obj.update(np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 0.0]))
        norm = np.linalg.norm(obj.embedding)
        self.assertAlmostEqual(norm, 1.0, places=5)

    def test_update_velocity_on_second_update(self):
        """Velocity should be updated after the first update (update_count > 0)."""
        obj = self._make_obj()
        # First update sets position and increments count
        obj.update(np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]))
        # Second update: position changed, so velocity should be non-zero
        obj.update(np.array([1.0, 0.0, 0.0]), np.array([10.0, 0.0, 0.0]))
        # velocity should have a positive x-component
        self.assertGreater(obj.velocity[0], 0.0)

    def test_update_confidence_capped_at_one(self):
        """Confidence should never exceed 1.0."""
        obj = self._make_obj()
        for _ in range(50):
            obj.update(np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]))
        self.assertLessEqual(obj.confidence, 1.0)

    def test_update_permanence_capped_at_one(self):
        """Permanence strength should never exceed 1.0."""
        obj = self._make_obj()
        for _ in range(50):
            obj.update(np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]))
        self.assertLessEqual(obj.permanence_strength, 1.0)


class TestPhysicsModelDataclass(unittest.TestCase):
    """Tests for the PhysicsModel dataclass."""

    def test_default_gravity(self):
        pm = PhysicsModel()
        np.testing.assert_array_almost_equal(pm.gravity, [0.0, -9.8, 0.0])

    def test_default_friction(self):
        pm = PhysicsModel()
        self.assertAlmostEqual(pm.friction, 0.1)

    def test_default_elasticity(self):
        pm = PhysicsModel()
        self.assertAlmostEqual(pm.collision_elasticity, 0.7)

    def test_default_transition_matrix_none(self):
        pm = PhysicsModel()
        self.assertIsNone(pm.transition_matrix)

    def test_predict_stationary_object(self):
        """A stationary object at origin should fall due to gravity."""
        pm = PhysicsModel()
        pos = np.array([0.0, 100.0, 0.0])
        vel = np.zeros(3)
        new_pos, new_vel = pm.predict_next_state(pos, vel, dt=1.0)
        # Gravity: acceleration = [0, -9.8, 0] (no friction on zero vel)
        # new_vel = [0, -9.8, 0]
        # new_pos = [0, 100, 0] + [0,0,0]*1 + 0.5*[0,-9.8,0]*1 = [0, 95.1, 0]
        np.testing.assert_array_almost_equal(new_vel, [0.0, -9.8, 0.0])
        np.testing.assert_array_almost_equal(new_pos, [0.0, 95.1, 0.0])

    def test_predict_moving_object_friction(self):
        """A horizontally moving object should decelerate due to friction."""
        pm = PhysicsModel()
        pos = np.zeros(3)
        vel = np.array([10.0, 0.0, 0.0])
        new_pos, new_vel = pm.predict_next_state(pos, vel, dt=1.0)
        # Friction force in x: -0.1 * 10 = -1.0
        # acceleration_x = 0 - 1.0 = -1.0
        # new_vel_x = 10 + (-1)*1 = 9.0
        self.assertAlmostEqual(new_vel[0], 9.0)

    def test_predict_zero_dt(self):
        """With dt=0, position and velocity should not change."""
        pm = PhysicsModel()
        pos = np.array([1.0, 2.0, 3.0])
        vel = np.array([4.0, 5.0, 6.0])
        new_pos, new_vel = pm.predict_next_state(pos, vel, dt=0.0)
        np.testing.assert_array_almost_equal(new_pos, pos)
        np.testing.assert_array_almost_equal(new_vel, vel)

    def test_predict_custom_gravity(self):
        """Physics model with custom gravity should use it."""
        pm = PhysicsModel(gravity=np.array([0.0, 0.0, -5.0]))
        pos = np.zeros(3)
        vel = np.zeros(3)
        new_pos, new_vel = pm.predict_next_state(pos, vel, dt=1.0)
        self.assertAlmostEqual(new_vel[2], -5.0)


class TestStateTransitionDataclass(unittest.TestCase):
    """Tests for the StateTransition dataclass."""

    def test_creation(self):
        st = StateTransition(
            state_before=np.zeros(4),
            action=None,
            state_after=np.ones(4),
            timestamp=0.0,
        )
        np.testing.assert_array_equal(st.state_before, np.zeros(4))
        np.testing.assert_array_equal(st.state_after, np.ones(4))
        self.assertIsNone(st.action)
        self.assertEqual(st.timestamp, 0.0)

    def test_default_prediction_error(self):
        st = StateTransition(
            state_before=np.zeros(4),
            action=None,
            state_after=np.ones(4),
            timestamp=0.0,
        )
        self.assertEqual(st.prediction_error, 0.0)

    def test_with_action(self):
        action = np.array([1.0, 2.0])
        st = StateTransition(
            state_before=np.zeros(4),
            action=action,
            state_after=np.ones(4),
            timestamp=1.0,
        )
        np.testing.assert_array_equal(st.action, action)


class TestVariableDataclass(unittest.TestCase):
    """Tests for the Variable dataclass."""

    def test_creation(self):
        v = Variable(name="temperature", value=25.0)
        self.assertEqual(v.name, "temperature")
        self.assertEqual(v.value, 25.0)

    def test_default_observable(self):
        v = Variable(name="x", value=0)
        self.assertTrue(v.observable)

    def test_default_continuous(self):
        v = Variable(name="x", value=0)
        self.assertFalse(v.continuous)

    def test_default_prior_none(self):
        v = Variable(name="x", value=0)
        self.assertIsNone(v.prior_distribution)

    def test_with_prior(self):
        prior = {"low": 0.3, "high": 0.7}
        v = Variable(name="x", value="low", prior_distribution=prior)
        self.assertEqual(v.prior_distribution, prior)


class TestCausalEdgeDataclass(unittest.TestCase):
    """Tests for the CausalEdge dataclass."""

    def test_creation(self):
        edge = CausalEdge(
            cause="A",
            effect="B",
            relation_type=CausalRelationType.CAUSES,
        )
        self.assertEqual(edge.cause, "A")
        self.assertEqual(edge.effect, "B")
        self.assertEqual(edge.relation_type, CausalRelationType.CAUSES)

    def test_default_strength(self):
        edge = CausalEdge(
            cause="A", effect="B",
            relation_type=CausalRelationType.CAUSES,
        )
        self.assertEqual(edge.strength, 1.0)

    def test_default_mechanism_none(self):
        edge = CausalEdge(
            cause="A", effect="B",
            relation_type=CausalRelationType.CAUSES,
        )
        self.assertIsNone(edge.mechanism)

    def test_default_learned_from_intervention(self):
        edge = CausalEdge(
            cause="A", effect="B",
            relation_type=CausalRelationType.CAUSES,
        )
        self.assertFalse(edge.learned_from_intervention)

    def test_with_mechanism(self):
        fn = lambda x: x * 2
        edge = CausalEdge(
            cause="A", effect="B",
            relation_type=CausalRelationType.CAUSES,
            mechanism=fn,
        )
        self.assertIs(edge.mechanism, fn)


class TestCausalObjectDataclass(unittest.TestCase):
    """Tests for the CausalObject dataclass."""

    def test_creation(self):
        obj = CausalObject(
            object_id="car_0",
            object_type="vehicle",
            position=np.array([1.0, 2.0, 3.0]),
            velocity=np.zeros(3),
            properties={"color": "red"},
            last_observed=0.0,
        )
        self.assertEqual(obj.object_id, "car_0")
        self.assertEqual(obj.object_type, "vehicle")
        self.assertEqual(obj.properties["color"], "red")

    def test_default_existence_confidence(self):
        obj = CausalObject(
            object_id="x", object_type="t",
            position=np.zeros(3), velocity=np.zeros(3),
            properties={}, last_observed=0.0,
        )
        self.assertEqual(obj.existence_confidence, 1.0)

    def test_default_permanence(self):
        obj = CausalObject(
            object_id="x", object_type="t",
            position=np.zeros(3), velocity=np.zeros(3),
            properties={}, last_observed=0.0,
        )
        self.assertTrue(obj.permanence)


# ===================================================================
# WorldModel initialization tests
# ===================================================================

class TestWorldModelInitialization(unittest.TestCase):
    """Tests for WorldModel.__init__."""

    def test_default_parameters(self):
        wm = WorldModel(random_seed=42)
        self.assertEqual(wm.state_dim, 64)
        self.assertEqual(wm.max_objects, 100)
        self.assertAlmostEqual(wm.permanence_decay, 0.01)
        self.assertEqual(wm.prediction_horizon, 10)
        self.assertAlmostEqual(wm.learning_rate, 0.05)

    def test_custom_parameters(self):
        wm = WorldModel(
            state_dim=32,
            max_objects=50,
            permanence_decay=0.05,
            prediction_horizon=5,
            learning_rate=0.1,
            random_seed=0,
        )
        self.assertEqual(wm.state_dim, 32)
        self.assertEqual(wm.max_objects, 50)
        self.assertAlmostEqual(wm.permanence_decay, 0.05)
        self.assertEqual(wm.prediction_horizon, 5)
        self.assertAlmostEqual(wm.learning_rate, 0.1)

    def test_empty_objects_at_init(self):
        wm = WorldModel(random_seed=42)
        self.assertEqual(len(wm.objects), 0)

    def test_object_counter_starts_at_zero(self):
        wm = WorldModel(random_seed=42)
        self.assertEqual(wm.object_counter, 0)

    def test_physics_model_created(self):
        wm = WorldModel(random_seed=42)
        self.assertIsInstance(wm.physics, PhysicsModel)

    def test_transition_weights_shape(self):
        dim = 32
        wm = WorldModel(state_dim=dim, random_seed=42)
        self.assertEqual(wm.transition_weights.shape, (dim, dim))

    def test_action_weights_shape(self):
        dim = 32
        wm = WorldModel(state_dim=dim, random_seed=42)
        self.assertEqual(wm.action_weights.shape, (dim, wm.action_dim))

    def test_transition_weights_stable(self):
        """Transition weights should have spectral radius < 1 for stability."""
        wm = WorldModel(state_dim=16, random_seed=42)
        norm = np.linalg.norm(wm.transition_weights)
        self.assertLessEqual(norm, 1.0 + 1e-6)

    def test_initial_statistics_zero(self):
        wm = WorldModel(random_seed=42)
        self.assertEqual(wm.total_observations, 0)
        self.assertEqual(wm.objects_created, 0)
        self.assertEqual(wm.objects_removed, 0)
        self.assertEqual(wm.total_predictions, 0)
        self.assertEqual(wm.total_surprises, 0)

    def test_state_covariance_shape(self):
        dim = 16
        wm = WorldModel(state_dim=dim, random_seed=42)
        self.assertEqual(wm.state_covariance.shape, (dim, dim))

    def test_prediction_errors_deque_empty(self):
        wm = WorldModel(random_seed=42)
        self.assertEqual(len(wm.prediction_errors), 0)

    def test_state_transitions_deque_empty(self):
        wm = WorldModel(random_seed=42)
        self.assertEqual(len(wm.state_transitions), 0)

    def test_random_seed_determinism(self):
        """Two models with the same seed should have identical weights."""
        wm1 = WorldModel(state_dim=16, random_seed=99)
        wm2 = WorldModel(state_dim=16, random_seed=99)
        np.testing.assert_array_equal(
            wm1.transition_weights, wm2.transition_weights
        )
        np.testing.assert_array_equal(
            wm1.action_weights, wm2.action_weights
        )


# ===================================================================
# Object tracking and permanence tests
# ===================================================================

class TestWorldModelObjectTracking(unittest.TestCase):
    """Tests for object creation, matching, and tracking."""

    def setUp(self):
        self.dim = 16
        self.wm = WorldModel(state_dim=self.dim, random_seed=42)

    def test_observe_creates_new_object(self):
        """Observing a new object should create it in the model."""
        obs = [_make_observation(self.dim, position=[1, 2, 3], seed=0)]
        result = self.wm.observe(obs)
        self.assertEqual(result['total_objects'], 1)
        self.assertEqual(len(result['new_objects']), 1)
        self.assertEqual(self.wm.objects_created, 1)

    def test_observe_increments_total_observations(self):
        obs = [_make_observation(self.dim, seed=0)]
        self.wm.observe(obs)
        self.assertEqual(self.wm.total_observations, 1)
        self.wm.observe(obs)
        self.assertEqual(self.wm.total_observations, 2)

    def test_observe_multiple_objects(self):
        """Observing multiple distinct objects should create all of them."""
        observations = [
            _make_observation(self.dim, position=[0, 0, 0], seed=0),
            _make_observation(self.dim, position=[100, 100, 100], seed=1),
            _make_observation(self.dim, position=[200, 200, 200], seed=2),
        ]
        result = self.wm.observe(observations)
        self.assertEqual(result['total_objects'], 3)
        self.assertEqual(len(result['new_objects']), 3)

    def test_observe_returns_expected_keys(self):
        obs = [_make_observation(self.dim, seed=0)]
        result = self.wm.observe(obs)
        for key in ('updates', 'new_objects', 'occluded', 'predictions', 'total_objects'):
            self.assertIn(key, result)

    def test_object_ids_sequential(self):
        """Object IDs should follow the pattern obj_0, obj_1, ..."""
        obs1 = [_make_observation(self.dim, position=[0, 0, 0], seed=0)]
        obs2 = [_make_observation(self.dim, position=[100, 100, 100], seed=1)]
        self.wm.observe(obs1)
        self.wm.observe(obs2)
        ids = sorted(self.wm.objects.keys())
        self.assertEqual(ids, ["obj_0", "obj_1"])

    def test_new_object_starts_with_moderate_confidence(self):
        """Newly created objects should have confidence = 0.5."""
        obs = [_make_observation(self.dim, position=[0, 0, 0], seed=0)]
        self.wm.observe(obs)
        obj = list(self.wm.objects.values())[0]
        self.assertAlmostEqual(obj.confidence, 0.5)

    def test_matched_object_update(self):
        """Re-observing an identical object at the same position should update it."""
        embedding = _make_embedding(self.dim, seed=10)
        obs = [{'embedding': embedding, 'position': np.array([5.0, 5.0, 5.0])}]
        self.wm.observe(obs)
        self.assertEqual(self.wm.objects_created, 1)

        # Observe same object again (same embedding, same position)
        self.wm.observe(obs)
        # Should still be 1 object (updated, not new)
        self.assertEqual(len(self.wm.objects), 1)

    def test_max_objects_enforced(self):
        """When max_objects is exceeded, the least confident should be removed."""
        wm = WorldModel(state_dim=self.dim, max_objects=3, random_seed=42)
        for i in range(5):
            obs = [_make_observation(self.dim, position=[i * 1000, 0, 0], seed=i + 100)]
            wm.observe(obs)
        self.assertLessEqual(len(wm.objects), 3)

    def test_object_properties_stored(self):
        """Properties passed in observation should be stored on the object."""
        obs = [_make_observation(self.dim, seed=0, properties={"color": "blue", "size": 3})]
        self.wm.observe(obs)
        obj = list(self.wm.objects.values())[0]
        self.assertEqual(obj.properties.get("color"), "blue")
        self.assertEqual(obj.properties.get("size"), 3)


class TestWorldModelObjectPermanence(unittest.TestCase):
    """Tests for object permanence behavior when objects are no longer observed."""

    def setUp(self):
        self.dim = 16
        self.wm = WorldModel(
            state_dim=self.dim,
            permanence_decay=0.01,
            random_seed=42,
        )

    def test_unobserved_object_becomes_occluded(self):
        """An object that is not re-observed should transition to OCCLUDED."""
        # Create an object
        emb1 = _make_embedding(self.dim, seed=50)
        obs1 = [{'embedding': emb1, 'position': np.array([0.0, 0.0, 0.0])}]
        self.wm.observe(obs1)
        obj_id = list(self.wm.objects.keys())[0]
        self.assertEqual(self.wm.objects[obj_id].state, ObjectState.VISIBLE)

        # Observe a completely different object (first one is not seen)
        emb2 = _make_embedding(self.dim, seed=999)
        obs2 = [{'embedding': emb2, 'position': np.array([500.0, 500.0, 500.0])}]
        self.wm.observe(obs2)

        # The first object should now be occluded (if it still exists)
        if obj_id in self.wm.objects:
            self.assertEqual(self.wm.objects[obj_id].state, ObjectState.OCCLUDED)

    def test_occluded_object_confidence_decays(self):
        """Confidence of an occluded object should decrease over time."""
        emb = _make_embedding(self.dim, seed=50)
        obs = [{'embedding': emb, 'position': np.array([0.0, 0.0, 0.0])}]
        self.wm.observe(obs)
        obj_id = list(self.wm.objects.keys())[0]
        initial_confidence = self.wm.objects[obj_id].confidence

        # Observe nothing relevant several times (object becomes occluded)
        for i in range(5):
            emb_other = _make_embedding(self.dim, seed=200 + i)
            obs_other = [{'embedding': emb_other, 'position': np.array([1000.0 * (i + 1), 0.0, 0.0])}]
            self.wm.observe(obs_other)

        if obj_id in self.wm.objects:
            self.assertLess(self.wm.objects[obj_id].confidence, initial_confidence)

    def test_query_existing_object(self):
        """query_object_permanence on a tracked object should return exists=True."""
        obs = [_make_observation(self.dim, seed=0)]
        self.wm.observe(obs)
        obj_id = list(self.wm.objects.keys())[0]
        result = self.wm.query_object_permanence(obj_id)
        self.assertTrue(result['exists'])
        self.assertEqual(result['object_id'], obj_id)
        self.assertIn('confidence', result)
        self.assertIn('position', result)
        self.assertIn('velocity', result)

    def test_query_nonexistent_object(self):
        """query_object_permanence on an unknown ID should return exists=False."""
        result = self.wm.query_object_permanence("nonexistent_obj")
        self.assertFalse(result['exists'])
        self.assertEqual(result['message'], 'Object not in world model')

    def test_query_returns_state_value(self):
        """The returned state should be a string from ObjectState values."""
        obs = [_make_observation(self.dim, seed=0)]
        self.wm.observe(obs)
        obj_id = list(self.wm.objects.keys())[0]
        result = self.wm.query_object_permanence(obj_id)
        self.assertIn(result['state'], [s.value for s in ObjectState])

    def test_occluded_object_position_predicted(self):
        """An occluded object should have its position predicted by the physics model."""
        emb = _make_embedding(self.dim, seed=50)
        obs = [{'embedding': emb, 'position': np.array([0.0, 100.0, 0.0])}]
        self.wm.observe(obs)
        obj_id = list(self.wm.objects.keys())[0]
        initial_pos = self.wm.objects[obj_id].position.copy()

        # Make it occluded by observing something else
        emb_other = _make_embedding(self.dim, seed=999)
        obs_other = [{'embedding': emb_other, 'position': np.array([500.0, 500.0, 500.0])}]
        self.wm.observe(obs_other)

        if obj_id in self.wm.objects:
            # Position should have changed due to physics prediction (gravity)
            current_pos = self.wm.objects[obj_id].position
            self.assertFalse(np.allclose(current_pos, initial_pos),
                             "Occluded object position should be predicted by physics")


# ===================================================================
# Physics simulation tests
# ===================================================================

class TestWorldModelPhysicsSimulation(unittest.TestCase):
    """Tests for the physics simulation within WorldModel."""

    def test_physics_model_default_gravity(self):
        wm = WorldModel(random_seed=42)
        np.testing.assert_array_almost_equal(
            wm.physics.gravity, [0.0, -9.8, 0.0]
        )

    def test_physics_predict_returns_tuple(self):
        wm = WorldModel(random_seed=42)
        pos = np.array([0.0, 0.0, 0.0])
        vel = np.array([1.0, 0.0, 0.0])
        result = wm.physics.predict_next_state(pos, vel, dt=0.1)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

    def test_physics_predict_shapes(self):
        wm = WorldModel(random_seed=42)
        pos = np.array([0.0, 0.0, 0.0])
        vel = np.array([1.0, 2.0, 3.0])
        new_pos, new_vel = wm.physics.predict_next_state(pos, vel, dt=1.0)
        self.assertEqual(new_pos.shape, (3,))
        self.assertEqual(new_vel.shape, (3,))

    def test_learn_physics_from_trajectory(self):
        """learn_physics should update gravity estimate from a trajectory."""
        wm = WorldModel(state_dim=16, random_seed=42)
        original_gravity = wm.physics.gravity.copy()

        # Create a trajectory with constant downward acceleration
        trajectory = []
        for t in range(10):
            pos = np.array([0.0, 100.0 - 0.5 * 5.0 * t * t, 0.0])
            trajectory.append({'position': pos})

        wm.learn_physics(trajectory)
        # Gravity estimate should have changed
        self.assertFalse(
            np.allclose(wm.physics.gravity, original_gravity),
            "Gravity should be updated after learning from trajectory"
        )

    def test_learn_physics_short_trajectory_noop(self):
        """Trajectory with fewer than 3 points should not change physics."""
        wm = WorldModel(state_dim=16, random_seed=42)
        original_gravity = wm.physics.gravity.copy()
        wm.learn_physics([{'position': np.zeros(3)}, {'position': np.ones(3)}])
        np.testing.assert_array_equal(wm.physics.gravity, original_gravity)

    def test_learn_physics_empty_trajectory_noop(self):
        """Empty trajectory should not cause errors."""
        wm = WorldModel(state_dim=16, random_seed=42)
        wm.learn_physics([])  # Should not raise

    def test_generate_predictions_returns_list(self):
        """_generate_predictions should return a list."""
        wm = WorldModel(state_dim=16, random_seed=42)
        obs = [_make_observation(16, seed=0)]
        wm.observe(obs)
        predictions = wm._generate_predictions()
        self.assertIsInstance(predictions, list)

    def test_predictions_contain_future_positions(self):
        """Each object prediction should include future position forecasts."""
        wm = WorldModel(state_dim=16, random_seed=42)
        obs = [_make_observation(16, position=[0, 50, 0], seed=0)]
        wm.observe(obs)
        predictions = wm._generate_predictions()
        # Find the prediction for the object (not global state)
        obj_preds = [p for p in predictions if 'object_id' in p]
        self.assertGreater(len(obj_preds), 0)
        self.assertIn('future_positions', obj_preds[0])
        self.assertGreater(len(obj_preds[0]['future_positions']), 0)

    def test_prediction_confidence_decays(self):
        """Future position predictions should have decreasing confidence."""
        wm = WorldModel(state_dim=16, prediction_horizon=5, random_seed=42)
        obs = [_make_observation(16, seed=0)]
        wm.observe(obs)
        predictions = wm._generate_predictions()
        obj_preds = [p for p in predictions if 'object_id' in p]
        if obj_preds:
            futures = obj_preds[0]['future_positions']
            confidences = [f['confidence'] for f in futures]
            # Each successive confidence should be less than or equal to the previous
            for i in range(1, len(confidences)):
                self.assertLessEqual(confidences[i], confidences[i - 1])


# ===================================================================
# State transition tests
# ===================================================================

class TestWorldModelStateTransitions(unittest.TestCase):
    """Tests for state transition learning and prediction."""

    def setUp(self):
        self.dim = 16
        self.wm = WorldModel(state_dim=self.dim, random_seed=42)

    def test_learn_transition_stores_transition(self):
        """Providing global_state should store a state transition."""
        state = np.random.RandomState(0).randn(self.dim)
        obs = [_make_observation(self.dim, seed=0)]
        self.wm.observe(obs, global_state=state)
        self.assertEqual(len(self.wm.state_transitions), 1)

    def test_multiple_transitions_stored(self):
        """Multiple observations with global_state should accumulate transitions."""
        rng = np.random.RandomState(0)
        for i in range(5):
            state = rng.randn(self.dim)
            obs = [_make_observation(self.dim, seed=i, position=[i * 100, 0, 0])]
            self.wm.observe(obs, global_state=state)
        self.assertEqual(len(self.wm.state_transitions), 5)

    def test_transition_deque_maxlen(self):
        """State transitions deque should respect its maxlen of 1000."""
        self.assertEqual(self.wm.state_transitions.maxlen, 1000)

    def test_prediction_errors_tracked(self):
        """After multiple transitions, prediction errors should be tracked."""
        rng = np.random.RandomState(0)
        for i in range(3):
            state = rng.randn(self.dim)
            obs = [_make_observation(self.dim, seed=i, position=[i * 100, 0, 0])]
            self.wm.observe(obs, global_state=state)
        # First transition has no predecessor to compare, so errors start from 2nd
        self.assertGreater(len(self.wm.prediction_errors), 0)

    def test_total_predictions_incremented(self):
        """total_predictions should increment with each learned transition (after first)."""
        rng = np.random.RandomState(0)
        for i in range(3):
            state = rng.randn(self.dim)
            obs = [_make_observation(self.dim, seed=i, position=[i * 100, 0, 0])]
            self.wm.observe(obs, global_state=state)
        self.assertGreater(self.wm.total_predictions, 0)

    def test_transition_weights_updated(self):
        """Transition weights should change after learning from transitions."""
        initial_weights = self.wm.transition_weights.copy()
        rng = np.random.RandomState(0)
        for i in range(5):
            state = rng.randn(self.dim)
            obs = [_make_observation(self.dim, seed=i, position=[i * 100, 0, 0])]
            self.wm.observe(obs, global_state=state)
        self.assertFalse(
            np.allclose(self.wm.transition_weights, initial_weights),
            "Transition weights should be updated via Hebbian learning"
        )

    def test_transition_weights_norm_bounded(self):
        """Transition weights norm should remain bounded (<= 1)."""
        rng = np.random.RandomState(0)
        for i in range(20):
            state = rng.randn(self.dim) * 10  # Large states to stress normalization
            obs = [_make_observation(self.dim, seed=i, position=[i * 100, 0, 0])]
            self.wm.observe(obs, global_state=state)
        norm = np.linalg.norm(self.wm.transition_weights)
        self.assertLessEqual(norm, 1.0 + 1e-6)

    def test_predict_with_intervention_empty_without_transitions(self):
        """predict_with_intervention should return empty list with no prior transitions."""
        action = np.ones(self.wm.action_dim)
        result = self.wm.predict_with_intervention(action, steps=3)
        self.assertEqual(result, [])

    def test_predict_with_intervention_returns_states(self):
        """predict_with_intervention should return predicted future states."""
        rng = np.random.RandomState(0)
        # Build up some transitions first
        for i in range(3):
            state = rng.randn(self.dim)
            obs = [_make_observation(self.dim, seed=i, position=[i * 100, 0, 0])]
            self.wm.observe(obs, global_state=state)

        action = rng.randn(self.wm.action_dim)
        predictions = self.wm.predict_with_intervention(action, steps=5)
        self.assertEqual(len(predictions), 5)
        for pred in predictions:
            self.assertEqual(pred.shape, (self.dim,))

    def test_predict_with_intervention_action_padding(self):
        """Action shorter than action_dim should be zero-padded."""
        rng = np.random.RandomState(0)
        for i in range(3):
            state = rng.randn(self.dim)
            obs = [_make_observation(self.dim, seed=i, position=[i * 100, 0, 0])]
            self.wm.observe(obs, global_state=state)

        short_action = np.array([1.0, 2.0])  # Much shorter than action_dim=16
        predictions = self.wm.predict_with_intervention(short_action, steps=2)
        self.assertEqual(len(predictions), 2)

    def test_predict_with_intervention_normalized(self):
        """Predictions from intervention should be normalized."""
        rng = np.random.RandomState(0)
        for i in range(3):
            state = rng.randn(self.dim)
            obs = [_make_observation(self.dim, seed=i, position=[i * 100, 0, 0])]
            self.wm.observe(obs, global_state=state)

        action = rng.randn(self.wm.action_dim)
        predictions = self.wm.predict_with_intervention(action, steps=3)
        for pred in predictions:
            norm = np.linalg.norm(pred)
            self.assertAlmostEqual(norm, 1.0, places=4)


# ===================================================================
# Hidden state inference tests
# ===================================================================

class TestWorldModelHiddenStateInference(unittest.TestCase):
    """Tests for Bayesian hidden state inference."""

    def setUp(self):
        self.dim = 16
        self.wm = WorldModel(state_dim=self.dim, random_seed=42)

    def test_infer_hidden_state_returns_tuple(self):
        obs = np.random.RandomState(0).randn(self.dim)
        result = self.wm.infer_hidden_state(obs)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

    def test_infer_hidden_state_shapes(self):
        obs = np.random.RandomState(0).randn(self.dim)
        state, uncertainty = self.wm.infer_hidden_state(obs)
        self.assertEqual(state.shape, (self.dim,))
        self.assertEqual(uncertainty.shape, (self.dim,))

    def test_infer_hidden_state_with_prior(self):
        obs = np.random.RandomState(0).randn(self.dim)
        prior = np.random.RandomState(1).randn(self.dim)
        state, uncertainty = self.wm.infer_hidden_state(obs, prior_state=prior)
        self.assertEqual(state.shape, (self.dim,))

    def test_infer_hidden_state_short_observation(self):
        """Short observation vectors should be zero-padded."""
        obs = np.array([1.0, 2.0, 3.0])  # Much shorter than state_dim
        state, uncertainty = self.wm.infer_hidden_state(obs)
        self.assertEqual(state.shape, (self.dim,))

    def test_uncertainty_non_negative(self):
        """Uncertainty (diagonal of covariance) should be non-negative."""
        obs = np.random.RandomState(0).randn(self.dim)
        _, uncertainty = self.wm.infer_hidden_state(obs)
        self.assertTrue(np.all(uncertainty >= -1e-10))


# ===================================================================
# World state embedding tests
# ===================================================================

class TestWorldModelStateEmbedding(unittest.TestCase):
    """Tests for get_world_state_embedding."""

    def setUp(self):
        self.dim = 16
        self.wm = WorldModel(state_dim=self.dim, random_seed=42)

    def test_empty_world_returns_zeros(self):
        """With no objects, embedding should be all zeros."""
        embedding = self.wm.get_world_state_embedding()
        self.assertEqual(embedding.shape, (self.dim,))
        np.testing.assert_array_equal(embedding, np.zeros(self.dim))

    def test_with_objects_returns_nonzero(self):
        """With objects, embedding should generally be non-zero."""
        obs = [_make_observation(self.dim, seed=0)]
        self.wm.observe(obs)
        # Manually boost permanence_strength so the weight is nonzero
        obj = list(self.wm.objects.values())[0]
        obj.permanence_strength = 0.5
        embedding = self.wm.get_world_state_embedding()
        self.assertEqual(embedding.shape, (self.dim,))
        self.assertGreater(np.linalg.norm(embedding), 0.0)

    def test_embedding_normalized(self):
        """World state embedding should be approximately unit norm (when non-zero objects exist)."""
        obs = [_make_observation(self.dim, seed=0)]
        self.wm.observe(obs)
        obj = list(self.wm.objects.values())[0]
        obj.permanence_strength = 0.5
        embedding = self.wm.get_world_state_embedding()
        if np.linalg.norm(embedding) > 0:
            self.assertAlmostEqual(np.linalg.norm(embedding), 1.0, places=4)


# ===================================================================
# Serialization / deserialization tests
# ===================================================================

class TestWorldModelSerialization(unittest.TestCase):
    """Tests for serialize() and deserialize()."""

    def setUp(self):
        self.dim = 16
        self.wm = WorldModel(state_dim=self.dim, max_objects=50, random_seed=42)
        # Add some objects
        for i in range(3):
            obs = [_make_observation(self.dim, position=[i * 100, i * 10, 0], seed=i)]
            self.wm.observe(obs)

    def test_serialize_returns_dict(self):
        data = self.wm.serialize()
        self.assertIsInstance(data, dict)

    def test_serialize_contains_expected_keys(self):
        data = self.wm.serialize()
        expected_keys = {
            'state_dim', 'max_objects', 'permanence_decay',
            'prediction_horizon', 'learning_rate', 'objects',
            'transition_weights', 'action_weights', 'physics', 'stats',
        }
        for key in expected_keys:
            self.assertIn(key, data, f"Missing key: {key}")

    def test_serialize_state_dim(self):
        data = self.wm.serialize()
        self.assertEqual(data['state_dim'], self.dim)

    def test_serialize_max_objects(self):
        data = self.wm.serialize()
        self.assertEqual(data['max_objects'], 50)

    def test_serialize_objects_count(self):
        data = self.wm.serialize()
        self.assertEqual(len(data['objects']), len(self.wm.objects))

    def test_serialize_object_fields(self):
        """Each serialized object should contain required fields."""
        data = self.wm.serialize()
        for oid, obj_data in data['objects'].items():
            for field in ('object_id', 'embedding', 'position', 'velocity',
                          'properties', 'state', 'confidence', 'permanence_strength'):
                self.assertIn(field, obj_data, f"Missing field '{field}' in object {oid}")

    def test_serialize_physics(self):
        data = self.wm.serialize()
        self.assertIn('gravity', data['physics'])
        self.assertIn('friction', data['physics'])
        self.assertIn('collision_elasticity', data['physics'])

    def test_serialize_transition_weights_shape(self):
        data = self.wm.serialize()
        tw = np.array(data['transition_weights'])
        self.assertEqual(tw.shape, (self.dim, self.dim))

    def test_serialize_stats(self):
        data = self.wm.serialize()
        stats = data['stats']
        self.assertIn('total_objects', stats)
        self.assertIn('visible_objects', stats)
        self.assertIn('total_observations', stats)

    def test_deserialize_roundtrip_parameters(self):
        """Deserialize should restore the model parameters."""
        data = self.wm.serialize()
        restored = WorldModel.deserialize(data)
        self.assertEqual(restored.state_dim, self.wm.state_dim)
        self.assertEqual(restored.max_objects, self.wm.max_objects)
        self.assertAlmostEqual(restored.permanence_decay, self.wm.permanence_decay)
        self.assertEqual(restored.prediction_horizon, self.wm.prediction_horizon)
        self.assertAlmostEqual(restored.learning_rate, self.wm.learning_rate)

    def test_deserialize_roundtrip_weights(self):
        """Transition and action weights should survive serialization roundtrip."""
        data = self.wm.serialize()
        restored = WorldModel.deserialize(data)
        np.testing.assert_array_almost_equal(
            restored.transition_weights, self.wm.transition_weights
        )
        np.testing.assert_array_almost_equal(
            restored.action_weights, self.wm.action_weights
        )

    def test_deserialize_roundtrip_physics(self):
        """Physics parameters should survive serialization roundtrip."""
        data = self.wm.serialize()
        restored = WorldModel.deserialize(data)
        np.testing.assert_array_almost_equal(
            restored.physics.gravity, self.wm.physics.gravity
        )
        self.assertAlmostEqual(restored.physics.friction, self.wm.physics.friction)
        self.assertAlmostEqual(
            restored.physics.collision_elasticity,
            self.wm.physics.collision_elasticity,
        )

    def test_deserialize_roundtrip_objects(self):
        """Objects should survive serialization roundtrip."""
        data = self.wm.serialize()
        restored = WorldModel.deserialize(data)
        self.assertEqual(len(restored.objects), len(self.wm.objects))
        for oid in self.wm.objects:
            self.assertIn(oid, restored.objects)
            orig = self.wm.objects[oid]
            rest = restored.objects[oid]
            self.assertEqual(rest.object_id, orig.object_id)
            np.testing.assert_array_almost_equal(rest.embedding, orig.embedding)
            np.testing.assert_array_almost_equal(rest.position, orig.position)
            np.testing.assert_array_almost_equal(rest.velocity, orig.velocity)
            self.assertEqual(rest.state, orig.state)
            self.assertAlmostEqual(rest.confidence, orig.confidence)
            self.assertAlmostEqual(rest.permanence_strength, orig.permanence_strength)

    def test_deserialize_empty_model(self):
        """Serializing and deserializing an empty model should work."""
        empty_wm = WorldModel(state_dim=8, random_seed=0)
        data = empty_wm.serialize()
        restored = WorldModel.deserialize(data)
        self.assertEqual(len(restored.objects), 0)
        self.assertEqual(restored.state_dim, 8)


# ===================================================================
# get_stats tests
# ===================================================================

class TestWorldModelStats(unittest.TestCase):
    """Tests for get_stats()."""

    def setUp(self):
        self.dim = 16
        self.wm = WorldModel(state_dim=self.dim, random_seed=42)

    def test_stats_returns_dict(self):
        stats = self.wm.get_stats()
        self.assertIsInstance(stats, dict)

    def test_stats_expected_keys(self):
        stats = self.wm.get_stats()
        expected = {
            'total_objects', 'visible_objects', 'occluded_objects',
            'total_observations', 'objects_created', 'objects_removed',
            'total_predictions', 'total_surprises', 'surprise_rate',
            'mean_prediction_error', 'physics_gravity', 'physics_friction',
            'transition_matrix_norm',
        }
        for key in expected:
            self.assertIn(key, stats, f"Missing stats key: {key}")

    def test_stats_initial_values(self):
        stats = self.wm.get_stats()
        self.assertEqual(stats['total_objects'], 0)
        self.assertEqual(stats['visible_objects'], 0)
        self.assertEqual(stats['occluded_objects'], 0)
        self.assertEqual(stats['total_observations'], 0)
        self.assertEqual(stats['objects_created'], 0)
        self.assertEqual(stats['objects_removed'], 0)

    def test_stats_after_observation(self):
        obs = [_make_observation(self.dim, seed=0)]
        self.wm.observe(obs)
        stats = self.wm.get_stats()
        self.assertEqual(stats['total_objects'], 1)
        self.assertEqual(stats['visible_objects'], 1)
        self.assertEqual(stats['total_observations'], 1)
        self.assertEqual(stats['objects_created'], 1)

    def test_stats_surprise_rate_zero_initially(self):
        stats = self.wm.get_stats()
        self.assertEqual(stats['surprise_rate'], 0.0)

    def test_stats_mean_prediction_error_zero_initially(self):
        stats = self.wm.get_stats()
        self.assertEqual(stats['mean_prediction_error'], 0.0)

    def test_stats_physics_gravity_is_list(self):
        stats = self.wm.get_stats()
        self.assertIsInstance(stats['physics_gravity'], list)
        self.assertEqual(len(stats['physics_gravity']), 3)

    def test_stats_transition_matrix_norm_positive(self):
        stats = self.wm.get_stats()
        self.assertGreaterEqual(stats['transition_matrix_norm'], 0.0)


# ===================================================================
# Edge case and robustness tests
# ===================================================================

class TestWorldModelEdgeCases(unittest.TestCase):
    """Tests for edge cases and robustness."""

    def setUp(self):
        self.dim = 16

    def test_observe_empty_list(self):
        """Observing an empty list should not error."""
        wm = WorldModel(state_dim=self.dim, random_seed=42)
        result = wm.observe([])
        self.assertEqual(result['total_objects'], 0)

    def test_observe_short_embedding(self):
        """An embedding shorter than state_dim should be zero-padded."""
        wm = WorldModel(state_dim=self.dim, random_seed=42)
        obs = [{'embedding': np.array([1.0, 0.0]), 'position': np.zeros(3)}]
        result = wm.observe(obs)
        self.assertEqual(result['total_objects'], 1)

    def test_observe_long_embedding(self):
        """An embedding longer than state_dim should be truncated."""
        wm = WorldModel(state_dim=self.dim, random_seed=42)
        obs = [{'embedding': np.ones(100), 'position': np.zeros(3)}]
        result = wm.observe(obs)
        self.assertEqual(result['total_objects'], 1)

    def test_observe_short_position(self):
        """A position shorter than 3D should be zero-padded."""
        wm = WorldModel(state_dim=self.dim, random_seed=42)
        obs = [{'embedding': np.ones(self.dim), 'position': np.array([1.0])}]
        result = wm.observe(obs)
        self.assertEqual(result['total_objects'], 1)
        obj = list(wm.objects.values())[0]
        self.assertEqual(len(obj.position), 3)

    def test_observe_no_embedding_key(self):
        """Observation without embedding should use random default."""
        wm = WorldModel(state_dim=self.dim, random_seed=42)
        obs = [{'position': np.array([1.0, 2.0, 3.0])}]
        result = wm.observe(obs)
        self.assertEqual(result['total_objects'], 1)

    def test_observe_no_position_key(self):
        """Observation without position should default to zeros."""
        wm = WorldModel(state_dim=self.dim, random_seed=42)
        obs = [{'embedding': np.ones(self.dim)}]
        result = wm.observe(obs)
        self.assertEqual(result['total_objects'], 1)
        obj = list(wm.objects.values())[0]
        np.testing.assert_array_equal(obj.position, np.zeros(3))

    def test_learn_transition_short_state(self):
        """A global state shorter than state_dim should be zero-padded."""
        wm = WorldModel(state_dim=self.dim, random_seed=42)
        short_state = np.array([1.0, 2.0])
        obs = [_make_observation(self.dim, seed=0)]
        wm.observe(obs, global_state=short_state)
        self.assertEqual(len(wm.state_transitions), 1)

    def test_consecutive_empty_observations(self):
        """Multiple empty observations should not error."""
        wm = WorldModel(state_dim=self.dim, random_seed=42)
        for _ in range(10):
            wm.observe([])
        self.assertEqual(wm.total_observations, 10)
        self.assertEqual(len(wm.objects), 0)

    def test_observe_with_global_state_none(self):
        """Passing global_state=None should skip transition learning."""
        wm = WorldModel(state_dim=self.dim, random_seed=42)
        obs = [_make_observation(self.dim, seed=0)]
        wm.observe(obs, global_state=None)
        self.assertEqual(len(wm.state_transitions), 0)


if __name__ == "__main__":
    unittest.main()
