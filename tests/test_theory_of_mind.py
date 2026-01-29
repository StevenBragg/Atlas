"""
Comprehensive tests for the Theory of Mind module.

Tests cover:
- TheoryOfMind initialization (default and custom parameters)
- Agent modeling (add, eviction, agent types, relationships)
- Mental state tracking (observe, infer, decay, predict)
- Social reasoning (cooperative, competitive, false belief, perspective)
- AgentType and MentalStateType enum values
- Serialization and deserialization (serialize / deserialize)
- Social learning and trust updates
- Statistics tracking

All tests are deterministic and pass reliably.
"""

import os
import sys
import unittest
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from self_organizing_av_system.core.backend import xp
from self_organizing_av_system.core.theory_of_mind import (
    TheoryOfMind,
    AgentType,
    MentalStateType,
    SocialRelation,
    MentalState,
    AgentModel,
    SocialContext,
    Observation,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
SEED = 42
DIM = 32


def _make_tom(**kwargs):
    """Create a TheoryOfMind with small, deterministic defaults."""
    defaults = dict(embedding_dim=DIM, max_agents=10, random_seed=SEED)
    defaults.update(kwargs)
    return TheoryOfMind(**defaults)


def _vec(dim=DIM, value=None, seed=None):
    """Return a deterministic vector."""
    if value is not None:
        return np.full(dim, value, dtype=np.float64)
    rng = np.random.RandomState(seed if seed is not None else 0)
    return rng.randn(dim)


# ===================================================================
# 1. Enum value tests
# ===================================================================
class TestAgentTypeEnum(unittest.TestCase):
    """Verify every AgentType member and its string value."""

    def test_members(self):
        expected = {
            "SELF": "self",
            "HUMAN": "human",
            "AI": "ai",
            "ANIMAL": "animal",
            "GENERIC": "generic",
        }
        for name, value in expected.items():
            member = AgentType[name]
            self.assertEqual(member.value, value)

    def test_member_count(self):
        self.assertEqual(len(AgentType), 5)

    def test_from_value(self):
        self.assertIs(AgentType("human"), AgentType.HUMAN)
        self.assertIs(AgentType("ai"), AgentType.AI)


class TestMentalStateTypeEnum(unittest.TestCase):
    """Verify every MentalStateType member and its string value."""

    def test_members(self):
        expected = {
            "BELIEF": "belief",
            "DESIRE": "desire",
            "INTENTION": "intention",
            "EMOTION": "emotion",
            "ATTENTION": "attention",
            "KNOWLEDGE": "knowledge",
        }
        for name, value in expected.items():
            member = MentalStateType[name]
            self.assertEqual(member.value, value)

    def test_member_count(self):
        self.assertEqual(len(MentalStateType), 6)


class TestSocialRelationEnum(unittest.TestCase):
    """Verify every SocialRelation member and its string value."""

    def test_members(self):
        expected = {
            "COOPERATIVE": "cooperative",
            "COMPETITIVE": "competitive",
            "NEUTRAL": "neutral",
            "HIERARCHICAL": "hierarchical",
            "RECIPROCAL": "reciprocal",
        }
        for name, value in expected.items():
            member = SocialRelation[name]
            self.assertEqual(member.value, value)

    def test_member_count(self):
        self.assertEqual(len(SocialRelation), 5)


# ===================================================================
# 2. Dataclass tests
# ===================================================================
class TestMentalStateDataclass(unittest.TestCase):
    """Tests for the MentalState dataclass."""

    def test_creation(self):
        content = np.zeros(DIM)
        ms = MentalState(
            state_type=MentalStateType.BELIEF,
            content=content,
            confidence=0.8,
            source="test",
        )
        self.assertEqual(ms.state_type, MentalStateType.BELIEF)
        self.assertAlmostEqual(ms.confidence, 0.8)
        self.assertEqual(ms.source, "test")
        np.testing.assert_array_equal(ms.content, content)
        self.assertIsInstance(ms.timestamp, float)

    def test_decay_reduces_confidence(self):
        ms = MentalState(
            state_type=MentalStateType.INTENTION,
            content=np.zeros(DIM),
            confidence=1.0,
            source="test",
        )
        ms.decay(rate=0.1)
        self.assertAlmostEqual(ms.confidence, 0.9)

    def test_decay_default_rate(self):
        ms = MentalState(
            state_type=MentalStateType.DESIRE,
            content=np.zeros(DIM),
            confidence=1.0,
            source="test",
        )
        ms.decay()
        self.assertAlmostEqual(ms.confidence, 0.99)

    def test_decay_multiple(self):
        ms = MentalState(
            state_type=MentalStateType.EMOTION,
            content=np.zeros(DIM),
            confidence=1.0,
            source="test",
        )
        for _ in range(10):
            ms.decay(rate=0.1)
        expected = 1.0 * (0.9 ** 10)
        self.assertAlmostEqual(ms.confidence, expected, places=6)


class TestAgentModelDataclass(unittest.TestCase):
    """Tests for the AgentModel dataclass."""

    def test_defaults(self):
        am = AgentModel(agent_id="a1", agent_type=AgentType.HUMAN)
        self.assertEqual(am.agent_id, "a1")
        self.assertEqual(am.agent_type, AgentType.HUMAN)
        self.assertIsInstance(am.mental_states, dict)
        self.assertIsInstance(am.beliefs, dict)
        self.assertIsInstance(am.desires, list)
        self.assertIsInstance(am.intentions, list)
        self.assertIsInstance(am.knowledge_state, dict)
        self.assertIsNone(am.personality_embedding)
        self.assertEqual(am.relationship, SocialRelation.NEUTRAL)
        self.assertAlmostEqual(am.trust_level, 0.5)
        self.assertAlmostEqual(am.predictability, 0.5)
        self.assertEqual(len(am.interaction_history), 0)

    def test_custom_values(self):
        personality = np.ones(DIM)
        am = AgentModel(
            agent_id="a2",
            agent_type=AgentType.AI,
            personality_embedding=personality,
            relationship=SocialRelation.COOPERATIVE,
            trust_level=0.9,
        )
        self.assertEqual(am.relationship, SocialRelation.COOPERATIVE)
        self.assertAlmostEqual(am.trust_level, 0.9)
        np.testing.assert_array_equal(am.personality_embedding, personality)


class TestSocialContextDataclass(unittest.TestCase):
    """Tests for the SocialContext dataclass."""

    def test_creation(self):
        sc = SocialContext(
            agents_present=["a1", "a2"],
            shared_attention=np.zeros(DIM),
            social_norms={"politeness": True},
            power_dynamics={"a1": 0.6, "a2": 0.4},
            common_ground={"topic": np.ones(DIM)},
        )
        self.assertEqual(len(sc.agents_present), 2)
        self.assertIn("politeness", sc.social_norms)
        self.assertAlmostEqual(sc.power_dynamics["a1"], 0.6)


class TestObservationDataclass(unittest.TestCase):
    """Tests for the Observation dataclass."""

    def test_creation(self):
        obs = Observation(
            agent_id="a1",
            action=np.ones(DIM),
            context=np.zeros(DIM),
            outcome=None,
        )
        self.assertEqual(obs.agent_id, "a1")
        self.assertIsNone(obs.outcome)
        self.assertIsInstance(obs.timestamp, float)


# ===================================================================
# 3. TheoryOfMind initialization
# ===================================================================
class TestTheoryOfMindInit(unittest.TestCase):
    """Tests for TheoryOfMind.__init__."""

    def test_default_parameters(self):
        tom = TheoryOfMind(random_seed=SEED)
        self.assertEqual(tom.embedding_dim, 64)
        self.assertEqual(tom.max_agents, 50)
        self.assertAlmostEqual(tom.belief_update_rate, 0.1)
        self.assertEqual(tom.simulation_steps, 5)
        self.assertAlmostEqual(tom.learning_rate, 0.05)

    def test_custom_parameters(self):
        tom = _make_tom(
            embedding_dim=16,
            max_agents=5,
            belief_update_rate=0.2,
            simulation_steps=3,
            learning_rate=0.01,
        )
        self.assertEqual(tom.embedding_dim, 16)
        self.assertEqual(tom.max_agents, 5)
        self.assertAlmostEqual(tom.belief_update_rate, 0.2)
        self.assertEqual(tom.simulation_steps, 3)
        self.assertAlmostEqual(tom.learning_rate, 0.01)

    def test_self_model_initialized(self):
        tom = _make_tom()
        self.assertIsNotNone(tom.self_model)
        self.assertEqual(tom.self_model.agent_id, "self")
        self.assertEqual(tom.self_model.agent_type, AgentType.SELF)
        self.assertAlmostEqual(tom.self_model.trust_level, 1.0)
        self.assertAlmostEqual(tom.self_model.predictability, 1.0)
        self.assertEqual(tom.self_model.relationship, SocialRelation.NEUTRAL)
        self.assertIsNotNone(tom.self_model.personality_embedding)
        self.assertEqual(
            tom.self_model.personality_embedding.shape, (DIM,)
        )

    def test_initial_statistics(self):
        tom = _make_tom()
        self.assertEqual(tom.total_inferences, 0)
        self.assertEqual(tom.successful_predictions, 0)
        self.assertEqual(tom.total_predictions, 0)
        self.assertEqual(tom.interaction_count, 0)
        self.assertEqual(len(tom.observations), 0)

    def test_agents_dict_empty(self):
        tom = _make_tom()
        self.assertEqual(len(tom.agents), 0)

    def test_weight_shapes(self):
        tom = _make_tom()
        self.assertEqual(tom.inference_weights.shape, (DIM, DIM))
        self.assertEqual(tom.inference_bias.shape, (DIM,))
        self.assertEqual(tom.prediction_weights.shape, (DIM, DIM))
        self.assertEqual(tom.prediction_bias.shape, (DIM,))


# ===================================================================
# 4. Agent modeling
# ===================================================================
class TestAddAgent(unittest.TestCase):
    """Tests for adding and managing agents."""

    def test_add_default_agent(self):
        tom = _make_tom()
        aid = tom.add_agent()
        self.assertEqual(aid, "agent_0")
        self.assertIn(aid, tom.agents)
        agent = tom.agents[aid]
        self.assertEqual(agent.agent_type, AgentType.GENERIC)
        self.assertEqual(agent.relationship, SocialRelation.NEUTRAL)

    def test_add_typed_agent(self):
        tom = _make_tom()
        aid = tom.add_agent(agent_type=AgentType.HUMAN)
        self.assertEqual(tom.agents[aid].agent_type, AgentType.HUMAN)

    def test_add_agent_with_relationship(self):
        tom = _make_tom()
        aid = tom.add_agent(relationship=SocialRelation.COOPERATIVE)
        self.assertEqual(tom.agents[aid].relationship, SocialRelation.COOPERATIVE)

    def test_add_agent_with_personality(self):
        tom = _make_tom()
        personality = np.ones(DIM) / np.sqrt(DIM)
        aid = tom.add_agent(personality=personality)
        np.testing.assert_array_almost_equal(
            tom.agents[aid].personality_embedding, personality
        )

    def test_auto_personality_normalized(self):
        tom = _make_tom()
        aid = tom.add_agent()
        p = tom.agents[aid].personality_embedding
        norm = np.linalg.norm(p)
        # Should be approximately unit norm (normalized with +1e-8)
        self.assertAlmostEqual(norm, 1.0, places=3)

    def test_agent_counter_increments(self):
        tom = _make_tom()
        a0 = tom.add_agent()
        a1 = tom.add_agent()
        a2 = tom.add_agent()
        self.assertEqual(a0, "agent_0")
        self.assertEqual(a1, "agent_1")
        self.assertEqual(a2, "agent_2")
        self.assertEqual(tom.agent_counter, 3)

    def test_max_agents_eviction(self):
        tom = _make_tom(max_agents=3)
        ids = []
        for _ in range(3):
            ids.append(tom.add_agent())
        self.assertEqual(len(tom.agents), 3)

        # Adding a 4th should evict the oldest (least recently interacted)
        new_id = tom.add_agent()
        self.assertEqual(len(tom.agents), 3)
        self.assertIn(new_id, tom.agents)

    def test_all_agent_types_can_be_added(self):
        tom = _make_tom()
        for at in AgentType:
            if at == AgentType.SELF:
                continue
            aid = tom.add_agent(agent_type=at)
            self.assertEqual(tom.agents[aid].agent_type, at)


# ===================================================================
# 5. Observation and mental state inference
# ===================================================================
class TestObserve(unittest.TestCase):
    """Tests for observing agents and inferring mental states."""

    def test_observe_known_agent(self):
        tom = _make_tom()
        aid = tom.add_agent()
        action = _vec(DIM, seed=1)
        context = _vec(DIM, seed=2)
        result = tom.observe(aid, action, context)
        self.assertIn('intention', result)
        self.assertIn('belief', result)
        self.assertIn('embedding', result['intention'])
        self.assertIn('confidence', result['intention'])

    def test_observe_unknown_agent_auto_adds(self):
        tom = _make_tom()
        action = _vec(DIM, seed=1)
        context = _vec(DIM, seed=2)
        # "unknown_agent" is not in tom.agents, so observe should auto-add
        result = tom.observe("unknown_agent", action, context)
        self.assertIn('intention', result)
        # A new agent should have been created
        self.assertEqual(len(tom.agents), 1)

    def test_observe_updates_interaction_count(self):
        tom = _make_tom()
        aid = tom.add_agent()
        action = _vec(DIM, seed=1)
        context = _vec(DIM, seed=2)
        tom.observe(aid, action, context)
        self.assertEqual(tom.interaction_count, 1)
        tom.observe(aid, action, context)
        self.assertEqual(tom.interaction_count, 2)

    def test_observe_records_observation(self):
        tom = _make_tom()
        aid = tom.add_agent()
        tom.observe(aid, _vec(DIM, seed=1), _vec(DIM, seed=2))
        self.assertEqual(len(tom.observations), 1)

    def test_observe_updates_agent_mental_states(self):
        tom = _make_tom()
        aid = tom.add_agent()
        tom.observe(aid, _vec(DIM, seed=1), _vec(DIM, seed=2))
        agent = tom.agents[aid]
        self.assertIn('current_intention', agent.mental_states)
        self.assertIn('current_belief', agent.mental_states)

    def test_observe_pads_short_vectors(self):
        tom = _make_tom()
        aid = tom.add_agent()
        short_action = np.array([1.0, 2.0, 3.0])
        short_context = np.array([4.0, 5.0])
        result = tom.observe(aid, short_action, short_context)
        self.assertIn('intention', result)
        # Should not raise; padding should handle it

    def test_observe_truncates_long_vectors(self):
        tom = _make_tom()
        aid = tom.add_agent()
        long_action = np.ones(DIM * 3)
        long_context = np.ones(DIM * 3)
        result = tom.observe(aid, long_action, long_context)
        self.assertIn('intention', result)

    def test_observe_with_outcome(self):
        tom = _make_tom()
        aid = tom.add_agent()
        action = _vec(DIM, seed=1)
        context = _vec(DIM, seed=2)
        outcome = _vec(DIM, seed=3)
        result = tom.observe(aid, action, context, outcome=outcome)
        self.assertIn('intention', result)

    def test_desire_inferred_after_three_observations(self):
        tom = _make_tom()
        aid = tom.add_agent()
        for i in range(3):
            tom.observe(aid, _vec(DIM, seed=i), _vec(DIM, seed=i + 100))
        agent = tom.agents[aid]
        self.assertIn('inferred_desire', agent.mental_states)

    def test_desire_not_inferred_before_three_observations(self):
        tom = _make_tom()
        aid = tom.add_agent()
        for i in range(2):
            tom.observe(aid, _vec(DIM, seed=i), _vec(DIM, seed=i + 100))
        agent = tom.agents[aid]
        self.assertNotIn('inferred_desire', agent.mental_states)

    def test_intentions_list_capped_at_ten(self):
        tom = _make_tom()
        aid = tom.add_agent()
        for i in range(15):
            tom.observe(aid, _vec(DIM, seed=i), _vec(DIM, seed=i + 100))
        agent = tom.agents[aid]
        self.assertLessEqual(len(agent.intentions), 10)

    def test_total_inferences_tracked(self):
        tom = _make_tom()
        aid = tom.add_agent()
        for i in range(5):
            tom.observe(aid, _vec(DIM, seed=i), _vec(DIM, seed=i + 100))
        self.assertEqual(tom.total_inferences, 5)


# ===================================================================
# 6. Prediction
# ===================================================================
class TestPredictAction(unittest.TestCase):
    """Tests for predicting agent actions."""

    def test_predict_unknown_agent_returns_zeros(self):
        tom = _make_tom()
        predicted, confidence = tom.predict_action("nonexistent", _vec(DIM, seed=0))
        np.testing.assert_array_equal(predicted, np.zeros(DIM))
        self.assertAlmostEqual(confidence, 0.0)

    def test_predict_returns_correct_shapes(self):
        tom = _make_tom()
        aid = tom.add_agent()
        tom.observe(aid, _vec(DIM, seed=1), _vec(DIM, seed=2))
        predicted, confidence = tom.predict_action(aid, _vec(DIM, seed=3))
        self.assertEqual(predicted.shape, (DIM,))
        self.assertIsInstance(confidence, float)

    def test_predict_normalizes_output(self):
        tom = _make_tom()
        aid = tom.add_agent()
        tom.observe(aid, _vec(DIM, seed=1), _vec(DIM, seed=2))
        predicted, _ = tom.predict_action(aid, _vec(DIM, seed=3))
        norm = np.linalg.norm(predicted)
        # Should be approximately unit norm
        self.assertAlmostEqual(norm, 1.0, places=3)

    def test_predict_stores_predicted_action(self):
        tom = _make_tom()
        aid = tom.add_agent()
        tom.observe(aid, _vec(DIM, seed=1), _vec(DIM, seed=2))
        tom.predict_action(aid, _vec(DIM, seed=3))
        agent = tom.agents[aid]
        self.assertIn('predicted_action', agent.mental_states)

    def test_predict_pads_short_context(self):
        tom = _make_tom()
        aid = tom.add_agent()
        predicted, confidence = tom.predict_action(aid, np.array([1.0, 2.0]))
        self.assertEqual(predicted.shape, (DIM,))

    def test_prediction_accuracy_updates(self):
        """After a prediction then observation, accuracy stats update."""
        tom = _make_tom()
        aid = tom.add_agent()
        context = _vec(DIM, seed=0)
        action = _vec(DIM, seed=1)
        # First observation to populate mental states
        tom.observe(aid, action, context)
        # Now predict
        tom.predict_action(aid, context)
        # Then observe again -- this should trigger accuracy update
        tom.observe(aid, action, context)
        self.assertGreaterEqual(tom.total_predictions, 1)


# ===================================================================
# 7. False belief inference
# ===================================================================
class TestFalseBelief(unittest.TestCase):
    """Tests for false belief inference."""

    def test_false_belief_unknown_agent(self):
        tom = _make_tom()
        result = tom.infer_false_belief("no_such", np.zeros(DIM))
        self.assertIn('error', result)

    def test_false_belief_with_observation(self):
        tom = _make_tom()
        aid = tom.add_agent()
        true_state = np.ones(DIM)
        agent_obs = np.zeros(DIM)
        result = tom.infer_false_belief(aid, true_state, agent_observation=agent_obs)
        self.assertIn('has_false_belief', result)
        self.assertIn('divergence', result)
        self.assertIn('similarity', result)
        self.assertTrue(result['has_false_belief'])  # very different states

    def test_false_belief_aligned_states(self):
        tom = _make_tom()
        aid = tom.add_agent()
        state = np.ones(DIM) / np.sqrt(DIM)
        result = tom.infer_false_belief(aid, state, agent_observation=state)
        # Same state => no false belief
        self.assertFalse(result['has_false_belief'])
        self.assertAlmostEqual(result['similarity'], 1.0, places=3)

    def test_false_belief_uses_stored_belief(self):
        tom = _make_tom()
        aid = tom.add_agent()
        # Give agent a context belief via observation
        context = _vec(DIM, seed=10)
        tom.observe(aid, _vec(DIM, seed=11), context)
        # Now infer without explicit agent_observation
        true_state = _vec(DIM, seed=99)
        result = tom.infer_false_belief(aid, true_state)
        self.assertIn('has_false_belief', result)

    def test_false_belief_no_belief_no_observation(self):
        tom = _make_tom()
        aid = tom.add_agent()
        # Agent has no beliefs at all, no observation provided
        true_state = np.ones(DIM)
        result = tom.infer_false_belief(aid, true_state)
        # Agent belief defaults to zeros, so should have false belief
        self.assertTrue(result['has_false_belief'])

    def test_false_belief_pads_short_vectors(self):
        tom = _make_tom()
        aid = tom.add_agent()
        short_true = np.array([1.0, 0.0])
        short_obs = np.array([0.0, 1.0])
        result = tom.infer_false_belief(aid, short_true, agent_observation=short_obs)
        self.assertIn('has_false_belief', result)


# ===================================================================
# 8. Perspective simulation
# ===================================================================
class TestSimulatePerspective(unittest.TestCase):
    """Tests for perspective simulation."""

    def test_unknown_agent_returns_empty(self):
        tom = _make_tom()
        result = tom.simulate_perspective("nonexistent", np.zeros(DIM))
        self.assertEqual(result, [])

    def test_returns_correct_number_of_steps(self):
        tom = _make_tom(simulation_steps=7)
        aid = tom.add_agent()
        trajectory = tom.simulate_perspective(aid, _vec(DIM, seed=0))
        self.assertEqual(len(trajectory), 7)

    def test_custom_steps_override(self):
        tom = _make_tom(simulation_steps=5)
        aid = tom.add_agent()
        trajectory = tom.simulate_perspective(aid, _vec(DIM, seed=0), steps=3)
        self.assertEqual(len(trajectory), 3)

    def test_trajectory_embeddings_correct_shape(self):
        tom = _make_tom()
        aid = tom.add_agent()
        trajectory = tom.simulate_perspective(aid, _vec(DIM, seed=0))
        for embedding in trajectory:
            self.assertEqual(embedding.shape, (DIM,))

    def test_trajectory_embeddings_are_normalized(self):
        tom = _make_tom()
        aid = tom.add_agent()
        trajectory = tom.simulate_perspective(aid, _vec(DIM, seed=0))
        for embedding in trajectory:
            norm = np.linalg.norm(embedding)
            self.assertAlmostEqual(norm, 1.0, places=3)

    def test_pads_short_scenario(self):
        tom = _make_tom()
        aid = tom.add_agent()
        short = np.array([1.0, 2.0])
        trajectory = tom.simulate_perspective(aid, short, steps=2)
        self.assertEqual(len(trajectory), 2)


# ===================================================================
# 9. Cooperative reasoning
# ===================================================================
class TestCooperativeReasoning(unittest.TestCase):
    """Tests for cooperative reasoning."""

    def test_cooperative_with_no_valid_agents(self):
        tom = _make_tom()
        result = tom.reason_cooperatively(
            agent_ids=["fake1", "fake2"],
            shared_goal=_vec(DIM, seed=0),
            context=_vec(DIM, seed=1),
        )
        self.assertEqual(result['agents'], [])
        self.assertAlmostEqual(result['coordination_quality'], 0.0)
        self.assertIsNone(result['recommended_strategy'])

    def test_cooperative_structure(self):
        tom = _make_tom()
        a1 = tom.add_agent()
        a2 = tom.add_agent()
        result = tom.reason_cooperatively(
            agent_ids=[a1, a2],
            shared_goal=_vec(DIM, seed=0),
            context=_vec(DIM, seed=1),
        )
        self.assertIn('agents', result)
        self.assertIn('coordination_quality', result)
        self.assertIn('trust_levels', result)
        self.assertIn('recommended_strategy', result)
        self.assertEqual(len(result['agents']), 2)

    def test_trust_levels_reported(self):
        tom = _make_tom()
        a1 = tom.add_agent()
        result = tom.reason_cooperatively(
            agent_ids=[a1],
            shared_goal=_vec(DIM, seed=0),
            context=_vec(DIM, seed=1),
        )
        self.assertIn(a1, result['trust_levels'])

    def test_recommended_strategy_is_string_or_none(self):
        tom = _make_tom()
        a1 = tom.add_agent()
        result = tom.reason_cooperatively(
            agent_ids=[a1],
            shared_goal=_vec(DIM, seed=0),
            context=_vec(DIM, seed=1),
        )
        strategy = result['recommended_strategy']
        self.assertIn(
            strategy,
            ['full_cooperation', 'cautious_cooperation', 'independent_action'],
        )


# ===================================================================
# 10. Competitive reasoning
# ===================================================================
class TestCompetitiveReasoning(unittest.TestCase):
    """Tests for competitive reasoning."""

    def test_unknown_opponent(self):
        tom = _make_tom()
        result = tom.reason_competitively("none", np.ones(DIM))
        self.assertIn('error', result)

    def test_competitive_structure(self):
        tom = _make_tom()
        opp = tom.add_agent()
        result = tom.reason_competitively(opp, _vec(DIM, seed=0))
        self.assertIn('opponent_id', result)
        self.assertIn('predicted_opponent_action', result)
        self.assertIn('opponent_confidence', result)
        self.assertIn('goal_conflict', result)
        self.assertIn('recommended_action', result)
        self.assertIn('strategy', result)
        self.assertIn('opponent_predictability', result)

    def test_competitive_with_explicit_opponent_goal(self):
        tom = _make_tom()
        opp = tom.add_agent()
        own = _vec(DIM, seed=0)
        opp_goal = _vec(DIM, seed=1)
        result = tom.reason_competitively(opp, own, opponent_goal=opp_goal)
        self.assertIsInstance(result['goal_conflict'], float)

    def test_competitive_with_context(self):
        tom = _make_tom()
        opp = tom.add_agent()
        result = tom.reason_competitively(
            opp, _vec(DIM, seed=0), context=_vec(DIM, seed=2)
        )
        self.assertIn('strategy', result)

    def test_strategy_values(self):
        tom = _make_tom()
        opp = tom.add_agent()
        result = tom.reason_competitively(opp, _vec(DIM, seed=0))
        self.assertIn(
            result['strategy'],
            ['counter', 'unpredictable', 'potential_cooperation'],
        )


# ===================================================================
# 11. Social learning
# ===================================================================
class TestLearnFromObservation(unittest.TestCase):
    """Tests for social learning."""

    def test_learn_from_known_agent(self):
        tom = _make_tom()
        aid = tom.add_agent()
        demo = [(_vec(DIM, seed=i), _vec(DIM, seed=i + 50)) for i in range(4)]
        result = tom.learn_from_observation(aid, demo)
        self.assertEqual(result['demonstrator_id'], aid)
        self.assertEqual(result['patterns_learned'], 4)

    def test_learn_from_unknown_agent_auto_adds(self):
        tom = _make_tom()
        demo = [(_vec(DIM, seed=0), _vec(DIM, seed=1))]
        result = tom.learn_from_observation("unknown_demo", demo)
        self.assertEqual(result['patterns_learned'], 1)
        # Agent should be auto-created
        self.assertIn(result['demonstrator_id'], tom.agents)

    def test_trust_increases_after_learning(self):
        tom = _make_tom()
        aid = tom.add_agent()
        initial_trust = tom.agents[aid].trust_level
        demo = [(_vec(DIM, seed=i), _vec(DIM, seed=i + 50)) for i in range(3)]
        tom.learn_from_observation(aid, demo)
        self.assertGreater(tom.agents[aid].trust_level, initial_trust)

    def test_trust_capped_at_one(self):
        tom = _make_tom()
        aid = tom.add_agent()
        tom.agents[aid].trust_level = 0.99
        demo = [(_vec(DIM, seed=0), _vec(DIM, seed=1))]
        tom.learn_from_observation(aid, demo)
        self.assertLessEqual(tom.agents[aid].trust_level, 1.0)

    def test_empty_demonstration(self):
        tom = _make_tom()
        aid = tom.add_agent()
        result = tom.learn_from_observation(aid, [])
        self.assertEqual(result['patterns_learned'], 0)


# ===================================================================
# 12. Trust and relationship management
# ===================================================================
class TestTrustAndRelationship(unittest.TestCase):
    """Tests for trust updates and relationship setting."""

    def test_positive_trust_update(self):
        tom = _make_tom()
        aid = tom.add_agent()
        initial = tom.agents[aid].trust_level
        tom.update_trust(aid, 1.0)
        self.assertGreater(tom.agents[aid].trust_level, initial)

    def test_negative_trust_update(self):
        tom = _make_tom()
        aid = tom.add_agent()
        initial = tom.agents[aid].trust_level
        tom.update_trust(aid, -1.0)
        self.assertLess(tom.agents[aid].trust_level, initial)

    def test_trust_clamped_above_zero(self):
        tom = _make_tom()
        aid = tom.add_agent()
        tom.agents[aid].trust_level = 0.05
        tom.update_trust(aid, -10.0)
        self.assertGreaterEqual(tom.agents[aid].trust_level, 0.0)

    def test_trust_clamped_below_one(self):
        tom = _make_tom()
        aid = tom.add_agent()
        tom.agents[aid].trust_level = 0.95
        tom.update_trust(aid, 10.0)
        self.assertLessEqual(tom.agents[aid].trust_level, 1.0)

    def test_trust_update_unknown_agent_no_error(self):
        tom = _make_tom()
        # Should not raise
        tom.update_trust("nonexistent", 0.5)

    def test_set_relationship(self):
        tom = _make_tom()
        aid = tom.add_agent()
        tom.set_relationship(aid, SocialRelation.COMPETITIVE)
        self.assertEqual(tom.agents[aid].relationship, SocialRelation.COMPETITIVE)

    def test_set_relationship_unknown_agent_no_error(self):
        tom = _make_tom()
        # Should not raise
        tom.set_relationship("nonexistent", SocialRelation.COOPERATIVE)

    def test_all_relationships_settable(self):
        tom = _make_tom()
        aid = tom.add_agent()
        for rel in SocialRelation:
            tom.set_relationship(aid, rel)
            self.assertEqual(tom.agents[aid].relationship, rel)


# ===================================================================
# 13. Statistics
# ===================================================================
class TestGetStats(unittest.TestCase):
    """Tests for get_stats method."""

    def test_stats_keys(self):
        tom = _make_tom()
        stats = tom.get_stats()
        expected_keys = {
            'total_agents',
            'total_inferences',
            'total_predictions',
            'successful_predictions',
            'prediction_accuracy',
            'interaction_count',
            'observation_history_size',
            'avg_trust',
            'avg_predictability',
        }
        self.assertEqual(set(stats.keys()), expected_keys)

    def test_stats_initial_values(self):
        tom = _make_tom()
        stats = tom.get_stats()
        self.assertEqual(stats['total_agents'], 0)
        self.assertEqual(stats['total_inferences'], 0)
        self.assertEqual(stats['total_predictions'], 0)
        self.assertEqual(stats['successful_predictions'], 0)
        self.assertAlmostEqual(stats['prediction_accuracy'], 0.0)
        self.assertEqual(stats['interaction_count'], 0)
        self.assertEqual(stats['observation_history_size'], 0)
        self.assertAlmostEqual(stats['avg_trust'], 0.0)
        self.assertAlmostEqual(stats['avg_predictability'], 0.0)

    def test_stats_after_activity(self):
        tom = _make_tom()
        aid = tom.add_agent()
        for i in range(3):
            tom.observe(aid, _vec(DIM, seed=i), _vec(DIM, seed=i + 100))
        stats = tom.get_stats()
        self.assertEqual(stats['total_agents'], 1)
        self.assertEqual(stats['total_inferences'], 3)
        self.assertEqual(stats['interaction_count'], 3)
        self.assertEqual(stats['observation_history_size'], 3)
        self.assertAlmostEqual(stats['avg_trust'], 0.5)


# ===================================================================
# 14. Serialization / Deserialization
# ===================================================================
class TestSerialization(unittest.TestCase):
    """Tests for serialize and deserialize."""

    def test_serialize_empty_tom(self):
        tom = _make_tom()
        data = tom.serialize()
        self.assertEqual(data['embedding_dim'], DIM)
        self.assertEqual(data['max_agents'], 10)
        self.assertEqual(data['agents'], {})
        self.assertIn('inference_weights', data)
        self.assertIn('prediction_weights', data)
        self.assertIn('stats', data)

    def test_serialize_with_agents(self):
        tom = _make_tom()
        a1 = tom.add_agent(agent_type=AgentType.HUMAN)
        tom.set_relationship(a1, SocialRelation.COOPERATIVE)
        data = tom.serialize()
        self.assertIn(a1, data['agents'])
        agent_data = data['agents'][a1]
        self.assertEqual(agent_data['agent_type'], 'human')
        self.assertEqual(agent_data['relationship'], 'cooperative')

    def test_serialize_after_observations(self):
        tom = _make_tom()
        aid = tom.add_agent()
        for i in range(3):
            tom.observe(aid, _vec(DIM, seed=i), _vec(DIM, seed=i + 100))
        data = tom.serialize()
        agent_data = data['agents'][aid]
        self.assertIn('beliefs', agent_data)
        self.assertIn('desires', agent_data)

    def test_roundtrip_config(self):
        tom = _make_tom(
            embedding_dim=16,
            max_agents=5,
            belief_update_rate=0.2,
            simulation_steps=3,
            learning_rate=0.01,
        )
        data = tom.serialize()
        tom2 = TheoryOfMind.deserialize(data)
        self.assertEqual(tom2.embedding_dim, 16)
        self.assertEqual(tom2.max_agents, 5)
        self.assertAlmostEqual(tom2.belief_update_rate, 0.2)
        self.assertEqual(tom2.simulation_steps, 3)
        self.assertAlmostEqual(tom2.learning_rate, 0.01)

    def test_roundtrip_weights(self):
        tom = _make_tom()
        data = tom.serialize()
        tom2 = TheoryOfMind.deserialize(data)
        np.testing.assert_array_almost_equal(
            tom2.inference_weights, tom.inference_weights
        )
        np.testing.assert_array_almost_equal(
            tom2.inference_bias, tom.inference_bias
        )
        np.testing.assert_array_almost_equal(
            tom2.prediction_weights, tom.prediction_weights
        )
        np.testing.assert_array_almost_equal(
            tom2.prediction_bias, tom.prediction_bias
        )

    def test_roundtrip_agents(self):
        tom = _make_tom()
        aid = tom.add_agent(
            agent_type=AgentType.AI,
            relationship=SocialRelation.HIERARCHICAL,
        )
        tom.agents[aid].trust_level = 0.8
        tom.agents[aid].predictability = 0.7

        data = tom.serialize()
        tom2 = TheoryOfMind.deserialize(data)

        self.assertIn(aid, tom2.agents)
        a2 = tom2.agents[aid]
        self.assertEqual(a2.agent_type, AgentType.AI)
        self.assertEqual(a2.relationship, SocialRelation.HIERARCHICAL)
        self.assertAlmostEqual(a2.trust_level, 0.8)
        self.assertAlmostEqual(a2.predictability, 0.7)

    def test_roundtrip_agent_beliefs_and_desires(self):
        tom = _make_tom()
        aid = tom.add_agent()
        # Observe enough times to build beliefs and desires
        for i in range(5):
            tom.observe(aid, _vec(DIM, seed=i), _vec(DIM, seed=i + 100))

        data = tom.serialize()
        tom2 = TheoryOfMind.deserialize(data)

        a_orig = tom.agents[aid]
        a_new = tom2.agents[aid]

        # Compare beliefs
        for key in a_orig.beliefs:
            np.testing.assert_array_almost_equal(
                a_new.beliefs[key], a_orig.beliefs[key]
            )

        # Compare desires
        self.assertEqual(len(a_new.desires), len(a_orig.desires))
        for d_orig, d_new in zip(a_orig.desires, a_new.desires):
            np.testing.assert_array_almost_equal(d_new, d_orig)

    def test_roundtrip_personality_none(self):
        """Agent with personality_embedding=None survives roundtrip."""
        tom = _make_tom()
        aid = tom.add_agent()
        tom.agents[aid].personality_embedding = None

        data = tom.serialize()
        tom2 = TheoryOfMind.deserialize(data)
        self.assertIsNone(tom2.agents[aid].personality_embedding)

    def test_deserialize_no_agents_key(self):
        """Deserialize handles missing 'agents' key gracefully."""
        tom = _make_tom()
        data = tom.serialize()
        data.pop('agents', None)
        tom2 = TheoryOfMind.deserialize(data)
        self.assertEqual(len(tom2.agents), 0)

    def test_serialized_stats_present(self):
        tom = _make_tom()
        data = tom.serialize()
        self.assertIn('stats', data)
        self.assertIn('total_agents', data['stats'])


# ===================================================================
# 15. Integration / end-to-end
# ===================================================================
class TestIntegration(unittest.TestCase):
    """End-to-end integration tests combining multiple operations."""

    def test_full_workflow(self):
        """Add agent, observe, predict, check stats, serialize/deserialize."""
        tom = _make_tom()

        # Add agents
        a1 = tom.add_agent(agent_type=AgentType.HUMAN)
        a2 = tom.add_agent(agent_type=AgentType.AI)
        tom.set_relationship(a1, SocialRelation.COOPERATIVE)
        tom.set_relationship(a2, SocialRelation.COMPETITIVE)

        # Observe multiple times
        for i in range(5):
            tom.observe(a1, _vec(DIM, seed=i), _vec(DIM, seed=i + 50))
            tom.observe(a2, _vec(DIM, seed=i + 200), _vec(DIM, seed=i + 250))

        # Predict
        p1, c1 = tom.predict_action(a1, _vec(DIM, seed=99))
        p2, c2 = tom.predict_action(a2, _vec(DIM, seed=100))
        self.assertEqual(p1.shape, (DIM,))
        self.assertEqual(p2.shape, (DIM,))

        # Cooperative reasoning
        coop = tom.reason_cooperatively([a1, a2], _vec(DIM, seed=0), _vec(DIM, seed=1))
        self.assertEqual(len(coop['agents']), 2)

        # Competitive reasoning
        comp = tom.reason_competitively(a2, _vec(DIM, seed=0))
        self.assertIn('strategy', comp)

        # False belief
        fb = tom.infer_false_belief(a1, _vec(DIM, seed=0), agent_observation=_vec(DIM, seed=99))
        self.assertIn('has_false_belief', fb)

        # Perspective
        traj = tom.simulate_perspective(a1, _vec(DIM, seed=0))
        self.assertEqual(len(traj), tom.simulation_steps)

        # Stats
        stats = tom.get_stats()
        self.assertEqual(stats['total_agents'], 2)
        self.assertEqual(stats['interaction_count'], 10)

        # Serialize and deserialize
        data = tom.serialize()
        tom2 = TheoryOfMind.deserialize(data)
        self.assertEqual(len(tom2.agents), 2)
        self.assertIn(a1, tom2.agents)
        self.assertIn(a2, tom2.agents)

    def test_observe_then_learn_workflow(self):
        """Observe an agent, then use social learning from them."""
        tom = _make_tom()
        aid = tom.add_agent(agent_type=AgentType.HUMAN)

        # Observe first
        for i in range(3):
            tom.observe(aid, _vec(DIM, seed=i), _vec(DIM, seed=i + 50))

        # Then learn
        demo = [(_vec(DIM, seed=i + 20), _vec(DIM, seed=i + 70)) for i in range(3)]
        result = tom.learn_from_observation(aid, demo)
        self.assertEqual(result['patterns_learned'], 3)
        self.assertGreater(tom.agents[aid].trust_level, 0.5)

    def test_xp_backend_is_usable(self):
        """Verify the imported xp backend works for basic operations."""
        arr = xp.zeros(10)
        self.assertEqual(arr.shape, (10,))
        arr2 = xp.ones((3, 3))
        self.assertEqual(arr2.shape, (3, 3))


if __name__ == '__main__':
    unittest.main()
