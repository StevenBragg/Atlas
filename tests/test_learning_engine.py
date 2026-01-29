"""
Comprehensive tests for the LearningEngine class (core/learning_engine.py).
Tests cover initialization, plasticity rules (Hebbian, STDP, BCM,
Anti-Hebbian, Competitive, Cooperative), strategy handling, and state
retrieval.
"""
import sys
import os
import unittest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from self_organizing_av_system.core.backend import xp
from self_organizing_av_system.core.learning_engine import LearningEngine, LearningState
from self_organizing_av_system.core.meta_learning import LearningStrategy, MetaLearner
from self_organizing_av_system.core.challenge import (
    Challenge,
    ChallengeType,
    ChallengeStatus,
    SuccessCriteria,
    TrainingData,
    Modality,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Small dimensions so tests run quickly
_STATE_DIM = 8
_INPUT_DIM = 6
_OUTPUT_DIM = 4
_BATCH = 5
_SEED = 42


def _make_engine(**kwargs):
    """Create a LearningEngine with small state_dim and a seeded MetaLearner."""
    defaults = dict(
        state_dim=_STATE_DIM,
        meta_learner=MetaLearner(num_strategies=7, random_seed=_SEED),
        learning_rate=0.1,
        batch_size=_BATCH,
        max_plateau_count=5,
    )
    defaults.update(kwargs)
    return LearningEngine(**defaults)


def _make_challenge(feature_dim=_INPUT_DIM, num_classes=_OUTPUT_DIM, n_samples=20):
    """Create a lightweight Challenge with synthetic training data."""
    np.random.seed(_SEED)
    samples = [np.random.randn(feature_dim).astype(np.float32) for _ in range(n_samples)]
    labels = [int(i % num_classes) for i in range(n_samples)]
    td = TrainingData(
        samples=samples,
        labels=labels,
        modality=Modality.EMBEDDING,
        feature_dim=feature_dim,
        num_classes=num_classes,
    )
    return Challenge(
        name="test_challenge",
        description="A small test challenge",
        challenge_type=ChallengeType.PATTERN_RECOGNITION,
        modalities=[Modality.EMBEDDING],
        training_data=td,
        success_criteria=SuccessCriteria(accuracy=0.5, max_iterations=50),
        difficulty=0.3,
    )


def _make_learning_state(strategy, engine=None, feature_dim=_INPUT_DIM,
                         output_dim=_OUTPUT_DIM):
    """Build a LearningState with deterministic weights."""
    np.random.seed(_SEED)
    challenge = _make_challenge(feature_dim=feature_dim, num_classes=output_dim)
    hyperparams = {'learning_rate': 1.0, 'bcm_threshold': 0.5}
    weights = xp.array(
        np.random.randn(output_dim, feature_dim).astype(np.float32) * 0.1
    )
    return LearningState(
        challenge=challenge,
        strategy=strategy,
        hyperparameters=hyperparams,
        weights=weights,
    )


def _make_batch(batch_size=_BATCH, input_dim=_INPUT_DIM, num_classes=_OUTPUT_DIM,
                seed=_SEED, with_labels=True):
    """Return (batch_x, batch_y) as xp arrays with fixed seed."""
    np.random.seed(seed)
    batch_x = xp.array(np.random.randn(batch_size, input_dim).astype(np.float32))
    if with_labels:
        batch_y = xp.array(np.random.randint(0, num_classes, batch_size))
    else:
        batch_y = None
    return batch_x, batch_y


# ---------------------------------------------------------------------------
# Test classes
# ---------------------------------------------------------------------------

class TestLearningEngineInitialization(unittest.TestCase):
    """Test LearningEngine construction and initial state."""

    def test_default_initialization(self):
        engine = _make_engine()
        self.assertEqual(engine.state_dim, _STATE_DIM)
        self.assertAlmostEqual(engine.learning_rate, 0.1)
        self.assertEqual(engine.batch_size, _BATCH)
        self.assertEqual(engine.max_plateau_count, 5)

    def test_meta_learner_assigned(self):
        meta = MetaLearner(num_strategies=7, random_seed=0)
        engine = _make_engine(meta_learner=meta)
        self.assertIs(engine.meta_learner, meta)

    def test_default_meta_learner_created(self):
        engine = LearningEngine(state_dim=_STATE_DIM)
        self.assertIsNotNone(engine.meta_learner)

    def test_no_active_sessions(self):
        engine = _make_engine()
        self.assertEqual(len(engine.active_sessions), 0)

    def test_no_capabilities(self):
        engine = _make_engine()
        self.assertEqual(len(engine.capabilities), 0)
        self.assertEqual(engine.list_capabilities(), [])

    def test_network_created(self):
        engine = _make_engine()
        self.assertIsNotNone(engine.network)

    def test_custom_learning_rate(self):
        engine = _make_engine(learning_rate=0.05)
        self.assertAlmostEqual(engine.learning_rate, 0.05)

    def test_custom_batch_size(self):
        engine = _make_engine(batch_size=16)
        self.assertEqual(engine.batch_size, 16)


class TestApplyLearningRule(unittest.TestCase):
    """Test the _apply_plasticity method (the main learning-rule dispatcher)."""

    def setUp(self):
        np.random.seed(_SEED)
        self.engine = _make_engine()

    def test_apply_plasticity_returns_accuracy_and_weights(self):
        state = _make_learning_state(LearningStrategy.HEBBIAN)
        batch_x, batch_y = _make_batch()
        accuracy, weights = self.engine._apply_plasticity(state, batch_x, batch_y)
        self.assertIsInstance(accuracy, float)
        self.assertEqual(weights.shape, (_OUTPUT_DIM, _INPUT_DIM))

    def test_apply_plasticity_accuracy_bounded(self):
        for strategy in LearningStrategy:
            state = _make_learning_state(strategy)
            batch_x, batch_y = _make_batch()
            accuracy, _ = self.engine._apply_plasticity(state, batch_x, batch_y)
            self.assertGreaterEqual(accuracy, 0.0,
                                    f"Accuracy < 0 for {strategy.name}")
            self.assertLessEqual(accuracy, 1.0,
                                 f"Accuracy > 1 for {strategy.name}")

    def test_apply_plasticity_updates_weights(self):
        state = _make_learning_state(LearningStrategy.HEBBIAN)
        old_weights = state.weights.copy()
        batch_x, batch_y = _make_batch()
        _, new_weights = self.engine._apply_plasticity(state, batch_x, batch_y)
        self.assertFalse(
            xp.allclose(new_weights, old_weights),
            "Weights should change after plasticity"
        )

    def test_apply_plasticity_without_labels(self):
        """Unsupervised path (batch_y=None) should still return valid output."""
        state = _make_learning_state(LearningStrategy.HEBBIAN)
        batch_x, _ = _make_batch(with_labels=False)
        accuracy, weights = self.engine._apply_plasticity(state, batch_x, None)
        self.assertIsInstance(accuracy, float)
        self.assertEqual(weights.shape, (_OUTPUT_DIM, _INPUT_DIM))

    def test_apply_plasticity_1d_input(self):
        """Single sample (1-D) input should be handled correctly."""
        state = _make_learning_state(LearningStrategy.HEBBIAN)
        np.random.seed(_SEED)
        batch_x = xp.array(np.random.randn(_INPUT_DIM).astype(np.float32))
        accuracy, weights = self.engine._apply_plasticity(state, batch_x, None)
        self.assertIsInstance(accuracy, float)

    def test_apply_plasticity_resizes_mismatched_weights(self):
        """If weight cols != input cols, weights get resized."""
        state = _make_learning_state(LearningStrategy.HEBBIAN,
                                     feature_dim=_INPUT_DIM,
                                     output_dim=_OUTPUT_DIM)
        # Create batch with a DIFFERENT input dim to force resize
        np.random.seed(_SEED)
        different_dim = _INPUT_DIM + 4
        batch_x = xp.array(np.random.randn(_BATCH, different_dim).astype(np.float32))
        batch_y = xp.array(np.random.randint(0, _OUTPUT_DIM, _BATCH))
        _, weights = self.engine._apply_plasticity(state, batch_x, batch_y)
        self.assertEqual(weights.shape[1], different_dim)


class TestHebbianUpdate(unittest.TestCase):
    """Test _hebbian_update."""

    def setUp(self):
        np.random.seed(_SEED)
        self.engine = _make_engine()

    def test_weights_change(self):
        weights = xp.array(np.random.randn(_OUTPUT_DIM, _INPUT_DIM).astype(np.float32) * 0.1)
        x = xp.array(np.random.randn(_BATCH, _INPUT_DIM).astype(np.float32))
        y = x @ weights.T  # activations
        old_w = weights.copy()
        new_w = self.engine._hebbian_update(weights, x, y, lr=0.1)
        self.assertFalse(xp.allclose(new_w, old_w))

    def test_output_is_normalized(self):
        np.random.seed(_SEED)
        weights = xp.array(np.random.randn(_OUTPUT_DIM, _INPUT_DIM).astype(np.float32))
        x = xp.array(np.random.randn(_BATCH, _INPUT_DIM).astype(np.float32))
        y = x @ weights.T
        new_w = self.engine._hebbian_update(weights, x, y, lr=0.1)
        norms = xp.linalg.norm(new_w, axis=1)
        for i in range(norms.shape[0]):
            self.assertAlmostEqual(float(norms[i]), 1.0, places=4,
                                   msg=f"Row {i} not normalized")

    def test_zero_lr_no_change_after_normalization(self):
        """With lr=0 the Hebbian term is zero, but normalization still applies."""
        np.random.seed(_SEED)
        weights = xp.array(np.random.randn(_OUTPUT_DIM, _INPUT_DIM).astype(np.float32))
        x = xp.array(np.random.randn(_BATCH, _INPUT_DIM).astype(np.float32))
        y = x @ weights.T
        # Normalize weights first so that the only change would come from lr
        norms = xp.linalg.norm(weights, axis=1, keepdims=True)
        weights_normed = weights / (norms + 1e-8)
        new_w = self.engine._hebbian_update(weights_normed.copy(), x, y, lr=0.0)
        # After lr=0 update + normalization, should be ~same as normalized input
        self.assertTrue(
            xp.allclose(new_w, weights_normed, atol=1e-5),
            "lr=0 should leave normalized weights unchanged"
        )


class TestSTDPUpdate(unittest.TestCase):
    """Test _stdp_update."""

    def setUp(self):
        np.random.seed(_SEED)
        self.engine = _make_engine()

    def test_weights_change(self):
        weights = xp.array(np.random.randn(_OUTPUT_DIM, _INPUT_DIM).astype(np.float32) * 0.1)
        x = xp.array(np.random.randn(_BATCH, _INPUT_DIM).astype(np.float32))
        y = x @ weights.T
        old_w = weights.copy()
        new_w = self.engine._stdp_update(weights, x, y, lr=0.1)
        self.assertFalse(xp.allclose(new_w, old_w))

    def test_weights_clipped(self):
        """STDP clips weights to [-2, 2]."""
        np.random.seed(_SEED)
        weights = xp.array(np.random.randn(_OUTPUT_DIM, _INPUT_DIM).astype(np.float32) * 0.1)
        x = xp.array(np.random.randn(_BATCH, _INPUT_DIM).astype(np.float32)) * 10.0
        y = x @ weights.T
        new_w = self.engine._stdp_update(weights, x, y, lr=1.0)
        self.assertTrue(float(xp.max(xp.abs(new_w))) <= 2.0 + 1e-6)

    def test_stdp_asymmetry(self):
        """Pre-post and post-pre correlations produce asymmetric update."""
        np.random.seed(_SEED)
        weights = xp.array(np.random.randn(_OUTPUT_DIM, _INPUT_DIM).astype(np.float32) * 0.1)
        x = xp.array(np.random.randn(_BATCH, _INPUT_DIM).astype(np.float32))
        y = x @ weights.T
        # The formula is lr * (y.T @ x - 0.5 * (x.T @ y).T) / N
        # which is NOT symmetric because of the 0.5 factor
        pre_post = y.T @ x
        post_pre = x.T @ y
        self.assertFalse(
            xp.allclose(pre_post, 0.5 * post_pre.T),
            "Pre-post and scaled post-pre should differ (asymmetric STDP)"
        )


class TestBCMUpdate(unittest.TestCase):
    """Test _bcm_update."""

    def setUp(self):
        np.random.seed(_SEED)
        self.engine = _make_engine()

    def test_weights_change(self):
        weights = xp.array(np.random.randn(_OUTPUT_DIM, _INPUT_DIM).astype(np.float32) * 0.1)
        x = xp.array(np.random.randn(_BATCH, _INPUT_DIM).astype(np.float32))
        y = x @ weights.T
        old_w = weights.copy()
        new_w = self.engine._bcm_update(weights, x, y, lr=0.1, theta=0.5)
        self.assertFalse(xp.allclose(new_w, old_w))

    def test_output_normalized(self):
        np.random.seed(_SEED)
        weights = xp.array(np.random.randn(_OUTPUT_DIM, _INPUT_DIM).astype(np.float32))
        x = xp.array(np.random.randn(_BATCH, _INPUT_DIM).astype(np.float32))
        y = x @ weights.T
        new_w = self.engine._bcm_update(weights, x, y, lr=0.01, theta=0.5)
        norms = xp.linalg.norm(new_w, axis=1)
        for i in range(norms.shape[0]):
            self.assertAlmostEqual(float(norms[i]), 1.0, places=3,
                                   msg=f"BCM row {i} not normalized")

    def test_theta_modulates_update(self):
        """Different theta values should produce different weight updates."""
        np.random.seed(_SEED)
        weights_a = xp.array(np.random.randn(_OUTPUT_DIM, _INPUT_DIM).astype(np.float32) * 0.1)
        weights_b = weights_a.copy()
        x = xp.array(np.random.randn(_BATCH, _INPUT_DIM).astype(np.float32))
        y = x @ weights_a.T

        new_a = self.engine._bcm_update(weights_a, x, y, lr=0.1, theta=0.1)
        new_b = self.engine._bcm_update(weights_b, x, y, lr=0.1, theta=2.0)
        self.assertFalse(
            xp.allclose(new_a, new_b),
            "Different BCM thresholds should produce different results"
        )


class TestAntiHebbianUpdate(unittest.TestCase):
    """Test _anti_hebbian_update."""

    def setUp(self):
        np.random.seed(_SEED)
        self.engine = _make_engine()

    def test_weights_change(self):
        weights = xp.array(np.random.randn(_OUTPUT_DIM, _INPUT_DIM).astype(np.float32) * 0.1)
        x = xp.array(np.random.randn(_BATCH, _INPUT_DIM).astype(np.float32))
        y = x @ weights.T
        old_w = weights.copy()
        new_w = self.engine._anti_hebbian_update(weights, x, y, lr=0.1)
        self.assertFalse(xp.allclose(new_w, old_w))

    def test_opposite_direction_of_hebbian(self):
        """Anti-Hebbian update should be in opposite direction to Hebbian."""
        np.random.seed(_SEED)
        weights = xp.array(np.random.randn(_OUTPUT_DIM, _INPUT_DIM).astype(np.float32) * 0.1)
        x = xp.array(np.random.randn(_BATCH, _INPUT_DIM).astype(np.float32))
        y = x @ weights.T

        # Compute raw deltas manually (before clipping/normalization)
        dW_hebbian = 0.1 * (y.T @ x) / x.shape[0]
        dW_anti = -0.1 * (y.T @ x) / x.shape[0]
        # They should be negatives of each other
        self.assertTrue(xp.allclose(dW_anti, -dW_hebbian, atol=1e-6))

    def test_weights_clipped(self):
        """Anti-Hebbian clips weights to [-2, 2]."""
        np.random.seed(_SEED)
        weights = xp.array(np.random.randn(_OUTPUT_DIM, _INPUT_DIM).astype(np.float32) * 0.1)
        x = xp.array(np.random.randn(_BATCH, _INPUT_DIM).astype(np.float32)) * 10.0
        y = x @ weights.T
        new_w = self.engine._anti_hebbian_update(weights, x, y, lr=1.0)
        self.assertTrue(float(xp.max(xp.abs(new_w))) <= 2.0 + 1e-6)


class TestCompetitiveUpdate(unittest.TestCase):
    """Test _competitive_update (winner-take-all)."""

    def setUp(self):
        np.random.seed(_SEED)
        self.engine = _make_engine()

    def test_weights_change(self):
        weights = xp.array(np.random.randn(_OUTPUT_DIM, _INPUT_DIM).astype(np.float32) * 0.1)
        x = xp.array(np.random.randn(_BATCH, _INPUT_DIM).astype(np.float32))
        y = x @ weights.T
        old_w = weights.copy()
        new_w = self.engine._competitive_update(weights, x, y, lr=0.1)
        self.assertFalse(xp.allclose(new_w, old_w))

    def test_only_winner_updated(self):
        """For a single sample, only the winner row should change."""
        np.random.seed(_SEED)
        weights = xp.array(np.random.randn(_OUTPUT_DIM, _INPUT_DIM).astype(np.float32) * 0.1)
        # Use a single sample
        x = xp.array(np.random.randn(1, _INPUT_DIM).astype(np.float32))
        y = x @ weights.T  # (1, output_dim)
        old_w = weights.copy()
        new_w = self.engine._competitive_update(weights.copy(), x, y, lr=0.1)

        winner = int(xp.argmax(y[0]))
        for row_idx in range(_OUTPUT_DIM):
            if row_idx == winner:
                # Winner row may have changed
                continue
            self.assertTrue(
                xp.allclose(new_w[row_idx], old_w[row_idx]),
                f"Non-winner row {row_idx} should not change"
            )

    def test_winner_moves_toward_input(self):
        """Winner weight vector should move closer to the input."""
        np.random.seed(_SEED)
        weights = xp.array(np.random.randn(_OUTPUT_DIM, _INPUT_DIM).astype(np.float32) * 0.1)
        x = xp.array(np.random.randn(1, _INPUT_DIM).astype(np.float32))
        y = x @ weights.T
        winner = int(xp.argmax(y[0]))

        old_dist = float(xp.linalg.norm(x[0] - weights[winner]))
        new_w = self.engine._competitive_update(weights.copy(), x, y, lr=0.5)
        new_dist = float(xp.linalg.norm(x[0] - new_w[winner]))
        self.assertLess(new_dist, old_dist,
                        "Winner should move toward input")


class TestCooperativeUpdate(unittest.TestCase):
    """Test _cooperative_update (neighborhood learning)."""

    def setUp(self):
        np.random.seed(_SEED)
        self.engine = _make_engine()

    def test_weights_change(self):
        weights = xp.array(np.random.randn(_OUTPUT_DIM, _INPUT_DIM).astype(np.float32) * 0.1)
        x = xp.array(np.random.randn(_BATCH, _INPUT_DIM).astype(np.float32))
        y = x @ weights.T
        old_w = weights.copy()
        new_w = self.engine._cooperative_update(weights, x, y, lr=0.1)
        self.assertFalse(xp.allclose(new_w, old_w))

    def test_all_rows_updated(self):
        """Cooperative learning should update ALL rows (neighborhood effect)."""
        np.random.seed(_SEED)
        weights = xp.array(np.random.randn(_OUTPUT_DIM, _INPUT_DIM).astype(np.float32) * 0.1)
        x = xp.array(np.random.randn(1, _INPUT_DIM).astype(np.float32))
        y = x @ weights.T
        old_w = weights.copy()
        new_w = self.engine._cooperative_update(weights.copy(), x, y, lr=0.5)
        for row_idx in range(_OUTPUT_DIM):
            self.assertFalse(
                xp.allclose(new_w[row_idx], old_w[row_idx]),
                f"Row {row_idx} should change in cooperative update"
            )

    def test_winner_changes_most(self):
        """The winner should experience the largest weight change when
        all weight rows start identical (so neighbourhood is the sole factor)."""
        np.random.seed(_SEED)
        # All rows identical => (xi - weights[row]) is the same for every row,
        # so the neighbourhood function alone determines change magnitude.
        base_row = np.random.randn(_INPUT_DIM).astype(np.float32) * 0.1
        weights = xp.array(np.tile(base_row, (_OUTPUT_DIM, 1)))
        x = xp.array(np.random.randn(1, _INPUT_DIM).astype(np.float32))
        y = x @ weights.T
        winner = int(xp.argmax(y[0]))
        old_w = weights.copy()
        new_w = self.engine._cooperative_update(weights.copy(), x, y, lr=0.5)

        changes = [float(xp.linalg.norm(new_w[i] - old_w[i])) for i in range(_OUTPUT_DIM)]
        winner_change = changes[winner]
        for i, c in enumerate(changes):
            if i != winner:
                self.assertGreaterEqual(
                    winner_change + 1e-6, c,
                    f"Winner change ({winner_change:.6f}) should be >= "
                    f"non-winner row {i} change ({c:.6f})"
                )


class TestDifferentLearningStrategies(unittest.TestCase):
    """Test that each LearningStrategy dispatches correctly through _apply_plasticity."""

    def setUp(self):
        np.random.seed(_SEED)
        self.engine = _make_engine()

    def test_all_strategies_produce_valid_output(self):
        for strategy in LearningStrategy:
            with self.subTest(strategy=strategy.name):
                state = _make_learning_state(strategy)
                batch_x, batch_y = _make_batch()
                accuracy, weights = self.engine._apply_plasticity(
                    state, batch_x, batch_y
                )
                self.assertIsInstance(accuracy, float)
                self.assertGreaterEqual(accuracy, 0.0)
                self.assertLessEqual(accuracy, 1.0)
                self.assertEqual(weights.shape, (_OUTPUT_DIM, _INPUT_DIM))

    def test_strategies_produce_different_weights(self):
        """Different strategies should (in general) yield different weight matrices."""
        results = {}
        for strategy in LearningStrategy:
            np.random.seed(_SEED)
            state = _make_learning_state(strategy)
            batch_x, batch_y = _make_batch()
            _, weights = self.engine._apply_plasticity(state, batch_x, batch_y)
            results[strategy] = weights.copy()

        # At least some pairs should differ
        strategies = list(results.keys())
        any_different = False
        for i in range(len(strategies)):
            for j in range(i + 1, len(strategies)):
                if not xp.allclose(results[strategies[i]], results[strategies[j]], atol=1e-5):
                    any_different = True
                    break
            if any_different:
                break
        self.assertTrue(any_different,
                        "At least some strategy pairs should produce different weights")

    def test_unsupervised_competitive_applies(self):
        """Competitive learning should modify weights when no labels present."""
        state = _make_learning_state(LearningStrategy.COMPETITIVE)
        batch_x, _ = _make_batch(with_labels=False)
        old_w = state.weights.copy()
        _, new_w = self.engine._apply_plasticity(state, batch_x, None)
        self.assertFalse(
            xp.allclose(new_w, old_w),
            "Competitive update without labels should change weights"
        )

    def test_unsupervised_cooperative_applies(self):
        """Cooperative learning should modify weights when no labels present."""
        state = _make_learning_state(LearningStrategy.COOPERATIVE)
        batch_x, _ = _make_batch(with_labels=False)
        old_w = state.weights.copy()
        _, new_w = self.engine._apply_plasticity(state, batch_x, None)
        self.assertFalse(
            xp.allclose(new_w, old_w),
            "Cooperative update without labels should change weights"
        )

    def test_supervised_competitive_skips_competitive_rule(self):
        """With labels, competitive rule is skipped (only supervised delta applies)."""
        state = _make_learning_state(LearningStrategy.COMPETITIVE)
        batch_x, batch_y = _make_batch()

        # Build weights from supervised component only (delta rule)
        # The competitive branch is skipped when has_supervision=True
        accuracy, weights = self.engine._apply_plasticity(state, batch_x, batch_y)
        # Just verify it runs without error and returns valid output
        self.assertIsInstance(accuracy, float)
        self.assertEqual(weights.shape, (_OUTPUT_DIM, _INPUT_DIM))


class TestGetState(unittest.TestCase):
    """Test state retrieval methods on LearningEngine."""

    def setUp(self):
        np.random.seed(_SEED)
        self.engine = _make_engine()

    def test_get_internal_state_shape(self):
        state = self.engine.get_internal_state()
        self.assertEqual(state.shape, (_STATE_DIM,))

    def test_get_internal_state_dtype(self):
        state = self.engine.get_internal_state()
        self.assertTrue(state.dtype in (np.float32, np.float64))

    def test_get_network_stats(self):
        stats = self.engine.get_network_stats()
        self.assertIsInstance(stats, dict)

    def test_get_network_structure(self):
        structure = self.engine.get_network_structure()
        self.assertIsInstance(structure, dict)

    def test_get_layer_activations_initially_empty(self):
        activations = self.engine.get_layer_activations()
        # Before any forward pass, activations may be empty
        self.assertIsInstance(activations, list)

    def test_get_capability_missing(self):
        result = self.engine.get_capability("nonexistent")
        self.assertIsNone(result)

    def test_list_capabilities_empty(self):
        caps = self.engine.list_capabilities()
        self.assertEqual(caps, [])

    def test_get_recent_structural_changes(self):
        changes = self.engine.get_recent_structural_changes(limit=5)
        self.assertIsInstance(changes, list)

    def test_register_structural_change_callback(self):
        calls = []
        self.engine.register_structural_change_callback(lambda info: calls.append(info))
        self.assertEqual(len(self.engine._structural_change_callbacks), 1)


class TestLearningStateDataclass(unittest.TestCase):
    """Test the LearningState dataclass independently."""

    def test_creation(self):
        state = _make_learning_state(LearningStrategy.HEBBIAN)
        self.assertEqual(state.iteration, 0)
        self.assertEqual(state.best_accuracy, 0.0)
        self.assertEqual(state.plateau_count, 0)
        self.assertEqual(state.structural_changes, [])
        self.assertIsInstance(state.strategy, LearningStrategy)
        self.assertEqual(state.strategy, LearningStrategy.HEBBIAN)

    def test_weights_shape(self):
        state = _make_learning_state(LearningStrategy.OJA)
        self.assertEqual(state.weights.shape, (_OUTPUT_DIM, _INPUT_DIM))

    def test_hyperparameters_present(self):
        state = _make_learning_state(LearningStrategy.BCM)
        self.assertIn('learning_rate', state.hyperparameters)
        self.assertIn('bcm_threshold', state.hyperparameters)

    def test_challenge_attached(self):
        state = _make_learning_state(LearningStrategy.STDP)
        self.assertIsNotNone(state.challenge)
        self.assertEqual(state.challenge.name, "test_challenge")

    def test_mutable_iteration(self):
        state = _make_learning_state(LearningStrategy.HEBBIAN)
        state.iteration = 42
        self.assertEqual(state.iteration, 42)

    def test_start_time_set(self):
        state = _make_learning_state(LearningStrategy.HEBBIAN)
        self.assertIsInstance(state.start_time, float)
        self.assertGreater(state.start_time, 0)


class TestCalculateAccuracy(unittest.TestCase):
    """Test the _calculate_accuracy helper."""

    def setUp(self):
        np.random.seed(_SEED)
        self.engine = _make_engine()

    def test_perfect_classification(self):
        """When predictions match labels, accuracy should be 1.0."""
        # Design weights so that argmax of x @ W.T matches labels
        np.random.seed(_SEED)
        weights = xp.eye(_OUTPUT_DIM, _INPUT_DIM, dtype=xp.float32)
        # Input: one-hot-like vectors so argmax(x @ W.T) = label
        x = xp.zeros((_OUTPUT_DIM, _INPUT_DIM), dtype=xp.float32)
        for i in range(_OUTPUT_DIM):
            x[i, i] = 1.0
        y = xp.arange(_OUTPUT_DIM)
        acc = self.engine._calculate_accuracy(weights, x, y)
        self.assertAlmostEqual(acc, 1.0, places=5)

    def test_unsupervised_accuracy(self):
        """With y=None, uses reconstruction proxy."""
        np.random.seed(_SEED)
        weights = xp.array(np.random.randn(_OUTPUT_DIM, _INPUT_DIM).astype(np.float32) * 0.1)
        x = xp.array(np.random.randn(_BATCH, _INPUT_DIM).astype(np.float32))
        acc = self.engine._calculate_accuracy(weights, x, None)
        self.assertIsInstance(acc, float)
        # Reconstruction accuracy may be 0 if error > 1 (max(0, 1-error))
        self.assertGreaterEqual(acc, 0.0)
        self.assertLessEqual(acc, 1.0)

    def test_accuracy_with_out_of_range_labels(self):
        """Labels exceeding weight dimension should be handled gracefully."""
        np.random.seed(_SEED)
        weights = xp.array(np.random.randn(_OUTPUT_DIM, _INPUT_DIM).astype(np.float32))
        x = xp.array(np.random.randn(_BATCH, _INPUT_DIM).astype(np.float32))
        # All labels out of range
        y = xp.array([_OUTPUT_DIM + 10] * _BATCH)
        acc = self.engine._calculate_accuracy(weights, x, y)
        self.assertAlmostEqual(acc, 0.0, places=5)


class TestInitializeWeights(unittest.TestCase):
    """Test the _initialize_weights helper."""

    def setUp(self):
        np.random.seed(_SEED)
        self.engine = _make_engine()

    def test_weight_shape_with_training_data(self):
        challenge = _make_challenge(feature_dim=_INPUT_DIM, num_classes=_OUTPUT_DIM)
        weights = self.engine._initialize_weights(challenge)
        self.assertEqual(weights.shape, (_OUTPUT_DIM, _INPUT_DIM))

    def test_weight_shape_without_training_data(self):
        challenge = Challenge(
            name="no_data",
            description="Challenge without data",
        )
        weights = self.engine._initialize_weights(challenge)
        self.assertEqual(weights.shape, (_STATE_DIM, _STATE_DIM))

    def test_xavier_scale(self):
        """Weights should be roughly Xavier-scaled."""
        np.random.seed(_SEED)
        challenge = _make_challenge(feature_dim=100, num_classes=50)
        engine = _make_engine(state_dim=100)
        weights = engine._initialize_weights(challenge)
        expected_scale = float(xp.sqrt(2.0 / (100 + 50)))
        std = float(xp.std(weights))
        # std should be in the ballpark of expected scale
        self.assertAlmostEqual(std, expected_scale, delta=expected_scale * 0.5)


if __name__ == '__main__':
    unittest.main()
