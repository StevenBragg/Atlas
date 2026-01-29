"""
Comprehensive tests for the MultimodalAssociation class
(core/multimodal_association.py).

Tests cover initialization, binding (update), cross-modal prediction,
association modes (HEBBIAN, STDP, COMPETITIVE), learning, serialization,
and state management.
"""
import sys
import os
import unittest
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from self_organizing_av_system.core.backend import xp
from self_organizing_av_system.core.multimodal_association import (
    MultimodalAssociation,
    AssociationMode,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
VISUAL_SIZE = 16
AUDIO_SIZE = 8
SEED = 42


def _make_association(**overrides):
    """Create a MultimodalAssociation with small default sizes."""
    defaults = dict(
        modality_sizes={"visual": VISUAL_SIZE, "audio": AUDIO_SIZE},
        association_size=12,
        learning_rate=0.01,
        random_seed=SEED,
    )
    defaults.update(overrides)
    return MultimodalAssociation(**defaults)


def _random_visual(seed=0):
    rng = xp.random.RandomState(seed)
    v = rng.rand(VISUAL_SIZE)
    return v / (xp.linalg.norm(v) + 1e-8)


def _random_audio(seed=1):
    rng = xp.random.RandomState(seed)
    a = rng.rand(AUDIO_SIZE)
    return a / (xp.linalg.norm(a) + 1e-8)


# ===================================================================
# 1. Initialization Tests
# ===================================================================
class TestInitialization(unittest.TestCase):
    """Test MultimodalAssociation construction and initial state."""

    def test_default_modality_sizes(self):
        ma = _make_association()
        self.assertEqual(ma.modality_sizes, {"visual": VISUAL_SIZE, "audio": AUDIO_SIZE})

    def test_explicit_association_size(self):
        ma = _make_association(association_size=20)
        self.assertEqual(ma.association_size, 20)

    def test_auto_association_size(self):
        """When association_size is None it should default to max modality size."""
        ma = MultimodalAssociation(
            modality_sizes={"visual": VISUAL_SIZE, "audio": AUDIO_SIZE},
            association_size=None,
            random_seed=SEED,
        )
        self.assertEqual(ma.association_size, max(VISUAL_SIZE, AUDIO_SIZE))

    def test_learning_rate_stored(self):
        ma = _make_association(learning_rate=0.05)
        self.assertAlmostEqual(ma.learning_rate, 0.05)

    def test_association_mode_enum(self):
        ma = _make_association(association_mode=AssociationMode.STDP)
        self.assertEqual(ma.association_mode, AssociationMode.STDP)

    def test_association_mode_string(self):
        ma = _make_association(association_mode="competitive")
        self.assertEqual(ma.association_mode, AssociationMode.COMPETITIVE)

    def test_forward_weights_shapes(self):
        ma = _make_association()
        self.assertEqual(ma.forward_weights["visual"].shape, (12, VISUAL_SIZE))
        self.assertEqual(ma.forward_weights["audio"].shape, (12, AUDIO_SIZE))

    def test_backward_weights_shapes(self):
        ma = _make_association()
        self.assertEqual(ma.backward_weights["visual"].shape, (VISUAL_SIZE, 12))
        self.assertEqual(ma.backward_weights["audio"].shape, (AUDIO_SIZE, 12))

    def test_association_activity_initialized_to_zero(self):
        ma = _make_association()
        self.assertTrue(xp.allclose(ma.association_activity, xp.zeros(12)))

    def test_default_modality_weights(self):
        ma = _make_association()
        for w in ma.modality_weights.values():
            self.assertAlmostEqual(w, 0.5)

    def test_custom_modality_weights(self):
        ma = _make_association(modality_weights={"visual": 0.7, "audio": 0.3})
        self.assertAlmostEqual(ma.modality_weights["visual"], 0.7)
        self.assertAlmostEqual(ma.modality_weights["audio"], 0.3)

    def test_initial_update_count_zero(self):
        ma = _make_association()
        self.assertEqual(ma.update_count, 0)

    def test_attention_weights_initialized(self):
        ma = _make_association()
        for modality in ("visual", "audio"):
            self.assertIn(modality, ma.attention_weights)
            self.assertAlmostEqual(ma.attention_weights[modality], 1.0)

    def test_single_modality(self):
        ma = MultimodalAssociation(
            modality_sizes={"touch": 4},
            association_size=6,
            random_seed=SEED,
        )
        self.assertIn("touch", ma.forward_weights)
        self.assertEqual(ma.forward_weights["touch"].shape, (6, 4))

    def test_three_modalities(self):
        ma = MultimodalAssociation(
            modality_sizes={"visual": 16, "audio": 8, "touch": 4},
            association_size=10,
            random_seed=SEED,
        )
        self.assertEqual(len(ma.forward_weights), 3)
        self.assertEqual(len(ma.backward_weights), 3)

    def test_normalization_mode_stored(self):
        for mode in ("softmax", "max", "none"):
            ma = _make_association(normalization_mode=mode)
            self.assertEqual(ma.normalization_mode, mode)

    def test_sparse_coding_flag(self):
        ma = _make_association(use_sparse_coding=False)
        self.assertFalse(ma.use_sparse_coding)

    def test_enable_attention_flag(self):
        ma = _make_association(enable_attention=False)
        self.assertFalse(ma.enable_attention)


# ===================================================================
# 2. Bind (update) Tests
# ===================================================================
class TestBind(unittest.TestCase):
    """Test the update() method which binds visual and audio inputs."""

    def setUp(self):
        self.ma = _make_association()
        self.visual = _random_visual()
        self.audio = _random_audio()

    def test_update_returns_dict(self):
        result = self.ma.update({"visual": self.visual, "audio": self.audio})
        self.assertIsInstance(result, dict)

    def test_update_contains_association_activity(self):
        result = self.ma.update({"visual": self.visual, "audio": self.audio})
        self.assertIn("association_activity", result)
        self.assertEqual(result["association_activity"].shape, (12,))

    def test_update_increments_count(self):
        self.ma.update({"visual": self.visual, "audio": self.audio})
        self.assertEqual(self.ma.update_count, 1)
        self.ma.update({"visual": self.visual, "audio": self.audio})
        self.assertEqual(self.ma.update_count, 2)

    def test_reconstructions_returned(self):
        result = self.ma.update({"visual": self.visual, "audio": self.audio})
        self.assertIn("visual", result["reconstructions"])
        self.assertIn("audio", result["reconstructions"])
        self.assertEqual(result["reconstructions"]["visual"].shape, (VISUAL_SIZE,))
        self.assertEqual(result["reconstructions"]["audio"].shape, (AUDIO_SIZE,))

    def test_reconstruction_errors_returned(self):
        result = self.ma.update({"visual": self.visual, "audio": self.audio})
        self.assertIn("visual", result["reconstruction_errors"])
        self.assertIn("audio", result["reconstruction_errors"])
        for err in result["reconstruction_errors"].values():
            self.assertIsInstance(err, float)
            self.assertGreaterEqual(err, 0.0)

    def test_weight_changes_returned(self):
        result = self.ma.update({"visual": self.visual, "audio": self.audio})
        self.assertIn("weight_changes", result)
        for key in ("visual", "audio"):
            self.assertIn(key, result["weight_changes"])

    def test_update_with_single_modality(self):
        result = self.ma.update({"visual": self.visual})
        self.assertIn("visual", result["reconstructions"])
        self.assertNotIn("audio", result["reconstructions"])

    def test_learning_disabled(self):
        fw_before = self.ma.forward_weights["visual"].copy()
        self.ma.update(
            {"visual": self.visual, "audio": self.audio},
            learning_enabled=False,
        )
        fw_after = self.ma.forward_weights["visual"]
        self.assertTrue(xp.allclose(fw_before, fw_after))

    def test_attention_focus(self):
        result = self.ma.update(
            {"visual": self.visual, "audio": self.audio},
            attention_focus="visual",
        )
        # After focusing on visual, visual attention weight should be boosted
        self.assertGreater(
            self.ma.attention_weights["visual"],
            self.ma.attention_weights["audio"],
        )

    def test_association_activity_not_all_zero_after_update(self):
        self.ma.update({"visual": self.visual, "audio": self.audio})
        self.assertFalse(xp.allclose(self.ma.association_activity, 0.0))

    def test_softmax_normalization_sums_to_one(self):
        ma = _make_association(normalization_mode="softmax")
        ma.update({"visual": self.visual, "audio": self.audio})
        total = float(xp.sum(ma.association_activity))
        self.assertAlmostEqual(total, 1.0, places=5)

    def test_max_normalization_bounded(self):
        ma = _make_association(normalization_mode="max")
        ma.update({"visual": self.visual, "audio": self.audio})
        max_val = float(xp.max(xp.abs(ma.association_activity)))
        self.assertLessEqual(max_val, 1.0 + 1e-6)


# ===================================================================
# 3. Cross-modal Prediction Tests
# ===================================================================
class TestPredictVisualFromAudio(unittest.TestCase):
    """Test predicting visual output given audio input."""

    def setUp(self):
        self.ma = _make_association()
        self.visual = _random_visual()
        self.audio = _random_audio()

    def test_prediction_shape(self):
        pred = self.ma.get_cross_modal_prediction("audio", self.audio, "visual")
        self.assertEqual(pred.shape, (VISUAL_SIZE,))

    def test_prediction_is_finite(self):
        pred = self.ma.get_cross_modal_prediction("audio", self.audio, "visual")
        self.assertTrue(xp.all(xp.isfinite(pred)))

    def test_prediction_normalized(self):
        pred = self.ma.get_cross_modal_prediction("audio", self.audio, "visual")
        max_val = float(xp.max(xp.abs(pred)))
        # Should be normalised to 1 (or 0 if all-zero)
        self.assertTrue(max_val <= 1.0 + 1e-6)

    def test_different_inputs_different_predictions(self):
        pred1 = self.ma.get_cross_modal_prediction("audio", _random_audio(10), "visual")
        pred2 = self.ma.get_cross_modal_prediction("audio", _random_audio(20), "visual")
        self.assertFalse(xp.allclose(pred1, pred2))

    def test_unknown_source_returns_zeros(self):
        pred = self.ma.get_cross_modal_prediction("unknown", self.audio, "visual")
        self.assertTrue(xp.allclose(pred, 0.0))

    def test_unknown_target_returns_zeros(self):
        pred = self.ma.get_cross_modal_prediction("audio", self.audio, "unknown")
        self.assertTrue(xp.allclose(pred, 0.0))


class TestPredictAudioFromVisual(unittest.TestCase):
    """Test predicting audio output given visual input."""

    def setUp(self):
        self.ma = _make_association()
        self.visual = _random_visual()

    def test_prediction_shape(self):
        pred = self.ma.get_cross_modal_prediction("visual", self.visual, "audio")
        self.assertEqual(pred.shape, (AUDIO_SIZE,))

    def test_prediction_is_finite(self):
        pred = self.ma.get_cross_modal_prediction("visual", self.visual, "audio")
        self.assertTrue(xp.all(xp.isfinite(pred)))

    def test_prediction_normalized(self):
        pred = self.ma.get_cross_modal_prediction("visual", self.visual, "audio")
        max_val = float(xp.max(xp.abs(pred)))
        self.assertTrue(max_val <= 1.0 + 1e-6)

    def test_different_inputs_different_predictions(self):
        pred1 = self.ma.get_cross_modal_prediction("visual", _random_visual(30), "audio")
        pred2 = self.ma.get_cross_modal_prediction("visual", _random_visual(40), "audio")
        self.assertFalse(xp.allclose(pred1, pred2))


# ===================================================================
# 4. Association Mode Tests
# ===================================================================
class TestHebbianMode(unittest.TestCase):
    """Test Hebbian association mode."""

    def test_weights_change_after_update(self):
        ma = _make_association(
            association_mode=AssociationMode.HEBBIAN,
            association_threshold=0.0,
        )
        fw_before = ma.forward_weights["visual"].copy()
        v = _random_visual()
        a = _random_audio()
        ma.update({"visual": v, "audio": a})
        fw_after = ma.forward_weights["visual"]
        # Weights should have changed (at least some elements)
        self.assertFalse(xp.allclose(fw_before, fw_after))

    def test_repeated_updates_converge(self):
        """Repeated identical patterns should reduce reconstruction error."""
        ma = _make_association(
            association_mode=AssociationMode.HEBBIAN,
            learning_rate=0.05,
            association_threshold=0.0,
        )
        v = _random_visual()
        a = _random_audio()
        errors = []
        for _ in range(30):
            result = ma.update({"visual": v, "audio": a})
            errors.append(result["reconstruction_errors"].get("visual", 1.0))
        # Error at the end should be no greater than at the start
        self.assertLessEqual(errors[-1], errors[0] + 0.1)


class TestSTDPMode(unittest.TestCase):
    """Test STDP association mode."""

    def test_stdp_initializes(self):
        ma = _make_association(association_mode=AssociationMode.STDP)
        self.assertEqual(ma.association_mode, AssociationMode.STDP)

    def test_first_update_uses_hebbian_fallback(self):
        """On the first call there is no prior reconstruction; Hebbian is used."""
        ma = _make_association(
            association_mode=AssociationMode.STDP,
            association_threshold=0.0,
        )
        fw_before = ma.forward_weights["visual"].copy()
        v = _random_visual()
        a = _random_audio()
        ma.update({"visual": v, "audio": a})
        fw_after = ma.forward_weights["visual"]
        self.assertFalse(xp.allclose(fw_before, fw_after))

    def test_subsequent_update_uses_prediction_error(self):
        """The second update should use the cached reconstruction for STDP."""
        ma = _make_association(
            association_mode=AssociationMode.STDP,
            association_threshold=0.0,
        )
        v = _random_visual()
        a = _random_audio()
        ma.update({"visual": v, "audio": a})
        # After first update, last_reconstruction should be populated
        self.assertIn("visual", ma.last_reconstruction)
        # Second update now runs the STDP branch
        fw_before = ma.forward_weights["visual"].copy()
        ma.update({"visual": v, "audio": a})
        fw_after = ma.forward_weights["visual"]
        self.assertFalse(xp.allclose(fw_before, fw_after))


class TestCompetitiveMode(unittest.TestCase):
    """Test Competitive association mode."""

    def test_competitive_initializes(self):
        ma = _make_association(association_mode=AssociationMode.COMPETITIVE)
        self.assertEqual(ma.association_mode, AssociationMode.COMPETITIVE)

    def test_competitive_update_changes_weights(self):
        ma = _make_association(
            association_mode=AssociationMode.COMPETITIVE,
            association_threshold=0.0,
        )
        fw_before = ma.forward_weights["visual"].copy()
        v = _random_visual()
        a = _random_audio()
        ma.update({"visual": v, "audio": a})
        fw_after = ma.forward_weights["visual"]
        self.assertFalse(xp.allclose(fw_before, fw_after))

    def test_winner_take_all_selectivity(self):
        """Only winning neurons should have their weights updated."""
        ma = _make_association(
            association_mode=AssociationMode.COMPETITIVE,
            association_threshold=0.0,
        )
        v = _random_visual()
        a = _random_audio()
        ma.update({"visual": v, "audio": a}, learning_enabled=False)
        # Identify winning neurons
        winners = ma.association_activity > xp.mean(ma.association_activity)
        losers = ~winners
        # Now do a learning update
        fw_before = ma.forward_weights["audio"].copy()
        ma.update({"visual": v, "audio": a}, learning_enabled=True)
        fw_after = ma.forward_weights["audio"]
        delta = xp.abs(fw_after - fw_before)
        # Losers should have smaller changes than winners (on average)
        if int(xp.sum(winners)) > 0 and int(xp.sum(losers)) > 0:
            winner_change = float(xp.mean(delta[winners.flatten()]))
            loser_change = float(xp.mean(delta[losers.flatten()]))
            self.assertGreaterEqual(winner_change + 1e-9, loser_change)


class TestAdaptiveMode(unittest.TestCase):
    """Test Adaptive association mode."""

    def test_adaptive_initializes(self):
        ma = _make_association(association_mode=AssociationMode.ADAPTIVE)
        self.assertEqual(ma.association_mode, AssociationMode.ADAPTIVE)

    def test_adaptive_update_runs(self):
        ma = _make_association(association_mode=AssociationMode.ADAPTIVE)
        v = _random_visual()
        a = _random_audio()
        result = ma.update({"visual": v, "audio": a})
        self.assertIn("association_activity", result)


# ===================================================================
# 5. Learn (weight update) Tests
# ===================================================================
class TestLearn(unittest.TestCase):
    """Test the learning behaviour of update()."""

    def setUp(self):
        self.ma = _make_association(
            learning_rate=0.05,
            association_threshold=0.0,
        )
        self.visual = _random_visual()
        self.audio = _random_audio()

    def test_forward_weights_updated(self):
        fw_before = self.ma.forward_weights["visual"].copy()
        self.ma.update({"visual": self.visual, "audio": self.audio})
        fw_after = self.ma.forward_weights["visual"]
        self.assertFalse(xp.allclose(fw_before, fw_after))

    def test_backward_weights_updated(self):
        bw_before = self.ma.backward_weights["visual"].copy()
        self.ma.update({"visual": self.visual, "audio": self.audio})
        bw_after = self.ma.backward_weights["visual"]
        self.assertFalse(xp.allclose(bw_before, bw_after))

    def test_weight_deltas_tracked(self):
        self.ma.update({"visual": self.visual, "audio": self.audio})
        for modality in ("visual", "audio"):
            self.assertIn(modality, self.ma.weight_deltas)

    def test_stability_tracking(self):
        """After many identical updates, modalities may become stable."""
        ma = _make_association(
            learning_rate=0.001,
            stability_threshold=10.0,
            association_threshold=0.0,
        )
        v = _random_visual()
        a = _random_audio()
        for _ in range(5):
            result = ma.update({"visual": v, "audio": a})
        # With a very high stability threshold, associations should be stable
        for modality in ("visual", "audio"):
            if modality in result.get("stability", {}):
                self.assertGreaterEqual(result["stability"][modality], 0.0)

    def test_regularization_shrinks_weights(self):
        """With high regularization, weight magnitudes should decrease."""
        ma = _make_association(
            learning_rate=0.0,
            regularization_strength=0.1,
            decay_rate=0.0,
            association_threshold=0.0,
        )
        fw_norm_before = float(xp.linalg.norm(ma.forward_weights["visual"]))
        v = _random_visual()
        a = _random_audio()
        ma.update({"visual": v, "audio": a})
        fw_norm_after = float(xp.linalg.norm(ma.forward_weights["visual"]))
        self.assertLess(fw_norm_after, fw_norm_before)

    def test_decay_reduces_weights(self):
        """With high decay and no learning, weights shrink."""
        ma = _make_association(
            learning_rate=0.0,
            regularization_strength=0.0,
            decay_rate=0.5,
            association_threshold=0.0,
        )
        fw_norm_before = float(xp.linalg.norm(ma.forward_weights["visual"]))
        v = _random_visual()
        a = _random_audio()
        ma.update({"visual": v, "audio": a})
        fw_norm_after = float(xp.linalg.norm(ma.forward_weights["visual"]))
        self.assertLess(fw_norm_after, fw_norm_before)

    def test_reconstruction_error_history_grows(self):
        v = _random_visual()
        a = _random_audio()
        for _ in range(5):
            self.ma.update({"visual": v, "audio": a})
        self.assertEqual(len(self.ma.reconstruction_errors["visual"]), 5)
        self.assertEqual(len(self.ma.reconstruction_errors["audio"]), 5)


# ===================================================================
# 6. Serialization / get_state Tests
# ===================================================================
class TestSerialization(unittest.TestCase):
    """Test serialize() and deserialize() roundtrip."""

    def setUp(self):
        self.ma = _make_association()
        v = _random_visual()
        a = _random_audio()
        # Perform a few updates so there is non-trivial state
        for _ in range(3):
            self.ma.update({"visual": v, "audio": a})

    def test_serialize_returns_dict(self):
        data = self.ma.serialize()
        self.assertIsInstance(data, dict)

    def test_serialize_keys(self):
        data = self.ma.serialize()
        expected_keys = {
            "modality_sizes", "association_size", "learning_rate",
            "association_threshold", "association_mode", "modality_weights",
            "normalization_mode", "lateral_inhibition", "use_sparse_coding",
            "stability_threshold", "forward_weights", "backward_weights",
            "attention_weights", "stable_associations",
            "mean_reconstruction_error", "update_count",
        }
        self.assertTrue(expected_keys.issubset(set(data.keys())))

    def test_serialize_weights_are_lists(self):
        data = self.ma.serialize()
        for modality in ("visual", "audio"):
            self.assertIsInstance(data["forward_weights"][modality], list)
            self.assertIsInstance(data["backward_weights"][modality], list)

    def test_deserialize_roundtrip(self):
        data = self.ma.serialize()
        restored = MultimodalAssociation.deserialize(data)
        self.assertEqual(restored.association_size, self.ma.association_size)
        self.assertAlmostEqual(restored.learning_rate, self.ma.learning_rate)
        self.assertEqual(restored.association_mode, self.ma.association_mode)
        self.assertEqual(restored.update_count, self.ma.update_count)

    def test_deserialized_weights_match(self):
        data = self.ma.serialize()
        restored = MultimodalAssociation.deserialize(data)
        for modality in ("visual", "audio"):
            self.assertTrue(
                xp.allclose(
                    restored.forward_weights[modality],
                    self.ma.forward_weights[modality],
                    atol=1e-6,
                )
            )
            self.assertTrue(
                xp.allclose(
                    restored.backward_weights[modality],
                    self.ma.backward_weights[modality],
                    atol=1e-6,
                )
            )

    def test_deserialized_predictions_match(self):
        data = self.ma.serialize()
        restored = MultimodalAssociation.deserialize(data)
        audio = _random_audio(99)
        pred_orig = self.ma.get_cross_modal_prediction("audio", audio, "visual")
        pred_rest = restored.get_cross_modal_prediction("audio", audio, "visual")
        self.assertTrue(xp.allclose(pred_orig, pred_rest, atol=1e-6))

    def test_serialized_update_count(self):
        data = self.ma.serialize()
        self.assertEqual(data["update_count"], 3)

    def test_serialize_stable_associations(self):
        data = self.ma.serialize()
        self.assertIsInstance(data["stable_associations"], list)


# ===================================================================
# 7. Stats & Activity Tests
# ===================================================================
class TestStats(unittest.TestCase):
    """Test get_stats() and get_multimodal_activity()."""

    def setUp(self):
        self.ma = _make_association()
        v = _random_visual()
        a = _random_audio()
        self.ma.update({"visual": v, "audio": a})

    def test_get_stats_returns_dict(self):
        stats = self.ma.get_stats()
        self.assertIsInstance(stats, dict)

    def test_get_stats_keys(self):
        stats = self.ma.get_stats()
        for key in ("modalities", "association_size", "association_mode",
                     "reconstruction_errors", "weight_deltas",
                     "stable_associations", "weight_sparsity", "update_count"):
            self.assertIn(key, stats)

    def test_get_stats_modalities(self):
        stats = self.ma.get_stats()
        self.assertEqual(sorted(stats["modalities"]), ["audio", "visual"])

    def test_get_stats_update_count(self):
        stats = self.ma.get_stats()
        self.assertEqual(stats["update_count"], 1)

    def test_get_multimodal_activity_keys(self):
        activity = self.ma.get_multimodal_activity()
        for key in ("activations", "sparsity", "avg_activation", "active_count"):
            self.assertIn(key, activity)

    def test_get_multimodal_activity_shape(self):
        activity = self.ma.get_multimodal_activity()
        self.assertEqual(activity["activations"].shape, (12,))

    def test_weight_sparsity_in_range(self):
        stats = self.ma.get_stats()
        for modality, sparsity in stats["weight_sparsity"].items():
            self.assertGreaterEqual(sparsity, 0.0)
            self.assertLessEqual(sparsity, 1.0)


# ===================================================================
# 8. Weights Property Tests
# ===================================================================
class TestWeightsProperty(unittest.TestCase):
    """Test the weights property getter and setter."""

    def test_getter_returns_copies(self):
        ma = _make_association()
        w = ma.weights
        # Mutating the returned copy should not affect internal state
        w["visual"][:] = 999.0
        self.assertFalse(xp.allclose(ma.forward_weights["visual"], 999.0))

    def test_setter_updates_forward_weights(self):
        ma = _make_association()
        new_fw = {
            "visual": xp.ones((12, VISUAL_SIZE)) * 0.5,
            "audio": xp.ones((12, AUDIO_SIZE)) * 0.3,
        }
        ma.weights = new_fw
        self.assertTrue(xp.allclose(ma.forward_weights["visual"], 0.5))
        self.assertTrue(xp.allclose(ma.forward_weights["audio"], 0.3))

    def test_setter_rejects_non_dict(self):
        ma = _make_association()
        with self.assertRaises(ValueError):
            ma.weights = "bad"

    def test_setter_handles_shape_mismatch(self):
        """Setting weights with wrong shape should create new random weights."""
        ma = _make_association()
        # Provide wrong shape
        bad_weights = {
            "visual": xp.ones((5, 5)),
        }
        ma.weights = bad_weights
        # Should still have valid shape
        self.assertEqual(ma.forward_weights["visual"].shape, (12, VISUAL_SIZE))


# ===================================================================
# 9. Integration & Edge-case Tests
# ===================================================================
class TestIntegrationAndEdgeCases(unittest.TestCase):
    """Integration and boundary tests."""

    def test_zero_input(self):
        ma = _make_association()
        v = xp.zeros(VISUAL_SIZE)
        a = xp.zeros(AUDIO_SIZE)
        result = ma.update({"visual": v, "audio": a})
        # Should not crash; reconstructions should be finite
        for r in result["reconstructions"].values():
            self.assertTrue(xp.all(xp.isfinite(r)))

    def test_large_input(self):
        ma = _make_association()
        v = xp.ones(VISUAL_SIZE) * 1e6
        a = xp.ones(AUDIO_SIZE) * 1e6
        result = ma.update({"visual": v, "audio": a})
        act = result["association_activity"]
        self.assertTrue(xp.all(xp.isfinite(act)))

    def test_integrate_multiple_modalities(self):
        ma = _make_association()
        v = _random_visual()
        a = _random_audio()
        pred = ma.integrate_multiple_modalities(
            {"visual": v, "audio": a}, target_modality="visual"
        )
        self.assertEqual(pred.shape, (VISUAL_SIZE,))
        self.assertTrue(xp.all(xp.isfinite(pred)))

    def test_cross_modal_prediction_symmetry(self):
        """Both directions should produce finite, non-zero results."""
        ma = _make_association()
        v = _random_visual()
        a = _random_audio()
        p1 = ma.get_cross_modal_prediction("visual", v, "audio")
        p2 = ma.get_cross_modal_prediction("audio", a, "visual")
        self.assertTrue(xp.all(xp.isfinite(p1)))
        self.assertTrue(xp.all(xp.isfinite(p2)))
        self.assertFalse(xp.allclose(p1, 0.0))
        self.assertFalse(xp.allclose(p2, 0.0))

    def test_many_updates_no_crash(self):
        ma = _make_association()
        v = _random_visual()
        a = _random_audio()
        for _ in range(100):
            ma.update({"visual": v, "audio": a})
        self.assertEqual(ma.update_count, 100)
        self.assertTrue(xp.all(xp.isfinite(ma.association_activity)))

    def test_reconstruction_error_history_capped(self):
        """Error history should not exceed 1000 entries."""
        ma = _make_association()
        v = _random_visual()
        a = _random_audio()
        for _ in range(1100):
            ma.update({"visual": v, "audio": a})
        self.assertLessEqual(len(ma.reconstruction_errors["visual"]), 1000)

    def test_no_lateral_inhibition(self):
        ma = _make_association(lateral_inhibition=0.0)
        v = _random_visual()
        a = _random_audio()
        result = ma.update({"visual": v, "audio": a})
        self.assertTrue(xp.all(xp.isfinite(result["association_activity"])))

    def test_disable_sparse_coding(self):
        ma = _make_association(use_sparse_coding=False, lateral_inhibition=0.5)
        v = _random_visual()
        a = _random_audio()
        result = ma.update({"visual": v, "audio": a})
        self.assertTrue(xp.all(xp.isfinite(result["association_activity"])))


if __name__ == "__main__":
    unittest.main()
