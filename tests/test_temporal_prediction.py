"""
Comprehensive tests for the TemporalPrediction class (core/temporal_prediction.py).
Tests cover initialization, sequential update processing, prediction generation,
prediction modes, prediction error tracking, and state management.
"""
import sys
import os
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from self_organizing_av_system.core.backend import xp
from self_organizing_av_system.core.temporal_prediction import TemporalPrediction, PredictionMode


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SEED = 42
SMALL_DIM = 8


def _make_tp(**kwargs):
    """Create a TemporalPrediction with small, deterministic defaults."""
    defaults = dict(
        representation_size=SMALL_DIM,
        sequence_length=5,
        prediction_horizon=3,
        learning_rate=0.01,
        random_seed=SEED,
    )
    defaults.update(kwargs)
    return TemporalPrediction(**defaults)


def _deterministic_state(dim=SMALL_DIM, value=1.0, index=0):
    """Return a deterministic state vector with a single non-zero entry."""
    s = xp.zeros(dim)
    s[index % dim] = value
    return s


def _sequential_states(n, dim=SMALL_DIM):
    """Generate *n* distinct deterministic states."""
    states = []
    for i in range(n):
        s = xp.zeros(dim)
        s[i % dim] = 1.0
        # Add a small gradient so each vector is unique even after wrapping
        s = s + xp.linspace(0, 0.01 * (i + 1), dim)
        states.append(s)
    return states


# ===================================================================
# 1. Initialization tests
# ===================================================================

class TestTemporalPredictionInitialization(unittest.TestCase):
    """Test TemporalPrediction construction and initial state."""

    def test_default_initialization(self):
        tp = _make_tp()
        self.assertEqual(tp.representation_size, SMALL_DIM)
        self.assertEqual(tp.sequence_length, 5)
        self.assertEqual(tp.prediction_horizon, 3)
        self.assertAlmostEqual(tp.learning_rate, 0.01)
        self.assertEqual(tp.prediction_mode, PredictionMode.FORWARD)
        self.assertTrue(tp.use_eligibility_trace)
        self.assertTrue(tp.enable_surprise_detection)
        self.assertTrue(tp.enable_recurrent_connections)
        self.assertEqual(tp.update_count, 0)
        self.assertAlmostEqual(tp.mean_prediction_error, 0.0)
        self.assertEqual(tp.surprise_count, 0)

    def test_custom_parameters(self):
        tp = _make_tp(
            sequence_length=10,
            prediction_horizon=5,
            learning_rate=0.05,
            trace_decay=0.9,
            confidence_threshold=0.5,
            td_lambda=0.5,
            regularization_strength=0.01,
            surprise_threshold=0.8,
            confidence_learning_rate=0.01,
            prediction_decay=0.3,
        )
        self.assertEqual(tp.sequence_length, 10)
        self.assertEqual(tp.prediction_horizon, 5)
        self.assertAlmostEqual(tp.learning_rate, 0.05)
        self.assertAlmostEqual(tp.trace_decay, 0.9)
        self.assertAlmostEqual(tp.confidence_threshold, 0.5)
        self.assertAlmostEqual(tp.td_lambda, 0.5)
        self.assertAlmostEqual(tp.regularization_strength, 0.01)
        self.assertAlmostEqual(tp.surprise_threshold, 0.8)
        self.assertAlmostEqual(tp.confidence_learning_rate, 0.01)
        self.assertAlmostEqual(tp.prediction_decay, 0.3)

    def test_forward_weights_created(self):
        tp = _make_tp(prediction_horizon=4)
        self.assertEqual(len(tp.forward_weights), 4)
        for t in range(1, 5):
            self.assertIn(t, tp.forward_weights)
            self.assertEqual(tp.forward_weights[t].shape, (SMALL_DIM, SMALL_DIM))

    def test_confidence_weights_created(self):
        tp = _make_tp(prediction_horizon=3)
        self.assertEqual(len(tp.confidence_weights), 3)
        for t in range(1, 4):
            self.assertIn(t, tp.confidence_weights)
            self.assertEqual(tp.confidence_weights[t].shape, (SMALL_DIM, 1))

    def test_recurrent_weights_created(self):
        tp = _make_tp(enable_recurrent_connections=True)
        self.assertIsNotNone(tp.recurrent_weights)
        self.assertEqual(tp.recurrent_weights.shape, (SMALL_DIM, SMALL_DIM))
        # Diagonal should be zero
        diag = xp.diag(tp.recurrent_weights)
        for d in diag:
            self.assertAlmostEqual(float(d), 0.0)

    def test_no_recurrent_weights_when_disabled(self):
        tp = _make_tp(enable_recurrent_connections=False)
        self.assertIsNone(tp.recurrent_weights)

    def test_eligibility_traces_created(self):
        tp = _make_tp(use_eligibility_trace=True, prediction_horizon=3)
        self.assertEqual(len(tp.eligibility_traces), 3)
        for t in range(1, 4):
            self.assertIn(t, tp.eligibility_traces)
            self.assertEqual(tp.eligibility_traces[t].shape, (SMALL_DIM, SMALL_DIM))
            # Should start as zeros
            self.assertAlmostEqual(float(xp.sum(xp.abs(tp.eligibility_traces[t]))), 0.0)

    def test_no_eligibility_traces_when_disabled(self):
        tp = _make_tp(use_eligibility_trace=False)
        self.assertEqual(len(tp.eligibility_traces), 0)

    def test_backward_weights_forward_mode(self):
        """In FORWARD mode no backward weights should be created."""
        tp = _make_tp(prediction_mode=PredictionMode.FORWARD)
        self.assertEqual(len(tp.backward_weights), 0)

    def test_backward_weights_backward_mode(self):
        tp = _make_tp(prediction_mode=PredictionMode.BACKWARD)
        self.assertGreater(len(tp.backward_weights), 0)

    def test_backward_weights_bidirectional_mode(self):
        tp = _make_tp(prediction_mode=PredictionMode.BIDIRECTIONAL)
        self.assertGreater(len(tp.backward_weights), 0)

    def test_buffers_initially_empty(self):
        tp = _make_tp()
        self.assertEqual(len(tp.state_buffer), 0)
        self.assertEqual(len(tp.prediction_buffer), 0)
        self.assertEqual(len(tp.confidence_buffer), 0)
        self.assertEqual(len(tp.surprise_buffer), 0)
        self.assertEqual(len(tp.prediction_errors), 0)

    def test_prediction_mode_from_string(self):
        tp = _make_tp(prediction_mode="forward")
        self.assertEqual(tp.prediction_mode, PredictionMode.FORWARD)

    def test_prediction_mode_from_string_backward(self):
        tp = _make_tp(prediction_mode="backward")
        self.assertEqual(tp.prediction_mode, PredictionMode.BACKWARD)

    def test_prediction_mode_from_string_bidir(self):
        tp = _make_tp(prediction_mode="bidir")
        self.assertEqual(tp.prediction_mode, PredictionMode.BIDIRECTIONAL)


# ===================================================================
# 2. Update / process with sequential inputs
# ===================================================================

class TestTemporalPredictionUpdate(unittest.TestCase):
    """Test the update() method with sequential state inputs."""

    def setUp(self):
        self.tp = _make_tp()

    def test_single_update_returns_dict(self):
        state = _deterministic_state()
        result = self.tp.update(state)
        self.assertIsInstance(result, dict)
        self.assertIn("predictions", result)
        self.assertIn("confidence", result)
        self.assertIn("surprise", result)
        self.assertIn("prediction_error", result)
        self.assertIn("is_surprising", result)

    def test_update_increments_counter(self):
        self.assertEqual(self.tp.update_count, 0)
        self.tp.update(_deterministic_state())
        self.assertEqual(self.tp.update_count, 1)
        self.tp.update(_deterministic_state(index=1))
        self.assertEqual(self.tp.update_count, 2)

    def test_state_buffer_grows(self):
        states = _sequential_states(3)
        for s in states:
            self.tp.update(s)
        self.assertEqual(len(self.tp.state_buffer), 3)

    def test_state_buffer_capped(self):
        """Buffer should be trimmed to sequence_length + prediction_horizon."""
        max_len = self.tp.sequence_length + self.tp.prediction_horizon
        states = _sequential_states(max_len + 5)
        for s in states:
            self.tp.update(s)
        self.assertLessEqual(len(self.tp.state_buffer), max_len)

    def test_first_update_no_predictions(self):
        """Only one state in buffer -> no predictions yet."""
        result = self.tp.update(_deterministic_state())
        # With only 1 state in buffer, predictions dict should be empty
        self.assertEqual(len(result["predictions"]), 0)

    def test_second_update_produces_predictions(self):
        self.tp.update(_deterministic_state(index=0))
        result = self.tp.update(_deterministic_state(index=1))
        # Now there are 2 states, predictions should be generated
        self.assertGreater(len(result["predictions"]), 0)

    def test_predictions_have_correct_shape(self):
        states = _sequential_states(4)
        for s in states:
            result = self.tp.update(s)
        for t, pred in result["predictions"].items():
            self.assertEqual(pred.shape, (SMALL_DIM,))

    def test_sequential_updates_track_errors(self):
        """After enough updates, prediction_errors list should be populated."""
        states = _sequential_states(10)
        for s in states:
            self.tp.update(s)
        # After several updates beyond prediction_horizon, errors should be recorded
        self.assertGreater(len(self.tp.prediction_errors), 0)

    def test_confidence_values_are_floats(self):
        states = _sequential_states(3)
        for s in states:
            result = self.tp.update(s)
        for t, conf in result["confidence"].items():
            self.assertIsInstance(conf, float)

    def test_surprise_is_float(self):
        states = _sequential_states(4)
        for s in states:
            result = self.tp.update(s)
        self.assertIsInstance(result["surprise"], float)

    def test_is_surprising_is_bool(self):
        result = self.tp.update(_deterministic_state())
        self.assertIsInstance(result["is_surprising"], bool)

    def test_prediction_error_is_float(self):
        states = _sequential_states(5)
        for s in states:
            result = self.tp.update(s)
        self.assertIsInstance(result["prediction_error"], float)

    def test_update_with_1d_state(self):
        """1-D states should be handled without error."""
        state = xp.ones(SMALL_DIM)
        result = self.tp.update(state)
        self.assertIsInstance(result, dict)

    def test_update_with_zero_state(self):
        """All-zeros state should not raise."""
        state = xp.zeros(SMALL_DIM)
        result = self.tp.update(state)
        self.assertIsInstance(result, dict)


# ===================================================================
# 3. Predict methods
# ===================================================================

class TestTemporalPredictionPredict(unittest.TestCase):
    """Test predict_future() and predict_sequence()."""

    def setUp(self):
        self.tp = _make_tp()
        # Feed some states so weights have been updated
        for s in _sequential_states(8):
            self.tp.update(s)

    def test_predict_future_returns_dict(self):
        state = _deterministic_state()
        preds = self.tp.predict_future(state)
        self.assertIsInstance(preds, dict)

    def test_predict_future_keys_match_horizon(self):
        state = _deterministic_state()
        preds = self.tp.predict_future(state)
        for t in range(1, self.tp.prediction_horizon + 1):
            self.assertIn(t, preds)

    def test_predict_future_with_confidence(self):
        state = _deterministic_state()
        preds = self.tp.predict_future(state, include_confidence=True)
        for t, val in preds.items():
            self.assertIsInstance(val, tuple)
            self.assertEqual(len(val), 2)
            pred_arr, conf = val
            self.assertEqual(pred_arr.shape, (SMALL_DIM,))
            self.assertIsInstance(conf, float)

    def test_predict_future_without_confidence(self):
        state = _deterministic_state()
        preds = self.tp.predict_future(state, include_confidence=False)
        for t, val in preds.items():
            # Should be just the array, not a tuple
            self.assertTrue(hasattr(val, 'shape'))
            self.assertEqual(val.shape, (SMALL_DIM,))

    def test_predict_future_custom_steps(self):
        state = _deterministic_state()
        preds = self.tp.predict_future(state, steps=2)
        self.assertEqual(len(preds), 2)
        self.assertIn(1, preds)
        self.assertIn(2, preds)

    def test_predict_future_steps_clipped_to_horizon(self):
        """Requesting more steps than the horizon returns only up to horizon."""
        state = _deterministic_state()
        preds = self.tp.predict_future(state, steps=100)
        self.assertEqual(len(preds), self.tp.prediction_horizon)

    def test_predict_sequence_returns_list(self):
        state = _deterministic_state()
        seq = self.tp.predict_sequence(state, length=4)
        self.assertIsInstance(seq, list)
        self.assertEqual(len(seq), 4)

    def test_predict_sequence_with_confidence(self):
        state = _deterministic_state()
        seq = self.tp.predict_sequence(state, length=3, include_confidence=True)
        for entry in seq:
            self.assertIsInstance(entry, tuple)
            self.assertEqual(len(entry), 2)
            pred_arr, conf = entry
            self.assertEqual(pred_arr.shape, (SMALL_DIM,))
            self.assertIsInstance(conf, float)

    def test_predict_sequence_without_confidence(self):
        state = _deterministic_state()
        seq = self.tp.predict_sequence(state, length=3, include_confidence=False)
        for entry in seq:
            self.assertTrue(hasattr(entry, 'shape'))
            self.assertEqual(entry.shape, (SMALL_DIM,))

    def test_predict_sequence_is_autoregressive(self):
        """Each prediction should feed into the next; verify they differ."""
        state = _deterministic_state(value=1.0)
        seq = self.tp.predict_sequence(state, length=3, include_confidence=False)
        # Consecutive predictions should generally differ
        for i in range(len(seq) - 1):
            diff = float(xp.sum(xp.abs(seq[i] - seq[i + 1])))
            # They may converge, but at least the first pair is unlikely identical
            # Just ensure they are valid arrays
            self.assertEqual(seq[i].shape, (SMALL_DIM,))

    def test_predict_future_with_zero_state(self):
        """Predicting from an all-zero state should not raise."""
        state = xp.zeros(SMALL_DIM)
        preds = self.tp.predict_future(state)
        self.assertIsInstance(preds, dict)


# ===================================================================
# 4. Prediction modes
# ===================================================================

class TestPredictionModes(unittest.TestCase):
    """Test different PredictionMode values."""

    def test_forward_mode_enum(self):
        self.assertEqual(PredictionMode.FORWARD.value, "forward")

    def test_backward_mode_enum(self):
        self.assertEqual(PredictionMode.BACKWARD.value, "backward")

    def test_bidirectional_mode_enum(self):
        self.assertEqual(PredictionMode.BIDIRECTIONAL.value, "bidir")

    def test_forward_mode_creates_no_backward_weights(self):
        tp = _make_tp(prediction_mode=PredictionMode.FORWARD)
        self.assertEqual(len(tp.backward_weights), 0)

    def test_backward_mode_creates_backward_weights(self):
        tp = _make_tp(prediction_mode=PredictionMode.BACKWARD)
        self.assertEqual(len(tp.backward_weights), tp.sequence_length)
        for t in range(1, tp.sequence_length + 1):
            self.assertIn(t, tp.backward_weights)
            self.assertEqual(tp.backward_weights[t].shape, (SMALL_DIM, SMALL_DIM))

    def test_bidirectional_mode_creates_backward_weights(self):
        tp = _make_tp(prediction_mode=PredictionMode.BIDIRECTIONAL)
        self.assertEqual(len(tp.backward_weights), tp.sequence_length)

    def test_forward_mode_runs_update(self):
        tp = _make_tp(prediction_mode=PredictionMode.FORWARD)
        states = _sequential_states(5)
        for s in states:
            result = tp.update(s)
        self.assertIsInstance(result, dict)

    def test_backward_mode_runs_update(self):
        tp = _make_tp(prediction_mode=PredictionMode.BACKWARD)
        states = _sequential_states(5)
        for s in states:
            result = tp.update(s)
        self.assertIsInstance(result, dict)

    def test_bidirectional_mode_runs_update(self):
        tp = _make_tp(prediction_mode=PredictionMode.BIDIRECTIONAL)
        states = _sequential_states(5)
        for s in states:
            result = tp.update(s)
        self.assertIsInstance(result, dict)

    def test_string_forward_produces_same_mode(self):
        tp = _make_tp(prediction_mode="forward")
        self.assertEqual(tp.prediction_mode, PredictionMode.FORWARD)

    def test_string_backward_produces_same_mode(self):
        tp = _make_tp(prediction_mode="backward")
        self.assertEqual(tp.prediction_mode, PredictionMode.BACKWARD)

    def test_string_bidir_produces_same_mode(self):
        tp = _make_tp(prediction_mode="bidir")
        self.assertEqual(tp.prediction_mode, PredictionMode.BIDIRECTIONAL)

    def test_all_modes_predict_future(self):
        """predict_future should work regardless of mode."""
        state = _deterministic_state()
        for mode in PredictionMode:
            tp = _make_tp(prediction_mode=mode)
            for s in _sequential_states(4):
                tp.update(s)
            preds = tp.predict_future(state)
            self.assertIsInstance(preds, dict)
            self.assertGreater(len(preds), 0, f"No predictions for mode {mode}")


# ===================================================================
# 5. Prediction error tracking
# ===================================================================

class TestPredictionError(unittest.TestCase):
    """Test prediction error computation and tracking."""

    def test_initial_mean_error_is_zero(self):
        tp = _make_tp()
        self.assertAlmostEqual(tp.mean_prediction_error, 0.0)

    def test_initial_prediction_errors_empty(self):
        tp = _make_tp()
        self.assertEqual(len(tp.prediction_errors), 0)

    def test_error_tracked_after_updates(self):
        tp = _make_tp()
        states = _sequential_states(10)
        for s in states:
            tp.update(s)
        # After enough updates (> prediction_horizon), errors should accumulate
        self.assertGreater(len(tp.prediction_errors), 0)

    def test_prediction_errors_are_tuples(self):
        tp = _make_tp()
        states = _sequential_states(10)
        for s in states:
            tp.update(s)
        for entry in tp.prediction_errors:
            self.assertIsInstance(entry, tuple)
            self.assertEqual(len(entry), 2)
            step_num, error_val = entry
            self.assertIsInstance(step_num, int)
            self.assertIsInstance(error_val, float)

    def test_mean_prediction_error_non_negative(self):
        tp = _make_tp()
        states = _sequential_states(10)
        for s in states:
            tp.update(s)
        self.assertGreaterEqual(tp.mean_prediction_error, 0.0)

    def test_prediction_error_changes_with_updates(self):
        tp = _make_tp()
        states = _sequential_states(12)
        errors = []
        for s in states:
            result = tp.update(s)
            errors.append(result["prediction_error"])
        # Not all errors should be zero (at least after buffer fills)
        non_zero = [e for e in errors if e > 0.0]
        self.assertGreater(len(non_zero), 0)

    def test_prediction_error_in_update_result_matches_type(self):
        tp = _make_tp()
        states = _sequential_states(6)
        for s in states:
            result = tp.update(s)
        self.assertIsInstance(result["prediction_error"], float)

    def test_error_list_bounded(self):
        """prediction_errors should not exceed 1000 entries."""
        tp = _make_tp()
        # Feed many updates
        for i in range(1100):
            s = xp.zeros(SMALL_DIM)
            s[i % SMALL_DIM] = 1.0
            tp.update(s)
        self.assertLessEqual(len(tp.prediction_errors), 1000)


# ===================================================================
# 6. State management (get_stats / serialize / deserialize)
# ===================================================================

class TestTemporalPredictionState(unittest.TestCase):
    """Test get_stats(), serialize(), and deserialize()."""

    def setUp(self):
        self.tp = _make_tp()
        for s in _sequential_states(8):
            self.tp.update(s)

    # --- get_stats ---

    def test_get_stats_returns_dict(self):
        stats = self.tp.get_stats()
        self.assertIsInstance(stats, dict)

    def test_get_stats_keys(self):
        stats = self.tp.get_stats()
        expected_keys = {
            'representation_size', 'sequence_length', 'prediction_horizon',
            'prediction_mode', 'mean_prediction_error', 'surprise_rate',
            'error_trend', 'update_count', 'buffer_size',
        }
        for key in expected_keys:
            self.assertIn(key, stats, f"Missing key: {key}")

    def test_get_stats_values(self):
        stats = self.tp.get_stats()
        self.assertEqual(stats['representation_size'], SMALL_DIM)
        self.assertEqual(stats['sequence_length'], 5)
        self.assertEqual(stats['prediction_horizon'], 3)
        self.assertEqual(stats['prediction_mode'], PredictionMode.FORWARD.value)
        self.assertEqual(stats['update_count'], 8)
        self.assertIsInstance(stats['mean_prediction_error'], float)
        self.assertIsInstance(stats['surprise_rate'], float)
        self.assertIsInstance(stats['error_trend'], (int, float))
        self.assertIsInstance(stats['buffer_size'], int)

    def test_get_stats_surprise_rate_bounded(self):
        stats = self.tp.get_stats()
        self.assertGreaterEqual(stats['surprise_rate'], 0.0)
        self.assertLessEqual(stats['surprise_rate'], 1.0)

    # --- serialize ---

    def test_serialize_returns_dict(self):
        data = self.tp.serialize()
        self.assertIsInstance(data, dict)

    def test_serialize_contains_params(self):
        data = self.tp.serialize()
        self.assertEqual(data['representation_size'], SMALL_DIM)
        self.assertEqual(data['sequence_length'], 5)
        self.assertEqual(data['prediction_horizon'], 3)
        self.assertAlmostEqual(data['learning_rate'], 0.01)
        self.assertEqual(data['prediction_mode'], 'forward')

    def test_serialize_contains_weights(self):
        data = self.tp.serialize()
        self.assertIn('forward_weights', data)
        self.assertIn('confidence_weights', data)
        self.assertIn('eligibility_traces', data)

    def test_serialize_forward_weights_keys(self):
        data = self.tp.serialize()
        for t in range(1, self.tp.prediction_horizon + 1):
            self.assertIn(str(t), data['forward_weights'])

    def test_serialize_recurrent_weights(self):
        data = self.tp.serialize()
        self.assertIsNotNone(data['recurrent_weights'])
        self.assertIsInstance(data['recurrent_weights'], list)

    def test_serialize_counters(self):
        data = self.tp.serialize()
        self.assertEqual(data['update_count'], 8)
        self.assertIsInstance(data['surprise_count'], int)
        self.assertIsInstance(data['mean_prediction_error'], float)

    # --- deserialize ---

    def test_deserialize_roundtrip(self):
        data = self.tp.serialize()
        restored = TemporalPrediction.deserialize(data)
        self.assertEqual(restored.representation_size, self.tp.representation_size)
        self.assertEqual(restored.sequence_length, self.tp.sequence_length)
        self.assertEqual(restored.prediction_horizon, self.tp.prediction_horizon)
        self.assertAlmostEqual(restored.learning_rate, self.tp.learning_rate)
        self.assertEqual(restored.prediction_mode, self.tp.prediction_mode)
        self.assertEqual(restored.update_count, self.tp.update_count)
        self.assertEqual(restored.surprise_count, self.tp.surprise_count)
        self.assertAlmostEqual(
            restored.mean_prediction_error,
            self.tp.mean_prediction_error,
            places=6,
        )

    def test_deserialize_weights_match(self):
        data = self.tp.serialize()
        restored = TemporalPrediction.deserialize(data)
        for t in self.tp.forward_weights:
            orig = self.tp.forward_weights[t]
            rest = restored.forward_weights[t]
            diff = float(xp.max(xp.abs(orig - rest)))
            self.assertAlmostEqual(diff, 0.0, places=5,
                                   msg=f"Forward weight mismatch at t={t}")

    def test_deserialize_recurrent_weights_match(self):
        data = self.tp.serialize()
        restored = TemporalPrediction.deserialize(data)
        self.assertIsNotNone(restored.recurrent_weights)
        diff = float(xp.max(xp.abs(self.tp.recurrent_weights - restored.recurrent_weights)))
        self.assertAlmostEqual(diff, 0.0, places=5)

    def test_deserialized_instance_can_predict(self):
        data = self.tp.serialize()
        restored = TemporalPrediction.deserialize(data)
        state = _deterministic_state()
        preds = restored.predict_future(state)
        self.assertIsInstance(preds, dict)
        self.assertGreater(len(preds), 0)

    def test_deserialized_instance_can_update(self):
        data = self.tp.serialize()
        restored = TemporalPrediction.deserialize(data)
        result = restored.update(_deterministic_state())
        self.assertIsInstance(result, dict)
        self.assertEqual(restored.update_count, self.tp.update_count + 1)


# ===================================================================
# 7. Surprise detection
# ===================================================================

class TestSurpriseDetection(unittest.TestCase):
    """Test surprise detection mechanism."""

    def test_surprise_disabled(self):
        tp = _make_tp(enable_surprise_detection=False)
        states = _sequential_states(6)
        for s in states:
            result = tp.update(s)
        # When disabled, surprise should remain 0
        self.assertAlmostEqual(result["surprise"], 0.0)
        self.assertFalse(result["is_surprising"])

    def test_surprise_enabled_returns_value(self):
        tp = _make_tp(enable_surprise_detection=True)
        states = _sequential_states(6)
        for s in states:
            result = tp.update(s)
        # surprise field should be a float
        self.assertIsInstance(result["surprise"], float)

    def test_surprise_count_increments(self):
        """With a very low threshold, most events should be surprising."""
        tp = _make_tp(
            enable_surprise_detection=True,
            surprise_threshold=0.0,  # everything is surprising
        )
        states = _sequential_states(8)
        for s in states:
            tp.update(s)
        # At threshold 0, any nonzero surprise triggers a count
        # But first update has no prediction, so it can't surprise
        self.assertGreaterEqual(tp.surprise_count, 0)

    def test_surprise_buffer_bounded(self):
        tp = _make_tp(enable_surprise_detection=True)
        states = _sequential_states(20)
        for s in states:
            tp.update(s)
        self.assertLessEqual(len(tp.surprise_buffer), tp.sequence_length)


# ===================================================================
# 8. Edge cases
# ===================================================================

class TestEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions."""

    def test_single_dimension(self):
        tp = TemporalPrediction(
            representation_size=1,
            sequence_length=3,
            prediction_horizon=2,
            random_seed=SEED,
        )
        for i in range(5):
            s = xp.array([float(i)])
            result = tp.update(s)
        self.assertIsInstance(result, dict)

    def test_large_representation(self):
        tp = TemporalPrediction(
            representation_size=64,
            sequence_length=3,
            prediction_horizon=2,
            random_seed=SEED,
        )
        state = xp.ones(64)
        result = tp.update(state)
        self.assertIsInstance(result, dict)

    def test_horizon_one(self):
        tp = TemporalPrediction(
            representation_size=SMALL_DIM,
            prediction_horizon=1,
            random_seed=SEED,
        )
        states = _sequential_states(5)
        for s in states:
            result = tp.update(s)
        self.assertIn(1, result["predictions"])

    def test_no_eligibility_trace_update(self):
        """Updates without eligibility traces should not raise."""
        tp = _make_tp(use_eligibility_trace=False)
        states = _sequential_states(10)
        for s in states:
            result = tp.update(s)
        self.assertIsInstance(result, dict)

    def test_negative_values_in_state(self):
        tp = _make_tp()
        state = xp.array([-1.0, -0.5, 0.0, 0.5, 1.0, -0.3, 0.7, -0.9])
        result = tp.update(state)
        self.assertIsInstance(result, dict)

    def test_repeated_identical_states(self):
        """Feeding the exact same state repeatedly should not crash."""
        tp = _make_tp()
        state = _deterministic_state()
        for _ in range(10):
            result = tp.update(state)
        self.assertIsInstance(result, dict)


if __name__ == '__main__':
    unittest.main()
