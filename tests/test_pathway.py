"""
Comprehensive tests for the NeuralPathway class (core/pathway.py).
Tests cover initialization, feedforward processing, learning, prediction,
top-down generation, pruning, structural plasticity, and state management.
"""
import sys
import os
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from self_organizing_av_system.core.backend import xp
from self_organizing_av_system.core.pathway import NeuralPathway


def _seed():
    """Set a fixed random seed for deterministic tests."""
    xp.random.seed(42)


class TestNeuralPathwayInitialization(unittest.TestCase):
    """Test NeuralPathway construction and initial state."""

    def setUp(self):
        _seed()

    def test_default_parameters(self):
        pw = NeuralPathway(name="test", input_size=8, layer_sizes=[8, 4])
        self.assertEqual(pw.name, "test")
        self.assertEqual(pw.input_size, 8)
        self.assertEqual(pw.num_layers, 2)
        self.assertIsNone(pw.current_input)
        self.assertIsNone(pw.predicted_next_input)
        self.assertEqual(len(pw.error_history), 0)
        # previous_layer_activations should have one entry per layer, all None
        self.assertEqual(len(pw.previous_layer_activations), 2)
        for act in pw.previous_layer_activations:
            self.assertIsNone(act)

    def test_custom_parameters(self):
        pw = NeuralPathway(
            name="custom",
            input_size=8,
            layer_sizes=[8, 4],
            layer_k_winners=[2, 1],
            learning_rates=[0.05, 0.02],
            use_recurrent=False,
        )
        self.assertEqual(pw.layers[0].k_winners, 2)
        self.assertEqual(pw.layers[1].k_winners, 1)
        self.assertAlmostEqual(pw.layers[0].learning_rate, 0.05)
        self.assertAlmostEqual(pw.layers[1].learning_rate, 0.02)
        # Recurrent connections should NOT be initialized
        for layer in pw.layers:
            self.assertIsNone(layer.recurrent_weights)

    def test_layer_creation_sizes(self):
        pw = NeuralPathway(name="sizes", input_size=8, layer_sizes=[8, 4])
        # First layer: input_size=8, layer_size=8
        self.assertEqual(pw.layers[0].input_size, 8)
        self.assertEqual(pw.layers[0].layer_size, 8)
        # Second layer: input_size=8 (output of first), layer_size=4
        self.assertEqual(pw.layers[1].input_size, 8)
        self.assertEqual(pw.layers[1].layer_size, 4)

    def test_layer_names(self):
        pw = NeuralPathway(name="demo", input_size=8, layer_sizes=[8, 4])
        self.assertEqual(pw.layers[0].name, "demo_layer_0")
        self.assertEqual(pw.layers[1].name, "demo_layer_1")

    def test_default_k_winners(self):
        # k_winners defaults to ~10% of layer size, minimum 1
        pw = NeuralPathway(name="kw", input_size=8, layer_sizes=[8, 4])
        # 10% of 8 = 0.8 -> max(1, 0) = 1; int(0.8)=0 -> max(1,0)=1
        self.assertGreaterEqual(pw.layers[0].k_winners, 1)
        self.assertGreaterEqual(pw.layers[1].k_winners, 1)

    def test_default_learning_rates_decrease(self):
        pw = NeuralPathway(name="lr", input_size=8, layer_sizes=[8, 4, 2])
        lr0 = pw.layers[0].learning_rate
        lr1 = pw.layers[1].learning_rate
        lr2 = pw.layers[2].learning_rate
        self.assertGreater(lr0, lr1)
        self.assertGreater(lr1, lr2)

    def test_recurrent_connections_default(self):
        pw = NeuralPathway(name="rec", input_size=8, layer_sizes=[8, 4])
        for layer in pw.layers:
            self.assertIsNotNone(layer.recurrent_weights)

    def test_single_layer_pathway(self):
        pw = NeuralPathway(name="single", input_size=8, layer_sizes=[4])
        self.assertEqual(pw.num_layers, 1)
        self.assertEqual(pw.layers[0].input_size, 8)
        self.assertEqual(pw.layers[0].layer_size, 4)


class TestNeuralPathwayProcess(unittest.TestCase):
    """Test feedforward processing through the pathway."""

    def setUp(self):
        _seed()
        self.pw = NeuralPathway(name="proc", input_size=8, layer_sizes=[8, 4])

    def test_feedforward_returns_array(self):
        inputs = xp.random.rand(8)
        output = self.pw.process(inputs)
        self.assertIsNotNone(output)
        self.assertEqual(output.shape, (4,))

    def test_output_shape_matches_last_layer(self):
        _seed()
        pw = NeuralPathway(name="shape", input_size=8, layer_sizes=[8, 4, 2])
        inputs = xp.random.rand(8)
        output = pw.process(inputs)
        self.assertEqual(output.shape, (2,))

    def test_input_size_validation(self):
        wrong_input = xp.random.rand(5)
        with self.assertRaises(ValueError):
            self.pw.process(wrong_input)

    def test_current_input_stored(self):
        inputs = xp.random.rand(8)
        self.pw.process(inputs)
        self.assertIsNotNone(self.pw.current_input)
        self.assertTrue(xp.allclose(self.pw.current_input, inputs))

    def test_output_non_negative(self):
        # Activations should be non-negative (ReLU / k-WTA)
        inputs = xp.random.rand(8)
        output = self.pw.process(inputs)
        self.assertTrue(xp.all(output >= 0))

    def test_process_with_time_step(self):
        inputs = xp.random.rand(8)
        output = self.pw.process(inputs, time_step=10)
        self.assertEqual(output.shape, (4,))

    def test_process_multiple_times(self):
        # Process several inputs sequentially; should not error
        for t in range(5):
            inputs = xp.random.rand(8)
            output = self.pw.process(inputs, time_step=t * 10)
            self.assertEqual(output.shape, (4,))


class TestNeuralPathwayLearn(unittest.TestCase):
    """Test learning across all layers."""

    def setUp(self):
        _seed()
        self.pw = NeuralPathway(
            name="learn",
            input_size=8,
            layer_sizes=[8, 4],
            use_recurrent=False,
        )

    def _get_all_weights(self):
        """Snapshot all neuron weights in the pathway."""
        all_weights = []
        for layer in self.pw.layers:
            layer_weights = [n.weights.copy() for n in layer.neurons]
            all_weights.append(layer_weights)
        return all_weights

    def test_learn_no_input_noop(self):
        # If process has not been called, learn should be a no-op
        old = self._get_all_weights()
        self.pw.learn('oja')
        new = self._get_all_weights()
        for l_old, l_new in zip(old, new):
            for w_old, w_new in zip(l_old, l_new):
                self.assertTrue(xp.allclose(w_old, w_new))

    def test_learn_oja(self):
        # Process a strong input to trigger activation, then learn
        inputs = xp.abs(xp.random.rand(8)) + 0.5
        self.pw.process(inputs, time_step=100)
        old_weights = self._get_all_weights()
        self.pw.learn('oja')
        new_weights = self._get_all_weights()
        # At least one weight in the pathway should have changed
        any_changed = False
        for l_old, l_new in zip(old_weights, new_weights):
            for w_old, w_new in zip(l_old, l_new):
                if not xp.allclose(w_old, w_new, atol=1e-10):
                    any_changed = True
                    break
            if any_changed:
                break
        self.assertTrue(any_changed, "Oja learning should change at least some weights")

    def test_learn_hebbian(self):
        inputs = xp.abs(xp.random.rand(8)) + 0.5
        self.pw.process(inputs, time_step=100)
        old_weights = self._get_all_weights()
        self.pw.learn('hebbian')
        new_weights = self._get_all_weights()
        any_changed = False
        for l_old, l_new in zip(old_weights, new_weights):
            for w_old, w_new in zip(l_old, l_new):
                if not xp.allclose(w_old, w_new, atol=1e-10):
                    any_changed = True
                    break
            if any_changed:
                break
        self.assertTrue(any_changed, "Hebbian learning should change at least some weights")

    def test_learn_stdp(self):
        inputs = xp.abs(xp.random.rand(8)) + 0.5
        self.pw.process(inputs, time_step=100)
        old_weights = self._get_all_weights()
        self.pw.learn('stdp')
        new_weights = self._get_all_weights()
        any_changed = False
        for l_old, l_new in zip(old_weights, new_weights):
            for w_old, w_new in zip(l_old, l_new):
                if not xp.allclose(w_old, w_new, atol=1e-10):
                    any_changed = True
                    break
            if any_changed:
                break
        self.assertTrue(any_changed, "STDP learning should change at least some weights")


class TestNeuralPathwayPredictNext(unittest.TestCase):
    """Test predict_next returns predictions for each layer."""

    def setUp(self):
        _seed()
        self.pw = NeuralPathway(name="pred", input_size=8, layer_sizes=[8, 4])

    def test_predictions_list_length(self):
        preds = self.pw.predict_next()
        self.assertEqual(len(preds), 2)

    def test_prediction_shapes(self):
        preds = self.pw.predict_next()
        self.assertEqual(preds[0].shape, (8,))
        self.assertEqual(preds[1].shape, (4,))

    def test_predictions_zeros_before_processing(self):
        # Before any input is processed, activations are zero so predictions
        # should be zero vectors (recurrent on zero activations).
        preds = self.pw.predict_next()
        for pred in preds:
            self.assertTrue(xp.allclose(pred, 0.0))

    def test_predictions_after_processing(self):
        # After processing, at least some predictions may be nonzero
        inputs = xp.abs(xp.random.rand(8)) + 0.5
        self.pw.process(inputs, time_step=100)
        preds = self.pw.predict_next()
        self.assertEqual(len(preds), 2)
        # Each prediction should be non-negative (ReLU applied)
        for pred in preds:
            self.assertTrue(xp.all(pred >= 0))


class TestNeuralPathwayGenerateFromTop(unittest.TestCase):
    """Test top-down generation from top layer activation."""

    def setUp(self):
        _seed()
        self.pw = NeuralPathway(name="gen", input_size=8, layer_sizes=[8, 4])

    def test_output_shape(self):
        top_act = xp.random.rand(4)
        result = self.pw.generate_from_top(top_act)
        self.assertEqual(result.shape, (8,))

    def test_output_range_clipped(self):
        top_act = xp.random.rand(4) * 5.0
        result = self.pw.generate_from_top(top_act)
        self.assertTrue(xp.all(result >= 0.0))
        self.assertTrue(xp.all(result <= 1.0))

    def test_zero_input_returns_valid(self):
        top_act = xp.zeros(4)
        result = self.pw.generate_from_top(top_act)
        self.assertEqual(result.shape, (8,))
        # All zeros in produces clipped zeros
        self.assertTrue(xp.all(result >= 0.0))
        self.assertTrue(xp.all(result <= 1.0))

    def test_does_not_modify_input(self):
        top_act = xp.random.rand(4)
        original = top_act.copy()
        self.pw.generate_from_top(top_act)
        self.assertTrue(xp.allclose(top_act, original))


class TestNeuralPathwayPrune(unittest.TestCase):
    """Test pruning weak connections across all layers."""

    def setUp(self):
        _seed()
        self.pw = NeuralPathway(name="prune", input_size=8, layer_sizes=[8, 4])

    def test_returns_dict(self):
        stats = self.pw.prune_pathway(threshold=0.01)
        self.assertIsInstance(stats, dict)

    def test_stats_have_expected_keys(self):
        stats = self.pw.prune_pathway(threshold=0.01)
        self.assertIn("layer_0", stats)
        self.assertIn("layer_1", stats)

    def test_stats_values_are_ints(self):
        stats = self.pw.prune_pathway(threshold=0.01)
        for key, val in stats.items():
            self.assertIsInstance(val, int)

    def test_high_threshold_prunes_more(self):
        # A very high threshold should prune many connections
        stats_low = self.pw.prune_pathway(threshold=0.001)
        # Recreate pathway for a fresh comparison
        _seed()
        pw2 = NeuralPathway(name="prune2", input_size=8, layer_sizes=[8, 4])
        stats_high = pw2.prune_pathway(threshold=10.0)
        total_low = sum(stats_low.values())
        total_high = sum(stats_high.values())
        self.assertGreaterEqual(total_high, total_low)


class TestNeuralPathwayAddNeurons(unittest.TestCase):
    """Test add_neurons_where_needed structural plasticity."""

    def setUp(self):
        _seed()
        self.pw = NeuralPathway(name="add", input_size=8, layer_sizes=[8, 4])

    def test_returns_dict(self):
        result = self.pw.add_neurons_where_needed(max_new_per_layer=1)
        self.assertIsInstance(result, dict)

    def test_keys_per_layer(self):
        result = self.pw.add_neurons_where_needed()
        self.assertIn("layer_0", result)
        self.assertIn("layer_1", result)

    def test_no_additions_without_history(self):
        # Without enough history (100 entries), no neurons should be added
        result = self.pw.add_neurons_where_needed()
        for key, val in result.items():
            self.assertEqual(val, 0)

    def test_additions_with_high_activation_history(self):
        # Manually fill history so the condition is met
        for layer in self.pw.layers:
            layer.mean_activation_history = [0.5] * 200
            layer.sparsity_history = [0.5] * 200
        old_sizes = [layer.layer_size for layer in self.pw.layers]
        result = self.pw.add_neurons_where_needed(max_new_per_layer=1)
        # At least one layer should have a neuron added
        total_added = sum(result.values())
        self.assertGreater(total_added, 0)
        # Verify actual layer sizes grew
        for i, layer in enumerate(self.pw.layers):
            self.assertEqual(layer.layer_size, old_sizes[i] + result[f"layer_{i}"])


class TestNeuralPathwayReplaceDeadNeurons(unittest.TestCase):
    """Test replace_dead_neurons structural plasticity."""

    def setUp(self):
        _seed()
        self.pw = NeuralPathway(
            name="replace",
            input_size=8,
            layer_sizes=[8, 4],
            use_recurrent=False,
        )

    def test_returns_dict(self):
        result = self.pw.replace_dead_neurons()
        self.assertIsInstance(result, dict)

    def test_keys_per_layer(self):
        result = self.pw.replace_dead_neurons()
        self.assertIn("layer_0", result)
        self.assertIn("layer_1", result)

    def test_no_replacements_without_input(self):
        # Without a processed input, no replacements should happen
        result = self.pw.replace_dead_neurons()
        total = sum(result.values())
        self.assertEqual(total, 0)

    def test_replaces_dead_neurons_after_processing(self):
        # Process an input so current_input is set
        inputs = xp.abs(xp.random.rand(8)) + 0.5
        self.pw.process(inputs, time_step=100)
        # All neurons start with recent_mean_activation == 0.0 (below threshold)
        # which qualifies them as dead. However, after processing some may have
        # activated. Set all to dead manually for a definitive test.
        for layer in self.pw.layers:
            for neuron in layer.neurons:
                neuron.recent_mean_activation = 0.0
        result = self.pw.replace_dead_neurons(min_activation_threshold=0.01)
        total = sum(result.values())
        self.assertGreater(total, 0)


class TestNeuralPathwayGetState(unittest.TestCase):
    """Test get_pathway_state returns a well-formed dictionary."""

    def setUp(self):
        _seed()
        self.pw = NeuralPathway(name="state", input_size=8, layer_sizes=[8, 4])

    def test_returns_dict(self):
        state = self.pw.get_pathway_state()
        self.assertIsInstance(state, dict)

    def test_expected_keys(self):
        state = self.pw.get_pathway_state()
        self.assertIn('name', state)
        self.assertIn('num_layers', state)
        self.assertIn('layers', state)

    def test_name_value(self):
        state = self.pw.get_pathway_state()
        self.assertEqual(state['name'], "state")

    def test_num_layers_value(self):
        state = self.pw.get_pathway_state()
        self.assertEqual(state['num_layers'], 2)

    def test_layers_list_length(self):
        state = self.pw.get_pathway_state()
        self.assertEqual(len(state['layers']), 2)

    def test_layer_state_keys(self):
        state = self.pw.get_pathway_state()
        for layer_state in state['layers']:
            self.assertIn('name', layer_state)
            self.assertIn('input_size', layer_state)
            self.assertIn('layer_size', layer_state)
            self.assertIn('learning_rate', layer_state)
            self.assertIn('threshold', layer_state)
            self.assertIn('activations', layer_state)


class TestNeuralPathwayGetAllLayerActivations(unittest.TestCase):
    """Test get_all_layer_activations."""

    def setUp(self):
        _seed()
        self.pw = NeuralPathway(name="act", input_size=8, layer_sizes=[8, 4])

    def test_returns_list(self):
        acts = self.pw.get_all_layer_activations()
        self.assertIsInstance(acts, list)

    def test_list_length(self):
        acts = self.pw.get_all_layer_activations()
        self.assertEqual(len(acts), 2)

    def test_shapes_match_layers(self):
        acts = self.pw.get_all_layer_activations()
        self.assertEqual(acts[0].shape, (8,))
        self.assertEqual(acts[1].shape, (4,))

    def test_all_zeros_before_processing(self):
        acts = self.pw.get_all_layer_activations()
        for act in acts:
            self.assertTrue(xp.allclose(act, 0.0))

    def test_returns_copies(self):
        # Modifying returned activations should not affect the pathway
        acts = self.pw.get_all_layer_activations()
        acts[0][:] = 999.0
        original = self.pw.layers[0].activations
        self.assertFalse(xp.allclose(original, 999.0))


class TestNeuralPathwayGetPredictionError(unittest.TestCase):
    """Test get_prediction_error computation."""

    def setUp(self):
        _seed()
        self.pw = NeuralPathway(name="err", input_size=8, layer_sizes=[8, 4])

    def test_returns_float(self):
        # Process first so layers have activations
        inputs = xp.random.rand(8)
        self.pw.process(inputs, time_step=0)
        actual_next = xp.random.rand(8)
        error = self.pw.get_prediction_error(actual_next)
        self.assertIsInstance(error, float)

    def test_error_non_negative(self):
        inputs = xp.random.rand(8)
        self.pw.process(inputs, time_step=0)
        actual_next = xp.random.rand(8)
        error = self.pw.get_prediction_error(actual_next)
        self.assertGreaterEqual(error, 0.0)

    def test_identical_prediction_low_error(self):
        # Process and then generate a predicted input; comparing with itself
        # should yield relatively low error
        inputs = xp.random.rand(8)
        self.pw.process(inputs, time_step=0)
        top_act = self.pw.layers[-1].activations
        predicted = self.pw.generate_from_top(top_act)
        error = self.pw.get_prediction_error(predicted)
        # The error comparing the generated input with itself should be small
        self.assertLess(error, 0.5)

    def test_stores_last_prediction_error(self):
        inputs = xp.random.rand(8)
        self.pw.process(inputs, time_step=0)
        actual_next = xp.random.rand(8)
        error = self.pw.get_prediction_error(actual_next)
        self.assertTrue(hasattr(self.pw, '_last_prediction_error'))
        self.assertAlmostEqual(float(self.pw._last_prediction_error), error, places=5)

    def test_mismatched_sizes(self):
        # get_prediction_error handles size mismatches by truncating
        inputs = xp.random.rand(8)
        self.pw.process(inputs, time_step=0)
        actual_next = xp.random.rand(12)
        error = self.pw.get_prediction_error(actual_next)
        self.assertIsInstance(error, float)
        self.assertGreaterEqual(error, 0.0)


class TestNeuralPathwayRepr(unittest.TestCase):
    """Test __repr__ method."""

    def setUp(self):
        _seed()
        self.pw = NeuralPathway(name="repr_test", input_size=8, layer_sizes=[8, 4])

    def test_contains_class_name(self):
        r = repr(self.pw)
        self.assertIn("NeuralPathway", r)

    def test_contains_pathway_name(self):
        r = repr(self.pw)
        self.assertIn("repr_test", r)

    def test_contains_layer_sizes(self):
        r = repr(self.pw)
        self.assertIn("8", r)
        self.assertIn("4", r)


if __name__ == '__main__':
    unittest.main()
