"""
Comprehensive tests for the VisualProcessor class
(self_organizing_av_system/models/visual/processor.py).

Tests cover initialization, frame processing (process_frame),
patch extraction (_extract_patches), motion detection
(_compute_motion_features), contrast normalization, and
grayscale/color conversion handling.
"""
import sys
import os
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# The processor module uses a bare `from core.pathway import NeuralPathway`,
# so ensure self_organizing_av_system is also on sys.path.
sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), '..', 'self_organizing_av_system'),
)

import numpy as np
from self_organizing_av_system.core.backend import xp
from self_organizing_av_system.models.visual.processor import VisualProcessor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _seed():
    """Set a fixed random seed for deterministic tests."""
    np.random.seed(42)
    xp.random.seed(42)


def _make_gray_image(height=32, width=32, seed=0):
    """Create a deterministic synthetic grayscale image (uint8, 2-D)."""
    rng = np.random.RandomState(seed)
    return rng.randint(0, 256, size=(height, width), dtype=np.uint8)


def _make_color_image(height=32, width=32, seed=0):
    """Create a deterministic synthetic RGB image (uint8, H x W x 3)."""
    rng = np.random.RandomState(seed)
    return rng.randint(0, 256, size=(height, width, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Small-processor factory (keeps every test fast)
# ---------------------------------------------------------------------------

def _small_processor(use_grayscale=True, contrast_normalize=True,
                     layer_sizes=None, patch_size=4, stride=4):
    """Return a VisualProcessor configured for 16x16 input with tiny layers."""
    if layer_sizes is None:
        layer_sizes = [20, 10]
    return VisualProcessor(
        input_width=16,
        input_height=16,
        use_grayscale=use_grayscale,
        patch_size=patch_size,
        stride=stride,
        contrast_normalize=contrast_normalize,
        layer_sizes=layer_sizes,
    )


# ===================================================================
# 1.  Initialization
# ===================================================================

class TestVisualProcessorInitialization(unittest.TestCase):
    """Test VisualProcessor construction and initial state."""

    def setUp(self):
        _seed()

    def test_default_parameters(self):
        vp = VisualProcessor()
        self.assertEqual(vp.input_width, 64)
        self.assertEqual(vp.input_height, 64)
        self.assertTrue(vp.use_grayscale)
        self.assertEqual(vp.patch_size, 8)
        self.assertEqual(vp.stride, 4)
        self.assertTrue(vp.contrast_normalize)
        self.assertEqual(vp.num_channels, 1)

    def test_custom_parameters(self):
        vp = VisualProcessor(
            input_width=16, input_height=16,
            use_grayscale=False, patch_size=4,
            stride=2, contrast_normalize=False,
            layer_sizes=[20, 10],
        )
        self.assertEqual(vp.input_width, 16)
        self.assertEqual(vp.input_height, 16)
        self.assertFalse(vp.use_grayscale)
        self.assertEqual(vp.patch_size, 4)
        self.assertEqual(vp.stride, 2)
        self.assertFalse(vp.contrast_normalize)
        self.assertEqual(vp.num_channels, 3)

    def test_derived_parameters_grayscale(self):
        vp = _small_processor(use_grayscale=True, patch_size=4, stride=2)
        # patches_w = (16 - 4) // 2 + 1 = 7
        self.assertEqual(vp.patches_w, 7)
        self.assertEqual(vp.patches_h, 7)
        self.assertEqual(vp.num_patches, 49)
        # patch_dim = 4 * 4 * 1 = 16
        self.assertEqual(vp.patch_dim, 16)

    def test_derived_parameters_color(self):
        vp = _small_processor(use_grayscale=False, patch_size=4, stride=2)
        self.assertEqual(vp.num_channels, 3)
        # patch_dim = 4 * 4 * 3 = 48
        self.assertEqual(vp.patch_dim, 48)

    def test_patch_coordinates_precomputed(self):
        vp = _small_processor(patch_size=4, stride=2)
        self.assertEqual(len(vp.patch_coords), vp.num_patches)
        # First coordinate must be (0, 0)
        self.assertEqual(vp.patch_coords[0], (0, 0))
        # Second coordinate must be (0, stride)
        self.assertEqual(vp.patch_coords[1], (0, 2))

    def test_default_layer_sizes(self):
        vp = VisualProcessor(input_width=16, input_height=16)
        layer_sizes = [l.layer_size for l in vp.visual_pathway.layers]
        self.assertEqual(layer_sizes, [200, 100, 50])

    def test_custom_layer_sizes(self):
        vp = _small_processor(layer_sizes=[20, 10])
        layer_sizes = [l.layer_size for l in vp.visual_pathway.layers]
        self.assertEqual(layer_sizes, [20, 10])

    def test_initial_state(self):
        vp = _small_processor()
        self.assertEqual(vp.frame_count, 0)
        self.assertIsNone(vp.current_frame)
        self.assertIsNone(vp.current_patches)
        self.assertIsNone(vp.previous_frame)
        self.assertIsNone(vp.motion_features)

    def test_pathway_input_size_equals_patch_dim(self):
        vp = _small_processor(use_grayscale=True)
        self.assertEqual(vp.visual_pathway.input_size, vp.patch_dim)

    def test_pathway_name_is_visual(self):
        vp = _small_processor()
        self.assertEqual(vp.visual_pathway.name, "visual")

    def test_repr_contains_class_and_dimensions(self):
        vp = _small_processor()
        r = repr(vp)
        self.assertIn("VisualProcessor", r)
        self.assertIn("16x16", r)


# ===================================================================
# 2.  process_frame() -- the main process method
# ===================================================================

class TestVisualProcessorProcessFrame(unittest.TestCase):
    """Test process_frame() with synthetic image data."""

    def setUp(self):
        _seed()
        self.vp = _small_processor()

    def test_returns_ndarray(self):
        frame = _make_color_image(32, 32, seed=1)
        result = self.vp.process_frame(frame)
        self.assertIsInstance(result, np.ndarray)

    def test_output_shape_matches_last_layer(self):
        frame = _make_color_image(32, 32, seed=1)
        result = self.vp.process_frame(frame)
        # The last layer has 10 neurons
        self.assertEqual(result.shape, (10,))

    def test_frame_count_increments(self):
        frame = _make_color_image(32, 32, seed=1)
        self.assertEqual(self.vp.frame_count, 0)
        self.vp.process_frame(frame)
        self.assertEqual(self.vp.frame_count, 1)
        self.vp.process_frame(frame)
        self.assertEqual(self.vp.frame_count, 2)

    def test_current_frame_stored(self):
        frame = _make_color_image(32, 32, seed=1)
        self.vp.process_frame(frame)
        self.assertIsNotNone(self.vp.current_frame)
        self.assertEqual(self.vp.current_frame.shape, (16, 16))

    def test_current_patches_stored(self):
        frame = _make_color_image(32, 32, seed=1)
        self.vp.process_frame(frame)
        self.assertIsNotNone(self.vp.current_patches)
        self.assertEqual(len(self.vp.current_patches), self.vp.num_patches)

    def test_previous_frame_is_none_after_first(self):
        frame = _make_color_image(32, 32, seed=1)
        self.vp.process_frame(frame)
        # current_frame was None before the first call, so previous_frame
        # remains None after the first frame.
        self.assertIsNone(self.vp.previous_frame)

    def test_previous_frame_set_after_second(self):
        self.vp.process_frame(_make_color_image(32, 32, seed=1))
        self.vp.process_frame(_make_color_image(32, 32, seed=2))
        self.assertIsNotNone(self.vp.previous_frame)

    def test_motion_features_none_after_first(self):
        self.vp.process_frame(_make_color_image(32, 32, seed=1))
        self.assertIsNone(self.vp.motion_features)

    def test_motion_features_computed_after_second(self):
        self.vp.process_frame(_make_color_image(32, 32, seed=1))
        self.vp.process_frame(_make_color_image(32, 32, seed=2))
        self.assertIsNotNone(self.vp.motion_features)

    def test_output_values_are_finite(self):
        frame = _make_color_image(32, 32, seed=1)
        result = self.vp.process_frame(frame)
        self.assertTrue(np.all(np.isfinite(result)))

    def test_output_non_negative(self):
        frame = _make_color_image(32, 32, seed=1)
        result = self.vp.process_frame(frame)
        self.assertTrue(np.all(result >= 0))

    def test_process_with_time_step(self):
        frame = _make_color_image(32, 32, seed=1)
        result = self.vp.process_frame(frame, time_step=42)
        self.assertEqual(result.shape, (10,))

    def test_process_grayscale_input_directly(self):
        """A 2-D (already gray) frame should be handled by a grayscale processor."""
        gray_frame = _make_gray_image(32, 32, seed=1)
        result = self.vp.process_frame(gray_frame)
        self.assertEqual(result.shape, (10,))

    def test_process_multiple_frames_sequentially(self):
        for i in range(5):
            result = self.vp.process_frame(
                _make_color_image(32, 32, seed=i), time_step=i * 10,
            )
            self.assertEqual(result.shape, (10,))
        self.assertEqual(self.vp.frame_count, 5)

    def test_deterministic_output(self):
        """Identical seeds and inputs must give identical outputs."""
        frame = _make_color_image(32, 32, seed=99)

        _seed()
        vp1 = _small_processor()
        result1 = vp1.process_frame(frame)

        _seed()
        vp2 = _small_processor()
        result2 = vp2.process_frame(frame)

        np.testing.assert_array_almost_equal(result1, result2)


# ===================================================================
# 3.  _extract_patches()
# ===================================================================

class TestVisualProcessorExtractPatches(unittest.TestCase):
    """Test _extract_patches() method."""

    def setUp(self):
        _seed()

    # -- grayscale ---------------------------------------------------

    def test_grayscale_patch_count(self):
        vp = _small_processor(use_grayscale=True, patch_size=4, stride=4)
        frame = np.random.rand(16, 16).astype(np.float32)
        patches = vp._extract_patches(frame)
        # (16-4)//4+1 = 4 per axis -> 4*4 = 16 patches
        self.assertEqual(len(patches), 16)

    def test_grayscale_patch_dimension(self):
        vp = _small_processor(use_grayscale=True, patch_size=4, stride=4)
        frame = np.random.rand(16, 16).astype(np.float32)
        patches = vp._extract_patches(frame)
        for patch in patches:
            # 4*4*1 = 16
            self.assertEqual(patch.shape, (16,))

    # -- color -------------------------------------------------------

    def test_color_patch_count(self):
        vp = _small_processor(use_grayscale=False, patch_size=4, stride=4)
        frame = np.random.rand(16, 16, 3).astype(np.float32)
        patches = vp._extract_patches(frame)
        self.assertEqual(len(patches), 16)

    def test_color_patch_dimension(self):
        vp = _small_processor(use_grayscale=False, patch_size=4, stride=4)
        frame = np.random.rand(16, 16, 3).astype(np.float32)
        patches = vp._extract_patches(frame)
        for patch in patches:
            # 4*4*3 = 48
            self.assertEqual(patch.shape, (48,))

    # -- overlapping patches -----------------------------------------

    def test_overlapping_patches_count(self):
        vp = _small_processor(use_grayscale=True, patch_size=4, stride=2)
        frame = np.random.rand(16, 16).astype(np.float32)
        patches = vp._extract_patches(frame)
        # (16-4)//2+1 = 7 per axis -> 7*7 = 49
        self.assertEqual(len(patches), 49)

    # -- value correctness ------------------------------------------

    def test_patch_values_match_source_region(self):
        """Extracted patch values must correspond to the correct frame region."""
        vp = _small_processor(use_grayscale=True, patch_size=4, stride=4)
        frame = np.arange(256, dtype=np.float32).reshape(16, 16)
        patches = vp._extract_patches(frame)

        # First patch: rows 0-3, cols 0-3
        expected_first = frame[0:4, 0:4].flatten()
        np.testing.assert_array_equal(patches[0], expected_first)

        # Second patch: rows 0-3, cols 4-7
        expected_second = frame[0:4, 4:8].flatten()
        np.testing.assert_array_equal(patches[1], expected_second)

    def test_color_patch_values_match_source_region(self):
        vp = _small_processor(use_grayscale=False, patch_size=4, stride=4)
        frame = np.arange(16 * 16 * 3, dtype=np.float32).reshape(16, 16, 3)
        patches = vp._extract_patches(frame)

        expected_first = frame[0:4, 0:4, :].flatten()
        np.testing.assert_array_equal(patches[0], expected_first)

    def test_all_patches_are_flat_arrays(self):
        vp = _small_processor(use_grayscale=True, patch_size=4, stride=4)
        frame = np.random.rand(16, 16).astype(np.float32)
        patches = vp._extract_patches(frame)
        for patch in patches:
            self.assertEqual(len(patch.shape), 1)


# ===================================================================
# 4.  _compute_motion_features()  (detect_motion)
# ===================================================================

class TestVisualProcessorMotionDetection(unittest.TestCase):
    """Test _compute_motion_features() method."""

    def setUp(self):
        _seed()
        self.vp = _small_processor()

    def test_identical_frames_produce_zero_motion(self):
        frame = np.random.rand(16, 16).astype(np.float32)
        motion = self.vp._compute_motion_features(frame, frame)
        np.testing.assert_array_almost_equal(motion, np.zeros((16, 16)))

    def test_different_frames_produce_nonzero_motion(self):
        frame1 = np.zeros((16, 16), dtype=np.float32)
        frame2 = np.ones((16, 16), dtype=np.float32)
        motion = self.vp._compute_motion_features(frame1, frame2)
        self.assertTrue(np.all(motion > 0))

    def test_motion_is_non_negative(self):
        frame1 = np.random.rand(16, 16).astype(np.float32)
        frame2 = np.random.rand(16, 16).astype(np.float32)
        motion = self.vp._compute_motion_features(frame1, frame2)
        self.assertTrue(np.all(motion >= 0))

    def test_motion_shape_grayscale(self):
        frame1 = np.random.rand(16, 16).astype(np.float32)
        frame2 = np.random.rand(16, 16).astype(np.float32)
        motion = self.vp._compute_motion_features(frame1, frame2)
        self.assertEqual(motion.shape, (16, 16))

    def test_motion_shape_color(self):
        """Color frame diff is reduced via max across channels -> 2-D output."""
        frame1 = np.random.rand(16, 16, 3).astype(np.float32)
        frame2 = np.random.rand(16, 16, 3).astype(np.float32)
        motion = self.vp._compute_motion_features(frame1, frame2)
        self.assertEqual(motion.shape, (16, 16))

    def test_motion_equals_absolute_difference(self):
        frame1 = np.full((16, 16), 0.3, dtype=np.float32)
        frame2 = np.full((16, 16), 0.7, dtype=np.float32)
        motion = self.vp._compute_motion_features(frame1, frame2)
        np.testing.assert_array_almost_equal(
            motion, np.full((16, 16), 0.4), decimal=5,
        )

    def test_motion_symmetric(self):
        """abs(a - b) == abs(b - a), so motion should be the same regardless of order."""
        frame1 = np.random.rand(16, 16).astype(np.float32)
        frame2 = np.random.rand(16, 16).astype(np.float32)
        motion_ab = self.vp._compute_motion_features(frame1, frame2)
        motion_ba = self.vp._compute_motion_features(frame2, frame1)
        np.testing.assert_array_almost_equal(motion_ab, motion_ba)

    def test_motion_computed_during_process_frame(self):
        """After two process_frame calls the motion_features attribute is set."""
        self.vp.process_frame(_make_color_image(32, 32, seed=1))
        self.assertIsNone(self.vp.motion_features)
        self.vp.process_frame(_make_color_image(32, 32, seed=2))
        self.assertIsNotNone(self.vp.motion_features)
        self.assertEqual(self.vp.motion_features.shape, (16, 16))


# ===================================================================
# 5.  Contrast normalization
# ===================================================================

class TestVisualProcessorContrastNormalization(unittest.TestCase):
    """Test contrast normalization inside _preprocess_frame()."""

    def setUp(self):
        _seed()

    def test_output_range_with_normalization(self):
        vp = _small_processor(contrast_normalize=True)
        frame = _make_color_image(32, 32, seed=1)
        processed = vp._preprocess_frame(frame)
        # After contrast normalization and rescaling, values fall in [0, 1]
        self.assertGreaterEqual(float(np.min(processed)), -1e-6)
        self.assertLessEqual(float(np.max(processed)), 1.0 + 1e-6)

    def test_output_range_without_normalization(self):
        vp = _small_processor(contrast_normalize=False)
        frame = _make_color_image(32, 32, seed=1)
        processed = vp._preprocess_frame(frame)
        # Simple /255 -> [0, 1]
        self.assertGreaterEqual(float(np.min(processed)), 0.0)
        self.assertLessEqual(float(np.max(processed)), 1.0)

    def test_uniform_image_handled_gracefully(self):
        """Uniform image (std=0): normalization branch is skipped."""
        vp = _small_processor(contrast_normalize=True)
        uniform_frame = np.full((32, 32, 3), 128, dtype=np.uint8)
        processed = vp._preprocess_frame(uniform_frame)
        # std == 0 so the normalization `if std > 0` is skipped;
        # result equals pixel_value / 255.0
        expected_val = 128.0 / 255.0
        np.testing.assert_array_almost_equal(
            processed, np.full((16, 16), expected_val), decimal=4,
        )

    def test_normalization_changes_non_uniform_image(self):
        """A non-uniform image should differ with and without normalization."""
        vp_norm = _small_processor(contrast_normalize=True)
        vp_no = _small_processor(contrast_normalize=False)
        frame = _make_color_image(32, 32, seed=5)
        processed_norm = vp_norm._preprocess_frame(frame)
        processed_no = vp_no._preprocess_frame(frame)
        self.assertFalse(np.allclose(processed_norm, processed_no))

    def test_output_is_floating_point(self):
        vp = _small_processor(contrast_normalize=True)
        frame = _make_color_image(32, 32, seed=1)
        processed = vp._preprocess_frame(frame)
        self.assertTrue(np.issubdtype(processed.dtype, np.floating))

    def test_normalized_min_is_zero(self):
        """After contrast normalization the minimum value should be ~0."""
        vp = _small_processor(contrast_normalize=True)
        frame = _make_color_image(32, 32, seed=3)
        processed = vp._preprocess_frame(frame)
        self.assertAlmostEqual(float(np.min(processed)), 0.0, places=5)

    def test_normalized_max_is_one(self):
        """After contrast normalization the maximum value should be ~1."""
        vp = _small_processor(contrast_normalize=True)
        frame = _make_color_image(32, 32, seed=3)
        processed = vp._preprocess_frame(frame)
        self.assertAlmostEqual(float(np.max(processed)), 1.0, places=5)


# ===================================================================
# 6.  Grayscale / color conversion handling
# ===================================================================

class TestVisualProcessorGrayscaleColorHandling(unittest.TestCase):
    """Test grayscale and color conversion paths."""

    def setUp(self):
        _seed()

    def test_rgb_converted_to_grayscale(self):
        vp = _small_processor(use_grayscale=True, contrast_normalize=False)
        color_frame = _make_color_image(32, 32, seed=1)
        processed = vp._preprocess_frame(color_frame)
        self.assertEqual(len(processed.shape), 2)
        self.assertEqual(processed.shape, (16, 16))

    def test_already_grayscale_input_unchanged_dims(self):
        vp = _small_processor(use_grayscale=True, contrast_normalize=False)
        gray_frame = _make_gray_image(32, 32, seed=1)
        processed = vp._preprocess_frame(gray_frame)
        self.assertEqual(len(processed.shape), 2)
        self.assertEqual(processed.shape, (16, 16))

    def test_color_mode_preserves_channels(self):
        vp = _small_processor(use_grayscale=False, contrast_normalize=False)
        color_frame = _make_color_image(32, 32, seed=1)
        processed = vp._preprocess_frame(color_frame)
        self.assertEqual(len(processed.shape), 3)
        self.assertEqual(processed.shape, (16, 16, 3))

    def test_color_mode_values_in_unit_range(self):
        vp = _small_processor(use_grayscale=False, contrast_normalize=False)
        color_frame = _make_color_image(32, 32, seed=1)
        processed = vp._preprocess_frame(color_frame)
        self.assertGreaterEqual(float(np.min(processed)), 0.0)
        self.assertLessEqual(float(np.max(processed)), 1.0)

    def test_grayscale_num_channels_attribute(self):
        vp = _small_processor(use_grayscale=True)
        self.assertEqual(vp.num_channels, 1)

    def test_color_num_channels_attribute(self):
        vp = _small_processor(use_grayscale=False)
        self.assertEqual(vp.num_channels, 3)

    def test_full_pipeline_grayscale(self):
        """End-to-end: color image through grayscale pipeline."""
        vp = _small_processor(use_grayscale=True)
        result = vp.process_frame(_make_color_image(32, 32, seed=1))
        self.assertEqual(result.shape, (10,))
        self.assertTrue(np.all(np.isfinite(result)))

    def test_full_pipeline_color(self):
        """End-to-end: color image through color pipeline."""
        vp = _small_processor(use_grayscale=False)
        result = vp.process_frame(_make_color_image(32, 32, seed=1))
        self.assertEqual(result.shape, (10,))
        self.assertTrue(np.all(np.isfinite(result)))

    def test_resizing_to_target_dimensions(self):
        """Input of arbitrary size must be resized to input_width x input_height."""
        vp = _small_processor(use_grayscale=True, contrast_normalize=False)
        large_frame = _make_color_image(200, 300, seed=7)
        processed = vp._preprocess_frame(large_frame)
        self.assertEqual(processed.shape, (16, 16))

    def test_resizing_color_to_target_dimensions(self):
        vp = _small_processor(use_grayscale=False, contrast_normalize=False)
        large_frame = _make_color_image(200, 300, seed=7)
        processed = vp._preprocess_frame(large_frame)
        self.assertEqual(processed.shape, (16, 16, 3))


# ===================================================================
# 7.  Stats and auxiliary methods
# ===================================================================

class TestVisualProcessorStats(unittest.TestCase):
    """Test get_stats(), get_pathway_state(), and learn()."""

    def setUp(self):
        _seed()
        self.vp = _small_processor()

    def test_get_stats_returns_dict(self):
        stats = self.vp.get_stats()
        self.assertIsInstance(stats, dict)

    def test_get_stats_expected_keys(self):
        stats = self.vp.get_stats()
        for key in ("frames_processed", "input_dimensions", "num_patches",
                     "patch_size", "layer_sizes", "use_grayscale"):
            self.assertIn(key, stats)

    def test_frames_processed_tracks_count(self):
        self.vp.process_frame(_make_color_image(32, 32, seed=1))
        self.vp.process_frame(_make_color_image(32, 32, seed=2))
        stats = self.vp.get_stats()
        self.assertEqual(stats["frames_processed"], 2)

    def test_input_dimensions_string(self):
        stats = self.vp.get_stats()
        self.assertEqual(stats["input_dimensions"], "16x16")

    def test_layer_sizes_in_stats(self):
        stats = self.vp.get_stats()
        self.assertEqual(stats["layer_sizes"], [20, 10])

    def test_get_pathway_state(self):
        state = self.vp.get_pathway_state()
        self.assertIsInstance(state, dict)
        self.assertIn("name", state)
        self.assertEqual(state["name"], "visual")

    def test_learn_does_not_error_without_input(self):
        """learn() before any process_frame should be a safe no-op."""
        self.vp.learn('oja')  # should not raise

    def test_learn_after_processing(self):
        """learn() after processing a frame should not raise."""
        self.vp.process_frame(_make_color_image(32, 32, seed=1))
        self.vp.learn('oja')  # should not raise


if __name__ == '__main__':
    unittest.main()
