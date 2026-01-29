"""
Comprehensive tests for the AudioProcessor class
(self_organizing_av_system/models/audio/processor.py).

Tests cover initialization, audio chunk processing (process_audio_chunk),
waveform processing (process_waveform), buffer management (_update_buffer),
spectrogram computation (_compute_spectrogram, _compute_full_spectrogram),
spectrogram frame processing (_process_spectrogram_frame), temporal features
(_compute_temporal_features), receptive field access (get_receptive_fields,
visualize_receptive_field), learning, get_pathway_state, get_stats, and repr.
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
from self_organizing_av_system.models.audio.processor import AudioProcessor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _seed():
    """Set a fixed random seed for deterministic tests."""
    np.random.seed(42)
    xp.random.seed(42)


def _make_audio_chunk(length=512, seed=0):
    """Create a deterministic synthetic mono audio chunk in [-1, 1]."""
    rng = np.random.RandomState(seed)
    return rng.uniform(-1.0, 1.0, size=length).astype(np.float32)


def _sine_wave(freq, n_samples, sr):
    """Pure sine wave at *freq* Hz."""
    t = np.arange(n_samples, dtype=np.float64) / sr
    return np.sin(2.0 * np.pi * freq * t)


def _white_noise(n_samples, amplitude=0.1, seed=123):
    """Deterministic white noise."""
    rng = np.random.RandomState(seed)
    return (amplitude * rng.randn(n_samples)).astype(np.float64)


def _make_waveform(duration_sec=0.1, sample_rate=22050, freq=440.0, seed=0):
    """Create a deterministic synthetic mono waveform (sine + noise)."""
    rng = np.random.RandomState(seed)
    n_samples = int(duration_sec * sample_rate)
    t = np.linspace(0, duration_sec, n_samples, endpoint=False)
    sine = 0.5 * np.sin(2 * np.pi * freq * t)
    noise = 0.05 * rng.randn(n_samples)
    return (sine + noise).astype(np.float32)


# ---------------------------------------------------------------------------
# Small-processor factory (keeps every test fast)
# ---------------------------------------------------------------------------

# Shared small/fast defaults used by most tests.
_DEFAULTS = dict(
    sample_rate=8000,
    window_size=256,
    hop_length=128,
    n_mels=16,
    min_freq=50,
    max_freq=3500,
    normalize=True,
    layer_sizes=[32, 16],
)


def _small_processor(**overrides):
    """Return an AudioProcessor configured with small sizes for fast testing."""
    params = {**_DEFAULTS, **overrides}
    return AudioProcessor(**params)


# ===================================================================
# 1.  Initialization
# ===================================================================

class TestAudioProcessorInitialization(unittest.TestCase):
    """Test AudioProcessor construction and initial state."""

    def setUp(self):
        _seed()

    # -- scalar attributes ---------------------------------------------------

    def test_default_sample_rate(self):
        ap = _small_processor()
        self.assertEqual(ap.sample_rate, 8000)

    def test_custom_sample_rate(self):
        ap = _small_processor(sample_rate=16000)
        self.assertEqual(ap.sample_rate, 16000)

    def test_default_window_size(self):
        ap = _small_processor()
        self.assertEqual(ap.window_size, 256)

    def test_default_hop_length(self):
        ap = _small_processor()
        self.assertEqual(ap.hop_length, 128)

    def test_default_n_mels(self):
        ap = _small_processor()
        self.assertEqual(ap.n_mels, 16)

    def test_frequency_range(self):
        ap = _small_processor()
        self.assertEqual(ap.min_freq, 50)
        self.assertEqual(ap.max_freq, 3500)

    def test_normalize_flag_true(self):
        ap = _small_processor(normalize=True)
        self.assertTrue(ap.normalize)

    def test_normalize_flag_false(self):
        ap = _small_processor(normalize=False)
        self.assertFalse(ap.normalize)

    def test_all_defaults_from_class(self):
        """Using AudioProcessor() with no args should use class defaults."""
        ap = AudioProcessor()
        self.assertEqual(ap.sample_rate, 22050)
        self.assertEqual(ap.window_size, 1024)
        self.assertEqual(ap.hop_length, 512)
        self.assertEqual(ap.n_mels, 64)
        self.assertEqual(ap.min_freq, 50)
        self.assertEqual(ap.max_freq, 8000)
        self.assertTrue(ap.normalize)

    # -- mel filterbank ------------------------------------------------------

    def test_mel_filterbank_shape(self):
        ap = _small_processor(n_mels=16, window_size=256)
        expected_cols = 256 // 2 + 1
        self.assertEqual(ap.mel_fb.shape, (16, expected_cols))

    def test_mel_filterbank_non_negative(self):
        ap = _small_processor()
        self.assertTrue(np.all(ap.mel_fb >= 0))

    def test_mel_filterbank_rows_have_energy(self):
        """Every mel filter should have at least one non-zero coefficient."""
        ap = _small_processor()
        for i in range(ap.n_mels):
            self.assertGreater(np.sum(ap.mel_fb[i]), 0,
                               f"Mel filter {i} is entirely zero")

    # -- audio buffer --------------------------------------------------------

    def test_audio_buffer_shape(self):
        ap = _small_processor(window_size=256)
        self.assertEqual(ap.audio_buffer.shape, (512,))

    def test_audio_buffer_initialized_to_zeros(self):
        ap = _small_processor(window_size=256)
        np.testing.assert_array_equal(ap.audio_buffer, np.zeros(512))

    # -- tracking state ------------------------------------------------------

    def test_initial_frame_count(self):
        ap = _small_processor()
        self.assertEqual(ap.frame_count, 0)

    def test_initial_spectrogram_is_none(self):
        ap = _small_processor()
        self.assertIsNone(ap.current_spectrogram)

    def test_initial_current_frame_is_none(self):
        ap = _small_processor()
        self.assertIsNone(ap.current_frame)

    def test_initial_frame_energy_is_none(self):
        ap = _small_processor()
        self.assertIsNone(ap.frame_energy)

    def test_initial_temporal_features_is_none(self):
        ap = _small_processor()
        self.assertIsNone(ap.temporal_features)

    # -- neural pathway ------------------------------------------------------

    def test_default_layer_sizes_when_none(self):
        """When no layer_sizes provided, defaults are [150, 75, 40]."""
        _seed()
        ap = AudioProcessor(
            sample_rate=8000, window_size=256, hop_length=128,
            n_mels=16, min_freq=50, max_freq=3500,
        )
        sizes = [layer.layer_size for layer in ap.audio_pathway.layers]
        self.assertEqual(sizes, [150, 75, 40])

    def test_custom_layer_sizes(self):
        ap = _small_processor(layer_sizes=[20, 10, 5])
        sizes = [layer.layer_size for layer in ap.audio_pathway.layers]
        self.assertEqual(sizes, [20, 10, 5])

    def test_pathway_input_size_matches_n_mels(self):
        ap = _small_processor(n_mels=16)
        self.assertEqual(ap.audio_pathway.input_size, 16)

    def test_pathway_name_is_audio(self):
        ap = _small_processor()
        self.assertEqual(ap.audio_pathway.name, "audio")

    def test_pathway_use_recurrent_false(self):
        """Audio pathway is created with use_recurrent=False, so no recurrent weights."""
        ap = _small_processor()
        for layer in ap.audio_pathway.layers:
            self.assertIsNone(layer.recurrent_weights)

    # -- repr ----------------------------------------------------------------

    def test_repr_contains_class_name(self):
        ap = _small_processor()
        self.assertIn("AudioProcessor", repr(ap))

    def test_repr_contains_n_mels(self):
        ap = _small_processor(n_mels=16)
        self.assertIn("n_mels=16", repr(ap))

    def test_repr_contains_freq_range(self):
        ap = _small_processor(min_freq=50, max_freq=3500)
        r = repr(ap)
        self.assertIn("50-3500Hz", r)


# ===================================================================
# 2.  _update_buffer()
# ===================================================================

class TestAudioProcessorUpdateBuffer(unittest.TestCase):
    """Test _update_buffer() method."""

    def setUp(self):
        _seed()
        self.ap = _small_processor(window_size=256)  # buffer size = 512

    def test_buffer_updated_with_chunk(self):
        chunk = np.ones(100, dtype=np.float32)
        self.ap._update_buffer(chunk)
        np.testing.assert_array_almost_equal(
            self.ap.audio_buffer[-100:], np.ones(100)
        )

    def test_buffer_shifts_old_data(self):
        """Inserting two chunks should shift the buffer left."""
        chunk1 = np.ones(100, dtype=np.float32) * 1.0
        self.ap._update_buffer(chunk1)
        chunk2 = np.ones(100, dtype=np.float32) * 2.0
        self.ap._update_buffer(chunk2)
        np.testing.assert_array_almost_equal(
            self.ap.audio_buffer[-100:], np.full(100, 2.0)
        )
        np.testing.assert_array_almost_equal(
            self.ap.audio_buffer[-200:-100], np.full(100, 1.0)
        )

    def test_chunk_larger_than_buffer(self):
        """If chunk is larger than buffer, only first buffer_size samples used."""
        big_chunk = np.arange(1000, dtype=np.float32)
        self.ap._update_buffer(big_chunk)
        np.testing.assert_array_almost_equal(
            self.ap.audio_buffer, big_chunk[:512]
        )

    def test_small_chunk(self):
        """A very small chunk should only fill the end of the buffer."""
        small_chunk = np.array([5.0, 6.0, 7.0], dtype=np.float32)
        self.ap._update_buffer(small_chunk)
        np.testing.assert_array_almost_equal(
            self.ap.audio_buffer[-3:], np.array([5.0, 6.0, 7.0])
        )

    def test_buffer_length_preserved(self):
        """Buffer length should stay constant regardless of chunk size."""
        for size in [10, 100, 256, 512, 1024]:
            chunk = np.ones(size, dtype=np.float32)
            self.ap._update_buffer(chunk)
            self.assertEqual(self.ap.audio_buffer.shape[0], 512)


# ===================================================================
# 3.  _compute_spectrogram() (single frame)
# ===================================================================

class TestAudioProcessorComputeSpectrogram(unittest.TestCase):
    """Test _compute_spectrogram() for a single audio buffer."""

    def setUp(self):
        _seed()

    def test_output_shape(self):
        ap = _small_processor(n_mels=16, window_size=256)
        audio_data = np.random.randn(512).astype(np.float32)
        result = ap._compute_spectrogram(audio_data)
        self.assertEqual(result.shape, (16,))

    def test_output_is_finite(self):
        ap = _small_processor(n_mels=16, window_size=256)
        audio_data = np.random.randn(512).astype(np.float32)
        result = ap._compute_spectrogram(audio_data)
        self.assertTrue(np.all(np.isfinite(result)))

    def test_normalized_output_range(self):
        """With normalize=True, output should be in [0, 1]."""
        ap = _small_processor(n_mels=16, window_size=256, normalize=True)
        audio_data = np.random.randn(512).astype(np.float32)
        result = ap._compute_spectrogram(audio_data)
        self.assertGreaterEqual(float(np.min(result)), -1e-6)
        self.assertLessEqual(float(np.max(result)), 1.0 + 1e-6)

    def test_normalized_min_near_zero(self):
        ap = _small_processor(n_mels=16, window_size=256, normalize=True)
        audio_data = np.random.randn(512).astype(np.float32)
        result = ap._compute_spectrogram(audio_data)
        self.assertAlmostEqual(float(np.min(result)), 0.0, places=4)

    def test_normalized_max_near_one(self):
        ap = _small_processor(n_mels=16, window_size=256, normalize=True)
        audio_data = np.random.randn(512).astype(np.float32)
        result = ap._compute_spectrogram(audio_data)
        self.assertAlmostEqual(float(np.max(result)), 1.0, places=4)

    def test_unnormalized_output_returns_db_values(self):
        """With normalize=False, output is in dB scale (should be <= 0)."""
        ap = _small_processor(n_mels=16, window_size=256, normalize=False)
        audio_data = np.random.randn(512).astype(np.float32)
        result = ap._compute_spectrogram(audio_data)
        self.assertTrue(np.all(result <= 0.0 + 1e-6))

    def test_different_inputs_produce_different_outputs(self):
        ap = _small_processor(n_mels=16, window_size=256)
        audio1 = np.random.RandomState(1).randn(512).astype(np.float32)
        audio2 = np.random.RandomState(2).randn(512).astype(np.float32)
        result1 = ap._compute_spectrogram(audio1)
        result2 = ap._compute_spectrogram(audio2)
        self.assertFalse(np.allclose(result1, result2))

    def test_silent_input(self):
        """Zero audio should produce a valid (finite) spectrogram frame."""
        ap = _small_processor(n_mels=16, window_size=256)
        silent = np.zeros(512, dtype=np.float32)
        result = ap._compute_spectrogram(silent)
        self.assertTrue(np.all(np.isfinite(result)))

    def test_deterministic(self):
        ap = _small_processor(n_mels=16, window_size=256)
        buf = _sine_wave(440, 512, ap.sample_rate)
        spec1 = ap._compute_spectrogram(buf)
        spec2 = ap._compute_spectrogram(buf)
        np.testing.assert_array_equal(spec1, spec2)

    def test_sine_has_energy(self):
        ap = _small_processor(n_mels=16, window_size=256)
        buf = _sine_wave(440, 512, ap.sample_rate)
        spec = ap._compute_spectrogram(buf)
        self.assertGreater(np.sum(spec), 0.0)


# ===================================================================
# 4.  _compute_full_spectrogram() (entire waveform)
# ===================================================================

class TestAudioProcessorComputeFullSpectrogram(unittest.TestCase):
    """Test _compute_full_spectrogram() for an entire audio waveform."""

    def setUp(self):
        _seed()

    def test_output_shape_rows(self):
        ap = _small_processor(n_mels=16, window_size=256, hop_length=128)
        waveform = _make_waveform(duration_sec=0.1, sample_rate=8000)
        result = ap._compute_full_spectrogram(waveform)
        self.assertEqual(result.shape[0], 16)

    def test_output_has_multiple_frames(self):
        ap = _small_processor(n_mels=16, window_size=256, hop_length=128)
        waveform = _make_waveform(duration_sec=0.1, sample_rate=8000)
        result = ap._compute_full_spectrogram(waveform)
        self.assertGreater(result.shape[1], 1)

    def test_output_is_finite(self):
        ap = _small_processor(n_mels=16, window_size=256, hop_length=128)
        waveform = _make_waveform(duration_sec=0.1, sample_rate=8000)
        result = ap._compute_full_spectrogram(waveform)
        self.assertTrue(np.all(np.isfinite(result)))

    def test_normalized_output_range(self):
        ap = _small_processor(n_mels=16, normalize=True)
        waveform = _make_waveform(duration_sec=0.1, sample_rate=8000)
        result = ap._compute_full_spectrogram(waveform)
        self.assertGreaterEqual(float(np.min(result)), -1e-6)
        self.assertLessEqual(float(np.max(result)), 1.0 + 1e-6)

    def test_unnormalized_output_all_non_positive(self):
        ap = _small_processor(n_mels=16, normalize=False)
        waveform = _make_waveform(duration_sec=0.1, sample_rate=8000)
        result = ap._compute_full_spectrogram(waveform)
        self.assertTrue(np.all(result <= 0.0 + 1e-6))

    def test_longer_waveform_more_frames(self):
        ap = _small_processor(n_mels=16, window_size=256, hop_length=128)
        short_wf = _make_waveform(duration_sec=0.05, sample_rate=8000)
        long_wf = _make_waveform(duration_sec=0.2, sample_rate=8000)
        result_short = ap._compute_full_spectrogram(short_wf)
        result_long = ap._compute_full_spectrogram(long_wf)
        self.assertGreater(result_long.shape[1], result_short.shape[1])

    def test_deterministic(self):
        ap = _small_processor(n_mels=16, window_size=256, hop_length=128)
        waveform = _make_waveform(duration_sec=0.1, sample_rate=8000)
        s1 = ap._compute_full_spectrogram(waveform)
        s2 = ap._compute_full_spectrogram(waveform)
        np.testing.assert_array_equal(s1, s2)


# ===================================================================
# 5.  _process_spectrogram_frame()
# ===================================================================

class TestAudioProcessorProcessSpectrogramFrame(unittest.TestCase):
    """Test _process_spectrogram_frame() method."""

    def setUp(self):
        _seed()
        self.ap = _small_processor(n_mels=16, layer_sizes=[32, 16])

    def test_returns_ndarray(self):
        spec_frame = np.random.rand(16).astype(np.float32)
        result = self.ap._process_spectrogram_frame(spec_frame)
        self.assertIsInstance(result, np.ndarray)

    def test_output_shape_matches_last_layer(self):
        spec_frame = np.random.rand(16).astype(np.float32)
        result = self.ap._process_spectrogram_frame(spec_frame)
        self.assertEqual(result.shape, (16,))

    def test_current_frame_stored(self):
        spec_frame = np.random.rand(16).astype(np.float32)
        self.ap._process_spectrogram_frame(spec_frame)
        np.testing.assert_array_equal(self.ap.current_frame, spec_frame)

    def test_frame_energy_computed(self):
        spec_frame = np.ones(16, dtype=np.float32) * 0.5
        self.ap._process_spectrogram_frame(spec_frame)
        self.assertAlmostEqual(float(self.ap.frame_energy), 8.0, places=4)

    def test_frame_energy_zero_for_zeros(self):
        spec_frame = np.zeros(16, dtype=np.float32)
        self.ap._process_spectrogram_frame(spec_frame)
        self.assertAlmostEqual(float(self.ap.frame_energy), 0.0, places=4)

    def test_raises_on_wrong_shape(self):
        wrong_frame = np.random.rand(32).astype(np.float32)
        with self.assertRaises(ValueError):
            self.ap._process_spectrogram_frame(wrong_frame)

    def test_with_time_step(self):
        spec_frame = np.random.rand(16).astype(np.float32)
        result = self.ap._process_spectrogram_frame(spec_frame, time_step=42)
        self.assertEqual(result.shape, (16,))

    def test_output_is_finite(self):
        spec_frame = np.random.rand(16).astype(np.float32)
        result = self.ap._process_spectrogram_frame(spec_frame)
        self.assertTrue(np.all(np.isfinite(result)))

    def test_output_non_negative(self):
        """Activations after competition should be non-negative."""
        spec_frame = np.random.rand(16).astype(np.float32)
        result = self.ap._process_spectrogram_frame(spec_frame)
        self.assertTrue(np.all(result >= 0))


# ===================================================================
# 6.  process_audio_chunk() -- main chunk processing method
# ===================================================================

class TestAudioProcessorProcessAudioChunk(unittest.TestCase):
    """Test process_audio_chunk() with synthetic audio chunks."""

    def setUp(self):
        _seed()
        self.ap = _small_processor(n_mels=16, window_size=256, layer_sizes=[32, 16])

    def test_returns_ndarray(self):
        chunk = _make_audio_chunk(length=256, seed=1)
        result = self.ap.process_audio_chunk(chunk)
        self.assertIsInstance(result, np.ndarray)

    def test_output_shape_matches_last_layer(self):
        chunk = _make_audio_chunk(length=256, seed=1)
        result = self.ap.process_audio_chunk(chunk)
        self.assertEqual(result.shape, (16,))

    def test_frame_count_increments(self):
        chunk = _make_audio_chunk(length=256, seed=1)
        self.assertEqual(self.ap.frame_count, 0)
        self.ap.process_audio_chunk(chunk)
        self.assertEqual(self.ap.frame_count, 1)
        self.ap.process_audio_chunk(chunk)
        self.assertEqual(self.ap.frame_count, 2)

    def test_current_frame_updated(self):
        chunk = _make_audio_chunk(length=256, seed=1)
        self.ap.process_audio_chunk(chunk)
        self.assertIsNotNone(self.ap.current_frame)
        self.assertEqual(self.ap.current_frame.shape, (16,))

    def test_frame_energy_updated(self):
        chunk = _make_audio_chunk(length=256, seed=1)
        self.ap.process_audio_chunk(chunk)
        self.assertIsNotNone(self.ap.frame_energy)

    def test_with_time_step(self):
        chunk = _make_audio_chunk(length=256, seed=1)
        result = self.ap.process_audio_chunk(chunk, time_step=10)
        self.assertEqual(result.shape, (16,))

    def test_output_is_finite(self):
        chunk = _make_audio_chunk(length=256, seed=1)
        result = self.ap.process_audio_chunk(chunk)
        self.assertTrue(np.all(np.isfinite(result)))

    def test_output_non_negative(self):
        chunk = _make_audio_chunk(length=256, seed=1)
        result = self.ap.process_audio_chunk(chunk)
        self.assertTrue(np.all(result >= 0))

    def test_process_multiple_chunks_sequentially(self):
        for i in range(5):
            chunk = _make_audio_chunk(length=256, seed=i)
            result = self.ap.process_audio_chunk(chunk, time_step=i * 10)
            self.assertEqual(result.shape, (16,))
        self.assertEqual(self.ap.frame_count, 5)

    def test_small_chunk(self):
        """A chunk smaller than window size should still work."""
        chunk = _make_audio_chunk(length=64, seed=1)
        result = self.ap.process_audio_chunk(chunk)
        self.assertEqual(result.shape, (16,))

    def test_deterministic_output(self):
        """Identical seeds and inputs must give identical outputs."""
        chunk = _make_audio_chunk(length=256, seed=99)

        _seed()
        ap1 = _small_processor(n_mels=16, window_size=256, layer_sizes=[32, 16])
        result1 = ap1.process_audio_chunk(chunk)

        _seed()
        ap2 = _small_processor(n_mels=16, window_size=256, layer_sizes=[32, 16])
        result2 = ap2.process_audio_chunk(chunk)

        np.testing.assert_array_almost_equal(result1, result2)

    def test_process_chunk_with_noise(self):
        noise = _white_noise(1024)
        result = self.ap.process_audio_chunk(noise)
        self.assertEqual(result.shape[0],
                         self.ap.audio_pathway.layers[-1].layer_size)

    def test_process_chunk_with_sine(self):
        chunk = _sine_wave(440, 1024, self.ap.sample_rate)
        result = self.ap.process_audio_chunk(chunk)
        self.assertEqual(result.shape[0],
                         self.ap.audio_pathway.layers[-1].layer_size)


# ===================================================================
# 7.  process_waveform() -- full waveform processing
# ===================================================================

class TestAudioProcessorProcessWaveform(unittest.TestCase):
    """Test process_waveform() with synthetic waveforms."""

    def setUp(self):
        _seed()
        self.ap = _small_processor(
            n_mels=16, window_size=256, hop_length=128,
            layer_sizes=[32, 16]
        )

    def test_returns_list(self):
        waveform = _make_waveform(duration_sec=0.1, sample_rate=8000)
        result = self.ap.process_waveform(waveform)
        self.assertIsInstance(result, list)

    def test_list_elements_are_ndarrays(self):
        waveform = _make_waveform(duration_sec=0.1, sample_rate=8000)
        result = self.ap.process_waveform(waveform)
        for elem in result:
            self.assertIsInstance(elem, np.ndarray)

    def test_each_element_shape_matches_last_layer(self):
        waveform = _make_waveform(duration_sec=0.1, sample_rate=8000)
        result = self.ap.process_waveform(waveform)
        for elem in result:
            self.assertEqual(elem.shape, (16,))

    def test_frame_count_increments_by_num_frames(self):
        waveform = _make_waveform(duration_sec=0.1, sample_rate=8000)
        result = self.ap.process_waveform(waveform)
        self.assertEqual(self.ap.frame_count, len(result))

    def test_current_spectrogram_stored(self):
        waveform = _make_waveform(duration_sec=0.1, sample_rate=8000)
        self.ap.process_waveform(waveform)
        self.assertIsNotNone(self.ap.current_spectrogram)
        self.assertEqual(self.ap.current_spectrogram.shape[0], 16)

    def test_with_time_step(self):
        waveform = _make_waveform(duration_sec=0.1, sample_rate=8000)
        result = self.ap.process_waveform(waveform, time_step=100)
        self.assertGreater(len(result), 0)

    def test_all_activations_finite(self):
        waveform = _make_waveform(duration_sec=0.1, sample_rate=8000)
        result = self.ap.process_waveform(waveform)
        for elem in result:
            self.assertTrue(np.all(np.isfinite(elem)))

    def test_all_activations_non_negative(self):
        waveform = _make_waveform(duration_sec=0.1, sample_rate=8000)
        result = self.ap.process_waveform(waveform)
        for elem in result:
            self.assertTrue(np.all(elem >= 0))

    def test_longer_waveform_more_frames(self):
        short_wf = _make_waveform(duration_sec=0.05, sample_rate=8000)
        result_short = self.ap.process_waveform(short_wf)

        _seed()
        ap2 = _small_processor(
            n_mels=16, window_size=256, hop_length=128,
            layer_sizes=[32, 16]
        )
        long_wf = _make_waveform(duration_sec=0.2, sample_rate=8000)
        result_long = ap2.process_waveform(long_wf)
        self.assertGreater(len(result_long), len(result_short))

    def test_non_empty_result(self):
        waveform = _make_waveform(duration_sec=0.1, sample_rate=8000)
        result = self.ap.process_waveform(waveform)
        self.assertGreater(len(result), 0)


# ===================================================================
# 8.  _compute_temporal_features()
# ===================================================================

class TestAudioProcessorTemporalFeatures(unittest.TestCase):
    """Test _compute_temporal_features() method."""

    def setUp(self):
        _seed()
        self.ap = _small_processor(n_mels=16)

    def test_output_shape(self):
        spec_history = np.random.rand(16, 10).astype(np.float32)
        result = self.ap._compute_temporal_features(spec_history)
        self.assertEqual(result.shape, (16,))

    def test_constant_input_zero_features(self):
        """Constant input across time should give zero temporal features."""
        spec_history = np.ones((16, 10), dtype=np.float32) * 0.5
        result = self.ap._compute_temporal_features(spec_history)
        np.testing.assert_allclose(result, 0.0, atol=1e-12)

    def test_varying_input_nonzero_features(self):
        spec_history = np.random.rand(16, 10).astype(np.float32)
        result = self.ap._compute_temporal_features(spec_history)
        self.assertTrue(np.any(result > 0))

    def test_output_non_negative(self):
        """Temporal features (mean of abs diff) should be non-negative."""
        spec_history = np.random.rand(16, 10).astype(np.float32)
        result = self.ap._compute_temporal_features(spec_history)
        self.assertTrue(np.all(result >= -1e-10))

    def test_output_is_finite(self):
        spec_history = np.random.rand(16, 10).astype(np.float32)
        result = self.ap._compute_temporal_features(spec_history)
        self.assertTrue(np.all(np.isfinite(result)))

    def test_two_frame_history(self):
        """Temporal features with just 2 frames should still work."""
        spec_history = np.random.rand(16, 2).astype(np.float32)
        result = self.ap._compute_temporal_features(spec_history)
        self.assertEqual(result.shape, (16,))

    def test_step_change_produces_expected_features(self):
        """A step change from 0 to 1 at a single time."""
        spec_history = np.zeros((16, 5), dtype=np.float32)
        spec_history[:, 2:] = 1.0
        result = self.ap._compute_temporal_features(spec_history)
        # diff has 4 columns, only one is nonzero (column 2-1=1 has diff=1)
        expected = np.full(16, 0.25)
        np.testing.assert_array_almost_equal(result, expected, decimal=5)

    def test_onset_produces_large_flux(self):
        """A sudden onset should yield higher flux than a steady signal."""
        n = self.ap.n_mels
        steady = np.ones((n, 10)) * 0.5
        onset = np.zeros((n, 10))
        onset[:, 5:] = 1.0
        feats_steady = self.ap._compute_temporal_features(steady)
        feats_onset = self.ap._compute_temporal_features(onset)
        self.assertGreater(np.sum(feats_onset), np.sum(feats_steady))


# ===================================================================
# 9.  get_receptive_fields()
# ===================================================================

class TestAudioProcessorReceptiveFields(unittest.TestCase):
    """Test get_receptive_fields() method."""

    def setUp(self):
        _seed()
        self.ap = _small_processor(n_mels=16, layer_sizes=[32, 16])

    def test_layer_0_receptive_fields_shape(self):
        fields = self.ap.get_receptive_fields(layer_idx=0)
        self.assertEqual(fields.shape, (32, 16))

    def test_layer_1_receptive_fields_shape(self):
        fields = self.ap.get_receptive_fields(layer_idx=1)
        self.assertEqual(fields.shape, (16, 32))

    def test_receptive_fields_are_finite(self):
        fields = self.ap.get_receptive_fields(layer_idx=0)
        self.assertTrue(np.all(np.isfinite(fields)))

    def test_receptive_fields_not_all_zero(self):
        """After initialization, weights should not all be zero."""
        fields = self.ap.get_receptive_fields(layer_idx=0)
        self.assertFalse(np.allclose(fields, 0))

    def test_default_layer_idx(self):
        """Default layer_idx=0 should work."""
        fields = self.ap.get_receptive_fields()
        self.assertEqual(fields.shape, (32, 16))


# ===================================================================
# 10. visualize_receptive_field()
# ===================================================================

class TestAudioProcessorVisualizeReceptiveField(unittest.TestCase):
    """Test visualize_receptive_field() method."""

    def setUp(self):
        _seed()
        self.ap = _small_processor(n_mels=16, layer_sizes=[32, 16])

    def test_layer_0_returns_1d(self):
        rf = self.ap.visualize_receptive_field(layer_idx=0, neuron_idx=0)
        self.assertEqual(len(rf.shape), 1)

    def test_layer_0_shape_matches_input(self):
        rf = self.ap.visualize_receptive_field(layer_idx=0, neuron_idx=0)
        self.assertEqual(rf.shape[0], 16)

    def test_layer_0_finite(self):
        rf = self.ap.visualize_receptive_field(layer_idx=0, neuron_idx=0)
        self.assertTrue(np.all(np.isfinite(rf)))

    def test_higher_layer_returns_1d(self):
        rf = self.ap.visualize_receptive_field(layer_idx=1, neuron_idx=0)
        self.assertEqual(len(rf.shape), 1)

    def test_higher_layer_shape_matches_n_mels(self):
        """Higher layer visualization should be in input feature space (n_mels)."""
        rf = self.ap.visualize_receptive_field(layer_idx=1, neuron_idx=0)
        self.assertEqual(rf.shape[0], 16)

    def test_higher_layer_normalized_range(self):
        """Higher layer RF should be normalized to [0, 1]."""
        rf = self.ap.visualize_receptive_field(layer_idx=1, neuron_idx=0)
        self.assertGreaterEqual(float(np.min(rf)), -1e-6)
        self.assertLessEqual(float(np.max(rf)), 1.0 + 1e-6)

    def test_invalid_negative_layer_index_raises(self):
        with self.assertRaises(ValueError):
            self.ap.visualize_receptive_field(layer_idx=-1, neuron_idx=0)

    def test_out_of_bounds_layer_index_raises(self):
        with self.assertRaises(ValueError):
            self.ap.visualize_receptive_field(layer_idx=5, neuron_idx=0)

    def test_invalid_negative_neuron_index_raises(self):
        with self.assertRaises(ValueError):
            self.ap.visualize_receptive_field(layer_idx=0, neuron_idx=-1)

    def test_out_of_bounds_neuron_index_raises(self):
        with self.assertRaises(ValueError):
            self.ap.visualize_receptive_field(layer_idx=0, neuron_idx=100)

    def test_different_neurons_may_differ(self):
        """Different neurons should generally have different receptive fields."""
        rf0 = self.ap.visualize_receptive_field(layer_idx=0, neuron_idx=0)
        rf1 = self.ap.visualize_receptive_field(layer_idx=0, neuron_idx=1)
        self.assertFalse(np.allclose(rf0, rf1))


# ===================================================================
# 11. learn()
# ===================================================================

class TestAudioProcessorLearn(unittest.TestCase):
    """Test learn() method."""

    def setUp(self):
        _seed()
        self.ap = _small_processor(n_mels=16, window_size=256, layer_sizes=[32, 16])

    def _snapshot_weights(self):
        return [n.weights.copy()
                for n in self.ap.audio_pathway.layers[0].neurons]

    def test_learn_does_not_error_without_input(self):
        """learn() before any processing should be a safe no-op."""
        self.ap.learn('oja')  # should not raise

    def test_learn_without_processing_is_noop(self):
        before = self._snapshot_weights()
        self.ap.learn('oja')
        after = self._snapshot_weights()
        for b, a in zip(before, after):
            np.testing.assert_array_equal(b, a)

    def test_learn_oja_after_processing(self):
        chunk = _make_audio_chunk(length=256, seed=1)
        self.ap.process_audio_chunk(chunk)
        self.ap.learn('oja')  # should not raise

    def test_learn_hebbian(self):
        chunk = _make_audio_chunk(length=256, seed=1)
        self.ap.process_audio_chunk(chunk)
        self.ap.learn('hebbian')  # should not raise

    def test_learn_stdp(self):
        chunk = _make_audio_chunk(length=256, seed=1)
        self.ap.process_audio_chunk(chunk)
        self.ap.learn('stdp')  # should not raise

    def test_learn_oja_modifies_weights(self):
        """After processing and learning, weights should change."""
        chunk = _sine_wave(440, 1024, self.ap.sample_rate)
        before = self._snapshot_weights()
        self.ap.process_audio_chunk(chunk)
        self.ap.learn('oja')
        after = self._snapshot_weights()
        any_changed = any(
            not np.allclose(b, a, atol=1e-12) for b, a in zip(before, after))
        self.assertTrue(any_changed,
                        "Oja learning should modify at least one weight")

    def test_learn_hebbian_modifies_weights(self):
        chunk = _sine_wave(440, 1024, self.ap.sample_rate)
        before = self._snapshot_weights()
        self.ap.process_audio_chunk(chunk)
        self.ap.learn('hebbian')
        after = self._snapshot_weights()
        any_changed = any(
            not np.allclose(b, a, atol=1e-12) for b, a in zip(before, after))
        self.assertTrue(any_changed,
                        "Hebbian learning should modify at least one weight")


# ===================================================================
# 12. get_pathway_state()
# ===================================================================

class TestAudioProcessorPathwayState(unittest.TestCase):
    """Test get_pathway_state() method."""

    def setUp(self):
        _seed()
        self.ap = _small_processor(n_mels=16, layer_sizes=[32, 16])

    def test_returns_dict(self):
        state = self.ap.get_pathway_state()
        self.assertIsInstance(state, dict)

    def test_contains_name(self):
        state = self.ap.get_pathway_state()
        self.assertIn('name', state)
        self.assertEqual(state['name'], 'audio')

    def test_contains_num_layers(self):
        state = self.ap.get_pathway_state()
        self.assertIn('num_layers', state)
        self.assertEqual(state['num_layers'], 2)

    def test_contains_layers_list(self):
        state = self.ap.get_pathway_state()
        self.assertIn('layers', state)
        self.assertIsInstance(state['layers'], list)
        self.assertEqual(len(state['layers']), 2)

    def test_layer_state_has_expected_keys(self):
        state = self.ap.get_pathway_state()
        for layer_state in state['layers']:
            self.assertIn('name', layer_state)
            self.assertIn('input_size', layer_state)
            self.assertIn('layer_size', layer_state)

    def test_state_after_processing(self):
        chunk = _make_audio_chunk(length=256, seed=1)
        self.ap.process_audio_chunk(chunk)
        state = self.ap.get_pathway_state()
        self.assertEqual(state['name'], 'audio')


# ===================================================================
# 13. get_stats()
# ===================================================================

class TestAudioProcessorStats(unittest.TestCase):
    """Test get_stats() method."""

    def setUp(self):
        _seed()
        self.ap = _small_processor(
            n_mels=16, window_size=256,
            min_freq=50, max_freq=3500,
            layer_sizes=[32, 16]
        )

    def test_returns_dict(self):
        stats = self.ap.get_stats()
        self.assertIsInstance(stats, dict)

    def test_expected_keys(self):
        stats = self.ap.get_stats()
        for key in ("frames_processed", "sample_rate", "n_mels",
                     "window_size", "freq_range", "layer_sizes"):
            self.assertIn(key, stats)

    def test_frames_processed_initially_zero(self):
        stats = self.ap.get_stats()
        self.assertEqual(stats["frames_processed"], 0)

    def test_frames_processed_tracks_count(self):
        chunk = _make_audio_chunk(length=256, seed=1)
        self.ap.process_audio_chunk(chunk)
        self.ap.process_audio_chunk(chunk)
        stats = self.ap.get_stats()
        self.assertEqual(stats["frames_processed"], 2)

    def test_sample_rate_in_stats(self):
        stats = self.ap.get_stats()
        self.assertEqual(stats["sample_rate"], 8000)

    def test_n_mels_in_stats(self):
        stats = self.ap.get_stats()
        self.assertEqual(stats["n_mels"], 16)

    def test_window_size_in_stats(self):
        stats = self.ap.get_stats()
        self.assertEqual(stats["window_size"], 256)

    def test_freq_range_in_stats(self):
        stats = self.ap.get_stats()
        self.assertEqual(stats["freq_range"], "50-3500Hz")

    def test_layer_sizes_in_stats(self):
        stats = self.ap.get_stats()
        self.assertEqual(stats["layer_sizes"], [32, 16])


# ===================================================================
# 14. Frequency analysis verification
# ===================================================================

class TestFrequencyAnalysis(unittest.TestCase):
    """Verify that audio frequency content is correctly captured."""

    def setUp(self):
        _seed()
        self.ap = _small_processor()
        self.sr = self.ap.sample_rate

    def test_mel_spectrum_low_freq_activates_lower_bands(self):
        """A 200 Hz tone should put more energy in the lower mel bands."""
        buf = _sine_wave(200, self.ap.window_size * 2, self.sr)
        spec = self.ap._compute_spectrogram(buf)
        mid = self.ap.n_mels // 2
        self.assertGreater(np.sum(spec[:mid]), np.sum(spec[mid:]))

    def test_mel_spectrum_high_freq_activates_upper_bands(self):
        """A 3000 Hz tone should put more energy in the upper mel bands."""
        buf = _sine_wave(3000, self.ap.window_size * 2, self.sr)
        spec = self.ap._compute_spectrogram(buf)
        mid = self.ap.n_mels // 2
        self.assertGreater(np.sum(spec[mid:]), np.sum(spec[:mid]))

    def test_distinct_frequencies_produce_distinct_spectra(self):
        buf_a = _sine_wave(300, self.ap.window_size * 2, self.sr)
        buf_b = _sine_wave(2500, self.ap.window_size * 2, self.sr)
        spec_a = self.ap._compute_spectrogram(buf_a)
        spec_b = self.ap._compute_spectrogram(buf_b)
        self.assertGreater(np.linalg.norm(spec_a - spec_b), 0.1)

    def test_mel_filterbank_covers_frequency_range(self):
        """At least one filter should respond inside the configured range."""
        fb = self.ap.mel_fb
        fft_freqs = np.linspace(0, self.sr / 2.0, fb.shape[1])
        in_range = (fft_freqs >= self.ap.min_freq) & (fft_freqs <= self.ap.max_freq)
        self.assertGreater(fb[:, in_range].sum(), 0)


# ===================================================================
# 15. End-to-end integration tests
# ===================================================================

class TestAudioProcessorIntegration(unittest.TestCase):
    """End-to-end integration tests combining multiple operations."""

    def setUp(self):
        _seed()

    def test_chunk_then_learn_cycle(self):
        """Process multiple chunks with interleaved learning."""
        ap = _small_processor(n_mels=16, window_size=256, layer_sizes=[32, 16])
        for i in range(3):
            chunk = _make_audio_chunk(length=256, seed=i)
            result = ap.process_audio_chunk(chunk, time_step=i)
            ap.learn('oja')
            self.assertEqual(result.shape, (16,))
        self.assertEqual(ap.frame_count, 3)

    def test_waveform_then_learn(self):
        """Process a full waveform then learn."""
        ap = _small_processor(n_mels=16, window_size=256, hop_length=128, layer_sizes=[32, 16])
        waveform = _make_waveform(duration_sec=0.1, sample_rate=8000)
        activations = ap.process_waveform(waveform)
        ap.learn('oja')
        self.assertGreater(len(activations), 0)
        self.assertGreater(ap.frame_count, 0)

    def test_stats_after_waveform_processing(self):
        ap = _small_processor(n_mels=16, window_size=256, hop_length=128, layer_sizes=[32, 16])
        waveform = _make_waveform(duration_sec=0.1, sample_rate=8000)
        activations = ap.process_waveform(waveform)
        stats = ap.get_stats()
        self.assertEqual(stats["frames_processed"], len(activations))

    def test_pathway_state_after_processing(self):
        ap = _small_processor(n_mels=16, window_size=256, layer_sizes=[32, 16])
        chunk = _make_audio_chunk(length=256, seed=1)
        ap.process_audio_chunk(chunk)
        state = ap.get_pathway_state()
        self.assertEqual(state["name"], "audio")
        self.assertEqual(state["num_layers"], 2)

    def test_receptive_fields_after_learning(self):
        """Receptive fields should still be valid after learning."""
        ap = _small_processor(n_mels=16, window_size=256, layer_sizes=[32, 16])
        chunk = _make_audio_chunk(length=256, seed=1)
        ap.process_audio_chunk(chunk)
        ap.learn('oja')
        fields = ap.get_receptive_fields(layer_idx=0)
        self.assertEqual(fields.shape, (32, 16))
        self.assertTrue(np.all(np.isfinite(fields)))

    def test_single_layer_processor(self):
        """AudioProcessor with a single layer should work."""
        ap = _small_processor(n_mels=16, window_size=256, layer_sizes=[10])
        chunk = _make_audio_chunk(length=256, seed=1)
        result = ap.process_audio_chunk(chunk)
        self.assertEqual(result.shape, (10,))

    def test_three_layer_processor(self):
        """AudioProcessor with three layers should work."""
        ap = _small_processor(n_mels=16, window_size=256, layer_sizes=[32, 20, 10])
        chunk = _make_audio_chunk(length=256, seed=1)
        result = ap.process_audio_chunk(chunk)
        self.assertEqual(result.shape, (10,))

    def test_full_pipeline_chunk_stats_state(self):
        """Complete pipeline: init, process, learn, stats, state."""
        ap = _small_processor(n_mels=16, window_size=256, layer_sizes=[32, 16])
        chunk = _sine_wave(440, 512, ap.sample_rate)
        result = ap.process_audio_chunk(chunk)
        ap.learn('oja')
        stats = ap.get_stats()
        state = ap.get_pathway_state()
        self.assertEqual(result.shape, (16,))
        self.assertEqual(stats["frames_processed"], 1)
        self.assertEqual(state["name"], "audio")
        self.assertTrue(np.all(np.isfinite(result)))


if __name__ == '__main__':
    unittest.main()
