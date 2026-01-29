"""
Comprehensive tests for the GPU backend abstraction module.

Tests cover the public API: xp, HAS_GPU, to_cpu, to_gpu, sync,
get_backend_info, get_array_module, memory_pool_info, and clear_memory_pool.

All tests are deterministic and pass reliably in both CPU-only and GPU environments.
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from self_organizing_av_system.core.backend import (
    xp,
    HAS_GPU,
    to_cpu,
    to_gpu,
    sync,
    get_backend_info,
    get_array_module,
    memory_pool_info,
    clear_memory_pool,
)


class TestXpAvailability(unittest.TestCase):
    """Tests that the xp array module is available and usable."""

    def test_xp_is_not_none(self):
        """xp must be a valid module, never None."""
        self.assertIsNotNone(xp)

    def test_xp_has_array_creation_functions(self):
        """xp must expose standard array-creation routines."""
        for attr in ("zeros", "ones", "array", "empty", "arange", "linspace"):
            self.assertTrue(
                hasattr(xp, attr),
                f"xp is missing expected attribute '{attr}'",
            )

    def test_xp_is_numpy_on_cpu(self):
        """When no GPU is present, xp should be the numpy module."""
        if not HAS_GPU:
            self.assertIs(xp, np)


class TestHasGpuFlag(unittest.TestCase):
    """Tests for the HAS_GPU boolean flag."""

    def test_has_gpu_is_bool(self):
        """HAS_GPU must be a Python bool."""
        self.assertIsInstance(HAS_GPU, bool)

    def test_has_gpu_consistent_with_xp(self):
        """HAS_GPU should agree with the type of xp."""
        if HAS_GPU:
            self.assertNotEqual(xp.__name__, "numpy")
        else:
            self.assertEqual(xp.__name__, "numpy")


class TestToCpu(unittest.TestCase):
    """Tests for the to_cpu conversion function."""

    def test_numpy_array_roundtrip(self):
        """A NumPy array passed through to_cpu should remain a NumPy array."""
        arr = np.array([1.0, 2.0, 3.0])
        result = to_cpu(arr)
        self.assertIsInstance(result, np.ndarray)
        np.testing.assert_array_equal(result, arr)

    def test_scalar_conversion(self):
        """A plain Python scalar should be converted to a numpy array."""
        result = to_cpu(42)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.item(), 42)

    def test_list_conversion(self):
        """A plain Python list should be converted to a numpy array."""
        result = to_cpu([1, 2, 3])
        self.assertIsInstance(result, np.ndarray)
        np.testing.assert_array_equal(result, np.array([1, 2, 3]))

    def test_preserves_dtype(self):
        """to_cpu should preserve the dtype of the input array."""
        for dtype in (np.float32, np.float64, np.int32):
            arr = np.array([1, 2, 3], dtype=dtype)
            result = to_cpu(arr)
            self.assertEqual(result.dtype, dtype)

    def test_preserves_shape(self):
        """to_cpu should preserve multi-dimensional shapes."""
        arr = np.zeros((3, 4, 5))
        result = to_cpu(arr)
        self.assertEqual(result.shape, (3, 4, 5))

    def test_empty_array(self):
        """to_cpu should handle empty arrays without error."""
        arr = np.array([])
        result = to_cpu(arr)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.size, 0)

    def test_xp_array_to_cpu(self):
        """An array created via xp should survive to_cpu as a numpy array."""
        arr = xp.array([10.0, 20.0, 30.0])
        result = to_cpu(arr)
        self.assertIsInstance(result, np.ndarray)
        np.testing.assert_array_equal(result, np.array([10.0, 20.0, 30.0]))


class TestToGpu(unittest.TestCase):
    """Tests for the to_gpu transfer function."""

    def test_returns_array(self):
        """to_gpu must return an array-like object."""
        arr = np.array([1.0, 2.0, 3.0])
        result = to_gpu(arr)
        self.assertTrue(hasattr(result, "shape"))

    def test_cpu_fallback_returns_same_object(self):
        """On CPU, to_gpu should return the original numpy array unchanged."""
        if not HAS_GPU:
            arr = np.array([4.0, 5.0, 6.0])
            result = to_gpu(arr)
            self.assertIs(result, arr)

    def test_preserves_values(self):
        """Values must survive the to_gpu -> to_cpu round-trip."""
        arr = np.array([[1, 2], [3, 4]], dtype=np.float64)
        gpu_arr = to_gpu(arr)
        cpu_arr = to_cpu(gpu_arr)
        np.testing.assert_array_equal(cpu_arr, arr)

    def test_preserves_shape_roundtrip(self):
        """Shape must survive the to_gpu -> to_cpu round-trip."""
        arr = np.zeros((2, 3, 4))
        gpu_arr = to_gpu(arr)
        cpu_arr = to_cpu(gpu_arr)
        self.assertEqual(cpu_arr.shape, (2, 3, 4))

    def test_idempotent_on_xp_array(self):
        """Calling to_gpu on an xp array should not error."""
        arr = xp.ones((3,))
        result = to_gpu(to_cpu(arr))
        cpu_result = to_cpu(result)
        np.testing.assert_array_equal(cpu_result, np.ones((3,)))


class TestSync(unittest.TestCase):
    """Tests for the sync function."""

    def test_sync_no_error(self):
        """sync() must complete without raising any exception."""
        try:
            sync()
        except Exception as exc:
            self.fail(f"sync() raised {type(exc).__name__}: {exc}")

    def test_sync_returns_none(self):
        """sync() should return None."""
        result = sync()
        self.assertIsNone(result)

    def test_sync_multiple_calls(self):
        """Calling sync() multiple times in a row must not error."""
        for _ in range(5):
            sync()


class TestGetBackendInfo(unittest.TestCase):
    """Tests for the get_backend_info function."""

    def setUp(self):
        self.info = get_backend_info()

    def test_returns_dict(self):
        """get_backend_info must return a dictionary."""
        self.assertIsInstance(self.info, dict)

    def test_has_backend_key(self):
        """Result must include a 'backend' key."""
        self.assertIn("backend", self.info)

    def test_backend_value(self):
        """'backend' should be either 'numpy' or 'cupy'."""
        self.assertIn(self.info["backend"], ("numpy", "cupy"))

    def test_has_gpu_key(self):
        """Result must include a 'has_gpu' key."""
        self.assertIn("has_gpu", self.info)

    def test_has_gpu_value_is_bool(self):
        """'has_gpu' value must be boolean."""
        self.assertIsInstance(self.info["has_gpu"], bool)

    def test_has_device_name_key(self):
        """Result must include a 'device_name' key."""
        self.assertIn("device_name", self.info)

    def test_has_device_memory_key(self):
        """Result must include a 'device_memory_gb' key."""
        self.assertIn("device_memory_gb", self.info)

    def test_has_device_count_key(self):
        """Result must include a 'device_count' key."""
        self.assertIn("device_count", self.info)

    def test_cpu_backend_values(self):
        """On CPU, device fields should be None / 0."""
        if not HAS_GPU:
            self.assertEqual(self.info["backend"], "numpy")
            self.assertFalse(self.info["has_gpu"])
            self.assertIsNone(self.info["device_name"])
            self.assertIsNone(self.info["device_memory_gb"])
            self.assertEqual(self.info["device_count"], 0)

    def test_consistent_with_has_gpu(self):
        """The info dict's has_gpu must match the module-level HAS_GPU."""
        self.assertEqual(self.info["has_gpu"], HAS_GPU)


class TestGetArrayModule(unittest.TestCase):
    """Tests for the get_array_module function."""

    def test_numpy_array_returns_numpy(self):
        """get_array_module on a numpy array should return numpy."""
        arr = np.array([1, 2, 3])
        mod = get_array_module(arr)
        self.assertIs(mod, np)

    def test_xp_array_returns_module(self):
        """get_array_module on an xp array should return a valid module."""
        arr = xp.array([1, 2, 3])
        mod = get_array_module(arr)
        self.assertTrue(hasattr(mod, "array"))
        self.assertTrue(hasattr(mod, "zeros"))

    def test_returns_numpy_on_cpu(self):
        """On CPU, get_array_module should always return numpy."""
        if not HAS_GPU:
            arr = xp.ones((5,))
            mod = get_array_module(arr)
            self.assertIs(mod, np)


class TestMemoryPoolInfo(unittest.TestCase):
    """Tests for the memory_pool_info function."""

    def test_returns_none_on_cpu(self):
        """On CPU, memory_pool_info must return None."""
        if not HAS_GPU:
            result = memory_pool_info()
            self.assertIsNone(result)

    def test_returns_dict_on_gpu(self):
        """On GPU, memory_pool_info must return a dict with expected keys."""
        if HAS_GPU:
            result = memory_pool_info()
            self.assertIsInstance(result, dict)
            for key in ("used_bytes", "total_bytes", "used_gb", "total_gb"):
                self.assertIn(key, result)

    def test_no_error(self):
        """memory_pool_info must not raise, regardless of backend."""
        try:
            memory_pool_info()
        except Exception as exc:
            self.fail(f"memory_pool_info() raised {type(exc).__name__}: {exc}")


class TestClearMemoryPool(unittest.TestCase):
    """Tests for the clear_memory_pool function."""

    def test_no_error(self):
        """clear_memory_pool must complete without raising."""
        try:
            clear_memory_pool()
        except Exception as exc:
            self.fail(f"clear_memory_pool() raised {type(exc).__name__}: {exc}")

    def test_returns_none(self):
        """clear_memory_pool should return None."""
        result = clear_memory_pool()
        self.assertIsNone(result)

    def test_multiple_calls(self):
        """Repeated calls to clear_memory_pool must not error."""
        for _ in range(3):
            clear_memory_pool()


class TestXpArrayOperations(unittest.TestCase):
    """Tests for basic array operations using the xp backend."""

    # ---- creation -----------------------------------------------------------

    def test_zeros(self):
        """xp.zeros should create an all-zero array."""
        arr = to_cpu(xp.zeros((3, 4)))
        np.testing.assert_array_equal(arr, np.zeros((3, 4)))

    def test_ones(self):
        """xp.ones should create an all-one array."""
        arr = to_cpu(xp.ones((2, 5)))
        np.testing.assert_array_equal(arr, np.ones((2, 5)))

    def test_arange(self):
        """xp.arange should produce a sequential range."""
        arr = to_cpu(xp.arange(10))
        np.testing.assert_array_equal(arr, np.arange(10))

    def test_linspace(self):
        """xp.linspace should produce evenly spaced values."""
        arr = to_cpu(xp.linspace(0, 1, 5))
        np.testing.assert_allclose(arr, np.linspace(0, 1, 5))

    def test_full(self):
        """xp.full should fill an array with a constant."""
        arr = to_cpu(xp.full((2, 3), 7.0))
        np.testing.assert_array_equal(arr, np.full((2, 3), 7.0))

    def test_eye(self):
        """xp.eye should produce an identity matrix."""
        arr = to_cpu(xp.eye(4))
        np.testing.assert_array_equal(arr, np.eye(4))

    # ---- arithmetic ---------------------------------------------------------

    def test_add(self):
        """Element-wise addition should work."""
        a = xp.array([1.0, 2.0, 3.0])
        b = xp.array([4.0, 5.0, 6.0])
        result = to_cpu(a + b)
        np.testing.assert_array_equal(result, np.array([5.0, 7.0, 9.0]))

    def test_subtract(self):
        """Element-wise subtraction should work."""
        a = xp.array([10.0, 20.0, 30.0])
        b = xp.array([1.0, 2.0, 3.0])
        result = to_cpu(a - b)
        np.testing.assert_array_equal(result, np.array([9.0, 18.0, 27.0]))

    def test_multiply(self):
        """Element-wise multiplication should work."""
        a = xp.array([2.0, 3.0, 4.0])
        b = xp.array([5.0, 6.0, 7.0])
        result = to_cpu(a * b)
        np.testing.assert_array_equal(result, np.array([10.0, 18.0, 28.0]))

    def test_scalar_multiply(self):
        """Scalar multiplication should broadcast correctly."""
        a = xp.array([1.0, 2.0, 3.0])
        result = to_cpu(a * 3.0)
        np.testing.assert_array_equal(result, np.array([3.0, 6.0, 9.0]))

    # ---- linear algebra -----------------------------------------------------

    def test_dot_1d(self):
        """xp.dot on 1D vectors should return a scalar."""
        a = xp.array([1.0, 2.0, 3.0])
        b = xp.array([4.0, 5.0, 6.0])
        result = float(to_cpu(xp.dot(a, b)))
        self.assertAlmostEqual(result, 32.0)

    def test_dot_2d(self):
        """xp.dot on 2D arrays should perform matrix multiplication."""
        a = xp.array([[1.0, 2.0], [3.0, 4.0]])
        b = xp.array([[5.0, 6.0], [7.0, 8.0]])
        result = to_cpu(xp.dot(a, b))
        expected = np.array([[19.0, 22.0], [43.0, 50.0]])
        np.testing.assert_array_almost_equal(result, expected)

    def test_matmul_operator(self):
        """The @ operator should perform matrix multiplication."""
        a = xp.eye(3)
        b = xp.array([[1.0], [2.0], [3.0]])
        result = to_cpu(a @ b)
        np.testing.assert_array_equal(result, np.array([[1.0], [2.0], [3.0]]))

    def test_transpose(self):
        """Array transpose should swap axes."""
        a = xp.array([[1, 2, 3], [4, 5, 6]])
        result = to_cpu(a.T)
        expected = np.array([[1, 4], [2, 5], [3, 6]])
        np.testing.assert_array_equal(result, expected)

    # ---- reductions ---------------------------------------------------------

    def test_sum(self):
        """xp.sum should return the correct total."""
        arr = xp.array([1.0, 2.0, 3.0, 4.0])
        result = float(to_cpu(xp.sum(arr)))
        self.assertAlmostEqual(result, 10.0)

    def test_mean(self):
        """xp.mean should return the correct average."""
        arr = xp.array([2.0, 4.0, 6.0, 8.0])
        result = float(to_cpu(xp.mean(arr)))
        self.assertAlmostEqual(result, 5.0)

    def test_max(self):
        """xp.max should return the maximum element."""
        arr = xp.array([3.0, 1.0, 4.0, 1.0, 5.0])
        result = float(to_cpu(xp.max(arr)))
        self.assertAlmostEqual(result, 5.0)

    def test_min(self):
        """xp.min should return the minimum element."""
        arr = xp.array([3.0, 1.0, 4.0, 1.0, 5.0])
        result = float(to_cpu(xp.min(arr)))
        self.assertAlmostEqual(result, 1.0)

    def test_argmax(self):
        """xp.argmax should return the index of the maximum."""
        arr = xp.array([3.0, 1.0, 4.0, 1.0, 5.0])
        result = int(to_cpu(xp.argmax(arr)))
        self.assertEqual(result, 4)

    # ---- shape manipulation -------------------------------------------------

    def test_reshape(self):
        """xp.reshape should change the shape without altering data."""
        arr = xp.arange(12)
        result = to_cpu(xp.reshape(arr, (3, 4)))
        expected = np.arange(12).reshape(3, 4)
        np.testing.assert_array_equal(result, expected)

    def test_concatenate(self):
        """xp.concatenate should join arrays along an axis."""
        a = xp.array([1, 2, 3])
        b = xp.array([4, 5, 6])
        result = to_cpu(xp.concatenate([a, b]))
        np.testing.assert_array_equal(result, np.array([1, 2, 3, 4, 5, 6]))

    # ---- math functions -----------------------------------------------------

    def test_exp(self):
        """xp.exp should compute the exponential."""
        arr = xp.array([0.0, 1.0])
        result = to_cpu(xp.exp(arr))
        np.testing.assert_array_almost_equal(result, np.array([1.0, np.e]))

    def test_sqrt(self):
        """xp.sqrt should compute the square root."""
        arr = xp.array([0.0, 1.0, 4.0, 9.0])
        result = to_cpu(xp.sqrt(arr))
        np.testing.assert_array_almost_equal(result, np.array([0.0, 1.0, 2.0, 3.0]))

    def test_abs(self):
        """xp.abs should compute absolute values."""
        arr = xp.array([-3.0, -1.0, 0.0, 2.0])
        result = to_cpu(xp.abs(arr))
        np.testing.assert_array_equal(result, np.array([3.0, 1.0, 0.0, 2.0]))


if __name__ == "__main__":
    unittest.main()
