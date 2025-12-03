"""
GPU Backend Abstraction for Atlas Neural System.

Provides a unified interface that uses CuPy for GPU acceleration
when available, falling back to NumPy for CPU execution.

Usage:
    from .backend import xp, to_cpu, to_gpu, get_backend_info, sync

    # Use xp instead of np for all array operations
    weights = xp.random.randn(100, 100)
    result = xp.dot(weights, inputs)

    # Convert to CPU numpy array when needed (e.g., for I/O)
    cpu_result = to_cpu(result)
"""

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Try to import CuPy for GPU acceleration
try:
    import cupy as cp
    import numpy as np

    # Test that CUDA is actually available
    try:
        cp.cuda.runtime.getDeviceCount()
        HAS_GPU = True
        xp = cp  # Use CuPy as the array backend
        logger.info("CuPy GPU backend initialized successfully")
    except cp.cuda.runtime.CUDARuntimeError:
        HAS_GPU = False
        xp = np
        logger.warning("CuPy installed but CUDA not available, falling back to NumPy")

except ImportError:
    import numpy as np
    HAS_GPU = False
    xp = np  # Fall back to NumPy
    logger.info("CuPy not available, using NumPy CPU backend")


def get_backend_info() -> Dict[str, Any]:
    """
    Get information about the current backend.

    Returns:
        Dictionary with backend information including:
        - backend: 'cupy' or 'numpy'
        - has_gpu: whether GPU is available
        - device_name: GPU name if available
        - device_memory: GPU memory in GB if available
    """
    info = {
        'backend': 'cupy' if HAS_GPU else 'numpy',
        'has_gpu': HAS_GPU,
        'device_name': None,
        'device_memory_gb': None,
        'device_count': 0,
    }

    if HAS_GPU:
        try:
            import cupy as cp
            device = cp.cuda.Device()
            info['device_count'] = cp.cuda.runtime.getDeviceCount()
            info['device_name'] = cp.cuda.runtime.getDeviceProperties(device.id)['name'].decode()
            info['device_memory_gb'] = device.mem_info[1] / (1024**3)
        except Exception as e:
            logger.warning(f"Could not get GPU device info: {e}")

    return info


def to_cpu(array: Any) -> 'np.ndarray':
    """
    Convert array to CPU numpy array.

    Args:
        array: Input array (CuPy or NumPy)

    Returns:
        NumPy array on CPU
    """
    if HAS_GPU and hasattr(array, 'get'):
        # CuPy array - transfer to CPU
        return array.get()
    else:
        # Already a NumPy array or scalar
        import numpy as np
        return np.asarray(array)


def to_gpu(array: Any) -> Any:
    """
    Convert array to GPU if available.

    Args:
        array: Input array (NumPy)

    Returns:
        CuPy array on GPU if available, otherwise original array
    """
    if HAS_GPU:
        import cupy as cp
        if hasattr(array, 'get'):
            # Already a CuPy array
            return array
        else:
            # Convert NumPy to CuPy
            return cp.asarray(array)
    else:
        return array


def sync() -> None:
    """
    Synchronize GPU operations.

    Call this before timing or when you need to ensure
    all GPU operations are complete.
    """
    if HAS_GPU:
        import cupy as cp
        cp.cuda.Stream.null.synchronize()


def get_array_module(array: Any) -> Any:
    """
    Get the array module (numpy or cupy) for the given array.

    Args:
        array: Input array

    Returns:
        numpy or cupy module
    """
    if HAS_GPU:
        import cupy as cp
        return cp.get_array_module(array)
    else:
        import numpy as np
        return np


def set_device(device_id: int = 0) -> None:
    """
    Set the GPU device to use.

    Args:
        device_id: GPU device ID (default 0)
    """
    if HAS_GPU:
        import cupy as cp
        cp.cuda.Device(device_id).use()
        logger.info(f"Using GPU device {device_id}")


def memory_pool_info() -> Optional[Dict[str, float]]:
    """
    Get GPU memory pool information.

    Returns:
        Dictionary with memory info or None if no GPU
    """
    if not HAS_GPU:
        return None

    import cupy as cp
    mempool = cp.get_default_memory_pool()

    return {
        'used_bytes': mempool.used_bytes(),
        'total_bytes': mempool.total_bytes(),
        'used_gb': mempool.used_bytes() / (1024**3),
        'total_gb': mempool.total_bytes() / (1024**3),
    }


def clear_memory_pool() -> None:
    """
    Clear the GPU memory pool to free memory.
    """
    if HAS_GPU:
        import cupy as cp
        mempool = cp.get_default_memory_pool()
        mempool.free_all_blocks()
        logger.debug("GPU memory pool cleared")


# Export the main interface
__all__ = [
    'xp',
    'HAS_GPU',
    'to_cpu',
    'to_gpu',
    'sync',
    'get_backend_info',
    'get_array_module',
    'set_device',
    'memory_pool_info',
    'clear_memory_pool',
]
