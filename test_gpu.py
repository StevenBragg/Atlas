"""Verify GPU is being used for Atlas computations."""
import sys
sys.path.insert(0, '.')

from self_organizing_av_system.core.backend import (
    xp, HAS_GPU, get_backend_info, memory_pool_info, to_cpu
)

print("=" * 60)
print("Atlas GPU Diagnostic")
print("=" * 60)

# Backend info
info = get_backend_info()
print(f"\nBackend: {info['backend'].upper()}")
print(f"GPU Available: {info['has_gpu']}")
if info['has_gpu']:
    print(f"GPU Device: {info['device_name']}")
    print(f"GPU Memory: {info['device_memory_gb']:.1f} GB")

# Test array operations
print("\n--- Testing Array Operations ---")
test_array = xp.random.randn(1000, 1000).astype(xp.float32)
print(f"Test array type: {type(test_array).__module__}.{type(test_array).__name__}")
print(f"Array on GPU: {hasattr(test_array, 'device')}")

# Matrix multiplication
result = test_array @ test_array.T
print(f"MatMul result type: {type(result).__module__}.{type(result).__name__}")

# Memory usage
if HAS_GPU:
    mem = memory_pool_info()
    print(f"\nGPU Memory Used: {mem['used_gb']:.3f} GB")
    print(f"GPU Memory Pool: {mem['total_gb']:.3f} GB")

# Test learning engine uses GPU
print("\n--- Testing Learning Engine ---")
from self_organizing_av_system.core.learning_engine import LearningEngine
from self_organizing_av_system.core.curriculum_system import CurriculumSystem, CurriculumLevel

engine = LearningEngine()
curriculum = CurriculumSystem()

# Check network weights are on GPU
network = engine.network
print(f"Network layers: {len(network.layers)}")
for i, layer in enumerate(network.layers):
    weights = layer.weights
    print(f"  Layer {i} weights type: {type(weights).__module__}.{type(weights).__name__}")
    if hasattr(weights, 'device'):
        print(f"  Layer {i} on GPU: True (device {weights.device})")

# Run a quick learning test and check memory
print("\n--- Running Learning Test ---")
level_info = curriculum.CURRICULUM[CurriculumLevel.LEVEL_1_BASIC]
challenge = curriculum.create_challenge_object(level_info.challenges[0])

# Check training data is moved to GPU
samples, labels = challenge.training_data.get_batch(32)
from self_organizing_av_system.core.backend import to_gpu
gpu_samples = to_gpu(samples)
print(f"Training batch type: {type(gpu_samples).__module__}.{type(gpu_samples).__name__}")

if HAS_GPU:
    mem = memory_pool_info()
    print(f"\nGPU Memory After Setup: {mem['used_gb']:.3f} GB")

print("\n" + "=" * 60)
if HAS_GPU:
    print("SUCCESS: GPU acceleration is active!")
else:
    print("WARNING: Running on CPU only")
print("=" * 60)
