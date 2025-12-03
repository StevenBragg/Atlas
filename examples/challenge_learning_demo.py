"""
ATLAS Challenge-Based Learning Demonstration

This example demonstrates how to use the ChallengeLearner to teach
ATLAS new tasks through challenges.

Key features demonstrated:
1. Learning from natural language descriptions
2. Learning from structured data
3. Multiple modalities (vision, audio, time series, etc.)
4. Automatic strategy selection (Hebbian, STDP, BCM, etc.)
5. Progress tracking and curriculum learning

All learning uses biology-inspired local plasticity rules - NO backpropagation!
"""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from self_organizing_av_system.core import (
    ChallengeLearner,
    Challenge,
    ChallengeType,
    Modality,
    TrainingData,
    SuccessCriteria,
    learn_challenge,
)


def print_header(title: str) -> None:
    """Print a section header."""
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def demonstrate_natural_language_learning():
    """Demonstrate learning from natural language descriptions."""
    print_header("NATURAL LANGUAGE CHALLENGE LEARNING")

    learner = ChallengeLearner(state_dim=64, verbose=True)

    # Example 1: Pattern recognition challenge
    print("\n1. Pattern Recognition Challenge")
    print("-" * 40)
    result = learner.learn(
        "Learn to classify images of handwritten digits with 80% accuracy"
    )
    print(f"   Result: {'Success' if result.success else 'Failed'}")
    print(f"   Strategy used: {result.strategy_used}")

    # Example 2: Prediction challenge
    print("\n2. Prediction Challenge")
    print("-" * 40)
    result = learner.learn(
        "Learn to predict stock prices from historical time series data"
    )
    print(f"   Result: {'Success' if result.success else 'Failed'}")
    print(f"   Strategy used: {result.strategy_used}")

    # Example 3: Anomaly detection challenge
    print("\n3. Anomaly Detection Challenge")
    print("-" * 40)
    result = learner.learn(
        "Detect unusual patterns in sensor readings"
    )
    print(f"   Result: {'Success' if result.success else 'Failed'}")
    print(f"   Strategy used: {result.strategy_used}")

    return learner


def demonstrate_structured_data_learning():
    """Demonstrate learning from structured data."""
    print_header("STRUCTURED DATA CHALLENGE LEARNING")

    learner = ChallengeLearner(state_dim=64, verbose=True)

    # Generate synthetic classification data
    print("\n1. Classification from Synthetic Data")
    print("-" * 40)

    n_samples = 200
    n_features = 32
    n_classes = 5

    # Create clustered data (simulates real classification task)
    data = []
    labels = []
    for i in range(n_samples):
        class_id = i % n_classes
        # Each class has a different mean
        mean = np.zeros(n_features)
        mean[class_id * 6:(class_id + 1) * 6] = 1.0
        sample = mean + np.random.randn(n_features) * 0.3
        data.append(sample)
        labels.append(class_id)

    result = learner.learn_from_data(
        data=data,
        labels=labels,
        success_criteria={"accuracy": 0.7, "max_iterations": 100},
        name="synthetic_classification",
        description="Classify synthetic clusters",
    )

    print(f"   Final accuracy: {result.accuracy:.2%}")
    print(f"   Iterations: {result.iterations}")

    # Generate time series prediction data
    print("\n2. Time Series Prediction")
    print("-" * 40)

    # Create sine wave with noise
    t = np.linspace(0, 4 * np.pi, 300)
    signal = np.sin(t) + 0.2 * np.random.randn(len(t))

    # Create input-output pairs (predict next value from window)
    window_size = 10
    ts_data = []
    ts_labels = []
    for i in range(len(signal) - window_size - 1):
        ts_data.append(signal[i:i + window_size])
        ts_labels.append(signal[i + window_size])

    result = learner.learn_from_data(
        data=ts_data,
        labels=[[l] for l in ts_labels],  # Reshape for compatibility
        success_criteria={"accuracy": 0.6, "max_iterations": 100},
        name="time_series_prediction",
        description="Predict next value in time series",
        modality=Modality.TIME_SERIES,
    )

    print(f"   Final accuracy: {result.accuracy:.2%}")
    print(f"   Strategy: {result.strategy_used}")

    return learner


def demonstrate_multimodal_learning():
    """Demonstrate learning across multiple modalities."""
    print_header("MULTIMODAL CHALLENGE LEARNING")

    learner = ChallengeLearner(state_dim=64, verbose=True)

    # Simulate vision data (flattened images)
    print("\n1. Vision Modality")
    print("-" * 40)
    vision_data = [np.random.randn(28 * 28) for _ in range(100)]
    vision_labels = [i % 10 for i in range(100)]

    result = learner.learn_from_data(
        data=vision_data,
        labels=vision_labels,
        success_criteria={"accuracy": 0.6, "max_iterations": 50},
        name="vision_classification",
        modality=Modality.VISION,
    )
    print(f"   Accuracy: {result.accuracy:.2%}")

    # Simulate audio data (spectrograms)
    print("\n2. Audio Modality")
    print("-" * 40)
    audio_data = [np.random.randn(128) for _ in range(100)]  # MFCC features
    audio_labels = [i % 5 for i in range(100)]

    result = learner.learn_from_data(
        data=audio_data,
        labels=audio_labels,
        success_criteria={"accuracy": 0.6, "max_iterations": 50},
        name="audio_classification",
        modality=Modality.AUDIO,
    )
    print(f"   Accuracy: {result.accuracy:.2%}")

    # Simulate sensor data
    print("\n3. Sensor Modality")
    print("-" * 40)
    sensor_data = [np.random.randn(16) for _ in range(100)]
    sensor_labels = [i % 3 for i in range(100)]

    result = learner.learn_from_data(
        data=sensor_data,
        labels=sensor_labels,
        success_criteria={"accuracy": 0.6, "max_iterations": 50},
        name="sensor_classification",
        modality=Modality.SENSOR,
    )
    print(f"   Accuracy: {result.accuracy:.2%}")

    return learner


def demonstrate_curriculum_learning():
    """Demonstrate curriculum learning with increasing difficulty."""
    print_header("CURRICULUM LEARNING DEMONSTRATION")

    learner = ChallengeLearner(state_dim=64, verbose=False)

    difficulties = [0.2, 0.4, 0.6, 0.8]
    results = []

    for diff in difficulties:
        print(f"\nDifficulty level: {diff:.0%}")
        print("-" * 40)

        # Harder challenges have more noise
        n_features = 32
        n_samples = 100
        noise_level = diff

        data = []
        labels = []
        for i in range(n_samples):
            class_id = i % 5
            mean = np.zeros(n_features)
            mean[class_id * 6:(class_id + 1) * 6] = 1.0
            sample = mean + np.random.randn(n_features) * noise_level
            data.append(sample)
            labels.append(class_id)

        challenge = Challenge(
            name=f"curriculum_{diff:.0%}",
            description=f"Classification with {diff:.0%} noise",
            challenge_type=ChallengeType.PATTERN_RECOGNITION,
            modalities=[Modality.EMBEDDING],
            training_data=TrainingData(
                samples=data,
                labels=labels,
                modality=Modality.EMBEDDING,
            ),
            success_criteria=SuccessCriteria(
                accuracy=0.6,
                max_iterations=100,
            ),
            difficulty=diff,
        )

        result = learner.learn(challenge)
        results.append(result)

        print(f"  Strategy: {result.strategy_used}")
        print(f"  Accuracy: {result.accuracy:.2%}")
        print(f"  Iterations: {result.iterations}")

    # Summary
    print("\n" + "-" * 40)
    print("Curriculum Summary:")
    for diff, result in zip(difficulties, results):
        status = "PASS" if result.success else "FAIL"
        print(f"  {diff:.0%} difficulty: {result.accuracy:.2%} [{status}]")

    return learner


def demonstrate_learned_capabilities():
    """Demonstrate using learned capabilities."""
    print_header("LEARNED CAPABILITIES")

    learner = ChallengeLearner(state_dim=32, verbose=False)

    # Train on a simple task
    print("\n1. Training a capability...")
    data = [np.random.randn(32) for _ in range(100)]
    labels = [i % 3 for i in range(100)]

    result = learner.learn_from_data(
        data=data,
        labels=labels,
        success_criteria={"accuracy": 0.6, "max_iterations": 100},
        name="test_capability",
    )

    # List capabilities
    print("\n2. Learned Capabilities:")
    print("-" * 40)
    for cap in learner.get_capabilities():
        print(f"  - {cap.name}")
        print(f"    Type: {cap.challenge_type.name}")
        print(f"    Proficiency: {cap.proficiency:.2%}")
        print(f"    Modalities: {[m.name for m in cap.modalities]}")

    # Apply capability to new data
    if result.capability_id:
        print("\n3. Applying Capability to New Data:")
        print("-" * 40)
        new_data = np.random.randn(5, 32)
        predictions = learner.apply_capability(result.capability_id, new_data)
        print(f"  Input shape: {new_data.shape}")
        print(f"  Output shape: {predictions.shape}")
        print(f"  Predictions: {np.argmax(predictions, axis=1)}")

    # Get statistics
    print("\n4. Overall Statistics:")
    print("-" * 40)
    stats = learner.get_stats()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")

    return learner


def main():
    """Run all demonstrations."""
    print("\n" + "#" * 60)
    print("#" + " " * 58 + "#")
    print("#  ATLAS CHALLENGE-BASED LEARNING DEMONSTRATION" + " " * 11 + "#")
    print("#" + " " * 58 + "#")
    print("#  All learning uses biology-inspired local rules" + " " * 9 + "#")
    print("#  (Hebbian, STDP, BCM, etc.) - NO backpropagation!" + " " * 7 + "#")
    print("#" + " " * 58 + "#")
    print("#" * 60)

    try:
        # Run demonstrations
        demonstrate_natural_language_learning()
        demonstrate_structured_data_learning()
        demonstrate_multimodal_learning()
        demonstrate_curriculum_learning()
        demonstrate_learned_capabilities()

        print("\n" + "=" * 60)
        print("ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY!")
        print("=" * 60)

        print("\nKey Takeaways:")
        print("  - ATLAS learns from natural language OR structured data")
        print("  - Supports all modalities: vision, audio, text, sensors, etc.")
        print("  - Uses biology-inspired learning (Hebbian, STDP, BCM, etc.)")
        print("  - NO backpropagation - all learning is local")
        print("  - Automatic strategy selection via meta-learning")
        print("  - Curriculum learning adjusts difficulty automatically")
        print("  - Learned capabilities can be applied to new data")

    except Exception as e:
        print(f"\nError during demonstration: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
