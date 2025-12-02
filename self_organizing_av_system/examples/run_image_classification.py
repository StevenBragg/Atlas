#!/usr/bin/env python3
"""
Image Classification Learning Demo for ATLAS

This example demonstrates how ATLAS learns to classify images autonomously
using biologically-plausible learning mechanisms:
- Unsupervised feature learning through competitive Hebbian learning
- Prototype-based classification with self-organizing representations
- Meta-learning for automatic optimization of learning strategies

Usage:
    python run_image_classification.py [--dataset synthetic|mnist|folder]
                                       [--epochs 20]
                                       [--classes 10]
                                       [--samples 500]
                                       [--folder /path/to/images]
"""

import argparse
import sys
import os
import time
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.image_classification import (
    ImageClassificationLearner,
    ImageDataset,
    quick_train_classifier,
    ClassificationResult,
)


def print_header(text: str) -> None:
    """Print a formatted header."""
    print("\n" + "=" * 60)
    print(f" {text}")
    print("=" * 60)


def print_confusion_matrix(matrix: np.ndarray, class_names: list) -> None:
    """Print a formatted confusion matrix."""
    print("\nConfusion Matrix:")
    print("-" * 40)

    # Header
    header = "True\\Pred |"
    for name in class_names:
        header += f" {name[:6]:>6}"
    print(header)
    print("-" * len(header))

    # Rows
    for i, name in enumerate(class_names):
        row = f"{name[:9]:>9} |"
        for j in range(len(class_names)):
            row += f" {matrix[i, j]:>6}"
        print(row)


def demo_synthetic_classification(
    num_classes: int = 5,
    samples_per_class: int = 100,
    epochs: int = 15,
    verbose: bool = True,
) -> float:
    """
    Demonstrate classification on synthetic data.

    This creates a synthetic dataset with distinguishable patterns
    and trains the classifier to recognize them.
    """
    print_header("Synthetic Dataset Classification Demo")

    print(f"\nGenerating synthetic dataset:")
    print(f"  - Classes: {num_classes}")
    print(f"  - Samples per class: {samples_per_class}")
    print(f"  - Total samples: {num_classes * samples_per_class}")

    # Generate synthetic data
    dataset = ImageDataset.generate_synthetic(
        num_samples_per_class=samples_per_class,
        num_classes=num_classes,
        image_size=(28, 28),
        pattern_type="shapes",
    )

    # Split into train/test
    train_data, test_data = dataset.split(train_ratio=0.8)

    print(f"\nDataset split:")
    print(f"  - Training samples: {len(train_data)}")
    print(f"  - Test samples: {len(test_data)}")

    # Create classifier
    print("\nInitializing classifier...")
    classifier = ImageClassificationLearner(
        num_classes=num_classes,
        feature_layers=[128, 64],  # Simpler feature hierarchy
        num_prototypes_per_class=5,
        learning_rate=0.05,
        use_meta_learning=True,
        use_feature_learning=False,  # Raw features work well for synthetic data
    )

    # Train
    print(f"\nTraining for {epochs} epochs...")
    print("-" * 60)

    start_time = time.time()
    metrics = classifier.train_on_dataset(
        train_data,
        epochs=epochs,
        batch_size=32,
        validation_split=0.1,
        verbose=verbose,
    )
    train_time = time.time() - start_time

    print("-" * 60)
    print(f"Training completed in {train_time:.2f} seconds")

    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_accuracy = classifier.evaluate(test_data, verbose=True)

    # Show confusion matrix
    conf_matrix = classifier.get_confusion_matrix(test_data)
    print_confusion_matrix(conf_matrix, dataset.class_names)

    # Show some example predictions
    print("\nExample predictions:")
    print("-" * 40)
    for i in range(min(5, len(test_data))):
        image, true_label = test_data[i]
        result = classifier.classify(image)
        status = "correct" if result.predicted_class == true_label else "WRONG"
        print(f"  Sample {i}: true={dataset.class_names[true_label]}, "
              f"pred={result.predicted_label} ({result.confidence:.2f}) [{status}]")

    # Show classifier stats
    print("\nClassifier Statistics:")
    stats = classifier.get_stats()
    print(f"  - Epochs trained: {stats['epochs_trained']}")
    print(f"  - Samples seen: {stats['samples_seen']}")
    print(f"  - Current strategy: {stats['current_strategy']}")
    if 'meta_learning_stats' in stats:
        ml_stats = stats['meta_learning_stats']
        print(f"  - Meta-learning updates: {ml_stats.get('total_updates', 0)}")
        print(f"  - Curriculum difficulty: {ml_stats.get('curriculum_difficulty', 0):.2f}")

    return test_accuracy


def demo_folder_classification(
    folder_path: str,
    epochs: int = 20,
    verbose: bool = True,
) -> float:
    """
    Demonstrate classification on images from a folder.

    Expected folder structure:
        folder_path/
            class_0/
                image1.png
                image2.jpg
            class_1/
                ...
    """
    print_header("Folder Dataset Classification Demo")

    print(f"\nLoading images from: {folder_path}")

    try:
        dataset = ImageDataset.from_folder(
            folder_path,
            image_size=(32, 32),
            grayscale=True,
        )
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return 0.0

    print(f"  - Classes found: {dataset.num_classes}")
    print(f"  - Class names: {dataset.class_names}")
    print(f"  - Total samples: {len(dataset)}")

    # Split into train/test
    train_data, test_data = dataset.split(train_ratio=0.8)

    print(f"\nDataset split:")
    print(f"  - Training samples: {len(train_data)}")
    print(f"  - Test samples: {len(test_data)}")

    # Create and train classifier
    classifier = ImageClassificationLearner(
        num_classes=dataset.num_classes,
        feature_layers=[256, 128, 64],
        num_prototypes_per_class=3,
        use_meta_learning=True,
    )

    print(f"\nTraining for {epochs} epochs...")
    metrics = classifier.train_on_dataset(
        train_data,
        epochs=epochs,
        verbose=verbose,
    )

    # Evaluate
    test_accuracy = classifier.evaluate(test_data, verbose=True)

    # Confusion matrix
    conf_matrix = classifier.get_confusion_matrix(test_data)
    print_confusion_matrix(conf_matrix, dataset.class_names)

    return test_accuracy


def demo_numpy_classification(
    images_path: str,
    labels_path: str = None,
    epochs: int = 20,
    verbose: bool = True,
) -> float:
    """
    Demonstrate classification on numpy arrays.
    """
    print_header("NumPy Dataset Classification Demo")

    print(f"\nLoading from: {images_path}")

    try:
        dataset = ImageDataset.from_numpy(
            images_path,
            labels_path=labels_path,
        )
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return 0.0

    print(f"  - Samples: {len(dataset)}")
    print(f"  - Classes: {dataset.num_classes}")

    # Train
    train_data, test_data = dataset.split(train_ratio=0.8)

    classifier = ImageClassificationLearner(
        num_classes=dataset.num_classes,
        use_meta_learning=True,
    )

    classifier.train_on_dataset(train_data, epochs=epochs, verbose=verbose)

    test_accuracy = classifier.evaluate(test_data, verbose=True)

    return test_accuracy


def demo_online_learning(
    num_classes: int = 5,
    num_samples: int = 500,
    verbose: bool = True,
) -> float:
    """
    Demonstrate online (sample-by-sample) learning.

    This shows how the classifier can learn incrementally
    as data arrives one sample at a time.
    """
    print_header("Online Learning Demo")

    print(f"\nSimulating online learning scenario:")
    print(f"  - Classes: {num_classes}")
    print(f"  - Total samples: {num_samples}")

    # Generate synthetic data
    dataset = ImageDataset.generate_synthetic(
        num_samples_per_class=num_samples // num_classes,
        num_classes=num_classes,
        image_size=(28, 28),
        pattern_type="shapes",
    )

    # Split for final evaluation
    train_data, test_data = dataset.split(train_ratio=0.8)

    # Create classifier
    classifier = ImageClassificationLearner(
        num_classes=num_classes,
        feature_layers=[128, 64],
        num_prototypes_per_class=2,
        learning_rate=0.03,
        use_meta_learning=True,
    )

    # Online learning
    print("\nOnline learning progress:")
    print("-" * 40)

    window_size = 50
    correct_window = []

    for i in range(len(train_data)):
        image, label = train_data[i]

        # Train on single sample and get whether prediction was correct
        was_correct = classifier.train_single(image, label)
        correct_window.append(was_correct)

        if len(correct_window) > window_size:
            correct_window.pop(0)

        # Print progress every 50 samples
        if (i + 1) % 50 == 0:
            recent_acc = sum(correct_window) / len(correct_window)
            print(f"  Sample {i+1}/{len(train_data)}: "
                  f"recent accuracy = {recent_acc:.4f}")

    print("-" * 40)

    # Final evaluation
    print("\nFinal evaluation on test set:")
    test_accuracy = classifier.evaluate(test_data, verbose=True)

    return test_accuracy


def demo_save_load(
    num_classes: int = 5,
    samples_per_class: int = 50,
    epochs: int = 5,
) -> None:
    """
    Demonstrate saving and loading a trained classifier.
    """
    print_header("Save/Load Demo")

    # Generate data
    dataset = ImageDataset.generate_synthetic(
        num_samples_per_class=samples_per_class,
        num_classes=num_classes,
        image_size=(28, 28),
    )
    train_data, test_data = dataset.split(0.8)

    # Train classifier
    print("\n1. Training initial classifier...")
    classifier = ImageClassificationLearner(num_classes=num_classes)
    classifier.train_on_dataset(train_data, epochs=epochs, verbose=False)

    original_accuracy = classifier.evaluate(test_data)
    print(f"   Original accuracy: {original_accuracy:.4f}")

    # Save
    save_path = "/tmp/atlas_classifier.pkl"
    print(f"\n2. Saving to {save_path}...")
    classifier.save(save_path)

    # Load
    print("\n3. Loading from file...")
    loaded_classifier = ImageClassificationLearner.load(save_path)

    loaded_accuracy = loaded_classifier.evaluate(test_data)
    print(f"   Loaded accuracy: {loaded_accuracy:.4f}")

    # Verify
    if abs(original_accuracy - loaded_accuracy) < 0.001:
        print("\nSave/Load verification: PASSED")
    else:
        print("\nSave/Load verification: FAILED (accuracy mismatch)")

    # Continue training the loaded model
    print("\n4. Continuing training on loaded model...")
    loaded_classifier.train_on_dataset(train_data, epochs=3, verbose=False)
    continued_accuracy = loaded_classifier.evaluate(test_data)
    print(f"   Accuracy after continued training: {continued_accuracy:.4f}")

    # Cleanup
    os.remove(save_path)


def main():
    parser = argparse.ArgumentParser(
        description="ATLAS Image Classification Learning Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with synthetic data (default)
  python run_image_classification.py

  # Run with more classes and epochs
  python run_image_classification.py --classes 10 --epochs 30 --samples 200

  # Run with images from a folder
  python run_image_classification.py --dataset folder --folder /path/to/images

  # Run online learning demo
  python run_image_classification.py --demo online

  # Run all demos
  python run_image_classification.py --demo all
        """
    )

    parser.add_argument(
        '--dataset',
        type=str,
        default='synthetic',
        choices=['synthetic', 'folder', 'numpy'],
        help='Dataset type to use (default: synthetic)'
    )

    parser.add_argument(
        '--demo',
        type=str,
        default='basic',
        choices=['basic', 'online', 'saveload', 'all'],
        help='Demo mode to run (default: basic)'
    )

    parser.add_argument(
        '--epochs',
        type=int,
        default=15,
        help='Number of training epochs (default: 15)'
    )

    parser.add_argument(
        '--classes',
        type=int,
        default=5,
        help='Number of classes for synthetic data (default: 5)'
    )

    parser.add_argument(
        '--samples',
        type=int,
        default=100,
        help='Samples per class for synthetic data (default: 100)'
    )

    parser.add_argument(
        '--folder',
        type=str,
        default=None,
        help='Path to image folder (for folder dataset)'
    )

    parser.add_argument(
        '--numpy',
        type=str,
        default=None,
        help='Path to numpy file (for numpy dataset)'
    )

    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Reduce verbosity'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )

    args = parser.parse_args()

    # Set random seed
    np.random.seed(args.seed)

    # Print welcome message
    print("\n" + "=" * 60)
    print(" ATLAS - Autonomous Image Classification Learning")
    print(" Using biologically-plausible learning mechanisms")
    print("=" * 60)

    verbose = not args.quiet

    if args.demo == 'all':
        # Run all demos
        print("\nRunning all demos...\n")

        demo_synthetic_classification(
            num_classes=args.classes,
            samples_per_class=args.samples,
            epochs=args.epochs,
            verbose=verbose,
        )

        demo_online_learning(
            num_classes=args.classes,
            num_samples=args.samples * args.classes,
            verbose=verbose,
        )

        demo_save_load(
            num_classes=args.classes,
            samples_per_class=args.samples // 2,
        )

    elif args.demo == 'online':
        demo_online_learning(
            num_classes=args.classes,
            num_samples=args.samples * args.classes,
            verbose=verbose,
        )

    elif args.demo == 'saveload':
        demo_save_load(
            num_classes=args.classes,
            samples_per_class=args.samples,
        )

    else:  # basic demo
        if args.dataset == 'folder':
            if args.folder is None:
                print("Error: --folder path required for folder dataset")
                sys.exit(1)
            demo_folder_classification(
                folder_path=args.folder,
                epochs=args.epochs,
                verbose=verbose,
            )

        elif args.dataset == 'numpy':
            if args.numpy is None:
                print("Error: --numpy path required for numpy dataset")
                sys.exit(1)
            demo_numpy_classification(
                images_path=args.numpy,
                epochs=args.epochs,
                verbose=verbose,
            )

        else:  # synthetic
            demo_synthetic_classification(
                num_classes=args.classes,
                samples_per_class=args.samples,
                epochs=args.epochs,
                verbose=verbose,
            )

    print("\n" + "=" * 60)
    print(" Demo completed successfully!")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    main()
